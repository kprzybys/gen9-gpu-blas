# Attention on the Gen9 GPU: performance model (stage 1)

Status: model (sections 1-7), stage 2 result (section 8: criterion not met in OpenCL C) and stage 2b result
(section 9: criterion met by a hand-scheduled assembly kernel, 296 GFLOPS = 2.8x the CPU). Evidence tags as in [performance-model.md](performance-model.md)
([SPEC], [MEAS], [SW], [EST], [HYP]). Stage 2 (a standalone kernel in `bench/`) starts only from this prediction,
and its result is judged against the stop criterion in section 6.

## 1. Why

The ggml backend runs only MUL_MAT (batch >= 32) and, opt-in, the fused FFN block on the GPU
(`gen9_dev_supports_op` in `ggml-backend/ggml-gen9.cpp`). Attention (ggml flash attention), RMSNorm, RoPE and the
bias ADDs stay on the CPU.

Measured on HD 530 (desktop i5-6500, no power-limit throttling) with the same 512-token ubatch at growing KV depth d
(the matrix products are identical at every depth, so the increase is attention) [MEAS]
(`results/attention-depth-hd530-2026-11-13.txt`):

| Model (Q4_K_M) | GEN9 t(d0) | t(d3584) | attention at d3584 | CPU attention rate | OpenVINO GPU attention rate |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | 3.12 s | 6.09 s | 49 % of the ubatch | 106 GFLOPS | 135 GFLOPS |
| Llama-3.2-1B | 2.44 s | 4.69 s | 48 % | 107 GFLOPS | 97 GFLOPS |

Attention cost is linear in d. On long prompts (RAG, code, summaries) it is the largest CPU part left in prompt
processing. OpenVINO's GPU attention on the same GPU is no faster than the CPU, so there is no existing Gen9 kernel to
copy; the question is whether a kernel built on this project's measured constants can do clearly better.

## 2. Shapes (GGUF headers) [MEAS]

| Model | L | H (query heads) | H_kv | d_head | group g = H / H_kv | H * L * d_head | Notes |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 28 | 12 | 2 | 128 | 6 | 43 008 | first target |
| Llama-3.2-1B | 16 | 32 | 8 | 64 | 4 | 32 768 | second target (small d_head) |
| Qwen2.5-7B | 28 | 28 | 4 | 128 | 7 | 100 352 | long launches, see section 5 |
| Gemma-2-2B | 26 | 8 | 4 | 256 | 2 | 53 248 | later: logit soft-capping, sliding window 4096 |

## 3. Work per 512-token ubatch

With n_q = 512 queries at depth d, causal inside the ubatch, each query sees on average n_k = d + 256 keys.

- **FLOP** (Q * K^T and P * V, 2 FLOP per multiply-add): F = 4 * n_q * n_k * (H * L * d_head).
  Qwen at d = 3584: F = 4 * 512 * 3840 * 43 008 = 3.38e11.
- **Softmax**: one exp and about 4 other FP32 ops per score; scores = n_q * n_k * H * L.
  Qwen at d = 3584: 6.6e8 exp + 2.6e9 ops.
- **K/V bytes** (F16 KV cache): one pass over the cache = 2 * (d + n_q) * d_head * H_kv * L * 2 B.
  Qwen at d = 3584: 117 MB, 4.2 MB per layer. If a work-group handles B_q queries for all g query heads of one KV
  head, the cache is read n_q / B_q times: B_q = 32 gives 16 passes = 1.9 GB.

## 4. Ceilings

| Resource | Ceiling | Time for Qwen at d = 3584 | Source |
|---|---|---|---|
| FP32 multiply-add (FP16 inputs, FP32 accumulator, mixed-mode `mad`) | 410 GFLOPS | 0.82 s | [MEAS] FMA 410; [SW] hf -> f folds into `mad` |
| FP16 multiply-add (FP16 accumulator) | 803 GFLOPS | 0.42 s | [MEAS] |
| K/V reads, 1.9 GB, per-layer cache (4.2 MB) inside LLC (6 MB) | 60-65 GB/s | 0.03 s | [MEAS] block reads 8-16 MB |
| same from RAM (worst case) | 20 GB/s | 0.10 s | [MEAS] |
| Softmax exp | 13.3 Gop/s `exp`, 44.0 `native_exp` (HD 530, 1.0 GHz) | 0.050 s / 0.015 s | [MEAS] `results/attn-micro-hd530-2026-11-13.txt` |
| SLM reads (K/V tiles) | 177 GB/s consecutive or broadcast, 96 GB/s stride 2 (HD 530, 1.0 GHz) | 2-byte element must serve >= 2 multiply-adds from registers at 300 GFLOPS | [MEAS] same file |
| Softmax other FP32 ops | 200 Gop/s | 0.013 s | [MEAS] FP32 mul/add |

Attention here is compute-bound: the K/V traffic is a few percent of the multiply-add time even from RAM, as long
as all g query heads of a KV head reuse one K/V tile.

Accuracy decides the multiply-add ceiling. Scores are dot products of d_head = 128 terms that feed exp, so an
absolute score error becomes a relative weight error. With FP16 accumulation the error is of order
sqrt(128) * 4.9e-4 times the summed magnitudes (section 6 of performance-model.md), too much for scores of size
10-30. The first variant therefore accumulates Q * K^T in FP32 (410 GFLOPS ceiling). P * V (P in [0, 1]) uses FP16
products with an FP32 output accumulator and the online rescaling of flash attention. A variant with FP16 partial
sums over short chunks (as in the accurate FP16 GEMM) is the second step, only if the accuracy test allows it.

## 5. Kernel constraints from known facts

- **Registers.** Private arrays above about 128 B per lane go to scratch, which kcl does not configure: silently
  wrong results [SW]. Separate vector variables stay in registers. The output accumulator must be split across the
  lanes of a subgroup: for d_head = 128 in SIMD8, 4 query rows x 128 values = 64 floats per lane = 64 of 128 GRF
  (one GRF per SIMD8 float variable). SIMD16 with 8 rows needs 64 two-GRF variables = all 128 GRF, so SIMD8 with
  4 rows per subgroup is the starting point. `lib/gen_kernels.py` rejects kernels
  that spill.
- **GQA reuse.** One work-group = one KV head x B_q queries x all g query heads, so each K/V tile is read once for
  g heads (Qwen g = 6, Llama g = 4).
- **KV cache layout.** llama.cpp keeps K and V per layer as rows of H_kv * d_head values per token in host memory,
  reached by the GPU through userptr. One head of one token is 256 B contiguous (Qwen). Row streams of 256 B read at
  14-18 GB/s [MEAS], enough here (section 4), but userptr loses 2x when hundreds of 4 KB pages are touched at once
  [MEAS]; keep a work-group's K/V walk sequential.
- **Launch length.** The GPU is preempted after about 640 ms with silently wrong results (pitfalls.md). Qwen 1.5B at
  d = 3584 is 1.2e10 FLOP per layer, about 40-60 ms per launch: fine. Qwen 7B at d = 32k is 2.4e11 FLOP per layer,
  about 0.8-1.2 s: a launch must cover a slice of query blocks. Rule: at most 2e10 FLOP per launch.
- **64-bit address arithmetic** in the inner loop is emulated [SW]: compute row pointers before the loop.
- **Shared power (laptop only).** With the fused FFN the CPU clock fell from 2500 to 1750 MHz and CPU attention
  slowed by 48 % (`results/ggml-backend-ffn-fusion-2026-11-13.txt`). Moving attention to the GPU removes that
  dependence, so the laptop gain should be larger than the desktop one [HYP].

## 6. Prediction and stop criterion

[HYP], to be checked by stage 2 on HD 530 (desktop) and UHD 620 (laptop):

| Quantity | FP32-accumulation variant | FP16-partial-sum variant |
|---|---|---|
| Effective attention rate (multiply-add + softmax), Qwen shape, n_k = 4096 | 200-300 GFLOPS (50-75 % of 410) | 400-550 GFLOPS |
| Extra attention time d0 -> d3584 (CPU: 2.97 s) | 1.05-1.58 s (1.9-2.8x faster) | 0.57-0.79 s (3.8-5.2x) |
| Whole 512-token ubatch at d = 3584 (now 6.09 s) | 4.0-4.6 s (-25 to -34 %) | 3.5-3.8 s (-38 to -42 %) |
| Llama (d_head = 64: smaller tiles, lower efficiency) | 160-250 GFLOPS | 320-450 GFLOPS |

The whole-ubatch line assumes everything else stays as now: 3.12 s at d = 0 minus about 0.21 s of in-ubatch CPU
attention.

**Stop criterion for stage 2:** the standalone kernel must be at least **2.5x faster than the CPU attention** on the
Qwen shape at n_k = 4096 (effective rate >= 265 GFLOPS including softmax; about -30 % of the ubatch at d = 3584),
with output error vs an FP64 reference max |O - O_ref| / max |O_ref| <= 2e-3 (FP16 output rounding is 4.9e-4).
Below 2.5x the gain does not justify the integration work (a new ggml op, masks, three model families); the result is
recorded and the direction closed. The FP32-accumulation variant sits on the threshold by this model, so stage 2 also
tries the FP16-partial-sum variant before deciding.

## 7. Gaps to measure before or during stage 2

| Quantity | Why | Microbenchmark |
|---|---|---|
| ~~`exp` throughput~~ | measured: softmax costs 1 % with `native_exp` (3-5 % with precise `exp`) | done, `bench/attn_micro.sh` |
| ~~SLM read bandwidth~~ | measured: 177 GB/s, broadcast as fast as consecutive, stride 2 halves it | done, `bench/attn_micro.sh` |
| ~~Mixed-mode `mad` rate~~ | measured in stage 2: a packed FP16 source folds into an FP32 `mad` at the full FP32 rate (323 GFLOPS in the same loop either way) | done, `k_mix` in `bench/attn_micro.cl` |
| CPU attention rate on the laptop (UHD 620, PL1) | the baseline there; expected below the desktop's 106 GFLOPS | the depth sweep of `results/attention-depth-hd530-2026-11-13.txt` on the laptop |

## 8. Stage 2 result

Measured on HD 530 (desktop, 1.0 GHz) with a standalone fused kernel (`bench/attn.cl`, host `bench/attn.c`,
runner `bench/attn.sh`); full table in `results/attn-kernel-hd530-2026-11-13.txt` [MEAS].

| Variant | Qwen shape (n_k 4096) | Llama shape | Error vs FP64 | Note |
|---|---|---|---|---|
| A: FP32 accumulation, SIMD8, lane = key for Q K^T (K pre-transposed, packed FP16 folded into `mad`), lane = dim for P V, 2 rows (Qwen) / 4 rows (Llama) per subgroup, no SLM, work-group 128 | **215 GFLOPS** (2.0x CPU) | **234 GFLOPS** (2.2x) | 2e-6 | no spill; prediction 200-300 met at its lower end |
| A2: A + register prefetch of K/V | 224 (spills 128 B) | 237 | 2e-6 | no real gain: load latency is not the limit |
| B: FP16 Q and FP16 partial sums of Q K^T over 8 dims | 192 | 183-211 | 2.4-2.6e-3 | slower and above the 2e-3 bound |
| C: non-fused | skipped | | | see the results file |

**Verdict: the stop criterion (>= 265 GFLOPS, 2.5x the CPU, on the Qwen shape) is not met; the direction is
closed at this level (OpenCL C through IGC 24.35).** What limits it:

- The compiled instruction stream itself: with K/V pointers frozen so that every read hits (diagnostic `ADV=0`)
  the same kernel runs at 261-269 GFLOPS, about 68 % of the 384 GFLOPS FP32 peak at 1.0 GHz. About half of the
  `mad`s have a src1/src2 GRF bank conflict, which IGC does not let us control (the project's GEMMs avoid it only in
  hand-written assembly). So even perfect memory behaviour would only touch the threshold.
- Registers: FP32 Q and O take 32 GRF per row at d_head 128, so a subgroup holds 2 rows (3 or more spill), and each
  K/V element read serves only 2 rows. Halving the bytes per `mad` (Llama, 4 rows) did not help, nor did
  register prefetch, and n_k 512 (all in L3) gives the same rate: the remaining 13-28 % loss on real K/V streams is
  neither bandwidth nor load latency; it is not explained yet.
- Mixed-mode `mad` (FP16 operand, FP32 accumulator) is not the problem: it runs at the full FP32 rate
  (`k_mix` in `bench/attn_micro.cl`: 323 GFLOPS for both).
- FP16 arithmetic, the only way past the FP32 ceiling, costs accuracy: 8-dim FP16 partial sums already give
  2.4-2.6e-3 (> 2e-3). The suggested cause "broadcast-scalar FP16 `mad`s lose the double rate" is [HYP]: only
  two-operand `mul`/`add` were measured that way, and the asm HGEMM runs `mad (16) :hf` with a scalar operand at
  777 GFLOPS; see [isa-notes.md](isa-notes.md) for the alternative (SIMD8 `:hf` packs 8 halves per instruction).

Implications for stage 3: integration is not started. A 2.0x gain on attention would cut a Qwen 512-token ubatch at
d = 3584 by about 25 % (6.09 -> ~4.6 s) at the cost of a new ggml op with masks for three model families; by the
rule of section 6 that is below the threshold. The direction could be reopened only by a hand-scheduled assembly
kernel (bank-conflict-free register allocation, as `asm/` does for GEMM, with 4 rows per thread via Q in SLM), whose
ceiling from the `ADV=0` diagnostic plus the conflict share is about 300-330 GFLOPS — or on a GPU with more registers.

## 9. Stage 2b result (assembly)

Measured on HD 530 (desktop, 1.0 GHz) with the same host and error test as stage 2; generator `asm/attn_gen.py`
(with `asm/isa.py`), template `asm/attn.cl`, `make -C asm attn` (`DEVICE=skl` for HD 530, `whl` for UHD 620);
`bench/attn.c` loads the binary with `CL=<file>.bin`. Full table, predictions and instruction counts in
`results/attn-asm-hd530-2026-11-13.txt` [MEAS].

| Variant | Qwen shape | Llama shape | Error | Note |
|---|---|---|---|---|
| R: IGC's loop of variant A, operands remapped (Q permuted, p copies), conflicts 512 -> 0 per 1024 mads | 218 (+0 %) | 247 (+5 %) | 2e-6 | conflicts are not the Qwen limit |
| template `asm/attn.cl` (OpenCL C: Q/O via block reads/writes, frees registers) | 231 | 246 | 2e-6 | base of S and C |
| S2: whole loop generated on IGC's prologue/epilogue, K/V ring of 2 slots (loads 1 step = 32 mads ahead), conflict-free | 288 | (no registers) | 2e-6 | criterion met |
| S2e / S4e: exp in place to free registers, ring of 2 / 4 slots | 273 / 281 | 305 | 2e-6 | exp in place costs 5 %, NS 4 +3 % |
| **C: separate p copies, l and m moved to idle registers; ring 4 (Qwen) / 2 (Llama)** | **295-296** (2.8x CPU) | **310** (2.9x) | 2.1e-6 / 2.0e-6 | best |

The laptop UHD 620 confirms it: 308 GFLOPS (Qwen, at ~1046 MHz) and 342 (Llama, 1100 MHz) with the same binaries
built for `whl`.

**Verdict: the stop criterion (>= 265 GFLOPS, 2.5x the CPU, error <= 2e-3) is met** with 296 GFLOPS and error
2.1e-6. What made the difference is scheduling, not bank conflicts: IGC issues each K/V block read right before
its use, and with ~6 threads per EU the read latency stays exposed; the hand loop keeps the next one to three steps
in flight (+25 % over the same template in OpenCL C). Removing the conflicts alone gave nothing for Qwen.

What limits it now: the first explanation below ([EST], L3 read bandwidth) is contradicted by the OA counters
(`results/oa-attention-hd530-2026-11-13.txt`): stalls are 2-3 %, both FPU pipes are busy 86-88 % of active time,
L3 traffic is 112-118 GB/s, and the loss is ~20 % (Qwen) / 13 % (Llama) of EU time without a thread [MEAS];
[HYP] work distribution (work-group granularity, grid tail); `LOCAL=24` is 3.5 % faster with less idle time.
The original estimate, kept for the record: L3 read bandwidth for Qwen. At 2 rows per thread each 128 B block read feeds 16 SIMD8
mads (8 B per mad), and 296 GFLOPS is 148 GB/s = 148 B per clock, the block-read ceiling measured on UHD 620
(155-158 GB/s at ~1.1 GHz). More speed would need more query rows per thread, but FP32 Q and O already take 64 of
128 GRF at d_head 128, and SLM shares the L3 data port. Llama (4 rows, 4 B per mad) is not bandwidth-bound; at 81 %
of peak it is held by the non-mad instructions (~15 %) and the softmax of 4 rows.

Implications for stage 3: by the rule of section 6 the integration is now justified. With 296 GFLOPS instead of
106 the attention of a Qwen 512-token ubatch at d = 3584 (3.4e11 FLOP) takes ~1.1 s instead of ~3.2 s, so the
ubatch would go from 6.09 s to about 4.0 s (-33 %) [EST], more on the laptop where the CPU also loses clock to
the GPU. Open points that the standalone kernel does not cover:
- causal mask inside the ubatch (only the last key blocks of each query block; a masked tail or the CPU for the
  in-ubatch part), and the K layout: the kernel reads K transposed in 8-key groups (`k_kt`, 0.28 ms per layer at
  n_k 4096, under 1 %); a transposed K cache would avoid the pass;
- one binary per shape (D, G, H_kv, rows per thread) and per device, built from the template by the generator; the
  generator reads IGC's own loop (symbolic pass with asserts), so a new IGC/NEO version must be re-checked: the
  generator refuses any ocloc other than 24.35.30872.36 (override `ATTN_OCLOC_ANY=1`), stops on any line form of
  IGC's loop or epilogue that `asm/isa.py` does not understand, and `make -C asm attn-check` compares the binaries
  for both devices with the SHA-256 in `asm/attn_sha256.txt`;
- **K buffer padding**: the ring prefetches the first NS steps of the next key block also after the last block, so
  the K-transposed buffer is read up to (NS-1)*128 + (NK-1)*D*16 + 128 bytes past the end of the last KV head
  (Qwen ring 4: 2560 B; Llama ring 2: 1280 B; loaded, never used). The integration must allocate it with at least
  NK*D*16 extra bytes (one block of 8*NK keys: 4096 / 2048 B; covers every ring depth since NS <= D/8);
  `bench/attn.c` pads exactly that (`KT_PAD` overrides) instead of the earlier blanket 64 KB;
- host-side preconditions the hand loop does not check: n_k > 0 and a multiple of 8*NK (no tail: llama.cpp's KV
  length must be padded or the tail handled separately), n_q*G a multiple of the rows per thread, full SIMD8
  subgroups (work-group size a multiple of 8 dividing the global size; `bench/attn.c` rejects anything else);
- launch length: Qwen 1.5B at d = 3584 is ~40 ms per layer, far from the 640 ms preemption limit; long contexts
  and larger models need the per-launch FLOP rule of section 5;
- Llama and Gemma shapes need their own row count and ring depth (Llama: 4 rows, ring 2; Gemma d_head 256 does not
  fit 2 FP32 rows and needs a different split).

