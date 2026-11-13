# Attention on the Gen9 GPU: performance model (stage 1)

Status: model only, no kernel yet. Evidence tags as in [performance-model.md](performance-model.md)
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
| Softmax exp | unknown (math box) | 0.01-0.05 s if 15-60 Gop/s | gap, section 7 |
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
| `exp` throughput (FP32, native and `-cl-fast-relaxed-math`) | softmax; if below about 10 Gop/s it reaches 10 % of the multiply-add time | FMA-loop style kernel (`bench/arith_throughput_latency.sh`) |
| SLM read bandwidth and bank conflicts | K/V tiles shared by the subgroups of a work-group | block and scattered SLM reads, 1-8 subgroups |
| Mixed-mode `mad` (hf sources, f accumulator) throughput in SIMD8 | the FP32-accumulation ceiling assumes the full 410 GFLOPS | same FMA loop with half inputs |
| CPU attention rate on the laptop (UHD 620, PL1) | the baseline there; expected below the desktop's 106 GFLOPS | the depth sweep of `results/attention-depth-hd530-2026-11-13.txt` on the laptop |
