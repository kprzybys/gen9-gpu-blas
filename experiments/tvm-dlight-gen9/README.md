# Gen9 schedule rules for TVM DLight and an MLC-LLM preset

Status (2026-11-16): **written, not built, not run.** The patches apply cleanly and the Python compiles; no TVM or
MLC-LLM build exists on our machines, so nothing here is measured. Treat every constant as a starting point for a
sweep, derived from this project's measurements of hand-written kernels.

Background: MLC-LLM compiles models with TVM; kernels come from DLight rules that pick constants by target kind
(`docs/kernel-guide.md` section 6). There is no Intel branch, so an Intel GPU gets the generic constants (scalar
GEMV loads, 8x8 matmul). These patches add a Gen9 branch, selected by `"device": "intel_gen9"` in the target.

## Files

| File | Applies to | Content |
|---|---|---|
| `relax-dlight-gen9.patch` | `github.com/mlc-ai/relax` at `45221f267` (MLC-LLM's TVM submodule) | Gen9 branches in `python/tvm/s_tir/dlight/gpu/{gemv,low_batch_gemv,matmul}.py` |
| `mlc-llm-preset-gen9.patch` | `github.com/mlc-ai/mlc-llm` at `3b7663a` | presets `intel:gen9` (OpenCL) and `intel:gen9-vulkan` in `python/mlc_llm/support/auto_target.py` |

Apply with `git apply <patch>` inside the checkout (`3rdparty/tvm` for the relax patch). Use with
`mlc_llm compile ... --device intel:gen9` (or `intel:gen9-vulkan`). A hand-written target string with
`-device=intel_gen9` selects the same branches.

## Constants and where they come from

Predicate in every rule: `target.kind.name in ("opencl", "vulkan") and "intel_gen9" in str(target.attrs)`.

| Rule | Constant | Value | Evidence in this project |
|---|---|---|---|
| all | threads per group | 256 | `clinfo`: max work-group size 256 on the UHD 620 |
| GEMV (NK) | lanes along K (`TAG_R = threadIdx.x`) | | a sub-group must read one contiguous stretch of a row (`docs/skinny-gemm.md`, block reads) |
| GEMV (NK) | `VEC_C` | 4 | 8 B per lane in FP16 as `skinny_r`; 16 B in FP32 = one 64 B line per message (`docs/performance-model.md`, access width) |
| GEMV (NK) | `TR` x 8 B | 128 x 8 B = 1 KB per row per step | 256 B: 14-18 GB/s, 512 B: 19-20, 1 KB: 21-24 (`results/skinny-gemm-rework-2026-11-12.txt`) |
| GEMV (NK) | `TS` (rows per group) | 2 (tall), 4 (wide), 1 x 256 for dynamic shapes | many rows touched at once cost 2x on 4 KB pages (`results/gemv-n-2026-11-12.txt`) |
| GEMV (NK) | `VEC_LOAD` | 8 B | quantized words: 2 x uint32 = 16 int4 values per lane; 16 B FP16 vectors compile to byte reads (`docs/kernel-guide.md` §2) |
| GEMV (NK) | `LOAD_V_SHARED` | off | vector from L3 (~155 GB/s) vs SLM (~177 GB/s): no gain worth a barrier for M = 1 [EST] |
| GEMV (NK) | `UNROLL` | 16 | >= 2 reads in flight per thread (L3 155 vs 117 GB/s), without spills |
| GEMV (KN) | lanes along N, `TS, TR` | 64, 4 | like `skinny_n4` (a work-item owns consecutive columns) |
| low-batch GEMV | as GEMV, `TILE_S = 2` (fixed by DLight) | | 2 rows x bucket (<= 4) = <= 8 accumulators per lane, far below the spill limit |
| Matmul | output tile | 64 x 64 (`block 16 x 8`, `micro 4 x 8`) | CLBlast's tuned MWG = NWG = 64 on the UHD 620 [EXT] |
| Matmul | `micro 4 x 8` | 32 FP32 accumulators = 128 B per lane | the largest private array IGC 24.35 keeps in registers (`float s[8][4]`); `s[8][8]` spills (`results/skinny-gemm-rework-2026-11-12.txt`) |
| Matmul | `vector_size` | 4 | 16 B per lane loads |
| Matmul | `storage_align` | on | SLM stride-2 reads run at 96 vs 177 GB/s (`results/attn-micro-hd530-2026-11-13.txt`) |
| Matmul | `use_shared` | on | **candidate**: CLBlast picks no local memory on the UHD 620; never measured here (`docs/kernel-guide.md` §2) |
| preset | local memory | 64 KB (OpenCL and Vulkan) | `clinfo`, `vulkaninfo` (ANV: 64 KB shared, 1024 invocations, kept at 256) |

Not used on purpose: sub-group shuffles (`thread_warp_size` 1; TVM's OpenCL path has no Intel sub-group support in
these schedules), SIMD width (IGC chooses; TVM cannot request `intel_reqd_sub_group_size`).

## Known limits of DLight on Gen9

What our kernels do and DLight cannot express, so expect a gap to the hand-written path even after tuning:

- **block reads** (`intel_sub_group_block_read*`), which gave FP16 GEMV 1.2-1.4x;
- **packed panels** for GEMM (page locality at large N: 15-20x collapse without packing);
- **FP32 accumulation choice per kernel** and register-parity tricks at the ISA level;
- every call dequantizes int4 inside the kernel; for k <= 8 rows our measurement says this is issue-bound and only
  at parity with an FP16 cache (`docs/kernel-guide.md` §1).

## Plan to check (only on request; building TVM + MLC-LLM is a large job)

1. Build relax with OpenCL (and/or Vulkan) and MLC-LLM; compile Qwen2.5-1.5B `q4f16_1` with `opencl:generic` and
   with `intel:gen9`.
2. Correctness first: same output tokens / perplexity on both.
3. Measure prefill (pp512) and token generation against each other, under `gpu-lock`, alternating runs; then one
   comparison with the GEN9 ggml backend on the same model.
4. Sweep: GEMV `TR` {64, 128, 256}, `UNROLL` {8, 16, 64}; Matmul `use_shared` {on, off}, `micro` {4x4, 4x8, 8x4};
   MLC `low_batch_specialization` range {8, 16} (our skinny path wins up to 16 rows against our packed GEMM).
5. Check the generated OpenCL (`--debug-dump`) for spills: `ocloc` `spill_size` must be 0.
