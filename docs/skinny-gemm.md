# Skinny GEMM (small M, e.g. LLM inference shapes)

Skinny GEMM is `C (M x N) = alpha·op(A)·op(B) + beta·C` where one of the dimensions M or N is small (<= 16) and the other
operand is a large weight matrix. Typical case: token-by-token generation in a language model, where M is the number of tokens
or beams processed at once (1-64) and the weights are N x K (like `nn.Linear`, `C = A·Wᵀ`) or K x N (`C = A·B`).

It is an ordinary FP16/FP32 GEMM, so it lives in `gen9blas` and `gen9blas_gemm` selects it automatically; the BLAS layer
(`cblas_sgemm`, see [api.md](api.md)) gets it for free. FP64 keeps the packed path. AI-specific kernels (quantization, softmax,
layer norm, a ggml backend) and weights kept in packed form between calls are not BLAS operations and are out of scope of this
library ([roadmap.md](roadmap.md)).

Evidence tags as in `results/*.txt`: [MEAS] measurement, [HYP] unconfirmed hypothesis.

## Why the generic GEMM is slow here

The generic path packs both operands into tiles (read and write of the whole weight matrix: 32 MB in FP16 for 4096 x 4096) and
the kernel computes tiles padded to full height, so for M <= 32 the time hardly depends on M. Baseline [MEAS] (mains power, USM,
N = K = 4096):

| M | HGEMM A·Wᵀ | HGEMM A·B | SGEMM A·Wᵀ | SGEMM A·B |
|---|---|---|---|---|
| 1-16 | 10.4-10.6 ms (3.2 GB/s of weights) | 6.7-6.9 ms (4.9 GB/s) | 14.0 ms (4.8 GB/s) | 10.3-10.5 ms (6.4 GB/s) |
| 32 | 11.2 ms | 7.5 ms | 14.3 ms | 11.0 ms |
| 64 | 12.5 ms (172 GFLOPS) | 8.8 ms (244) | 17.0 ms (127) | 13.3 ms (161) |
| 256 | 21.3 ms (403 GFLOPS) | 17.6 ms (488) | 34.6 ms (248) | 30.9 ms (278) |
| gemv N (M = 1) | 1.93 ms (17.4 GB/s) | - | 3.01 ms (22.3 GB/s) | - |

## Model (written before the code; see [performance-model.md](performance-model.md))

- Bytes: the weights N·K·e (e = 2 for FP16, 4 for FP32) are read once; A (M·K·e) and C (M·N·e) are negligible for M <= 64.
- Memory-bound time: `t_mem = N·K·e / BW`, with BW ~ 22-25 GB/s (FP32 gemv and GEMV N on USM) [MEAS]. For N = K = 4096:
  FP16 1.3-1.5 ms, FP32 2.7-3.0 ms.
- Compute-bound time: `t_comp = 2·M·N·K / F`, with F ~ 750 (FP16) / 370 (FP32) GFLOPS for the assembly loops. For a simple
  kernel without assembly, 50-70% of the FMA peak (410 GFLOPS) is expected [HYP], i.e. F ~ 200-290 GFLOPS.
- Crossover `M* = F·e / (2·BW)`: FP16 ~ 9-13, FP32 ~ 18-26 for the non-assembly kernel.
- Prediction for N = K = 4096, USM: FP16 M <= 8 1.4-1.8 ms, M = 16 1.9-2.7 ms; FP32 M <= 8 2.7-3.1 ms; ordinary memory
  (userptr) up to 20% slower as long as a kernel touches few pages at a time (lesson from GEMV N, `results/gemv-n-2026-11-12.txt`).

## Design

Common to all kernels (`skinny_*` in `bench/level12.cl`, selected in `gemm_skinny` in `lib/gen9blas.c`):

- no packing: the operands are read directly (USM or userptr), the weights once per tile of rows of the small operand;
- FP32 accumulation also for FP16 (accuracy at least as good as `GEN9BLAS_HS`, at no cost when memory-bound);
- the alpha/beta epilogue is in the same kernel;
- arbitrary strides: `A[m][k] = A[m·sam + k·sak]`, `C[m][n] = C[m·scm + n·scn]`; transposed operands and slices need no copy;
- when N is the small dimension the roles are swapped, `Cᵀ = op(B)ᵀ·op(A)ᵀ`, and C is written with swapped strides;
- FP64 is not provided (local memory and registers); D keeps the packed path.

### Weights contiguous along K (A·Wᵀ)

- **`skinny_k1`, `skinny_k4`, `skinny_k8`** (MT = 1, 4, 8 rows of A per group; RW = 4, 4, 2 weight rows per group). A work-group
  is 16 subgroups of SIMD16 on the same RW weight rows (few pages at once: the lesson from GEMV N on userptr, as in `gemv_nw`). A
  lane reads 4 consecutive k, a subgroup 64 k contiguously, and the subgroups interleave with a step of 1024. Partial sums of
  the subgroups are combined in local memory. In FP16, when `sak == 1`, K is a multiple of 64 and the pointers are 16 B aligned,
  the weights and A are read with block reads (`intel_sub_group_block_read_us4`), which raised the FP16 gain by 1.2-1.4x
  (scattered 2 B reads were the limit). MT = 8 with RW = 4 (32 accumulators) spills registers, hence RW = 2.
- **`skinny_r4`, `skinny_r8`** (MT = 4 for M = 2-4, MT = 8 for M = 5-16; M > 8 is processed in tiles of 8 with the M tile varying
  fastest in the grid, so the second read of the weights comes from the last-level cache). SIMD8 subgroups, 8 weight rows per
  group, `SKINNY_S` = 16 subgroups per group. SIMD8 makes one float per lane exactly one GRF, so the 64 accumulators
  (8 weight rows x MT = 8) fit into the 128 GRF, and the mixed-mode half-to-float conversion is natively SIMD8 (in SIMD16 Gen9
  splits it in two). A lane reads 8 B (FP16: 4 k, FP32: 2 k), so a subgroup message is one 64 B line. Accumulation is an explicit
  `fma` per component, with the accumulators declared as separate vectors `s0..s7`. The epilogue is a transposed reduction in
  the subgroup (`shuffle_xor`, 64 -> 8 values per lane), then local memory `part[SKINNY_S][8·MT]`, a barrier, and a sum of the
  parts (about 300 instructions per thread, a small fraction of the loop).
- **Selection** (`gemm_skinny`): `skinny_r*` is used when weights and A are contiguous along K, the small side is >= 2, K is a
  multiple of `8·KL` (KL = 4 for FP16, 2 for FP32: K % 32 for FP16, K % 16 for FP32), pointers and strides are 16 B aligned
  (checked on the host: the GPU address has the same offset within a page as the host address), and either the small side is > 4
  or the weight row is at most 16 KB. Otherwise `skinny_k*` is used. `GEN9BLAS_SKINNY_R=0` forces `skinny_k*` (comparisons).
  M = 1 always uses `skinny_k1`.

### Weights contiguous along N (A·B)

- **`skinny_n4`** (MT = 4) plus **`skinny_n2`**. A work-item handles 4 consecutive columns; grid dimension 1 is a part of K times
  a tile of M (tiles vary fastest, so consecutive tiles read the same chunk of weights from the cache). A is loaded once per 16 k
  spread across the lanes and broadcast (`sub_group_broadcast`); the first version read `A[m][k]` separately for each k (a message
  per FMA) and was instruction-bound (~60 GFLOPS). Partial sums go to a work buffer `P[part][m][n]` (FP32) and `skinny_n2` adds the
  parts and does the alpha/beta epilogue. The number of parts is chosen for ~8 thousand work-items per tile of M. MT = 8 spills
  registers, so MT = 4 only; for M > 4 the weights are read once per tile of 4 rows.

### Threshold

The skinny path is used when `min(M, N) <= 16` for FP16 and FP32 (`GEN9BLAS_SKINNY=n` sets another threshold for all types,
`GEN9BLAS_SKINNY=0` disables it). Measured (8 shapes, USM and ordinary memory, `results/skinny-gemm-2026-11-12.txt`): at M = 16
the skinny kernels are faster almost everywhere (FP16 1.0-2.0x, FP32 0.9-2.2x; weaker only for 51866 x 1280), at M = 32 FP16 is
0.3-1.0x and FP32 0.8-1.2x of the packed path. Above 16 the packed assembly path stays.

## Results

Measurement: `python3 bench/skinny_gemm.py usm|plain` (time, GB/s of weights, GFLOPS, CPU, error against FP64). Mains power,
medians. Full tables: `results/skinny-gemm-2026-11-12.txt` (first version, all shapes) and
`results/skinny-gemm-rework-2026-11-12.txt` (block reads and the `skinny_r*` rework). Stages for `A·Wᵀ`, USM, ms [MEAS]:
(0) packed path, (1) first skinny version, (2) with FP16 block reads, (3) with `skinny_r*` (current).

| Weights N x K, type | M | (0) packed | (1) | (2) | (3) current | (0) -> (3) |
|---|---|---|---|---|---|---|
| 4096 x 4096 FP16 | 1 | 10.38 | 1.85 | 1.52 | 1.52 | 6.8x |
| | 2 | 10.81 | 2.35 | 2.27 | 1.55 | 7.0x |
| | 4 | 11.12 | 2.43 | 2.30 | 1.68 | 6.6x |
| | 8 | 11.00 | 3.92 | 2.94 | 1.90 | 5.8x |
| | 16 | 10.77 | 6.83 | 4.92 | 3.19 | 3.4x |
| 11008 x 4096 FP16 | 1 | 28.24 | 4.31 | 3.71 | 3.71 | 7.6x |
| | 8 | 27.09 | 9.00 | 6.50 | 4.21 | 6.4x |
| | 16 | 28.13 | 16.27 | 11.69 | 7.21 | 3.9x |
| 4096 x 4096 FP32 | 1 | 14.15 | 3.10 | 3.23 | 3.07 | 4.6x |
| | 4 | 13.89 | 3.73 | 3.52 | 3.48 | 4.0x |
| | 8 | 14.05 | 4.89 | 4.70 | 3.47 | 4.0x |
| | 16 | 13.89 | 6.93 | 6.88 | 4.11 | 3.4x |

For M = 1 stages (2) and (3) are the same kernel (differences are scatter). First-version results for `A·B` (FP16 4096 x 4096,
ms, packed -> skinny): M = 1 6.49 -> 1.94, M = 4 6.74 -> 2.00, M = 8 7.11 -> 2.75, M = 16 7.23 -> 4.34; FP32 `A·B`: 10.58 -> 3.43,
10.39 -> 3.34, 11.26 -> 4.93, 10.57 -> 8.26. LLM shapes at M = 1 (first version): 11008 x 4096 FP16 6.6-6.9x, FP32 4.9-5.3x;
51866 x 1280 FP16 4.5-5.5x, FP32 4.0-5.2x. Ordinary (userptr) memory: FP16 4096 x 4096, M = 1, 2.19-2.29 ms; the gain is usually
10-20% smaller than on USM.

Accuracy (FP32 accumulation): FP16 error against FP64 (relative to the maximum) 1.2e-2 - 2.9e-2 -> 2.5e-4 - 3.8e-4 (~50x
smaller); FP32 2e-6 -> 1.3e-7 - 1.2e-6. CPU per call from Python: 0.1-0.6 ms, unchanged. Overall speedup of the first version
against the packed path: 1.3-6.9x for M <= 16.

Plan criterion (set before the rework): FP16 4096 x 4096 with M <= 8 within 1.8 ms (>= 18 GB/s of weights) on USM, M = 16 within
3 ms, FP32 M <= 8 within 3.5 ms, no regression for M = 1. Status: met for M <= 4 (FP16 1.55-1.68 ms) and for FP32 M <= 8
(3.07-3.48 ms); FP16 M = 8 (1.90 ms) and M = 16 (3.19 ms) are close but not met.

### Prediction versus measurement (gaps of the model)

- The memory-bound case (M = 1) hit the prediction: FP16 1.5-1.85 ms against 1.4-1.8; FP32 3.1 against 2.7-3.1.
- For M >= 4 the measured compute rate was 70-180 GFLOPS (first version) and 168-225 GFLOPS at M = 16 (rework), against the
  predicted 200-290. Gaps found:
  (a) traffic of A from L3 grows as M/RW (every group reads all of A) and RW is limited by registers;
  (b) in `A·B` with M above MT the weights are read `ceil(M/MT)` times, from the cache only when the K-part fragment fits;
  (c) 64-bit address arithmetic in the loop (multiplication is emulated on Gen9) cost 1.2-1.9x; row pointers are now computed
  once before the loop;
  (d) register limit: 16 float accumulators per lane in SIMD16 with 4 weight vectors; 32 already spill to scratch;
  (e) weight bandwidth depends on the bytes read contiguously from a row per group step (`SKINNY_S x 64 B`): 256 B gives
  14-18 GB/s, 512 B ~19-20, 1 KB 21-24 [MEAS]; the older kernels read 2-4 KB; hence `SKINNY_S` = 16;
  (f) with long rows (44 KB, FP32 K = 11008) and 8 rows per group the bandwidth drops to 16-17 GB/s [HYP: pages / TLB]; hence
  the 16 KB row limit for M <= 4 (FP32 11008-row cases were 0.7x of the old kernel without it);
  (g) for M = 8-16 the time is closer to `t_mem + t_comp` than to `max(t_mem, t_comp)` (FP16 4096 x 4096, M = 8: 1.4 + 0.64 ~
  1.9-2.0 ms): a thread loads, waits, then computes, with too little overlap [HYP].
- Instruction mix: for the earlier `skinny_k4_h` the loop was 165 instructions for 64 useful FMA (the dot product plus
  accumulator was not fused, and mixed-mode operations split into 2 x SIMD8), about 25% of the FMA peak, which explains the
  measured 110-120 GFLOPS (27-29% of 410 [MEAS]). In the first `skinny_r8` version of the FP16 block loop, 256 of 293
  instructions were `mad` (87%), better than the predicted 77% because the conversions went into mixed-mode `mad`.

### Pitfalls found on the way

- Register spills: kernels that spill to scratch computed wrong results through `kcl` (it does not configure scratch), but
  correct ones through NEO. `lib/gen_kernels.py` now rejects kernels with a non-zero spill size, so `kcl` reports an error
  instead of computing wrongly. `float s[8][8]` in `skinny_r8` was placed in private memory by the compiler (private size
  2048); separate vectors `s0..s7` keep it in registers. A declared private size alone is not rejected, because the GEMV
  kernels declare one but never touch scratch (see [roadmap.md](roadmap.md)).
- See also [pitfalls.md](pitfalls.md).

## Comparison with ViennaCL 1.7.1

ViennaCL 1.7.1 (OpenCL on NEO, `bench/compare/viennacl_skinny.cpp`) against `libgen9blas` (USM, `gen9blas.sgemm` /
`gen9blas.gemv` from Python), FP32 `C = A·Wᵀ`, mains power, 2 alternating runs, median ms. ViennaCL has no FP16. The GEMV
row is `y = W·x` (M = 1). Full table: `results/viennacl-skinny-2026-11-12.txt`.

| Weights N x K | Operation | M | ViennaCL ms | gen9blas ms | gen9blas faster |
|---|---|---|---|---|---|
| 4096 x 4096 | gemv | 1 | 3.154 | 3.224 | 1.0x |
| | gemm | 1 / 8 / 16 | 44.3 / 53.1 / 54.4 | 3.43 / 3.70 / 4.06 | 12.9x / 14.4x / 13.4x |
| 11008 x 4096 | gemv | 1 | 7.771 | 9.250 | 0.8x |
| | gemm | 1 / 8 / 16 | 115.0 / 105.3 / 131.3 | 8.62 / 9.75 / 10.96 | 13.3x / 10.8x / 12.0x |
| 4096 x 11008 | gemv | 1 | 7.632 | 7.652 | 1.0x |
| | gemm | 1 / 8 / 16 | 30.9 / 30.5 / 33.0 | 8.33 / 10.37 / 10.78 | 3.7x / 2.9x / 3.1x |
| 2048 x 2048 | gemv | 1 | 0.784 | 1.091 | 0.7x |
| | gemm | 1 / 8 / 16 | 13.65 / 14.42 / 15.17 | 1.04 / 1.09 / 1.27 | 13.1x / 13.3x / 12.0x |

- GEMM at M = 1-16: ViennaCL needs 30-135 ms (1.1-6 GB/s of weights) because its GEMM computes full-height tiles, like the old
  packed path here, so the time hardly depends on M. `libgen9blas` takes 1.0-11 ms (13-22 GB/s): 2.9-14.6x faster.
- GEMV (M = 1, the counterpart of skinny GEMM in ViennaCL): a tie at 4096 x 4096 and 4096 x 11008; `libgen9blas` is slower at
  11008 x 4096 (0.8x) and 2048 x 2048 (0.7x: 1.09 against 0.78 ms; part of it is the call from Python, ~0.03-0.1 ms).
- Instead of M separate GEMVs: `libgen9blas` M = 16 (FP32 4096 x 4096: 4.06 ms) costs 1.3x of one ViennaCL GEMV (3.15 ms); 16
  GEMV calls in ViennaCL take ~50 ms, about 12x longer.

Other comparisons (OpenVINO, oneDNN, CLBlast, ggml Vulkan) are in [benchmarks.md](benchmarks.md); they cover square GEMM, not the
skinny case.

## Tests

`python3 lib/test_skinny.py` forces the skinny kernels for `min(M, N) <= 64` (`GEN9BLAS_SKINNY=64`) and compares with NumPy in
FP64: both weight layouts, transposes of A and B (`.T` views), small N (role swap), alpha/beta, C as a slice (row stride), odd
sizes and K tails (K mod 8, K mod 128), M around the tile sizes (1, 3, 4, 5, 8, 9, 16, 17, 33, 64), USM and plain memory, FP16 and
FP32. It is part of `make -C lib check`.

## Open work

Overlapping loads with compute for M = 8-16, a weight-bandwidth microbenchmark, scratch detection in the kernel generator, the
same rework for `A·B`, and a "constant weights" API are listed in [roadmap.md](roadmap.md).
