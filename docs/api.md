# gen9blas API

`gen9blas` is a BLAS (levels 1-3, FP16 / FP32 / FP64) for the Intel UHD Graphics 620 (Gen9.5) integrated GPU. It has
hand-written Gen9 assembly GEMM loops (see [performance-model.md](performance-model.md)) and talks to the GPU directly
through the Linux i915 driver, without Intel's OpenCL runtime (NEO): `lib/kcl.c` implements the small subset of OpenCL the
library needs on top of i915 command buffers ([no-neo.md](no-neo.md)). The library does not link `libOpenCL`.

Four artifacts are built by `make -C lib`:

| File | What it is |
|---|---|
| `lib/libgen9blas.so` | the C library (`lib/gen9blas.h`) |
| `lib/gen9blas.py` | Python (ctypes + NumPy) wrapper around `libgen9blas.so` |
| `lib/libgen9blas_blas.so` | drop-in BLAS/CBLAS: same symbols as the system `libblas.so.3`; S and D functions on the GPU, the rest forwarded to OpenBLAS |
| `lib/libgen9blas_neo.so`, `lib/libgen9blas_blas_neo.so` | the same code on top of NEO (OpenCL), for comparisons only |

Tested only on Intel UHD Graphics 620 (Whiskey Lake GT2) with the Linux i915 driver. `kcl` submits command buffers to the
kernel driver itself, and a bug can hang the GPU.

## Conventions

- All matrices are row-major. Scalars (alpha, beta, c, s) are `double` in the C API and in Python.
- Types (`type` argument): `GEN9BLAS_H` = FP16 with FP16 accumulation (fastest), `GEN9BLAS_HS` = FP16 accumulated in chunks of
  `GEN9BLAS_KC` = 32 products into FP32 (more accurate), `GEN9BLAS_S` = FP32, `GEN9BLAS_D` = FP64 (levels 1-3). At levels 1-2
  `GEN9BLAS_HS` is the same as `GEN9BLAS_H`; FP16 is computed in FP32 and reductions in FP64.
- FP16 elements are `uint16_t` in C and `float16` in NumPy. Pointers are passed as `void *`.
- Return value `0` = success, `-1` = error; the description is `gen9blas_error()`.
- Constants: `GEN9BLAS_LEFT` / `GEN9BLAS_RIGHT` (side), `GEN9BLAS_LOWER` / `GEN9BLAS_UPPER` (uplo). `trans` / `transA`:
  0 = none, 1 = transpose. `diag`: 0 = regular, 1 = unit diagonal.
- Relative error (to the sum of |a·b|): `hgemm` ~1e-4, accurate FP16 (`HS`) ~1e-5, `sgemm` ~1e-8, FP64 ~1e-15.

## C API (`lib/gen9blas.h`)

The header is the authority for signatures; this section lists what is there.

Initialization and memory:

| Function | Purpose |
|---|---|
| `gen9blas_init(dir)` | initialize; must be called before anything else. `dir` is only used by the NEO builds (directory with the `.cl` sources and `gen_*.bin` binaries); in the default build the kernels are compiled in |
| `gen9blas_alloc(bytes)`, `gen9blas_free(p)` | zero-copy memory: USM host memory (a GEM object mapped to the CPU, aligned to 4096 B) |
| `gen9blas_is_usm(p, bytes)` | 1 if `[p, p+bytes)` lies inside one `gen9blas_alloc` allocation |
| `gen9blas_available(type)` | 1 if the kernel of the given type has been loaded |
| `gen9blas_last_time_ms()` | time of the last call (with `GEN9BLAS_PROFILE=1`: kernel time; without: the whole call) |
| `gen9blas_last_zero_copy()` | GEMM: bits 1 = A, 2 = B, 4 = C went without a copy |
| `gen9blas_error()` | text of the last error |

Level 3 (all take `type`; FP16 / FP32 / FP64 unless noted):

| Function | Computes | Notes |
|---|---|---|
| `gen9blas_gemm`, `gen9blas_hgemm`, `gen9blas_sgemm` | C = alpha·op(A)·op(B) + beta·C | transposes, any M, N, K. When min(M, N) <= 16 (FP16, FP32) the skinny kernel without packing is used, FP32 accumulation, 1.3-6.9x faster ([skinny-gemm.md](skinny-gemm.md)); `GEN9BLAS_SKINNY=0` disables it. `hgemm` / `sgemm` are shortcuts without transposes |
| `gen9blas_gemm_batched` | C_b = alpha·A_b·B_b + beta·C_b for a strided batch | **one kernel launch** for the whole batch; strides in elements |
| `gen9blas_symm` | C = alpha·A·B + beta·C, A symmetric (side, uplo) | A is expanded to a full matrix on the GPU |
| `gen9blas_syrk` | uplo triangle of C = alpha·op(A)·op(A)ᵀ + beta·C | blocked: only the bands needed for the triangle (~62% of the GEMM work) |
| `gen9blas_syr2k` | uplo triangle of C = alpha·(op(A)·op(B)ᵀ + op(B)·op(A)ᵀ) + beta·C | blocked, K' = 2K, operands merged on the GPU |
| `gen9blas_trmm` | B = alpha·op(A)·B or alpha·B·op(A), A triangular | blocked: only the non-zero range of A (~62% of the GEMM work) |
| `gen9blas_trsm` | solve op(A)·X = alpha·B or X·op(A) = alpha·B (X overwrites B) | FP32, FP64 and FP16 (`H`, `HS`); blocked (256), entirely on the GPU (inverses of the diagonal blocks in FP64) |

Levels 1-2 (`type` = `S`, `D` or `H`; vectors with stride `inc`, a negative stride works as in BLAS: the pointer is to the
lowest address):

| Function | Computes |
|---|---|
| `gen9blas_axpy`, `_scal`, `_copy`, `_swap`, `_rot`, `_rotm` | y = a·x + y, x = a·x, y = x, x <-> y, Givens rotation, modified rotation (`param` as in BLAS: flag, h11, h21, h12, h22) |
| `gen9blas_dot`, `_asum`, `_nrm2`, `_iamax` | x·y, sum of abs(x), 2-norm, 0-based index of the first max abs(x); the result goes through a pointer |
| `gen9blas_rotg`, `gen9blas_rotmg` | rotation parameters, computed on the CPU (Netlib reference algorithms; agree with OpenBLAS); FP32 only |
| `gen9blas_gemv` | y = alpha·op(A)·x + beta·y (beta = 0: y is not read) |
| `gen9blas_ger` | A = alpha·x·yᵀ + A |
| `gen9blas_symv` | y = alpha·A·x + beta·y, A symmetric (uplo triangle) |
| `gen9blas_trmv`, `gen9blas_trsv` | x = op(A)·x; solve op(A)·z = x in place. TRSV is blocked (256) with an FP32 work vector |
| `gen9blas_syr`, `gen9blas_syr2` | A = alpha·x·xᵀ + A; A = alpha·(x·yᵀ + y·xᵀ) + A (uplo triangle) |
| `gen9blas_gbmv`, `_sbmv`, `_tbmv`, `_tbsv` | banded: general, symmetric, triangular (product, solve) |
| `gen9blas_spmv`, `_tpmv`, `_tpsv`, `_spr`, `_spr2` | packed: symmetric, triangular, rank-1 and rank-2 updates |

Banded and packed matrices use the row-major layout of CBLAS `RowMajor` (formats are described in the header). The products
use one work-item per result element (speed like OpenBLAS: SPMV n = 4096 8.3 ms vs 8.5 ms). TBSV/TPSV are computed by a single
work-group row by row (inherently sequential), 6-40x slower than the CPU (tbsv n = 10^5, K = 8: 189 vs 4.5 ms), so the BLAS layer
sends them to the CPU in `auto` mode.

### Zero-copy memory

CPU involvement per call is small because the GPU reads the caller's memory directly:

- **Ordinary process memory** (NumPy arrays, any C memory) goes to the GPU without a copy through i915 userptr (`kcl_userptr`).
  Userptr objects are cached between calls: the first use of an array pins its pages (~0.45 µs of CPU per 4 KB, about the
  cost of a copy), later calls cost almost nothing. The amount of pinned memory is limited by `KCL_USERPTR_MB` (default 1024).
  Slices (row stride), transposed views and strided vectors work, and operands may overlap. Using the same arrays in
  consecutive calls: SGEMM 4096 costs 0.47 ms of CPU instead of 28 ms, axpy/dot with 10^7 elements take 0.15-0.18x of the time
  (`results/userptr-2026-11-12.txt`).
- **USM host memory** from `gen9blas_alloc` (Python: `gen9blas.empty`) needs no pinning at first use. An allocation costs
  ~0.5 ms of CPU per MB, so the Python wrapper keeps a pool (up to `gen9blas.POOL_MAX` = 1 GiB; `gen9blas.free_pool()` returns it
  to the driver).
- The copy through the CPU remains as a fallback: for a region larger than the limit, with `GEN9BLAS_NO_USERPTR=1`, and in the NEO
  builds.

## Python module (`lib/gen9blas.py`)

```python
import sys; sys.path.insert(0, "lib")          # or put lib/ on PYTHONPATH
import numpy as np, gen9blas

C = gen9blas.hgemm(A, B)                        # float16, FP16 accumulation; accurate=True: chunked accumulation to FP32
C = gen9blas.sgemm(A.T, B, alpha=2.0)           # float32; transposed views without a copy
C = gen9blas.gemm_batched(A3, B3)               # A3 (count, M, K), B3 (count, K, N) -> (count, M, N)
X = gen9blas.empty((n, n), np.float16)          # USM memory; plain arrays are zero-copy too (userptr)
gen9blas.axpy(0.5, x, y); d = gen9blas.dot(x, y)
y = gen9blas.gemv(A, x); gen9blas.ger(1.0, x, y, A)
```

The module loads `libgen9blas.so` from its own directory (`GEN9BLAS_LIB` selects another build, e.g. `libgen9blas_neo.so`) and
initializes the library on first use. The type follows the NumPy dtype: `float16`, `float32`, `float64`. The `accurate=True`
argument selects `GEN9BLAS_HS` for FP16.

| Group | Functions |
|---|---|
| Level 3 | `gemm`, `hgemm`, `sgemm`, `gemm_batched`, `symm`, `syrk`, `syr2k`, `trmm`, `trsm` |
| Level 1 | `axpy`, `scal`, `copy`, `swap`, `rot`, `rotm`, `rotg`, `rotmg`, `dot`, `asum`, `nrm2`, `iamax` |
| Level 2 | `gemv`, `ger`, `symv`, `trmv`, `trsv`, `syr`, `syr2`, `gbmv`, `sbmv`, `tbmv`, `tbsv`, `spmv`, `tpmv`, `tpsv`, `spr`, `spr2` |
| Memory, timing | `empty`, `free_pool`, `POOL_MAX`, `last_time_ms`, `last_zero_copy` |

`trmm` and `trsm` return a new array (a copy of B is modified); level-1/2 "in place" operations require a result array of type
float32/float16 (inputs of another type are converted). Arrays and their views, also with a stride or a negative stride,
need no copy (leading dimensions come from the NumPy strides).

## BLAS/CBLAS drop-in: `libgen9blas_blas.so`

The library exports the same 393 symbols as the system `libblas.so.3` (OpenBLAS 0.3): CBLAS and the Fortran interface
(`lib/blas_symbols.txt`). 110 of them are implemented on the GPU: the S and D functions of levels 2-3 including banded and
packed ones (D is generated from the S code by `lib/gen_blas_d.py`), and CBLAS level 1 (with `srotm` / `drotm`). The remaining
283 (C and Z types, `rotg` / `rotmg`, Fortran level 1, `sdsdot` / `dsdot`) jump straight to OpenBLAS on the CPU
(`lib/gen9blas_blas_forward.c`, generated from the symbol list by `lib/gen_forward.py`).

```
LD_PRELOAD=$PWD/lib/libgen9blas_blas.so python3 program.py      # NumPy/SciPy from Ubuntu packages (libblas.so.3)
GEN9BLAS_BLAS=gpu|cpu|auto   GEN9BLAS_BLAS_INFO=1   GEN9BLAS_BLAS_CPU=<path to the CPU BLAS>
```

Modes (`GEN9BLAS_BLAS`): `auto` (default), `gpu` (every S/D call on the GPU), `cpu` (everything on the CPU).

Rules of `auto`. The criterion is the CPU time of the process per call (measured on mains power;
`bench/auto_thresholds.py`, `results/auto-thresholds-2026-11-12.txt`): a call goes to the GPU when the work on the CPU would
cost more than the fixed cost of a GPU call (~20-250 µs of CPU).

| Level | Goes to the GPU from |
|---|---|
| 3 | 4 MFLOP |
| 1 | 2^20 elements |
| 2 | 2^22 (M·N); from 2^16 when OpenBLAS runs more than one thread and splits the function across threads (axpy, gemv, ger, symv, trmv, syr, syr2: the threads spin and burn 6-8x the wall time in CPU) |

Always on the CPU: alpha = 0, empty dimensions, stride 0, invalid arguments (the CPU BLAS reports them), TBSV/TPSV, and a call
after a GPU error (a message is printed once). USM and ordinary memory are treated alike (userptr, no copy). Column-major is
handled as row-major of the transposed matrix (swap dimensions, side, uplo, trans), so the same kernels are used. GPU calls
run under a mutex (NumPy releases the GIL).

Tests: `python3 lib/test_blas.py` (all S functions, both layouts, Fortran, all options, negative strides, USM, forwarding of
Z), `python3 lib/test_blas_d.py` (the same for D) and `python3 lib/test_blas_banded.py` (banded and packed, differential
against OpenBLAS on three inputs).

NumPy 1.26 (Ubuntu), float32, n = 4096 (`A @ B` etc.; CPU = % of one core):

| | OpenBLAS | libgen9blas_blas |
|---|---|---|
| `A @ B` (sgemm) | 1058 ms, 697% | **424 ms, 11%** |
| `A.T @ A` (ssyrk) | 665 ms, 700% | 397 ms, 37% |
| `scipy.linalg.blas.sgemm` | 1481 ms, 587% | 810 ms, 53% |
| `A @ x` (sgemv, 64 MB copy) | 3 ms, 700% | 12 ms, 67% |
| `A @ B` float64 (dgemm) | 2060 ms, 755% | 2650 ms, **10%** |
| `A.T @ A` float64 | 1056 ms, 777% | 1749 ms, 14% |
| `A @ x` float64 (128 MB copy) | 7 ms, 621% | 27 ms, 77% |

Limitation: LAPACK inside OpenBLAS (e.g. `scipy.linalg.solve_triangular`, `np.linalg.solve`) calls its BLAS functions
directly, bypassing the exported symbols, so the preload does not intercept them.

## Performance summary

GEMM kernel, N >= 1024: FP16 715-765, FP16 accurate 645-660, FP32 364-369 GFLOPS. The full table is in
[roadmap.md](roadmap.md) and [benchmarks.md](benchmarks.md).

Level 3, whole call, `gen9blas.empty` arrays, n = 4096, GFLOPS from the standard formula (n³ for SYRK/TRMM/TRSM, 2n³ for GEMM),
best of 5 (`python3 bench/level3_cpu.py 4096`; CPU = % of one core during the call):

| n = 4096 | FP32 | FP16 | CPU |
|---|---|---|---|
| GEMM | 354-358 | 698 | 4-5% |
| SYMM / SYR2K | 325-328 / 237-241 | 569-571 / 462-463 | 5-7% |
| SYRK / TRMM | 269-271 / 259 | 507-510 / 467 | 5-7% |
| TRSM (SciPy/OpenBLAS on the CPU for comparison: 107) | 249 | 394 | 5-7% |

FP64 (arrays float64; main kernel `dgemm_v` in OpenCL C, the rest derived from `sgemm.cl` by `lib/gen_dgemm.py`), n = 2048:
DGEMM 56, DSYMM 54, DSYRK / DSYR2K / DTRMM / DTRSM 41-42 GFLOPS, CPU 5-9%. Peak FP64 of the GPU (FMA loop) is 90-94 GFLOPS;
OpenBLAS DGEMM on the CPU reaches 71-83 GFLOPS at 760-780% CPU.

Levels 1-2 (`python3 bench/l12_gen9blas.py float32|float16`; vectors 2^24, matrices 8192²; GB/s and CPU, `gen9blas.empty` arrays):

| | FP32 GPU | FP16 GPU | FP32 NumPy/OpenBLAS on the CPU |
|---|---|---|---|
| axpy | 23.7 GB/s, 11% | 15.5 GB/s, 7% | 14.5 (SciPy saxpy, 1 thread) |
| dot / nrm2 | 16.8 / 13.6, 8-9% | 8.5 / 7.2, 6-8% | 14.5 / 12.7, 100% |
| iamax | 16.2, 11% | 8.1, 9% | 3.0, 100% |
| gemv N / T | 22.4 / 25.5, 8% | 14.2 / 16.9, 7-8% | 24.0 / 19.8, **750-790%** |
| ger | 17.3, 7% | 8.3, 4% | 4.7, 100% |
| symv* | 17.9, 7% | 9.1, 7% | 45.6, 741% |
| trmv N / T** | 22.8 / 15.2, 8-10% | 13.5 / 6.6, 6-10% | 21.2 / 23.5, 785% |
| trsv N / T** | 4.9 / 7.8, **24-36%** | 2.3 / 4.0, 17-34% | 13.9 / 13.7, 100% |
| syr / syr2 | 13.3 / 9.5, 6-7% | 6.4 / 4.6, 4-5% | 24.6 / 24.1, 780% |

\* SYMV: GB/s from the size of the whole matrix; the CPU reads the triangle once, the GPU twice (N and T passes).
\*\* Triangle: GB/s from half of the matrix. TRSV: 2 kernel launches per block of 64 (n = 8192: 256), so the driver occupies
the CPU while submitting; the GPU is slower than one CPU thread (a version without global synchronization was slower).
These table values are from an earlier build; since then level 1 reductions, TRSV (blocks of 256) and SYMV (one pass) were
sped up, see [roadmap.md](roadmap.md) and `bench/level12_speed.py`.

FP16 on the CPU (NumPy) is 0.1-0.7 GB/s.

Batched GEMM (best of 5, `gen9blas.empty` arrays): FP16 59 (256 x 64³), 116 (64 x 128³), 225 (16 x 256³) GFLOPS; FP32 60 / 90 /
149. With alpha = 1, beta = 0 and sizes that are a multiple of the tile the kernel writes straight to C (no epilogue).

CPU load: all the work is on the GPU; the CPU only submits and sleeps in `GEM_WAIT` (in NEO `clFinish` spins actively; with
`GEN9BLAS_WAIT=busy` the NEO builds use `clFinish`). SYRK / SYR2K / TRMM / TRSM pack the data once, then run several kernel
launches on fragments (sub-buffers) with a triangular epilogue; the number of bands is 4 (`GEN9BLAS_BLOCKS`). Fixed cost of a
call (small data, `gen9blas.empty` arrays, from Python): ~145 µs wall, ~85 µs CPU, of which ~25 µs is the driver and ~50 µs
the Python wrapper; from C, our part is ~3-4 µs and the rest is i915 and CPU C-states (`results/overhead-2026-11-12.txt`).

## Environment variables

All variables are read once per process (on first use). "Value" lists what is accepted; anything else is the default.

`GEN9BLAS_*` (library and BLAS layer):

| Variable | Values, default | Effect |
|---|---|---|
| `GEN9BLAS_BLAS` | `auto` (default), `gpu`, `cpu` | mode of `libgen9blas_blas.so` (above) |
| `GEN9BLAS_BLAS_INFO` | set (any value) | the BLAS layer prints each call and where it went (`GPU` / `CPU`) on stderr |
| `GEN9BLAS_BLAS_CPU` | path; default `libopenblas.so.0`, then `/usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3` | the CPU BLAS used for forwarded and CPU-side calls |
| `GEN9BLAS_DIR` | path; default `../bench` relative to the library | kernel directory passed to `gen9blas_init` (used by the NEO builds only) |
| `GEN9BLAS_LIB` | path; default `libgen9blas.so` next to `gen9blas.py` | Python module: library to load (e.g. a NEO build) |
| `GEN9BLAS_PROFILE` | set (any value) | profiling: stage times on stderr, `gen9blas_last_time_ms()` returns the kernel time. Off by default because queue profiling costs ~20 µs of CPU per event |
| `GEN9BLAS_SKINNY` | integer; default: threshold 16 for FP16 and FP32, none for FP64 | skinny GEMM is used when min(M, N) <= n; `0` disables it (old packed path) |
| `GEN9BLAS_SKINNY_R` | `0` | use the older `skinny_k4` / `skinny_k8` kernels instead of `skinny_r4` / `skinny_r8` (comparisons) |
| `GEN9BLAS_BLOCKS` | integer; default 4 for n >= 512, else 1 | number of bands in blocked SYRK / SYR2K / TRMM / TRSM |
| `GEN9BLAS_GEMV_N` | `old`, `grouped`; default: grouped for N >= 1024 | force a variant of GEMV N (comparisons) |
| `GEN9BLAS_PACK` | `old` | disable the faster FP16/FP32 packing kernels `*_q` / `*_w2` (comparisons) |
| `GEN9BLAS_NO_USERPTR` | set (any value) | disable userptr zero-copy for ordinary memory: copy through the CPU as before (comparisons) |
| `GEN9BLAS_WAIT` | `busy` | NEO builds only: wait with `clFinish` (spins) instead of sleeping |

`KCL_*` (the OpenCL-subset layer `lib/kcl.c`; mostly diagnostics):

| Variable | Values, default | Effect |
|---|---|---|
| `KCL_USERPTR_MB` | integer MB; default 1024 | limit of process memory pinned for userptr |
| `KCL_PROFILE_KERNELS` | set (any value) | timestamps around each kernel; the sum of times per kernel name is printed at process exit |
| `KCL_WAIT_EACH` | set (any value) | wait for the GPU after every kernel (diagnostics) |
| `KCL_PHASES` | set (any value) | wall time and thread CPU time per submission phase, printed at exit (`results/overhead-2026-11-12.txt`) |
| `KCL_ARB` | `age` | EU thread arbitration: default is round-robin (as NEO on Gen9; important for GEMM, K = 8192 is 1.7x faster); `age` = hardware default |
| `KCL_L3` | `slm` | always configure L3 with SLM (as NEO on KBL/CFL); by default only in batches that contain SLM kernels; no measurable difference |
| `KCL_MOCS` | integer index; default 2 | MOCS table index for buffer surfaces (diagnostics) |
| `KCL_ADDR` | `0` (default), `1`, `2` | GPU virtual address alignment: 0 = 64 KB, 1 = 2 MB for objects of 2 MB and more, 2 = additional 0-15 x 4 KB offset (diagnostics) |

## Building

Requirements: Linux with the i915 driver and access to `/dev/dri/renderD128`; GCC; Python 3 with NumPy (tests, wrapper);
`ocloc` from Intel's compute-runtime packages (the kernels are compiled at build time); the Mesa header `lib/third_party/mesa/gen9_pack.h`
is in the repository. OpenBLAS (`libopenblas.so.0`) is needed at run time for `libgen9blas_blas.so`. The NEO builds additionally
need `libOpenCL` and a working NEO. The assembly GEMM binaries (`bench/gen_h.bin`, `bench/gen_hd.bin`, `bench/gen_s.bin`) are
tracked; they are rebuilt without a GPU by `make -C asm`.

```
make -C lib           # kernels.c (from bench/*.cl and bench/gen_*.bin through ocloc), libgen9blas.so, libgen9blas_blas.so, *_neo.so
make -C lib check     # builds, then runs all 8 test suites in sequence and stops at the first failure
make -C lib test      # only test_basic.py
make -C lib clean
```

The suites run by `check`: `test_basic.py`, `test_level3.py`, `test_level12.py`, `test_userptr.py`, `test_skinny.py`,
`test_blas.py`, `test_blas_d.py`, `test_blas_banded.py` (in `lib/`). The build generator `lib/gen_kernels.py` rejects kernels
that spill registers to scratch memory, because `kcl` does not configure scratch ([pitfalls.md](pitfalls.md)).

Run the tests only on a machine where a GPU hang is acceptable.
