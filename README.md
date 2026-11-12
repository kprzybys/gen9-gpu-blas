# gen9-gpu-blas

BLAS levels 1–3 (FP16, FP32, FP64) for the Intel UHD Graphics 620 integrated GPU (Gen9.5, Whiskey Lake GT2), with
GEMM inner loops written in Gen9 assembly, and `kcl`, a minimal OpenCL subset that builds command buffers itself and
submits them straight to the Linux `i915` kernel driver. No OpenCL runtime (NEO) is needed to run the library.

The library is called `gen9blas`: `libgen9blas.so` + `gen9blas.h` for C, the `gen9blas` module for Python/NumPy, and
`libgen9blas_blas.so`, a drop-in BLAS/CBLAS that can be injected into existing programs with `LD_PRELOAD`.

The repository is also a worked example of programming a GPU from public documentation: a performance model with
evidence-tagged constants, a reproducible assembly build, and every measurement kept as a text file in `results/`.

**Safety: `kcl` writes raw GPU command buffers. A bug in a kernel, or in the code that submits it, can hang the GPU.**
On a laptop the display runs on the same GPU, so a hang may freeze the screen until the kernel resets the engine, or
force a reboot. Save your work first, do not run experiments on a machine you cannot reboot, and do not change
`lib/kcl.c`, `asm/` or `experiments/no-neo/` casually. See [docs/pitfalls.md](docs/pitfalls.md).

## Tested configuration

All results in this repository come from one machine; nothing else has been tested.

| Item | Value |
|---|---|
| GPU | Intel UHD Graphics 620, Whiskey Lake GT2 (Gen9.5), max 1100 MHz; CPU Core i5-8265U |
| Kernel driver | Linux 7.0 (7.0.0-42 / 7.0.0-43 in the result files), `i915` (not `xe`) |
| Device access | `/dev/dri/renderD128` (hard-coded first render node), IOMMU enabled |
| Offline compiler | `ocloc` 24.35 (Intel compute-runtime 24.35.30872.36, IGC 1.0.17537.24), always `-device whl` |
| CPU BLAS layer / reference | OpenBLAS 0.3.26 (`libopenblas0-pthread`), 8 hardware threads |
| Python | Python 3.12.3, NumPy 1.26.4 (SciPy for some tests) |
| NEO (OpenCL) | 24.35.30872.36, used only for comparisons and for the `*_neo.so` library variants |
| Other libraries compared | CLBlast 1.6.2, oneDNN v3.6, OpenVINO 2026.1, ggml/llama.cpp 2cbe0ec (Vulkan, Mesa ANV 25.2.8), ViennaCL 1.7.1 |

Hardware constants, topology and the evidence behind them: [docs/hardware.md](docs/hardware.md).

## Headline results

GFLOPS = 2·M·N·K / time. Square matrices, row-major, data in zero-copy (USM) memory, mains power, GPU at about
1100 MHz. Run-to-run variation is about ±10% (see [docs/benchmarks.md](docs/benchmarks.md)); numbers from different
rows and files are different runs. Theoretical FP32 FMA peak: 422.4 GFLOPS; measured FMA throughput: 402.8 GFLOPS
(`results/micro-2026-11-11.txt`, `results/arith-throughput-latency-2026-11-11.txt`).

### gen9blas GEMM

| Operation | n = 1024 | n = 2048 | n = 3000 | n = 4096 | Source |
|---|---|---|---|---|---|
| SGEMM (FP32), whole call | 303.0 | 341.6 | 332.8 | 343.6 | `results/compare-onednn-2026-11-11.txt` |
| HGEMM (FP16, FP16 accumulation), whole call | 447.6 | 632.4 | 644.3 | 698.9 | `results/compare-onednn-2026-11-11.txt` |
| HGEMM accurate (FP16, FP32 accumulation per 32 products), whole call | 445.5 | 564.1 | 580.2 | 617.7 | `results/compare-onednn-2026-11-11.txt` |
| DGEMM (FP64), whole call | 53.7 | 56.4 | - | 57.3 | `results/compare-l3-2026-11-11.txt` |
| HGEMM assembly kernel only (no packing) | 751.1 | 732.6 | - | 777.3 | `results/hgemm-compare-2026-11-11.txt` |
| HGEMM assembly kernel incl. GPU packing of A and B | 605.2 | 646.8 | - | 731.2 | `results/hgemm-compare-2026-11-11.txt` |

Relative error (`results/compare-onednn-2026-11-11.txt`, n = 4096, average / maximum): SGEMM 1.6e-08 / 7.8e-08,
HGEMM 1.4e-04 / 8.5e-04, HGEMM accurate 1.5e-05 / 5.6e-05. DGEMM is slower than the CPU: OpenBLAS reaches
137.8 / 99.7 / 92.3 GFLOPS at n = 1024 / 2048 / 4096 on 8 threads (`results/compare-l3-2026-11-11.txt`).

### Skinny GEMM (small M, as in LLM inference)

C = A·Wᵀ, FP16 weights 4096 × 4096, time per call in ms, USM, median
(`results/skinny-gemm-rework-2026-11-12.txt`, section "STAGE COMPARISON"; model: [docs/skinny-gemm.md](docs/skinny-gemm.md)).

| M | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Packing path (before the skinny kernels) | 10.38 | 10.81 | 11.12 | 11.00 | 10.77 |
| Skinny kernels (current) | 1.52 | 1.55 | 1.68 | 1.90 | 3.19 |

The memory-bound limit for these weights is 32 MB at 22–25 GB/s, i.e. 1.3–1.5 ms (same file).

### Comparison with other libraries

Same machine; both sides of a row come from the file named in the row. "gen9blas" = this library, whole call.

| Competitor | Case | Competitor | gen9blas | Source |
|---|---|---|---|---|
| oneDNN v3.6 (GPU) | FP32 n = 1024 | 345.2 | 303.0 | `results/compare-onednn-2026-11-11.txt` |
| oneDNN v3.6 (GPU) | FP16 n = 1024 | 702.9 | 447.6 | `results/compare-onednn-2026-11-11.txt` |
| oneDNN v3.6 (GPU) | FP32 n = 4096 | 331.8 | 343.6 | `results/compare-onednn-2026-11-11.txt` |
| oneDNN v3.6 (GPU) | FP16 n = 4096 | 120.3 | 698.9 | `results/compare-onednn-2026-11-11.txt` |
| OpenVINO 2026.1 (GPU) | FP32 n = 4096, B as input / B constant | 90.9 / 299.5 | 346.7 | `results/compare-openvino-2026-11-11.txt` |
| OpenVINO 2026.1 (GPU) | FP16 n = 4096, B as input / B constant | 305.1 / 546.8 | 698.7 | `results/compare-openvino-2026-11-11.txt` |
| ggml-vulkan (Mesa ANV) | FP32 n = 2048 | 131 | 341.6 | `results/compare-ggml-vulkan-2026-11-11.txt`, `results/compare-onednn-2026-11-11.txt` |
| ggml-vulkan (Mesa ANV) | FP16 n = 2048 (A FP16, B FP32) | 111 | 632.4 | same two files |
| CLBlast (GPU) | SGEMM n = 4096 | 43.3 | 350.8 | `results/compare-l3-2026-11-11.txt` |
| CLBlast (GPU) | HGEMM n = 4096 (vs assembly kernel) | 242.6 | 777.3 | `results/hgemm-compare-2026-11-11.txt` |
| ViennaCL 1.7.1 | SGEMM NN, 1976³ | 59.6 | 320.0 | `results/viennacl-2026-11-12.txt` |
| ViennaCL 1.7.1 | DGEMM NN, 1976³ | 34.2 | 54.0 | `results/viennacl-2026-11-12.txt` |
| ViennaCL 1.7.1 | skinny FP32 4096², M = 1, ms (lower is better) | 44.343 | 3.428 | `results/viennacl-skinny-2026-11-12.txt` |
| ViennaCL 1.7.1 | skinny FP32 4096², M = 16, ms (lower is better) | 54.384 | 4.059 | `results/viennacl-skinny-2026-11-12.txt` |
| OpenBLAS 0.3.26 (CPU, 8 threads) | SGEMM n = 4096 | 305.1 | 350.8 | `results/compare-l3-2026-11-11.txt` |

Where the competitor wins: oneDNN is faster at n = 1024 (FP32 and FP16), and OpenBLAS on the CPU is faster for DGEMM.
The ggml row goes through a different framework path (graph call, FP32 B), so it is an indication, not a like-for-like
kernel test. Energy per call and CPU load (the library leaves the CPU almost idle: 4–8% of one core for large GEMM)
are in `results/compare-l3-power-2026-11-11.txt` and `results/compare-onednn-2026-11-11.txt`.

## Quick start

Requirements: Linux with `i915` and read/write access to `/dev/dri/renderD128` (usually the `render` group), `gcc`,
`ocloc` 24.35 (generates the kernels), Python 3 + NumPy. OpenBLAS (`libopenblas.so.0`) is needed for the BLAS layer.

```sh
sh scripts/fetch_docs.sh                         # optional: reference documents/sources into third_party/ (not needed to build)
make -C asm check                                # optional, no GPU: rebuild the 6 assembly binaries, compare with the tracked ones
make -C lib libgen9blas.so libgen9blas_blas.so   # NEO-free libraries (plain `make -C lib` also builds the *_neo.so variants, needs libOpenCL)
make -C lib check                                # runs all 8 test suites on the GPU
```

Python (NumPy arrays, zero-copy through `userptr`):

```python
import sys; sys.path.insert(0, "lib")
import numpy as np, gen9blas
A = np.random.rand(2048, 2048).astype(np.float32); B = np.random.rand(2048, 2048).astype(np.float32)
C = gen9blas.sgemm(A, B)                          # also hgemm (FP16), gemm_batched, symm, syrk, syr2k, trmm, trsm, gemv, axpy, dot, ...
X = gen9blas.empty((2048, 2048), np.float16)      # USM memory from the library pool (optional)
```

C (header `lib/gen9blas.h`, all matrices row-major; full reference in [docs/api.md](docs/api.md)):

```c
#include "gen9blas.h"
gen9blas_init("bench");                            /* directory with the kernel files (bench/) */
float *A = gen9blas_alloc(sizeof(float) * M * K);  /* USM; ordinary malloc memory also works */
gen9blas_sgemm(M, N, K, 1.0, A, K, B, N, 0.0, C, N);
```
```sh
gcc -O2 prog.c -Ilib -Llib -lgen9blas -Wl,-rpath,$PWD/lib
```

Existing programs: NumPy/SciPy from the distribution use the system `libblas.so.3`. With the preloaded layer, S and D
functions of levels 2–3 and CBLAS level 1 go to the GPU when it pays off; everything else is forwarded to OpenBLAS.

```sh
LD_PRELOAD=$PWD/lib/libgen9blas_blas.so python3 program.py
GEN9BLAS_BLAS=gpu|cpu|auto   GEN9BLAS_BLAS_INFO=1      # force a side / log every call and where it went
```

Reproducing the measurements: [docs/benchmarks.md](docs/benchmarks.md).

## Limitations

- One GPU model, one machine, one driver stack (see "Tested configuration"). Other Gen9/Gen9.5 parts may work but are
  untested; newer generations and the `xe` driver are not supported.
- `kcl` is a subset of OpenCL, just enough for the kernels of this library. It does not configure scratch space: a
  kernel that spills registers would compute wrong results (the kernel generator rejects the cases found so far). It
  does not export `cl*` symbols, and it uses a single device (the first render node).
- FP64 GEMM (about 55 GFLOPS) is slower than OpenBLAS on the CPU. HGEMM accumulates in FP16 (relative error about
  1e-4); use the accurate variant (`GEN9BLAS_HS`) when that matters.
- The GPU shares RAM, last-level cache and power budget with the CPU. Results move by about ±10% with desktop load,
  and CPU costs are 2–3x higher on battery.
- In the BLAS layer only S and D functions go to the GPU; complex types, `rotg`/`rotmg` and Fortran level 1 run on the CPU.
- The results depend on specific software behaviour ([SW] in the model): compiler register allocation, driver
  submission rules. Other `ocloc`/IGC or kernel versions may change them. See [docs/performance-model.md](docs/performance-model.md).

## Repository layout

| Path | Contents |
|---|---|
| `lib/` | `libgen9blas` (`gen9blas.c`, `kcl.c`, generated kernels), Python module `gen9blas.py`, BLAS/CBLAS layer, 8 test suites |
| `asm/` | Reproducible build of the six hand-assembled GEMM binaries (`make -C asm`), generators, microbenchmarks |
| `bench/` | OpenCL C kernels, measurement programs and scripts; `bench/compare/`: comparisons with other libraries |
| `experiments/no-neo/` | Step-by-step programs that drive i915 directly (first kernel, probes, chains, userptr) |
| `experiments/isa/` | ISA-level experiments (kernels for disassembly, memory and arithmetic microbenchmarks) |
| `results/` | Raw measurement output with evidence tags, one file per experiment and date |
| `docs/` | Documentation (index below) |
| `scripts/fetch_docs.sh` | Downloads reference documents and source excerpts into `third_party/` (not in git) |
| `third_party/` | Local copies of the reference material (created by the script, not redistributed) |
| `lib/third_party/mesa/` | `gen9_pack.h` generated from Mesa genxml (MIT, see `NOTICE`) |

## Documentation

| Document | Contents |
|---|---|
| [docs/api.md](docs/api.md) | C and Python API, BLAS layer, environment variables |
| [docs/hardware.md](docs/hardware.md) | UHD 620 topology, clocks, memory system, measured constants |
| [docs/performance-model.md](docs/performance-model.md) | Constants vs dynamic factors, software behaviour, predicting a kernel's speed |
| [docs/no-neo.md](docs/no-neo.md) | How `kcl` submits work to i915 without NEO: command stream, state, thread payload, userptr |
| [docs/skinny-gemm.md](docs/skinny-gemm.md) | Small-M GEMM: model, kernels, results |
| [docs/pitfalls.md](docs/pitfalls.md) | Things that hang, miscompute or silently slow down |
| [docs/history.md](docs/history.md) | How the design evolved and why |
| [docs/benchmarks.md](docs/benchmarks.md) | Measurement method and how to reproduce each result file |
| [docs/references.md](docs/references.md) | External documents and source code used, versions, checksums |
| [docs/roadmap.md](docs/roadmap.md) | Open problems and possible next steps |

Evidence tags used in the documents and in `results/`: [SPEC] Intel documentation; [MEAS] our measurement of a hardware
constant; [SW] software behaviour, fixed for a given driver/compiler version; [EST] estimate from third-party code
(Mesa); [HYP] unconfirmed hypothesis.

## License

Code: MIT (`LICENSE`). Documentation and results: CC BY 4.0 (`LICENSE-docs`). Third-party notices: `NOTICE`.
The Intel PRM and whitepaper PDFs and the third-party source excerpts that `scripts/fetch_docs.sh` downloads are not
part of this repository and keep their own licenses.

## How to cite

See `CITATION.cff` (GitHub offers "Cite this repository" from it). Version 0.1.0 is planned.
<!-- Zenodo DOI: to be added after the 0.1.0 release is archived -->

> Przybyszewski, K. gen9-gpu-blas: BLAS levels 1–3 for Intel UHD 620 (Gen9.5) with hand-written assembly GEMM loops
> and a direct i915 OpenCL subset, version 0.1.0. https://github.com/kprzybys/gen9-gpu-blas

## AI assistance

Developed with AI assistance (Claude, Anthropic).
