---
title: 'gen9-gpu-blas: BLAS and neural-network kernels for Intel Gen9 integrated GPUs, driven directly through the Linux i915 driver'
tags:
  - C
  - BLAS
  - GEMM
  - GPU
  - Intel Gen9
  - assembly
  - linear algebra
authors:
  - name: Karol Przybyszewski
    orcid: 0009-0001-9311-2022
    affiliation: 1
affiliations:
  - name: TODO (affiliation, to be confirmed by the author)
    index: 1
date: 15 November 2026
bibliography: paper.bib
---

# Summary

`gen9-gpu-blas` is a C library (`libgen9blas`) that provides BLAS levels 1–3 in FP16, FP32 and FP64 for Intel Gen9
integrated GPUs, the graphics part of many laptops and small desktops sold between 2015 and 2020 (UHD Graphics 620,
HD Graphics 530). It also provides the building blocks of neural-network inference on top of these
kernels: GEMM with FP16 or quantized weights packed once on the GPU, fused epilogues, elementwise operators with the
semantics of ggml [@ggml], and flash attention. The library has two unusual properties. First, the inner loops of
its GEMM and attention kernels are written in Gen9 assembly, generated from frozen OpenCL C templates by scripts that
replace the compiler's loop with a hand-scheduled one. Second, it does not use Intel's OpenCL runtime: a small
component, `kcl`, implements the subset of OpenCL the library needs by building GPU command buffers itself and
submitting them to the Linux `i915` kernel driver. The library exposes a C API (`gen9blas.h`, `gen9blas_ml.h`), a
Python/NumPy module, a drop-in BLAS/CBLAS shared object that can be injected into unmodified programs, and Rust
bindings.

# Statement of need

Integrated GPUs of this generation sit idle in most of the machines that carry them, including school and office
computers. On the UHD 620 the theoretical FP32 peak is 422 GFLOPS [@gen9whitepaper] and 403 GFLOPS of FMA throughput are
measured, comparable to the CPU cores next to them. The vendor libraries for them have moved on to newer architectures. On the test machine,
oneDNN [@onednn] reaches 120 GFLOPS for an FP16 GEMM with n = 4096, CLBlast [@clblast] 43 GFLOPS for SGEMM and
ViennaCL [@viennacl] 60 GFLOPS, and the Vulkan backend of ggml reaches 131 GFLOPS (FP32, n = 2048). `gen9-gpu-blas` reaches
344 GFLOPS for SGEMM (81 % of the theoretical FMA peak of 422 GFLOPS, whole call including data packing) and 699 GFLOPS
for HGEMM at n = 4096, while the CPU stays almost idle (4–8 % of one core during large GEMMs). All numbers above come
from text files in the repository's `results/` directory, each with its method and raw output. Where the comparison
is lost, it is stated: oneDNN is faster for n = 1024, and the CPU (OpenBLAS [@openblas]) is faster for FP64.

The intended users are:

- people who want to use these GPUs for linear algebra or local inference, for example the GEN9 backend for
  llama.cpp and whisper.cpp built on this library [@ggmlintel];
- researchers who study how far a documented but unsupported GPU can be pushed with public information only.

For the second group, the repository doubles as a worked example. It contains:

- a performance model whose constants are tagged as measured, read from source code, or hypothesized;
- a reproducible assembly build (`make -C asm check` rebuilds every binary and compares hashes);
- a symbolic checker that verifies a generated loop computes the same values as the compiler's loop;
- notes on the hardware behaviour discovered on the way, for example that a kernel running longer than the preemption
  timeout is silently cut short.

# State of the field

High-performance GEMM on CPUs follows the packing and blocking structure described by @goto2008 and implemented in
BLIS [@blis] and OpenBLAS. On GPUs, assembly-level tuning has repeatedly beaten compiled code [@volkov2008; @lai2013].
For Intel GPUs the maintained options are oneDNN and OpenVINO, which target Intel's own runtimes and newer GPUs, and
portable OpenCL libraries such as CLBlast and ViennaCL, which do not reach the hardware limits on Gen9.
`gen9-gpu-blas` applies the packed-GEMM structure to Gen9, writes the microkernels in the GPU's assembly, and replaces
the runtime with a direct driver interface. To our knowledge no other maintained library does this for Gen9.

# Software design

- **Kernels.** OpenCL C kernels (`lib/kernels/`) are compiled offline with a pinned `ocloc`. The GEMM and attention
  inner loops come from generators in `asm/` that disassemble the compiler's output, replace the loop body with a
  scheduled one and check the result for register-bank conflicts, scratch use and equivalence with the original
  loop. The binaries and their descriptors are embedded in the library, so it needs no runtime compiler.
- **kcl.** About 900 lines of C implement buffers, kernels, argument setting, dispatch and waiting directly on
  `i915` (execbuffer, userptr, context parameters), using Mesa's generated command headers [@mesa]. It splits
  long GEMMs into launches shorter than the preemption timeout and checks the engine reset counter after each wait,
  so that a lost result is reported instead of returned.
- **Library.** Kernel selection uses tile and launch-size rules derived from the performance model. Zero-copy access
  to process memory goes through i915 userptr objects. Weights can be packed once on the GPU and reused, which is
  what inference workloads need.
- **API 0.2.** Two headers separate the stable BLAS part from the operators that follow ggml and may change.
  Features are queried at run time (`gen9blas_has`), errors are typed status codes, and only the API symbols are
  exported. The CMake build installs a relocatable package (`find_package(gen9blas)`, `pkg-config`), and programs with
  their own kernels link `kcl` as a static library with a helper that generates their kernel tables.
- **Portability.** The same binary runs on Gen9 and Gen9.5 parts: an HD 530 runs the UHD 620 kernels unchanged at
  0.89–0.94 of the speed, about the clock ratio. A Gen8 (Braswell) path selected at start-up is experimental, with
  FP32 and FP64 only and SGEMM at 63–81 GFLOPS.

# Quality control

- Python test suites compare every routine with NumPy or OpenBLAS (`ctest -L gpu`), and a separate suite exercises
  every error code.
- A golden regression check rebuilds all components and requires bit-identical results from fixed inputs. It covers
  kernel binaries and tables, 59 BLAS result hashes and the test suites, and was run on two machines for every change
  of the 0.2 restructuring.
- The Rust bindings are checked against the C headers by a script that compares every function signature and
  constant.

# Research impact statement

The GEN9 backend for llama.cpp built on this library [@ggmlintel] processes a 512-token prompt of Qwen2.5-1.5B
(Q4_K_M) at 171 tokens/s on the UHD 620, 3.3 times faster than OpenVINO's GPU path on the same machine. Energy per
token is measured with RAPL in that repository's `results/`. The same kernels run a GPU speech pipeline (log-mel spectrogram,
voice activity detection, speech segmentation, speaker embeddings) checked against CPU references (bit-identical
log-mel; the same segments as the reference segmenter). This shows that a
generation of GPUs considered obsolete can carry current inference workloads.

# AI usage disclosure

The software, its documentation and this paper were developed with AI assistance (Claude Code, Anthropic). The
author decided the direction of the work, the sources of knowledge, and when to test and when to stop. The AI
gathered and aggregated knowledge, wrote code, ran tests and recorded results. Every performance and correctness
claim is backed by a measurement file in the repository. Changes are checked by the bit-exact regression suite
described above.

# Acknowledgements

We thank the authors of Mesa, the Intel Graphics Compiler and the Linux `i915` driver for the public documentation
and source code that made this work possible.

# References
