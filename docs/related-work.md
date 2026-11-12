# Related work

A survey of projects and papers similar to this repository, from a web search on 2026-11-12 (six general web queries).
It is a starting point, not a complete review: the links were found by a search engine and most were **not read in
full**. Claims below come from search-result summaries unless marked otherwise; verify before citing. Software this
repository already uses or compares against is listed in [references.md](references.md).

Summary: no project was found that combines the three parts of this repository — hand-written Gen9 assembly GEMM,
compute submission straight to `i915` without an OpenCL runtime, and a BLAS library for an Intel integrated GPU. Each
part has related work, listed below. Six queries are not enough to claim that no such project exists; a targeted
GitHub code search (for example `execbuffer2 GPGPU_WALKER`, `gen9 sgemm asm`) is still to do.

## 1. BLAS and GEMM on Intel integrated GPUs

| Project | What it is | Relation to this repository |
|---|---|---|
| [CLBlast](https://cnugteren.github.io/clblast/clblast.html) ([paper, arXiv 1705.05249](https://arxiv.org/pdf/1705.05249v2.pdf)) | Tuned OpenCL BLAS; publishes per-device tuning results, including [Skylake ULT GT2 (i5-6200U)](https://cnugteren.github.io/clblast/results/skylakeultgt2.html) | Already the OpenCL baseline here (see [references.md](references.md) §4). The Skylake GT2 page is a third-party reference point close to the HD 530 run (`results/skylake-hd530-2026-11-12.txt`) |
| [Intel clGPU](https://github.com/intel/clGPU) | Intel technical preview with an "Intel clBLAS" optimized for Intel GPUs; the supported list includes HD Graphics 620 | Possible additional baseline; status and maintenance not checked |
| [SGEMM for Intel Processor Graphics](https://www.codeproject.com/Articles/994769/SGEMM-for-Intel-Processor-Graphics) | Article on SGEMM kernels with `cl_intel_subgroups`, SIMD-8 | Same kernel style as the OpenCL C kernels here, but no assembly |
| [Intel OpenCL GEMM sample](https://intel.com/content/dam/develop/external/us/en/documents/intel-ocl-gemm.pdf) | Blocked GEMM sample (`gemm_nn`, `gemm_nt`) | Basic sample, not a tuned Gen9 kernel |
| [MAGMA GEMM on Intel GPUs (ICL/UTK, 2022)](https://netlib3.cs.utk.edu/files/publications/2022/icl-utk-1616-2022.pdf) | Parameterized GEMM tuned on NVIDIA A100, AMD MI100 and Intel Gen9 | Academic reference for Gen9 GEMM speed |
| [gemm_optimization](https://github.com/GPUs/gemm_optimization) | Benchmark of clBLAS, CLBlast, MIOpenGemm, MKL (CPU), cuBLAS | No UHD 620 results seen |
| [SGEMM optimization on GPU (Intel community PDF)](https://community.intel.com/cipcp26785/attachments/cipcp26785/oneapi-data-parallel-c/2629/1/sgemm-optimization-on-gpu-v4.pdf), [IXPUG 2023 paper 851](https://www.ixpug.org/images/docs/Annual_Conference_2023/IXPUG_Annual_Conference_2023_paper_851.pdf) | GEMM optimization on Intel GPUs | Not read; architecture targeted (Gen9 or Xe) unknown |

## 2. Gen assembly and ISA tooling

| Source | What it is | Relation |
|---|---|---|
| [igt-gpu-tools `assembler/` (Android mirror)](https://android.googlesource.com/platform/external/igt-gpu-tools/+/f20ae1750/assembler) | Old Gen assembler (gen4asm lineage, `gen8_instruction.c`, `gram.y`) | Older alternative to IGA; maintenance status unknown |
| [IGC vISA documentation (Debian sources)](https://sources.debian.org/src/intel-graphics-compiler/1.0.17791.18-1/documentation/visa/1_introduction.md) | Virtual ISA between compilers and native Gen code | Already used (IGC sparse checkout, [references.md](references.md) §3) |
| [Intel forum: GPU ISA code execution](https://community.intel.com/t5/GPU-Compute-Software/GPU-ISA-code-execution/m-p/1283987) | Question on running hand-written ISA on the GPU | Shows demand for what `asm/` does |

## 3. Submitting compute work without a runtime (`kcl`)

| Source | What it says | Relation |
|---|---|---|
| [dri-devel, June 2014: "looking for i915 direct programming sample application"](https://lists.freedesktop.org/archives/dri-devel/2014-June/061372.html) | Someone wants to run hand-written code on the EUs through i915; no working example in the thread | The problem `kcl` and [no-neo.md](no-neo.md) solve |
| [Intel forum: example of an app submitting work directly to the GPU, bypassing the KMD](https://community.intel.com/t5/OpenCL-for-CPU/Example-of-app-submitting-work-directly-to-GPU-bypassing-KMD/m-p/1150875) | Moderator describes the idea; no example given | Same demand |
| [compute-runtime issue #600](https://github.com/intel/compute-runtime/issues/600) | Low-latency "direct submission" needs OEM kernel driver changes; upstream uses the legacy model | Context for call-overhead numbers in [api.md](api.md) |
| [i915 GuC submission / DRM scheduler docs](https://docs.kernel.org/gpu/rfc/i915_scheduler.html) | Kernel documentation of execbuf submission | Background |
| Beignet ([Debian package](https://packages.debian.org/bullseye/beignet-opencl-icd)) | Open-source OpenCL for Ivy Bridge to Coffee Lake, LLVM-based; Intel archived the repository on 2023-01-07 | A full runtime, not minimal, but its source shows Gen9 command buffers from another angle |

## 4. Same method on other hardware

| Source | What it is | Relation |
|---|---|---|
| DeepPerf, PPoPP 2017 ([PDF](https://static.aminer.org/pdf/20170130/pdfs/ppopp/c7hwcafxtjlpqeesb0wl926oskzvr1di.pdf), [code](https://github.com/QPromise/DeepPerf)) | Reverse-engineered ISA encoding, own assembler, assembly microbenchmarks, optimized kernels for NVIDIA Kepler/Maxwell/Pascal | Closest in method: model from microbenchmarks, then kernels |
| [Automated Instruction Encoding Synthesis for Modern GPU ISA Compression (arXiv 2609.18662)](https://arxiv.org/pdf/2609.18662) | Automated recovery of GPU instruction encodings (NVIDIA) | Not read |

## Possible uses

1. **Comparison with published numbers.** CLBlast results for Skylake GT2 and the MAGMA Gen9 paper give third-party
   numbers to set next to `results/` and [performance-model.md](performance-model.md).
2. **Positioning.** The dri-devel thread and the Intel forum questions show that people looked for a minimal
   runtime-free compute example on i915 and did not get one; `kcl` with [no-neo.md](no-neo.md) answers that.
3. **Open search.** GitHub code search for small repositories that general web search misses.
