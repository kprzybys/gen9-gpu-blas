# gen9-gpu-blas

BLAS (levels 1–3, FP16/FP32/FP64) for the Intel UHD Graphics 620 (Gen9.5) integrated GPU, with hand-written
Gen9 assembly GEMM loops and a minimal OpenCL subset (`kcl`) that submits work directly to the Linux i915 driver.
Library name: `gen9blas` (`libgen9blas.so`, `gen9blas.h`, Python module `gen9blas`).

**Status: work in progress — repository being prepared for publication.** Code comments, notes and result files are
still partly in Polish and are being translated.

Tested only on Intel UHD Graphics 620 (Whiskey Lake GT2) with Linux i915. `kcl` submits command buffers directly
to the kernel driver; a bug can hang the GPU.

## License

Code: MIT (`LICENSE`). Documentation and results: CC BY 4.0 (`LICENSE-docs`). Third-party material: `NOTICE`.

Developed with AI assistance (Claude, Anthropic).
