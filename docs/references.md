# References

Every external document and source used in this project, with versions, where it came from and what was taken from it.
Nothing listed here is redistributed in the repository except the generated header `lib/third_party/mesa/gen9_pack.h`
(see `NOTICE`). `scripts/fetch_docs.sh` downloads the documents and source excerpts into `third_party/` (ignored by git)
for local use; the documentation refers to them as `third_party/<path>` or by document name.

Evidence tags: [SPEC] Intel documentation; [MEAS] our measurement; [SW] software behaviour for a given version;
[EST] estimate from third-party code; [HYP] hypothesis. Overview: [README.md](../README.md).

## 1. Intel documents

### Open Source Graphics Programmer's Reference Manual for Kaby Lake (PRM KBL)

Gen9.5 (Kaby Lake, Coffee Lake, Whiskey Lake) is covered by the Kaby Lake volumes. Citation form in this repository:
"PRM KBL Vol. N, <section or page>".

- Download: `https://cdrdv2.intel.com/v1/dl/getContent/<document id>` (the Intel index page blocks automated clients,
  the direct links work). Mirror of the same volumes: `https://www.x.org/docs/intel/KBL/`.
- License: Intel publishes the PRMs under CC BY-ND 4.0 (Creative Commons Attribution-NoDerivatives). **They are not
  redistributed here**; the script downloads them from Intel. Quotations in this repository are short and cite volume
  and page.
- Page numbers in the documentation refer to the PDFs with the checksums below. If a checksum differs after download
  (`scripts/fetch_docs.sh` warns), Intel has re-issued the file and pages may have moved.

| Vol. | Document id | File in `third_party/pdf/` | sha256 |
|---|---|---|---|
| 1 Preface | 723910 | `intel-gfx-prm-osrc-kbl-vol01-preface.pdf` | `1833ab7fcd7f132f88504e2b243fe357f7f86ac1350838aacf407e291c03dbaf` |
| 2a Command reference: Instructions | 723916 | `intel-gfx-prm-osrc-kbl-vol02a-commandreference-instructions.pdf` | `0f2bd4d5fdcdb502606bda01d91247a4101a076c8f1ffeb620fc42beb3f141a4` |
| 2b Command reference: Enumerations | 723917 | `intel-gfx-prm-osrc-kbl-vol02b-commandreference-enumerations.pdf` | `bcae959ab2f87fb7cf00412d396a97fb7e62b3824bc131e411677d97833d8e0c` |
| 2c Command reference: Registers, part 1 | 723921 | `intel-gfx-prm-osrc-kbl-vol02c-commandreference-registers-part1.pdf` | `c6fcd0256fe9dcc51695d68e7a12e19a87e17b40004ff23277f1a452caa72faf` |
| 2c Command reference: Registers, part 2 | 723924 | `intel-gfx-prm-osrc-kbl-vol02c-commandreference-registers-part2_0.pdf` | `542454f3c013ab291296b9fb26cbc9fe16c465783f7436fe43e728dcde01de8f` |
| 2d Command reference: Structures | 723936 | `intel-gfx-prm-osrc-kbl-vol02d-commandreference-structures.pdf` | `3aca1f3230584dc598799871ba901f5b40fcbd01f19edd41dd167a9e6768a805` |
| 3 GPU overview | 723943 | `intel-gfx-prm-osrc-kbl-vol03-gpu_overview.pdf` | `9aec33ce96790de7b0402744591788041221a9e2ec92fc2ae280003f505591ae` |
| 4 Configurations | 723945 | `intel-gfx-prm-osrc-kbl-vol04-configurations.pdf` | `aa1d9bc58ad40bf5ef914de8edcfbda825bfdd279944b719f4c406fda897a481` |
| 5 Memory views | 723948 | `intel-gfx-prm-osrc-kbl-vol05-memory_views.pdf` | `8b15ca979b4f356888dd0b4b46482cd05457949974c1dde1abd6914f911fd8e0` |
| 6 Command stream programming | 723951 | `intel-gfx-prm-osrc-kbl-vol06-command_stream_programming.pdf` | `432bda751f1ef8925ee65eec3f2dedc7359c88441a03cbdef601e5f62686cc9b` |
| 7 3D / Media / GPGPU | 723952 | `intel-gfx-prm-osrc-kbl-vol07-3d_media_gpgpu.pdf` | `22aa547d0b634af217917d71642791f522ace6a9567aed74451c040fb99a7bee` |

What was taken from them (selection; section and page are given where the text uses them):

- Vol. 7, "Programming the GPGPU Pipeline" (pp. 647–665): the command sequence for a compute dispatch (pipeline select,
  state base addresses, media VFE state, interface descriptors, GPGPU_WALKER), used by `kcl` and
  `experiments/no-neo/` ([no-neo.md](no-neo.md)). Vol. 7 "Register Region Restrictions" and "Mixed Mode Float": rules
  the assembly generators must respect (for example hf to f mixed-mode operations are split into two SIMD8).
- Vol. 2a: EU instruction set, encodings and operand restrictions (assembly, `asm/`).
- Vol. 2d: bit-level structures (`INTERFACE_DESCRIPTOR_DATA`, shared local memory size encoding, p. 318, which differs from
  the genxml gen80 table, see [pitfalls.md](pitfalls.md)); Vol. 2c/2b: registers and enumerations.
- Vol. 3, 4: GT2 configuration and topology (slices, subslices, EUs, threads per EU) for [hardware.md](hardware.md).
- Vol. 5, 6: surface state, memory views, command streamer and ring/batch programming.

### Whitepaper: The Compute Architecture of Intel Processor Graphics Gen9 (v1.0)

- File: `third_party/pdf/the-compute-architecture-of-intel-processor-graphics-gen9-v1d0.pdf`
- URL: `https://www.intel.com/content/dam/develop/external/us/en/documents/the-compute-architecture-of-intel-processor-graphics-gen9-v1d0.pdf`
  (no `cdrdv2` document id)
- sha256: `5a944ad0bf9b078816480017bae4c31bfb927678c525e0f35b25154464a59f14`
- Used for: execution unit organization (threads per EU, FPU pipes, FP16/FP32/FP64 rates), cache and memory hierarchy,
  peak-throughput arithmetic ([hardware.md](hardware.md), [performance-model.md](performance-model.md)). Intel's
  product page (ARK) supplies the nominal clocks of the Core i5-8265U / UHD 620.

## 2. Khronos OpenCL extension specifications

Fetched from `https://registry.khronos.org/OpenCL/extensions/intel/<name>.html` (converted to Markdown with pandoc) into
`third_party/khronos/`; `cl_intel_media_block_io` is a `.txt` specification.

| Extension | Used for |
|---|---|
| `cl_intel_subgroups` (and `_short`, `_char`) | `intel_sub_group_block_read*/write*`, `intel_sub_group_shuffle_xor` in the GEMM and skinny kernels |
| `cl_intel_required_subgroup_size` | `intel_reqd_sub_group_size` (SIMD8/16/32 per kernel) |
| `cl_intel_spirv_subgroups` | Sub-group built-ins in SPIR-V form (reference for what the compiler emits) |
| `cl_intel_media_block_io` | 2D block read/write messages (`k_media` in `bench/memory.cl`) |
| `cl_intel_unified_shared_memory`, `cl_khr_fp16`, `cl_khr_fp64` | Referenced by kernels and by the NEO comparison paths; not fetched |

## 3. Third-party source code

All versions are the ones installed on the test machine or used to generate files; the commit is the tag's commit. The
sources are consulted for reference or used to generate a header, never copied into the library except as noted.

### Mesa 25.2.8 (`mesa-25.2.8`, commit `f97d7dcca0e51d7e20ed4342bb5ea8afea7db7fa`)

Source: `https://gitlab.freedesktop.org/mesa/mesa.git`, sparse checkout of `src/intel/compiler` and `src/intel/genxml`.

- `src/intel/genxml/gen90.xml` (with `gen80`, `gen75`, `gen70`, `gen60`, `gen50`, `gen45`, `gen40`) and
  `gen_pack_header.py`: bit-level definitions of Gen9 commands and structures. `lib/third_party/mesa/gen9_pack.h` is
  generated from them (`make gen9_pack` in `experiments/no-neo/`); Intel copyright and MIT license are kept in the
  file. [SPEC]-level facts that genxml and the PRM disagree on are called out in [pitfalls.md](pitfalls.md).
- `src/intel/compiler` (`brw_eu_validate.c`, `brw_opt_bank_conflicts.cpp`, `brw_schedule_instructions.cpp`, `brw_asm*`,
  `brw_disasm.c`, `brw_eu_*`, `brw_gram.y`, `brw_lex.l`): operand-restriction validator, register-bank conflict rules,
  instruction-latency heuristics, assembler grammar. The latency numbers (scattered messages ~300 cycles, untyped
  surface 200–600, atomics ~14000) are compiler-author assumptions, tagged [EST], not measurements.
- Mesa ANV 25.2.8 (the Vulkan driver) is the backend measured in `results/compare-ggml-vulkan-2026-11-11.txt`.
- License: MIT (Mesa). Header produced for this repository is covered in `NOTICE`.

### Intel Graphics Compiler (IGC) 1.0.17537.24 (`igc-1.0.17537.24`, commit `36a6b4e5190a181df613ab4cc6db5115ca8d56f8`)

Source: `https://github.com/intel/intel-graphics-compiler.git`, tag `igc-1.0.17537.24`, sparse checkout of `visa/iga`
and `documentation`. This is the version matching the `ocloc` 24.35 used for all builds.

- `visa/iga` (IGA, the Intel Graphics Assembler library, also used by `ocloc asm`/`ocloc disasm`) and `documentation/visa`
  (vISA instruction documentation): syntax and encoding of Gen9 assembly for `asm/` and for reading compiler output.
- Compiler behaviour observed through `ocloc` ([SW]): register allocation (including the GRF bank conflicts in `mad`
  chains that motivated hand-written assembly), private-memory spilling thresholds, payload layout in `.ze_info`.
- License: MIT.

### Intel Compute Runtime (NEO) 24.35.30872.36 (`neo-24.35.30872.36`, commit `2cacae673fabf74bcd77ed8bece2ac3df9781f91`)

Source: `https://github.com/intel/compute-runtime.git`, tag `24.35.30872.36`. NEO is the OpenCL runtime on the test machine
(package `intel-opencl-icd-legacy1`); the library does not use it, except in the `*_neo.so` variants and in comparisons.
Files consulted (27, fetched by `scripts/fetch_docs.sh`): `shared/source/gen9/*` (preamble, command encoder,
`reg_configs.h`), `shared/source/generated/gen9/hw_cmds_generated_gen9.inl`, `shared/source/helpers/*`
(state base address, preamble, local ids, per-thread data, L3 ranges), `shared/source/command_container/*`,
`opencl/source/gen9/gpgpu_walker_gen9.cpp`, `opencl/source/command_queue/gpgpu_walker*`, `opencl/source/helpers/hardware_commands_helper*`.
Taken from it: how a working Gen9 compute dispatch is programmed (order of commands, preamble, SLM size encoding
`computeSlmValues`, thread payload and local-id generation) for [no-neo.md](no-neo.md). License: MIT.

### IGT GPU Tools 2.6 (`igt-2.6`, tag `v2.6`, commit `916c79cfb7675ca01607dcd821f8c49d4f633a6f`)

Source: `https://gitlab.freedesktop.org/drm/igt-gpu-tools.git`, directory `lib`. Files: `gpgpu_fill.c/.h`,
`gpu_cmds.c/.h`, `gen9_render.h`, `gen8_media.h`, `gen7_media.h`, `intel_batchbuffer.h`, `intel_reg.h`.
`gpgpu_fill` is the smallest example of a Gen9 compute kernel submitted through i915 without NEO and was the starting
point of `experiments/no-neo/first.c`. Pitfall taken from its comments [EST]: the instruction-buffer size in
STATE_BASE_ADDRESS needs its Modify Enable bit or the GPU hangs. License: MIT.

### Linux i915 driver, v7.0 (`linux-i915`)

Source: `https://raw.githubusercontent.com/torvalds/linux/v7.0/drivers/gpu/drm/i915/gem/` (same version as the running
kernel family), files `i915_gem_userptr.c`, `i915_gem_pages.c`, `i915_gem_object.c`, `i915_gem_execbuffer.c`.
Read to understand `DRM_IOCTL_I915_GEM_USERPTR` (page pinning with `pin_user_pages_fast`, scatter-gather table, DMA
mapping through the IOMMU, invalidation by mmu notifier) and execbuffer rules, which explain the measured userptr costs
([no-neo.md](no-neo.md), `results/userptr-2026-11-12.txt`). License: GPL-2.0, not redistributed.

## 4. Software compared against (not part of the repository)

| Software | Version | Role |
|---|---|---|
| CLBlast | 1.6.2 (distribution package) | OpenCL BLAS baseline (`bench/compare/l3.c`, `bench/gemm_clblast.c`) |
| OpenBLAS | 0.3.26 | CPU baseline and CPU side of the BLAS layer |
| oneDNN | v3.6 (GPU runtime OpenCL, `jit:gemm`) | Last version with a kernel catalog for Gen9 (`results/compare-onednn-2026-11-11.txt`) |
| OpenVINO | 2026.1 | GPU plugin MatMul (`results/compare-openvino-2026-11-11.txt`) |
| ggml / llama.cpp | commit 2cbe0ec, Vulkan backend on Mesa ANV 25.2.8 | `results/compare-ggml-vulkan-2026-11-11.txt` |
| ViennaCL | 1.7.1 | `dense_blas` benchmark, repeated from the OpenBenchmarking profile `pts/viennacl` (public result 2103278-IB-KABYLAKET61) |

## 5. Documentation of this repository

[api.md](api.md), [hardware.md](hardware.md), [performance-model.md](performance-model.md), [no-neo.md](no-neo.md),
[skinny-gemm.md](skinny-gemm.md), [pitfalls.md](pitfalls.md), [history.md](history.md), [benchmarks.md](benchmarks.md),
[roadmap.md](roadmap.md), [related-work.md](related-work.md).
