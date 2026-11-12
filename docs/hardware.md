# Hardware: Intel UHD Graphics 620 (Gen9.5) and Core i5-8265U

State as of 2026-11-11. This page collects the hardware facts used by gen9blas and where each one comes from.
How these constants enter performance predictions is in [performance-model.md](performance-model.md).

Sources:

- **[ARK]** Intel product page: https://www.intel.com/content/www/us/en/products/sku/149088/intel-core-i58265u-processor-6m-cache-up-to-3-90-ghz/specifications.html
- **[WP]** Intel whitepaper on the Gen9 architecture (details in "Compute units" below).
- **[PRM]** Intel Open Source Graphics PRM for Kaby Lake ("PRM KBL Vol. N"), local copies in `third_party/txt/` (fetched by `scripts/fetch_docs.sh`, not in git).
- **[SYS]** readout from the test system (lspci, clinfo, vainfo, vulkaninfo, glxinfo, sysfs).
- **[DBG]** i915 debugfs (details in "Topology and driver state").

Test system: Linux 7.0 with the i915 driver (kernel 7.0.0-42/43 in the result files), OpenCL by Intel NEO 24.35.30872.36, Mesa 25.2.8.

## Graphics: Intel UHD Graphics 620 (Whiskey Lake-U GT2, Gen9.5)

### According to Intel [ARK]

| Parameter | Value |
|---|---|
| GPU name | Intel UHD Graphics for 8th Generation Intel Processors |
| Base clock | 300 MHz |
| Max dynamic clock | 1.10 GHz |
| Max video memory | 32 GB (shared with RAM) |
| Outputs | eDP / DP / HDMI / DVI |
| 4K support | yes, 60 Hz |
| Max HDMI resolution | 4096x2304 @ 24 Hz |
| Max DP resolution | 4096x2304 @ 60 Hz |
| Max eDP resolution | 4096x2304 @ 60 Hz |
| Number of displays | 3 |
| DirectX | 12 |
| OpenGL | 4.5 |
| Intel Quick Sync Video | yes |
| Intel Clear Video / Clear Video HD | yes / yes |
| Device ID | 3EA0, 3EA1 |

### System readout [SYS]

| Parameter | Value |
|---|---|
| PCI device | 00:02.0, [8086:3ea0] |
| Kernel driver | i915 |
| DRM nodes | /dev/dri/card1, /dev/dri/renderD128 |
| Execution units (EU) | 24 |
| Clock (sysfs) | 300-1100 MHz |
| Memory visible in OpenCL | 14.09 GiB global, max single allocation 4 GiB, local memory 64 KiB |
| Max work-group size | 256 (multiple of 32) |
| Theoretical performance (derived) | about 0.42 TFLOPS FP32 / 0.84 TFLOPS FP16 (24 EU x 16 FLOP/clock x 1.1 GHz) |

### Compute units: full specification

Source **[WP]**: Intel, "The Compute Architecture of Intel Processor Graphics Gen9", v1.0, 14 August 2015
(S. Junkins), https://www.intel.com/content/dam/develop/external/us/en/documents/the-compute-architecture-of-intel-processor-graphics-gen9-v1d0.pdf
(local copy: `third_party/pdf/the-compute-architecture-of-intel-processor-graphics-gen9-v1d0.pdf`).
The document describes Gen9 (Skylake, HD 530). UHD 620 is Gen9.5 with the same EU, subslice and slice design.
clinfo and debugfs confirm it: 1 slice, 3 subslices, 8 EU, 7 threads per EU.
Per-second values are computed for 1.1 GHz and are theoretical peaks.

#### Hierarchy

| Level | Count | Source |
|---|---|---|
| Slice | 1 | [WP], [SYS], [DBG] |
| Subslice | 3 (8 EU each) | [WP], [SYS], [DBG] |
| EU | 24 | [WP], [SYS], [DBG] |
| Hardware threads | 168 (24 x 7) | [WP], [SYS] (7 threads per EU) |
| Max work-items in flight | 5376 (168 x SIMD-32) | [WP] |

#### Execution Unit (EU) [WP]

| Element | Specification |
|---|---|
| Hardware threads | 7 per EU; SMT combined with fine-grained interleaving (IMT); round-robin scheduling |
| GRF (general register file) | 128 registers x 32 B = 4 KB per thread, 28 KB per EU, 672 KB for the whole GPU |
| ARF | separate architecture register file for per-thread state (instruction pointer etc.) |
| Instruction issue | up to 4 instructions per clock, from 4 different threads, to up to 4 functional units |
| Functional units | 2 x SIMD FPU, 1 branch unit, 1 send unit |
| SIMD FPU | physically 4 FP32/INT32 lanes or 8 FP16/INT16 lanes per FPU; FMA (MAD) every clock |
| Extended math and FP64 | only one of the two FPUs (transcendentals and double precision) |
| INT32 | native in both FPUs |
| Logical SIMD width | SIMD-8, -16 and -32, split into SIMD-4 operations |
| GRF -> FPU bandwidth | 96 B/clock read and 32 B/clock write per EU (for MAD) |
| Branch unit | tracks branches and their nesting, active-lane mask |
| Send unit | all memory, sampler and system communication operations |

#### Peak performance (whole GPU, 24 EU)

| Type | Per clock | At 1.1 GHz | Derivation |
|---|---|---|---|
| FP32 | 384 FLOP | 422.4 GFLOPS | 24 x 2 FPU x SIMD-4 x (mul+add) [WP] |
| FP16 | 768 FLOP | 844.8 GFLOPS | 2 x FP32, because of 8 FP16 lanes per FPU [WP] |
| FP64 | 96 FLOP | 105.6 GFLOPS | 24 x 1 FPU x SIMD-4 x 2 x 1/2 (1/4 of FP32) [WP] |
| INT32 | 192 IOP | 211.2 GIOPS | 24 x 2 FPU x SIMD-4 x add [WP] |
| INT16 | 384 IOP | 422.4 GIOPS | derived from 8 INT16 lanes per FPU |
| INT8 dot product (DP4a) | none | - | no such instruction (appears in Gen12; general knowledge, not verified in a document) |
| Matrix units (XMX) | none | - | [WP] describes no such units |

Measured throughput against these peaks (FP32 410, FP16 803 GFLOPS, FP64 90-94) is in
[performance-model.md](performance-model.md), section 2.

#### Subslice (x3) [WP]

| Element | Specification |
|---|---|
| EU | 8 (56 threads) |
| Local thread dispatcher | 1 |
| Instruction cache | own; misses go to L3 |
| Sampler | 1, read-only; sampler L1 and L2 caches; BC1-BC7/DXT decompression; point, bilinear, trilinear, anisotropic filtering; native NV12 |
| Data port | 1 load/store unit, scatter/gather, merges accesses into 64 B lines; serves SLM |
| Interface to L3 | 64 B/clock read and write per data port |
| Shared Local Memory | 64 KB per subslice, 192 KB in total; 64 B bus; more banks than L3; not coherent with the rest of memory |
| Hardware barriers | up to 16 active thread groups per subslice |

#### Slice (x1) [WP]

| Element | Specification |
|---|---|
| L3 | 768 KB per slice, of which 512 KB usable as application data cache; the rest for pipeline buffers and SLM; partitioned. clinfo shows 768 KiB |
| L3 bandwidth | 192 B/clock in total (3 x 64 B), about 211 GB/s at 1.1 GHz |
| L3 fill from memory | 64 B/clock |
| Atomics | hardware read-modify-write on L3 and SLM; 32-bit, including float min/max/cmpxchg (new in Gen9) |
| Barriers | hardware barrier logic |
| Cache line | 64 B along the whole path: SIMD-16 x 32 bit = 64 B register = 64 B bus = 64 B L3 line |
| Preemption | at compute thread level (thread-level preemption) |

#### Connection to the rest of the system [WP] / [ARK]

| Element | Specification |
|---|---|
| GTI (GPU <-> ring/LLC) | 64 B/clock read; write 64 B/clock or 32 B/clock in the low-power configuration |
| LLC | shared with the CPU, 6 MB [ARK] |
| eDRAM | none (GT2) |
| RAM | DDR4-2400 / LPDDR3-2133, 2 channels, max 37.5 GB/s shared with the CPU [ARK]. This is the real limit for data outside the caches |

Data port message types used by the kernels (byte scattered, dword scattered, untyped surface, OWord block, media block):
PRM KBL Vol. 7 (3D/Media/GPGPU) and Vol. 2d (structures).

#### Programming model [SYS] (clinfo)

| Element | Value |
|---|---|
| Sub-group sizes | 8, 16, 32 |
| Max work-group | 256, max 32 sub-groups |
| Preferred work-group size multiple | 32 |
| Native vector widths | char 16, short 8, int 4, long 1, half 8, float 1, double 1 |
| FP16 / FP32 / FP64 | in hardware; denormals, INF/NaN, all rounding modes, FMA IEEE 754-2008 |
| FP32 div/sqrt | correctly rounded |
| Atomics (OpenCL) | relaxed, acq/rel, seq_cst; work-group, device and all-devices scope; 64 B alignment |
| SVM / USM | coarse-grained SVM; USM host and device with atomics |
| Global cache | 768 KiB, 64 B line |
| Device IP | 9.20.0 (Gen9.5) |

### Programming interfaces [SYS]

| API | Version / driver |
|---|---|
| OpenGL | 4.6 Core (Mesa 25.2.8), GLSL 4.60 |
| OpenGL ES | 3.2 |
| Vulkan | 1.4.318 (Mesa) |
| OpenCL | 3.0 (Intel NEO 24.35.30872.36), OpenCL C 1.2, cl_khr_fp16, cl_khr_fp64, subgroups |
| VA-API | iHD driver 26.4.1 |
| OpenVINO | installed, sees CPU and GPU devices |

Note: Intel lists OpenGL 4.5, while the Mesa driver exposes 4.6.

### Hardware video (VA-API) [SYS]

| Codec | Decode | Encode |
|---|---|---|
| H.264 (CBP / Main / High) | yes | yes (also LP mode) |
| HEVC Main (8-bit) | yes | yes |
| HEVC Main10 (10-bit) | yes | yes |
| VP9 profile 0 and 2 | yes | no |
| VP8 | yes | yes |
| MPEG-2 | yes | yes |
| JPEG | yes | yes |
| VC-1 | yes | no |
| AV1 | no | no |

### Topology and driver state [DBG]

Source **[DBG]**: i915 debugfs (`/sys/kernel/debug/dri/1/`, root only).

| Parameter | Value |
|---|---|
| Graphics / media version | 9 / 9, GT2 |
| Platform in i915 | COFFEELAKE (Whiskey Lake is a variant of it) |
| Topology | 1 slice (mask 0x1), 3 subslices (mask 0x7), 8 EU per subslice, 24 EU in total |
| Power gating | EU: yes; slice and subslice: no |
| LLC shared with CPU | yes |
| RC6 / RPS / runtime PM | yes / yes / yes |
| Stolen memory | 32 MB |
| PPGTT | 48-bit, IOMMU enabled |
| Engines | rcs0 (render/compute), bcs0 (blitter), vcs0 (video codec, MFX), vecs0 (video enhancement) |
| CS timestamp clock | 12 MHz (83 ns) |
| GPU memory pages | 4 KB (`has_64k_pages: no`) |

### Firmware and clocks [DBG]

| Element | State |
|---|---|
| GuC | kbl_guc_70.1.1, RUNNING, no failures in logs |
| HuC | kbl_huc_4.0.0, RUNNING, authenticated for all workloads |
| DMC (display) | kbl_dmc 1.4 |
| `i915.enable_guc` | 2: HuC is loaded, job scheduling through execlists, not through GuC |
| RPn / RPe / RP1 | 300 / 300 / 300 MHz |
| RP0 / boost / max | 1100 / 1100 / 1100 MHz (no overclocking) |
| RPS thresholds | up at 95%, down at 85% busy |
| Video Turbo Mode | yes |

Idle-desktop reference measurement, 2026-11-11 (`intel_gpu_top`, light office work): about 487 MHz, RC6 31%,
GPU 2.2 W, whole processor 5.7 W. The video decoder was busy 1.3%. The engines ran without resets or hangs.

## Processor: Intel Core i5-8265U [ARK]

| Parameter | Value |
|---|---|
| Family | 8th generation Core i5, code name Whiskey Lake, mobile |
| Lithography | 14 nm |
| Cores / threads | 4 / 8 |
| Base / max turbo clock | 1.60 GHz / 3.90 GHz |
| Cache | 6 MB Intel Smart Cache |
| Bus speed | 4 GT/s |
| TDP | 15 W |
| cTDP-up | 25 W (base 1.80 GHz) |
| cTDP-down | 10 W (base 800 MHz) |
| Memory types | DDR4-2400, LPDDR3-2133 |
| Max memory / channels | 64 GB / 2 |
| Max memory bandwidth | 37.5 GB/s |
| ECC | no |
| Launch | Q3 2018 |
| Status | Discontinued |
| End of servicing (Servicing Updates and Interactive Support) | 31 March 2026 |

Installed RAM on the test system [SYS]: 15 GiB visible.

## Practical consequences for BLAS

- Compute peaks: FP32 422.4, FP16 844.8, FP64 105.6 GFLOPS at 1.1 GHz; there are no integer dot-product or matrix units.
- Memory-bound routines (BLAS levels 1 and 2) are limited by RAM shared with the CPU: about 20 GB/s measured for the GPU
  against 37.5 GB/s theoretical, so dual-channel operation matters. See [performance-model.md](performance-model.md), sections 2 and 6.
- The GPU and the CPU share one power budget; sustained load behaviour is described in [performance-model.md](performance-model.md), section 4.
