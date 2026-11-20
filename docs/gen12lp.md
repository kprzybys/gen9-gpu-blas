# Gen12LP (Tiger Lake, Xe-LP) port: facts before code

Status (2026-11-19): **G1 microbenchmarks through NEO (section 6); G2-G4 done: the FP16/FP32 library runs through kcl on
Gen12LP and passes its test suites (section 8, `results/gen12lp-a1-2026-11-19.txt`); Gen9/Gen8 kernel tables unchanged.
2026-11-20: Qwen3 attention from `asm/attn.cl` compiled by IGC; Qwen3-Embedding through the ggml GEN9 backend beats
llama.cpp Vulkan on the same GPU (section 9).** This file collects what has to change in kcl, in the kernels and in
the build for a Gen12LP GPU (Tiger Lake UHD Graphics G4, 48 EU), following the method of
[performance-model.md](performance-model.md) and the Gen8 port ([gen8.md](gen8.md)): facts with sources first, then
a model with written predictions, then one change at a time and a test. Work happens on the `gen12` branch; `main`
stays unchanged until the Gen9 tests pass unchanged on the Gen9 machines.

Conventions as in [gen8.md](gen8.md): **[SPEC]** Intel documentation or Mesa genxml; **[SW]** driver or compiler
source of a named version; **[MEAS]** read on the target machine without submitting GPU work (i915 debugfs, sysfs,
`clinfo`, `dmidecode`); **[EXP]** compiler experiment (ocloc, no GPU work); **[VENDOR]** Intel marketing/architecture
material, not a specification; **[HYP]** unconfirmed.

Target machine: a Dell Vostro 15 3510 with a Core i3-1115G4, Debian testing, Linux 7.3.4, intel-opencl-icd (NEO)
26.27.39122, IGC 2.40.14.

## 1. Target hardware

| Quantity | Value | Source |
|---|---|---|
| GPU | Tiger Lake-LP GT2 "UHD Graphics G4", PCI 0x9a78 rev 1, graphics version 12 | [MEAS] `lspci -nn`, i915 debugfs `i915_capabilities` |
| EU / layout | 48 EU = 1 slice x 3 dual-subslices (i915 calls them subslices) x 16 EU; masks 0x7 and 0xffff each | [MEAS] debugfs `gt0/sseu_status`, `gt0/sseu_topology` |
| Threads | 7 per EU [VENDOR, HYP until read from NEO hw_info]; 48 x 7 = 336 | |
| GPU clock | 100-1250 MHz; RP1 = RPe = 350 MHz; no overclock range | [MEAS] debugfs `gt0/frequency`, sysfs `gt_*_freq_mhz` |
| Peak (1.25 GHz) | FP32 8 FMA per EU per clock = 16 FLOP: **960 GFLOPS**; FP16 2x = 1920; INT8 DP4a 4 MAC per lane: 8 x 4 x 2 = 64 op per EU per clock = **3840 GOPS**; **no FP64** | [VENDOR] Xe-LP: one 8-wide FP/INT ALU + 2-wide extended math per EU (Gen9: two 4-wide FPUs); FP64 absence [MEAS] `clinfo` (no `cl_khr_fp64`) |
| Ratio to the UHD 620 (24 EU, 1.1 GHz, 422 GFLOPS FP32) | FP32/FP16 **2.27x**; INT8 has no Gen9 counterpart (Gen9 has no DP4a) | [performance-model.md](performance-model.md) |
| L3 | OpenCL reports a 1.875 MiB read/write global memory cache, 64 B lines | [MEAS] `clinfo` |
| SLM | 64 KB per work-group; max work-group 512 (Gen9: 256) | [MEAS] `clinfo` |
| Sub-groups | SIMD 8, 16 and 32; max 64 sub-groups per work-group | [MEAS] `clinfo` |
| Memory | 2 x 8 GB DDR4-2667, 64-bit each, dual channel: **42.7 GB/s** theoretical (UHD 620 machine: 37.5, i.e. only **1.14x**); shared with the CPU | [MEAS] `dmidecode -t 17` |
| CPU LLC | 6 MB, shared with the GPU (`has_llc: yes`, `has_snoop: no`) | [MEAS] `lscpu`, `i915_capabilities` |
| GPU address space | 48-bit PPGTT (`ppgtt-size: 48`, type 2 = full) | [MEAS] `i915_capabilities` |
| Timestamp | OpenCL profiling timer resolution 52 ns, consistent with 19.2 MHz [HYP: CS timestamp frequency not yet read through GETPARAM]; Gen9: 12 MHz | [MEAS] `clinfo` |
| Engine scheduling | rcs0: `preempt_timeout_ms` 7500, `heartbeat_interval_ms` 2500, `timeslice_duration_ms` 1; execlists (GuC submission off, GuC used only to authenticate HuC) | [MEAS] sysfs `engine/rcs0/*`, `dmesg` |
| Power | i3-1115G4: CPU and GPU share one package power budget (PL1 15 W nominal); the sustained GPU clock under load is unknown | [VENDOR]; [HYP] |
| Accessible | user in `render` and `video`; `/dev/dri/renderD128` | [MEAS] |

Consequence for the model: compute-bound work (GEMM prefill, FP16/FP32) can gain up to 2.27x over the UHD 620,
memory-bound work (GEMV, decode) only about 1.14x, and quantized weights gain most where DP4a replaces Gen9's
int8 -> int16/float conversions.

## 2. Compiler

- **ocloc 24.35 (the version used for all Gen9 builds) does not know Gen12**: `-device tgllp` fails with "Cannot get HW
  Info"; its device list ends at Gen11 (`icllp`, `ehl`, `jsl`) [EXP]. 24.35 is the last release with Gen8-Gen11.
  Gen12 needs a second ocloc: 26.27.39122 (the version matching the NEO on the target) works, unpacked from the
  Debian `intel-ocloc` package into a user directory (no system install) [EXP].
- Output format is **zebin** (ELF with `.ze_info`, `.text.<kernel>`, `.spv`, `.misc.buildOptions`,
  `.note.intelgt.metrics`); `ocloc disasm` 26.27 writes only `sections.txt` for it, so the assembly is taken from
  IGC dumps (`IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=...`) [EXP].
- **The library sources do not compile for Gen12 as they are**: `hgemm.cl` and `sgemm.cl` (block inverses
  `trsm_inv0_*`/`trsm_inv1_*` accumulate in `double`), `hgemm.cl` (a reduction "sum in double") and `level12.cl`
  without `-DDBL` (line 207 uses `double`) fail with "use of type 'double' requires cl_khr_fp64 support"; `dgemm.cl`
  and `level12.cl -DDBL` cannot compile at all [EXP]. FP64 emulation (`IGC_EnableDPEmulation=1`,
  `-cl-fp64-gen-emu`) did not help: the build still fails [EXP]. Needed: `#ifdef cl_khr_fp64` guards with an FP32
  (or compensated) fallback for these helpers, and no FP64 programs on Gen12.
- With `double` mapped to `float` (only to get statistics, not a correct build) all FP16/FP32 programs compile
  (hgemm 74, sgemm 27, level12 FP16 33, FP32 33 kernel bodies) [EXP].
- **GRF**: every kernel gets `numGRF=128` (4 KB per thread, as on Gen9), `numAcc=2`, `numSWSB=16` (16 scoreboard
  tokens) [EXP, IGC dump header].
- **Spills** (kcl has no scratch support, so these kernels would be marked unsupported) [EXP]:

  | Kernel | SIMD | spill size | on whl (Gen9.5) |
  |---|---|---|---|
  | `hgemm_16x32kc` | 16 | 896 B | spills too ([gen8.md](gen8.md) section 2) |
  | `hgemm_16x32mx` | 16 | 1472 B | spills too |
  | `gen9blas_dequant_b16` | 16 | 3424 B | no |
  | `skinny_n4_h` | 16 | 768 B | no |
  | `skinny_n4_f` | 16 | 9408 B | no |

- **SIMD width**: kernels without `intel_reqd_sub_group_size` get SIMD32 on Gen12 in most cases (packing
  `pack_a/b*`, `copy16_f`, `gemv_t1b_f`, `gemv_t2_f`, `syr2_f`, ...: 52 of 74 bodies in hgemm, 23 of 27 in sgemm,
  12 of 33 in level12 FP32) [EXP]. kcl and the descriptor reader must handle SIMD32 dispatch (to be checked in
  `lib/kcl.c`); kernels that assume a sub-group size must declare it.
- **64-bit integer add is emulated**: IGC passes `-hasNoInt64Add` to vISA for TGLLP, and a trivial
  `c[get_global_id(0)]` kernel already contains 4 `addc` pairs [EXP]. `size_t` index arithmetic costs more than on
  Gen9; `int`/`uint` indices where the range allows it.
- **DP4a is emitted**: for `dot_acc_sat_4x8packed_ss_int` (`cl_khr_integer_dot_product`), for
  `dot_4x8packed_ss_int`, and also for a plain `char4` multiply-add written by hand, IGC emits
  `dp4a (16|M0) ... :d` (two SIMD16 halves of a SIMD32 kernel) [EXP, `dp4a.cl` probe]. So int8 dot products need no
  intrinsics or assembly.
- `.full_options` of TGLLP kernels include `-activeThreadsOnlyBarrier`, `-waClearHDCWrite`, `-PVCSendWARWA`,
  `-enablePreemption` [EXP]; their effect on a hand-written command stream is in section 5.

## 3. ISA differences that matter for `asm/` and hand-written binaries

Details with file:line and PRM pages: [gen12lp-sources-notes.md](gen12lp-sources-notes.md) (sources in `third_party/`,
fetched 2026-11-18: Linux v7.3 i915, NEO 26.27.39122.14, Mesa 25.2.8 genxml, IGC 2.40.14 headers, Tiger Lake Open
Source PRM vol 2a-14 with SHA-256 sums).

- **Software scoreboarding**: Gen12 instructions carry SWSB annotations (RegDist, SBID tokens, `sync`); the hardware no
  longer tracks register dependencies for the assembler. New instruction encoding; `send` replaces `sends`/`sendsc` [SPEC, SW].
- No FP64, no native 64-bit integer add (`-hasNoInt64Add`), no DWord x DWord `mul` at full rate [SPEC, EXP].
- **Consequence: the Gen9 assembler binaries (`gen_h.bin`, `gen_hd.bin`, `gen_s.bin`, attention `*_tokc.bin`) and the
  sources in `asm/` cannot be reused.** Gen12 starts with the OpenCL C kernels only; a Gen12 assembler path (IGA syntax
  with SWSB) is a separate later step, and only where OpenCL C demonstrably loses.

## 4. Driver (i915, Linux 7.3)

| Topic | Fact | Source |
|---|---|---|
| Coherency | `has_llc: yes`: GPU and CPU share the LLC, so Gen9's model (WB mmaps, no clflush) applies; the Gen8 non-LLC fixes are not needed | [MEAS], [SW] `i915_pci.c` |
| MOCS | index into a per-platform table; on TGL index 2 = "L3 + LLC, write-back", 3 = uncached. kcl's index 2 is already right | [SW] `intel_mocs.c:164-176` (`GEN11_MOCS_ENTRIES`), 264 (`tgl_mocs_table`) |
| L3 configuration | `0x7034` (L3CNTLREG) is reserved on Gen12; the L3 register is L3ALLOCREG `0xB134` (NEO's value lives in `gen12lp/reg_configs.h:22`, but NEO emits no L3 LRI at runtime) | [SW] NEO; [SPEC] notes section 2 |
| `0xE404` (thread arbitration) | not on the user-writable list: a user-batch LRI is dropped with a violation error. On Gen9 round-robin gave about 1.7x on long-K SGEMM (362 vs 211-220 GFLOPS) ([performance-model.md](performance-model.md)); on Gen12 this knob is gone [HYP: whether the default arbitration is already better on Gen12 is unmeasured] | [SW] `intel_workarounds.c`, `i915_cmd_parser.c` |
| Preemption | thread-group level on Gen12 (Gen9: command level); rcs0 `preempt_timeout_ms` 7500, so one thread group longer than 7.5 s still gets the context reset [HYP, inferred] | [SW] `intel_workarounds.c`; [MEAS] sysfs |
| Timestamp | 19.2 MHz through `I915_PARAM_CS_TIMESTAMP_FREQUENCY` (kcl already reads it since the Gen8 port) | [SW] `intel_gt_clock_utils.c` |

## 5. Command stream differences (kcl emits, Gen9 -> Gen12LP)

| Command / field | Change | Source |
|---|---|---|
| GPGPU_WALKER, MEDIA_VFE_STATE, MEDIA_INTERFACE_DESCRIPTOR_LOAD, MEDIA_STATE_FLUSH, MI_BATCH_BUFFER_END | kept with the same lengths (walker 15, VFE 9, IDL 4 DW); COMPUTE_WALKER is XeHP and later only | [SPEC] genxml gen120 |
| packs | generate `gen12_pack.h` (GFX12); buffer RENDER_SURFACE_STATE fields unchanged | [SPEC] |
| STATE_BASE_ADDRESS | 22 DW (Gen9: 19); new DW 19-21 (bindless sampler base/size) may stay 0 | [SPEC] |
| STATE_COMPUTE_MODE | new, 2 DW; NEO sets Force Non-Coherent = 2 | [SW] NEO |
| PIPELINE_SELECT | same 1 DW, but Wa_1607854226: PIPE_CONTROL (CS stall, RT/depth/HDC flush) -> PIPELINE_SELECT(3D) -> STATE_BASE_ADDRESS -> PIPELINE_SELECT(GPGPU) before MEDIA_VFE_STATE; NEO and Mesa both do this | [SW] NEO, Mesa `iris_state.c` |
| PIPE_CONTROL | new HDC Pipeline Flush bit (DW0 bit 9); never set Constant Cache Invalidate (PRM vol14 hang note); CS stall + RT flush + HDC flush before SBA | [SPEC], [SW] |
| LRI `0x7034`, `0xE404` | drop both on Gen12 (section 4); L3 stays at the kernel's default unless an LRI to `0xB134` proves useful (test) | |
| MEDIA_VFE_STATE | max threads 48 x 7 - 1 = 335; **URB entry allocation 0x782 (kcl's Gen9 value) violates the 32 KB limit, NEO uses 1024**; new DW3 bit 6 Fused EU Dispatch (1 = legacy), to be driven by `require_disable_eufusion` from `.ze_info` | [SPEC], [SW] NEO |
| INTERFACE_DESCRIPTOR_DATA | layout and SLM encoding as Gen9 (1K = 1 ... 64K = 7, so kcl's non-Gen8 branch); at most 64 threads per group from the walker's width field | [SPEC] |
| SIMD32 | GPGPU_WALKER SIMD size field 2 = SIMD32; to be checked in kcl and `gen_kernels.py` (section 2: most packing and level-2 kernels are SIMD32 on Gen12) | [SPEC], [EXP] |

Unresolved (notes section 7): L3 size (OpenCL reports 1920 KB, the PRM table gives 2880 KB for 1 x 3 x 16); the
stepping of this 0x9a78 part (NEO's barrier workarounds apply to A0-B only); the value of IGC's `waClearHDCWrite` for TGL.

## 6. Model and predictions (before any measurement)

Written before running anything, so that a mismatch shows a gap in the model.

| Measurement (OpenCL through NEO, `bench/clbench`, `bench/gen12_g1.sh`) | Prediction | Reasoning | Measured 2026-11-18 [MEAS] |
|---|---|---|---|
| FP32 FMA chains (`peak.cl k_fmav`) | 850-930 GFLOPS if the GPU holds about 1.2 GHz; less if the package power limit lowers the clock | 960 peak; the UHD 620 reached 97 % | **945 GFLOPS (98.5 %)**, clock held 1250 MHz (short run, CPU idle): slightly above the range, the power limit did not bite |
| FP16 FMA | 1650-1850 GFLOPS | 2x FP32; UHD 620 reached 95 % | **1811 (94 %)** |
| INT8 DP4a chains | 3.0-3.7 TOPS | 3840 peak; dp4a is a single 3-source ALU instruction like `mad` | **8 chains: 2.93 TOPS (76 %), below the range; 16 chains: 3.59 TOPS (93.5 %)**. Gap in the model: latency. A dependent dp4a (SIMD32) takes 14.6 clocks, FP32 `fma` 11.0, so 8 accumulators per SIMD32 thread do not hide it. Saturating and plain forms are equal |
| Block read bandwidth from RAM (>= 64 MB) | 25-32 GB/s | 42.7 theoretical; the UHD 620 reached 19-27 of 37.5 (50-72 %) | **29.4-31.8 GB/s (69-74 %)** |
| Read bandwidth from L3 (<= 1 MB) | [HYP] about the UHD 620 per clock (1 slice), so about 1.1-1.2x in GB/s | no source yet | **217 GB/s at 1 MB** (UHD 620: 156 with the same OWord x2 kernel, 1.39x); 8 MB (beyond L3 + LLC share) 126 GB/s |

Raw output: `results/tigerlake-g4-g1-2026-11-18.txt`. A first DP4a kernel with loop-invariant operands was folded by
the compiler (20 TOPS, invalid); the kernel now feeds the accumulator back as an operand.

**Rule for Gen12LP int8 kernels [MEAS]: at least 16 independent dp4a accumulators per SIMD32 thread** (or more threads
with fewer each) to reach more than 90 % of the DP4a rate.

What this means for the first real kernel: a Q8_0/Q4 GEMV (decode) is memory-bound on both GPUs, so DP4a will not make it
faster than about 1.1-1.3x the UHD 620; a quantized GEMM (prefill) is where DP4a can show: int8 at up to 4x the FP32
rate means dequantization to FP16 (the Gen9 path) is no longer the cheapest way.

## 7. Plan

| Step | What | Where | Needs |
|---|---|---|---|
| G1 | microbenchmarks through NEO (no kcl): FP32/FP16 FMA, DP4a, memory bandwidth; compare with section 6 | `bench/clbench`, `bench/peak.cl`, a DP4a kernel, `bench/memory.cl` on the target | nothing in the library |
| G2 (done) | `#ifdef cl_khr_fp64` guards in `hgemm.cl`, `sgemm.cl`, `level12.cl`; Gen9 builds unchanged (golden) | `lib/kernels/` | G1 not needed |
| G3 (done) | second ocloc in the build (`GEN9BLAS_OCLOC_GEN12`), `gen_kernels.py` target `tgllp` (zebin, SIMD32, no assembler binaries) | `lib/CMakeLists.txt`, `lib/gen_kernels.py` | G2 |
| G4 (done) | kcl Gen12 command stream and driver changes (sections 4-5) | `lib/kcl.c`, a `kcl_gen12.c` like `kcl_gen8.c` | G3, facts in 4-5 |
| G5 | first DP4a kernel (Q8_0 GEMM for prefill), measured against the section 6 prediction | `lib/kernels/`, ggml backend later | G1, G4 (or through NEO first) |

Open questions: the sustained GPU clock under the package power limit; whether kcl handles SIMD32; Gen12 versions of the
five spilling kernels; FP64 users in the library API (`gen9blas_dgemm` etc.) must report "unsupported" on Gen12.

## 8. G2-G4 result (2026-11-19) [MEAS]

What was built (branch `gen12`):
- **G2**: `trsm_inv*` in `hgemm.cl` / `sgemm.cl` accumulate in `INV_T` (double with `cl_khr_fp64`, float without);
  `level12.cl` `F(red)` without FP64 sums in FP32 with compensation (Neumaier per lane, TwoSum tree in SLM) and stores a
  float pair per work-group, which `lib/src/level12.c` adds in double. Preprocessor-only on Gen9/Gen8: `kernels.c` and
  `kernels_gen8.c` are byte-identical to the tables before the change.
- **G3**: `lib/gen_kernels.py tgllp` (ocloc from `OCLOC`, `.ze_info` read from the ELF section, FP16/FP32 programs
  only, `require_disable_eufusion` / `has_dpas` marked unsupported), CMake `GEN9BLAS_OCLOC_GEN12` (version-checked,
  26.27.039122) or a prebuilt `kernels_gen12.c`. 166 kernel bodies, 92 of them SIMD32; 160 supported.
- **G4**: `lib/kcl.c` detects Tiger Lake by PCI id; batch preamble: PIPE_CONTROL (CS stall, DC, RT, depth, HDC flush)
  -> PIPELINE_SELECT 3D -> STATE_BASE_ADDRESS (22 DW) -> STATE_COMPUTE_MODE (force non-coherent) -> PIPELINE_SELECT GPGPU
  -> PIPE_CONTROL (no constant cache invalidate) -> MEDIA_VFE_STATE (URB 1024) -> MIDL; no L3CNTLREG, no 0xE404; HDC
  flush in the barrier between kernels. The other commands use the Gen9 packs: genxml gen120 keeps every field kcl sets
  at the same bits (checked field by field). SIMD32 dispatch needed no change.
- Library: no FP64 type, main GEMM kernels from OpenCL C (`hgemm_v`, `hgemm_kc_v`, `sgemm_v`; tgllp compiles with
  `-DPLAIN_PACK_A2` / `-DGEN8_PLAIN_PACK`, because those kernels read plain packed tiles, not the assembler layouts).

Found on the way (each by a failing run):
- i915 has **no read-only PTEs on Gen11/12** (`gen8_ppgtt.c:1025`): `I915_USERPTR_READ_ONLY` fails with ENODEV. kcl makes
  userptr objects writable on Gen12 and copies ranges that are mapped read-only (checked in `/proc/self/maps`) -- this
  matters for read-only mmap'd model files in the ggml backend.
- `skinny_n4_h/_f` spill on Gen12 (768 / 9408 B; less unrolling does not remove it): skinny GEMM with weights contiguous
  along N falls back to the packed path (FP16 through the accurate kernel).
- `gen9blas_dequant_b16` does not spill with the real build (the 3424 B in section 2 came from the double -> float
  statistics build).

Tests on the Gen12 GPU: basic, errors, level3, level12, userptr, packed, long_launch all OK; skinny 430/432 (two FP32
cases 2.6e-6 against a 2e-6 tolerance, from the fallback above); blas, blas_d, blas_banded OK after installing
OpenBLAS on the host.
GEMM through kcl: hgemm 1144 GFLOPS (63 % of the measured FP16 peak), accurate hgemm 951, sgemm 551 (58 % of FP32).

## 9. Attention and Qwen3-Embedding (2026-11-20) [MEAS]

The Gen9 assembler attention cannot run on Gen12 (no SWSB), so the Gen12 table gets `attn_qwen3_tokc.bin` from the
OpenCL C template `asm/attn.cl` compiled by IGC (`-DD=128 -DG=2 -DHKV=8 -DRQ=2 -DNK=2 -DQTOK -DCAUSAL=1 -DCLMASK`).
`-DCLMASK` masks the keys past `nv` inside the last partial key block, which the Gen9 binaries handle in assembler.
The Gen9 attention binaries rebuilt on a Gen9 machine still match `asm/attn_sha256.txt`.

Qwen3-Embedding-0.6B Q8_0 through llama-server with the flags of kp-docsearch, 32 texts of about 1200 characters per
request, 5 minutes each (`ggml-intel-igpu` `bench/embed_rate.py`, results in its
`results/embed-qwen3-gen12-2026-11-20.txt`). Control vector cosine against the Gen9 reference is 0.999997 in every run,
with no GPU resets:

| Backend | fragments/h | CPU cores used |
|---|---|---|
| GEN9, GPU attention, `-t 4` | 5673 | 2.84 |
| GEN9, GPU attention, `-t 2` | 5412 | 1.02 |
| GEN9, GPU attention, `-t 1` | 5065 | 0.33 |
| GEN9, attention on the CPU, `-t 4` | 4578 | |
| llama.cpp Vulkan, `-ngl 99` | 4796 | 0.21 |

For a server running all day, `-t 1` or `-t 2` is the sensible choice: still faster than Vulkan at a fraction of the CPU.
Backend `test_attn` on Gen12 (`ggml-intel-igpu` `results/test-attn-gen12-2026-11-20.txt`): all mask modes of the Gen9
suite pass (NMSE <= 2.2e-12, bound 5e-4), every node on the GPU, 2.2x the 4-thread CPU at a typical 430-token chunk
and 3.2-3.3x at 2048 tokens; shapes without a Gen12 binary (whisper D 64, Qwen2.5-7B) stay on ggml-cpu. Gen9 golden
on the second UHD 620 machine: 11/11 suites, kernel tables byte-identical (`results/gen12-golden-gp-2026-11-20.txt`).

