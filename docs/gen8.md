# Gen8 (Cherryview/Braswell) port: facts before code

Status (2026-11-13): **implemented on the `gen8` branch (section 6); FP32 and FP64 GEMM, TRSM and level-1 kernels run correctly on the Cherryview (sections 6a, 6b).** This file collects what has to change in
kcl and in the kernels for a Gen8 Cherryview GPU, following the method of [performance-model.md](performance-model.md):
facts with sources first, then one change at a time and a test. Work happens on the `gen8` branch; `main` stays Gen9-only
until the Gen9 tests pass unchanged on both Gen9 machines.

Conventions: **[SPEC]** Intel PRM for Cherryview/Braswell (`third_party/txt/intel-gfx-prm-osrc-chv-bsw-*`; **p. = the page
number printed on the page**, the PDF page is higher by 35 in vol07, 4 in vol04, 17 in vol02a, 39 in vol02c, 12 in vol02d)
or Mesa genxml; **[SW]** driver or compiler source (Linux v7.0 i915, Mesa 25.2.8, NEO 24.35.30872.36); **[MEAS]** read on the
target machine (i915 getparam, topology query, debugfs, sysfs; no GPU work submitted); **[SCAN]** static scan with
`scripts/chv_scan.py` (kernels compiled with `ocloc -device bdw`); **[EXP]** compiler experiment; **[HYP]** unconfirmed.
All cited sources: `sh scripts/fetch_docs.sh`.

## 1. Target hardware

| Quantity | Value | Source |
|---|---|---|
| GPU | Cherryview/Braswell "HD Graphics 400", PCI 0x22b1 rev 0x35, graphics version 8 (Celeron N3010, 2 cores) | [MEAS] i915 debugfs `i915_capabilities` |
| EU / threads / layout | 12 EU (1 slice, 2 subslices x 6 of 8 EU), 7 threads/EU = 84 threads; 42 per subslice | [MEAS] GETPARAM EU_TOTAL, topology query; [SPEC] vol04 p.5 (2x6) |
| GPU clock | 200-600 MHz (RPe = RP1 = 320 MHz) | [MEAS] sysfs, `i915_frequency_info` |
| Peak (600 MHz) | FP32 192 FLOP/clk = **115 GFLOPS**; FP16 384 = 230; **FP64 24 = 14.4** (1/8 of FP32) | [SPEC] vol04 p.5 |
| L3 / SLM | 384 KB in 2 banks; SLM 0 or 128 KB; no LLC/L4 | [SPEC] vol04 p.5 |
| Memory | one DDR3-1600 module, single channel: **12.8 GB/s** theoretical (UHD 620: 37.5) | [MEAS] dmidecode |
| GPU address space | 32-bit per-context PPGTT: **4 GB** (Gen9: 48-bit) | [MEAS] `ppgtt-size: 32`, CONTEXT_PARAM_GTT_SIZE = 4096 MB |
| CPU cache sharing | `has_llc: no`, `has_snoop: yes` | [MEAS] `i915_capabilities`; [SW] `i915_pci.c` (chv_info) |
| IOMMU | off for the GPU (no DMAR table) | [MEAS] |
| Timestamp counter | 12.5 MHz (80 ns/tick, 36 bits); Gen9: 12 MHz | [MEAS] GETPARAM CS_TIMESTAMP_FREQUENCY; [SW] `intel_gt_clock_utils.c:117`; [SPEC] vol02c p.1174 (width) |
| Reset | per-engine reset (`reset=3`), hangcheck on; no GPU hang in this machine's kernel log | [MEAS] |

## 2. Compiler

- `ocloc` 24.35 (the version used for all Gen9 builds) knows Gen8 only as Broadwell: `-device bdw` works, `chv`, `bsw` and
  `0x22b1` fail with "Cannot get HW Info" [EXP]. Its device list for Gen8 contains only `bdw` [EXP]; whether any NEO
  release ever supported Cherryview was not checked.
- The bdw output is a patch-token binary (`PTM.txt`, `*_KernelHeap.asm`), not zebin, and `-internal_options
  -cl-intel-enable-zebin` does not change that [EXP]. `lib/gen_kernels.py` therefore has a patch-token reader (token
  layouts: IGC `patch_list.h`/`patch_shared.h`, mapping as in NEO `kernel_descriptor_from_patchtokens.cpp`);
  `make -C lib kernels_gen8.c` builds the Gen8 table `kcl_programs_gen8` (the six `.cl` programs of the library, `-noFP64bRegion`; kernels with A64 QWord messages are marked unsupported; the probes of `bench/gen8_probe.cl` go only into `kernels_gen8_probe.c` for `lib/test_gen8_probe`).
- Check of the reader (`make -C lib check-patchtokens`): Gen9.5 compiled in both formats; of the kernels with identical
  ISA, 235 have identical descriptors and 8 differ only in accepted ways (both readers' rules in
  `lib/check_patchtokens.py`), 0 errors; 16 kernels get different code from the two compile paths and are not compared
  [EXP]. Side finding: 6 Gen9 GEMV kernels (`gemv_n_*`, `gemv_nw_*`) declare 1 KB scratch in `.ze_info`
  (`per_thread_memory_buffers`, not read by the zebin reader) but contain no scratch messages, so Gen9 is unaffected.
- **Scratch**: 8 kernels spill to scratch when built for bdw (`hgemm_16x32`, `hgemm_16x32g/kc/mx/p/q`, `hgemm_v`,
  `gemv_n_f`; whl: only `hgemm_16x32kc`, `hgemm_16x32mx`). kcl has no scratch support and `lib/gen_kernels.py` marks
  such kernels unsupported [SCAN].
- `bench/gen_h.bin`, `gen_hd.bin`, `gen_s.bin` are Gen9 binaries and would need Gen8 versions.
- Not evaluated: Beignet, the older Intel OpenCL stack, is said to have supported Cherryview [HYP, unchecked].

## 3. Cherryview EU restrictions vs code compiled for Broadwell

Cherryview adds rules that Broadwell does not have. When a source or destination is 64-bit (DF/Q/UQ) **or the instruction is
an integer DWord x DWord `mul`** [SPEC vol07 p.823-824; SW `elk_eu_validate.c:1736-1830`, gated on `INTEL_PLATFORM_CHV`]:

| # | Rule |
|---|---|
| R1 | Align1 regions: (a) source and destination horizontal strides aligned to the same qword; (b) VStride = Width x HStride; (c) equal subregister offsets (scalar source exempt). PRM examples of legal forms: `mov (4) r10.0:df r11.0<16;8,2>:f`, `mul (4) r10.0<2>:d r11.0<8;4,2>:d` |
| R2 | no indirect addressing |
| R3 | no ARF operands (accumulator, address, flag), no `mac`, no AccWrEn |
| R4 | no DepCtrl |

Further rules on the same `mul` [SPEC vol02a p.1078]: "[CHV, BSW-A]: DW * W is not supported outside MACH macro"
(A stepping) and "[CHV, BSW]: When multiplying DW x DW, the dst cannot be accumulator"; the same page says 32x32 multiply
is native. Mesa's Gen8 compiler nevertheless never emits DW x DW `mul` on Cherryview (`has_integer_dword_mul = false`,
`intel_device_info.c:497`; Broadwell: true at 422) and lowers it to DW x UW sequences with a GRF destination
(`elk_fs.cpp:3716-3731`, used from 3989) [SW].

Scan of the six library programs (177 kernels) built for bdw with the options of `lib/gen_kernels.py`
[SCAN, `python3 scripts/chv_scan.py DIR`]:

| Hits | default build | with `IGC_VISAOptions=-noFP64bRegion` [EXP] |
|---|---|---|
| 64-bit, R1a stride (e.g. `mov (8\|M0) r68.0<1>:q r12.0<8;8,1>:d`, `mul q <- d x d`, `add q <- q + d`, `shl q <- d`) | 3523 | 0 |
| 64-bit, R1c offset | 37 | 0 |
| 64-bit, width-1 region `<2;1,0>:d` (qword-strided, but fails Mesa's literal R1b check) | 0 | 3509 |
| 64-bit, R2 indirect (`psv_d`, `trsv_block_d`) | 2 | 2 |
| 64-bit, R3: `math.invm` / `madm` with accumulators acc2-9 (FP64 division) | 363 | 363 |
| DW x DW `mul` with non-scalar regions, R1a (mostly `mul` + `mach` triples emulating 64-bit multiplies in index code) | 1359 | 1359 |
| DW x DW `mul`, R1b-w1 / R1c (`skinny_r4_h`) | 8 / 8 | 8 / 8 |
| kernels with no hit (besides the width-1 regions) | **3** | **53** |

- With `-noFP64bRegion` (a vISA option in IGC 1.0.17537.24) all 64-bit stride and offset violations become width-1
  regions. Clean then: `dgemm_v`, `hgemm_16x32` (but it needs scratch), all packing and expand/epilog kernels, `trsm_L/Y/store`,
  `copy16`, `gemv_t1b_f/d`. `sgemm_16x16p` and four `hgemm_16x32*` variants have a single DW x DW `mul`; GEMV/AXPY kernels
  6-39 [SCAN].
- The three kernels clean in the default build still contain scalar DW x DW `mul` (allowed by R1, but never emitted by
  Mesa on Cherryview), so "clean" is not "known safe" [HYP].
- `hgemm_16x32` on bdw computes in FP32 (`mad :f`) and uses FP16 only for conversions, so it would not get the 2x FP16
  rate [SCAN + HYP].

What a violation does on the hardware is **not documented**: wrong results, wrong addresses (page faults) or a hang are
all possible [HYP]. Tried without effect [EXP]: `-force_stos_opt` (identical code), `-32` (still 3096 64-bit stride hits,
changes the argument ABI), `mul24` in the source (still DW x DW), `IGC_ForceNoFP64bRegioning=1`,
`IGC_VISAOptions="-platform CHV"`. Ways forward:

1. **`-noFP64bRegion`** for all kernels (done above); open: does Cherryview accept the width-1 `<2;1,0>` form?
2. **Index code without 64-bit multiplies and DW x DW `mul`** in the remaining kernels. Origin of the 1359 DW x DW hits
   with `-noFP64bRegion` [SCAN]: 922 are `mul`/`mach` sequences emulating 64-bit multiplies (`size_t` index arithmetic,
   e.g. `get_global_id()` times a leading dimension), 411 other 32-bit index multiplies (mostly GEMV, SYMV, skinny GEMM),
   26 in the compiler's prolog of 23 kernels (linearising the 3-D local id for `get_sub_group_id()`). A test kernel
   `a[i * ld]` gives [EXP]: `size_t i` 15 mul/mach, 6 hits; `int i` 2 hits; a `ushort` factor 0 hits (DW x UW with a GRF
   destination, the Mesa recipe; stepping-A restriction above to be excluded); pointer increments instead of the
   multiply 0 hits. So source changes can remove them, at the cost of touching most Gen9 kernels.
3. **Rewrite remaining instructions in the assembly** and reassemble with `ocloc asm -device bdw` (the project already
   assembles kernels, see `asm/`); needs free registers [HYP].
4. FP64 division (`math.invm`/`madm`, R3) and the two indirect 64-bit moves need a separate decision (drop the FP64
   triangular solvers on Gen8 or rewrite them).
5. A hardware test of a kernel against the CPU decides between the remaining unknowns.

## 4. Driver (i915, Linux v7.0)

This section and section 5 describe the situation before the port; `lib/kcl.c` line numbers refer to the code on
`main` before it. What was changed is in section 6.

| Topic | Fact | Source |
|---|---|---|
| CPU/GPU coherency | A GEM_CREATE object on a non-LLC platform is `I915_CACHE_NONE` (not coherent); i915 clflushes it when pages are acquired, and in execbuffer only when `cache_dirty` is set, which plain stores through a WB mmap do not do. **kcl as written may hand the GPU stale data**: batch, ISA and constants (`lib/kcl.c:423-433`) and data buffers | [SW] `i915_gem_shmem.c:589-605`, `i915_gem_object.c:169`, `i915_gem_pages.c:37-42`, `i915_gem_execbuffer.c:2119` |
| Fix A | `I915_GEM_SET_CACHING(I915_CACHING_CACHED)`: allowed when the GPU can snoop (`!HAS_LLC && !HAS_SNOOP` -> ENODEV); gives PAT index 4 with `CHV_PPAT_SNOOP` (uncached is index 3, no snoop). That this makes CPU and GPU coherent in both directions is i915's model, not measured; the hardware never snoops PPGTT page tables and "some other special cycles" [HYP until tested] | [SW] `i915_gem_domain.c:356-364`, `intel_gtt.c:640-662` |
| Fix B | `I915_MMAP_OFFSET_WC` for objects the CPU only writes (batch, ISA, dynamic state) | [SW] i915 mmap types |
| Fix C | `SET_DOMAIN(CPU, write)` before CPU writes sets `cache_dirty`, so the next execbuffer clflushes | [SW] `i915_gem_domain.c:523`, `i915_gem_execbuffer.c:2119` |
| userptr | already `I915_CACHE_LLC` (= snooped on non-LLC); read-only userptr supported on gen8 | [SW] `i915_gem_userptr.c:530`, `gen8_ppgtt.c:1025` |
| Softpin >= 4 GB | fails with **EINVAL** (`range_overflows`); only ENOSPC is retried, nothing is moved silently. kcl starts at `1<<32` and never reuses addresses (`kbo_addr`), so it must allocate below 4 GB and reuse freed ranges | [SW] `i915_vma.c:854-858`, `i915_gem_execbuffer.c:816`; `lib/kcl.c:47, 68` |
| Preemption | none on gen8 (`can_preempt`: `GRAPHICS_VER > 8`); a long kernel owns the engine until it ends | [SW] `intel_execlists_submission.c:3313-3316` |
| Hang handling | heartbeat every 2.5 s raises the pulse priority (MIN, NORMAL, HEARTBEAT, BARRIER) and then resets the engine: about 10-12.5 s after the batch starts [derived from the code]. The 20 s request timeout is armed when the request is submitted, so batches queued behind a long one count too. 3 hangs within 120 s ban the context (execbuf then EIO) | [SW] `Kconfig.profile`, `intel_engine_heartbeat.c:183-196`, `i915_request.c:786`, `intel_reset.c:94-101`, `i915_gem_context_types.h:389`, `i915_gem_execbuffer.c:2560` |
| MOCS | gen8 has no MOCS table (tables only for gen9+); the field is decoded directly: Cherryview [6:5] memory type (3 = WB), [4:3] target cache (3 = L3). kcl's `2<<1` = 0x04 means **uncached, no L3**; Mesa uses **0x78** on gen8 including Cherryview | [SW] `intel_mocs.c:503-508`, `isl/isl.c:205-224`; [SPEC] `gen80.xml:225-236`, vol02d p.236 |
| LRI from a user batch | no command parser and no whitelist on gen8 render; the hardware allows 0x7034 (L3 config) and 0xE404 (TD_CTL2) from non-privileged batches; a disallowed LRI becomes a NOOP | [SW] `i915_cmd_parser.c:961-963`, `intel_workarounds.c:2157`; [SPEC] vol06 (text lines 1270-1290, 1332-1389) |

## 5. Command stream differences (kcl emits, Gen9 -> Cherryview)

| Command / field | Change | Source |
|---|---|---|
| all packs | generate `gen8_pack.h` from `gen80.xml` (`GFX8_*`) | [SPEC] genxml |
| PIPELINE_SELECT | no Mask Bits (15:2 reserved); MEDIA_VFE_STATE must follow it immediately (WaSendDummyVFEafterPipelineSelect) | [SPEC] vol02a p.1104, vol16 |
| STATE_BASE_ADDRESS | 16 DW instead of 19; no bindless fields | [SPEC] `gen80.xml:2741`, vol02a p.1127 |
| PIPE_CONTROL | with Command Streamer Stall, one of RT cache flush, depth cache flush, stall at pixel scoreboard, depth stall, post-sync operation or DC flush must also be set (not in the KBL text). `lib/kcl.c:296-297` sets CS stall with invalidations only | [SPEC] vol02a p.1098 |
| MEDIA_VFE_STATE | max threads 12 x 7 - 1 = **83** (kcl: `24 * 7 - 1`); scratch and URB fields unchanged, URB 0x782 fits (NEO uses it on gen8); Gen8-only Bypass Gateway Control (DW3 bit 6) left 0 like NEO | [SPEC] vol02a, vol04 p.5, `gen80.xml`; [SW] NEO `hw_cmds_generated_gen8.inl:711` |
| INTERFACE_DESCRIPTOR_DATA | SLM size in 4 KB steps: 4K=1, 8K=2, 16K=4, 32K=8, 64K=16 (Gen9.5: 1K=1 ... 64K=7); a barrier must be allocated whenever SLM > 0; at most **42 threads per group** (kcl allows 64) | [SPEC] vol02d p.223; [SW] NEO `gfx_core_helper_gen8.cpp:20-38` |
| L3CNTLREG 0x7034 | same layout; kcl's values 0x60000321 / 0x80000340 match rows of the Cherryview allocation tables (NEO uses them on Broadwell). Bit 9 (set in both) makes parity errors hang the GPU, and "s/w has to initialize URB or SLM to all 0's" once error detection is enabled | [SPEC] vol02c p.675-676, vol07 p.744; [SW] NEO `gen8/reg_configs.h` |
| 0xE404 (TD_CTL2) | bit fields not documented for Cherryview; NEO does not program thread arbitration on gen8 -> do not emit it on gen8 | [SPEC] vol06; [SW] NEO `gen8/hw_cmds_base.h:45` |
| timestamps | 12.5 MHz: `lib/kcl.c:350, 591` hardcode 12 MHz; take I915_PARAM_CS_TIMESTAMP_FREQUENCY | section 1 |
| unchanged | GPGPU_WALKER, MEDIA_STATE_FLUSH, MEDIA_INTERFACE_DESCRIPTOR_LOAD, MI_LOAD_REGISTER_IMM, MI_BATCH_BUFFER_END, RENDER_SURFACE_STATE (differs only in Gen9 fields kcl does not use) | [SPEC] genxml diff |

## 6. Implementation

| Step | What | Where |
|---|---|---|
| D1 | patch-token reader, `kernels_gen8.c` (`-noFP64bRegion`), check against zebin | `lib/gen_kernels.py`, `lib/check_patchtokens.py` |
| D2 | device parameters at init: chipset (CHV = 0x22b0-0x22b3), EU and subslice count, timestamp frequency, LLC, GTT size | `lib/kcl.c` `init()` |
| D3 | Gen8 command stream: PIPELINE_SELECT and STATE_BASE_ADDRESS from `gen8_pack.h` (own file `kcl_gen8.c`: the Mesa headers define the same enumerators), dummy MEDIA_VFE_STATE after PIPELINE_SELECT, no 0xE404, DC flush with the CS stall, MOCS 0x78, L3CNTLREG 0x60000121 always (bit 9 clear, SLM layout; Beignet uses the same value), SLM in 4 KB steps with a barrier, at most 42 threads per group, VFE threads and private memory from EU_TOTAL | `lib/kcl.c`, `lib/kcl_gen8.c` |
| D4 | without LLC: CPU-read objects snooped (SET_CACHING CACHED), CPU-written objects mapped WC + sfence before execbuffer; 4 GB PPGTT: addresses from 1 MB, freed ranges reused after the GPU is done with them | `lib/kcl.c` `kbo_new`, `kbo_addr` |
| D5 | gen9blas on Gen8: FP32 and FP64 only, `sgemm_v` from `sgemm.cl`, grouped GEMV for every N (no `gemv_n_f`), band splitting assumes 50 / 8 GFLOPS | `lib/gen9blas.c` |

The Gen9 command stream is unchanged; after D3 and after D5, `make -C lib check` passed on the HD 530 and the GEMM speeds
matched `results/skylake-hd530-2026-11-12.txt` within 0.6 % (n >= 2048).

## 6a. First run on the Cherryview (2026-11-12, `lib/test_gen8_smoke.py`, one step per run, guarded) [MEAS]

| Step | Kernels | Result |
|---|---|---|
| init | none (no GPU work) | Gen8 detected; FP32 and FP64 available |
| copy (n = 64, 4096, 2^20) | `copy16_f` (only width-1 `<2;1,0>` 64-bit regions) | correct, 1.4 ms in total |
| axpy (n = 64, 4096, 2^20) | `axpy_f` (+ 6 non-scalar DW x DW `mul`) | correct (error 0), 1.9 ms |
| SGEMM n = 16 | `skinny_n4_f`, `skinny_n2_f` | correct |
| SGEMM n = 64, 256 | `pkf_a`, `pkf_b`, `sgemm_v` | **wrong** (relative error ~1), no reset |
| DGEMM n = 16 | `pkd_a` | **GPU hang, engine reset** (detected by kcl; system unaffected) |

So kcl itself works on Gen8 (command stream, 4 GB address space, coherency, patch-token descriptors) and the width-1
64-bit regions of `-noFP64bRegion` executed correctly in `copy16_f`.

Analysis (offline, 2026-11-12):
- **SGEMM: a software bug of step D5, not the hardware.** On Gen8 the main kernel is `sgemm_v` from `bench/sgemm.cl`,
  which reads the plain packed layout of `pack_af`/`pack_bf` (`sgemm.cl:46-53`, 156-181), while the packing kernels
  `pkf_a`/`pkf_b` write the interleaved layout of the Gen9 assembler kernel (`PKF_IDX_A`/`PKF_IDX_B`, `sgemm.cl:131-133`).
  The same arguments, a different data layout; on Gen9 this pair never runs. A numpy emulation of "pack with PKF_IDX,
  multiply as `sgemm_v` reads" with the test's random inputs predicts relative errors 1.4546 (n = 64) and 1.3259
  (n = 256), printed by the test as 1.5e+00 and 1.3e+00 as measured [EXP]. So `pkf_a`, `pkf_b` and `sgemm_v` most likely
  executed correctly [HYP until re-run with a consistent layout].
- **The DW x DW multiply hypothesis is contradicted:** `skinny_n2_f`, `skinny_n4_f` and `axpy_f` ran correctly with
  non-scalar DW x DW `mul` and `mul`/`mach` sequences whose products feed the store addresses [SCAN + MEAS]. Not yet
  exercised: `mach` with a non-zero high part, `math.invm`/`madm` (R3), the indirect moves (R2).
- **DGEMM hang: unexplained.** It is in `pkd_a` (its start timestamp was written, its end not); `pkd_a` has no loop. The
  only instruction forms it uses that no completed kernel used are A64 qword gather/scatter messages and `mov :df`
  [SCAN]. The CHV PRM defines A64 QWord scattered messages (vol07 p.129, 193), but Mesa never emits them [HYP: the
  candidate cause].
- Fixed (`GEN8_PLAIN_PACK` in the bdw build of `bench/sgemm.cl`) and re-run 2026-11-13 [MEAS]: SGEMM n = 16, 64, 100,
  256 and batched 2 x 64 correct (max relative error 1e-7 .. 6e-7; kernels `pkf_a`, `pkf_b`, `sgemm_v`,
  `gen9blas_epilog_fb`, `skinny_*`), no reset. This confirms the layout diagnosis and that these kernels, including their
  DW x DW multiplies and the `mul`/`mach` sequence with non-zero batch index, execute correctly on the Cherryview.
- FP64 copy n = 64 (`copy16_d`: A64 qword gather/scatter and `mov :df`, no multiplies), 2026-11-13 [MEAS]: **GPU hang,
  engine reset** (detected by kcl; system unaffected). So the DGEMM hang does not come from index
  arithmetic; the candidates left are the A64 qword scattered messages (most likely) and DF-typed moves [HYP].
  `-cl-intel-force-enable-a64WA` (internal option) does not change these messages [EXP].
- **Root cause confirmed, 2026-11-13** (`lib/test_gen8_probe`, kernels in `bench/gen8_probe.cl`, compiled stateless) [MEAS]:
  `probe_df_dword` (DF multiply, A64 dword gather/scatter only) is correct (64 of 64); `probe_qword` (the A64 QWord
  gathering read `0x042402FF` of the hanging kernels, integer data, no DF type) **hangs the GPU** (engine reset; system
  unaffected). So on the Cherryview the A64 QWord scattered *read* must not be used. The CHV PRM documents
  the message (vol07 p.193) but has no "A64 Qword Scattered Read" MSD page (vol02a lists Dword read/write and only the
  Qword *write*); Mesa never emits 64-bit memory messages on Gen8 (`elk_fs_nir.cpp`: 64-bit global loads are split into
  32-bit ones). Not tested on its own: the A64 QWord scattered write (the probe's compiler used dword scatter for it).
- Fix (`GEN8_CHV`, defined only in the bdw build, `lib/gen_kernels.py` `GEN8_DEFINES`): element access through macros
  (`RD`/`WR`, `LV4T`/`SV4T`, `V4`/`VG`/`COPYG` in `bench/level12.cl`; `RD`/`WR`/`RDD`/`WRD` in `bench/sgemm.cl` and
  `bench/dgemm_core.cl`, which `lib/gen_dgemm.py` combines into `dgemm.cl`). On Gen9 they expand to the former plain
  indexing (Gen9 `kernels.c` unchanged, md5 02e260eb); on Gen8 FP64 data moves as uint pairs (dword messages). The same
  build also avoids the other two FP64 forms CHV forbids: a `sub_group_broadcast` of a double with a variable lane (an
  indirect DF move, R2; now two dword broadcasts) and FP64 division (`math.invm`/`madm` with accumulators, R3; now an FP32
  reciprocal refined by three Newton steps plus a quotient correction, DF mul/fma only, both operands first scaled into
  [1, 2) by their exponents; zero, infinity and NaN handled on the bit patterns as IEEE does). A CPU emulation with exact
  FMA and a reciprocal error up to 1e-3 [EXP]: all 324 pairs of special and extreme values (signed zeros, inf, NaN,
  subnormals, the largest number, 1e±300) equal IEEE `a / b`, 299 946 of 300 000 random pairs over the whole range are
  correctly rounded and the other 54 are subnormal results off by 1 ulp (double rounding in the final scaling). Scan
  of the Gen8 build [SCAN]: no QWord messages and no R2/R3/R4 hits in any FP32 or FP64 kernel; only the FP16
  `trsm_inv*_h` kernels keep them (FP16 is not used on Gen8).

## 6b. FP64 after the fix (2026-11-13, `lib/test_gen8_smoke.py`) [MEAS]

| Step | Kernels | Result |
|---|---|---|
| dcopy n = 64, 65 | `copy16_d` (hung before the fix) | correct |
| daxpy n = 64, 4096, 2^20 | `axpy_d` | error 0 |
| DGEMM n = 16, 64, 256 | `pkd_a`, `pkd_b`, `dgemm_v` | max relative error 1e-16 .. 4e-16 |
| DTRSM n = 64 | `trsm_inv0..3_d` (emulated division), `trsm_L/Y/store_d`, DGEMM, `gen9blas_epilog_trid` | relative residual 1.1e-16 |

No reset, system unaffected. (The division used then was replaced later by the full-range version above; see
section 6c for its hardware test.)

## 6c. Test suites on the Cherryview (2026-11-13) [MEAS]

FP16 cases are skipped (`gen9blas.available`); the three `test_blas*` suites (drop-in BLAS layer, which forwards to and
compares with OpenBLAS on the CPU) were not run: OpenBLAS is not installed on the target and is not needed by the GPU
library. `test_level12` skips ROTG/ROTMG/ROTM when SciPy is missing (their reference).

| Suite | Result |
|---|---|
| `test_basic` (SGEMM, all transposes, copy / zero-copy) | all OK |
| `test_level3` (GEMM, SYMM, SYRK, SYR2K, TRMM, TRSM, batched, views; FP32, FP64) | all OK |
| `test_level12` (levels 1 and 2, FP32 and FP64) | all OK |
| `test_userptr` | all OK |
| `test_skinny` | **wrong results** exactly in the cases that use `skinny_r4_f` / `skinny_r8_f` (K a multiple of 16, M = 2, 4, 8, B transposed); the cases on `skinny_k*` pass. With `skinny_k*` only: all OK |
| `test_level3` with `GEN9BLAS_MAX_LAUNCH_MS=0.5` | all OK |
| `test_long_launch` (DGEMM n = 4096, SGEMM n = 8192, batched DGEMM 2 x 3072) | first **two GPU resets**: the band-split launches went into one batch, which on Gen8 (no preemption) is one request, longer than the heartbeat allows (~10 s). With one request per kernel on Gen8 (`lib/kcl.c` `enqueue_`): all OK |

`skinny_r*` differ from the passing SLM/barrier kernels (`skinny_k4_f`, `trsv_block_f`, same 16 threads per group) by
SIMD8 and the A64 aligned OWord block read x4 (`intel_sub_group_block_read2`); which of the two (if either) is the cause
is not known [HYP]. They are only the faster variant, so on Gen8 `gen9blas.c` uses `skinny_k*` instead (as with
`GEN9BLAS_SKINNY_R=0`).

## 7. Open questions and known limitations

- Why `skinny_r4/r8` give wrong results (SIMD8? A64 OWord block read x4?); Gen8 uses `skinny_k*` instead (6c).
- The A64 QWord scattered *write* was never tested on its own (no kernel of the Gen8 build uses it any more).
- The accuracy of `native_recip` on the hardware is not measured; `div_chv` tolerates a 1e-3 error in emulation, and the
  hardware test of the division (6c) is the check.
- Snooping: coherent in all tests so far (including command-streamer fetches of the WC-mapped batch); its bandwidth cost
  against uncached memory with clflush is not measured. No performance has been measured on Gen8 at all.
- `est_ms` assumes 50 / 8 GFLOPS (FP32 / FP64) for band splitting, not measured; at the idle clock the peaks are lower.
  With one request per kernel only a single launch has to fit the ~10 s heartbeat; only GEMM is band-split.
- The 4 GB address space: when it is full, kcl waits for the GPU and retries, but does not evict cached userptr objects
  (up to `KCL_USERPTR_MB`, default 1 GB); `KCL_ADDR` (2 MB alignment, colouring) is ignored on Gen8.
- Five GPU hangs during the bring-up (2026-11-12/13) were all engine resets; the system kept running each time.
