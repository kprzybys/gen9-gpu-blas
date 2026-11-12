# Running kernels without NEO: directly on i915

State as of 2026-11-12. Evidence tags (as in `results/*.txt`): [SPEC] Intel documentation; [MEAS] our measurement on the
test machine; [SW] behaviour of a specific software version; [EST] taken from third-party code (Mesa, igt, NEO);
[HYP] unconfirmed hypothesis. Section numbers (6a, 6d, 6g, ...) are cited from code comments and result files; they are stable.

> **Safety warning.** `kcl` (`lib/kcl.c`) builds command-streamer batch buffers itself and submits them to the Linux
> i915 driver with `DRM_IOCTL_I915_GEM_EXECBUFFER2`. There is no NEO validation in between. A bug (wrong command length,
> wrong address in a stateless kernel, an endless loop) can hang the GPU: i915 resets the render engine, the desktop
> may freeze for a fraction of a second up to several seconds, and a context that hangs the GPU repeatedly is banned
> until the process ends (section 6). Everything here was tested **only on one machine: Intel UHD 620 (WHL GT2, device
> 0x3EA0, revision 2), Linux 7.0 i915**. Do not run it on other hardware, and do not run the experiments on a machine
> where an unsaved desktop session matters.

## 1. Why, and what changes

Before: `gen9blas` -> OpenCL API -> NEO (queue, memory, waiting) -> i915 -> GPU. Without NEO: the library builds a
**batch buffer** of command-streamer commands itself and sends it to i915 with one `ioctl`. The kernels (binaries from
`ocloc`, or from our assembler, see `asm/Makefile`) are unchanged. What this was expected to give:

- Submission cost: about 25 us CPU per kernel in NEO ([SW], see `docs/performance-model.md`); without NEO several kernels
  of one operation (packing A, packing B, GEMM, epilogue) go into **one** batch buffer and one `ioctl`.
- Waiting: `DRM_IOCTL_I915_GEM_WAIT` sleeps in the Linux kernel until the interrupt, with no busy loop and no
  `nanosleep` polling of our own [SPEC: i915_drm.h].
- Memory: `GEM_USERPTR` turns ordinary process memory (NumPy arrays) into a GPU object without a copy, with none of the
  USM / `USE_HOST_PTR` pitfalls in NEO [MEAS: HAS_USERPTR_PROBE = 1; LLC = 1, so memory is coherent through the shared cache].
- Full control over synchronisation between kernels (PIPE_CONTROL only where it is needed).

## 2. The i915 interface on the test machine [MEAS] (`experiments/no-neo/probe.c`, Linux 7.0, renderD128, group `render`)

| Parameter | Value | Meaning |
|---|---|---|
| CHIPSET_ID / REVISION | 0x3EA0 / 2 | WHL GT2 |
| EU_TOTAL / SUBSLICE_TOTAL | 24 / 3 | |
| Context GTT (PPGTT) | 2^48 B | full 48-bit addresses, each process has its own address space |
| HAS_EXEC_SOFTPIN | 1 | we choose the object's PPGTT address (`EXEC_OBJECT_PINNED`); no relocations |
| HAS_EXEC_NO_RELOC, HANDLE_LUT, BATCH_FIRST | 1, 1, 1 | |
| HAS_USERPTR_PROBE | 1 | `GEM_USERPTR` with the PROBE flag |
| HAS_LLC | 1 | GPU and CPU coherent through LLC |
| HAS_EXEC_FENCE / FENCE_ARRAY / TIMELINE_FENCES | 1 / 1 / 1 | waiting on fences (sync_file / syncobj) |
| HAS_WAIT_TIMEOUT | 1 | `GEM_WAIT` with a time limit |
| CS_TIMESTAMP_FREQUENCY | 12 MHz | PIPE_CONTROL timestamps (83 ns) |
| HAS_SECURE_BATCHES | 0 | no privileged commands |
| CMD_PARSER_VERSION | 10 | [HYP] on Gen9 the parser covers only the blitter (BCS), not render; not confirmed (see section 7) |
| HAS_SCHEDULER | 31 | priorities, preemption |

## 3. Command sequence of a GPGPU kernel (Gen9)

Sources: PRM KBL Vol. 7 "Programming the GPGPU Pipeline" (pp. 647-665), igt `gpgpu_fill.c` / `gpu_cmds.c`
(`third_party/igt-2.6`), NEO `hardware_commands_helper_base.inl` (`third_party/neo-24.35.30872.36`). Bit-level fields:
`third_party/mesa-25.2.8/genxml/gen90.xml` (+ gen80, gen75, gen60).

```
PIPELINE_SELECT (GPGPU, Gen9 selection mask)
STATE_BASE_ADDRESS        general = 0, stateless MOCS, surface/dynamic/instruction base, sizes (Modify Enable bit!)
MEDIA_VFE_STATE           scratch, max thread count, URB: number and size of entries, CURBE size
MEDIA_INTERFACE_DESCRIPTOR_LOAD   address of INTERFACE_DESCRIPTOR_DATA (relative to dynamic state base)
GPGPU_WALKER              SIMD (8/16/32), threads per group, X/Y/Z group grid, right and bottom edge masks,
                          address and length of indirect data (cross-thread + per-thread)
MEDIA_STATE_FLUSH         after the walker (required before the next MEDIA_CURBE_LOAD; optional with indirect data)
PIPE_CONTROL              synchronisation between dependent kernels, timestamp
MI_BATCH_BUFFER_END
```

- **INTERFACE_DESCRIPTOR_DATA** (gen80 onwards, 8 DW): kernel address (relative to instruction base, 64 B), FP mode,
  binding table (32 B), `Constant/Indirect URB Entry Read Length` = GRFs of per-thread data, `Cross-Thread Constant Data
  Read Length` = GRFs of shared data, threads per group, SLM size, barrier [SPEC + EST: NEO setGrfInfo].
- **Indirect data** instead of CURBE: a pointer in the walker (relative to indirect object base), length aligned to 64 B.
  In memory: cross-thread data first, then per-thread data for each thread of the group [EST: NEO]. In registers the
  order is reversed: "The X/Y/Z payload in the EU GRF comes before the Cross-Thread Constant Data" [SPEC PRM Vol. 7 p. 659].
- **igt pitfall:** the instruction buffer size in STATE_BASE_ADDRESS must have the Modify Enable bit set, otherwise the
  GPU hangs [EST: comment in igt].
- Consecutive walkers with indirect data need no flush between them (CURBE does need MEDIA_STATE_FLUSH) [SPEC p. 659].
- Preemption: a GPGPU_WALKER can be interrupted in the middle of a group (Gen8+), state is saved in the context image [SPEC p. 651].

## 4. Thread payload of IGC kernels [MEAS: disassembly of `hgemm_v` + `.ze_info`]

| Register | Contents | Source |
|---|---|---|
| R0 | GPGPU header: R0.1 = group ID X, R0.6 = Y, R0.7 = Z; R0.5[31:10] scratch pointer, [9] GPGPU, [8:0] FFTID; R0.4 binding table; R0.2 barrier/SLM; R0.0[29:24] SLM index | [SPEC PRM Vol. 7 pp. 662-663] |
| R1-R3 (SIMD16) | local ID X, Y, Z: 16 x u16 per register (`per_thread_payload_arguments: local_id, size 96`) | [MEAS] `hgemm_v` reads r1, r2, r3 as `:uw` |
| R4 onwards | shared data according to `payload_arguments` from `.ze_info` (offset 0 = r4.0) | [MEAS] M (offset 56) = r5.6, local_size (96) = r7.0-2 |

- Local IDs are computed by software (Gen9 does not generate them in hardware): a separate per-thread block for every
  thread of the group [SPEC p. 655; EST: NEO local_id_gen].
- `.ze_info` (from `ocloc disasm`) gives for each kernel: SIMD, GRF count, argument offsets, addressing mode (stateless /
  stateful + binding-table index), scratch and SLM size. `hgemm_v`: A and B stateless (64-bit address in the shared
  data), C stateful (BTI 0), which needs a RENDER_SURFACE_STATE of buffer type (or rewriting the kernel as stateless).

## 5. Memory

- **GEM objects:** `GEM_CREATE` (Linux kernel memory) or `GEM_USERPTR` (existing process memory, 4 KB aligned; used since
  2026-11-12, section 6g). CPU mapping: `GEM_MMAP_OFFSET` (MMAP_VERSION 1, GTT_VERSION 5) [MEAS].
- **Addresses:** softpin: we choose the object's PPGTT address (`EXEC_OBJECT_PINNED | EXEC_OBJECT_SUPPORTS_48B_ADDRESS`),
  the same at every submission; stateless kernels get that address directly [MEAS: SOFTPIN, GTT 2^48].
- **Coherence:** shared LLC (HAS_LLC = 1), so a CPU write is visible to the GPU without flushing CPU caches; on the GPU
  side, the L3 / data cache is flushed with PIPE_CONTROL before the CPU reads [SPEC: PIPE_CONTROL; details unverified].
- **MOCS** (cache policy in STATE_BASE_ADDRESS and surface state): Gen9 uses indices into a table set by i915
  [HYP: index 2 = WB LLC + L3 as in Mesa; not confirmed. Measured preference: 6f.5].
- **4 KB pages**, 4-level PPGTT [MEAS: debugfs, GTT 2^48].

## 6. Risks

- **GPU hang** (bad command buffer, bad command length, a loop in a kernel): i915 detects it (heartbeat) and resets the
  render engine. The reset also affects the desktop compositor for a fraction of a second up to a few seconds. A context
  that hangs the GPU several times is banned until the end of the process [EST: i915]. Start tests with minimal kernels
  and a time limit in `GEM_WAIT`.
- **No validation:** NEO checks arguments, `kcl` does not. A wrong address in a stateless kernel is a write to a random
  place in the process's PPGTT (context isolation, HAS_CONTEXT_ISOLATION, protects other processes).
- **Compatibility:** everything above applies to Gen9 and this i915 version. NEO legacy (24.35) is the last line for
  Gen9 anyway, which is why the NEO path remains only as a comparison build (`libgen9blas_neo.so`, `-DGEN9BLAS_NEO`).

## 6a. i915 time limits of the render engine [MEAS: `/sys/class/drm/card*/engine/rcs0`, 2026-11-11]

| Parameter | Value | Meaning |
|---|---|---|
| preempt_timeout_ms | 640 | when another context (e.g. the desktop) wants the GPU and ours cannot be preempted within 640 ms, the context is reset |
| heartbeat_interval_ms | 2500 | checks that the engine is alive |
| timeslice_duration_ms | 1 | switching between contexts of equal priority |
| stop_timeout_ms | 100 | |

**Event 2026-11-11:** ggml `test-backend-ops` (Vulkan, MUL_MAT n = 4096) -- "Resetting rcs0 for preemption time out",
"context reset due to GPU hang" (`journalctl -k`). This is why `results/compare-ggml-vulkan-2026-11-11.txt` has no n = 4096
result. Consequence: a single work group (the unit of GPGPU preemption) must last much less than 640 ms [HYP: on Gen9
i915 preempts GPGPU at group boundaries; not confirmed]. The gen9blas kernels stayed within it on 2026-11-11 (DGEMM 4096: 2.4 s in total),
but on 2026-11-12 DGEMM n = 3072 and 4096 reset the GPU with exactly this message (n = 2048, 0.3 s, works; not yet diagnosed,
see `results/numpy-preload-2026-11-12.txt`).

## 6b. Diagnostics

- The GPU state at the time of a hang is in `/sys/class/drm/card*/error` (root; reading clears it, so save the output to
  a file); decode it with `intel_error_decode` (intel-gpu-tools 1.28).
- Engine registers (L3CNTLREG, INSTDONE, HEAD/TAIL, ACTHD, BBADDR) read as 0 unless the GPU is awake (forcewake): the
  GPU sits in RC6 otherwise.
- `journalctl -k` (no root needed): i915 messages about hangs and resets.

## 6c. First experiment: it works [MEAS]

`experiments/no-neo/first.c` (`make -C experiments/no-neo && cd experiments/no-neo && ./first 16`): the kernel
`store` from `store.cl` (`p[i] = v + i`, SIMD16, stateless, compiled with `ocloc`, ISA cut out of the zebin by
`extract.py`) sent in our own command buffer (280 B) with the sequence from section 3. The commands are packed with the
functions of `lib/third_party/mesa/gen9_pack.h` (generated from Mesa genxml by `gen_pack_header.py`; regenerate
explicitly with `make gen9_pack`; the few Mesa helper functions it needs are in `lib/third_party/mesa/`).
**The result was correct the first time, with no GPU hang.**

Settings that worked: MEDIA_VFE_STATE as in NEO (max threads 167 = 24*7-1, the field is U16-1; 1 URB entry; entry size
0x782; CURBE 0); IDD: CrossThreadConstantDataReadLength = 2 (64 B), ConstantURBEntryReadLength = 3 (local IDs);
GPGPU_WALKER: IndirectDataLength = 192 (64 + 96), indirect data = shared, then u16 local IDs; all base addresses = the
heap (softpin 0x10000000), general = 0; MOCS 2<<1 (worked; whether it is WB is still [HYP]); PIPE_CONTROL (CS stall +
DC flush) before PIPELINE_SELECT and at the end, PIPE_CONTROL (invalidate state/instruction/constant cache) after
STATE_BASE_ADDRESS.

**Cost of one call** (same kernel, 2000 calls, median; `experiments/no-neo/neo_compare.c` for NEO):

| | time | CPU per call | core occupancy |
|---|---|---|---|
| NEO + clFinish (busy wait) | 34.4 us | 41-43 us | 112-118% |
| NEO + sleeping poll (like the old wait in gen9blas) | 90 us | 29 us | 29-30% |
| **without NEO: execbuffer2 + GEM_WAIT** | **36 us** | **7.1 us** | 19-20% |

Without NEO: latency as with busy waiting, with 4-6x less CPU (GEM_WAIT sleeps in the Linux kernel until the interrupt).
The buffer contains the whole state every time (PIPELINE_SELECT, STATE_BASE_ADDRESS, VFE); the GPU-side cost is included.

## 6d. Several dependent kernels in one buffer [MEAS]

`experiments/no-neo/chain.c` (+ `gen9i.h`: i915 and command-buffer handling; `kernels.py`: C header with kernel code and
offsets from `.ze_info`): a chain of K kernels (`store`, then K-1 x `copy` or `reverse`, `b[i] = a[N-1-i] + 1`) alternating
between two buffers. Env `CROSS=1` selects the `reverse` steps, `RACE=1` the race test; env `G9I_NO_MSF=1` (read by `gen9i.h`)
drops MEDIA_STATE_FLUSH.

**A barrier is necessary.** Race test (50 runs, buffers cleared before each, the cross dependency `reverse`, N = 2^20):
without a barrier 30/50 runs gave a wrong result, without a barrier and without MEDIA_STATE_FLUSH 38/50, **with a
PIPE_CONTROL barrier (CommandStreamerStall + DCFlush) 0/50**. Consecutive GPGPU_WALKERs overlap; MEDIA_STATE_FLUSH does
not order them. At N = 4096 and with an element-to-element dependency (`copy`) the race did not show up: the kernels are
too short to overlap. A race test must have a dependency between distant groups.

**Cost** (8 kernels, median of 200; NEO: `experiments/no-neo/neo_chain.c`, in-order queue, one wait at the end):

| N | without NEO: one buffer, barriers | NEO + clFinish | NEO + sleeping wait (like the old gen9blas wait) |
|---|---|---|---|
| 4096 | **72 us, CPU 10 us** | 96 us, CPU 108 us | 149 us, CPU 103 us |
| 65 536 | **399 us, CPU 18 us** | 391 us, CPU 409 us | 590 us, CPU 120 us |
| 1 048 576 | **4.52 ms, CPU 49 us** | 4.77 ms, CPU 4.76 ms | 4.83 ms, CPU 420 us |

Without NEO: time equal to NEO with busy waiting or shorter, CPU 8-25x lower than NEO with sleeping (about 12 us CPU per
`clEnqueueNDRangeKernel` in NEO versus about 10 us for a whole buffer). Submitting each kernel separately without NEO is
4.6x slower at small N (316 versus 72 us): put the kernels of one operation into one buffer.

## 6e. Binding table and full HGEMM without NEO [MEAS]

- `gen9i.h`: each kernel call has its own IDD (with its own binding table); the IDD array is loaded once per buffer
  (MEDIA_INTERFACE_DESCRIPTOR_LOAD completed at submit; up to 64 calls per buffer); 3D grid (group = 1 SIMD thread).
- A buffer through the binding table: RENDER_SURFACE_STATE as NEO `encodeBuffer` (SURFTYPE_BUFFER, RAW format 0x1ff,
  size-1 split into Width[6:0] / Height[20:7] / Depth[31:21], VALIGN_4, LINEAR, GPU coherent) + `buffer_address` in the
  shared data.
- **Pitfall:** the Binding Table Pointer in the IDD is bits 5..15, so at most 64 KB from the Surface State Base Address.
  A table further away gives a truncated pointer and the kernel writes to the wrong place (our first test: buffer
  untouched, no GPU hang). Fix: Surface State Base = start of a 64 KB area holding the tables and surface states.
- The `store.cl` kernels in the binding-table version (IGC "untyped surface read/write", target `chain_bti`): chain
  correct (race 0/50), time as stateless.
- **HGEMM** (`experiments/no-neo/gemm_h.c`): `pk_a2_q` (SIMD16) + `pk_bq_w2` (SIMD32), barrier, `hgemm_v` in one buffer.
  Correct the first time (mean error 1.4e-4, as the library FP16 path). Median (two series alternating with
  gen9blas/NEO, `bench/compare/l3 ... 3`):

| n | without NEO | gen9blas via NEO | CPU without NEO | CPU NEO |
|---|---|---|---|---|
| 512 | **0.68-0.72 ms** (373-393 GFLOPS) | 1.01-1.03 ms | 3.8-4.7% | 14-15% |
| 1024 | **3.76-4.16 ms** (517-572) | 4.09-4.76 ms | 1.4% | 9% |
| 2048 | **26.1 ms** (657) | 26.2-27.8 ms | 0.2% | 4% |
| 4096 | **195-205 ms** (671-703) | 198-216 ms | 0.05% | 3-4% |

  n = 1024: minimum 3.60 ms versus the oneDNN kernel alone 3.06-3.35 ms (no packing): the difference is packing.

## 6f. The whole library without NEO: the kcl layer [MEAS]

Goal: the whole library runs at the lowest possible hardware level. Implementation: `lib/kcl.c` + `lib/kcl.h` are exactly
the subset of OpenCL used by `lib/gen9blas.c` (about 25 functions), implemented on i915; `gen9blas.c` is almost unchanged
(program creation -> built-in kernels, the wait -> `GEM_WAIT`). Kernels are compiled at build time (`lib/gen_kernels.py`:
`ocloc` for `hgemm.cl`, `sgemm.cl`, `dgemm.cl`, `level12.cl` in three option sets (`-DHALF`, default, `-DDBL`), plus the
assembler binaries `gen_*.bin`) into `lib/kernels.c` (239 kernels, 1.28 MB of ISA at the time of the measurement;
argument descriptors from `.ze_info`, layout in `lib/kcl_desc.h`). The library exports no `cl*` symbols (the macros are
`kcl_*`) and does not link libOpenCL. `libgen9blas_neo.so` / `libgen9blas_blas_neo.so` (`-DGEN9BLAS_NEO`) are the old
path, kept for comparison only. Kernels with unsupported features (images, sampler, scratch) are flagged `unsupported`
by `gen_kernels.py` and `kcl` refuses to create them.

Model in kcl: a queue = a command buffer built on the fly (IDD + indirect data + binding table per call, PIPE_CONTROL
barrier between kernels), submitted on wait or when full (64 IDD); 8 rotating buffers (wait for a slot before reuse);
`cl_mem` = a GEM object (softpin, increasing addresses, never reused); "USM host" = a GEM object mapped WB;
multi-thread groups (local IDs as NEO: 32 B row / 64 B for SIMD32), SLM, barriers, private memory (private_size x SIMD x
168), program constants (`const_base`), PIPE_CONTROL timestamps (`GEN9BLAS_PROFILE`, `KCL_PROFILE_KERNELS`).

**Facts found while porting (all confirmed by measurement):**
1. **The buffer surface size must be rounded up to 4 B** (NEO: `alignUp(size, 4)`): the hardware checks RAW bounds per
   DWORD, so the last FP16 element of a buffer whose size is not divisible by 4 read as 0 (GEMM 1x1x1 = 0).
2. **SLM size encoding in the IDD on Gen9.5: log2(pow2(max(slm, 1K))) - 9 (1K = 1, 2K = 2, 4K = 3 ... 64K = 7)**
   (PRM KBL Vol. 2d p. 318, NEO `computeSlmValues`). The Gen8 encoding (Mesa genxml gen80.xml: 4K = 1, 8K = 2, 16K = 4 ...) does not apply; gen90.xml has the Gen9 one.
   With the Gen8 values, groups received 1 KB instead of 4 KB and overlapped in SLM (SYMM/TRMM correct for one group only).
3. **MI_LOAD_REGISTER_IMM to L3CNTLREG (0x7034) works from an unprivileged buffer** (a register read while the context
   was running gave 0x60000321, the SLM configuration). NEO on KBL/CFL always uses the SLM configuration; measured: without
   SLM slightly faster for GEMM without SLM (345 versus 334 GFLOPS, K = 4096). kcl uses SLM only in buffers that contain
   SLM kernels; `KCL_L3=slm` forces the NEO behaviour (diagnostics).
4. **Round-robin EU thread arbitration (MI_LOAD_REGISTER_IMM 0xE404 = 0x100, as NEO on Gen9) is crucial for GEMM with a
   long K:** SGEMM 4096 x 4096 x 8192: 211-220 GFLOPS (hardware default) -> 362 GFLOPS (round-robin, = NEO); at K = 4096
   no difference (358 / 358). Found by comparing the NEO preamble after rejecting other hypotheses (MOCS, L3, 2 MB address
   alignment, address colouring). `KCL_ARB=age` restores the hardware default.
5. MOCS (i915 SKL table indices): 0 = no cache (SSYR2K 4096: 1715 ms), 1 = 950, **2 = 877 (best, used)**, 3 = 890.
   `KCL_MOCS=<index>` overrides.
6. Aligning GPU addresses to 2 MB gives nothing: `transparent_hugepage/shmem_enabled = never`, so i915 uses 4 KB pages
   anyway. The option remains as `KCL_ADDR=1` (2 MB alignment for objects >= 2 MB; `KCL_ADDR=2` offsets addresses by
   0..15 x 4 KB; default 0 = 64 KB alignment).
7. CPU measurement pitfall: OpenBLAS (NumPy) threads spin in the background and inflate the process CPU time (49 ms CPU
   per 7 ms GEMM); `bench/level3_cpu.py` now sets `OPENBLAS_NUM_THREADS=1`.

**Result** (all test suites OK: `make -C lib check`, i.e. `test_basic`, `test_level3`, `test_level12` x3, `test_blas`,
`test_blas_d`, `test_blas_banded`, ...). Level 3, GFLOPS without NEO / NEO (CPU % without NEO / NEO),
`results/no-neo-level3-2026-11-11.txt`:

| | GEMM | SYMM | SYRK | SYR2K | TRMM | TRSM |
|---|---|---|---|---|---|---|
| FP32 n = 1024 | 287/292 (7/5) | **269/203** (5/10) | 189/187 (5/14) | 181/177 (3/8) | 175/156 (5/23) | 96/86 (3/28) |
| FP16 n = 1024 | 535/509 (3/12) | **473/408** (3/17) | 299/279 (6/23) | 287/274 (4/18) | 248/226 (13/34) | 124/107 (4/36) |
| FP32 n = 2048 | 341/339 (2/3) | **322/281** (0/4) | 246/239 (0/4) | 234/231 (0/5) | 233/223 (1/5) | 183/175 (1/10) |
| FP16 n = 2048 | 672/663 (0/3) | **601/429** (1/3) | 437/426 (1/5) | 413/404 (0/4) | 406/375 (2/5) | 277/256 (1/9) |
| FP32 n = 4096 | 349/349 (0/4) | 334/322 (0/4) | 266/258 (0/4) | 240/236 (0/4) | 250/248 (0/4) | 247/242 (0/5) |
| FP16 n = 4096 | 720/719 (0/3) | 588/569 (0/3) | 516/497 (0/3) | 486/458 (0/3) | 495/470 (0/4) | 438/417 (0/4) |

Small calls from Python (GEMM 64^3): 88-108 us versus 137-140 us, CPU 29 versus 75-79 us; 256^3: CPU 34-46 versus 82-86 us.

## 6g. Ordinary process memory without a copy: userptr [MEAS]

Goal: NumPy arrays (and any C program memory) go to the GPU without a CPU copy; before this only `gen9blas.empty` (USM)
did. Microbenchmark `experiments/no-neo/userptr.c` (`PHASES=1` splits the cost into phases), results
`results/userptr-2026-11-12.txt`, test `lib/test_userptr.py`, comparison `bench/compare/userptr.py`.

**Facts:**
1. `GEM_USERPTR` makes a GEM object from existing process memory (address and size 4 KB aligned: large NumPy arrays start
   16 B past a page start, so the object covers whole pages and the offset goes into the address). Flags: READ_ONLY works
   on this card, PROBE too; overlapping objects are allowed. LLC: no clflush needed (i915 `can_bypass_llc` = false), the GPU
   reads CPU writes and vice versa (checked, including data changed between calls).
2. **Pinning pages costs about 0.45 us CPU per 4 KB page** (at the first execbuf; 2 x 64 MB: 14-27 ms CPU), about the
   same as a copy (~10 GB/s). Source, i915 v7.0: `pin_user_pages_fast` + sg table + `i915_gem_gtt_prepare_pages`, i.e. DMA
   mapping of every page through the IOMMU (enabled for the GPU: group 0, `DMA-FQ`). A userptr created in every call
   therefore gains nothing; **an object kept between calls** does: the GPU reads it as fast as its own buffer (64 MB:
   9.1 versus 8.8 ms), CPU 127 us instead of 12 ms.
3. Process memory that changes (munmap, new mapping) is watched by the Linux kernel (mmu notifier): the object's pages are
   invalidated and pinned again at the next execbuf, so a cached object stays correct.
4. **An object wider than the current array is dangerous:** pages that once held the array may later belong to an mmap of
   a GEM object (`gen9blas.empty`, scratch buffers); such pages cannot be pinned and execbuf returns EFAULT (measured: a
   64 MB union object of two former arrays). `msync` does not detect it (the pages are mapped).
5. **Two objects on the same pages in one kernel lose writes:** they are two GPU addresses of the same memory line; a
   swap/rot of two arrays, one ending and the other starting in the same 64-byte line (16 B malloc header), lost exactly
   one element (x[0]), depending on addresses, reproducibly.

**Solution in kcl (`kcl_userptr`, `kcl_new_operation`):** a cache of userptr objects (LRU, 1 GB limit of pinned memory,
`KCL_USERPTR_MB`). The regions of one BLAS operation are collected and, at the first `clSetKernelArg`, split into groups
of overlapping page ranges; each group gets **one** object (fact 5), taken from the cache only for **exactly the same**
range (fact 4: the range is memory the caller supplies now), otherwise a new one (READ_ONLY when nothing in the group is
written). Objects of the current operation are not evicted from the cache. A region that is too large (> limit) or an
error: the old CPU copy (`GEN9BLAS_NO_USERPTR=1` forces it for comparison).

**Result** (the same NumPy arrays in successive calls, medians of 3 alternating rounds): SGEMM n = 1024-4096 CPU
0.02-0.20x (4096: 0.47 ms instead of 28 ms), time 0.82-0.96x; axpy/dot n = 1e6-1e7: time 0.15-0.24x, CPU 0.01-0.10x;
gemv 4096: time 0.65x, CPU 0.03x. New arrays in every call: CPU 0.27-0.34x (pinning instead of two copies).

## 7. Open gaps

1. ~~Minimal kernel~~ -- works (section 6c).
2. ~~Call cost~~ -- measured (6c); ~~several dependent kernels in one buffer~~ -- section 6d.
3. ~~MOCS~~ -- index 2 best, 0 = no cache (6f.5); the meaning of the other indices still comes from Mesa [EST].
4. ~~L3CNTLREG from an unprivileged buffer~~ -- works (6f.3).
5. Scratch (R0.5, MEDIA_VFE_STATE) -- no library kernel needs it (`gen_kernels.py` rejects such kernels).
6. Whether CMD_PARSER (v10) applies to the render engine on Gen9.
