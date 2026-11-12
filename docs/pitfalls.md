# Pitfalls and lessons

Things that cost time while building this library on an Intel UHD Graphics 620 (Gen9.5, Whiskey Lake GT2,
Linux i915). Each entry gives the symptom, the cause, what to do, and where the evidence is. Evidence tags as in
`results/*.txt`: [SPEC] Intel documentation, [MEAS] our measurement of a hardware constant, [SW] software behaviour
(fixed for a given driver/compiler version), [EST] estimate from third-party code, [HYP] unconfirmed hypothesis.

Versions: NEO 24.35.30872.36 (the last "legacy" line that supports Gen9), IGC 1.0.17537.24, Mesa 25.2.8, IGT 2.6,
Linux 7.0 i915. Most findings about NEO apply only to the OpenCL path (`libgen9blas_neo.so`, the comparison
programs in `bench/`); the default library (`libgen9blas.so`) uses `kcl` and does not go through NEO
(see `docs/no-neo.md`). Hardware constants are in `docs/hardware.md`, the model in `docs/performance-model.md`,
the chronology in `docs/history.md`.

Contents: 1 Compiler and tool chain, 2 Hardware, 3 Driver (NEO), 4 kcl and i915, 5 Kernel-writing lessons,
6 Measurement conditions, 7 Library and test lessons.

---

## 1. Compiler and tool chain (IGC, ocloc)

**Machine code built with `-device cfl` is rejected.**
Symptom: `clCreateProgramWithBinary` fails with `CL_INVALID_BINARY` (-42). Cause: the product family in
`.note.intelgt.compat` does not match the device. Do: build and assemble with `-device whl` (or `-device 0x3ea0`).
`-device cfl` is fine for disassembly only (`experiments/isa/dump.sh`). [SW] Source: `asm/disassemble.sh`,
`asm/assemble.sh`.

**An edited binary that still contains `.spv` is silently ignored.**
Symptom: editing the machine code does not change the run time. Cause: on a mismatch the driver recompiles the
kernel from the attached SPIR-V. Do: delete the `.spv` section (and its line in `sections.txt`) after
disassembling; `asm/disassemble.sh` does this. [SW]

**`ocloc asm` option quirks.**
`ocloc asm` (legacy 24.35) takes `-file <result>`, not `-out` as the help text says, and writes the result into
the `-dump` directory (`asm/assemble.sh` moves it). It uses `libiga64.so.1` 1.0.17537.24, matching IGC. [SW]
A standalone `iga64` from `libigc-tools` 2.42 also handles Gen9 (`-p=9`) and is handy for checking single
encodings.

**Patching a loop: replace everything between the label and the jump.**
Symptom: every micro-benchmark showed a false ceiling of about 360 GFLOPS. Cause: the compiler had placed 31 of
the 256 `mad` after the loop-counter instruction, before `cmp`/`jmpi`; a generator that cut only up to the
counter left 31 extra instructions per iteration. Do: replace all instructions between the loop label and the
back-jump. Source: `results/micro-2026-11-11.txt`, `asm/micro.py`.

**The compiler does not avoid register-bank conflicts.**
In OpenCL C GEMM loops 128 of 256 (SGEMM) or 160 of 256 (HGEMM 16x16) `mad` instructions had a src1/src2 bank
conflict (a 33% slowdown each); the best compiler HGEMM reached 654 GFLOPS vs 777 for the hand loop. Do: lay out
registers by hand (see "Register banks" below). Source: `results/hgemm-compare-2026-11-11.txt`. [SW]

**FP16 `mad()` is not fused without fast-math.**
Symptom: 200 GFLOPS instead of 800. Cause: IGC emits `mul` + `add` for half `mad()`. Do: use `fma()` (803 GFLOPS) or
`-cl-fast-relaxed-math` (791). [SW/MEAS] Source: `results/arith-throughput-latency-2026-11-11.txt`.

**FP16 two-operand ops with a broadcast scalar lose the double rate.**
`mul`/`add :hf` with a `<0;1,0>` scalar operand run at 205 Gop/s instead of 400 with a vector operand. [MEAS]

**Vector types inside one work-item defeat SIMD.**
A `half16 acc` plus `sub_group_broadcast` became 256 scalar `mad (1|M0)`. One value per SIMD lane, not vectors
per work-item. [SW] (`docs/history.md`, instruction-map section.)

**Full-precision transcendental functions are libraries.**
`sin()` FP32 = 568 instructions, FP64 `sqrt` = 145; INT64 multiply = 73. Use `native_*` or fast-math where accuracy
allows; avoid INT64 and `mul_hi` in hot loops. [SW] Source: `experiments/isa/map.cl`.

**Private arrays above about 128 B per lane go to scratch memory.**
Symptom: silently wrong results through `kcl` (correct through NEO). Cause: IGC puts e.g. `float s[8][8]` in
SIMD8 (256 B per lane) into private memory (scratch) even when fully unrolled; `kcl` does not configure scratch
(`MEDIA_VFE_STATE`). The same happens on register spill. Do: use separate vectors (`float8 s0..s7`) instead of
arrays; after compiling check `private_size`, `spill_size` and `per_thread_memory_buffers` in `.ze_info`.
`lib/gen_kernels.py` rejects kernels with `spill_size` or `per_thread_scratch_size`, but not `private_size`
alone (`gemv_n` declares 256 B without using it), so a scratch access is not detected automatically.
A fallback path in the same kernel (for example an unaligned 2-byte read path) also costs registers and can cause
the spill; check alignment on the host instead. [SW] Source: `results/skinny-gemm-2026-11-12.txt`,
`results/skinny-gemm-rework-2026-11-12.txt`.

**64-bit multiplication is emulated.**
Row-pointer arithmetic inside a loop cost 1.2-1.9x; compute pointers once before the loop. [SW] Source:
`results/skinny-gemm-2026-11-12.txt`.

**FP16 vector loads are split into byte reads.**
`vload8` of `half` becomes eight 4-byte "byte gathering" reads (as in `bench/packing.cl`), and copy or gemv with
8 FP16 per lane was *slower* (11-16 GB/s). Do: use `intel_sub_group_block_read_us*` (one OWord block message,
FP16 gemv 15-16.5 -> 20-25 GB/s) or process FP16 copies as 32-bit words. Source:
`results/message-16B-2026-11-12.txt`. [SW]

**Per-lane access width: 16 B per SIMD16 lane matches a 64 B line.**
With 32 B per lane (FP64) two messages each cover half of every line; a write without a prior read refetches the
line (COPY FP64 17.4 vs 23.6 GB/s once changed to 16 B per lane). [SW, explanation HYP]

**Sub-group-wide wait loops hang the GPU.**
A flag wait written as `while (atomic_load(...) == 0)` executed by all lanes of the sub-group hung the GPU
(compiler/divergence issue). Do: let one lane spin and follow with `sub_group_barrier`. A "sync-free" single-launch
TRSV built on this was 3-8x slower than the blocked version anyway (28/18 ms vs 54-181/70-163 ms, n = 8192 FP32,
N/T); the cost is in the dependency chain, not in barriers or atomics. Do not retry without a new idea. [SW]
Source: `docs/history.md` (levels 1-2 section).

---

## 2. Hardware

**Register banks: two, by register parity (not four as in Mesa).**
Mesa's model (`brw_opt_bank_conflicts.cpp`, [EST]) has 4 banks. On the UHD 620 only the parity of the register
number matters, and only src1 and src2 of a three-source `mad` conflict (src0 = dst does not). A conflict costs +1
clock per SIMD8 half: two vectors in the same bank 6 instead of 4 clocks (x0.66 measured); a broadcast scalar
src2 in the same parity 5 clocks (x0.8). Rule: put B (src1) and A (src2) in registers of different parity.
Same register for both: no conflict. Instruction compaction and the destination register do not matter. [MEAS]
Source: `results/micro-2026-11-11.txt`.

**Instruction rates that are not visible in the machine code.**
INT32 multiply runs at half rate (100 Gop/s) although it is one `mul (16) :d`; INT16 only speeds up multiplies with
a vector operand (x1.75); INT8 has no dot-product instruction (4 `mul` + 3 `add`); INT64 is emulated (add 1/4,
multiply about 1/14); FP64 is SIMD8 per instruction at 1/4 the FP32 rate. Compute indices with shifts and adds.
[MEAS] Source: `results/arith-throughput-latency-2026-11-11.txt`.

**Mixed-mode FP16/FP32 `mad` has FP32 rate.**
[SPEC] PRM KBL Vol. 7 ("Register Region Restrictions", "Mixed Mode Float"): execution size is counted in float
elements, SIMD16 is not allowed when the result is packed f16. An accumulate-in-FP32 FP16 kernel therefore has
FP32 throughput by construction.

**Memory hierarchy numbers.**
L3 delivers about 155 GB/s but only with at least two block reads in flight per thread (one at a time: 117 GB/s);
RAM about 20 GB/s for the GPU and 12.8-18 GB/s for the CPU, so about 20 GB/s is the platform limit, not the GPU's;
latency about 147 clocks (L3) and 440 (RAM, 400 ns). A kernel must issue reads hundreds of clocks before use. The
memory is dual channel (2 x 8 GB, 2400 MT/s); the early "single channel" hypothesis was wrong. [MEAS] Source:
`results/memory-bandwidth-2026-11-11.txt`, `results/memory-latency-2026-11-11.txt`.

**Large-N collapse is a TLB problem (4 KB pages).**
Symptom: OpenCL C GEMM with 2D block reads is fast up to N about 1728, then drops about 15-20x (SGEMM CLBlast: 35-60
GFLOPS at N = 2000, 2048, 3000, 4000, 4096). Cause: with 4 KB pages (`has_64k_pages: no`) a tile that reads many
distant rows touches too many pages. Packing A and B into contiguous panels removed the collapse for our
kernels [MEAS]; for CLBlast the same cause is probable but unproven (row padding did not help, so it is not cache
associativity) [HYP]. The same effect: packing kernels in which neighbouring work-items write 16 different panels
(`pack_b16`: 17.7 ms vs 5.6 ms) or read 16 rows per k (`pack_b_k16`: 57 ms vs 4.5), and a row-wise GEMV that
touches about 700 pages at once on userptr memory. Source: `results/hgemm-compare-2026-11-11.txt`,
`results/gemm0-2026-11-11.txt`, `results/gemv-n-2026-11-12.txt`.

**`send` header registers can be modified right after the send.**
The hardware reads the header at issue, so one header per stream is enough (`add +128` right after the `send`);
results are correct. [MEAS] Used in `asm/sgemm_asm.py`.

**Register dependencies of `send` -> `mad` are tracked by hardware.**
The Gen9 scoreboard handles them; hand-written loops need no explicit barriers. [SPEC/MEAS]

**FP16 reduction by 4096 ones gives 2048.**
Not a bug: FP16 cannot represent 2049. Use FP32 accumulation (`GEN9BLAS_HS`) when the sum matters.

---

## 3. Driver (NEO, the OpenCL path)

**NEO aborts when `CL_MEM_USE_HOST_PTR` buffers overlap within one call.**
Symptom: abort at `host_ptr_manager.cpp:252`. Cause: a zero-copy buffer overlapping another area of the same
memory with a different size (also one copied by `clEnqueueWriteBuffer`), for example `gemm(A[:512], A)` with an
aligned `A`, or SYRK row bands. Between calls it is safe (the registration disappears when the buffer is
released). Our tests missed it because they used unaligned arrays. Do: if operands of one call overlap, none goes
zero-copy; include aligned and overlapping cases in tests. [SW]

**Host memory for the GPU: USM host only.**
Symptom: random NEO failures when `USE_HOST_PTR` buffers over ordinary memory are created and released often:
`EXECBUFFER2` -> `ENOSPC`, `host_ptr_manager.cpp:252`, `reference_tracked_object.h:134`, segfaults. Cause: NEO closes
GPU objects asynchronously, a new buffer on the same pages gets an overlapping GPU address (softpin at the CPU
address). With a registration cache or without one, with a sleeping wait 2 of 4 runs crashed (with `clFinish` 6 of 6
passed); the NEO debug keys `EnableGemCloseWorker=0`, `EnableAsyncDestroyAllocations=0`, `EnableDeferredDeleter=0`
(`NEOReadDebugKeys=1`) still left 1 of 8 runs failing; map/unmap before release also aborted. Do: use USM host
(`clHostMemAllocINTEL` + `clSetKernelArgMemPointerINTEL`), never `USE_HOST_PTR` on ordinary memory and never
`clEnqueueRead/WriteBuffer` from host arrays (reading into a stack array also registers stack pages). [SW]

**`USE_HOST_PTR` on the inside of a USM allocation ignores the offset.**
The start of the allocation is used: a silently wrong result. Do not create `cl_mem` on USM; pass USM pointers as
kernel arguments (an interior pointer is fine). `clGetMemAllocInfoINTEL` distinguishes USM (type HOST) from
ordinary memory (UNKNOWN). [SW]

**USM allocation costs about 0.5 ms of CPU per MB.**
64 MB: 33 ms (free: 3 ms). Keep a pool (`gen9blas.empty`, `free_pool()`). Registering plain memory
(`USE_HOST_PTR`) costs about 10 ms per 64 MB, as much as a copy. [MEAS]

**`clWaitForEvents` / `clFinish` spin on the CPU.**
About 6 W extra and 100% of a core for short calls (it also explains the "low CPU use" advantage of any library
that waits by sleeping). Do: marker + `clFlush` + polling with `nanosleep` (20 us -> 0.5 ms), as `wait_gpu()`
in `lib/gen9blas.c`; `GEN9BLAS_WAIT=busy` restores `clFinish`. Whole package 25.1 -> 18.9 W at equal speed.
`clFinish` does not spin all the time (level 3, n = 4096: 12-26% of a core vs 6-12%). Waiting by sleeping is not
slower in medians over several seconds; the single "best of 5" figure had suggested 0.8 ms loss (it was GPU
warm-up). [MEAS] Source: `results/compare-l3-2026-11-11.txt`, `results/long-run-2026-11-11.txt`.

**A profiling-enabled queue costs CPU.**
The completion marker on a profiling queue: 21 us CPU per call vs 2.3 us without. Profile only with
`GEN9BLAS_PROFILE=1`. [MEAS]

**NEO has no `cl_khr_command_buffer`.**
Recording a launch sequence is impossible; per-launch CPU cost (about 12 us per `clEnqueueNDRangeKernel`) stays.
This is one reason for `kcl` (a whole operation in one command buffer). [SW]

**Third-party libraries on this stack.**
CLBlast 1.6.2: SGEMM collapses (see TLB above), and row-major SSYRK/SSYR2K for n > 128 returns wrong results in
about 82% of the triangle (column-major and FP16 are correct). OpenVINO: without an explicit
`INFERENCE_PRECISION_HINT` an FP32 input is silently computed in FP16. oneDNN: only v3.6 still has a Gen9 GEMM
catalogue; `benchdnn` reports kernel time, not call time. Compare like with like. Source:
`results/compare-l3-2026-11-11.txt`, `results/compare-openvino-2026-11-11.txt`,
`results/compare-onednn-2026-11-11.txt`.

---

## 4. kcl and i915 (no NEO)

Details of the method are in `docs/no-neo.md`. `kcl` submits command buffers directly to i915 and does no
validation: a wrong buffer can hang the GPU. Start new work with minimal kernels and a timeout in the wait.

**GPU hang and reset affect the desktop.**
i915 detects a hang (heartbeat 2500 ms) and resets the render engine; the desktop freezes for a fraction of a
second to seconds, and a context that hangs repeatedly is banned for the rest of the process [EST: i915]. A
preemption timeout of 640 ms (`/sys/class/drm/card*/engine/rcs0/preempt_timeout_ms`) applies when another context
wants the GPU: a single work-group must be far shorter than that [HYP: Gen9 preempts GPGPU at work-group
boundaries]. Observed: a Vulkan MUL_MAT n = 4096 test of another library triggered "Resetting rcs0 for preemption
time out". Diagnostics: `journalctl -k`, `/sys/class/drm/card*/error` (decode with `intel_error_decode`; reading
the file clears it, so save the output), engine registers read as 0 unless the GPU is awake (RC6).

**SLM size encoding in `INTERFACE_DESCRIPTOR_DATA`.**
On Gen9.5 it is log2(pow2(max(slm, 1 KB))) - 9 (1 KB = 1 ... 64 KB = 7). This differs from Gen8 (Mesa's `gen80.xml`
has 4 KB = 1, 8 KB = 2, 16 KB = 4 ...): using the Gen8 encoding or the gen80 definitions on Gen9.5 gives the wrong SLM
size. Source: PRM KBL, Mesa 25.2.8 `gen90.xml` (correct Gen9 encoding). [SPEC/MEAS]

**RAW surface size must be a multiple of 4 B.**
Otherwise the last FP16 element reads as 0. Round the size up. [MEAS]

**Thread arbitration policy: use round-robin.**
Without round-robin arbitration (a `MI_LOAD_REGISTER_IMM` of `0x100` into `DebugControlReg2`, register `0xE404`,
as NEO does on Gen9) a GEMM with K = 8192 lost 1.7x. `kcl` emits it by default; `KCL_ARB=age` restores the
hardware default for diagnostics. [MEAS]

**Binding Table Pointer must be within 64 KB of Surface State Base.** [SPEC/MEAS]

**Consecutive `GPGPU_WALKER`s overlap without a barrier.**
Add a `PIPE_CONTROL` (CS stall + DC flush) between dependent kernels in one command buffer. Race test (50 runs,
cross-dependency between distant groups, n = 2^20): 30/50 wrong without a barrier, 38/50 without barrier and
`MEDIA_STATE_FLUSH`, 0/50 with `PIPE_CONTROL`. `MEDIA_STATE_FLUSH` does not order them. A race test must have a
dependency between distant groups: with small n or element-to-element dependencies the kernels are too short to
overlap and the bug stays hidden. Source: `docs/no-neo.md`, `experiments/no-neo/chain.c`.

**Scratch is not configured.** See the private-array entry in section 1.

**userptr: pinning cost, stale ranges, shared lines.**
(a) Pinning costs about 0.45 us CPU per 4 KB page (IOMMU DMA mapping), about as much as a copy, so userptr objects
must be kept in a cache between calls (then 64 MB reads as fast as an own buffer and a call costs 127 us CPU
instead of 12 ms). (b) An object wider than the current array is dangerous: pages that were once the array may now
belong to a GEM object's mmap and cannot be pinned (`execbuf` returns `EFAULT`; `msync` does not detect it).
(c) Two objects over the same pages in one kernel lose writes to the shared 64 B line (swap/rot of adjacent arrays
lost exactly one element, e.g. `x[0]`, reproducibly): use one object per group of overlapping page ranges. kcl
does this in `kcl_userptr` / `kcl_new_operation`; `KCL_USERPTR_MB` limits pinned memory (1 GB). [MEAS/SW]
Source: `results/userptr-2026-11-12.txt`.

**userptr memory is slower for kernels that touch many pages at once.**
Row-wise GEMV: 10-11 GB/s on plain memory vs 22-26 on USM; the grouped variant (`GEN9BLAS_GEMV_N=grouped`)
gives 18.9-21 on plain memory. Source: `results/gemv-n-2026-11-12.txt`.

**Cost of a call without NEO depends on the power state.**
See section 6. Fixed cost, mains power: 48 us wall / 14-17 us CPU for a small axpy, of which our part is 3-4 us;
the rest is i915 (execbuf 8 us, wait). The hypothesis "too many objects in `execbuf`" was rejected (15 -> 7
objects, no change). Source: `results/overhead-2026-11-12.txt`.

---

## 5. Kernel-writing lessons

- **Packing mattered more than assembly.** Contiguous panels removed the TLB collapse (20x at large N); the hand
  loop then added 19%. Pack element-per-work-item (or 2-4 elements for FP16 with wider stores), never 16 panels
  per neighbour set. FP32 gains nothing from wider packing.
- **Kernels that read a mirrored or transposed matrix element-by-element with a stride are very slow in FP16**
  (SYMM FP16 56 -> 113 ms); use 16x16 tiles in local memory (`gen9blas_expand_*`).
- **Reductions limited by read instructions, not bandwidth.** 4 elements per work-item with `vload4`: FP32 13-17
  -> 19-23 GB/s, FP16 7-8 -> 13-16.
- **Read loops that wait on memory every iteration** cost about 260 us per TRSV block; unroll the reads.
- **Check results against a reference including aligned and overlapping arrays, odd sizes, both memory kinds.**
  Bugs found this way: FP16 copy with odd n copied half (work-item count computed with the wrong elements per
  work-item); per-type element size computed locally as a ternary (FP64 band routines copied half the matrix;
  now `elem_size()`); `gen9blas_init()` called twice in one process (BLAS layer, then the Python module) created
  a second OpenCL context whose buffers failed with error -50.

---

## 6. Measurement conditions

**Run-to-run noise is about 10%.** The desktop compositor, the screen locker and screen casting also use the GPU
(a locked screen took 47% of the render engine during a long run). A single measurement once suggested the
opposite of the truth (zero-copy slower; alternating series: 2-15% faster). Compare variants alternating, several
series, medians. Compare absolute numbers only within one series.

**Best-of-N hides warm-up and power limits.** Short runs include GPU warm-up. The CPU runs at turbo (PL2, 25-42 W)
for about 30-45 s and then drops to the long-term limit (PL1, about 17 W): OpenBLAS SGEMM 4096 297 GFLOPS at 41.5 W,
then 189 GFLOPS at 16.7 W; OpenBLAS FP32 ~300 GFLOPS short, ~130 for 90 s. The GPU stays at 1100 MHz and 745-761
GFLOPS for four minutes. Report sustained numbers for anything compared with the CPU. Source:
`results/long-run-2026-11-11.txt`, `results/compare-l3-power-2026-11-11.txt`.

**Mains vs battery.** On battery (AC online = 0, EPP `balance_power`) the CPU sits at about 900 MHz while the GPU
works, and every CPU cost (execbuf, command building, OpenBLAS) is 2-3x higher (bare execbuf + GEM_WAIT: 22-28 us
CPU instead of 7). Before measuring check `/sys/class/power_supply/*/online` and the frequency in `/proc/cpuinfo`
and compare only within the same conditions. Source: `results/overhead-2026-11-12.txt`.

**Threads of OpenBLAS inflate CPU measurements.** After loading, OpenBLAS threads spin (about 550% CPU in a process
that never calls it; 700% after calls). Set `OPENBLAS_NUM_THREADS=1` in the environment before importing NumPy;
measure GPU and CPU paths in separate processes. `bench/compare/l3.c` re-executes itself with
`OPENBLAS_NUM_THREADS=1` for the GPU libraries.

**Python in the loop adds about 30 us per call** (ctypes, `.ctypes.data`); measure fixed costs from C
(`bench/overhead.c`). CPU time per call from `CLOCK_THREAD_CPUTIME_ID`/process time (`perf` is not available with
`perf_event_paranoid = 4`).

**CPU % is a reporting artefact of the wait mode.** A library that waits by sleeping shows 4-13% CPU, one that
spins 100%; part of what looks like an advantage over CLBlast and OpenVINO on level 3 is only the wait method
(`WAIT=sleep` in the comparison programs removes it). Speed advantages are separate and genuine.

**Energy.** `intel_gpu_top` package power excludes RAM and the rest of the board; energy per call = package power x
call time. Lower GPU clocks save energy on memory-bound work (500 MHz: -40% energy for 2.2x time on GEMM).
Frequency is changed through `gt_{min,max,boost}_freq_mhz` (lower the boost first when decreasing, the other way
round when raising; restore 1100 afterwards; needs write permission, for example via a udev rule).

**Models tend to underestimate fixed costs.** The model-predicted auto-mode thresholds were too low (it
under-counted the GPU fixed cost and ignored the spinning OpenBLAS threads); the skinny GEMM compute ceiling
(200-290 GFLOPS predicted) was 70-180 measured. Treat predictions as [HYP] until measured. Source:
`results/auto-thresholds-2026-11-12.txt`, `results/skinny-gemm-2026-11-12.txt`.

**Test-harness traps.** `v.copy()` of a Fortran-ordered NumPy array gives C order (use `copy(order="K")`; NaN
patterns then matched in both libraries but data was permuted). Timing-dependent driver bugs disappear under `gdb`:
use `coredumpctl` and `strace -Z`, and repeat each test several times. `make` and `diff` output may be filtered or
shortened by shell wrappers; run tests and comparisons with the real binaries when in doubt.

---

## 7. Library and BLAS-layer lessons

- **Zero-copy buffers are not kept across calls on the NEO path**: after Python frees an array (munmap) the GPU
  might read stale pages. On the `kcl` path the userptr cache handles this (mmu notifier, exact-range reuse).
- **BLAS/CBLAS with negative increments** pass a pointer to the lowest address (as `gen9blas` does): do not
  offset it again.
- **`LD_PRELOAD` interposition covers only exported calls.** NumPy `sgemm`, `sgemv`, `ssyrk` are intercepted;
  LAPACK inside OpenBLAS (`solve_triangular`, `np.linalg.solve`) calls BLAS internally and bypasses the exports.
- **Macros with function-pointer types as arguments** break on commas inside the type (pass name first, type as
  `...`, use `__typeof__`). The BLAS layer (`lib/gen9blas_blas.c`) forwards 356 of 393 symbols to
  `libopenblas.so.0` with `jmp` and refuses a library that resolves to our own `cblas_sgemm` (loop protection).
- **Banded and packed solves on the GPU** (`tbsv`, `tpsv`: one group of 256, n barriers) cost more CPU than
  OpenBLAS; auto mode sends them to the CPU.
- **Level-3 triangular work:** SYRK/TRMM computing the full product do 2x the work; band decomposition into 4
  bands reaches about 62% of it (2 bands: too much work, 6-8: call overhead wins).
