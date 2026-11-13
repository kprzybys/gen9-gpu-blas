# Writing fast Gen9 kernels: the whole path

How to get a kernel for the UHD 620 / HD 530 (Gen9 / Gen9.5) as close to its ceiling as possible, from the model to
the instruction level. This file is the map; the details live in:

- [performance-model.md](performance-model.md): hardware constants, software behaviour, formulas (the ceiling);
- [isa-notes.md](isa-notes.md): the ISA level (workflow, 13 rules, verification checklist, OA counters);
- [pitfalls.md](pitfalls.md): symptoms and fixes; [benchmarks.md](benchmarks.md): measurement rules;
- [solvers.md](solvers.md): exact questions (register layout, feasibility proofs) with constraint solvers.

Evidence tags as in performance-model.md: **[MEAS]** results file, **[SW]** behaviour of NEO 24.35 / IGC,
**[SPEC]** PRM, **[EST]** Mesa heuristics, **[HYP]** unconfirmed. **[EXT]** marks a published external result that
this project has not reproduced (different driver, other hardware or other conditions); treat it as a hypothesis
until measured here. External sources are collected in
[reports/Generatory kerneli dla Gen9.md](../reports/Generatory%20kerneli%20dla%20Gen9.md) (2026-11-13, in Polish).

## 1. Order of work

Work top-down and stop at the first level that reaches the ceiling. Each level has its own levers; a lower level
cannot repair a mistake made above it.

| Step | Level | Typical levers | Typical gain (evidence) |
|---|---|---|---|
| 1 | Model and ceiling | roofline from [performance-model.md](performance-model.md) §6; a ceiling diagnostic of the real loop (e.g. attention `ADV=0`: pointers frozen, every read hits) | decides whether the kernel is memory- or compute-bound before any tuning |
| 2 | Data layout and memory access | packing into contiguous panels, page locality, message type, bytes per lane, reads in flight | GEMM large-N collapse removed: 32 -> 654 GFLOPS (TLB, 4 KB pages) [MEAS]; FP16 GEMV 15-16.5 -> 20-25 GB/s with block reads [MEAS] |
| 3 | Compute structure in OpenCL C | register tiling, accumulators as vectors, sub-group size, compile flags, work-group size | FP16 `mad` 200 -> 791 GFLOPS with `-cl-fast-relaxed-math` [MEAS]; CLBlast on Skylake GT2: 2D register tiling 125 -> 180, sub-group shuffle -> 230 GFLOPS [EXT, Beignet driver] |
| 4 | Small parameter sweep | tile sizes, rows per thread, prefetch distance, work-group size (§4) | single per cent (attention NS 2 -> 4: +3 % [MEAS]) |
| 5 | ISA (hand loop) | GRF bank parity, load scheduling, register allocation without spill | +5 % to +53 % over the best OpenCL C [MEAS], see [isa-notes.md](isa-notes.md) §1 |

What the data say about the split: most of the distance to the ceiling is closed at levels 2-3 (layout and source
structure); the ISA level adds the last few to few tens of per cent, and only for compute-bound loops. The same
split appears in external work (CLBlast 125 -> 230 GFLOPS by source changes; C-for-Metal with explicit registers
about +10 % over expert OpenCL on Gen11 [EXT]). Drop to ISA only under the conditions in
[isa-notes.md](isa-notes.md) §1; for memory-bound work it does not pay (packing 20x, assembly +19 %, [pitfalls.md](pitfalls.md) §5).

Also check the platform settings that act like a kernel change: EU thread arbitration round-robin (SGEMM K = 8192:
211-220 -> 362 GFLOPS) and MOCS index 2 (index 0 is uncached, 2x slower), both set by `lib/kcl.c` [MEAS]
([performance-model.md](performance-model.md) §3).

### Four checks before choosing or tuning a kernel

1. **Profile the whole pass first, kernels second.** The share a kernel can win is bounded by its share of the pass.
   In a pp512 pass (Qwen 1.5B Q4_K_M, ~220 t/s) the GEMM is 80 %, packing + epilogue 7 %, attention 3 %, and ~9 %
   is GPU idle: ~2 % wake-up after each wait, ~7 % CPU ops (swiglu, rms_norm, rope, add) [MEAS, `results/ggml-backend-pass-profile-attn-gpu-2026-11-13.txt`].
   Use `KCL_PROFILE_KERNELS=1` (kernel sums), `KCL_PHASES=1` (submission and wait) and `GEN9_INFO=1` (backend
   decisions) together, and separate one-time from per-pass cost by varying the pass count (`-r 1` vs `-r 6`): a
   per-call average of execbuf (0.75 ms) first read as 7-8 % of the pass was in fact first-use page zeroing; per
   pass it is ~0.035 ms per submission, 0.3 %. And before comparing a variant, check the counter that shows it ran
   (`GEN9_INFO`): a morning "fused FFN costs +5.6 % energy on Q4_K_M" compared two identical configurations, because
   the fusion was F16-only (c1b8586); once it ran it gave +3-8 % pp512.
2. **Decide port-bound vs issue-bound by measurement, with a microbenchmark in the same bytes-per-instruction ratio.**
   The OWord block-read port gives 7.1-7.3 B/clock/EU on HD 530 and 7.6 on UHD 620 with >= 2 reads in flight; up to
   ~12 SIMD8 FP32 mads per 128 B block hide under the reads. At 16 mads per block (Qwen attention) the loop is
   issue-bound at ~87 % of the port (OA: both FPU pipes 95 % of active time, stall 5 %) [MEAS,
   `results/l3port-2026-11-13.txt`]. Make sure the microbenchmark measures the hardware, not the compiler: two
   versions were discarded because IGC added ~25 register copies or an integer multiply per block (read the ISA).
   The OA counter `L3ShaderThroughput` matches true bytes once scaled by the OA clock.
3. **A fitted model parameter is not a hardware constant.** The multithreaded model fits the attention points with a
   "6 B/clock/EU port", but the measured port is 7.1+; the fitted value lumps issue overhead and latency together
   ([solvers.md](solvers.md) §5). Test a model's explanation separately from its fit before acting on it.
4. **Extra work per element must fit the slack of the loop it joins.** Decoding Q4_K weights inside the FP16 GEMM
   costs ~2 instructions per 16 mads (+12.5 %) on a loop that is already issue-bound with 0-1 free registers, so it
   wins at most +5-14 % and only where weights are dequantized per use (7B); the scales fold into the 32-k carry for
   ~3 % [EST, `results/qgemm-model-2026-11-13.txt`]. Count the added instructions against the loop's non-mad budget
   and the free registers (solver budget) before writing a fused kernel.

## 2. Rules at the OpenCL C level

Rules confirmed in this project (NEO 24.35, IGC from `ocloc` 24.35). Each links to its evidence.

**Memory access**

- Use sub-group block reads/writes (`intel_sub_group_block_read*`, `_us` for FP16) for contiguous data. `vload8` of
  `half` becomes byte-gathering reads and is slower (11-16 GB/s) [SW] ([pitfalls.md](pitfalls.md) §1).
- 16 B per lane in SIMD16 (one 64 B line per message); 32 B per lane splits lines and a write without a prior read
  loses about 1.4x [SW].
- Keep at least two block reads in flight per thread: L3 155 GB/s vs 117 with one at a time [MEAS]; the ceiling
  with >= 2 in flight is 7.1-7.6 B/clock/EU (HD 530 170-176 GB/s, UHD 620 200 GB/s; up to ~12 SIMD8 FP32 mads per
  128 B block hide under the reads, `results/l3port-2026-11-13.txt`) [MEAS]; issue reads
  hundreds of clocks before use (L3 about 147 clocks, RAM about 440) [MEAS] ([pitfalls.md](pitfalls.md) §2).
- Page locality first: with 4 KB pages a tile reading many distant rows collapses 15-20x at large N; pack into
  contiguous panels; never let neighbouring work-items write 16 panels or read 16 rows per k [MEAS]
  ([pitfalls.md](pitfalls.md) §2).
- Read loops limited by instruction count, not bandwidth: several elements per work-item (`vload4`: FP32 13-17 ->
  19-23 GB/s) [MEAS] ([pitfalls.md](pitfalls.md) §5).
- Strided element-wise reads of a transposed or mirrored matrix in FP16 are very slow; use 16x16 tiles in local
  memory [MEAS] ([pitfalls.md](pitfalls.md) §5).

**Registers**

- Accumulators as separate vector variables, not private arrays: a private array above about 128 B per lane goes to
  scratch even when fully unrolled (`float s[8][8]` in SIMD8) [SW]. Under kcl scratch is not configured -> silently
  wrong results, so `spill_size` and `private_size` must be 0 ([performance-model.md](performance-model.md) §3,
  [isa-notes.md](isa-notes.md) §4.5).
- Know the register cost of types: raw `ushort4` per lane takes 4 GRF in SIMD8, `uint2` 2 GRF [SW].
- Choose SIMD8 vs SIMD16 by the register budget ([isa-notes.md](isa-notes.md) §4.4).

**Arithmetic and addresses**

- FP16 `mad()` needs `-cl-fast-relaxed-math`, otherwise it splits into `mul` + `add` (200 instead of 791 GFLOPS)
  [MEAS]; the half -> float conversion folds into `mad` in SIMD8 [SW].
- 64-bit multiply is emulated: compute pointers before the loop (`(long)i * stride` inside cost 1.2-1.9x) [MEAS].

**Launch**

- Keep one launch under the preemption timeout (640 ms) or results under kcl can be silently wrong; split long
  launches ([pitfalls.md](pitfalls.md) §4, [isa-notes.md](isa-notes.md) §4.13).
- Group sub-groups that share data into one work-group (attention: work-group 128 vs smaller, +10 % for Qwen) [MEAS]
  (`results/attn-kernel-hd530-2026-11-13.txt`).
- Grouped (Triton-style) tile order did not help after packing (31-550 GFLOPS, only one size) [MEAS]
  ([history.md](history.md) §8).

**Candidates not yet verified here** (from external tuning data; each needs a measurement on NEO 24.35, on both
devices, before it becomes a rule):

- *Local memory or not.* CLBlast's tuned GEMM entries choose no local memory for A and B on the UHD 620 (SA = SB = 0,
  GEMMK = 1, KREG = 4, MWG = NWG = 64) and a different design on the HD 530 (GEMMK = 0, MWG = 128) [EXT]. The project's
  GEMM kernels also read packed panels directly; a measured SLM vs no-SLM comparison is missing.
- *Sub-group shuffle for broadcast* inside a register tile (CLBlast +28 % on Skylake GT2 under Beignet) [EXT].
- *Work-group size limits with local memory:* Halide issue #4410 reports `CL_INVALID_WORK_GROUP_SIZE` on Gen9 for a
  32x8 tile with shared memory [EXT].
- *Device-specific optimum at the source level.* The `skl` and `whl` binaries of one template are identical
  (`results/attn-kcl-hd530-2026-11-13.txt`), yet CLBlast tuned different parameters for the two parts [EXT]: the
  optimum depends on the part (EU count, memory), not on the ISA, so source parameters should be measured on both
  machines.

## 3. Rules for generators

Generators that write kernels (`asm/gemm_gen.py`, `asm/attn_gen.py`, `lib/gen_kernels.py`) follow the same pattern
as the best vendor generators (MaxAs, AMD Tensile, oneDNN's nGEN): parameters plus a human-designed loop [EXT]. Rules:

1. **Enforce rules in code, do not only print them.** A generated binary must fail the build if it has a bank
   conflict in a hot `mad`, `spill_size`, `private_size` or a scratch buffer. oneDNN's generator does bank-aware
   allocation itself (`bank_conflict_allocation`) [EXT]. Status (2026-11-13): `attn_gen.py` prints the conflict
   count, `gemm_gen.py` does not count it, `lib/gen_kernels.py` rejects spill/scratch for the kcl table; a variant
   with 8 conflicts built without an error. **Open:** a shared check in `asm/isa.py` used by all generators.
2. **Check equivalence symbolically before assembling** (`asm/loopcheck.py`, run by `gemm_gen.py`): the generated
   loop and IGC's loop are executed in a symbolic interpreter (memory reads as symbols, counter concrete) and the
   accumulators must hold identical expressions (same products, same order = bit-identical), or, when the template
   assumes another packed layout (h, s), identical expressions up to a consistent bijection of memory elements.
   Self-tests: a swapped register, a wrong load offset and a swapped accumulation order are all rejected.
   Attention (`attn_gen.py` sched, order differs from IGC by design): for every O element the multiset of
   (V element, identity of p) must equal IGC's, where p's identity is traced through the softmax to the multiset of
   (K element, Q element) of its score; checks both mad phases on all six shapes (incl. QTOK and CAUSAL), not the
   softmax arithmetic. Mutations (Q element, V register, K offset) are rejected. `ATTN_LOOPCHECK=0` skips it.
3. **Pin the tool chain and hash the output** (already done: `ocloc` version check, `make -C asm check`). Hashes
   catch regressions; they do not prove a new variant correct.
4. **Correctness gate per variant:** FP64 reference over the whole output (`STEP=1` in `bench/attn`), error bound
   fixed before the run; the poison binary is a negative control (the driver really runs our ISA)
   ([isa-notes.md](isa-notes.md) §5).
5. **Prefetch distance is one explicit parameter from the start** (attention `NS`; Triton calls the same thing
   `num_stages` [EXT]). IGC issues each read just before its use, so this is the parameter the compiler will not
   tune for you.
6. **Constraints before compiling and measuring.** Encode every structural condition as an assertion with a reason
   (attention: `NS <= D/8`, `nsteps % NS == 0`, a contiguous 8-register block for the ring). Like IREE's constraint
   pruning [EXT], this removes most of the space before anything runs.
7. **Bank-aware allocation in the allocator** (attention: `Alloc` with register parity), not as a fix-up pass after
   the fact; register fragmentation, not the count, can block a variant (Llama NS 4: 21 registers free, no
   contiguous even block of 8) [SW: offline generator run, 2026-11-13, not in `results/`].
8. **Select variants by shape, not by device:** key = (head dim, rows per thread, kind); `skl` and `whl` give the
   same code.

## 4. Sweeping variants

The parameter spaces in this project are small: the attention generator builds about 10 variants per shape, the
GEMM generator has three fixed kinds with no tile parameters. Search strategies from autotuners (dual annealing,
Bayesian optimisation, successive halving; useful above about 200 evaluations [EXT]) are not needed. Protocol:

1. Enumerate the variants the generator actually builds (assertions reject the rest).
2. Reject analytically: conflicts > 0, spill or scratch, launch longer than about 100 ms.
3. Correctness: FP64 reference, whole output.
4. Write the prediction (from the model) into the results file before the run.
5. Measure in alternating rounds (A B C A B C, separate processes, at least 3 rounds, median and range;
   [benchmarks.md](benchmarks.md) rule 4), nothing else on the GPU, clock recorded.
6. Keep the raw output in `results/`.

Noise: repeated runs within one session differ by about 0.3 % (HD 530, attention), comparisons across sessions or
machines by up to ±10 % [MEAS]; only differences above the within-session spread count, and only within one session.

## 5. External kernel compilers (Triton, TVM, MLIR/IREE, Halide)

None of them produces competitive Gen9 code (state November 2026, details in the report):

- Triton's Intel backend supports Max, Flex and Arc only; its fast path needs DPAS and 2D block loads, which Gen9
  lacks [EXT].
- TVM removed its `intel_graphics` schedules on 2025-02-15; the only Gen9 measurement (AutoTVM, 2019) reached
  0.62-1.09x of OpenVINO after tens of hours of tuning per model [EXT].
- MLIR's XeGPU/XeVM model Xe (DPAS, 2D blocks); IREE has no Intel configuration; Halide's GPU autoscheduler is
  trained on NVIDIA [EXT].
- All of them end in IGC or Mesa's compiler, so none controls GRF bank parity or load placement, the two levers that
  gave this project's ISA-level gains.

Use them as a source of ideas (sections 2-4), not of code.

## 6. Open items

- Shared generator check (section 3, rule 1); touches `asm/attn_gen.py`, coordinate with the attention work.
- Two-bank model: check what IGC's `hasBankCollision()` returns for SKL (the comment in `visa/GraphColor.cpp` says
  the src1/src2 collision is "upto pre-SKL"), repeat `results/micro-2026-11-11.txt` on the HD 530, extend to
  instructions other than `mad` ([isa-notes.md](isa-notes.md) §7). Mesa models Gen7-Gen9 with 4 banks [EST]; this
  project measured 2 banks by register parity [MEAS].
- Verify the section 2 candidates (local memory or not, shuffle broadcast) on NEO 24.35 on both devices; optionally
  one run of CLBlast's tuner (`clblast-utils` 1.6.2) on the UHD 620 as input.
- Documentation states IGC 1.0.17537.24 (from `ocloc` 24.35), while the installed `libigc1` is 1.0.17791.16; state
  which IGC compiled which binary.
