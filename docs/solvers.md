# Solvers for Gen9 kernels: when, how, and how to describe the kernel and the GPU

Constraint solvers (CP-SAT from Google OR-Tools, MIP, SMT) can answer exact questions about a kernel at the ISA level:
"does this register layout exist?", "what is the cheapest fix?", "is this impossible?". They cannot design a kernel.
This guide is based on the first use in this project (2026-11-13): `experiments/solver/`,
`results/solver-regalloc-2026-11-13.txt`. The levels of kernel work are in [kernel-guide.md](kernel-guide.md); the ISA
rules a model must encode are in [isa-notes.md](isa-notes.md) §4.

## 1. When a solver helps and when it does not

| Question | Tool | Example from this project |
|---|---|---|
| Does a placement satisfying hard rules exist; if not, prove it | CP-SAT feasibility | Llama attention NS 4: impossible, needs 8*NS + 21 registers, 45 free |
| Cheapest change that makes it legal | CP-SAT with a cost objective | Llama NK=1: one `mov (8)` of a Q register removes 8 bank conflicts |
| Is our heuristic optimal | solver vs the heuristic on all instances | attn_gen `Alloc` first-fit: 0 disagreements over 320 cases |
| Which tile / buffering fits before writing a kernel | budget model over a sweep of designs | accurate HGEMM 16x32: fits only with live-in epilogue registers and no spare GRF; 8x32 fits but reuses less |
| Placement in time (prefetch distance, send order) | scheduling model, **only after calibration** | single-thread model rejected: does not reproduce NS2 vs NS4 |
| Algorithm, data layout, packing, tiling | none: design work, then measurement | packing gave 20x (TLB); no solver would propose it |
| Choice among about 10 variants | enumeration and measurement ([kernel-guide.md](kernel-guide.md) §4) | NS / NK / RQ of attention |

Do the counting bound first. Most "does it fit" questions end with a sum of register demands vs free registers; the
solver earns its place when parity, contiguity, alignment and existing registers interact, and when you want a proof
plus a minimal fix instead of a yes/no.

## 2. Workflow

1. **Extract the instance from real artefacts, not from memory.** Use the generator's own analysis on IGC's code
   (here `attn_gen.igc_maps` and `live_in` on the disassembly from the pinned `ocloc`), so the solver sees exactly
   the registers IGC left. Record the tool chain version and the source commit with the instance.
2. **Write the model as a mirror of the generator's rules** (slot sizes, parities, which values are live across the
   loop). Keep it separate from the generator; do not import private functions that may change.
3. **Validate before optimising.**
   - Reproduce what is known: the model must agree with the existing heuristic on every case where the heuristic
     succeeds (here: 0 disagreements), and with every measured result it claims to predict.
   - A timing model must be calibrated against measurements (at least two points that differ in one parameter) and
     **rejected** if no parameter value reproduces them. Then do not run the optimisation: it would produce
     confident numbers from a wrong model (this happened with the single-thread schedule model, section 5).
4. **Solve a sweep, not one point.** All shapes x all generator options; record status (OPTIMAL / INFEASIBLE /
   FEASIBLE with a time limit), objective and time. An INFEASIBLE from CP-SAT is a proof only for the model as written.
5. **Feed the result back as code or as a documented limit:** a generator assertion, a fix (one `mov`), or a sentence
   in the design doc ("NS 4 needs Q in FP16"). Then build, check (FP64 reference, conflicts = 0, spill = 0) and measure
   on the GPU. A solver result is a prediction until measured.

## 3. Building the model (patterns for Gen9)

Registers are 128 GRF of 32 B (r0 reserved). Useful CP-SAT encodings (`experiments/solver/solve.py`):

| Rule | Encoding |
|---|---|
| A value in one register | integer `x` in 1..127 plus an interval `[x, x+1)` |
| Fixed register (IGC's, r0, pointers, epilogue inputs) | fixed interval |
| Block of n contiguous registers starting even (send destination, ring slot) | `b = 2*y`, interval `[b, b+n)` |
| Parity (bank of the 2-bank model, [isa-notes.md](isa-notes.md) §4.1) | `x = 2*y + p`, `p` fixed or a 0/1 variable |
| No two values in one register | `AddNoOverlap` over all intervals |
| Balanced parities (e.g. Q elements swapped pairwise between an even and an odd register) | `2 * sum(p_i) == count` |
| Relocating an existing register at a cost | Boolean `moved <=> x != original`, cost = number of `mov` (before and, if live after, after the loop) |
| Register pressure over time (scheduling) | interval from the send to the last read, `AddCumulative` with the budget |
| Order of an address stream (header increments) | `x_{i+1} >= x_i + issue_cost` |

Objectives: none (feasibility), number of `mov`s, makespan of a few unrolled iterations (scheduling: unroll 3-4
blocks and measure the middle ones, or write a proper modulo model). Keep models small: the allocation model solves
320 instances in 1.7 s; if a model needs minutes, aggregate (one interval per step of mads, not per `mad`).

## 4. Describing the kernel vs the GPU

Keep three things apart; mixing them is how wrong models hide.

1. **GPU description** (`gpu`), one per device, every number with an evidence tag and a source
   ([performance-model.md](performance-model.md), [hardware.md](hardware.md)):

   ```json
   {"device": "UHD 620 (Gen9.5 GT2)", "eu": 24, "threads_per_eu": 7,
    "grf": {"count": 128, "bytes": 32, "reserved": [0], "tag": "SPEC"},
    "banks": {"model": "parity of the register number, src1/src2 of mad", "penalty": "x0.66 mad rate",
              "tag": "MEAS", "source": "results/micro-2026-11-11.txt", "unknown": "other instructions"},
    "send": {"block_read_regs_used_here": 4, "ring_slot_regs": 8, "slot_align": "even", "tag": "SW"},
    "latency": {"l3_clk": 147, "ram_clk": 440, "tag": "MEAS"},
    "issue": {"note": "co-issue of mad with send/add unknown", "tag": "HYP"},
    "preempt_ms": 640}
   ```

   What is unknown stays explicitly unknown (`"tag": "HYP"`), so a model that depends on it is marked as such.
2. **Kernel instance** (`kernel`), extracted per shape and per build: generator, template, build options, tool chain
   (`ocloc` version), commit; IGC's fixed registers by role (Q, O, l, pointers, counter, epilogue inputs); the values
   the generator must place with their sizes and constraints; for scheduling, the instruction stream as steps
   (length, which sends feed it, first and last use offsets). Example: `experiments/solver/instances.json`.
3. **Model** (code) that joins the two, plus the **configuration** being asked about (NS, PE_IN_S, COMPACT,
   relocation allowed). Results go to `results/` with the versions of all three and the solver.

The split lets one kernel instance be checked against another device description (HD 530 vs UHD 620: same ISA, other
EU count) and lets a corrected GPU fact (e.g. the bank model) re-run every model without re-extracting kernels.

## 5. Lessons from the first use

- The allocation question was exact and well posed: the solver proved the Llama NS 4 limit (a count bound, not
  fragmentation as a review had assumed) and found a one-instruction fix for NK=1. Cheap, decisive, useful.
- The scheduling question was not: a single-thread model of prefetch distance gives a step function (prefetch either
  fully hides the latency or stalls every step) and cannot reproduce NS2 287.7 vs NS4 295.5 GFLOPS (both about
  0.9 of the issue bound). Latency on Gen9 is hidden mostly by the other threads of the EU [HYP]. A scheduling model
  needs several threads and the shared send port / L3 (a discrete-event simulation, calibrated with OA counters,
  [isa-notes.md](isa-notes.md) §6) before a solver can optimise anything.
- The second use (accurate HGEMM 16x32 vs 8x32, `results/solver-hgemm-budget-2026-11-13.txt`) was answered before
  any kernel existed, from overhead measured on the existing kernels. It also exposed a conservative rule in a
  generator: `gemm_gen.free_regs` blocks every register the epilogue mentions (9-17), while only 8-9 are live into
  it (1-2 of them send headers that can be zeroed after the loop). Check such rules before concluding "does not fit":
  the overhead model is part of the instance.
- A multithreaded model fixed it (`experiments/solver/mt_model.py`, `results/mt-model-2026-11-13.txt`): 7 threads per
  EU, 2 FPU pipes, a data-port FIFO with limited bytes per clock and a fixed latency. Fitted on the 3 Qwen points
  (L about 300 clk, about 6 B/clk/EU) and checked on 2 Llama points within 3 %. It says Qwen is bandwidth-bound and
  Llama is not, so prefetch depth is not the lever for either. Fit few parameters, then predict points the fit has
  not seen before trusting the model.
- And test the model's *explanation* separately from its fit. A microbenchmark (`bench/l3port.cl`,
  `results/l3port-2026-11-13.txt`) measured the port at 7.1-7.3 B/clk/EU on HD 530 (7.6 on UHD 620), and in the Qwen
  ratio (8 B per mad) it reproduces the real loop's 295-305 GFLOPS with OA showing issue-bound (FPU 95 % of active,
  stall 5 %) at ~87 % of the port. The fit still holds, but its "6 B/clk/EU port" lumps issue overhead and latency
  together: Qwen is issue-bound, not bandwidth-bound. A fitted parameter is not a hardware constant until measured.
- Compute was not the limit: 320 allocation solves took 1.7 s. A many-core machine (the 32-thread desktop, 32 threads) matters only
  for large scheduling models or wide sweeps; run there with `nice` and a fixed `num_workers`.

## 6. Running

```
# kp-laptop (pinned ocloc, no GPU needed): extract instances
python3 -I experiments/solver/extract.py "$PWD" experiments/solver/instances.json
# any machine with OR-Tools (the 32-thread desktop: ~/scratch/gen9-regalloc/.venv, `uv pip install ortools`)
python3 solve.py instances.json 4            # allocation sweep -> results.json + table
python3 sched_model.py calib                 # schedule-model calibration (rejected, see section 5)
```

## 7. Open

- Done 2026-11-13: equivalence checker `asm/loopcheck.py` (recommendation 4): gemm_gen h/hd/s (exact or up to a
  layout bijection) and attn_gen sched (product multisets traced through the softmax); remap (variant R) not covered.
- Measure the model's predictions (Qwen bandwidth bound, Llama NS3 +1.2 %) and extend it to GEMM; a solver on top of it only if it shows a lever.
- The GPU description as a file (`experiments/solver/gen9.json`) read by every model, instead of constants in code.
- The Llama NK=1 fix: verify that Q is not read after the loop, implement in the generator (attention work), measure.
