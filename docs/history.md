# How the library was developed

A technical history of `gen9blas`: the stages, the key measurements and the decisions that followed from them. All
work was done on one laptop (Core i5-8265U, Intel UHD Graphics 620, Whiskey Lake GT2, 24 EU, 1100 MHz; Linux
i915, NEO 24.35, IGC 1.0.17537.24). Every stage points to its `results/*.txt` file, which holds the raw output and
the prediction written before the measurement. Evidence tags as in the results files: [SPEC] Intel documentation,
[MEAS] our measurement of a hardware constant, [SW] software behaviour (for a given driver/compiler version), [EST]
estimate from third-party code, [HYP] unconfirmed hypothesis.

Two things changed late and shape how to read the early stages. First, the early GEMM work used OpenCL through
NEO; the default library today bypasses NEO (`kcl`, `docs/no-neo.md`), and the NEO path remains only as
`libgen9blas_neo.so` for comparison. Second, host memory handling went through three designs (copy, USM, userptr;
see "Host memory"). Gotchas are collected in `docs/pitfalls.md`; the numbers below are GFLOPS unless stated.

Contents: 1 Method, 2 Instruction map, 3 Arithmetic throughput and latency, 4 Reference points, 5 Assembly workflow,
6 Register banks, 7 Memory, 8 HGEMM, 9 SGEMM and the accurate variant, 10 Generator, batched, level 3,
11 Host memory and waiting, 12 Levels 1-2, 13 BLAS/CBLAS layer, 14 Comparisons, 15 Overhead and thresholds,
16 Skinny GEMM, Appendix A (reading Gen9 machine code), Appendix B (assembly workflow).

---

## 1. Method

The order of work was fixed in advance: (1) instruction set, (2) instruction throughput and latency,
(3) memory hierarchy, (4) threads and occupancy, (5) clock and energy. Each step produced a model
(`docs/performance-model.md`) with facts separated from guesses, then one variant and a measurement; every result
file starts with the prediction. Predictions were wrong often enough to be worth the discipline: the skinny-GEMM
compute ceiling (predicted 200-290 GFLOPS, measured 70-180) and the auto-mode thresholds (predicted too low).

## 2. Instruction map (`experiments/isa/`)

`experiments/isa/dump.sh <file.cl>` compiles a kernel with `ocloc` and writes the machine code of each kernel to
`experiments/isa/out/<file>/<kernel>.asm` (the `out/` directory is not kept in git). `experiments/isa/map.cl`
covers arithmetic, control flow, sub-group operations, SLM, atomics and sampler.

Findings [SW]:
- IGC picks SIMD32 by default: each vector instruction appears twice, `(16|M0)` and `(16|M16)`. One EU thread
  serves 32 work-items; 16 lanes is the widest FP32 instruction.
- The EU has no load/store: all memory access is a `send` message to the data port (`hdc.dc0`/`dc1`), sampler,
  gateway or spawner; latency is hidden by 7 threads per EU.
- Hardware: FP32 `mad`; FP32 divide and sqrt (`math` block); `native_*` transcendental functions as one `math`
  instruction; atomics (global atomic add of a SIMD32 group is coalesced into one `sends`); SLM atomics.
- Not hardware: FP16 `mad` (compiler emits `mul`+`add`, see below), INT64 multiply (73 instructions), full-precision
  `sin` (568), FP64 `sqrt` (145), INT8 dot product (no DP4a: 4 `mul` + 3 `add`), sub-group shuffle (`math.irem` plus
  indirect addressing `r[a0.0]`).
- Short `if/else` are predicated (both branches computed); `sub_group_reduce_add` is a register tree of 5 adds.

Documentation cross-check (PRM KBL Vol. 7 "Mixed Mode Float", Vol. 2d): mixed FP16/FP32 sources run at FP32 rate. `intel_sub_group_block_read4`
from a buffer compiles to an OWord aligned block read (128 B), `block_read8` from an image to a media block read (8 rows x 32 B),
the natural candidate for loading GEMM tiles (`asm/block_reads.cl`).

Anti-pattern found: vector types inside a work-item (`half16 acc` + broadcast) compile to 256 scalar `mad (1|M0)`;
write one value per SIMD lane instead.

## 3. Arithmetic throughput and latency

`bench/arith_throughput_latency.sh` (kernels `bench/peak.cl`, program `bench/clbench.c`): 131072 work-items,
work-group 256, 8 independent chains x 16 operations per loop, best of 5, event profiling, clock 1100 MHz throughout.
Peak = 24 EU x 2 FPU x 4 lanes x 1.1 GHz = 211.2 G lane-ops/s. Results:
`results/arith-throughput-latency-2026-11-11.txt`.

| Operation | Measured | Theory | Note |
|---|---|---|---|
| FP32 FMA | **410 GFLOPS** | 422.4 | 97%; `mad`, `fma`, `-cl-mad-enable` equal |
| FP16 `fma()` | **803 GFLOPS** | 845 | double rate; `mad()` without fast-math: 200 (mul+add); with `-cl-fast-relaxed-math`: 791 |
| FP16 mul/add, vector operand / broadcast scalar | 400 / 205 Gop/s | 422 | scalar operand loses the double rate |
| FP64 FMA | 90-94 GFLOPS | 105.6 | 1/4 of FP32 |
| INT32 add/xor / mul | 200 / **100** Gop/s | 211 | multiply at half rate |
| INT16 mul vector operand | 352 Gop/s | 422 | x1.75 |
| INT64 add+xor / mul | 50 / 12-15 Gop/s | | emulated |

Latency (one SIMD32 thread, one dependent chain; step = two SIMD16 instructions): FP32 13.4 clocks, FP16 10.9,
FP64 43.1, INT32 mul 23.2, INT16 mul 12.7. Seven threads with 8 chains hide it with margin.

Conclusions: Intel's theoretical figures are reachable to 95-97% (FP64 85-89%); FP16 is 2x FP32 only through
`fma()`/fast-math and vector operands; INT8 has no speed advantage (gain only in memory); FP16 `fma` is the fastest
"small precision" arithmetic on this chip.

## 4. Reference points (`results/gemm0-2026-11-11.txt`)

Before writing kernels: CLBlast 1.6.2 on the GPU (`bench/gemm_clblast.c`) and NumPy/OpenBLAS 0.3.26 on the CPU
(`bench/gemm_cpu.py`), square row-major matrices, best of 5, on mains power; power via `bench/power.sh`.

| N | CLBlast SGEMM | CLBlast HGEMM | CLBlast DGEMM | OpenBLAS f32 | OpenBLAS f64 |
|---|---|---|---|---|---|
| 256 | 86 | 76 | 33 | 308 | 136 |
| 1024 | **308** | 254 | 34 | 316 | 135 |
| 2048 | **54** | 188 | 33 | 299 | 104 |
| 4096 | **35** | 206 | 33 | 235 | 66 |

Findings: the CPU is stronger than expected (bursts up to 42 W, PL2; sustained load drops to PL1); CLBlast SGEMM collapses to
35-60 GFLOPS at N = 2000, 2048, 3000, 4000, 4096 (later attributed to TLB, sections 7-8); CLBlast HGEMM sums in FP16 (error
1.3e-4 mean, 1.3e-3 max, about 10 000x FP32) and ignores the double FP16 rate: the main opportunity. GPU DGEMM (33) is 4x below the
CPU; for N <= 256 the CPU is 2-5x faster (launch overhead). Targets: SGEMM steadily >= 300, HGEMM 500+, a mixed variant with
near-FP32 accuracy.

## 5. Assembly workflow (summary; details in Appendix B)

Hand-edited machine code does reach the GPU: removing half of the `mad` in a loop halved the time (82 -> 41 ms),
removing all gave 1 ms. The approach is a *template*: an OpenCL C kernel with the right signature lets the compiler
generate the prologue (argument loads, `get_global_id`), the `.ze_info` and the epilogue; only the hot loop is
replaced. Pitfalls ([SW]): `-device cfl` binaries are rejected, an attached `.spv` makes the driver recompile
silently, `ocloc asm` takes `-file`. Today everything is reproduced by `make -C asm`.

## 6. Register banks (`results/micro-2026-11-11.txt`)

Mesa 25.2.8 (`brw_opt_bank_conflicts.cpp`) describes 4 GRF banks (`bank = ((r & 0x40) >> 5) | (r & 1)`) and 3-source
`mad` conflicts on src2/src3 [EST]. `asm/micro.py` / `asm/micro.sh` build micro-tests from the template
`asm/fma32.cl`: 256 independent `mad (16) :f` per iteration, peak 422.4.

| Test | GFLOPS | Result |
|---|---|---|
| src1, src2 in different parity | 401-405 | no conflict |
| src1 = src2 | 413 | read once |
| src1, src2 same parity | 270-276 | **conflict, -33%** |
| src2 broadcast scalar, same parity / other parity | 322-326 / 402 | -20% / none |
| src0 same parity as src1 | 403 | src0 does not take part |
| result register rotation, `{Compacted}` or not | 403-410 | irrelevant |
| GEMM pattern (in-place accumulators, B vector, A scalar) | 408-410 | 97% of peak |
| GEMM pattern with A in B's parity | 326 | -20% |

Model: a SIMD16 `mad` occupies one FPU for 4 clocks (two SIMD8 halves x 2), two FPUs alternate (2 clocks effective);
a conflict adds 1 clock per SIMD8 half. **Two banks, by register parity**, only src1 and src2. Rules for a
hand-written loop: B (src1) and A (src2) in registers of different parity; accumulator anywhere; 8 in-place
accumulators hide the `mad` latency at 7 threads. The compiler does not enforce this.

## 7. Memory (`results/memory-bandwidth-2026-11-11.txt`, `results/memory-latency-2026-11-11.txt`)

Kernels in `bench/memory.cl` (`k_owb`, `k_owb2`, `k_media`, `k_lat`), SIMD8, 16384 work-items; `clbench` takes
`BUF=<bytes>` and `IMG=<w>x<h>`.

| Working set | OWord x1 | OWord x2 in flight | media block | latency |
|---|---|---|---|---|
| 64 KB-512 KB (L3) | 117 GB/s | **156-159** | 153 | 133 ns = 147 clocks |
| 4 MB | 101 | 156 | 83 | 400 ns = 440 clocks |
| 8-16 MB | 58-62 | 60-64 | 30-66 | 400 ns |
| 256 MB (RAM) | 18.5 | **20** | 19.5 | 400 ns |

L3 gives about 155 GB/s (74% of 211) only with two or more reads in flight. RAM gives about 20 GB/s (53% of the
37.5 GB/s DDR4-2400 dual-channel peak); the single-channel hypothesis was wrong (`dmidecode`: 2 x 8 GB), and the CPU
reaches 12.8 GB/s on one thread and 17-18 on 2-4: a platform limit. The CPU reaches 12.8 GB/s on one thread and 17-18 on 2-4: a platform limit (dual-channel DDR4-2400; the single-channel
hypothesis was wrong). Plateau from 4 MB suggests the GPU uses only part of the 6 MB LLC. Media block reads from L3 are as fast
as OWord reads and give a 2D block per message.

Budget for HGEMM at 800 GFLOPS: at least 5.2 FLOP/B from L3, 40 FLOP/B from RAM. A 16x32 register tile gives
1024 FLOP per 96 B per k step = 10.7 FLOP/B: enough for L3; RAM needs reuse through L3 across work-groups.

## 8. HGEMM: from OpenCL C to hand-written assembly (`results/hgemm-compare-2026-11-11.txt`)

Kernels `hgemm_16x32` (2D images), `hgemm_16x32g` (grouped order), `hgemm_16x32p` (packed data) in `bench/hgemm.cl`;
program `bench/hgemm.c` (env `PACK`, `ND1`, `LY`, `PAD`, `REPS`, `DUMP`); series `bench/hgemm_compare.sh`.

| Version (N = 4096) | GFLOPS | What changed |
|---|---|---|
| CLBlast HGEMM | 243 | reference |
| OpenCL C, 16x32 tile, media block reads from images | 32 (622 at N = 1024) | fast to N about 1728, then about 15x collapse |
| + grouped order (Triton-style) | 31-550 | helps only at group size 8 and N = 2048 |
| + **packing A and B** (contiguous tile panels, OWord block reads from a buffer) | **654** | collapse gone: **TLB** (4 KB pages, `has_64k_pages: no`) |
| + **hand-written loop** (`asm/hgemm_asm.py`) | **777** | 0 bank conflicts (compiler: 256/512), no `mov` in the loop, double-buffered half k steps |

Assembly: 733-777 GFLOPS for N >= 1024 (91-97% of measured FP16 peak 803, 87-92% of theory 845); 605-731 with GPU
packing included. Accuracy identical to CLBlast (FP16 accumulation).

| N | asm | asm + packing | CLBlast HGEMM | CLBlast SGEMM | OpenBLAS f32 |
|---|---|---|---|---|---|
| 256 | 520 | 279 | 77 | 91 | 316 |
| 1024 | 751 | 605 | 103-254 | 285 | 280 |
| 2048 | 733 | 647 | 224 | 68 | 325 |
| 4096 | 777 | 731 | 243 | 44 | 307 |

Power (N = 2048): asm 733 GFLOPS at 33 W package (CLBlast 26.3 W, OpenBLAS 36.2 W). Long run (`bench/long_run.sh`,
`results/long-run-2026-11-11.txt`): HGEMM asm N = 2048 for 4 minutes 745-761 GFLOPS, no throttling (1100 MHz, GPU 13.6 W);
OpenBLAS f32 for 90 s about 130 GFLOPS (about 300 when short: PL2 for 45 s, then PL1 17 W): beyond 45 s the GPU in FP16 is about
5.8x the CPU in FP32.

How the hand loop works (`asm/hgemm_asm.py`): the compiler's prologue and epilogue stay, the `k` loop is replaced. The accumulator
map (register -> row/column group of C) is found experimentally: the `map` mode writes the constant i into accumulator i, the
compiler's epilogue stores C, and reading C gives the mapping (`asm/hgemm_map.txt`, `make -C asm maps`). 32 accumulators, two
buffers x (A 8 + B0 8 + B1 8 registers) at even bases, 12 `send` headers. Layout A2 (`pack_a2`) puts A for even `kl` in an odd
register and vice versa, B row `kl` sits in a register of parity `kl`, so src1/src2 of every `mad` differ in parity. Order:
[256 `mad` from X][6 `send` into X][256 `mad` from Y][6 `send` into Y][counter, `cmp`, `jmpi`]; the Gen9 scoreboard handles
`send`->`mad` dependencies.

Lessons: the hand loop gave +19% over the best compiler code (777 vs 654) and 3.2x CLBlast; the main gain is
bank conflicts. Packing mattered more than assembly (it removed a 20x collapse). On the GPU HGEMM is 2.4x faster
and 2.4x more energy-efficient than the CPU in FP32 (short runs).

Packing (`pack_a`, `pack_a2`, ... in `bench/hgemm.cl`; N = 4096: A 5.8 ms, B 5.6 ms): "16 elements per work-item" versions were
slower (`pack_b16` 17.7 ms) because neighbouring work-items write 16 panels (pages). A later round (`bench/packing.c`,
`bench/packing.cl`, byte-exact against the reference) cut FP16 packing at n = 1024 / 4096 from 0.45 / 7.25 to 0.24 / 4.5 ms for B
(2 elements per work-item, `pk_bq_w2`) and from 0.54 / 8.9 to 0.31 / 5.7 ms for A (4 rows x 8 k, `pk_a2_q`); FP32 gains nothing.
The old kernels remain under `GEN9BLAS_PACK=old`. HGEMM n = 1024 call: 4.4-4.9 -> 4.1-4.5 ms.

## 9. SGEMM and the accurate FP16 variant

**SGEMM** (`bench/sgemm.cl`, `bench/sgemm.c`, `asm/sgemm_asm.py`): OpenCL C 16x16 with packing 342-351 GFLOPS (128 of 256 `mad`
conflicting); assembly **359-376 GFLOPS** (92% of FP32 peak), 0 conflicts, error as FP32 (about 1.5e-8). The B row is 2 GRFs
(even-`kl` rows in an even-based block, odd in an odd-based one: `pack_bf2`), buffers are quarters of k (too few contiguous free
registers for halves), read-ahead of 2 quarters; the script finds A/B pointers, counter and bound from the data flow. A `send`
header register may be changed right after the send (the hardware reads it at issue): one header per stream. CLBlast SGEMM with
padded rows still collapses (42-78): not cache-line conflicts; consistent with TLB, without direct proof.

**Accurate HGEMM** (`GEN9BLAS_HS`: FP16 data, FP32 partial sums). First in OpenCL C (`hgemm_16x32kc`, `hgemm_16x32mx`),
N = 1024 / 4096 and error (N = 4096):

| Variant | GFLOPS | Error mean / max |
|---|---|---|
| FP16 accumulation (`hgemm_16x32p`) | 620 / 633 | 1.3e-4 / 1.3e-3 |
| chunks KC = 16 -> FP32 | 371 / 383 | 1.1e-5 / 4.7e-5 |
| KC = 32 | 403 / 400 | 1.4e-5 / 6.3e-5 |
| KC = 64 | 426 / 434 | 2.0e-5 / 6.4e-5 |
| mixed (each product in FP32) | 230 / 234 | 2.9e-6 / 1.5e-5 |

Chunking gives 10-27x lower error at 60-70% speed; the loss is register spill (1280 B / 2880 B in `.ze_info`).
The assembly version (`asm/hgemm_kc_asm.py`, template `hgemm_16x16kc` in `asm/hgemm_kc.cl`): 16x16 tile, quarters of
k, layout A3 (`pack_a3_pad`); 8 k-quarters (32 products) in FP16, then `f += h` (mixed `add (8) :f <- :hf`, two per
row) and `h = 0`; 0 conflicts; the template analysis was extended to follow `mov :w` copies; accumulator maps `f` and
`h` read in two runs (`map_f`, `map_h`). Result **674-689 GFLOPS** (N = 1024-4096), error 1.4e-5 mean, 6.3e-5 max:
-11% speed for about 20x lower maximum error (OpenCL C version: 441). The gain from chunked sums appears only for
K > 32.

## 10. One generator, batched GEMM, level 3

**Generator** `asm/gemm_gen.py` (via `make -C asm`: `gen_h`, `gen_hd`, `gen_s` -> `bench/gen_*.bin`): HGEMM moved to a
single-pointer B layout, so one automatic analysis (A/B pointers from data use in `mad`, following `mov :w` copies) serves
all kinds. Templates are *batched* (matrix number b = pid / tile count, pointers offset by b x packed size; a single GEMM
is b = 0). Speed: H 715-765, HD 645-660, S 364-369. The `_v` kernels (`hgemm_v`, `hgemm_kc_v`, `sgemm_v`) take
`(Ap, Bp, C, M, N, ldc, K, Kfull)`: loop length independent of the panel stride, so any slice of data packed once can be
computed (offsets as sub-buffers, 128 B alignment, bands of multiples of 64 elements).

**Level 3 on top of GEMM** (`lib/gen9blas.c`):
- SYMM/TRMM: expand to a full matrix, then GEMM. SYRK/SYR2K: full product, only the `uplo` triangle written back; SYR2K is
  one GEMM with K' = 2K: [A|B]·[B|A]ᵀ. TRSM: reduced to L·Y = alpha·R, blocked (nb = 256): GEMM update plus inverse of
  the diagonal block.
- **Bands** (`GEN9BLAS_BLOCKS`, 4): the result is split into 4 bands, each one GEMM over the needed range only: about
  62% of the full product (2 bands: too much work; 6-8: call overhead wins). Operands are packed once and bands are
  slices of the packed data (`gen9blas_epilog_tri*` writes only the triangle).
- **Everything on the GPU** to offload the CPU: `gen9blas_expand_*` (full matrix from a triangle; 16x16 tiles in local
  memory, since a strided element-per-work-item mirror read took SYMM FP16 from 56 to 113 ms), `gen9blas_syr2k_merge_*`
  (was a CPU loop at 100% of a core), TRSM kernels `trsm_L_*`, `trsm_Y_*`, `trsm_store_*` and diagonal-block inverses in
  FP64 (`trsm_inv0..3_*`, recursive: 7 ms at n = 2048 instead of 23-190 ms). FP16 TRSM stores the inverses in FP16
  (175 / 285 GFLOPS, relative error 1-2e-3).

Progress (FP32 GFLOPS, n = 2048 / 4096): SSYRK 110 -> 156 (bands) -> 172-206 / 211 (pack once); STRMM 96 -> 148 -> 170-185 /
203; STRSM 59 -> 76 / 118 -> 104 / 177 (all on GPU) -> 190 / 249 (USM).

**Batched GEMM** (`gen9blas_gemm_batched`): FP16 kernel alone 407 (256 x 64³), 566 (64 x 128³), 636 (16 x 256³); whole call
57-158, later up to 225 with direct write (alpha = 1, beta = 0, M % 16 = 0). A loop of single calls: 1.1 GFLOPS (25x slower);
K = 32 reaches only 82 (the loop is too short to hide prologue and latency).

**Without NEO** (`results/no-neo-level3-2026-11-11.txt`): same code through `kcl` vs the NEO build, n = 4096 FP32: GEMM
349.4 / 349.2, SYMM 334 / 322, SYRK 266 / 258, TRSM 247 / 242; FP16 GEMM 720 / 719, TRSM 438 / 417; CPU 0% vs 4%.

## 11. Host memory, zero-copy and waiting

Host memory went through three designs (driver traps: `docs/pitfalls.md`, section 3).
1. **Copy** (with `CL_MEM_USE_HOST_PTR` for 4096-aligned pointers: 4-40% gain per call). Crashed NEO when operands
   overlapped in one call (fixed: no zero-copy for overlapping operands).
2. **USM host** (`gen9blas_alloc`, pool behind `gen9blas.empty`): after days of random NEO crashes with `USE_HOST_PTR` on
   ordinary memory (registration about 10 ms per 64 MB, as much as a copy), USM host ran 20 000 calls on varying pointers
   without a crash at 20-24 GB/s and 11% CPU. Level 3 at n = 4096 gained as a side effect: SGEMM 291-311 -> 354-358, STRSM
   190 -> 249, HGEMM 522-573 -> 698, HTRSM 308 -> 394 (no per-call registration).
3. **userptr without NEO** (`results/userptr-2026-11-12.txt`, `experiments/no-neo/userptr.c`, `lib/test_userptr.py`): ordinary
   process memory goes to the GPU without a copy. Pinning costs about 0.45 us per 4 KB page, so `kcl` keeps an LRU cache
   (1 GB, `KCL_USERPTR_MB`), one object per group of overlapping page ranges, reused only for exactly the same range
   (64 MB: 9.1 vs 8.8 ms against an own buffer, CPU 127 us instead of 12 ms). Repeated NumPy arrays: SGEMM n = 1024-4096 CPU
   0.02-0.20x, time 0.82-0.96x; axpy/dot time 0.15-0.24x, CPU 0.01-0.10x. `GEN9BLAS_NO_USERPTR` forces the old copy.

**Waiting.** NEO's wait spins on the CPU. Polling with `nanosleep` (20 us -> 0.5 ms) lowered package power from 25.1 to
18.9 W at equal speed (HGEMM about 41 GFLOPS/W); medians over 3 s runs show no speed loss. `GEN9BLAS_WAIT=busy` restores
`clFinish`.

**Energy** (`bench/level3_power.py` through `bench/power.sh`, n = 4096): moving level 3 to the GPU plus sleeping wait cut
energy per call 5-27% (HSYR2K 8.1 -> 5.9 J) and CPU share from 39-246% to 5-18% of a core. Lower GPU clock
(`gt_*_freq_mhz`): memory-bound level 1-2 are cheapest at 500 MHz (SAXPY 0.26 vs 0.53 J/GB at 1100, speed 20.4 vs 24.4
GB/s); GEMM n = 4096 at 500 MHz: -40% energy for 2.2x time.

## 12. Levels 1-2

One source `bench/level12.cl`, compiled three times (`-DHALF`: FP16 stored, FP32 computed; plain FP32; `-DDBL`; `-cl-std=CL3.0`).
Reductions accumulate in FP64 per work-item, up to 96 group results summed on the CPU. GEMV N: a sub-group of 16 handles 4
rows (`vload4`); GEMV T: work-item = 4 columns x a band of rows plus a summing kernel. Triangular variants use a mask on
(i, j) and an optional unit diagonal (SYMV = N pass plus T pass; SYR/SYR2 = GER with a mask); banded/packed forms use one
kernel `pmv` with an element function, solves `psv` use one group of 256 (n barriers) and cost more CPU than OpenBLAS, so
auto mode sends them to the CPU.

Speed-ups (`bench/level12_speed.py`): level 1 with 4 elements per work-item (reductions FP32 13-17 -> 19-23 GB/s, FP16 7-8
-> 13-16); TRSV blocks of 256 (sub-blocks in local memory, unrolled reads) n = 8192 FP32 27/18 -> 14/12.5 ms (N/T); SYMV in one
pass 8.2 -> 19.2 GB/s. A single-launch TRSV with flag waits was always slower (n = 8192, N/T: 54-181 / 70-163 ms vs 28/18);
the cost is the dependency chain, not the synchronisation primitives; a wait loop run by all lanes of a sub-group hung
the GPU. FP64: peak 72.6 GFLOPS (FP32 362 in the same test); DGEMM 56 GFLOPS (79%) with two sub-groups of 8 per 16x16 tile;
the FP64 BLAS layer is generated from the FP32 code (`lib/gen_blas_d.py`).

Fixes found by comparing with ViennaCL (`results/viennacl-2026-11-12.txt`, `results/message-16B-2026-11-12.txt`): 16 B per
lane for COPY and row-wise GEMV in FP64 (17.4 -> 24.2 GB/s), FP16 copy through the 32-bit kernel, FP16 GEMV through
`intel_sub_group_block_read_us*` (15-16.5 -> 20-25 GB/s).

## 13. BLAS/CBLAS layer

`libgen9blas_blas.so` exports the same 393 symbols as the system `libblas.so.3`: 37 implemented (S and D, level 2-3 in CBLAS
and Fortran, CBLAS level 1), 356 forwarded by `jmp` to `libopenblas.so.0` (`dlopen` in a constructor, refusing a library that
resolves to our own `cblas_sgemm`; generated by `lib/gen_forward.py` from `lib/blas_symbols.txt`). With `LD_PRELOAD`, NumPy
`sgemm`, `sgemv`, `ssyrk` are intercepted; LAPACK inside OpenBLAS is not. NumPy float64 n = 4096 `A @ B`: 2650 ms at 10% CPU vs
OpenBLAS 2060 ms at 755%. Per-call CPU/GPU choice: section 15.

## 14. Comparisons with other libraries

Programs in `bench/compare/` (`make -C bench/compare`, run from the repository root): same operation and data for
`gen9blas`, CLBlast 1.6.2 (GPU) and OpenBLAS 0.3.26 (CPU, 8 threads); row-major, alpha = 1, beta = 0; best of 5; CPU = process
CPU time / wall time; data already in GPU memory (`MODE=host` for ordinary memory).

**Level 3** (`results/compare-l3-2026-11-11.txt`, `results/compare-l3-power-2026-11-11.txt`; GFLOPS):

| | n | gen9blas | CLBlast | OpenBLAS (CPU) |
|---|---|---|---|---|
| SGEMM | 1024 / 2048 / 3000 / 4096 | 311 / 345 / 337 / 351 | 295 / **88 / 78 / 43** | 343 / 342 / 343 / 305 (~770% CPU) |
| STRSM | 1024 / 2048 / 4096 | 91 / 182 / 256 | 15 / 18 / 9 | 288 / 319 / 245 |
| HGEMM | 1024 / 2048 / 4096 | 442 / 628 / 699 | 259 / 235 / 242 | none |
| DGEMM | 2048 | 56 | 39 | 100 |

`gen9blas` beats CLBlast everywhere (FP32 at n >= 2048 4-12x, TRSM 6-27x, FP16 1.7-4x); CLBlast's row-major SSYRK/SSYR2K
returns wrong values for n > 128. Against OpenBLAS: tie for FP32 at large n, OpenBLAS 1.7-3x faster at n = 1024 and
1.6-2.5x in FP64, but it uses about 770% of the CPU against 2-13% of one core (part of that gap is only the wait mode).
Energy n = 4096: SGEMM 8.8 J vs CLBlast 44 vs OpenBLAS 12.2-19.2 J; HGEMM 3.8 vs 11.5 J.

**OpenVINO 2026.1** (`results/compare-openvino-2026-11-11.txt`, `bench/compare/ov_gemm.py`): the strongest rival with weights as
a graph constant (`OV_CONST=1`): equal at n = 1024-2048, `gen9blas` 7-28% faster at 4096, 1.5-3.5x with B as an ordinary
input; n = 3000 collapses OpenVINO (FP32 116, FP16 244). Its FP32 is more accurate (4e-9 vs 1.6e-8).

**oneDNN v3.6** (`results/compare-onednn-2026-11-11.txt`; the last release with a Gen9 GEMM catalogue; `benchdnn` reports kernel
time only): kernel to kernel `gen9blas` wins (FP32 +7-14%; FP16 +3-12% at n = 1024 and 4-6x at n >= 2048, where oneDNN falls to
120-200). At n = 1024 oneDNN's FP16 kernel (3.1-3.3 ms) beats the whole `gen9blas` call (4.8 ms: kernel 2.98 plus
packing/epilogue about 1.8), which motivated the packing work (section 8).

**ggml Vulkan (Mesa ANV)** (`results/compare-ggml-vulkan-2026-11-11.txt`): 92-142 GFLOPS, 2.5-7x slower; no cooperative
matrices on Gen9.

Summary (GEMM, best path of each library, n = 4096 / 1024, GFLOPS): `gen9blas` FP32 344 / 303, FP16 699 / 448 (kernel 763 /
721); oneDNN FP32 332 / 345, FP16 120 / 642-703; OpenVINO (constant weights) 300 / 277, 547 / 465; CLBlast 43 / 295, 242 / 259;
ggml Vulkan -- / 142, -- / 128; OpenBLAS 305 / 343. Rivals win only at n = 1024 (packing overhead) and in FP32 accuracy.

**ViennaCL 1.7.1** (`results/viennacl-2026-11-12.txt`, `bench/compare/viennacl_gen9blas.py`; the OpenBenchmarking test repeated
here): vectors and GEMV tie (0.95-1.08x) apart from two weaknesses later fixed; GEMM FP32 317-326 vs 22-60 (5-14x), FP64 about
55 vs 17-34; ViennaCL loses 1.5-2.7x on TN/TT, `gen9blas` is flat across transpositions.

## 15. Fixed call cost, auto thresholds, GEMV N on plain memory

(`results/overhead-2026-11-12.txt`, `results/auto-thresholds-2026-11-12.txt`, `results/gemv-n-2026-11-12.txt`.) After userptr the
old USM/plain thresholds had no basis. Criterion: process CPU time per call, GPU vs OpenBLAS with 1 and 8 threads.
1. **Thresholds** (`bench/auto_thresholds.py`): OpenBLAS splits axpy, gemv, ger, symv, trmv, syr over threads from about 2^14
   elements and the threads spin, so the GPU uses 2-10x less CPU there; for unsplit functions and 1 thread the GPU pays off from
   2^20 (level 1) and 2^22 (level 2). Implemented in `lib/gen9blas_blas.c`. The model had predicted them too low.
2. **Fixed cost** (`bench/overhead.c`, `KCL_PHASES=1`): mains power 48 us wall / 14-17 us CPU for a small axpy, our part 3-4 us,
   the rest i915; large calls 120-145 us CPU, half of it waking a core from deep C-states. On battery 2-3x more.
3. **GEMV N** (`bench/gemv_n.py`): slow (10-11 GB/s) only on plain (userptr) memory, 22-26 on USM; the grouped variant
   (`gemv_nw`) gives 18.9-21 there (`GEN9BLAS_GEMV_N=old|grouped`): userptr (4 KB pages) loses when a kernel touches many pages
   at once.

## 16. Skinny GEMM (small M or N, no packing)

(`docs/skinny-gemm.md`, `results/skinny-gemm-2026-11-12.txt`, `results/skinny-gemm-rework-2026-11-12.txt`,
`results/viennacl-skinny-2026-11-12.txt`; `bench/skinny_gemm.py`, `lib/test_skinny.py`.) Context: token-by-token generation
(M = 1-16 rows against large weights). Baseline: the GEMM packed the whole weight matrix and computed full-height tiles:
HGEMM M <= 16 at 4096² took 10.4 ms (3.2 GB/s of weights), independent of M.

New kernels in `bench/level12.cl` (FP16 and FP32; FP64 unchanged): `skinny_k1/k4/k8` (weights contiguous along K, A·Wᵀ,
`gemv_nw`-like layout) and `skinny_n4`/`skinny_n2` (weights contiguous along N, A·B, K parts plus reduction); FP32
accumulation also for FP16. Dispatch in `gen9blas_gemm_batched`: single GEMM with min(M, N) <= 16 (small N: roles swapped,
Cᵀ = op(B)ᵀ·op(A)ᵀ); `GEN9BLAS_SKINNY=0` disables, `GEN9BLAS_SKINNY=n` changes the threshold.

Result, FP16 A·Wᵀ 4096² (USM, median ms; memory floor 32 MB / 22-25 GB/s = 1.3-1.5 ms):

| M | old path | v3 (skinny) | block reads | rework (`skinny_r`) | old -> final |
|---|---|---|---|---|---|
| 1 | 10.38 | 1.85 | 1.52 | 1.52 | 6.8x |
| 4 | 11.12 | 2.43 | 2.30 | 1.68 | 6.6x |
| 8 | 11.00 | 3.92 | 2.94 | 1.90 | 5.8x |
| 16 | 10.77 | 6.83 | 4.92 | 3.19 | 3.4x |

FP32 4096²: M = 1 14.15 -> 3.07 ms, M = 8 14.05 -> 3.47, M = 16 13.89 -> 4.11. LLM shapes at M = 1 (11008x4096): FP16
6.6-6.9x, FP32 4.9-5.3x; FP16 error (FP32 accumulation) about 50x smaller. Compared with ViennaCL (FP32 A·Wᵀ, M = 1-16):
ViennaCL's GEMM takes 30-135 ms (1-6 GB/s of weights; full-height tiles), `gen9blas` 12-14x faster at 4096², and GEMV equal.

The rework (`skinny_r`, SIMD8, 8 weight rows, 4 sub-groups per group) came from an assembly diagnosis: the first loop had 165
instructions for 64 useful FMAs (dot not fused into the accumulator, mixed-mode ops split into two SIMD8, conversions),
about 25% of FMA peak. The model gaps found: weight bandwidth depends on the contiguous bytes read per row per group step
(256 B 14-18 GB/s, 512 B 19-20, 1 KB 21-24; [MEAS]); long rows (44 KB) with 8 rows per group drop to 16-17 GB/s [HYP: TLB];
for M = 8-16 the time is nearer t_mem + t_comp than the maximum (too little overlap [HYP]). Rework gain over the
block-read version: 1.2-1.8x for M = 2-16. Not done: A in local memory, larger RW with small MT, hand-written assembly.

Two traps from this stage (`docs/pitfalls.md`): spill/private arrays go to scratch that `kcl` does not configure (silently wrong
results; the generator now rejects spills), and 64-bit address multiplication in loops is emulated.

---

## Appendix A. Reading Gen9 EU machine code

```
mad (16|M16)  r113.0<1>:f  r8.3<0;0>:f  r117.0<2;1>:f  r8.2<0>:f  {Compacted}
 |    |   |     |     |  |    + region <vertical stride;width,horizontal stride>; <0;..> = scalar broadcast
 |    |   |     |     |  + type: f=FP32, hf=FP16, df=FP64, d/ud=INT32, w/uw=INT16, b/ub=INT8, q=INT64
 |    |   |     |     + destination stride
 |    |   |     + GRF register: r0-r127, each 32 B (8 x 32 bit)
 |    |   + channel-mask offset: M0 = lanes 0-15, M16 = lanes 16-31
 |    + SIMD width of the instruction (1, 2, 4, 8, 16)
 + opcode
```

- `(W)`: WriteEnable, executed regardless of the channel mask (scalar and "shared" operations).
- `(f0.0)` / `(~f0.0)`: predication by a flag register set by `cmp`.
- `{Compacted}`: 8 B encoding instead of 16 B.
- `send` / `sends`: message to a unit outside the EU; the comment names the target: `hdc.dc0` / `hdc.dc1` (data port:
  global memory, SLM, atomics), `sampler`, `gateway` (barriers), `spawner` + `{EOT}` (end of thread). `wr:a+b, rd:c` is the
  number of registers sent and received.
- `a0`: address register for indirect GRF addressing (`r[a0.0]`). `n0` + `wait`: notification, barrier wait.
- `acc`, `mach`: accumulator, the high half of a product.

Rules from the PRM (KBL Vol. 7 "Register Region Restrictions", "Execution Data Type") [SPEC]: float sources must share one type
except in mixed HF/F mode (`mad (8) r10.0.xy:f r11.0.xxxx:f r12.xyzw:hf r13.yyyy:hf`); ExecSize x largest element size <= 64 B; an
operand spans at most 2 adjacent GRFs; float and int sources may not be mixed. Block reads: OWord block read 1/2/4/8 OWord per
offset (mask ignored in the aligned form); media block read up to 256 B, width at most 32 B, linear pitch a multiple of 64 B, base
aligned to 32 B (PRM Vol. 7 data port, Vol. 2d `MH_MB`).

## Appendix B. Assembly workflow (current)

Pipeline (what `asm/disassemble.sh` and `asm/assemble.sh` do):

```
kernel.cl --ocloc compile -device whl--> zebin (ELF)
          --ocloc disasm--> directory: .text.<kernel>.asm (edit), .ze_info (argument/payload description), ...
          --(remove .spv and its line in sections.txt)--
          --ocloc asm -file X -dump dir--> dir/X --> clCreateProgramWithBinary
```

Rules ([SW], see `docs/pitfalls.md`): `-device whl`, no `.spv`, `-file` not `-out`. `bench/clbench` and the other bench programs
accept a `.bin` instead of `.cl`.

The repository does not keep a per-binary build script; everything is in `asm/Makefile`:

| Command | Effect |
|---|---|
| `make -C asm` | rebuilds the 6 binaries in `bench/` (`gen_h`, `gen_hd`, `gen_s`, `hgemm_asm`, `hgemm_kc_asm`, `sgemm_asm`) from the frozen templates `asm/gen_h.cl`, `gen_s.cl`, `hgemm.cl`, `hgemm_kc.cl`, `sgemm.cl` and the tracked accumulator maps `asm/*_map.txt`; no GPU needed |
| `make -C asm check` | rebuilds into a temporary directory and compares with `bench/*.bin` (`cmp`) |
| `make -C asm maps` | regenerates the maps on the GPU (`asm/maps.sh`; needs `bench/hgemm` and `bench/sgemm`: `make -C bench`) |
| `make -C lib` | embeds `bench/gen_h.bin`, `gen_hd.bin`, `gen_s.bin` into `lib/kernels.c` (via `lib/gen_kernels.py`) |

Steps inside the makefile: template (OpenCL C) -> `asm/disassemble.sh` -> directory `asm/<name>.d` -> generator in `loop` mode
(`asm/gemm_gen.py`, `asm/hgemm_asm.py`, `asm/hgemm_kc_asm.py`, `asm/sgemm_asm.py`) plus the map -> `.bin`. Generator modes:
`map` (write the constant i into accumulator i; running the kernel with `DUMP=1` prints which register is which element of C;
`map_f`/`map_h` for the two accumulator sets of the accurate variant), `loop` (replace the compiler's loop with the hand-written
one), `analyze` (print the pointer/counter analysis of the compiler's loop; also in `asm/loop_analysis.py`).

Template principle: IGC generates prologue, `.ze_info` and epilogue for a kernel with the real signature; the generator replaces
the instructions between the loop label and the back-jump (all of them; see `docs/pitfalls.md`), recognises the A/B pointers,
counter and bound from the data flow (following `mov :w` copies) and appends the hand loop. Single-instruction experiments:
`iga64 -p=9` (from `libigc-tools`).
