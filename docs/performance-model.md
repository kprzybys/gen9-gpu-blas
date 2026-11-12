# Performance model of the UHD 620 (Gen9.5) for gen9blas

Goal: predict kernel and call times **before** writing code, and use measurements only to confirm the prediction.
A mismatch between prediction and measurement means the model has a gap: fill it with a targeted microbenchmark,
not with another kernel variant.

Hardware facts and their Intel citations are in [hardware.md](hardware.md). Measurement method and how to reproduce
results is in [benchmarks.md](benchmarks.md). Chronology of how the numbers were obtained is in [history.md](history.md);
the gotchas behind several rows below are in [pitfalls.md](pitfalls.md).

Conventions:

- every number has a source and an evidence tag:
  - **[SPEC]**: Intel documentation (PRM KBL, Gen9 whitepaper, ARK; see [hardware.md](hardware.md));
  - **[MEAS]**: our measurement of a hardware constant (results file in `results/`, or the document named in the source column);
  - **[SW]**: software behaviour, fixed for a given driver and compiler version (NEO 24.35.30872.36, IGC 1.0.17537.24 as
    used by `ocloc` 24.35);
  - **[EST]**: estimate from third-party code (Mesa), not a document;
  - **[HYP]**: unconfirmed hypothesis;
- **constants** (sections 1-3) change only with the hardware or the software version; **dynamic** quantities
  (section 4) depend on the moment and must be controlled in every measurement;
- after a driver or compiler change, re-check every [SW] row.

## 1. Hardware constants from documentation [SPEC]

| Quantity | Value | Source |
|---|---|---|
| EU / threads per EU / subslices / slices | 24 / 7 / 3 x 8 EU / 1 | Gen9 whitepaper; confirmed by clinfo and i915 debugfs |
| GRF | 128 x 32 B per thread | Gen9 whitepaper |
| FPU | 2 per EU (SIMD-4 FP32 or SIMD-8 FP16, FMA every clock); FP64 on one of them only | Gen9 whitepaper |
| Peak at 1.1 GHz | FP32 422.4, FP16 844.8, FP64 105.6 GFLOPS | derived from the above |
| GRF -> FPU read | 96 B/clock, write 32 B/clock per EU | Gen9 whitepaper |
| Memory line | 64 B along the whole path (SIMD-16 x 4 B register, bus, L3 line) | Gen9 whitepaper |
| Data port | 1 per subslice, merges work-item accesses into 64 B lines; 64 B/clock to L3 | Gen9 whitepaper |
| L3 | 768 KB (512 KB for application data), 192 B/clock total = about 211 GB/s at 1.1 GHz | Gen9 whitepaper |
| SLM | 64 KB per subslice | Gen9 whitepaper |
| GPU <-> LLC (GTI) | 64 B/clock read; write 64 or 32 B/clock | Gen9 whitepaper |
| LLC | 6 MB, shared with the CPU | ARK |
| RAM | 2 channels DDR4-2400, 37.5 GB/s theoretical, shared with the CPU | ARK |
| GPU clock | 300-1100 MHz (RPn-RP0) | ARK, i915 debugfs |
| Timestamp clock | 12 MHz (83 ns) | i915 debugfs |
| GPU memory pages | 4 KB (`has_64k_pages: no`) | i915 debugfs |
| Data port message types | byte scattered (1/2/4 B per lane), dword scattered, untyped surface (1-4 channels), OWord block (1-8 x 16 B), media block (2D) | PRM KBL Vol. 7, Vol. 2d |

## 2. Hardware constants measured [MEAS]

| Quantity | Value | Source |
|---|---|---|
| FMA FP32 / FP16 / FP64 | 410 / 803 / 90-94 GFLOPS (97 / 95 / 85-89% of peak) | `results/arith-throughput-latency-2026-11-11.txt` (FMA loop microbenchmark, `bench/arith_throughput_latency.sh`); `results/micro-2026-11-11.txt` for FP32 (up to 410.8). A later FP64 run gave 72.6 GFLOPS, but FP32 in that run reached only 362, so the conditions were worse |
| FP32 mul/add, INT32 add | 200 Gop/s; INT32 mul 100 (half rate) | `results/arith-throughput-latency-2026-11-11.txt` |
| FP16 two-operand ops with a scalar operand (`<0;1,0>`) | lose the double rate (205 instead of 400 Gop/s) | `results/arith-throughput-latency-2026-11-11.txt` (`half k_mul`, `k_add` vs `k_mulv`, `k_addv`) |
| GRF banks | 2 banks by register-number parity; conflict only for src1/src2 of `mad` (410 -> 272 Gop/s); src0 does not matter | `results/micro-2026-11-11.txt` |
| Read latency (OWord block, dependent) | L3: 133 ns (147 clocks); LLC: 200-270 ns; RAM: 400 ns (440 clocks) | `results/memory-latency-2026-11-11.txt` |
| Block read bandwidth | <= 2 MB (L3/LLC): 116 (OWord x1), 156 (OWord x2), 154 GB/s (media); 8-16 MB: 60-65; >= 64 MB (RAM): 19-27 GB/s | `results/memory-bandwidth-2026-11-11.txt` |
| RAM bandwidth in practice | GPU about 20 GB/s (FP32 axpy 23.7 GB/s), one CPU thread 17-18 GB/s | `results/auto-thresholds-2026-11-12.txt` (header), `results/message-16B-2026-11-12.txt` |
| LLC threshold | bandwidth drops at a 4-8 MB working set | `results/memory-bandwidth-2026-11-11.txt` |
| Page pinning for userptr (first execbuf) | about 0.45 us CPU per 4 KB page (about 10 GB/s copy-equivalent; DMA mapping through the IOMMU, `DMA-FQ`) | `results/userptr-2026-11-12.txt` |
| GPU read from a userptr object | same as from a GEM buffer for sequential reads (64 MB: 9.1 vs 8.8 ms) | `results/userptr-2026-11-12.txt` |
| userptr with many pages touched at once | a kernel touching hundreds of 4 KB pages at once (gemv N: 4 rows per subgroup) loses 2x on userptr (10.5 vs 22.5 GB/s on USM); 1 page per row per group: 18.9 GB/s. [HYP]: larger PPGTT pages for GEM objects aligned to 64 KB, 4 KB only for userptr | `results/gemv-n-2026-11-12.txt` |
| Writes to one 64 B line through two GPU addresses (two objects over the same pages) in one kernel | one write is lost | `results/userptr-2026-11-12.txt`, `lib/test_userptr.py`, [no-neo.md](no-neo.md) |

## 3. Software behaviour [SW] (NEO 24.35.30872.36, IGC 1.0.17537.24 from `ocloc` 24.35)

| Quantity | Value | Source |
|---|---|---|
| CPU cost of `clEnqueueNDRangeKernel` | about 25 us | [api.md](api.md) |
| Python wrapper (ctypes) | about 50 us per call | [api.md](api.md) |
| Fixed cost of a gen9blas call from Python | about 145 us wall, about 85 us CPU (no queue profiling) | [api.md](api.md) |
| Fixed cost of a libgen9blas call from C (mains power, CPU 3.7 GHz) [MEAS] | axpy n = 4096: 48 us wall / 14-17 us CPU (execbuf 8, GEM_WAIT 4 CPU / 35 wall, command building 2.4, userptr 0.7). On battery (CPU 900 MHz): 80-96 / 43-57 us. Large calls 120-145 us CPU (core waking from C6-C10; with the core held in C0, half) | `results/overhead-2026-11-12.txt` |
| One small kernel, submit + wait, C | NEO + `clFinish` 34 us / 41-43 us CPU; NEO + sleeping wait 90 us / 29 us CPU; **without NEO (i915 execbuffer2 + GEM_WAIT) 36 us / 7 us CPU** (different conditions in a later run: 57 us / 22-28 us CPU) | [no-neo.md](no-neo.md); `results/overhead-2026-11-12.txt` (bare i915 line) |
| 8 dependent kernels in one operation | without NEO (one buffer + barriers) N = 4096: 72 us / 10 us CPU; NEO + sleeping wait 149 us / 103 us CPU (about 12 us CPU per enqueue) [MEAS] | [no-neo.md](no-neo.md) |
| Consecutive GPGPU_WALKERs without a barrier | overlap (race observed 30/50); a PIPE_CONTROL with CS stall + DC flush is needed; MEDIA_STATE_FLUSH is not enough [MEAS] | [no-neo.md](no-neo.md), `experiments/no-neo/chain.c` |
| Whole HGEMM call without NEO | n = 512: 0.70 ms (NEO 1.02); n = 1024: 3.8-4.2 ms (NEO 4.1-4.8); CPU 1.4% (NEO 9%) | [no-neo.md](no-neo.md); `results/no-neo-level3-2026-11-11.txt` for the level-3 comparison |
| Binding Table Pointer in IDD | bits 5..15: at most 64 KB from Surface State Base Address | genxml gen80 [SPEC], measurement [MEAS] ([no-neo.md](no-neo.md)) |
| EU thread arbitration (`0xE404`) | round-robin (0x100): SGEMM K = 8192 362 GFLOPS; hardware default 211-220; K = 4096 no difference [MEAS] | [no-neo.md](no-neo.md); `KCL_ARB=age` in `lib/kcl.c` selects the default |
| SLM encoding in IDD (Gen9.5) | log2(pow2(max(slm, 1K))) - 9: 1K = 1 ... 64K = 7 (differs from Gen8: genxml gen80.xml 4K = 1 ...; gen90.xml matches) [SPEC] | PRM KBL Vol. 2d; NEO 24.35.30872.36 |
| RAW (stateful) buffer bounds | checked every 4 B: round the surface size up to 4 [MEAS] | [no-neo.md](no-neo.md); `lib/kcl.c` |
| L3CNTLREG (`0x7034`) from an unprivileged batch buffer | `MI_LOAD_REGISTER_IMM` works (read back 0x60000321) [MEAS] | [no-neo.md](no-neo.md); `lib/kcl.c` |
| MOCS (stateless/surface) | index 0 = uncached (2x slower), 2 best [MEAS] | [no-neo.md](no-neo.md); `KCL_MOCS` in `lib/kcl.c` |
| 2 MB pages in PPGTT | unavailable: shmem THP = never [MEAS] | [no-neo.md](no-neo.md) |
| Queue profiling | about 20 us CPU per event (hence `GEN9BLAS_PROFILE` only on request) | `lib/gen9blas.c` (comment at the profiling code) |
| `clWaitForEvents` / `clFinish` | spin actively (100% of a core); `wait_gpu()` in `lib/gen9blas.c` sleeps with `nanosleep` (`GEN9BLAS_WAIT=busy` restores `clFinish`) | `lib/gen9blas.c`, `bench/hgemm.c` |
| Host USM allocation | about 0.5 ms CPU per MB (hence the pool behind `gen9blas.empty`) | `lib/gen9blas.py` (comment at `_pool`) |
| Host memory for the GPU | host USM only; `USE_HOST_PTR` on ordinary memory -> random NEO crashes | [pitfalls.md](pitfalls.md) |
| `ocloc` compilation | `-device whl` (`cfl` gives CL_INVALID_BINARY); remove the `.spv` from the binary | `lib/gen_kernels.py`, [pitfalls.md](pitfalls.md) |
| Compiler output for simple packing kernels | **SIMD8, byte scattered messages** (read and write), also for `vload8`/`vstore16` -- **no** block read/write. `ocloc disasm` of `bench/packing.cl`: `pk_bq` write 16 B, `pk_bq_w2` 32 B, `pk_a2_q` 48 scattered reads + 16 scattered writes per thread | [pitfalls.md](pitfalls.md); `bench/packing.cl` |
| Access width per lane (SIMD16) [MEAS] | 16 B per lane: a message covers whole 64 B lines, the optimum. 32 B (4x FP64) splits lines between two messages -- a write without a prior read loses about 1.4x (FP64 COPY 17 -> 24 GB/s after the change); 8 B (4x FP16): smaller messages; 8x FP16 (16 B) the compiler splits into SIMD8 byte reads of 4 B (worse) | `results/message-16B-2026-11-12.txt` |
| Register spill (`spill_size` in `.ze_info`) [MEAS] | IGC spills registers to scratch when GRF runs out (skinny GEMM: 32 FP32 accumulators per lane in SIMD16 + 4 weight vectors -> 448-960 B); kcl does not configure scratch -> **silently wrong results**; `lib/gen_kernels.py` rejects such kernels | `results/skinny-gemm-2026-11-12.txt` |
| 64-bit address arithmetic in a hot loop [MEAS] | 64-bit multiply is emulated on Gen9: `(long)i * stride` in the inner loop cost 1.2-1.9x of kernel time (skinny GEMM); compute pointers before the loop | `results/skinny-gemm-2026-11-12.txt` |
| Skinny GEMM (no packing, FP32 accumulation) [MEAS] | M = 1: weights read at 18-27 GB/s (memory bound); compute 70-180 GFLOPS (FP16 A*B up to 177, A*W^T about 90, FP32 60-90) -- limited by A traffic from L3 (about M/RW) and the register limit | `results/skinny-gemm-2026-11-12.txt`; [skinny-gemm.md](skinny-gemm.md) |
| Private array > about 128 B per lane (IGC 24.35) [SW] | `float s[8][8]` in SIMD8 (256 B/lane) goes to private memory (`private_size`, scratch buffer) despite full unrolling; `s[8][4]` (128 B) stays in registers; separate vector variables always stay in registers. kcl does not configure scratch -> **silently wrong results** (correct through NEO). `private_size` may be declared without any access (gemv_n: 256 B) | `results/skinny-gemm-rework-2026-11-12.txt` |
| Mixed-mode hf -> f in `mad` (SIMD8) [SW] | the half conversion folds into `mad` (source `:hf`), no separate `mov`; in SIMD16 mixed mode is split into 2 x SIMD8 | `results/skinny-gemm-rework-2026-11-12.txt` |
| Raw `ushort4` per lane [SW] | IGC keeps 1 GRF per component (4 GRF for 8 B of data in SIMD8); `uint2` = 2 GRF | `results/skinny-gemm-rework-2026-11-12.txt` |
| Weight reads as row streams (`skinny_r`, 8 rows per group, RAM) [MEAS] | bandwidth depends on the contiguous bytes per row per group step: 256 B -> 14-18 GB/s, 512 B -> 19-20, 1 KB -> 21-24; 44 KB rows (FP32 K = 11008) -> 16-17 GB/s ([HYP]: pages/TLB) | `results/skinny-gemm-rework-2026-11-12.txt` |
| Skinny GEMM `skinny_r` (SIMD8, 8 weight rows, 16 subgroups, FP16 4096^2) [MEAS] | M = 2-4 1.55-1.68 ms; M = 8 1.9; M = 16 3.2 (168-225 GFLOPS); M = 8-16 closer to t_mem + t_comp than to the maximum ([HYP]: no overlap within a thread) | `results/skinny-gemm-rework-2026-11-12.txt` |
| Compiler and FP16 `mad()` | without `-cl-fast-relaxed-math` it splits into `mul` + `add` (200 instead of 791 GFLOPS) | `results/arith-throughput-latency-2026-11-11.txt` (`half k_mad` vs `k_fma`), [pitfalls.md](pitfalls.md) |
| Compiler and GRF banks in GEMM | 128-160 of 256 `mad` have a bank conflict (hence the hand-written assembly) | [pitfalls.md](pitfalls.md); `results/micro-2026-11-11.txt`; `asm/` |

Instruction latencies according to the Mesa scheduler heuristics
(`third_party/mesa-25.2.8/brw_schedule_instructions.cpp`, Mesa 25.2.8) [EST]: byte/dword scattered 300 clocks,
untyped surface 200-600, atomics 14 000. These are the compiler authors' assumptions, not measurements.

## 4. Dynamic quantities (control in every measurement)

| Quantity | Range | How to control |
|---|---|---|
| GPU clock | 300-1100 MHz; RPS raises it at 95% and lowers it at 85% busy [SPEC: i915 debugfs] | warm up before measuring; watch `intel_gpu_top`. Frequency limits are left at the driver defaults |
| CPU power limit | PL2 25-42 W for about 30-45 s, then PL1 about 17 W [MEAS] | the GPU fits in PL1 (16-18 W), the CPU does not: OpenBLAS SGEMM 297 -> 189 GFLOPS once the budget is used up (first value: `results/compare-l3-power-2026-11-11.txt`; sustained: [history.md](history.md); see also `results/long-run-2026-11-11.txt`) |
| Shared resources | RAM and LLC are shared with the CPU; the desktop also uses the GPU (compositor, screen lock, screen casting) | spread about +-10%: compare alternately, several rounds, use the median (`bench/compare/l3.sh`, see [benchmarks.md](benchmarks.md)) |
| Warm-up | first calls: compilation, allocations, low clock | skip the first call; short "best of 5" series can mislead (sleeping wait: apparent 0.8 ms) |
| OpenBLAS threads in the process | spin in the background (about 550% CPU) after the library is loaded; shared-memory routines (axpy, gemv, ger, symv, trmv, syr) use 6-8x wall time | `OPENBLAS_NUM_THREADS=1` for GPU measurements, set BEFORE importing NumPy |
| Power source | battery (EPP balance_power): CPU about 900 MHz while driving the GPU, CPU costs 2-3x higher | check `/sys/class/power_supply/A*/online`; compare under the same conditions |

## 5. Gaps: to be measured by a microbenchmark (do not base predictions on them yet)

| Quantity | Why needed | State |
|---|---|---|
| Range and organization of the GPU TLB | collapses when many pages are touched at once (`pack_b_k16`: 16 rows -> 7x slower; CLBlast at n >= 2048) | [HYP] TLB; the PRM gives no entry count |
| Write bandwidth per message type (byte scattered 2/4 B, dword scattered, OWord block write, media block write) | packing and epilogues are write-limited | partial (`results/message-16B-2026-11-12.txt`): untyped xyzw 16 B/lane stays at about 24 GB/s write+read; half-line writes about 17; byte 4 B about 16 -- block writes not measured |
| Scattered read bandwidth (2/4 B per line) vs block | as above | not measured |
| GPU gap between dependent kernels in a queue | calls with several kernels (pack A, pack B, GEMM, epilogue) | not measured; whole HGEMM call at n = 1024: about 0.4 ms outside kernel times |
| Thread dispatch cost vs grid size | small kernels | not measured |

## 6. Formulas

- **Memory-bound kernel:** t >= (bytes read + written) / bandwidth, where bandwidth depends on the working set
  (L3 / LLC / RAM, section 2) and on the message type (section 5, gap).
  Example: packing A in FP16 at n = 1024 moves 2 + 2 MB, so t >= 0.17 ms at 24 GB/s. It was 0.54 ms and is now 0.31 ms ([history.md](history.md)).
- **Compute-bound kernel:** t >= FLOP / measured peak (section 2), provided there are no bank conflicts and the loop fits in the GRF.
- **Library call:** t = CPU submit cost (section 3) + sum of kernel times + gaps between kernels (gap) + wake-up delay
  of the sleeping wait (depends on the polling interval).
- **Floating-point summation error:** summing K terms sequentially gives a relative error of order sqrt(K)*u on average and
  K*u in the worst case (u = 2^-24, about 6e-8 for FP32; 2^-11, about 4.9e-4 for FP16). With chunks of KC and a sum of
  chunks: of order (sqrt(KC) + sqrt(K/KC))*u. To be checked against measurements (see [roadmap.md](roadmap.md)).
