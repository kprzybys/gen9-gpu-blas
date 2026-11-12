# Benchmarks: method and reproduction

How the numbers in `results/` were obtained, what can disturb them, and the command that regenerates each file.
Overview of the results: [README.md](../README.md). Model that explains them: [performance-model.md](performance-model.md).

All measurements are on one machine (UHD 620 WHL GT2, Core i5-8265U, Linux 7.0 with `i915`); see the table in
[README.md](../README.md). Result files are named `<experiment>-<date>.txt`. The scripts that `tee` their output
write a file with the current date into `results/`; the others print to stdout and are redirected by hand. Numbers you
get will differ by the factors below, so compare orders of magnitude and ratios, not the third digit.

## 1. Conditions that change the result

These are the dynamic quantities of the performance model ([performance-model.md](performance-model.md), "dynamic
quantities"). Control or record each one for every measurement.

| Factor | Range | How to control or check |
|---|---|---|
| Power source | On battery (EPP `balance_power`) the CPU runs at about 900 MHz while the GPU works and CPU costs are 2–3x higher; on mains power (EPP `balance_performance`) the CPU runs at about 3.7 GHz | `cat /sys/class/power_supply/A*/online` (1 = mains). Measure on mains power and say so; several scripts print `power AC=<0/1>` into the result header. Do not compare numbers taken in different states. `results/auto-thresholds-2026-11-12.txt` contains both and marks the battery part as invalid for decisions |
| GPU clock | 300–1100 MHz; the frequency governor (RPS) raises the clock above 95% busy and lowers it below 85% [SPEC] | Warm up before measuring (first calls: compilation, allocation, low clock); `cat /sys/class/drm/card1/gt_max_freq_mhz`; watch the actual clock with `intel_gpu_top`. `bench/power.sh` and `bench/long_run.sh` record it |
| Processor power limit | PL2 25–42 W for about 30–45 s, then PL1 about 17 W [MEAS]. The GPU alone stays within PL1 (16–18 W); CPU-bound competitors do not: OpenBLAS SGEMM falls from 297 to 189 GFLOPS once the budget is used up | Long runs (`bench/long_run.sh`, 90 s CPU run); alternate GPU and CPU runs and let the machine cool down between them (the script waits 20 s) |
| Shared resources | RAM and last-level cache are shared with the CPU; the desktop compositor and other GPU users also run on this GPU | Variation of about ±10% between runs. Compare variants alternately (A B A B ...), in separate processes, take several rounds and report the median (range in brackets where it matters). A single run can suggest the opposite of the median |
| OpenBLAS threads | After the library is loaded its threads spin in the background (about 550% CPU). Functions that OpenBLAS splits across threads (axpy, gemv, ger, symv, trmv, syr) then use 6–8x the wall time as CPU time | `OPENBLAS_NUM_THREADS=1` when measuring the GPU path, set before NumPy is imported (the Python benchmarks do `os.environ.setdefault(...)` themselves). Use the default thread count only for the CPU baseline |
| Warm-up and wait mode | The first call includes kernel setup; short "best of 5" series can mislead when waiting is by sleep | Skip the first call (result lines print it separately as `1st call`). `GEN9BLAS_WAIT=busy` selects busy-waiting, the default sleeps |
| Memory kind | USM (`gen9blas_alloc` / `gen9blas.empty`) vs plain process memory (pinned through `userptr`, about 0.45 µs CPU per 4 KB page on first use) | State which one a result uses. Most results use USM (`gpu` mode); `MODE=host` of `bench/compare/l3` uses plain memory |

Statistics used: GEMM programs report the best of 5 calls (`hgemm`, `sgemm`, `l3` without a duration), or the mean over a
fixed duration when a duration argument is given (power runs); the Python scripts report the median of several rounds
or the best of 5 as stated in each file header. Error columns are computed on 256 random result elements against a
double-precision computation on the same rounded inputs (`err avg=... max=...`).

## 2. Power and clock

`bench/power.sh <command...>` runs a command (it should run at least 4 s) with `intel_gpu_top -J -s 500` alongside
and prints the median GPU clock, Render engine busy, GPU power and package power, dropping the first two samples and
the last. `intel_gpu_top` may need extra privileges (perf access) on your system. Energy per call = package power x call
time. `bench/level3_power.py` and `bench/compare/l3_power.sh` wrap it for level-3 operations.

## 3. Reproducing each result file

Build prerequisites: `make -C lib libgen9blas.so libgen9blas_blas.so` (and `make -C lib libgen9blas_neo.so` for NEO
comparisons); the measurement programs are built with `make -C bench` (`hgemm`, `sgemm`, `clbench`, `packing`, `l12`, `overhead`)
and `make -C bench optional` (`gemm_clblast`, needs CLBlast and OpenBLAS); library comparisons: `make -C bench/compare`. Run scripts from the repository root unless noted; the shell scripts
`cd` to their own directory. Comparison binaries need CLBlast (`libclblast`), OpenBLAS and an OpenCL ICD (NEO).

| Result file (`results/`) | What it contains | How to reproduce |
|---|---|---|
| `arith-throughput-latency-2026-11-11.txt` | Throughput and latency of arithmetic instructions (float, half, double, integer) | `sh bench/arith_throughput_latency.sh` (builds `bench/clbench`, kernels in `bench/peak.cl`) |
| `micro-2026-11-11.txt` | Assembly microbenchmarks: register-bank conflicts, broadcast operands | `sh asm/micro.sh` (generates binaries with `asm/micro.py`, runs `bench/clbench`) |
| `memory-bandwidth-2026-11-11.txt`, `memory-latency-2026-11-11.txt` | Block-read bandwidth (`k_owb`, `k_owb2`, `k_media`) and dependent-read latency (`k_lat`) versus buffer size | `bench/clbench` with the kernels of `bench/memory.cl` (buffer size is the compile-time `MASK`). No script sweeps the sizes: the table was assembled by hand from repeated runs |
| `gemm0-2026-11-11.txt` | Reference points: CLBlast (S/H/D) and OpenBLAS (f32/f64) for N = 128..4096 | `sh bench/gemm0.sh` (needs CLBlast; `bench/gemm_clblast.c`, `bench/gemm_cpu.py`) |
| `hgemm-compare-2026-11-11.txt` | HGEMM assembly vs OpenCL C vs CLBlast vs OpenBLAS, with and without packing, power | `sh bench/hgemm_compare.sh` (needs `bench/hgemm` and `bench/gemm_clblast` built, and `bench/hgemm_asm.bin`; the script runs `make -C asm` first) |
| `long-run-2026-11-11.txt` | 4000 HGEMM runs and 90 s of OpenBLAS, clock and power every 5 s | `sh bench/long_run.sh` (needs `intel_gpu_top`) |
| `compare-l3-2026-11-11.txt` | Level 3 (gemm, symm, syrk, syr2k, trmm, trsm; S/H/D): gen9blas vs CLBlast vs OpenBLAS; sleep-wait and host-memory variants | `make -C bench/compare l3` after `make -C lib`; then `sh bench/compare/l3.sh > results/compare-l3-<date>.txt` |
| `compare-l3-power-2026-11-11.txt` | Clock, power and energy of the level-3 cases above | `sh bench/compare/l3_power.sh` from the repository root |
| `compare-onednn-2026-11-11.txt` | oneDNN v3.6 (GPU) vs gen9blas, GEMM n = 1024..4096, FP32/FP16 | `sh bench/compare/onednn.sh > results/...`. oneDNN is built outside the repository (`cmake -DDNNL_CPU_RUNTIME=NONE -DDNNL_GPU_RUNTIME=OCL`, target `benchdnn`); point `BENCHDNN=` at it |
| `compare-openvino-2026-11-11.txt` | OpenVINO 2026.1 (GPU) vs gen9blas, plus energy | `sh bench/compare/ov.sh > results/...` (Python package `openvino`; `bench/compare/ov_gemm.py`) |
| `compare-ggml-vulkan-2026-11-11.txt` | ggml (llama.cpp 2cbe0ec) Vulkan backend on Mesa ANV | `test-backend-ops perf -o MUL_MAT -b Vulkan0` from llama.cpp, with a small local patch that adds square n x n cases (enabled by an environment variable). The patch is not part of this repository: the file records its result only |
| `viennacl-2026-11-12.txt` | ViennaCL 1.7.1 dense_blas test (OpenBenchmarking profile `pts/viennacl`) repeated on gen9blas, before and after the vector-width fixes | gen9blas side: `python3 bench/compare/viennacl_gen9blas.py [series]`. ViennaCL side: the `dense_blas` example of ViennaCL 1.7.1 built against the system OpenCL. The 2021 reference column is the public OpenBenchmarking result 2103278-IB-KABYLAKET61 (not redistributed) |
| `viennacl-skinny-2026-11-12.txt` | Skinny FP32 GEMM and GEMV, ViennaCL vs gen9blas | ViennaCL: `bench/compare/viennacl_skinny.cpp` (build line in its header). gen9blas: `python3 bench/skinny_gemm.py usm`. Runs alternate, median of 2 |
| `skinny-gemm-2026-11-12.txt` | Skinny GEMM versions, prediction and measurement, old path vs new | `python3 bench/skinny_gemm.py [usm\|plain] [label]`; `GEN9BLAS_SKINNY=0` selects the old path |
| `skinny-gemm-rework-2026-11-12.txt` | Rework of the skinny kernels (`skinny_r`), diagnosis from assembly, stage comparison | `python3 bench/skinny_gemm.py usm`; old path of the rework: `GEN9BLAS_SKINNY_R=0`; 3 alternating runs, medians. Assembly dumps come from `lib/.kernels/` after `make -C lib` |
| `gemv-n-2026-11-12.txt` | GEMV N, old vs grouped kernel, USM vs plain memory | `python3 bench/gemv_n.py usm` and `... plain` (the script alternates `GEN9BLAS_GEMV_N=old\|grouped` in separate processes, 3 rounds) |
| `message-16B-2026-11-12.txt` | Access width per lane vs bandwidth (copy and GEMV kernels) | `python3 bench/compare/viennacl_gen9blas.py` for the table; kernels in `bench/level12.cl`; ISA by `ocloc disasm` (see `asm/disassemble.sh`) |
| `overhead-2026-11-12.txt` | Fixed call cost, phase breakdown (execbuf, wait, command building, userptr) | `make -C bench overhead`; `KCL_PHASES=1 bench/overhead [n] [repetitions] [usm]` on mains power, then once on battery. The bare-i915 reference line comes from `experiments/no-neo/first` |
| `userptr-2026-11-12.txt` | userptr vs copy through the CPU | `sh bench/compare/userptr.sh` (3 alternating rounds; copy mode uses `GEN9BLAS_NO_USERPTR=1`); microbenchmark: `make -C experiments/no-neo userptr && experiments/no-neo/userptr` |
| `auto-thresholds-2026-11-12.txt` | Sizes at which the BLAS layer's auto mode sends level 1–2 calls to the GPU, CPU time per call | `python3 bench/auto_thresholds.py [rounds] [s\|d]` (needs `lib/libgen9blas_blas.so`; modes run alternately in separate processes). The file's battery section is superseded by its "MAINS POWER" section |
| `numpy-preload-2026-11-12.txt` | NumPy / SciPy with and without `LD_PRELOAD=libgen9blas_blas.so` (wall time, CPU %), routing check | `python3 bench/numpy_preload.py 4096 3`; `python3 bench/numpy_preload.py check 4096` |
| `level3-whole-call-2026-11-12.txt`, `level3-fp64-2026-11-12.txt` | Level 3 whole call n = 4096 (FP32, FP16) and FP64 at n = 2048 with OpenBLAS DGEMM | `python3 bench/level3_cpu.py 4096`; `bench/compare/l3 gen9blas <op> d 2048`, `bench/compare/l3 openblas gemm d 2048` |
| `level12-2026-11-12.txt` | Levels 1-2: GB/s and CPU %, FP32 / FP16 GPU, NumPy on the CPU | `python3 bench/l12_gen9blas.py float32` and `float16` |
| `batched-gemm-2026-11-12.txt`, `gemm-sizes-2026-11-12.txt` | Batched GEMM; whole-call GEMM FP16 / FP16 accurate / FP32 versus size | `python3 bench/batched_gemm.py`; `python3 bench/gemm_sizes.py` |
| `call-overhead-python-2026-11-12.txt` | Fixed cost of a call from Python (wrapper vs library) | `python3 bench/call_overhead.py 4096 2000 5` |
| `no-neo-level3-2026-11-11.txt` | Level 3 without NEO (`kcl`) vs the NEO variant, with CPU load | `OPENBLAS_NUM_THREADS=1 python3 bench/level3_cpu.py 1024 2048 4096` once with the default library and once with `GEN9BLAS_LIB=lib/libgen9blas_neo.so` (build with `make -C lib libgen9blas_neo.so`) |
| `skylake-hd530-2026-11-12.txt` | Second Gen9 part (HD 530, Skylake GT2) with the whl kernels: `make -C lib check`, GEMM vs size, level 3 | `lib/kernels.c` copied from a machine with `ocloc`; `make -C lib check`, `python3 bench/gemm_sizes.py`, `python3 bench/level3_cpu.py` (nice 19, `OPENBLAS_NUM_THREADS=1`) |
| `userptr-overhead-hd530-2026-11-12.txt` | HD 530 (GPU without IOMMU): fixed call cost, userptr vs copy, page pinning by phase | `KCL_PHASES=1 bench/overhead 4096 5000 [usm]`; `sh bench/compare/userptr.sh`; `experiments/no-neo/userptr` and `PHASES=1 experiments/no-neo/userptr 64` |
| `level12-hd530-2026-11-12.txt`, `level3-extra-hd530-2026-11-12.txt` | HD 530: levels 1-2; batched and skinny GEMM, DGEMM, 10-minute HGEMM run | `python3 bench/l12_gen9blas.py float32\|float16`; `bench/batched_gemm.py`; `bench/skinny_gemm.py usm`; DGEMM and the 10-minute loop: short one-off scripts described in the file headers |

Other measurement programs in `bench/`: `hgemm.c` / `sgemm.c` (single GEMM kernels; `hgemm` reads the environment variables `PACK`, `ND1`, `LY`, `REPS`,
`TRACE`), `packing.c` (packing kernels), `l12.c`, `l12_gen9blas.py`, `l12_cpu.py`, `level12_speed.py` (levels 1–2 bandwidth),
`level3_power.py` and `level3_cpu.py` (level 3, CPU time per call).

## 4. Checklist for a new measurement

1. Mains power, `cat /sys/class/power_supply/A*/online` prints 1; note the kernel version.
2. Close other GPU users where possible; warm up (skip the first call).
3. `OPENBLAS_NUM_THREADS=1` before NumPy for GPU-path measurements; CPU baselines with their own thread count stated.
4. Alternate the variants being compared, run separate processes, at least 3 rounds, report the median and the range.
5. For long GPU runs, record clock and power (`bench/power.sh`); for CPU baselines watch the PL2 to PL1 drop.
6. Write the conditions (date, power source, memory kind, method) at the top of the result file and tag every
   statement with its evidence class ([MEAS], [SW], [EST], [HYP]).
