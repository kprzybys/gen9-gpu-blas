"""NumPy / SciPy on the CPU (OpenBLAS) vs the same program with LD_PRELOAD=lib/libgen9blas_blas.so.

Run: python3 bench/numpy_preload.py [n] [rounds]      (default n = 4096, 3 rounds; float64 uses min(n, 2048), see below)
     python3 bench/numpy_preload.py check [n]         (GEN9BLAS_BLAS_INFO=1: where does each call go?)
The parent starts one child process per mode and round, alternating the modes (A B A B ...):
  cpu     plain NumPy, default OpenBLAS threads (the CPU baseline)
  preload LD_PRELOAD=lib/libgen9blas_blas.so, OPENBLAS_NUM_THREADS=1 (set before NumPy is imported; the GPU path
          needs no OpenBLAS threads, and they would spin and inflate the process CPU time)
float64 is measured at n = 2048: a single DGEMM dispatch of n = 4096 (2.4 s) exceeds the i915 preempt_timeout_ms (640 ms)
and resets the GPU on this machine (2026-11-12), see results/numpy-preload-2026-11-12.txt.
Cases: A @ B, A.T @ A, scipy.linalg.blas.s/dgemm, A @ x, for float32 and float64; ordinary NumPy arrays (userptr, no copy).
Per case: best wall time of 5 calls, CPU % of one core = process CPU time (all threads) / wall time, mean over the calls.
The parent prints the median over the rounds (range in brackets).
"""
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
LIB = os.path.join(HERE, "..", "lib", "libgen9blas_blas.so")


def child(n, reps, n64):
    import time
    import numpy as np
    import scipy.linalg.blas as sb
    rng = np.random.default_rng(1)
    out = {}
    n0 = n
    for dt, gemm in ((np.float32, sb.sgemm), (np.float64, sb.dgemm)):
        tag = np.dtype(dt).name
        n = n64 if dt is np.float64 else n0
        A = rng.uniform(-1, 1, (n, n)).astype(dt)
        B = rng.uniform(-1, 1, (n, n)).astype(dt)
        x = rng.uniform(-1, 1, n).astype(dt)
        cases = [("A @ B", lambda: A @ B), ("A.T @ A", lambda: A.T @ A),
                 ("scipy gemm", lambda: gemm(1.0, A, B)), ("A @ x", lambda: A @ x)]
        for name, f in cases:
            f()
            wall, cpu = [], []
            for _ in range(reps):
                t, c = time.perf_counter(), time.process_time()
                f()
                wall.append(time.perf_counter() - t)
                cpu.append(time.process_time() - c)
            out[f"{tag}|{name}"] = (min(wall), 100 * sum(cpu) / sum(wall))
    print(json.dumps(out))


def run(mode, n, reps, n64, info=False):
    env = dict(os.environ)
    env.pop("OPENBLAS_NUM_THREADS", None)
    if mode == "preload":
        env["LD_PRELOAD"] = os.path.realpath(LIB)
        env["OPENBLAS_NUM_THREADS"] = "1"
    if info:
        env["GEN9BLAS_BLAS_INFO"] = "1"
    p = subprocess.run([sys.executable, os.path.realpath(__file__), "child", str(n), str(reps), str(n64)],
                       env=env, capture_output=True, text=True, check=True)
    return json.loads(p.stdout.strip().splitlines()[-1]), p.stderr


def main():
    if sys.argv[1:2] == ["child"]:
        return child(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    if sys.argv[1:2] == ["check"]:
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
        _, err = run("preload", n, 1, min(n, 2048), info=True)
        counts = {}
        for line in err.splitlines():
            if line.startswith("gen9blas_blas:"):
                key = line.split(":", 1)[1].strip()
                counts[key] = counts.get(key, 0) + 1
        for k, v in sorted(counts.items()):
            print(f"  {v:3d} x {k}")
        return
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    res = {"cpu": [], "preload": []}
    for r in range(rounds):
        for mode in ("cpu", "preload"):
            res[mode].append(run(mode, n, 5, min(n, 2048))[0])
            print(f"  round {r + 1} {mode} done", file=sys.stderr)

    def fmt(mode, key):
        w = [d[key][0] * 1e3 for d in res[mode]]
        c = [d[key][1] for d in res[mode]]
        d = 2 if max(w) < 20 else 0
        return f"{statistics.median(w):7.{d}f} ms [{min(w):.{d}f}-{max(w):.{d}f}], CPU {statistics.median(c):4.0f}% [{min(c):.0f}-{max(c):.0f}]"
    print(f"float32 n = {n}, float64 n = {min(n, 2048)}, median of {rounds} rounds (range in brackets), best-of-5 wall time per round")
    print(f"{'case':24s} {'OpenBLAS (cpu)':>42s}   {'LD_PRELOAD libgen9blas_blas':>42s}")
    for key in res["cpu"][0]:
        tag, name = key.split("|")
        print(f"{name + ' ' + tag:24s} {fmt('cpu', key):>42s}   {fmt('preload', key):>42s}")


main()
