"""Thresholds of the auto mode in libgen9blas_blas (levels 1–2 on plain memory via userptr): process CPU time and wall time
per call, GPU vs OpenBLAS (1 and 8 threads). Results and prediction: results/auto-thresholds-2026-11-12.txt.

Run: python3 bench/auto_thresholds.py [rounds] [type: s|d]   (the parent runs the modes alternating in separate processes)
Modes: gpu (GEN9BLAS_BLAS=gpu, OPENBLAS_NUM_THREADS=1), cpu1, cpu8 (GEN9BLAS_BLAS=cpu, OPENBLAS_NUM_THREADS=1 / 8).
Cases: "same": the same arrays in consecutive calls; "new": new NumPy arrays (np.empty + fill)
before each call. Call CPU = (process CPU in the loop − main thread CPU for preparing the data) / R --
includes the spinning of OpenBLAS threads after the call and the userptr page pinning.
"""
import ctypes as C
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
ROW, NOTRANS, TRANS, UPPER, NONUNIT = 101, 111, 112, 121, 131
L1 = [1 << k for k in range(10, 25, 2)]
L2 = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
NEW_MAX = 1 << 22   # "new" only up to 16 MB per array (beyond that the filling dominates anyway)


def child(mode, t):
    # The parent sets GEN9BLAS_BLAS and OPENBLAS_NUM_THREADS in the process environment: on import NumPy loads the system
    # OpenBLAS, which reads the thread count at once (changing os.environ here would be too late)
    lib = C.CDLL(os.path.join(HERE, "..", "lib", "libgen9blas_blas.so"))
    dtype, ct = (np.float32, C.c_float) if t == "s" else (np.float64, C.c_double)
    f = lambda name: getattr(lib, name.replace("?", t))  # noqa: E731
    for name, rt in [("cblas_?dot", ct), ("cblas_?nrm2", ct), ("cblas_?asum", ct), ("cblas_i?amax", C.c_size_t)]:
        f(name).restype = rt
    p = lambda a: C.c_void_p(a.ctypes.data)  # noqa: E731
    rng = np.random.default_rng(1)

    source_data = (rng.uniform(-1, 1, 1 << 20) * 1e-3).astype(dtype)

    def fresh(*shape):   # fill by copying (cheap and constant; random generation made the "new" measurement noisy)
        a = np.empty(shape, dtype)
        fl, S = a.reshape(-1), source_data.size
        for i in range(0, fl.size, S):
            fl[i:i + S] = source_data[:min(S, fl.size - i)]
        return a

    def tri(n):
        A = fresh(n, n)
        A[np.diag_indices(n)] = 2.0   # well-conditioned triangular (trsv)
        return A

    a = ct(0.5)
    cases = {
        "axpy": (L1, lambda n: (fresh(n), fresh(n)), lambda n, x, y: f("cblas_?axpy")(n, a, p(x), 1, p(y), 1)),
        "dot": (L1, lambda n: (fresh(n), fresh(n)), lambda n, x, y: f("cblas_?dot")(n, p(x), 1, p(y), 1)),
        "nrm2": (L1, lambda n: (fresh(n),), lambda n, x: f("cblas_?nrm2")(n, p(x), 1)),
        "scal": (L1, lambda n: (fresh(n),), lambda n, x: f("cblas_?scal")(n, ct(0.999), p(x), 1)),
        "copy": (L1, lambda n: (fresh(n), fresh(n)), lambda n, x, y: f("cblas_?copy")(n, p(x), 1, p(y), 1)),
        "iamax": (L1, lambda n: (fresh(n),), lambda n, x: f("cblas_i?amax")(n, p(x), 1)),
        "gemv N": (L2, lambda n: (fresh(n, n), fresh(n), fresh(n)),
                   lambda n, A, x, y: f("cblas_?gemv")(ROW, NOTRANS, n, n, a, p(A), n, p(x), 1, ct(0.0), p(y), 1)),
        "gemv T": (L2, lambda n: (fresh(n, n), fresh(n), fresh(n)),
                   lambda n, A, x, y: f("cblas_?gemv")(ROW, TRANS, n, n, a, p(A), n, p(x), 1, ct(0.0), p(y), 1)),
        "ger": (L2, lambda n: (fresh(n, n), fresh(n), fresh(n)),
                lambda n, A, x, y: f("cblas_?ger")(ROW, n, n, ct(1e-3), p(x), 1, p(y), 1, p(A), n)),
        "symv": (L2, lambda n: (fresh(n, n), fresh(n), fresh(n)),
                 lambda n, A, x, y: f("cblas_?symv")(ROW, UPPER, n, a, p(A), n, p(x), 1, ct(0.0), p(y), 1)),
        "trmv": (L2, lambda n: (tri(n), fresh(n)), lambda n, A, x: f("cblas_?trmv")(ROW, UPPER, NOTRANS, NONUNIT, n, p(A), n, p(x), 1)),
        "trsv": (L2, lambda n: (tri(n), fresh(n)), lambda n, A, x: f("cblas_?trsv")(ROW, UPPER, NOTRANS, NONUNIT, n, p(A), n, p(x), 1)),
        "syr": (L2, lambda n: (fresh(n, n), fresh(n)), lambda n, A, x: f("cblas_?syr")(ROW, UPPER, n, ct(1e-3), p(x), 1, p(A), n)),
    }
    for name, (sizes, prep, call) in cases.items():
        for n in sizes:
            el = n if sizes is L1 else n * n
            data = prep(n)
            for _ in range(2):
                call(n, *data)
            t0 = time.perf_counter(); call(n, *data); t1 = time.perf_counter() - t0
            R = int(min(2000, max(5, 0.08 / max(t1, 1e-6))))
            time.sleep(0.2)   # OpenBLAS threads stop spinning after the previous case (otherwise their CPU lands here)
            # same arrays: a loop with the call (a loop without the call costs nothing)
            c0, s0 = time.process_time(), time.perf_counter()
            for _ in range(R):
                call(n, *data)
            result = {"f": name, "n": n, "case": "same", "t": (time.perf_counter() - s0) / R * 1e6,
                     "c": (time.process_time() - c0) / R * 1e6}
            print(json.dumps(result), flush=True)
            if el > NEW_MAX:
                continue
            del data
            Rn = max(5, min(R, 200))
            # new arrays: process CPU in the loop minus main-thread CPU for preparing (and freeing) the data --
            # OpenBLAS threads (also those spinning after the call) count in full; wall time -- the call itself
            c0, ts, base = time.process_time(), 0.0, 0.0
            for _ in range(Rn):
                b0 = time.thread_time(); d = prep(n); base += time.thread_time() - b0
                s0 = time.perf_counter(); call(n, *d); ts += time.perf_counter() - s0
                b0 = time.thread_time(); del d; base += time.thread_time() - b0
            result = {"f": name, "n": n, "case": "new", "t": ts / Rn * 1e6, "c": (time.process_time() - c0 - base) / Rn * 1e6}
            print(json.dumps(result), flush=True)


def parent(rounds, t):
    modes = ["gpu", "cpu1", "cpu8"]
    w = {}
    for r in range(rounds):
        for mode in (modes if r % 2 == 0 else modes[::-1]):
            print(f"# round {r + 1}/{rounds}: {mode}", file=sys.stderr, flush=True)
            env = dict(os.environ, GEN9BLAS_BLAS="gpu" if mode == "gpu" else "cpu",
                       OPENBLAS_NUM_THREADS="8" if mode == "cpu8" else "1")
            out = subprocess.run([sys.executable, __file__, "--child", mode, t], capture_output=True, text=True,
                                check=True, env=env).stdout
            for line in out.splitlines():
                d = json.loads(line)
                w.setdefault((d["f"], d["n"], d["case"]), {}).setdefault(mode, []).append((d["t"], d["c"]))
    print(f"type {t}; wall time / process CPU per call [µs], medians of {rounds} rounds; "
          "GPU/cpu8 and GPU/cpu1: CPU ratio (< 1: the GPU offloads)")
    print(f"{'function':8s} {'n':>9s} {'case':5s} | {'gpu':>17s} | {'cpu1':>17s} | {'cpu8':>17s} | {'CPU g/8':>7s} {'g/1':>6s} {'wall g/8':>10s}")
    for (fn, n, case_name), m in w.items():
        med = {k: np.median(np.array(v), axis=0) for k, v in m.items()}
        g, c1, c8 = med["gpu"], med["cpu1"], med["cpu8"]
        k = lambda x: f"{x[0]:8.1f} /{x[1]:8.1f}"  # noqa: E731
        print(f"{fn:8s} {n:9d} {case_name:5s} | {k(g)} | {k(c1)} | {k(c8)} | {g[1] / max(c8[1], 1e-9):7.2f} "
              f"{g[1] / max(c1[1], 1e-9):6.2f} {g[0] / max(c8[0], 1e-9):10.2f}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3])
    else:
        parent(int(sys.argv[1]) if len(sys.argv) > 1 else 3, sys.argv[2] if len(sys.argv) > 2 else "s")
