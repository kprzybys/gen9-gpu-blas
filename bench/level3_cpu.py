"""Level 3 of libgen9blas: time of the whole call and process CPU time (all threads) per call.

Run: python3 bench/level3_cpu.py [n ...]   (GEN9BLAS_LIB=path: another library version; GEN9BLAS_WAIT=busy)
CPU% = CPU time / wall time: how much of a core the call occupies (goal: close to 0 -- the CPU only submits and sleeps).
gen9blas.empty arrays (zero-copy). Best wall time of 5 repetitions, CPU time -- mean.
"""
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # OpenBLAS (NumPy) threads spin in the background and inflate the process CPU
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "lib"))
import gen9blas  # noqa: E402

rng = np.random.default_rng(1)


def make_array(n, dtype, diag=0.0):
    X = gen9blas.empty((n, n), dtype)
    X[:] = rng.uniform(-1, 1, (n, n))
    if diag:
        X[np.diag_indices(n)] = diag
    return X


for n in [int(a) for a in sys.argv[1:]] or [2048, 4096]:
    for dtype in (np.float32, np.float16):
        A, B, C = make_array(n, dtype, n), make_array(n, dtype), make_array(n, dtype)
        print(f"== n = {n}, {np.dtype(dtype).name}")
        for name, f, fl in [("gemm", lambda: gen9blas.gemm(A, B, C=C), 2 * n ** 3), ("symm", lambda: gen9blas.symm(A, B, C=C), 2 * n ** 3),
                             ("syrk", lambda: gen9blas.syrk(A, C=C), n ** 3), ("syr2k", lambda: gen9blas.syr2k(A, B, C=C), 2 * n ** 3),
                             ("trmm", lambda: gen9blas.trmm(A, B), n ** 3), ("trsm", lambda: gen9blas.trsm(A, B), n ** 3)]:
            f()
            wall, cpu_t = [], []
            for _ in range(5):
                t, c = time.perf_counter(), time.process_time()
                f()
                wall.append(time.perf_counter() - t); cpu_t.append(time.process_time() - c)
            t, c = min(wall), sum(cpu_t) / len(cpu_t)
            print(f"  {name:6s} {fl / t / 1e9:6.1f} GFLOPS  {t * 1e3:6.0f} ms   CPU {c * 1e3:6.2f} ms ({100 * c / np.mean(wall):4.1f}%)")
