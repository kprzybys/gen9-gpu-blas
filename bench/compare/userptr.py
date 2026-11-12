"""Measure userptr (plain NumPy arrays, zero-copy) against copying through the CPU (GEN9BLAS_NO_USERPTR=1).

Run: bench/compare/userptr.sh (both modes alternating, separate processes). For each case it prints the median
call time and process CPU per call (time.process_time; OPENBLAS_NUM_THREADS=1 so that OpenBLAS threads do not
inflate the CPU). "same": the same arrays in consecutive calls; "new": new arrays before each call
(time of the gen9blas call itself; first use of the memory = page pinning).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../lib"))
import gen9blas  # noqa: E402

rng = np.random.default_rng(1)


def measure(f, prepare=None, R=15):
    t, c = [], []
    args = prepare() if prepare else ()
    f(*args)
    for _ in range(R):
        if prepare:
            args = prepare()
        c0, t0 = time.process_time(), time.perf_counter()
        f(*args)
        t.append(time.perf_counter() - t0); c.append(time.process_time() - c0)
    return np.median(t) * 1e3, np.median(c) * 1e3


def rnd(*shape, dtype=np.float32):
    return rng.uniform(-1, 1, shape).astype(dtype)


P = {}
for n in (256, 1024, 2048, 4096):
    A, B = rnd(n, n), rnd(n, n)
    P[f"sgemm n={n} same"] = (lambda A=A, B=B: gen9blas.sgemm(A, B), None)
A, B = rnd(2048, 2048, dtype=np.float16), rnd(2048, 2048, dtype=np.float16)
P["hgemm n=2048 same"] = (lambda: gen9blas.hgemm(A, B), None)
P["sgemm n=1024 new"] = (lambda A, B: gen9blas.sgemm(A, B), lambda: (rnd(1024, 1024), rnd(1024, 1024)))
P["sgemm n=4096 new"] = (lambda A, B: gen9blas.sgemm(A, B), lambda: (rnd(4096, 4096), rnd(4096, 4096)))
for n in (10 ** 6, 10 ** 7):
    x, y = rnd(n), rnd(n)
    P[f"axpy n={n:.0e} same"] = (lambda x=x, y=y: gen9blas.axpy(1e-3, x, y), None)
    P[f"dot n={n:.0e} same"] = (lambda x=x, y=y: gen9blas.dot(x, y), None)
P["axpy n=1e+07 new"] = (lambda x, y: gen9blas.axpy(1e-3, x, y), lambda: (rnd(10 ** 7), rnd(10 ** 7)))
M, v, w = rnd(4096, 4096), rnd(4096), rnd(4096)
P["gemv 4096 same"] = (lambda: gen9blas.gemv(M, v, y=w), None)

for name, (f, prep) in P.items():
    t, c = measure(f, prep)
    print(f"{name:24s} {t:9.3f} ms  CPU {c:8.3f} ms", flush=True)
