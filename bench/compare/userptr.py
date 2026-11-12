"""Pomiar userptr (zwykłe tablice NumPy bez kopii) wobec kopii przez CPU (GEN9BLAS_NO_USERPTR=1).

Uruchom: bench/compare/userptr.sh (oba tryby na przemian, osobne procesy). Wypisuje dla każdego przypadku medianę
czasu wywołania i CPU procesu na wywołanie (time.process_time; OPENBLAS_NUM_THREADS=1, żeby wątki OpenBLAS nie
zawyżały CPU). „te same”: te same tablice w kolejnych wywołaniach; „nowe”: nowe tablice przed każdym wywołaniem
(czas samego wywołania gen9blas; pierwsze użycie pamięci = przypięcie stron).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../lib"))
import gen9blas  # noqa: E402

rng = np.random.default_rng(1)


def mierz(f, przygotuj=None, R=15):
    t, c = [], []
    args = przygotuj() if przygotuj else ()
    f(*args)
    for _ in range(R):
        if przygotuj:
            args = przygotuj()
        c0, t0 = time.process_time(), time.perf_counter()
        f(*args)
        t.append(time.perf_counter() - t0); c.append(time.process_time() - c0)
    return np.median(t) * 1e3, np.median(c) * 1e3


def los(*shape, typ=np.float32):
    return rng.uniform(-1, 1, shape).astype(typ)


P = {}
for n in (256, 1024, 2048, 4096):
    A, B = los(n, n), los(n, n)
    P[f"sgemm n={n} te same"] = (lambda A=A, B=B: gen9blas.sgemm(A, B), None)
A, B = los(2048, 2048, typ=np.float16), los(2048, 2048, typ=np.float16)
P["hgemm n=2048 te same"] = (lambda: gen9blas.hgemm(A, B), None)
P["sgemm n=1024 nowe"] = (lambda A, B: gen9blas.sgemm(A, B), lambda: (los(1024, 1024), los(1024, 1024)))
P["sgemm n=4096 nowe"] = (lambda A, B: gen9blas.sgemm(A, B), lambda: (los(4096, 4096), los(4096, 4096)))
for n in (10 ** 6, 10 ** 7):
    x, y = los(n), los(n)
    P[f"axpy n={n:.0e} te same"] = (lambda x=x, y=y: gen9blas.axpy(1e-3, x, y), None)
    P[f"dot n={n:.0e} te same"] = (lambda x=x, y=y: gen9blas.dot(x, y), None)
P["axpy n=1e+07 nowe"] = (lambda x, y: gen9blas.axpy(1e-3, x, y), lambda: (los(10 ** 7), los(10 ** 7)))
M, v, w = los(4096, 4096), los(4096), los(4096)
P["gemv 4096 te same"] = (lambda: gen9blas.gemv(M, v, y=w), None)

for nazwa, (f, prz) in P.items():
    t, c = mierz(f, prz)
    print(f"{nazwa:24s} {t:9.3f} ms  CPU {c:8.3f} ms", flush=True)
