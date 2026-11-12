"""Jedna operacja poziomu 3 libgen9blas w pętli przez ~6 s (do pomiaru mocy: bench/power.sh).

Uruchom: sh bench/power.sh python3 bench/level3_power.py <op> <n> <float32|float16>   (GEN9BLAS_LIB: inna wersja biblioteki)
Wypisuje czas wywołania i czas CPU na wywołanie; energia na wywołanie = moc Package (z power.sh) × czas wywołania.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "lib"))
import gen9blas  # noqa: E402

op, n, typ = sys.argv[1], int(sys.argv[2]), np.dtype(sys.argv[3]).type
rng = np.random.default_rng(1)


def tablica(diag=0.0):
    X = gen9blas.empty((n, n), typ)
    X[:] = rng.uniform(-1, 1, (n, n))
    if diag:
        X[np.diag_indices(n)] = diag
    return X


A, B, C = tablica(n), tablica(), tablica()
f = {"gemm": lambda: gen9blas.gemm(A, B, C=C), "symm": lambda: gen9blas.symm(A, B, C=C), "syrk": lambda: gen9blas.syrk(A, C=C),
     "syr2k": lambda: gen9blas.syr2k(A, B, C=C), "trmm": lambda: gen9blas.trmm(A, B), "trsm": lambda: gen9blas.trsm(A, B)}[op]
f()
k, t0, c0 = 0, time.perf_counter(), time.process_time()
while time.perf_counter() - t0 < 6.0:
    f()
    k += 1
t, c = (time.perf_counter() - t0) / k, (time.process_time() - c0) / k
print(f"  {op:6s} {sys.argv[3]} n={n}: {t * 1e3:6.0f} ms/wywolanie, CPU {100 * c / t:3.0f}%  ({k} wywolan)")
