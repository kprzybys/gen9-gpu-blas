"""Punkt odniesienia dla optymalizacji poziomów 1–2: GB/s i CPU (tablice gen9blas.empty, tylko GPU).

Uruchom: python3 bench/level12_speed.py [float32|float16|float64 ...]
GB/s z rozmiaru danych wzoru (trójkąt: pół macierzy, SYMV: trójkąt czytany raz = pół macierzy -- ile musi przeczytać
najlepszy algorytm). Wektory 2²⁴, macierze 8192². Najlepszy czas z 5, CPU -- średnia (100% = jeden rdzeń).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "lib"))
import gen9blas  # noqa: E402

rng = np.random.default_rng(1)
for nazwa_typu in sys.argv[1:] or ["float32", "float16"]:
    typ = np.dtype(nazwa_typu).type; e = np.dtype(typ).itemsize
    n1, n2 = 1 << 24, 8192 if e < 8 else 6144
    x = gen9blas.empty((n1,), typ); y = gen9blas.empty((n1,), typ); x[:] = rng.uniform(-1, 1, n1); y[:] = rng.uniform(-1, 1, n1)
    A = gen9blas.empty((n2, n2), typ); A[:] = rng.uniform(-1, 1, (n2, n2)) / n2; A[np.diag_indices(n2)] = 1
    xm, ym = x[:n2], y[:n2]
    print(f"== {nazwa_typu}")
    for nazwa, bajty, f in [
        ("dot", 2 * n1 * e, lambda: gen9blas.dot(x, y)), ("nrm2", n1 * e, lambda: gen9blas.nrm2(x)), ("asum", n1 * e, lambda: gen9blas.asum(x)),
        ("iamax", n1 * e, lambda: gen9blas.iamax(x)), ("axpy", 3 * n1 * e, lambda: gen9blas.axpy(0.5, x, y)),
        ("gemv N", n2 * n2 * e, lambda: gen9blas.gemv(A, xm, y=ym)), ("gemv T", n2 * n2 * e, lambda: gen9blas.gemv(A, xm, y=ym, trans=True)),
        ("symv", n2 * n2 * e / 2, lambda: gen9blas.symv(A, xm, y=ym)),
        ("trsv N", n2 * n2 * e / 2, lambda: gen9blas.trsv(A, xm)), ("trsv T", n2 * n2 * e / 2, lambda: gen9blas.trsv(A, xm, trans=True))]:
        f(); sc, cp = [], []
        for _ in range(5):
            t, c = time.perf_counter(), time.process_time(); f(); sc.append(time.perf_counter() - t); cp.append(time.process_time() - c)
        print(f"  {nazwa:8s} {bajty / min(sc) / 1e9:5.1f} GB/s  {min(sc) * 1e3:7.2f} ms  CPU {100 * np.mean(cp) / np.mean(sc):3.0f}%")
