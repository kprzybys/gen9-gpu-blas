"""Poziom 3 libgen9blas: czas całego wywołania i czas CPU procesu (wszystkie wątki) na wywołanie.

Uruchom: python3 bench/level3_cpu.py [n ...]   (GEN9BLAS_LIB=ścieżka: inna wersja biblioteki; GEN9BLAS_WAIT=aktywne)
CPU% = czas CPU / czas ściany: ile rdzenia zajmuje wywołanie (cel: blisko 0 -- CPU tylko zleca i śpi).
Tablice gen9blas.empty (zero-copy). Najlepszy czas ściany z 5 powtórzeń, czas CPU -- średnia.
"""
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # wątki OpenBLAS (NumPy) kręcą się w tle i zawyżają CPU procesu
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "lib"))
import gen9blas  # noqa: E402

rng = np.random.default_rng(1)


def tablica(n, typ, diag=0.0):
    X = gen9blas.empty((n, n), typ)
    X[:] = rng.uniform(-1, 1, (n, n))
    if diag:
        X[np.diag_indices(n)] = diag
    return X


for n in [int(a) for a in sys.argv[1:]] or [2048, 4096]:
    for typ in (np.float32, np.float16):
        A, B, C = tablica(n, typ, n), tablica(n, typ), tablica(n, typ)
        print(f"== n = {n}, {np.dtype(typ).name}")
        for nazwa, f, fl in [("gemm", lambda: gen9blas.gemm(A, B, C=C), 2 * n ** 3), ("symm", lambda: gen9blas.symm(A, B, C=C), 2 * n ** 3),
                             ("syrk", lambda: gen9blas.syrk(A, C=C), n ** 3), ("syr2k", lambda: gen9blas.syr2k(A, B, C=C), 2 * n ** 3),
                             ("trmm", lambda: gen9blas.trmm(A, B), n ** 3), ("trsm", lambda: gen9blas.trsm(A, B), n ** 3)]:
            f()
            sciana, cpu = [], []
            for _ in range(5):
                t, c = time.perf_counter(), time.process_time()
                f()
                sciana.append(time.perf_counter() - t); cpu.append(time.process_time() - c)
            t, c = min(sciana), sum(cpu) / len(cpu)
            print(f"  {nazwa:6s} {fl / t / 1e9:6.1f} GFLOPS  {t * 1e3:6.0f} ms   CPU {c * 1e3:6.0f} ms ({100 * c / np.mean(sciana):3.0f}%)")
