"""Krok 0: punkt odniesienia GEMM na CPU (NumPy -> OpenBLAS).

Użycie: python3 gemm_cpu.py <f32|f64> <N> [sekundy]
Format wyniku taki sam jak gemm_clblast. Dokładność liczona na 256 elementach względem float64.
"""
import sys
import time

import numpy as np

prec, n = sys.argv[1], int(sys.argv[2])
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
dt = {"f32": np.float32, "f64": np.float64}[prec]

rng = np.random.default_rng(1234)
a = rng.uniform(-1, 1, (n, n)).astype(dt)
b = rng.uniform(-1, 1, (n, n)).astype(dt)
a @ b  # rozgrzewka

if secs > 0:
    runs, start = 0, time.perf_counter()
    okno, t_okna, r_okna = 10.0, time.perf_counter(), 0
    while time.perf_counter() - start < secs:
        c = a @ b
        runs += 1
        r_okna += 1
        if time.perf_counter() - t_okna >= okno:   # ślad co ~10 s (test długotrwały)
            dt = time.perf_counter() - t_okna
            print(f"slad {time.perf_counter() - start:6.1f} s  {2 * n**3 * r_okna / dt / 1e9:7.1f} GFLOPS", file=sys.stderr)
            t_okna, r_okna = time.perf_counter(), 0
    best = (time.perf_counter() - start) / runs
else:
    best = 1e30
    for _ in range(5):
        t0 = time.perf_counter()
        c = a @ b
        best = min(best, time.perf_counter() - t0)

idx = np.random.default_rng(99).integers(0, n, (256, 2))
a64, b64 = a.astype(np.float64), b.astype(np.float64)
errs = []
for i, j in idx:
    prod = a64[i, :] * b64[:, j]
    errs.append(abs(float(c[i, j]) - prod.sum()) / np.abs(prod).sum())
print(f"openblas {prec}gemm N={n:<5} czas={best*1e3:9.3f} ms  {2*n**3/best/1e9:7.1f} GFLOPS  "
      f"blad_wzgl: sr={np.mean(errs):.2e} max={np.max(errs):.2e}{'  (tryb ciagly)' if secs else ''}")
