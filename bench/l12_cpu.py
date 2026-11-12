"""Pomiar CPU (NumPy/OpenBLAS) dla porównania energii: l12_cpu.py saxpy|sgemv <n> <sekundy> <watki>"""
import os, sys, time
os.environ["OPENBLAS_NUM_THREADS"] = sys.argv[4]
import numpy as np
co, n, sek = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
if co == "saxpy":
    x = np.full(n, 0.5, np.float32); y = np.full(n, 0.5, np.float32); b = 12 * n
    from scipy.linalg.blas import saxpy as f
    op = lambda: f(x, y, a=0.5)
else:
    A = np.full((n, n), 0.5, np.float32); x = np.full(n, 0.5, np.float32); b = 4 * n * n
    op = lambda: A @ x
t0 = time.perf_counter(); it = 0
while time.perf_counter() - t0 < sek:
    op(); it += 1
print(f"CPU({sys.argv[4]} w) {co} n={n}: {b * it / (time.perf_counter() - t0) / 1e9:.1f} GB/s")
