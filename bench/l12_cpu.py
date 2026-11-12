"""CPU measurement (NumPy/OpenBLAS) for energy comparison: l12_cpu.py saxpy|sgemv <n> <seconds> <threads>"""
import os, sys, time
os.environ["OPENBLAS_NUM_THREADS"] = sys.argv[4]
import numpy as np
op_name, n, secs = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
if op_name == "saxpy":
    x = np.full(n, 0.5, np.float32); y = np.full(n, 0.5, np.float32); b = 12 * n
    from scipy.linalg.blas import saxpy as f
    op = lambda: f(x, y, a=0.5)
else:
    A = np.full((n, n), 0.5, np.float32); x = np.full(n, 0.5, np.float32); b = 4 * n * n
    op = lambda: A @ x
t0 = time.perf_counter(); it = 0
while time.perf_counter() - t0 < secs:
    op(); it += 1
print(f"CPU({sys.argv[4]} thr) {op_name} n={n}: {b * it / (time.perf_counter() - t0) / 1e9:.1f} GB/s")
