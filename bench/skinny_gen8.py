"""Skinny SGEMM on Gen8: C = A (M x K) . W^T (W: N x K), gen9blas.empty arrays, best of 7 after a warm-up, weight GB/s.

Run: python3 bench/skinny_gen8.py   (GEN9BLAS_SKINNY_R=0: the skinny_k kernels instead of skinny_r, for comparison)
Results: results/braswell-gen8-tuned-2026-11-13.txt.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "lib"))
import gen9blas  # noqa: E402

rng = np.random.default_rng(1)
for N, K in ((4096, 4096), (11008, 4096)):
    W = gen9blas.empty((N, K), np.float32); W[:] = rng.uniform(-1, 1, (N, K))
    for M in (1, 2, 4, 8, 16):
        A = gen9blas.empty((M, K), np.float32); A[:] = rng.uniform(-1, 1, (M, K))
        f = lambda: gen9blas.sgemm(A, W.T)
        f(); f(); best = 1e9
        for _ in range(7):
            t = time.perf_counter(); f(); best = min(best, time.perf_counter() - t)
        C = f().astype(np.float64); R = A.astype(np.float64) @ W.T.astype(np.float64)
        e = np.max(np.abs(C - R)) / np.max(np.abs(R))
        print(f"N={N:6d} K={K} M={M:3d}  {best*1e3:8.3f} ms  {N*K*4/best/1e9:5.2f} GB/s  {2*M*N*K/best/1e9:6.2f} GFLOPS  err {e:.1e}", flush=True)
    del W; gen9blas.free_pool()
