"""Whole-call GEMM speed versus size: FP16, FP16 accurate, FP32 (gen9blas.empty arrays, includes packing).

Run: python3 bench/gemm_sizes.py [n ...]      (default 1024 2048 3072 4096)
Best of 5 calls after one warm-up call; GFLOPS = 2 n^3 / time.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "lib"))
import gen9blas  # noqa: E402

rng = np.random.default_rng(1)
print(f"{'n':>6s} {'FP16':>10s} {'FP16 accurate':>14s} {'FP32':>10s}   (GFLOPS)")
for n in [int(a) for a in sys.argv[1:]] or [1024, 2048, 3072, 4096]:
    row = []
    for dt, acc in ((np.float16, False), (np.float16, True), (np.float32, False)):
        if not gen9blas.available(dt, acc):   # Gen8: no FP16
            row.append(float("nan"))
            continue
        A, B, C = (gen9blas.empty((n, n), dt) for _ in range(3))
        A[:] = rng.uniform(-1, 1, (n, n))
        B[:] = rng.uniform(-1, 1, (n, n))
        gen9blas.gemm(A, B, C=C, accurate=acc)
        best = 1e9
        for _ in range(5):
            t = time.perf_counter()
            gen9blas.gemm(A, B, C=C, accurate=acc)
            best = min(best, time.perf_counter() - t)
        row.append(2 * n ** 3 / best / 1e9)
    print(f"{n:6d} {row[0]:10.1f} {row[1]:14.1f} {row[2]:10.1f}")
