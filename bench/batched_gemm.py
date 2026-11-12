"""Batched GEMM (gen9blas.gemm_batched): GFLOPS for FP16 and FP32, gen9blas.empty arrays.

Run: python3 bench/batched_gemm.py
Cases: 256 x 64^3, 64 x 128^3, 16 x 256^3 (count x M = N = K). Best of 5 calls after one warm-up call; GFLOPS = 2 M N K count / time.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "lib"))
import gen9blas  # noqa: E402

rng = np.random.default_rng(1)
print(f"{'case':16s} {'FP16':>12s} {'FP32':>12s}")
for cnt, m in ((256, 64), (64, 128), (16, 256)):
    row = []
    for dt in (np.float16, np.float32):
        A = gen9blas.empty((cnt, m, m), dt)
        B = gen9blas.empty((cnt, m, m), dt)
        C = gen9blas.empty((cnt, m, m), dt)
        A[:] = rng.uniform(-1, 1, A.shape)
        B[:] = rng.uniform(-1, 1, B.shape)
        gen9blas.gemm_batched(A, B, C=C)
        best = min((lambda t: (gen9blas.gemm_batched(A, B, C=C), time.perf_counter() - t)[1])(time.perf_counter()) for _ in range(5))
        row.append(f"{2 * cnt * m ** 3 / best / 1e9:7.1f} GFLOPS")
    print(f"{cnt} x {m}^3".ljust(16) + " " + " ".join(f"{r:>12s}" for r in row))
