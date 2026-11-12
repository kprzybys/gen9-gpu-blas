"""Skinny GEMM test (small M or N, no packing; docs/skinny-gemm.md) against NumPy in FP64.

Run: python3 lib/test_skinny.py   (forces the skinny kernel for min(M, N) ≤ 64 via GEN9BLAS_SKINNY=64, regardless of the
selection threshold in gen9blas.c; the old path is covered by test_basic and test_level3)
Cases: both weight layouts (A·Wᵀ and A·B), transposes of A and B (.T views), small N (role swap), alpha/beta, C as a slice
(row stride), odd sizes and K tails (K mod 8, K mod 128), M around the tiles (1, 3, 4, 5, 8, 9, 16, 17, 33, 64),
USM memory (gen9blas.empty) and plain (NumPy). FP16 and FP32.
"""
import os
import sys

import numpy as np

os.environ.setdefault("GEN9BLAS_SKINNY", "64")   # before loading the library (read once)
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import gen9blas  # noqa: E402

rng = np.random.default_rng(7)
TOL = {np.float16: 4e-3, np.float32: 2e-6}
bad = n_test = 0


def tab(shape, dt, usm):
    X = gen9blas.empty(shape, dt) if usm else np.empty(shape, dt)
    X[...] = rng.uniform(-1, 1, shape).astype(dt)
    return X


def check(desc, dt, result, ref):
    global bad, n_test
    n_test += 1
    e = np.max(np.abs(result.astype(np.float64) - ref)) / max(1e-30, np.max(np.abs(ref)))
    if not np.isfinite(e) or e > TOL[dt]:
        bad += 1
        print(f"BAD {desc}: error {e:.2e}")


for dt in (np.float16, np.float32):
    fn = gen9blas.hgemm if dt == np.float16 else gen9blas.sgemm
    label = np.dtype(dt).name
    for usm in (True, False):
        mem = "usm" if usm else "plain"
        for M, N, K in [(1, 1000, 777), (3, 517, 1029), (4, 2048, 128), (5, 64, 9), (8, 333, 4096), (9, 1500, 1500),
                        (16, 4096, 1000), (17, 130, 257), (33, 700, 300), (64, 256, 513), (1, 8, 3), (2, 7, 2000)]:
            for tA in (False, True):
                for tB in (False, True):
                    A = tab((K, M), dt, usm).T if tA else tab((M, K), dt, usm)
                    B = tab((N, K), dt, usm).T if tB else tab((K, N), dt, usm)
                    Ad, Bd = A.astype(np.float64), B.astype(np.float64)
                    desc = f"{label} {mem} M={M} N={N} K={K} tA={tA} tB={tB}"
                    check(desc, dt, fn(A, B), Ad @ Bd)
                    # small N: role swap (C = Bᵀ side), the same operands in reverse order
                    check(desc + " (small N)", dt, fn(B.T, A.T), Bd.T @ Ad.T)
            # alpha, beta, C as a slice of a larger array (row stride)
            A = tab((M, K), dt, usm); W = tab((N, K), dt, usm)
            big = tab((M + 2, N + 5), dt, usm); C = big[1:M + 1, 2:N + 2]
            C0 = C.astype(np.float64); D0 = big.copy()
            fn(A, W.T, alpha=0.75, beta=-1.5, C=C)
            check(f"{label} {mem} M={M} N={N} K={K} alpha/beta, C slice", dt, C,
                    0.75 * A.astype(np.float64) @ W.astype(np.float64).T - 1.5 * C0)
            mask = np.ones(big.shape, bool); mask[1:M + 1, 2:N + 2] = False
            if not np.array_equal(big[mask], D0[mask]):   # nothing outside the slice of C may change
                bad += 1; print(f"BAD {label} {mem} M={M} N={N}: change outside the slice of C")
print(f"tests: {n_test}")
print("RESULT: all OK" if not bad else f"RESULT: {bad} errors")
sys.exit(1 if bad else 0)
