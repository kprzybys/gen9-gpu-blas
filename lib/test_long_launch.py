"""Long GEMM launches: kcl has no mid-thread preemption, so a kernel running longer than the engine's preempt_timeout_ms
(640 ms) gets the engine reset when another context (e.g. the desktop) waits for the GPU, and its results are lost.
The library splits long launches into row bands (GEN9BLAS_MAX_LAUNCH_MS, default 100) and reports a reset as an error.

Run: python3 lib/test_long_launch.py
Cases that used to be single launches of 1-3 s: DGEMM n = 4096, SGEMM and accurate HGEMM n = 8192 (plain FP16
accumulation of 8192 products in (0, 1) loses precision by itself), batched DGEMM 2 x 3072.
Checked: rows at the start, middle and end of C against NumPy (float64), max relative error per row.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import gen9blas  # noqa: E402

rng = np.random.default_rng(3)
errors = 0


def rand(shape, dt):
    X = gen9blas.empty(shape, dt)
    X[:] = rng.uniform(0, 1, shape)
    return X


def check(name, A, B, C, tol):
    global errors
    n = A.shape[-2]
    rows = [0, n // 2, n - 1]
    ref = A[..., rows, :].astype(np.float64) @ B.astype(np.float64)
    e = float(np.max(np.abs(C[..., rows, :] - ref) / np.abs(ref)))
    ok = e <= tol
    errors += not ok
    print(f"{'OK ' if ok else 'ERR'} {name}: max rel err {e:.1e} (tolerance {tol:.0e})")


for name, dt, n, tol in [c for c in (("dgemm n=4096", np.float64, 4096, 1e-12), ("sgemm n=8192", np.float32, 8192, 1e-4),
                         ("hgemm accurate n=8192", np.float16, 8192, 5e-3)) if gen9blas.available(c[1], c[1] == np.float16)]:
    A, B = rand((n, n), dt), rand((n, n), dt)
    try:
        check(name, A, B, gen9blas.gemm(A, B, accurate=dt == np.float16), tol)
    except RuntimeError as e:
        errors += 1
        print(f"ERR {name}: {e}")
    del A, B
    gen9blas.free_pool()
A, B = rand((2, 3072, 3072), np.float64), rand((2, 3072, 3072), np.float64)
try:
    check("batched dgemm 2 x 3072", A, B, gen9blas.gemm_batched(A, B), 1e-12)
except RuntimeError as e:
    errors += 1
    print(f"ERR batched dgemm: {e}")

print("RESULT: all OK" if errors == 0 else f"RESULT: {errors} errors")
sys.exit(1 if errors else 0)
