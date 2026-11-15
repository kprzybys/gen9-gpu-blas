"""gen9blas_gemm_pb_skinny: small batches (M <= 16) on HS-packed weights, against float64 NumPy and gemm_pb_f16w.

Run: python3 lib/test_skinny_pb.py [--time]
Weights as ggml stores them (W[N][K] row-major, packed with transB = 1), activations F32 (ordinary NumPy memory: userptr).
Shapes of PaddleOCR-VL's text model (ERNIE 4.5 0.3B: d 1024, FFN 3072, K/V 256) plus odd sizes (N % 16, K % 32 != 0).
--time: median ms per call against gemm_pb_f16w (the backend's path for small batches today) and GB/s of FP16 weights.
"""
import ctypes
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import gen9blas  # noqa: E402

L = gen9blas._lib
_i, _p = ctypes.c_int, ctypes.c_void_p
L.gen9blas_pack_b.argtypes = [_i, _i, _i, _i, _p, _i]
L.gen9blas_pack_b.restype = _p
L.gen9blas_packed_free.argtypes = [_p]
L.gen9blas_gemm_pb_skinny.argtypes = [_i, _p, _i, _p, _p, _i, ctypes.POINTER(_i)]
L.gen9blas_gemm_pb_f16w.argtypes = [_i, _p, _i, _p, _p, _i, ctypes.POINTER(_i)]
L.gen9blas_skinny_pb_supported.restype = _i
HS = 2
gen9blas._init()
if not L.gen9blas_skinny_pb_supported():
    sys.exit("gen9blas_skinny_pb16_* missing from the kernel table")
rng = np.random.default_rng(11)
errors = 0
TIME = "--time" in sys.argv


def run(fn, M, A, P, N):
    C = np.full((M, N), np.nan, np.float32)
    nf = _i(0)
    rc = fn(M, A.ctypes.data, A.shape[1], P, C.ctypes.data, N, ctypes.byref(nf))
    if rc != 0:
        raise RuntimeError(f"rc {rc}: {L.gen9blas_error().decode()}")
    return C, nf.value


def med_ms(fn, M, A, P, N, reps=20):
    C = np.zeros((M, N), np.float32); nf = _i(0)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(M, A.ctypes.data, A.shape[1], P, C.ctypes.data, N, ctypes.byref(nf))
        ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts))


for (N, K) in [(1024, 1024), (256, 1024), (3072, 1024), (1024, 3072), (1000, 1000), (37, 96)]:
    W = (rng.standard_normal((N, K)) * 0.05).astype(np.float16)
    P = L.gen9blas_pack_b(HS, 1, K, N, W.ctypes.data, K)
    if not P:
        raise RuntimeError(L.gen9blas_error().decode())
    W64 = W.astype(np.float64)
    for M in (1, 2, 3, 4, 5, 8, 9, 12, 16):
        A = rng.standard_normal((M, K)).astype(np.float32)
        C, nf = run(L.gen9blas_gemm_pb_skinny, M, A, P, N)
        ref = A.astype(np.float64) @ W64.T
        mag = np.abs(A.astype(np.float64)) @ np.abs(W64.T)
        err = float(np.max(np.abs(C - ref) / mag))
        ok = err < 1e-5 and nf == 0 and np.isfinite(C).all()
        errors += not ok
        line = f"N {N:5} K {K:5} M {M:2}: rel err {err:.1e} nonfinite {nf} {'OK' if ok else 'FAIL'}"
        if TIME and M in (1, 4, 8, 16) and N >= 256:
            ts = med_ms(L.gen9blas_gemm_pb_skinny, M, A, P, N)
            tf = med_ms(L.gen9blas_gemm_pb_f16w, M, A, P, N)
            line += f" | skinny {ts:.3f} ms ({N * K * 2 / ts / 1e6:.1f} GB/s) vs gemm_pb_f16w {tf:.3f} ms ({tf / ts:.1f}x)"
        print(line, flush=True)
    L.gen9blas_packed_free(P)
print("errors:", errors)
sys.exit(1 if errors else 0)
