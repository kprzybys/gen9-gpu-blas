"""Packed B kept on the GPU: gen9blas_pack_b / gen9blas_gemm_pb / gen9blas_packed_free (ggml backend, step 1a).

Run: python3 lib/test_packed.py          (GEN9BLAS_PACK_BAND_MB=1 forces many packing / epilogue bands)
Ordinary NumPy memory (userptr path, as llama.cpp's mmap'd weights). Checks, against float64 NumPy on sampled rows and
against gen9blas_gemm on the same data: transposed and plain B, odd shapes (padding), alpha / beta (epilogue path),
reuse of one packed B with several A, B overwritten after packing (the packed copy must not change), FP16 / FP16
accurate / FP64, and a weight of the size of Qwen2.5-1.5B's output projection (151936 x 1536 FP32, 0.93 GB) with M = 512
(a 239 GFLOP product: split into launch bands).
"""
import ctypes
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import gen9blas  # noqa: E402

L = gen9blas._lib
_i, _p, _d = ctypes.c_int, ctypes.c_void_p, ctypes.c_double
L.gen9blas_pack_b.argtypes = [_i, _i, _i, _i, _p, _i]
L.gen9blas_pack_b.restype = _p
L.gen9blas_gemm_pb.argtypes = [_i, _i, _i, _d, _p, _i, _p, _d, _p, _i]
L.gen9blas_packed_free.argtypes = [_p]
H, S, HS, D = 0, 1, 2, 3
DT = {H: np.float16, S: np.float32, HS: np.float16, D: np.float64}
TOL = {H: 3e-3, S: 1e-6, HS: 3e-4, D: 1e-14}   # max relative error to sum|a*b| (README: hgemm ~1e-4, accurate ~1e-5, sgemm ~1e-8)
gen9blas._init()
rng = np.random.default_rng(7)
errors = 0


def pack(t, B, transB):
    ldb = B.shape[1]
    K, N = (B.shape[1], B.shape[0]) if transB else B.shape
    P = L.gen9blas_pack_b(t, transB, K, N, B.ctypes.data, ldb)
    if not P:
        raise RuntimeError(L.gen9blas_error().decode())
    return P, K, N


def gemm_pb(t, A, P, N, alpha=1.0, beta=0.0, C=None):
    M = A.shape[0]
    if C is None:
        C = np.zeros((M, N), DT[t])
    rc = L.gen9blas_gemm_pb(t, 0, M, alpha, A.ctypes.data, A.shape[1], P, beta, C.ctypes.data, N)
    if rc != 0:
        raise RuntimeError(L.gen9blas_error().decode())
    return C


def rel_err(C, A, Bkn, alpha=1.0, beta=0.0, C0=None, rows=None):
    """max over sampled rows of |C - ref| / (|alpha| sum|a b| + |beta c0|)."""
    M = A.shape[0]
    rows = rows if rows is not None else sorted({0, M // 2, M - 1})
    A64, B64 = A[rows].astype(np.float64), Bkn.astype(np.float64)
    ref = alpha * (A64 @ B64)
    mag = abs(alpha) * (np.abs(A64) @ np.abs(B64))
    if C0 is not None and beta != 0:
        ref += beta * C0[rows].astype(np.float64)
        mag += abs(beta) * np.abs(C0[rows].astype(np.float64))
    return float(np.max(np.abs(C[rows].astype(np.float64) - ref) / np.maximum(mag, 1e-30)))


def case(name, t, M, N, K, transB=1, alpha=1.0, beta=0.0, compare_gemm=True):
    global errors
    dt = DT[t]
    A = rng.uniform(-1, 1, (M, K)).astype(dt)
    B = rng.uniform(-1, 1, (N, K) if transB else (K, N)).astype(dt)
    Bkn = B.T if transB else B
    C0 = rng.uniform(-1, 1, (M, N)).astype(dt)
    P, _, _ = pack(t, B, transB)
    C = gemm_pb(t, A, P, N, alpha, beta, C0.copy())
    e = rel_err(C, A, Bkn, alpha, beta, C0)
    same = ""
    if compare_gemm:
        C2 = C0.copy()
        gen9blas._check(L.gen9blas_gemm(t, 0, transB, M, N, K, alpha, A.ctypes.data, K, B.ctypes.data, B.shape[1],
                                        beta, C2.ctypes.data, N))
        same = " identical to gemm" if np.array_equal(C, C2) else f" max |pb - gemm| {np.max(np.abs(C.astype(np.float64) - C2)):.2e}"
    ok = e <= TOL[t] and np.all(np.isfinite(C))
    errors += not ok
    print(f"{'OK ' if ok else 'ERR'} {name:42s} rel err {e:.2e}{same}")
    L.gen9blas_packed_free(P)


case("S  512x1536x1536 transB", S, 512, 1536, 1536)
case("S  512x256x1536 transB (K/V)", S, 512, 256, 1536)
case("S  512x8960x1536 transB (gate/up)", S, 512, 8960, 1536)
case("S  512x1536x8960 transB (down)", S, 512, 1536, 8960)
case("S  33x100x2051 transB (padding)", S, 33, 100, 2051)
case("S  37x1000x300 plain B (pb / w2)", S, 37, 1000, 300, transB=0)
case("S  64x1536x1536 alpha 0.5 beta -2 (epilogue)", S, 64, 1536, 1536, alpha=0.5, beta=-2.0)
case("H  256x1024x1024 transB", H, 256, 1024, 1024)
case("HS 256x1024x2048 transB", HS, 256, 1024, 2048)
case("D  64x512x512 transB", D, 64, 512, 512)

# reuse and independence from the original: pack once, overwrite B, multiply with two different A
A1 = rng.uniform(-1, 1, (128, 1536)).astype(np.float32)
A2 = rng.uniform(-1, 1, (512, 1536)).astype(np.float32)
B = rng.uniform(-1, 1, (1536, 1536)).astype(np.float32)
Bkn = B.T.copy()
P, K, N = pack(S, B, 1)
B[:] = 0.0
for nm, A in (("reuse 1 (M=128)", A1), ("reuse 2 (M=512)", A2), ("reuse 3 (M=128 again)", A1)):
    e = rel_err(gemm_pb(S, A, P, N), A, Bkn)
    ok = e <= TOL[S]
    errors += not ok
    print(f"{'OK ' if ok else 'ERR'} S  {nm + ', B zeroed after packing':40s} rel err {e:.2e}")
L.gen9blas_packed_free(P)

# output projection of Qwen2.5-1.5B: 151936 x 1536 FP32 (0.93 GB), M = 512 (llama-perplexity: all rows)
if os.environ.get("SKIP_BIG") is None:
    N, K, M = 151936, 1536, 512
    W = rng.uniform(-1, 1, (N, K)).astype(np.float32)
    A = rng.uniform(-1, 1, (M, K)).astype(np.float32)
    t0 = time.time(); P, _, _ = pack(S, W, 1); t_pack = time.time() - t0
    C = np.zeros((M, N), np.float32)
    t0 = time.time(); gemm_pb(S, A, P, N, C=C); t_mm = time.time() - t0
    cols = rng.choice(N, 4096, replace=False)
    e = rel_err(C[:, cols], A, W[cols].T)
    ok = e <= TOL[S]
    errors += not ok
    print(f"{'OK ' if ok else 'ERR'} S  head 512x151936x1536 (0.93 GB packed)       rel err {e:.2e}  "
          f"pack {t_pack * 1e3:.0f} ms ({W.nbytes / t_pack / 1e9:.2f} GB/s), gemm_pb {t_mm * 1e3:.0f} ms "
          f"({2 * M * N * K / t_mm / 1e9:.0f} GFLOPS)")
    L.gen9blas_packed_free(P)

print("errors:", errors)
sys.exit(1 if errors else 0)
