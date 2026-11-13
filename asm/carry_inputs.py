"""Inputs for check_simd16_carry.sh: accurate HGEMM (gemm_pb_f16w and gen9blas hgemm accurate) on edge shapes and on
FP16 data with subnormals, values near the FP16 range and inf/NaN; writes all output bytes to argv[1].
Run from the tree whose lib/ holds the library under test."""
import ctypes, sys
import numpy as np
sys.path.insert(0, "lib")
import gen9blas as g  # noqa: E402
g._init(); L = g._lib; _i, _p = ctypes.c_int, ctypes.c_void_p
L.gen9blas_pack_b.argtypes = [_i, _i, _i, _i, _p, _i]; L.gen9blas_pack_b.restype = _p
L.gen9blas_gemm_pb_f16w.argtypes = [_i, _p, _i, _p, _p, _i, ctypes.POINTER(_i)]
rng = np.random.default_rng(17); out = []
for M, K, N, kind in ((1, 32, 1, "n"), (17, 33, 47, "n"), (100, 513, 129, "n"), (512, 1536, 256, "n"), (64, 96, 64, "sub"),
                      (64, 96, 64, "big"), (64, 96, 64, "nan"), (300, 896, 512, "n")):
    A = rng.standard_normal((M, K)); B = rng.standard_normal((K, N)) * 0.1
    if kind == "sub": A *= 2.0 ** -20; B *= 2.0 ** -6           # products in the FP16 subnormal range
    if kind == "big": A *= 200.0; B *= 30.0                     # chunk sums near / beyond 65504
    if kind == "nan": A[3, 5] = np.inf; A[7, 9] = np.nan
    a16, b16 = A.astype(np.float16), B.astype(np.float16)
    out.append(g.hgemm(a16, b16, accurate=True).tobytes())
    W = b16.T.copy(); P = L.gen9blas_pack_b(2, 1, K, N, W.ctypes.data, K)
    Af = a16.astype(np.float32); C = np.zeros((M, N), np.float32); nf = _i()
    L.gen9blas_gemm_pb_f16w(M, Af.ctypes.data, K, P, C.ctypes.data, N, ctypes.byref(nf))
    out.append(C.tobytes()); out.append(bytes([nf.value]))
open(sys.argv[1], "wb").write(b"".join(out))
