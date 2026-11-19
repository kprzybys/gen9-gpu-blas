"""Status codes of the 0.2 API (gen9blas_status): one call per code, plus gen9blas_last_status for pointer functions.

Golden checks arithmetic; this checks the error paths (classification and propagation). Needs the GPU.
usage: python3 test_errors.py   ->  "RESULT: n/n OK"
"""
import ctypes, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen9blas as g  # noqa: E402

L = g._lib
_p, _i, _d = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
OK, DECLINED, ERR, UNSUPPORTED, INVALID, OOM, NO_DEVICE = 0, 1, -1, -2, -3, -4, -5
L.gen9blas_pack_b.restype = _p
L.gen9blas_pack_b.argtypes = [_i, _i, _i, _i, _p, _i]
L.gen9blas_gemm_pb.argtypes = [_i, _i, _d, _p, _i, _p, _d, _p, _i]
L.gen9blas_gemm_pb_skinny.argtypes = [_i, _p, _i, _p, _p, _i, _p]
L.gen9blas_packed_b_free.argtypes = [_p]
L.gen9blas_alloc.restype = _p
L.gen9blas_alloc.argtypes = [ctypes.c_size_t]
L.gen9blas_gemm.argtypes = [_i, _i, _i, _i, _i, _i, _d, _p, _i, _p, _i, _d, _p, _i]
L.gen9blas_gemv.argtypes = [_i, _i, _i, _i, _d, _p, _i, _p, _i, _d, _p, _i]
L.gen9blas_axpy.argtypes = [_i, _i, _d, _p, _i, _p, _i]

res = []
def check(name, got, want):
    ok = got == want
    res.append(ok)
    print(f"{'OK  ' if ok else 'FAIL'} {name:44s} got {got} want {want}" + ("" if ok else f"  ({L.gen9blas_error().decode()})"))

check("version 0.2.x", L.gen9blas_version() // 100, 2)
check("init", L.gen9blas_init(), OK)
check("init again (once per process)", L.gen9blas_init(), OK)
check("has(unknown feature) = 0", L.gen9blas_has(999), 0)
check("device_gen in (8, 9, 12)", L.gen9blas_device_gen() in (8, 9, 12), True)

a = np.ones((8, 8), np.float32); b = np.ones((8, 8), np.float32); c = np.zeros((8, 8), np.float32)
pa, pb, pc = a.ctypes.data, b.ctypes.data, c.ctypes.data
check("gemm ok", L.gen9blas_gemm(g.GEN9BLAS_S, 0, 0, 8, 8, 8, 1.0, pa, 8, pb, 8, 0.0, pc, 8), OK)
check("  last_status after success", L.gen9blas_last_status(), OK)
check("gemm type 7 -> UNSUPPORTED", L.gen9blas_gemm(7, 0, 0, 8, 8, 8, 1.0, pa, 8, pb, 8, 0.0, pc, 8), UNSUPPORTED)
x = np.ones(8, np.float32); y = np.zeros(8, np.float32)
check("gemv lda < N -> INVALID_ARG", L.gen9blas_gemv(g.GEN9BLAS_S, 0, 8, 8, 1.0, pa, 4, x.ctypes.data, 1, 0.0, y.ctypes.data, 1), INVALID)
check("axpy incx 0 -> INVALID_ARG", L.gen9blas_axpy(g.GEN9BLAS_S, 8, 1.0, x.ctypes.data, 0, y.ctypes.data, 1), INVALID)
check("gemm_pb(NULL) -> INVALID_ARG", L.gen9blas_gemm_pb(0, 8, 1.0, pa, 8, None, 0.0, pc, 8), INVALID)
check("pack_b type 7 -> NULL", L.gen9blas_pack_b(7, 0, 8, 8, pb, 8), None)
check("  last_status UNSUPPORTED", L.gen9blas_last_status(), UNSUPPORTED)
check("pack_b K = 0 -> NULL", L.gen9blas_pack_b(g.GEN9BLAS_S, 0, 0, 8, pb, 8), None)
check("  last_status INVALID_ARG", L.gen9blas_last_status(), INVALID)
if L.gen9blas_has(g.FEAT_FP16):
    w = np.ones((64, 64), np.float16)
    P = L.gen9blas_pack_b(g.GEN9BLAS_HS, 1, 64, 64, w.ctypes.data, 64)
    A = np.ones((17, 64), np.float32); C = np.zeros((17, 64), np.float32); nf = ctypes.c_int(0)
    check("gemm_pb_skinny M = 17 -> DECLINED", L.gen9blas_gemm_pb_skinny(17, A.ctypes.data, 64, P, C.ctypes.data, 64, ctypes.byref(nf)), DECLINED)
    S = L.gen9blas_pack_b(g.GEN9BLAS_S, 1, 64, 64, np.ones((64, 64), np.float32).ctypes.data, 64)
    check("gemm_pb_skinny on S-packed B -> INVALID_ARG", L.gen9blas_gemm_pb_skinny(4, A.ctypes.data, 64, S, C.ctypes.data, 64, ctypes.byref(nf)), INVALID)
    L.gen9blas_packed_b_free(P); L.gen9blas_packed_b_free(S)
check("alloc 2^44 B -> NULL", L.gen9blas_alloc(1 << 44), None)
check("  last_status OUT_OF_MEMORY", L.gen9blas_last_status(), OOM)
print(f"RESULT: {sum(res)}/{len(res)} {'OK' if all(res) else 'FAILED'}")
sys.exit(0 if all(res) else 1)
