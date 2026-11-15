"""Python wrapper (ctypes) for libgen9blas: BLAS (levels 1-3) on Intel UHD 620 with GEMM loops in assembly.

    import numpy as np, gen9blas
    C = gen9blas.hgemm(A, B)                        # float16, FP16 accumulation (fastest); accurate=True: chunked accumulation to FP32
    C = gen9blas.sgemm(A.T, B, alpha=2.0)           # float32; transposed views without a copy
    C = gen9blas.gemm_batched(A3, B3)               # batch: A3 (count, M, K), B3 (count, K, N) -> (count, M, N)
    C = gen9blas.symm(A, B, side="L", uplo="L")     # A symmetric
    C = gen9blas.syrk(A, uplo="L")                  # triangle uplo of A·Aᵀ
    C = gen9blas.syr2k(A, B, uplo="L")              # triangle uplo of A·Bᵀ + B·Aᵀ
    B = gen9blas.trmm(A, B, side="L", uplo="L")     # op(A)·B, A triangular (in place, on a copy of B)
    X = gen9blas.trsm(A, B, side="L", uplo="L")     # solve op(A)·X = B (float32 / float16)
    X = gen9blas.empty((n, n), np.float16)          # USM memory; plain arrays are zero-copy too (userptr, docs/no-neo.md 6g)
    C = gen9blas.sgemm(A[:, :k], B[:k].T)           # slices and .T views at levels 1-3 without a copy (ld from NumPy strides)
    gen9blas.axpy(0.5, x, y); d = gen9blas.dot(x, y)     # level 1 (also scal, copy, swap, rot, asum, nrm2, iamax)
    y = gen9blas.gemv(A, x); gen9blas.ger(1.0, x, y, A)  # level 2

The type (float16 / float32) follows the data type. Relative error (to the sum of |a·b|): hgemm ~1e-4, accurate ~1e-5, sgemm ~1e-8.
"""
import ctypes
import os
import weakref

import numpy as np

_HERE = os.path.dirname(os.path.realpath(__file__))
_lib = ctypes.CDLL(os.environ.get("GEN9BLAS_LIB", os.path.join(_HERE, "libgen9blas.so")))   # GEN9BLAS_LIB: another build of the library
_i, _f, _p, _l, _d = ctypes.c_int, ctypes.c_float, ctypes.c_void_p, ctypes.c_long, ctypes.c_double
_lib.gen9blas_init.argtypes = [ctypes.c_char_p]
_lib.gen9blas_gemm.argtypes = [_i, _i, _i, _i, _i, _i, _d, _p, _i, _p, _i, _d, _p, _i]
_lib.gen9blas_gemm_batched.argtypes = [_i, _i, _i, _i, _i, _i, _d, _p, _i, _l, _p, _i, _l, _d, _p, _i, _l, _i]
_lib.gen9blas_symm.argtypes = [_i, _i, _i, _i, _i, _d, _p, _i, _p, _i, _d, _p, _i]
_lib.gen9blas_syrk.argtypes = [_i, _i, _i, _i, _i, _d, _p, _i, _d, _p, _i]
_lib.gen9blas_syr2k.argtypes = [_i, _i, _i, _i, _i, _d, _p, _i, _p, _i, _d, _p, _i]
_lib.gen9blas_trmm.argtypes = [_i, _i, _i, _i, _i, _i, _i, _d, _p, _i, _p, _i]
_lib.gen9blas_trsm.argtypes = [_i, _i, _i, _i, _i, _i, _i, _d, _p, _i, _p, _i]
_lib.gen9blas_last_time_ms.restype = ctypes.c_double
_lib.gen9blas_alloc.argtypes = [ctypes.c_size_t]
_lib.gen9blas_alloc.restype = ctypes.c_void_p
_lib.gen9blas_free.argtypes = [ctypes.c_void_p]
_lib.gen9blas_error.restype = ctypes.c_char_p
GEN9BLAS_H, GEN9BLAS_S, GEN9BLAS_HS, GEN9BLAS_D = 0, 1, 2, 3
_SIDE = {"L": 0, "R": 1}
_TRI = {"L": 0, "U": 1}
_ready = False


def _init():
    global _ready
    if not _ready:
        if _lib.gen9blas_init(os.path.join(_HERE, "..", "bench").encode()) != 0:
            raise RuntimeError(_lib.gen9blas_error().decode())
        _ready = True


def _check(w):
    if w != 0:
        raise RuntimeError(_lib.gen9blas_error().decode())


# USM memory pool: a driver allocation costs ~0.5 ms CPU per MB (pages are prepared immediately; 64 MB = 33 ms),
# so released blocks wait in the pool for the next gen9blas.empty of the same (rounded) size.
_pool = {}
_pool_bytes = 0
POOL_MAX = 1 << 30


def _block_size(n):
    return max(4096, (n + 65535) // 65536 * 65536 if n > 65536 else (n + 4095) // 4096 * 4096)


def _release(p, n):
    global _pool_bytes
    if _pool_bytes + n <= POOL_MAX:
        _pool.setdefault(n, []).append(p); _pool_bytes += n
    else:
        _lib.gen9blas_free(p)


def free_pool():
    """Returns the memory blocks waiting in the pool to the driver (arrays in use stay)."""
    global _pool_bytes
    for blocks in _pool.values():
        for p in blocks:
            _lib.gen9blas_free(p)
    _pool.clear(); _pool_bytes = 0


def empty(shape, dtype=np.float16):
    """C-contiguous array in USM host memory: the GPU reads and writes it without a copy (also its views and slices).
    Plain NumPy arrays are zero-copy too (userptr: the first use pins the pages, about the cost of a copy); gen9blas.empty
    saves that first pinning. When all views of the array are gone, the block
    returns to the pool (up to POOL_MAX bytes) and is reused by later empty calls; free_pool() releases the pool."""
    global _pool_bytes
    _init()
    dtype = np.dtype(dtype)
    n = int(np.prod(shape)) * dtype.itemsize
    nb = _block_size(n)
    if _pool.get(nb):
        p = _pool[nb].pop(); _pool_bytes -= nb
    else:
        p = _lib.gen9blas_alloc(nb)
        if not p:
            raise MemoryError(_lib.gen9blas_error().decode())
    raw = (ctypes.c_char * nb).from_address(p)
    f = weakref.finalize(raw, _release, p, nb)
    f.atexit = False   # at program exit the driver frees all memory anyway
    return np.frombuffer(raw, dtype=np.uint8, count=n).view(dtype).reshape(shape)


def available(dt, accurate=False):
    """Whether the type can be used on this GPU (Gen8 Cherryview: no FP16 for now, docs/gen8.md)."""
    _init()
    return bool(_lib.gen9blas_available(_kind(dt, accurate)))


def _kind(dt, accurate):
    if np.dtype(dt) == np.float32:
        return GEN9BLAS_S
    if np.dtype(dt) == np.float64:
        return GEN9BLAS_D
    if np.dtype(dt) == np.float16:
        return GEN9BLAS_HS if accurate else GEN9BLAS_H
    raise TypeError("supported types: float16, float32, float64")


def _m3(X, dt, write=False):
    """Level 3 operand: (array, pointer, ld, trans) for a 2-D matrix. Strided rows (slices, e.g. A[:, :k],
    A[::2]) and transposed views are zero-copy (ld from NumPy strides); others are copied by the CPU (write: error)."""
    X, p, _, _, ld, tr = _mat(X, dt, write)
    return X, p, ld, tr


def _c(X, dt):
    return np.ascontiguousarray(X, dtype=dt)


def _copy_of(B, dt):
    """Copy of B (in-place result) in USM memory; contiguous B of the given type is copied by the GPU (no CPU work for B from gen9blas.empty)."""
    b = empty(B.shape, dt)
    if B.dtype == dt and B.flags.c_contiguous and B.size:
        _check(_lib.gen9blas_copy(_kind(dt, False), B.size, B.ctypes.data, 1, b.ctypes.data, 1))
    else:
        b[...] = B
    return b


def _output(C, shape, dt):
    """(C, given, ldc): new C from gen9blas.empty or the supplied C with contiguous rows (any row stride)."""
    if C is None:
        return empty(shape, dt), False, shape[-1]
    if C.shape != tuple(shape) or C.dtype != dt or C.strides[-1] != C.itemsize:
        raise ValueError(f"C must have shape {tuple(shape)}, type {np.dtype(dt).name} and contiguous rows")
    if C.ndim == 2:
        return C, True, _mat(C, dt, True)[4]
    if not C.flags.c_contiguous:
        raise ValueError("C (batch) must be C-contiguous")
    return C, True, shape[-1]


def gemm(A, B, alpha=1.0, beta=0.0, C=None, accurate=False):
    """C = alpha·A·B + beta·C (float16 or float32 according to A)."""
    _init()
    dt = A.dtype if A.dtype in (np.float16, np.float32, np.float64) else np.float32
    (M, K), (K2, N) = A.shape, B.shape
    if K != K2:
        raise ValueError(f"incompatible dimensions: {A.shape} · {B.shape}")
    a, pa, lda, ta = _m3(A, dt)
    b, pb, ldb, tb = _m3(B, dt)
    C, given, ldc = _output(C, (M, N), dt)
    _check(_lib.gen9blas_gemm(_kind(dt, accurate), ta, tb, M, N, K, alpha, pa, lda, pb, ldb,
                           beta if given else 0.0, C.ctypes.data, ldc))
    return C


def hgemm(A, B, alpha=1.0, beta=0.0, C=None, accurate=False):
    return gemm(A if A.dtype == np.float16 else A.astype(np.float16),
                B if B.dtype == np.float16 else B.astype(np.float16), alpha, beta, C, accurate)


def sgemm(A, B, alpha=1.0, beta=0.0, C=None):
    return gemm(A if A.dtype == np.float32 else A.astype(np.float32),
                B if B.dtype == np.float32 else B.astype(np.float32), alpha, beta, C)


def gemm_batched(A, B, alpha=1.0, beta=0.0, C=None, accurate=False):
    """Batch of multiplications: A (count, M, K), B (count, K, N) -> C (count, M, N), one kernel launch."""
    _init()
    dt = A.dtype if A.dtype in (np.float16, np.float32, np.float64) else np.float32
    a, b = _c(A, dt), _c(B, dt)
    (cnt, M, K), (cnt2, K2, N) = a.shape, b.shape
    if K != K2 or cnt != cnt2:
        raise ValueError(f"incompatible dimensions: {a.shape} · {b.shape}")
    C, given, _ = _output(C, (cnt, M, N), dt)
    _check(_lib.gen9blas_gemm_batched(_kind(dt, accurate), 0, 0, M, N, K, alpha, a.ctypes.data, K, M * K,
                                   b.ctypes.data, N, K * N, beta if given else 0.0, C.ctypes.data, N, M * N, cnt))
    return C


def symm(A, B, side="L", uplo="L", alpha=1.0, beta=0.0, C=None, accurate=False):
    _init()
    dt = B.dtype
    a, pa, n, lda, _, uplo = _square(A, dt, uplo)   # A.T view: the same matrix, triangle swapped
    b, pb, ldb, tb = _m3(B, dt)
    if tb:
        b, pb, ldb, tb = _m3(_c(B, dt), dt)
    M, N = B.shape
    C, given, ldc = _output(C, (M, N), dt)
    _check(_lib.gen9blas_symm(_kind(dt, accurate), _SIDE[side], _TRI[uplo], M, N, alpha, pa, lda,
                           pb, ldb, beta if given else 0.0, C.ctypes.data, ldc))
    return C


def syrk(A, uplo="L", trans=False, alpha=1.0, beta=0.0, C=None, accurate=False):
    _init()
    dt = A.dtype
    N, K = (A.shape[1], A.shape[0]) if trans else A.shape
    a, pa, lda, ta = _m3(A, dt)   # A.T view: trans flipped
    C, given, ldc = _output(C, (N, N), dt)
    if not given:
        C[...] = 0
    _check(_lib.gen9blas_syrk(_kind(dt, accurate), _TRI[uplo], int(trans) ^ ta, N, K, alpha, pa, lda,
                           beta if given else 0.0, C.ctypes.data, ldc))
    return C


def syr2k(A, B, uplo="L", trans=False, alpha=1.0, beta=0.0, C=None, accurate=False):
    _init()
    dt = A.dtype
    N, K = (A.shape[1], A.shape[0]) if trans else A.shape
    a, pa, lda, ta = _m3(A, dt)
    b, pb, ldb, tb = _m3(B, dt)
    if ta != tb:   # one trans flag for both: copy operand B to the layout of A
        b, pb, ldb, tb = _m3(np.ascontiguousarray(B.T, dtype=dt).T if ta else _c(B, dt), dt)
    C, given, ldc = _output(C, (N, N), dt)
    if not given:
        C[...] = 0
    _check(_lib.gen9blas_syr2k(_kind(dt, accurate), _TRI[uplo], int(trans) ^ ta, N, K, alpha, pa, lda,
                            pb, ldb, beta if given else 0.0, C.ctypes.data, ldc))
    return C


def trmm(A, B, side="L", uplo="L", transA=False, diag=False, alpha=1.0, accurate=False):
    """Returns alpha·op(A)·B (side="L") or alpha·B·op(A) (side="R"); A triangular. B is not modified."""
    _init()
    dt = B.dtype
    a, pa, _, lda, ta, uplo = _square(A, dt, uplo)   # A.T view: transA and triangle flipped
    b = _copy_of(B, dt)
    M, N = b.shape
    _check(_lib.gen9blas_trmm(_kind(dt, accurate), _SIDE[side], _TRI[uplo], int(transA) ^ ta, int(diag), M, N,
                           alpha, pa, lda, b.ctypes.data, N))
    return b


def trsm(A, B, side="L", uplo="L", transA=False, diag=False, alpha=1.0, accurate=False):
    """Returns X: op(A)·X = alpha·B (side="L") or X·op(A) = alpha·B (side="R").
    float32: FP32 accuracy; float16: GEMM in FP16 (accurate=True: chunked accumulation to FP32), block inverses in FP64 (on the GPU)."""
    _init()
    dt = B.dtype if B.dtype in (np.float16, np.float32, np.float64) else np.float32
    a, pa, _, lda, ta, uplo = _square(A, dt, uplo)   # A.T view: transA and triangle flipped
    b = _copy_of(B, dt)
    M, N = b.shape
    _check(_lib.gen9blas_trsm(_kind(dt, accurate), _SIDE[side], _TRI[uplo], int(transA) ^ ta, int(diag), M, N,
                           alpha, pa, lda, b.ctypes.data, N))
    return b


def last_time_ms():
    return _lib.gen9blas_last_time_ms()


def last_zero_copy():
    return _lib.gen9blas_last_zero_copy()


# ---------------- levels 1-2 (float32 / float16; FP16 computed in FP32, reductions in FP64) ----------------
# Arrays (and their views, also with a stride and a negative stride) are zero-copy: gen9blas.empty (USM) or plain (userptr).
# "In place" operations require float32/float16 in the result array; inputs of another type are converted.
_lib.gen9blas_axpy.argtypes = [_i, _i, _d, _p, _i, _p, _i]
_lib.gen9blas_scal.argtypes = [_i, _i, _d, _p, _i]
_lib.gen9blas_copy.argtypes = [_i, _i, _p, _i, _p, _i]
_lib.gen9blas_swap.argtypes = [_i, _i, _p, _i, _p, _i]
_lib.gen9blas_rot.argtypes = [_i, _i, _p, _i, _p, _i, _d, _d]
_lib.gen9blas_dot.argtypes = [_i, _i, _p, _i, _p, _i, ctypes.POINTER(_d)]
_lib.gen9blas_asum.argtypes = [_i, _i, _p, _i, ctypes.POINTER(_d)]
_lib.gen9blas_nrm2.argtypes = [_i, _i, _p, _i, ctypes.POINTER(_d)]
_lib.gen9blas_iamax.argtypes = [_i, _i, _p, _i, ctypes.POINTER(_l)]
_lib.gen9blas_gemv.argtypes = [_i, _i, _i, _i, _d, _p, _i, _p, _i, _d, _p, _i]
_lib.gen9blas_ger.argtypes = [_i, _i, _i, _d, _p, _i, _p, _i, _p, _i]


def _dtype_of(*X):
    """Compute type from the type of the first array: float16, float64 or (others) float32."""
    return np.float16 if X[0].dtype == np.float16 else np.float64 if X[0].dtype == np.float64 else np.float32


def _vec(x, dt, write=False):
    """(array, pointer to the lowest address, n, inc) for a 1-D vector; inc < 0 as in BLAS."""
    if x.ndim != 1:
        raise ValueError("expected a 1-D vector")
    if x.dtype != dt or x.strides[0] % x.itemsize or x.strides[0] == 0:
        if write:
            raise TypeError(f"the result must be a {np.dtype(dt).name} vector with a stride that is a multiple of the element size")
        x = np.ascontiguousarray(x, dtype=dt)
    n, inc = x.shape[0], x.strides[0] // x.itemsize
    return x, x.ctypes.data + (min(inc, 0) * (n - 1) * x.itemsize if n else 0), n, inc


def _mat(A, dt, write=False):
    """(array, pointer, rows, columns, ld, trans) for a row-major matrix with ld; a transposed view is zero-copy."""
    if A.ndim != 2:
        raise ValueError("expected a 2-D matrix")
    e = A.itemsize
    if A.dtype == dt and A.strides[1] == e and A.strides[0] >= A.shape[1] * e and A.strides[0] % e == 0:
        return A, A.ctypes.data, A.shape[0], A.shape[1], A.strides[0] // e, 0
    if not write and A.dtype == dt and A.strides[0] == e and A.strides[1] >= A.shape[0] * e and A.strides[1] % e == 0:
        return A, A.ctypes.data, A.shape[1], A.shape[0], A.strides[1] // e, 1
    if write:
        raise TypeError(f"the result must be a {np.dtype(dt).name} matrix with contiguous rows")
    A = np.ascontiguousarray(A, dtype=dt)
    return A, A.ctypes.data, A.shape[0], A.shape[1], A.shape[1], 0


def _kind_basic(dt):
    return GEN9BLAS_S if dt == np.float32 else GEN9BLAS_D if dt == np.float64 else GEN9BLAS_H


def axpy(a, x, y):
    """y = a·x + y (in place); returns y."""
    _init(); t = _dtype_of(y); y, py, n, iy = _vec(y, t, True); x, px, nx, ix = _vec(x, t)
    if nx != n:
        raise ValueError("lengths differ")
    _check(_lib.gen9blas_axpy(_kind_basic(t), n, a, px, ix, py, iy)); return y


def scal(a, x):
    """x = a·x (in place); returns x."""
    _init(); t = _dtype_of(x); x, px, n, ix = _vec(x, t, True)
    _check(_lib.gen9blas_scal(_kind_basic(t), n, a, px, ix)); return x


def copy(x, y=None):
    """y = x; without y: a new array (gen9blas.empty) of the type of x."""
    _init(); t = _dtype_of(y if y is not None else x)
    if y is None:
        y = empty(x.shape, t)
    y, py, n, iy = _vec(y, t, True); x, px, nx, ix = _vec(x, t)
    if nx != n:
        raise ValueError("lengths differ")
    _check(_lib.gen9blas_copy(_kind_basic(t), n, px, ix, py, iy)); return y


def swap(x, y):
    _init(); t = _dtype_of(x); x, px, n, ix = _vec(x, t, True); y, py, ny, iy = _vec(y, t, True)
    if ny != n:
        raise ValueError("lengths differ")
    _check(_lib.gen9blas_swap(_kind_basic(t), n, px, ix, py, iy)); return x, y


def rot(x, y, c, s):
    """x, y = c·x + s·y, c·y − s·x (in place)."""
    _init(); t = _dtype_of(x); x, px, n, ix = _vec(x, t, True); y, py, ny, iy = _vec(y, t, True)
    if ny != n:
        raise ValueError("lengths differ")
    _check(_lib.gen9blas_rot(_kind_basic(t), n, px, ix, py, iy, c, s)); return x, y


def dot(x, y):
    _init(); t = _dtype_of(x); x, px, n, ix = _vec(x, t); y, py, ny, iy = _vec(y, t)
    if ny != n:
        raise ValueError("lengths differ")
    w = _d(); _check(_lib.gen9blas_dot(_kind_basic(t), n, px, ix, py, iy, ctypes.byref(w))); return w.value


def asum(x):
    _init(); t = _dtype_of(x); x, px, n, ix = _vec(x, t)
    w = _d(); _check(_lib.gen9blas_asum(_kind_basic(t), n, px, ix, ctypes.byref(w))); return w.value


def nrm2(x):
    _init(); t = _dtype_of(x); x, px, n, ix = _vec(x, t)
    w = _d(); _check(_lib.gen9blas_nrm2(_kind_basic(t), n, px, ix, ctypes.byref(w))); return w.value


def iamax(x):
    """Index (0-based) of the first largest |x|; -1 for an empty vector."""
    _init(); t = _dtype_of(x); x, px, n, ix = _vec(x, t)
    w = _l(); _check(_lib.gen9blas_iamax(_kind_basic(t), n, px, ix, ctypes.byref(w))); return w.value


def gemv(A, x, alpha=1.0, beta=0.0, y=None, trans=False):
    """y = alpha·op(A)·x + beta·y; without y: a new array. An A.T view is zero-copy."""
    _init(); t = _dtype_of(A); A, pa, M, N, lda, tr = _mat(A, t); tr ^= int(trans)
    x, px, nx, ix = _vec(x, t)
    ny = N if tr else M
    if nx != (M if tr else N):
        raise ValueError("length of x does not match A")
    if y is None:
        y, beta = empty((ny,), t), 0.0
    y, py, n, iy = _vec(y, t, True)
    if n != ny:
        raise ValueError("length of y does not match A")
    _check(_lib.gen9blas_gemv(_kind_basic(t), tr, M, N, alpha, pa, lda, px, ix, beta, py, iy)); return y


def ger(alpha, x, y, A):
    """A = alpha·x·yᵀ + A (in place); returns A."""
    _init(); t = _dtype_of(A); A, pa, M, N, lda, _ = _mat(A, t, True)
    x, px, nx, ix = _vec(x, t); y, py, ny, iy = _vec(y, t)
    if nx != M or ny != N:
        raise ValueError("lengths of x, y do not match A")
    _check(_lib.gen9blas_ger(_kind_basic(t), M, N, alpha, px, ix, py, iy, pa, lda)); return A



_lib.gen9blas_symv.argtypes = [_i, _i, _i, _d, _p, _i, _p, _i, _d, _p, _i]
_lib.gen9blas_trmv.argtypes = [_i, _i, _i, _i, _i, _p, _i, _p, _i]
_lib.gen9blas_trsv.argtypes = [_i, _i, _i, _i, _i, _p, _i, _p, _i]
_lib.gen9blas_syr.argtypes = [_i, _i, _i, _d, _p, _i, _p, _i]
_lib.gen9blas_syr2.argtypes = [_i, _i, _i, _d, _p, _i, _p, _i, _p, _i]
_FLIP = {"L": "U", "U": "L"}


def _square(A, t, uplo):
    """(array, pointer, n, ld, transposed, uplo in memory): an A.T view is zero-copy, triangle swapped."""
    A, pa, M, N, lda, tr = _mat(A, t)
    if M != N:
        raise ValueError("expected a square matrix")
    return A, pa, N, lda, tr, _FLIP[uplo] if tr else uplo


def symv(A, x, uplo="L", alpha=1.0, beta=0.0, y=None):
    """y = alpha·A·x + beta·y, A symmetric (triangle uplo is used); without y: a new array."""
    _init(); t = _dtype_of(A); A, pa, n, lda, _, u = _square(A, t, uplo)
    x, px, nx, ix = _vec(x, t)
    if nx != n:
        raise ValueError("length of x does not match A")
    if y is None:
        y, beta = empty((n,), t), 0.0
    y, py, ny, iy = _vec(y, t, True)
    if ny != n:
        raise ValueError("length of y does not match A")
    _check(_lib.gen9blas_symv(_kind_basic(t), _TRI[u], n, alpha, pa, lda, px, ix, beta, py, iy)); return y


def trmv(A, x, uplo="L", trans=False, diag=False):
    """x = op(A)·x (in place), A triangular; diag=True: unit diagonal. Returns x."""
    _init(); t = _dtype_of(x); A, pa, n, lda, tr, u = _square(A, t, uplo)
    x, px, nx, ix = _vec(x, t, True)
    if nx != n:
        raise ValueError("length of x does not match A")
    _check(_lib.gen9blas_trmv(_kind_basic(t), _TRI[u], int(trans) ^ tr, int(diag), n, pa, lda, px, ix)); return x


def trsv(A, x, uplo="L", trans=False, diag=False):
    """Solves op(A)·z = x and stores z in x (in place); A triangular. Work vector in FP32. Returns x."""
    _init(); t = _dtype_of(x); A, pa, n, lda, tr, u = _square(A, t, uplo)
    x, px, nx, ix = _vec(x, t, True)
    if nx != n:
        raise ValueError("length of x does not match A")
    _check(_lib.gen9blas_trsv(_kind_basic(t), _TRI[u], int(trans) ^ tr, int(diag), n, pa, lda, px, ix)); return x


def syr(alpha, x, A, uplo="L"):
    """A = alpha·x·xᵀ + A on the triangle uplo (in place); returns A."""
    _init(); t = _dtype_of(A); A, pa, M, N, lda, _ = _mat(A, t, True)
    x, px, n, ix = _vec(x, t)
    if not M == N == n:
        raise ValueError("dimensions do not match")
    _check(_lib.gen9blas_syr(_kind_basic(t), _TRI[uplo], n, alpha, px, ix, pa, lda)); return A


def syr2(alpha, x, y, A, uplo="L"):
    """A = alpha·(x·yᵀ + y·xᵀ) + A on the triangle uplo (in place); returns A."""
    _init(); t = _dtype_of(A); A, pa, M, N, lda, _ = _mat(A, t, True)
    x, px, n, ix = _vec(x, t); y, py, ny, iy = _vec(y, t)
    if not M == N == n == ny:
        raise ValueError("dimensions do not match")
    _check(_lib.gen9blas_syr2(_kind_basic(t), _TRI[uplo], n, alpha, px, ix, py, iy, pa, lda)); return A


_lib.gen9blas_rotm.argtypes = [_i, _i, _p, _i, _p, _i, ctypes.POINTER(_d)]
_fp = ctypes.POINTER(_f)
_lib.gen9blas_rotg.argtypes = [_fp, _fp, _fp, _fp]
_lib.gen9blas_rotmg.argtypes = [_fp, _fp, _fp, _f, _fp]


def rotm(x, y, param):
    """x, y = h11·x + h12·y, h21·x + h22·y (in place); param: 5 numbers as in BLAS (flag, h11, h21, h12, h22)."""
    _init(); t = _dtype_of(x); x, px, n, ix = _vec(x, t, True); y, py, ny, iy = _vec(y, t, True)
    if ny != n:
        raise ValueError("lengths differ")
    pr = (_d * 5)(*[float(v) for v in param])
    _check(_lib.gen9blas_rotm(_kind_basic(t), n, px, ix, py, iy, pr)); return x, y


def rotg(a, b):
    """(r, z, c, s): Givens rotation zeroing b (CPU)."""
    w = [_f(a), _f(b), _f(), _f()]
    _lib.gen9blas_rotg(*[ctypes.byref(v) for v in w]); return tuple(v.value for v in w)


def rotmg(d1, d2, x1, y1):
    """(d1, d2, x1, param): modified Givens rotation (CPU); param as in BLAS (5 numbers)."""
    w = [_f(d1), _f(d2), _f(x1)]; pr = (_f * 5)()
    _lib.gen9blas_rotmg(*[ctypes.byref(v) for v in w], _f(y1), pr)
    return w[0].value, w[1].value, w[2].value, np.array(pr[:], np.float32)


# ---------------- level 2: banded and packed matrices (row layout as in CBLAS RowMajor) ----------------
# Band: array (rows, KL+KU+1), A[i, j] in row i, column KL + j − i (symmetric/triangular: upper KL = 0,
# KU = K; lower KL = K, KU = 0). Packed: vector n(n+1)/2 -- consecutive rows of the triangle (upper: j = i..n−1, lower: j = 0..i).
for _n, _t in {"gen9blas_gbmv": [_i] * 6 + [_d, _p, _i, _p, _i, _d, _p, _i], "gen9blas_sbmv": [_i] * 4 + [_d, _p, _i, _p, _i, _d, _p, _i],
               "gen9blas_tbmv": [_i] * 6 + [_p, _i, _p, _i], "gen9blas_tbsv": [_i] * 6 + [_p, _i, _p, _i],
               "gen9blas_spmv": [_i] * 3 + [_d, _p, _p, _i, _d, _p, _i], "gen9blas_tpmv": [_i] * 5 + [_p, _p, _i],
               "gen9blas_tpsv": [_i] * 5 + [_p, _p, _i], "gen9blas_spr": [_i] * 3 + [_d, _p, _i, _p],
               "gen9blas_spr2": [_i] * 3 + [_d, _p, _i, _p, _i, _p]}.items():
    getattr(_lib, _n).argtypes = _t


def _band(A, t, width, write=False):
    A, pa, M, W, lda, tr = _mat(A, t, write)
    if tr or W != width:
        raise ValueError(f"band: expected a row-major array with {width} columns")
    return A, pa, M, lda


def _packed(Ap, t, n, write=False):
    Ap, pp, m, inc = _vec(Ap, t, write)
    if inc != 1 or m != n * (n + 1) // 2:
        raise ValueError(f"packed: expected a contiguous vector of length {n * (n + 1) // 2}")
    return Ap, pp


def gbmv(A, x, M, N, KL, KU, alpha=1.0, beta=0.0, y=None, trans=False):
    """y = alpha·op(A)·x + beta·y, A: M×N banded (array (M, KL+KU+1))."""
    _init(); t = _dtype_of(A); A, pa, m, lda = _band(A, t, KL + KU + 1)
    if m != M:
        raise ValueError("number of band rows != M")
    x, px, nx, ix = _vec(x, t)
    if y is None:
        y, beta = empty((N if trans else M,), t), 0.0
    y, py, ny, iy = _vec(y, t, True)
    if nx != (M if trans else N) or ny != (N if trans else M):
        raise ValueError("lengths of x, y")
    _check(_lib.gen9blas_gbmv(_kind_basic(t), int(trans), M, N, KL, KU, alpha, pa, lda, px, ix, beta, py, iy)); return y


def sbmv(A, x, K, uplo="L", alpha=1.0, beta=0.0, y=None):
    """y = alpha·A·x + beta·y, A symmetric banded (array (n, K+1), triangle uplo)."""
    _init(); t = _dtype_of(A); A, pa, n, lda = _band(A, t, K + 1)
    x, px, nx, ix = _vec(x, t)
    if y is None:
        y, beta = empty((n,), t), 0.0
    y, py, ny, iy = _vec(y, t, True)
    if nx != n or ny != n:
        raise ValueError("lengths of x, y")
    _check(_lib.gen9blas_sbmv(_kind_basic(t), _TRI[uplo], n, K, alpha, pa, lda, px, ix, beta, py, iy)); return y


def tbmv(A, x, K, uplo="L", trans=False, diag=False):
    """x = op(A)·x (in place), A triangular banded (array (n, K+1))."""
    _init(); t = _dtype_of(x); A, pa, n, lda = _band(A, t, K + 1); x, px, nx, ix = _vec(x, t, True)
    _check(_lib.gen9blas_tbmv(_kind_basic(t), _TRI[uplo], int(trans), int(diag), n, K, pa, lda, px, ix)); return x


def tbsv(A, x, K, uplo="L", trans=False, diag=False):
    """Solves op(A)·z = x (in place), A triangular banded (array (n, K+1))."""
    _init(); t = _dtype_of(x); A, pa, n, lda = _band(A, t, K + 1); x, px, nx, ix = _vec(x, t, True)
    _check(_lib.gen9blas_tbsv(_kind_basic(t), _TRI[uplo], int(trans), int(diag), n, K, pa, lda, px, ix)); return x


def spmv(Ap, x, uplo="L", alpha=1.0, beta=0.0, y=None):
    """y = alpha·A·x + beta·y, A symmetric packed (vector n(n+1)/2)."""
    _init(); t = _dtype_of(Ap); x, px, n, ix = _vec(x, t); Ap, pp = _packed(Ap, t, n)
    if y is None:
        y, beta = empty((n,), t), 0.0
    y, py, ny, iy = _vec(y, t, True)
    _check(_lib.gen9blas_spmv(_kind_basic(t), _TRI[uplo], n, alpha, pp, px, ix, beta, py, iy)); return y


def tpmv(Ap, x, uplo="L", trans=False, diag=False):
    """x = op(A)·x (in place), A triangular packed."""
    _init(); t = _dtype_of(x); x, px, n, ix = _vec(x, t, True); Ap, pp = _packed(Ap, t, n)
    _check(_lib.gen9blas_tpmv(_kind_basic(t), _TRI[uplo], int(trans), int(diag), n, pp, px, ix)); return x


def tpsv(Ap, x, uplo="L", trans=False, diag=False):
    """Solves op(A)·z = x (in place), A triangular packed."""
    _init(); t = _dtype_of(x); x, px, n, ix = _vec(x, t, True); Ap, pp = _packed(Ap, t, n)
    _check(_lib.gen9blas_tpsv(_kind_basic(t), _TRI[uplo], int(trans), int(diag), n, pp, px, ix)); return x


def spr(alpha, x, Ap, uplo="L"):
    """A = alpha·x·xᵀ + A (packed, in place)."""
    _init(); t = _dtype_of(Ap); x, px, n, ix = _vec(x, t); Ap, pp = _packed(Ap, t, n, True)
    _check(_lib.gen9blas_spr(_kind_basic(t), _TRI[uplo], n, alpha, px, ix, pp)); return Ap


def spr2(alpha, x, y, Ap, uplo="L"):
    """A = alpha·(x·yᵀ + y·xᵀ) + A (packed, in place)."""
    _init(); t = _dtype_of(Ap); x, px, n, ix = _vec(x, t); y, py, ny, iy = _vec(y, t); Ap, pp = _packed(Ap, t, n, True)
    _check(_lib.gen9blas_spr2(_kind_basic(t), _TRI[uplo], n, alpha, px, ix, py, iy, pp)); return Ap
