"""Test of the BLAS/CBLAS layer (libgen9blas_blas.so) against NumPy (float64).

Run: python3 lib/test_blas.py
Forces the GPU (GEN9BLAS_BLAS=gpu) and checks in the log (GEN9BLAS_BLAS_INFO) that every S function went to the GPU. Tests: CBLAS
in row-major and column-major layout, the Fortran interface (column-major), all trans/uplo/side/diag combinations,
lda larger than the minimum, vector strides (also negative), level 1 on USM memory (gen9blas.empty from the same library),
functions forwarded to BLAS on the CPU (D, Z). Criterion: error relative to the formula scale ≤ 1e-5 (TRSV/TRSM: relative ≤ 1e-4).
"""
import ctypes
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.realpath(__file__))
os.environ.setdefault("GEN9BLAS_BLAS", "gpu")
os.environ["GEN9BLAS_BLAS_INFO"] = "1"
os.environ["GEN9BLAS_LIB"] = os.path.join(HERE, "libgen9blas_blas.so")   # gen9blas.empty z tego samego kontekstu OpenCL
import numpy as np  # noqa: E402

sys.path.insert(0, HERE)
import gen9blas  # noqa: E402

B = gen9blas._lib
log = tempfile.TemporaryFile()
stderr_saved = os.dup(2)
os.dup2(log.fileno(), 2)

rng = np.random.default_rng(5)
errors = 0
RM, CM = 101, 102
TR = {False: 111, True: 112}
UP = {"U": 121, "L": 122}
SD = {"L": 141, "R": 142}
DG = {False: 131, True: 132}
f, i, p, d = ctypes.c_float, ctypes.c_int, ctypes.c_void_p, ctypes.c_double
for name, sig in {
    "cblas_sgemm": [i] * 6 + [f, p, i, p, i, f, p, i], "cblas_ssymm": [i] * 5 + [f, p, i, p, i, f, p, i],
    "cblas_ssyrk": [i] * 5 + [f, p, i, f, p, i], "cblas_ssyr2k": [i] * 5 + [f, p, i, p, i, f, p, i],
    "cblas_strmm": [i] * 7 + [f, p, i, p, i], "cblas_strsm": [i] * 7 + [f, p, i, p, i],
    "cblas_sgemv": [i] * 4 + [f, p, i, p, i, f, p, i], "cblas_sger": [i] * 3 + [f, p, i, p, i, p, i],
    "cblas_ssymv": [i] * 3 + [f, p, i, p, i, f, p, i], "cblas_strmv": [i] * 5 + [p, i, p, i],
    "cblas_strsv": [i] * 5 + [p, i, p, i], "cblas_ssyr": [i] * 3 + [f, p, i, p, i], "cblas_ssyr2": [i] * 3 + [f, p, i, p, i, p, i],
    "cblas_saxpy": [i, f, p, i, p, i], "cblas_sscal": [i, f, p, i], "cblas_scopy": [i, p, i, p, i], "cblas_sswap": [i, p, i, p, i],
    "cblas_srot": [i, p, i, p, i, f, f], "cblas_srotm": [i, p, i, p, i, p], "cblas_sdot": [i, p, i, p, i], "cblas_sdsdot": [i, f, p, i, p, i],
    "cblas_dsdot": [i, p, i, p, i], "cblas_sasum": [i, p, i], "cblas_snrm2": [i, p, i], "cblas_isamax": [i, p, i],
    "cblas_dgemm": [i] * 6 + [d, p, i, p, i, d, p, i],
}.items():
    getattr(B, name).argtypes = sig
for name in ("cblas_sdot", "cblas_sdsdot", "cblas_sasum", "cblas_snrm2"):
    getattr(B, name).restype = f
B.cblas_dsdot.restype = d
B.cblas_isamax.restype = ctypes.c_size_t


def check(name, W, ref, scale, tol=1e-5):
    global errors
    e = float(np.max(np.abs(np.asarray(W, np.float64) - ref) / np.maximum(scale, 1e-30)))
    if not e <= tol:
        errors += 1
        print(f"  ERR {name}: {e:.1e}", file=sys.stdout)


def mat(r, c, layout, pad=3, dt=np.float32):
    """(r×c view, pointer, ld) in row-major (RM) or column-major (CM) layout, with ld larger than the minimum."""
    if layout == RM:
        base = rng.uniform(-1, 1, (r, c + pad)).astype(dt); v = base[:, :c]; return v, v.ctypes.data, c + pad
    base = np.asfortranarray(rng.uniform(-1, 1, (r + pad, c)).astype(dt)); v = base[:r, :]; return v, v.ctypes.data, r + pad


def vec(n, step):
    """(view, pointer to the lowest address, BLAS stride): stride < 0 -- element 0 at the highest address."""
    base = rng.uniform(-1, 1, (n - 1) * abs(step) + 1).astype(np.float32)
    v = base[::step] if step > 0 else base[::step]
    return v, base.ctypes.data, step


def op(X, t):
    return X.T if t else X


for layout in (RM, CM):
    lay_name = "row" if layout == RM else "column"
    M, N, K = 67, 45, 38
    for ta in (False, True):
        for tb in (False, True):
            A, pa, lda = mat(*((K, M) if ta else (M, K)), layout)
            Bm, pb, ldb = mat(*((N, K) if tb else (K, N)), layout)
            C, pc, ldc = mat(M, N, layout)
            Ad, Bd, Cd = (X.astype(np.float64) for X in (A, Bm, C))
            B.cblas_sgemm(layout, TR[ta], TR[tb], M, N, K, 1.5, pa, lda, pb, ldb, -0.5, pc, ldc)
            check(f"sgemm {lay_name} {ta}{tb}", C, 1.5 * op(Ad, ta) @ op(Bd, tb) - 0.5 * Cd, 1.5 * abs(op(Ad, ta)) @ abs(op(Bd, tb)) + 0.5 * abs(Cd))
    for side in "LR":
        n = M if side == "L" else N
        for uplo in "UL":
            A, pa, lda = mat(n, n, layout); Ad = A.astype(np.float64)
            T = np.triu(Ad) if uplo == "U" else np.tril(Ad); S = T + T.T - np.diag(np.diag(Ad))
            Bm, pb, ldb = mat(M, N, layout); C, pc, ldc = mat(M, N, layout); Bd, Cd = Bm.astype(np.float64), C.astype(np.float64)
            B.cblas_ssymm(layout, SD[side], UP[uplo], M, N, 0.7, pa, lda, pb, ldb, 0.3, pc, ldc)
            ref = 0.7 * (S @ Bd if side == "L" else Bd @ S) + 0.3 * Cd
            check(f"ssymm {lay_name} {side}{uplo}", C, ref, 0.7 * (abs(S) @ abs(Bd) if side == "L" else abs(Bd) @ abs(S)) + 0.3 * abs(Cd))
            for t in (False, True):
                for dg in (False, True):
                    Tm = T.copy()
                    if dg:
                        np.fill_diagonal(Tm, 1)
                    Bm, pb, ldb = mat(M, N, layout); Bd = Bm.astype(np.float64)
                    B.cblas_strmm(layout, SD[side], UP[uplo], TR[t], DG[dg], M, N, 2.0, pa, lda, pb, ldb)
                    ref = 2 * (op(Tm, t) @ Bd if side == "L" else Bd @ op(Tm, t))
                    check(f"strmm {lay_name} {side}{uplo}{t}{dg}", Bm, ref, 2 * (abs(op(Tm, t)) @ abs(Bd) if side == "L" else abs(Bd) @ abs(op(Tm, t))))
                    # TRSM: well-conditioned triangular
                    A2, pa2, lda2 = mat(n, n, layout); A2[:] /= n; A2[np.diag_indices(n)] = 1 + rng.uniform(0, 1, n)
                    T2 = (np.triu if uplo == "U" else np.tril)(A2.astype(np.float64))
                    if dg:
                        np.fill_diagonal(T2, 1)
                    Bm, pb, ldb = mat(M, N, layout); Bd = Bm.astype(np.float64)
                    B.cblas_strsm(layout, SD[side], UP[uplo], TR[t], DG[dg], M, N, 1.0, pa2, lda2, pb, ldb)
                    ref = np.linalg.solve(op(T2, t), Bd) if side == "L" else np.linalg.solve(op(T2, t).T, Bd.T).T
                    check(f"strsm {lay_name} {side}{uplo}{t}{dg}", Bm, ref, np.max(abs(ref)), 1e-4)
    for uplo in "UL":
        for t in (False, True):
            A, pa, lda = mat(*((K, N) if t else (N, K)), layout); Bm, pb, ldb = mat(*((K, N) if t else (N, K)), layout)
            C, pc, ldc = mat(N, N, layout); Ad, Bd, Cd = (X.astype(np.float64) for X in (A, Bm, C))
            mask = np.triu(np.ones((N, N), bool)) if uplo == "U" else np.tril(np.ones((N, N), bool))
            C0 = C.copy()
            B.cblas_ssyrk(layout, UP[uplo], TR[t], N, K, 0.5, pa, lda, 2.0, pc, ldc)
            ref = np.where(mask, 0.5 * op(Ad, t) @ op(Ad, t).T + 2 * Cd, Cd)
            check(f"ssyrk {lay_name} {uplo}{t}", C, ref, 0.5 * abs(op(Ad, t)) @ abs(op(Ad, t)).T + 2 * abs(Cd))
            C[:] = C0
            B.cblas_ssyr2k(layout, UP[uplo], TR[t], N, K, 0.5, pa, lda, pb, ldb, 2.0, pc, ldc)
            ref = np.where(mask, 0.5 * (op(Ad, t) @ op(Bd, t).T + op(Bd, t) @ op(Ad, t).T) + 2 * Cd, Cd)
            check(f"ssyr2k {lay_name} {uplo}{t}", C, ref, abs(op(Ad, t)) @ abs(op(Bd, t)).T + 2 * abs(Cd))
    # level 2
    M, N = 300, 210
    for t in (False, True):
        for kx, ky in ((1, 1), (2, -3), (-1, 2)):
            A, pa, lda = mat(M, N, layout); Ad = A.astype(np.float64)
            x, px, ix = vec(M if t else N, kx); y, py, iy = vec(N if t else M, ky)
            X, Y = x.astype(np.float64), y.astype(np.float64)
            B.cblas_sgemv(layout, TR[t], M, N, 0.5, pa, lda, px, ix, 2.0, py, iy)
            check(f"sgemv {lay_name} t={t} strides {kx},{ky}", y, 0.5 * op(Ad, t) @ X + 2 * Y, 0.5 * abs(op(Ad, t)) @ abs(X) + 2 * abs(Y))
    x, px, ix = vec(M, -2); y, py, iy = vec(N, 3); A, pa, lda = mat(M, N, layout)
    X, Y, Ad = x.astype(np.float64), y.astype(np.float64), A.astype(np.float64)
    B.cblas_sger(layout, M, N, -1.5, px, ix, py, iy, pa, lda)
    check(f"sger {lay_name}", A, Ad - 1.5 * np.outer(X, Y), abs(Ad) + 1.5 * np.outer(abs(X), abs(Y)))
    n = 260
    for uplo in "UL":
        A, pa, lda = mat(n, n, layout); Ad = A.astype(np.float64)
        T = np.triu(Ad) if uplo == "U" else np.tril(Ad); S = T + T.T - np.diag(np.diag(Ad))
        mask = np.triu(np.ones((n, n), bool)) if uplo == "U" else np.tril(np.ones((n, n), bool))
        x, px, ix = vec(n, -2); y, py, iy = vec(n, 1); X, Y = x.astype(np.float64), y.astype(np.float64)
        B.cblas_ssymv(layout, UP[uplo], n, 0.5, pa, lda, px, ix, 1.0, py, iy)
        check(f"ssymv {lay_name} {uplo}", y, 0.5 * S @ X + Y, 0.5 * abs(S) @ abs(X) + abs(Y))
        B.cblas_ssyr(layout, UP[uplo], n, 0.5, px, ix, pa, lda)
        check(f"ssyr {lay_name} {uplo}", A, np.where(mask, Ad + 0.5 * np.outer(X, X), Ad), abs(Ad) + 0.5 * np.outer(abs(X), abs(X)))
        A[:] = Ad
        B.cblas_ssyr2(layout, UP[uplo], n, 0.5, px, ix, py, iy, pa, lda)
        Y2 = y.astype(np.float64)
        check(f"ssyr2 {lay_name} {uplo}", A, np.where(mask, Ad + 0.5 * (np.outer(X, Y2) + np.outer(Y2, X)), Ad),
             abs(Ad) + 0.5 * (np.outer(abs(X), abs(Y2)) + np.outer(abs(Y2), abs(X))))
        for t in (False, True):
            for dg in (False, True):
                A3, pa3, lda3 = mat(n, n, layout)   # fresh matrix (A was changed by ssyr/ssyr2)
                Tm = (np.triu if uplo == "U" else np.tril)(A3.astype(np.float64))
                if dg:
                    np.fill_diagonal(Tm, 1)
                x, px, ix = vec(n, 2); X = x.astype(np.float64)
                B.cblas_strmv(layout, UP[uplo], TR[t], DG[dg], n, pa3, lda3, px, ix)
                check(f"strmv {lay_name} {uplo}{t}{dg}", x, op(Tm, t) @ X, abs(op(Tm, t)) @ abs(X))
                A2, pa2, lda2 = mat(n, n, layout); A2[:] /= n; A2[np.diag_indices(n)] = 1 + rng.uniform(0, 1, n)
                T2 = (np.triu if uplo == "U" else np.tril)(A2.astype(np.float64))
                if dg:
                    np.fill_diagonal(T2, 1)
                x, px, ix = vec(n, -1); X = x.astype(np.float64)
                B.cblas_strsv(layout, UP[uplo], TR[t], DG[dg], n, pa2, lda2, px, ix)
                ref = np.linalg.solve(op(T2, t), X)
                check(f"strsv {lay_name} {uplo}{t}{dg}", x, ref, np.max(abs(ref)), 1e-4)

# Fortran (column-major): selected functions, options as characters (also lowercase)
c, ir, fr = ctypes.c_char_p, ctypes.byref, ctypes.byref
I = lambda v: ctypes.byref(ctypes.c_int(v))  # noqa: E731
F = lambda v: ctypes.byref(ctypes.c_float(v))  # noqa: E731
M, N, K = 50, 40, 30
A, pa, lda = mat(M, K, CM); Bm, pb, ldb = mat(K, N, CM); C, pc, ldc = mat(M, N, CM)
Ad, Bd, Cd = (X.astype(np.float64) for X in (A, Bm, C))
B.sgemm_(b"N", b"n", I(M), I(N), I(K), F(1.0), p(pa), I(lda), p(pb), I(ldb), F(1.0), p(pc), I(ldc))
check("sgemm_", C, Ad @ Bd + Cd, abs(Ad) @ abs(Bd) + abs(Cd))
A, pa, lda = mat(M, N, CM); Ad = A.astype(np.float64); x, px, ix = vec(M, -1); y, py, iy = vec(N, 2)
X, Y = x.astype(np.float64), y.astype(np.float64)
B.sgemv_(b"T", I(M), I(N), F(2.0), p(pa), I(lda), p(px), I(ix), F(0.0), p(py), I(iy))
check("sgemv_ T", y, 2 * Ad.T @ X, 2 * abs(Ad.T) @ abs(X))
n = 70
A2, pa2, lda2 = mat(n, n, CM); A2[:] /= n; A2[np.diag_indices(n)] = 2
T2 = np.triu(A2.astype(np.float64)); Bm, pb, ldb = mat(n, 20, CM); Bd = Bm.astype(np.float64)
B.strsm_(b"L", b"u", b"T", b"N", I(n), I(20), F(1.0), p(pa2), I(lda2), p(pb), I(ldb))
check("strsm_", Bm, np.linalg.solve(T2.T, Bd), np.max(abs(Bd)), 1e-4)
C, pc, ldc = mat(n, n, CM); Cd = C.astype(np.float64); A, pa, lda = mat(n, 25, CM); Ad = A.astype(np.float64)
B.ssyrk_(b"L", b"N", I(n), I(25), F(1.0), p(pa), I(lda), F(0.0), p(pc), I(ldc))
check("ssyrk_", np.tril(C), np.tril(Ad @ Ad.T), abs(Ad) @ abs(Ad).T)

# level 1 CBLAS on USM
n = 100003
for kx, ky in ((1, 1), (3, 2)):
    xb = gen9blas.empty((n * kx,), np.float32); yb = gen9blas.empty((n * ky,), np.float32)
    xb[:] = rng.uniform(-1, 1, xb.shape); yb[:] = rng.uniform(-1, 1, yb.shape)
    x, y = xb[::kx], yb[::ky]; X, Y = x.astype(np.float64), y.astype(np.float64)
    px, py = xb.ctypes.data, yb.ctypes.data
    check(f"sdot {kx},{ky}", B.cblas_sdot(n, px, kx, py, ky), X @ Y, abs(X) @ abs(Y))
    check("dsdot", B.cblas_dsdot(n, px, kx, py, ky), X @ Y, abs(X) @ abs(Y), 1e-12)
    check("sdsdot", B.cblas_sdsdot(n, 3.0, px, kx, py, ky), 3 + X @ Y, 3 + abs(X) @ abs(Y))
    check("sasum", B.cblas_sasum(n, px, kx), abs(X).sum(), abs(X).sum())
    check("snrm2", B.cblas_snrm2(n, px, kx), np.linalg.norm(X), np.linalg.norm(X))
    if B.cblas_isamax(n, px, kx) != np.argmax(abs(X)):
        errors += 1; print("  ERR isamax")
    B.cblas_saxpy(n, 0.5, px, kx, py, ky); check("saxpy", y, 0.5 * X + Y, 0.5 * abs(X) + abs(Y))
    y[:] = Y; B.cblas_sscal(n, -2.0, py, ky); check("sscal", y, -2 * Y, 2 * abs(Y))
    y[:] = Y; B.cblas_scopy(n, px, kx, py, ky); check("scopy", y, X, 1, 0)
    y[:] = Y; B.cblas_sswap(n, px, kx, py, ky); check("sswap", np.concatenate([x, y]), np.concatenate([Y, X]), 1, 0)
    x[:], y[:] = X, Y; B.cblas_srot(n, px, kx, py, ky, 0.6, 0.8)
    check("srot", np.concatenate([x, y]), np.concatenate([0.6 * X + 0.8 * Y, 0.6 * Y - 0.8 * X]), 2, 1e-6)
    x[:], y[:] = X, Y; P = np.array([-1.0, 0.5, -0.3, 0.2, 1.5], np.float32)
    B.cblas_srotm(n, px, kx, py, ky, P.ctypes.data)
    check("srotm", np.concatenate([x, y]), np.concatenate([0.5 * X + 0.2 * Y, -0.3 * X + 1.5 * Y]), 3, 1e-6)

# D functions in the same library (details: test_blas_d.py); plain arrays -> copy, GPU forced
A = rng.uniform(-1, 1, (40, 30)); Bd = rng.uniform(-1, 1, (30, 20)); C = np.zeros((40, 20))
B.cblas_dgemm(RM, 111, 111, 40, 20, 30, 1.0, A.ctypes.data, 30, Bd.ctypes.data, 20, 0.0, C.ctypes.data, 20)
check("dgemm", C, A @ Bd, abs(A) @ abs(Bd), 1e-14)
# przekazywane do BLAS na CPU (zespolone)
z = (rng.uniform(-1, 1, 100) + 1j * rng.uniform(-1, 1, 100)).astype(np.complex128); wz = (ctypes.c_double * 2)()
B.cblas_zdotu_sub.argtypes = [i, p, i, p, i, p]
B.cblas_zdotu_sub(100, z.ctypes.data, 1, z.ctypes.data, 1, wz)
check("zdotu (CPU)", np.array([wz[0], wz[1]]), np.array([(z @ z).real, (z @ z).imag]), abs(z) @ abs(z), 1e-14)

# log: every S function on the GPU
os.dup2(stderr_saved, 2)
log.seek(0)
entries = log.read().decode().splitlines()
gpu = {w.split()[1] for w in entries if w.endswith("-> GPU")}
cpu = {w.split()[1] for w in entries if w.endswith("-> CPU")}
expected = {"sgemm", "ssymm", "ssyrk", "ssyr2k", "strmm", "strsm", "sgemv", "sger", "ssymv", "strmv", "strsv", "ssyr",
              "ssyr2", "saxpy", "sscal", "scopy", "sswap", "srot", "srotm", "sdot", "sasum", "snrm2", "isamax", "dgemm"}
if expected - gpu:
    errors += 1; print("  ERR: not on the GPU:", sorted(expected - gpu))
if cpu:
    errors += 1; print("  ERR: on the CPU:", sorted(cpu))
other = [w for w in entries if "->" not in w]
if other:
    print("  stderr:", other[:5])
print(f"calls on the GPU: {sum(w.endswith('-> GPU') for w in entries)}")
print("RESULT:", "all OK" if errors == 0 else f"{errors} errors")
sys.exit(1 if errors else 0)
