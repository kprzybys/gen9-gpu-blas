"""Level 1-2 BLAS test in libgen9blas against NumPy (float64).

Uruchom: python3 lib/test_level12.py
Stage B: SYMV, TRMV, TRSV (relative error of the solution to max|x|: 1e-5 / 2e-3), SYR, SYR2.
Criteria: error relative to the formula scale (e.g. Σ|a·x| + |y| for axpy, |A|·|x| for gemv): float32 ≤ 1e-6, float16 ≤ 2e-3
(rounding of the result to FP16, scale at least the smallest normal FP16 number; computed in FP32). Reductions (FP64 on the GPU): ≤ 1e-6 (float32), ≤ 1e-6 relative to
the exact sum of FP16 inputs. Cases: strides (also negative), unaligned pointers, length tails, operands on
shared pages, transposed views and submatrices, freeing and reallocating registered memory.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import gen9blas  # noqa: E402

rng = np.random.default_rng(11)
errors = 0
TOL = {np.float32: 1e-6, np.float16: 2e-3, np.float64: 1e-13}
SOLVE_TOL = {np.float32: 1e-5, np.float16: 3e-3, np.float64: 1e-12}   # relative error of solutions (TRSV, TBSV, TPSV)


def check(name, W, ref, scale, tol):
    global errors
    e = float(np.max(np.abs(np.asarray(W, np.float64) - ref) / np.maximum(scale, 6.2e-5 if tol > 1e-5 else 1e-30)))
    if not e <= tol:
        errors += 1
        print(f"  ERR {name}: {e:.1e} > {tol:.0e}")


def op_(X, t):
    return X.T if t else X


def vector(n, dt, step=1, shift=0):
    """Vector of length n with stride `step` (negative: reversed view) in an array shifted by `shift` elements."""
    base = rng.uniform(-1, 1, n * abs(step) + shift + 3).astype(dt)
    v = base[shift:shift + n * abs(step):abs(step)]
    return v[::-1] if step < 0 else v


for dt in (np.float32, np.float16, np.float64):
    tol = TOL[dt]
    print(f"== {np.dtype(dt).name}")
    for n in (1, 5, 255, 256, 1000, 100003):
        for kx, ky, px in ((1, 1, 0), (1, 1, 1), (3, -2, 5), (-1, 1, 2)):
            label = f"n={n} strides {kx},{ky} shift {px}"
            x, y = vector(n, dt, kx, px), vector(n, dt, ky, 1)
            X, Y = x.astype(np.float64), y.astype(np.float64)
            gen9blas.axpy(0.75, x, y); check("axpy " + label, y, 0.75 * X + Y, 0.75 * abs(X) + abs(Y), tol)
            y[:] = Y; gen9blas.scal(-1.5, y); check("scal " + label, y, -1.5 * Y, 1.5 * abs(Y), tol)
            y[:] = Y; gen9blas.copy(x, y); check("copy " + label, y, X, 1, 0)
            y[:] = Y; gen9blas.swap(x, y); check("swap " + label, np.concatenate([x, y]), np.concatenate([Y, X]), 1, 0)
            x[:], y[:] = X, Y
            gen9blas.rot(x, y, 0.6, 0.8)
            check("rot " + label, np.concatenate([x, y]), np.concatenate([0.6 * X + 0.8 * Y, 0.6 * Y - 0.8 * X]),
                 np.concatenate([abs(X) + abs(Y)] * 2), tol)
            x[:], y[:] = X, Y
            check("dot " + label, gen9blas.dot(x, y), X @ Y, abs(X) @ abs(Y), 1e-6)
            check("asum " + label, gen9blas.asum(x), abs(X).sum(), abs(X).sum(), 1e-6)
            check("nrm2 " + label, gen9blas.nrm2(x), np.linalg.norm(X), np.linalg.norm(X), 1e-6)
            i = gen9blas.iamax(x)
            if i != int(np.argmax(abs(X))):
                errors += 1; print(f"  ERR iamax {label}: {i} != {np.argmax(abs(X))}")
    # operands on shared pages / the same region
    v = rng.uniform(-1, 1, 3000).astype(dt); V = v.astype(np.float64)
    gen9blas.axpy(2.0, v[:1000], v[1000:2000]); check("axpy shared pages", v[1000:2000], 2 * V[:1000] + V[1000:2000], 3, tol)
    v[:] = V; check("dot x·x", gen9blas.dot(v, v), V @ V, V @ V, 1e-6)
    # GEMV / GER
    for M, N in ((1, 1), (7, 3), (64, 64), (257, 1031), (1000, 2), (3, 5000), (2048, 1536)):
        A = rng.uniform(-1, 1, (M, N)).astype(dt); Ad = A.astype(np.float64)
        x, xt = vector(N, dt), vector(M, dt, -2, 3)
        y0, yt0 = vector(M, dt, 2), vector(N, dt)
        Y0, YT0 = y0.astype(np.float64), yt0.astype(np.float64)
        y = gen9blas.gemv(A, x, alpha=1.5)
        check(f"gemv N {M}x{N}", y, 1.5 * Ad @ x.astype(np.float64), 1.5 * abs(Ad) @ abs(x.astype(np.float64)), tol)
        gen9blas.gemv(A, x, alpha=0.5, beta=-2.0, y=y0)
        check(f"gemv N beta {M}x{N}", y0, 0.5 * Ad @ x.astype(np.float64) - 2 * Y0,
             0.5 * abs(Ad) @ abs(x.astype(np.float64)) + 2 * abs(Y0), tol)
        gen9blas.gemv(A, xt, alpha=0.5, beta=1.0, y=yt0, trans=True)
        Xt = xt.astype(np.float64)
        check(f"gemv T {M}x{N}", yt0, 0.5 * Ad.T @ Xt + YT0, 0.5 * abs(Ad.T) @ abs(Xt) + abs(YT0), tol)
        xv = vector(M, dt); Xv = xv.astype(np.float64)
        yv = gen9blas.gemv(A.T, xv)   # transposed view without a copy
        check(f"gemv A.T {M}x{N}", yv, Ad.T @ Xv, abs(Ad.T) @ abs(Xv), tol)
        xg, yg = vector(M, dt, 1, 1), vector(N, dt, -3, 0)
        Xg, Yg = xg.astype(np.float64), yg.astype(np.float64)
        A2 = A.copy(); gen9blas.ger(-0.5, xg, yg, A2)
        check(f"ger {M}x{N}", A2, Ad - 0.5 * np.outer(Xg, Yg), abs(Ad) + 0.5 * np.outer(abs(Xg), abs(Yg)), tol)
    # submatrix (lda > N) with the vector in the same array
    B = rng.uniform(-1, 1, (300, 400)).astype(dt); Bd = B.astype(np.float64)
    y = gen9blas.gemv(B[10:200, 5:305], B[0, :300])
    check("gemv submatrix", y, Bd[10:200, 5:305] @ Bd[0, :300], abs(Bd[10:200, 5:305]) @ abs(Bd[0, :300]), tol)

# stage B: SYMV, TRMV, TRSV, SYR, SYR2 (both triangles, transposes, unit diagonal, A.T view, strides)
for dt in (np.float32, np.float16, np.float64):
    tol = TOL[dt]
    for n in (1, 5, 63, 64, 65, 200, 1000):
        A = rng.uniform(-1, 1, (n, n)).astype(dt); Ad = A.astype(np.float64)
        for uplo in "LU":
            T = np.tril(Ad) if uplo == "L" else np.triu(Ad)
            S = T + T.T - np.diag(np.diag(Ad))                   # symmetric from the triangle uplo
            x = vector(n, dt, -2, 1); X = x.astype(np.float64)
            y0 = vector(n, dt, 3); Y0 = y0.astype(np.float64)
            gen9blas.symv(A, x, uplo=uplo, alpha=0.5, beta=2.0, y=y0)
            check(f"symv {uplo} n={n}", y0, 0.5 * S @ X + 2 * Y0, 0.5 * abs(S) @ abs(X) + 2 * abs(Y0), tol)
            yv = gen9blas.symv(A.T, x, uplo=uplo)                         # A.T: the other triangle in memory
            T2 = np.tril(Ad.T) if uplo == "L" else np.triu(Ad.T); S2 = T2 + T2.T - np.diag(np.diag(Ad))
            check(f"symv A.T {uplo} n={n}", yv, S2 @ X, abs(S2) @ abs(X), tol)
            for trans in (False, True):
                for diag in (False, True):
                    Tm = T.copy()
                    if diag:
                        np.fill_diagonal(Tm, 1.0)
                    O = Tm.T if trans else Tm
                    label = f"{uplo} trans={int(trans)} diag={int(diag)} n={n}"
                    xx = vector(n, dt, 2); XX = xx.astype(np.float64)
                    gen9blas.trmv(A, xx, uplo=uplo, trans=trans, diag=diag)
                    check("trmv " + label, xx, O @ XX, abs(O) @ abs(XX), tol)
                    # TRSV: well-conditioned matrix -- large diagonal, and with unit diagonal small off-diagonal elements
                    tri = np.tril if uplo == "L" else np.triu
                    if diag:
                        Ab = (Ad / n).astype(dt)
                    else:
                        Ab = A.copy(); np.fill_diagonal(Ab, np.where(np.diag(Ad) < 0, -1, 1) * n / 4 + np.diag(Ad))
                    Tb = tri(Ab.astype(np.float64))
                    if diag:
                        np.fill_diagonal(Tb, 1.0)
                    Ob = Tb.T if trans else Tb
                    b = vector(n, dt, -1); B = b.astype(np.float64)
                    gen9blas.trsv(Ab, b, uplo=uplo, trans=trans, diag=diag)
                    ref = np.linalg.solve(Ob, B)
                    check("trsv " + label, b, ref, np.max(abs(ref)), SOLVE_TOL[dt])
            # SYR, SYR2: triangle uplo changed, the other unchanged
            xs, ys = vector(n, dt, 1, 2), vector(n, dt, -3, 0)
            Xs, Ys = xs.astype(np.float64), ys.astype(np.float64)
            mask = np.tril(np.ones((n, n), bool)) if uplo == "L" else np.triu(np.ones((n, n), bool))
            A1 = A.copy(); gen9blas.syr(-0.5, xs, A1, uplo=uplo)
            check(f"syr {uplo} n={n}", A1, np.where(mask, Ad - 0.5 * np.outer(Xs, Xs), Ad), abs(Ad) + 0.5 * np.outer(abs(Xs), abs(Xs)), tol)
            A2 = A.copy(); gen9blas.syr2(0.25, xs, ys, A2, uplo=uplo)
            check(f"syr2 {uplo} n={n}", A2, np.where(mask, Ad + 0.25 * (np.outer(Xs, Ys) + np.outer(Ys, Xs)), Ad),
                 abs(Ad) + 0.25 * (np.outer(abs(Xs), abs(Ys)) + np.outer(abs(Ys), abs(Xs))), tol)

# ROTG, ROTMG (CPU) and ROTM (GPU) against OpenBLAS (SciPy): random and extreme data (zeros, 1e-30, 1e30, d1 < 0)
import scipy.linalg.blas as sb  # noqa: E402
cases = [(0.0, 0.0), (0.0, 2.0), (3.0, 0.0), (1e-30, 3e-30), (1e30, -2e30), (-4.0, 3.0)] + \
            [tuple(rng.uniform(-10, 10, 2) * 10.0 ** rng.integers(-5, 5)) for _ in range(200)]
for a_, b_ in cases:
    r, z, c, s_ = gen9blas.rotg(a_, b_)
    c0, s0 = sb.srotg(a_, b_)
    if not (np.isclose(c, c0, rtol=1e-5, atol=1e-7) and np.isclose(s_, s0, rtol=1e-5, atol=1e-7)):
        errors += 1; print(f"  ERR rotg({a_}, {b_}): ({c}, {s_}) != ({c0}, {s0})")
    rr = np.float32(c) * np.float32(a_) + np.float32(s_) * np.float32(b_)   # r = c·a + s·b
    if not np.isclose(r, rr, rtol=1e-5, atol=1e-30):
        errors += 1; print(f"  ERR rotg r({a_}, {b_}): {r} != {rr}")
cases = [(1.0, 1.0, 0.0, 1.0), (1.0, 0.0, 1.0, 1.0), (-1.0, 2.0, 1.0, 1.0), (1e-10, 1.0, 1.0, 1e-3), (1e10, 1.0, 1e3, 1.0),
             (1.0, 1e-9, 2.0, 3.0), (1.0, 1e9, 2.0, 3.0)] + \
            [tuple(np.abs(rng.uniform(0.1, 10, 2)) ** rng.uniform(-3, 3, 2)) + tuple(rng.uniform(-5, 5, 2)) for _ in range(200)]
for d1, d2, x1, y1 in cases:
    w = gen9blas.rotmg(d1, d2, x1, y1)
    p0 = sb.srotmg(d1, d2, x1, y1)
    fl = w[3][0]
    relevant = {-2.0: [], -1.0: [1, 2, 3, 4], 0.0: [2, 3], 1.0: [1, 4]}.get(float(fl), [1, 2, 3, 4])
    if fl != p0[0] or not np.allclose(w[3][relevant], p0[relevant], rtol=1e-4, atol=1e-12):
        errors += 1; print(f"  ERR rotmg{(d1, d2, x1, y1)}: {w[3]} != {p0}")
for dt in (np.float32, np.float16, np.float64):
    for param in ([-1.0, 0.5, -0.3, 0.2, 1.5], [0.0, 9, -0.3, 0.2, 9], [1.0, 0.5, 9, 9, 1.5], [-2.0, 9, 9, 9, 9]):
        x = vector(1001, dt, -2, 1); y = vector(1001, dt, 3); X, Y = x.astype(np.float64), y.astype(np.float64)
        rt = np.float64 if dt == np.float64 else np.float32   # reference: OpenBLAS srotm / drotm
        rx, ry = (sb.drotm if dt == np.float64 else sb.srotm)(X.astype(rt), Y.astype(rt), np.array(param, rt))
        gen9blas.rotm(x, y, param)
        check(f"rotm {np.dtype(dt).name} flag {param[0]}", np.concatenate([x, y]), np.concatenate([rx, ry]).astype(np.float64),
             np.concatenate([abs(X) + abs(Y)] * 2) * 2, TOL[dt])

# Banded and packed (row layout): reference from a dense matrix packed into the format
def to_band(D, KL, KU):
    M, N = D.shape; P = np.zeros((M, KL + KU + 1), D.dtype)
    for i in range(M):
        for j in range(max(0, i - KL), min(N, i + KU + 1)):
            P[i, KL + j - i] = D[i, j]
    return P


def band_mask(M, N, KL, KU):
    i, j = np.indices((M, N)); return (j - i >= -KL) & (j - i <= KU)


def pack(D, uplo):
    n = D.shape[0]; return np.concatenate([D[i, i:] if uplo == "U" else D[i, :i + 1] for i in range(n)])


for dt in (np.float32, np.float16, np.float64):
    tol = TOL[dt]
    for M, N, KL, KU in ((1, 1, 0, 0), (50, 40, 3, 5), (300, 310, 0, 7), (257, 200, 30, 0), (100, 100, 99, 99)):
        D = rng.uniform(-1, 1, (M, N)).astype(dt).astype(np.float64) * band_mask(M, N, KL, KU)
        P = to_band(D, KL, KU).astype(dt)
        for tr in (False, True):
            x = vector(M if tr else N, dt, -2); y0 = vector(N if tr else M, dt, 3)
            X, Y0 = x.astype(np.float64), y0.astype(np.float64); O = D.T if tr else D
            gen9blas.gbmv(P, x, M, N, KL, KU, alpha=0.5, beta=2.0, y=y0, trans=tr)
            check(f"gbmv {M}x{N} KL={KL} KU={KU} tr={tr}", y0, 0.5 * O @ X + 2 * Y0, 0.5 * abs(O) @ abs(X) + 2 * abs(Y0), tol)
    for n, K in ((1, 0), (60, 4), (300, 17), (130, 129)):
        Dp = rng.uniform(-1, 1, (n, n)).astype(dt).astype(np.float64)
        for uplo in "UL":
            g = uplo == "U"
            T = Dp * band_mask(n, n, 0 if g else K, K if g else 0); S = T + T.T - np.diag(np.diag(T))
            P = to_band(T, 0 if g else K, K if g else 0).astype(dt)
            x = vector(n, dt, 2); X = x.astype(np.float64)
            check(f"sbmv n={n} K={K} {uplo}", gen9blas.sbmv(P, x, K, uplo=uplo, alpha=1.5), 1.5 * S @ X, 1.5 * abs(S) @ abs(X), tol)
            Tp = (np.triu if g else np.tril)(Dp); Sp = Tp + Tp.T - np.diag(np.diag(Tp))
            Ap = pack(Tp, uplo).astype(dt)
            y0 = vector(n, dt, -1); Y0 = y0.astype(np.float64)
            gen9blas.spmv(Ap, x, uplo=uplo, alpha=0.5, beta=-1.0, y=y0)
            check(f"spmv n={n} {uplo}", y0, 0.5 * Sp @ X - Y0, 0.5 * abs(Sp) @ abs(X) + abs(Y0), tol)
            for tr in (False, True):
                for dg in (False, True):
                    label = f"n={n} K={K} {uplo} tr={tr} diag={dg}"
                    Tm, Tpm = T.copy(), Tp.copy()
                    if dg:
                        np.fill_diagonal(Tm, 1); np.fill_diagonal(Tpm, 1)
                    x = vector(n, dt, -3); X = x.astype(np.float64)
                    gen9blas.tbmv(P, x, K, uplo=uplo, trans=tr, diag=dg)
                    check("tbmv " + label, x, op_(Tm, tr) @ X, abs(op_(Tm, tr)) @ abs(X), tol)
                    x = vector(n, dt, 1); X = x.astype(np.float64)
                    gen9blas.tpmv(Ap, x, uplo=uplo, trans=tr, diag=dg)
                    check("tpmv " + label, x, op_(Tpm, tr) @ X, abs(op_(Tpm, tr)) @ abs(X), tol)
                    # solutions: well conditioned (small off-diagonal elements, diagonal 1..2)
                    Ts = T / max(K, 1) / 4; np.fill_diagonal(Ts, 1 + abs(np.diag(T)))
                    Ps = to_band(Ts, 0 if g else K, K if g else 0).astype(dt); Ts = Ts.astype(dt).astype(np.float64)
                    if dg:
                        np.fill_diagonal(Ts, 1)
                    b = vector(n, dt, 2); Bv = b.astype(np.float64)
                    gen9blas.tbsv(Ps, b, K, uplo=uplo, trans=tr, diag=dg)
                    ref = np.linalg.solve(op_(Ts, tr), Bv)
                    check("tbsv " + label, b, ref, np.max(abs(ref)), SOLVE_TOL[dt])
                    Tq = Tp / n; np.fill_diagonal(Tq, 1 + abs(np.diag(Tp)))
                    Aq = pack(Tq, uplo).astype(dt); Tq = Tq.astype(dt).astype(np.float64)
                    if dg:
                        np.fill_diagonal(Tq, 1)
                    b = vector(n, dt, -1); Bv = b.astype(np.float64)
                    gen9blas.tpsv(Aq, b, uplo=uplo, trans=tr, diag=dg)
                    ref = np.linalg.solve(op_(Tq, tr), Bv)
                    check("tpsv " + label, b, ref, np.max(abs(ref)), SOLVE_TOL[dt])
            xs, ys = vector(n, dt, 1), vector(n, dt, -2); Xs, Ys = xs.astype(np.float64), ys.astype(np.float64)
            Ap1 = Ap.copy(); gen9blas.spr(-0.5, xs, Ap1, uplo=uplo)
            R = (np.triu if g else np.tril)(Tp - 0.5 * np.outer(Xs, Xs))
            check(f"chk n={n} {uplo}", Ap1, pack(R, uplo), pack(abs(Tp) + 0.5 * np.outer(abs(Xs), abs(Xs)), uplo), tol)
            Ap2 = Ap.copy(); gen9blas.spr2(0.25, xs, ys, Ap2, uplo=uplo)
            R = (np.triu if g else np.tril)(Tp + 0.25 * (np.outer(Xs, Ys) + np.outer(Ys, Xs)))
            check(f"spr2 n={n} {uplo}", Ap2, pack(R, uplo), pack(abs(Tp) + 0.5 * np.outer(abs(Xs), abs(Ys)) + 0.5 * np.outer(abs(Ys), abs(Xs)), uplo), tol)

# USM memory (gen9blas.empty): slices and zero-copy views, repeated allocation and freeing
for attempt in range(20):
    n = int(rng.integers(1, 3)) << 20
    x = gen9blas.empty((n + 7,), np.float32); y = gen9blas.empty((n + 5,), np.float32)
    x[:] = attempt; y[:] = 1
    gen9blas.axpy(1.0, x[3:n + 3], y[5:n + 5])
    if not ((y[5:n + 5] == attempt + 1).all() and (y[:5] == 1).all()):
        errors += 1; print(f"  ERR USM: attempt {attempt}")
    del x, y
for dt in (np.float32, np.float16, np.float64):   # the same set as above, but on USM memory (slices, strides, negative strides)
    tol = TOL[dt]
    A = gen9blas.empty((700, 900), dt); A[:] = rng.uniform(-1, 1, A.shape); Ad = A.astype(np.float64)
    v = gen9blas.empty((5000,), dt); v[:] = rng.uniform(-1, 1, v.shape); V = v.astype(np.float64)
    y = gen9blas.empty((2000,), dt); y[:] = 0
    gen9blas.gemv(A[50:650, 100:900], v[::-5][:800], alpha=2.0, y=y[1::2][:600])
    check("USM gemv slices", y[1::2][:600], 2 * Ad[50:650, 100:900] @ V[::-5][:800],
         2 * abs(Ad[50:650, 100:900]) @ abs(V[::-5][:800]), tol)
    if not (y[0::2] == 0).all():
        errors += 1; print("  ERR USM gemv: changed elements outside the result")
    check("USM dot", gen9blas.dot(v[10:3010:3], v[4000:3000:-1]), V[10:3010:3] @ V[4000:3000:-1], abs(V[10:3010:3]) @ abs(V[4000:3000:-1]), 1e-6)
    gen9blas.gemv(A.T, v[:700], y=y[:900], trans=False)
    check("USM gemv A.T", y[:900], Ad.T @ V[:700], abs(Ad.T) @ abs(V[:700]), tol)

print("RESULT:", "all OK" if errors == 0 else f"{errors} errors")
sys.exit(1 if errors else 0)
