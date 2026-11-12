"""Test BLAS poziomów 1–2 w libgen9blas względem NumPy (float64).

Uruchom: python3 lib/test_level12.py
Etap B: SYMV, TRMV, TRSV (błąd względny rozwiązania do max|x|: 1e-5 / 2e-3), SYR, SYR2.
Kryteria: błąd względem skali wzoru (np. Σ|a·x| + |y| dla axpy, |A|·|x| dla gemv): float32 ≤ 1e-6, float16 ≤ 2e-3
(zaokrąglenie wyniku do FP16, skala co najmniej najmniejsza liczba normalna FP16; liczenie w FP32). Redukcje (FP64 na GPU): ≤ 1e-6 (float32), ≤ 1e-6 względem
dokładnej sumy wejść FP16. Przypadki: kroki (także ujemne), wskaźniki niewyrównane, ogony długości, operandy na
wspólnych stronach, widoki transponowane i podmacierze, zwalnianie i ponowny przydział zarejestrowanej pamięci.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import gen9blas  # noqa: E402

rng = np.random.default_rng(11)
bledy = 0
KR = {np.float32: 1e-6, np.float16: 2e-3, np.float64: 1e-13}
ROZW = {np.float32: 1e-5, np.float16: 3e-3, np.float64: 1e-12}   # błąd względny rozwiązań (TRSV, TBSV, TPSV)


def ocen(nazwa, W, ref, skala, kr):
    global bledy
    e = float(np.max(np.abs(np.asarray(W, np.float64) - ref) / np.maximum(skala, 6.2e-5 if kr > 1e-5 else 1e-30)))
    if not e <= kr:
        bledy += 1
        print(f"  BLAD {nazwa}: {e:.1e} > {kr:.0e}")


def op_(X, t):
    return X.T if t else X


def wektor(n, typ, krok=1, przes=0):
    """Wektor długości n z krokiem `krok` (ujemny: odwrócony widok) w tablicy przesuniętej o `przes` elementów."""
    baza = rng.uniform(-1, 1, n * abs(krok) + przes + 3).astype(typ)
    v = baza[przes:przes + n * abs(krok):abs(krok)]
    return v[::-1] if krok < 0 else v


for typ in (np.float32, np.float16, np.float64):
    kr = KR[typ]
    print(f"== {np.dtype(typ).name}")
    for n in (1, 5, 255, 256, 1000, 100003):
        for kx, ky, px in ((1, 1, 0), (1, 1, 1), (3, -2, 5), (-1, 1, 2)):
            nz = f"n={n} kroki {kx},{ky} przes {px}"
            x, y = wektor(n, typ, kx, px), wektor(n, typ, ky, 1)
            X, Y = x.astype(np.float64), y.astype(np.float64)
            gen9blas.axpy(0.75, x, y); ocen("axpy " + nz, y, 0.75 * X + Y, 0.75 * abs(X) + abs(Y), kr)
            y[:] = Y; gen9blas.scal(-1.5, y); ocen("scal " + nz, y, -1.5 * Y, 1.5 * abs(Y), kr)
            y[:] = Y; gen9blas.copy(x, y); ocen("copy " + nz, y, X, 1, 0)
            y[:] = Y; gen9blas.swap(x, y); ocen("swap " + nz, np.concatenate([x, y]), np.concatenate([Y, X]), 1, 0)
            x[:], y[:] = X, Y
            gen9blas.rot(x, y, 0.6, 0.8)
            ocen("rot " + nz, np.concatenate([x, y]), np.concatenate([0.6 * X + 0.8 * Y, 0.6 * Y - 0.8 * X]),
                 np.concatenate([abs(X) + abs(Y)] * 2), kr)
            x[:], y[:] = X, Y
            ocen("dot " + nz, gen9blas.dot(x, y), X @ Y, abs(X) @ abs(Y), 1e-6)
            ocen("asum " + nz, gen9blas.asum(x), abs(X).sum(), abs(X).sum(), 1e-6)
            ocen("nrm2 " + nz, gen9blas.nrm2(x), np.linalg.norm(X), np.linalg.norm(X), 1e-6)
            i = gen9blas.iamax(x)
            if i != int(np.argmax(abs(X))):
                bledy += 1; print(f"  BLAD iamax {nz}: {i} != {np.argmax(abs(X))}")
    # operandy na wspólnych stronach / ten sam obszar
    v = rng.uniform(-1, 1, 3000).astype(typ); V = v.astype(np.float64)
    gen9blas.axpy(2.0, v[:1000], v[1000:2000]); ocen("axpy wspólne strony", v[1000:2000], 2 * V[:1000] + V[1000:2000], 3, kr)
    v[:] = V; ocen("dot x·x", gen9blas.dot(v, v), V @ V, V @ V, 1e-6)
    # GEMV / GER
    for M, N in ((1, 1), (7, 3), (64, 64), (257, 1031), (1000, 2), (3, 5000), (2048, 1536)):
        A = rng.uniform(-1, 1, (M, N)).astype(typ); Ad = A.astype(np.float64)
        x, xt = wektor(N, typ), wektor(M, typ, -2, 3)
        y0, yt0 = wektor(M, typ, 2), wektor(N, typ)
        Y0, YT0 = y0.astype(np.float64), yt0.astype(np.float64)
        y = gen9blas.gemv(A, x, alpha=1.5)
        ocen(f"gemv N {M}x{N}", y, 1.5 * Ad @ x.astype(np.float64), 1.5 * abs(Ad) @ abs(x.astype(np.float64)), kr)
        gen9blas.gemv(A, x, alpha=0.5, beta=-2.0, y=y0)
        ocen(f"gemv N beta {M}x{N}", y0, 0.5 * Ad @ x.astype(np.float64) - 2 * Y0,
             0.5 * abs(Ad) @ abs(x.astype(np.float64)) + 2 * abs(Y0), kr)
        gen9blas.gemv(A, xt, alpha=0.5, beta=1.0, y=yt0, trans=True)
        Xt = xt.astype(np.float64)
        ocen(f"gemv T {M}x{N}", yt0, 0.5 * Ad.T @ Xt + YT0, 0.5 * abs(Ad.T) @ abs(Xt) + abs(YT0), kr)
        xv = wektor(M, typ); Xv = xv.astype(np.float64)
        yv = gen9blas.gemv(A.T, xv)   # widok transponowany bez kopii
        ocen(f"gemv A.T {M}x{N}", yv, Ad.T @ Xv, abs(Ad.T) @ abs(Xv), kr)
        xg, yg = wektor(M, typ, 1, 1), wektor(N, typ, -3, 0)
        Xg, Yg = xg.astype(np.float64), yg.astype(np.float64)
        A2 = A.copy(); gen9blas.ger(-0.5, xg, yg, A2)
        ocen(f"ger {M}x{N}", A2, Ad - 0.5 * np.outer(Xg, Yg), abs(Ad) + 0.5 * np.outer(abs(Xg), abs(Yg)), kr)
    # podmacierz (lda > N) z wektorem w tej samej tablicy
    B = rng.uniform(-1, 1, (300, 400)).astype(typ); Bd = B.astype(np.float64)
    y = gen9blas.gemv(B[10:200, 5:305], B[0, :300])
    ocen("gemv podmacierz", y, Bd[10:200, 5:305] @ Bd[0, :300], abs(Bd[10:200, 5:305]) @ abs(Bd[0, :300]), kr)

# etap B: SYMV, TRMV, TRSV, SYR, SYR2 (oba trójkąty, transpozycje, jedynki na przekątnej, widok A.T, kroki)
for typ in (np.float32, np.float16, np.float64):
    kr = KR[typ]
    for n in (1, 5, 63, 64, 65, 200, 1000):
        A = rng.uniform(-1, 1, (n, n)).astype(typ); Ad = A.astype(np.float64)
        for uplo in "LU":
            T = np.tril(Ad) if uplo == "L" else np.triu(Ad)
            S = T + T.T - np.diag(np.diag(Ad))                   # symetryczna z trójkąta uplo
            x = wektor(n, typ, -2, 1); X = x.astype(np.float64)
            y0 = wektor(n, typ, 3); Y0 = y0.astype(np.float64)
            gen9blas.symv(A, x, uplo=uplo, alpha=0.5, beta=2.0, y=y0)
            ocen(f"symv {uplo} n={n}", y0, 0.5 * S @ X + 2 * Y0, 0.5 * abs(S) @ abs(X) + 2 * abs(Y0), kr)
            yv = gen9blas.symv(A.T, x, uplo=uplo)                         # A.T: inny trójkąt w pamięci
            T2 = np.tril(Ad.T) if uplo == "L" else np.triu(Ad.T); S2 = T2 + T2.T - np.diag(np.diag(Ad))
            ocen(f"symv A.T {uplo} n={n}", yv, S2 @ X, abs(S2) @ abs(X), kr)
            for trans in (False, True):
                for diag in (False, True):
                    Tm = T.copy()
                    if diag:
                        np.fill_diagonal(Tm, 1.0)
                    O = Tm.T if trans else Tm
                    nz = f"{uplo} trans={int(trans)} diag={int(diag)} n={n}"
                    xx = wektor(n, typ, 2); XX = xx.astype(np.float64)
                    gen9blas.trmv(A, xx, uplo=uplo, trans=trans, diag=diag)
                    ocen("trmv " + nz, xx, O @ XX, abs(O) @ abs(XX), kr)
                    # TRSV: macierz dobrze uwarunkowana -- duża przekątna, a przy jedynkach małe elementy poza nią
                    trojkat = np.tril if uplo == "L" else np.triu
                    if diag:
                        Ab = (Ad / n).astype(typ)
                    else:
                        Ab = A.copy(); np.fill_diagonal(Ab, np.where(np.diag(Ad) < 0, -1, 1) * n / 4 + np.diag(Ad))
                    Tb = trojkat(Ab.astype(np.float64))
                    if diag:
                        np.fill_diagonal(Tb, 1.0)
                    Ob = Tb.T if trans else Tb
                    b = wektor(n, typ, -1); B = b.astype(np.float64)
                    gen9blas.trsv(Ab, b, uplo=uplo, trans=trans, diag=diag)
                    ref = np.linalg.solve(Ob, B)
                    ocen("trsv " + nz, b, ref, np.max(abs(ref)), ROZW[typ])
            # SYR, SYR2: trójkąt uplo zmieniony, drugi bez zmian
            xs, ys = wektor(n, typ, 1, 2), wektor(n, typ, -3, 0)
            Xs, Ys = xs.astype(np.float64), ys.astype(np.float64)
            maska = np.tril(np.ones((n, n), bool)) if uplo == "L" else np.triu(np.ones((n, n), bool))
            A1 = A.copy(); gen9blas.syr(-0.5, xs, A1, uplo=uplo)
            ocen(f"syr {uplo} n={n}", A1, np.where(maska, Ad - 0.5 * np.outer(Xs, Xs), Ad), abs(Ad) + 0.5 * np.outer(abs(Xs), abs(Xs)), kr)
            A2 = A.copy(); gen9blas.syr2(0.25, xs, ys, A2, uplo=uplo)
            ocen(f"syr2 {uplo} n={n}", A2, np.where(maska, Ad + 0.25 * (np.outer(Xs, Ys) + np.outer(Ys, Xs)), Ad),
                 abs(Ad) + 0.25 * (np.outer(abs(Xs), abs(Ys)) + np.outer(abs(Ys), abs(Xs))), kr)

# ROTG, ROTMG (CPU) i ROTM (GPU) względem OpenBLAS (SciPy): dane losowe i skrajne (zera, 1e-30, 1e30, d1 < 0)
import scipy.linalg.blas as sb  # noqa: E402
przypadki = [(0.0, 0.0), (0.0, 2.0), (3.0, 0.0), (1e-30, 3e-30), (1e30, -2e30), (-4.0, 3.0)] + \
            [tuple(rng.uniform(-10, 10, 2) * 10.0 ** rng.integers(-5, 5)) for _ in range(200)]
for a_, b_ in przypadki:
    r, z, c, s_ = gen9blas.rotg(a_, b_)
    c0, s0 = sb.srotg(a_, b_)
    if not (np.isclose(c, c0, rtol=1e-5, atol=1e-7) and np.isclose(s_, s0, rtol=1e-5, atol=1e-7)):
        bledy += 1; print(f"  BLAD rotg({a_}, {b_}): ({c}, {s_}) != ({c0}, {s0})")
    rr = np.float32(c) * np.float32(a_) + np.float32(s_) * np.float32(b_)   # r = c·a + s·b
    if not np.isclose(r, rr, rtol=1e-5, atol=1e-30):
        bledy += 1; print(f"  BLAD rotg r({a_}, {b_}): {r} != {rr}")
przypadki = [(1.0, 1.0, 0.0, 1.0), (1.0, 0.0, 1.0, 1.0), (-1.0, 2.0, 1.0, 1.0), (1e-10, 1.0, 1.0, 1e-3), (1e10, 1.0, 1e3, 1.0),
             (1.0, 1e-9, 2.0, 3.0), (1.0, 1e9, 2.0, 3.0)] + \
            [tuple(np.abs(rng.uniform(0.1, 10, 2)) ** rng.uniform(-3, 3, 2)) + tuple(rng.uniform(-5, 5, 2)) for _ in range(200)]
for d1, d2, x1, y1 in przypadki:
    w = gen9blas.rotmg(d1, d2, x1, y1)
    p0 = sb.srotmg(d1, d2, x1, y1)
    fl = w[3][0]
    istotne = {-2.0: [], -1.0: [1, 2, 3, 4], 0.0: [2, 3], 1.0: [1, 4]}.get(float(fl), [1, 2, 3, 4])
    if fl != p0[0] or not np.allclose(w[3][istotne], p0[istotne], rtol=1e-4, atol=1e-12):
        bledy += 1; print(f"  BLAD rotmg{(d1, d2, x1, y1)}: {w[3]} != {p0}")
for typ in (np.float32, np.float16, np.float64):
    for param in ([-1.0, 0.5, -0.3, 0.2, 1.5], [0.0, 9, -0.3, 0.2, 9], [1.0, 0.5, 9, 9, 1.5], [-2.0, 9, 9, 9, 9]):
        x = wektor(1001, typ, -2, 1); y = wektor(1001, typ, 3); X, Y = x.astype(np.float64), y.astype(np.float64)
        rt = np.float64 if typ == np.float64 else np.float32   # wzorzec: OpenBLAS srotm / drotm
        rx, ry = (sb.drotm if typ == np.float64 else sb.srotm)(X.astype(rt), Y.astype(rt), np.array(param, rt))
        gen9blas.rotm(x, y, param)
        ocen(f"rotm {np.dtype(typ).name} flaga {param[0]}", np.concatenate([x, y]), np.concatenate([rx, ry]).astype(np.float64),
             np.concatenate([abs(X) + abs(Y)] * 2) * 2, KR[typ])

# Pasmowe i spakowane (układ wierszowy): wzorzec z gęstej macierzy pakowanej do formatu
def do_pasma(D, KL, KU):
    M, N = D.shape; P = np.zeros((M, KL + KU + 1), D.dtype)
    for i in range(M):
        for j in range(max(0, i - KL), min(N, i + KU + 1)):
            P[i, KL + j - i] = D[i, j]
    return P


def maska_pasma(M, N, KL, KU):
    i, j = np.indices((M, N)); return (j - i >= -KL) & (j - i <= KU)


def spakuj(D, uplo):
    n = D.shape[0]; return np.concatenate([D[i, i:] if uplo == "U" else D[i, :i + 1] for i in range(n)])


for typ in (np.float32, np.float16, np.float64):
    kr = KR[typ]
    for M, N, KL, KU in ((1, 1, 0, 0), (50, 40, 3, 5), (300, 310, 0, 7), (257, 200, 30, 0), (100, 100, 99, 99)):
        D = rng.uniform(-1, 1, (M, N)).astype(typ).astype(np.float64) * maska_pasma(M, N, KL, KU)
        P = do_pasma(D, KL, KU).astype(typ)
        for tr in (False, True):
            x = wektor(M if tr else N, typ, -2); y0 = wektor(N if tr else M, typ, 3)
            X, Y0 = x.astype(np.float64), y0.astype(np.float64); O = D.T if tr else D
            gen9blas.gbmv(P, x, M, N, KL, KU, alpha=0.5, beta=2.0, y=y0, trans=tr)
            ocen(f"gbmv {M}x{N} KL={KL} KU={KU} tr={tr}", y0, 0.5 * O @ X + 2 * Y0, 0.5 * abs(O) @ abs(X) + 2 * abs(Y0), kr)
    for n, K in ((1, 0), (60, 4), (300, 17), (130, 129)):
        Dp = rng.uniform(-1, 1, (n, n)).astype(typ).astype(np.float64)
        for uplo in "UL":
            g = uplo == "U"
            T = Dp * maska_pasma(n, n, 0 if g else K, K if g else 0); S = T + T.T - np.diag(np.diag(T))
            P = do_pasma(T, 0 if g else K, K if g else 0).astype(typ)
            x = wektor(n, typ, 2); X = x.astype(np.float64)
            ocen(f"sbmv n={n} K={K} {uplo}", gen9blas.sbmv(P, x, K, uplo=uplo, alpha=1.5), 1.5 * S @ X, 1.5 * abs(S) @ abs(X), kr)
            Tp = (np.triu if g else np.tril)(Dp); Sp = Tp + Tp.T - np.diag(np.diag(Tp))
            Ap = spakuj(Tp, uplo).astype(typ)
            y0 = wektor(n, typ, -1); Y0 = y0.astype(np.float64)
            gen9blas.spmv(Ap, x, uplo=uplo, alpha=0.5, beta=-1.0, y=y0)
            ocen(f"spmv n={n} {uplo}", y0, 0.5 * Sp @ X - Y0, 0.5 * abs(Sp) @ abs(X) + abs(Y0), kr)
            for tr in (False, True):
                for dg in (False, True):
                    nz = f"n={n} K={K} {uplo} tr={tr} diag={dg}"
                    Tm, Tpm = T.copy(), Tp.copy()
                    if dg:
                        np.fill_diagonal(Tm, 1); np.fill_diagonal(Tpm, 1)
                    x = wektor(n, typ, -3); X = x.astype(np.float64)
                    gen9blas.tbmv(P, x, K, uplo=uplo, trans=tr, diag=dg)
                    ocen("tbmv " + nz, x, op_(Tm, tr) @ X, abs(op_(Tm, tr)) @ abs(X), kr)
                    x = wektor(n, typ, 1); X = x.astype(np.float64)
                    gen9blas.tpmv(Ap, x, uplo=uplo, trans=tr, diag=dg)
                    ocen("tpmv " + nz, x, op_(Tpm, tr) @ X, abs(op_(Tpm, tr)) @ abs(X), kr)
                    # rozwiązania: dobrze uwarunkowane (małe elementy poza przekątną, przekątna 1..2)
                    Ts = T / max(K, 1) / 4; np.fill_diagonal(Ts, 1 + abs(np.diag(T)))
                    Ps = do_pasma(Ts, 0 if g else K, K if g else 0).astype(typ); Ts = Ts.astype(typ).astype(np.float64)
                    if dg:
                        np.fill_diagonal(Ts, 1)
                    b = wektor(n, typ, 2); Bv = b.astype(np.float64)
                    gen9blas.tbsv(Ps, b, K, uplo=uplo, trans=tr, diag=dg)
                    ref = np.linalg.solve(op_(Ts, tr), Bv)
                    ocen("tbsv " + nz, b, ref, np.max(abs(ref)), ROZW[typ])
                    Tq = Tp / n; np.fill_diagonal(Tq, 1 + abs(np.diag(Tp)))
                    Aq = spakuj(Tq, uplo).astype(typ); Tq = Tq.astype(typ).astype(np.float64)
                    if dg:
                        np.fill_diagonal(Tq, 1)
                    b = wektor(n, typ, -1); Bv = b.astype(np.float64)
                    gen9blas.tpsv(Aq, b, uplo=uplo, trans=tr, diag=dg)
                    ref = np.linalg.solve(op_(Tq, tr), Bv)
                    ocen("tpsv " + nz, b, ref, np.max(abs(ref)), ROZW[typ])
            xs, ys = wektor(n, typ, 1), wektor(n, typ, -2); Xs, Ys = xs.astype(np.float64), ys.astype(np.float64)
            Ap1 = Ap.copy(); gen9blas.spr(-0.5, xs, Ap1, uplo=uplo)
            R = (np.triu if g else np.tril)(Tp - 0.5 * np.outer(Xs, Xs))
            ocen(f"spr n={n} {uplo}", Ap1, spakuj(R, uplo), spakuj(abs(Tp) + 0.5 * np.outer(abs(Xs), abs(Xs)), uplo), kr)
            Ap2 = Ap.copy(); gen9blas.spr2(0.25, xs, ys, Ap2, uplo=uplo)
            R = (np.triu if g else np.tril)(Tp + 0.25 * (np.outer(Xs, Ys) + np.outer(Ys, Xs)))
            ocen(f"spr2 n={n} {uplo}", Ap2, spakuj(R, uplo), spakuj(abs(Tp) + 0.5 * np.outer(abs(Xs), abs(Ys)) + 0.5 * np.outer(abs(Ys), abs(Xs)), uplo), kr)

# pamięć USM (gen9blas.empty): wycinki i widoki bez kopii, wielokrotny przydział i zwolnienie
for proba in range(20):
    n = int(rng.integers(1, 3)) << 20
    x = gen9blas.empty((n + 7,), np.float32); y = gen9blas.empty((n + 5,), np.float32)
    x[:] = proba; y[:] = 1
    gen9blas.axpy(1.0, x[3:n + 3], y[5:n + 5])
    if not ((y[5:n + 5] == proba + 1).all() and (y[:5] == 1).all()):
        bledy += 1; print(f"  BLAD USM: proba {proba}")
    del x, y
for typ in (np.float32, np.float16, np.float64):   # ten sam zestaw co wyżej, ale na pamięci USM (wycinki, kroki, ujemne kroki)
    kr = KR[typ]
    A = gen9blas.empty((700, 900), typ); A[:] = rng.uniform(-1, 1, A.shape); Ad = A.astype(np.float64)
    v = gen9blas.empty((5000,), typ); v[:] = rng.uniform(-1, 1, v.shape); V = v.astype(np.float64)
    y = gen9blas.empty((2000,), typ); y[:] = 0
    gen9blas.gemv(A[50:650, 100:900], v[::-5][:800], alpha=2.0, y=y[1::2][:600])
    ocen("USM gemv wycinki", y[1::2][:600], 2 * Ad[50:650, 100:900] @ V[::-5][:800],
         2 * abs(Ad[50:650, 100:900]) @ abs(V[::-5][:800]), kr)
    if not (y[0::2] == 0).all():
        bledy += 1; print("  BLAD USM gemv: zmienione elementy poza wynikiem")
    ocen("USM dot", gen9blas.dot(v[10:3010:3], v[4000:3000:-1]), V[10:3010:3] @ V[4000:3000:-1], abs(V[10:3010:3]) @ abs(V[4000:3000:-1]), 1e-6)
    gen9blas.gemv(A.T, v[:700], y=y[:900], trans=False)
    ocen("USM gemv A.T", y[:900], Ad.T @ V[:700], abs(Ad.T) @ abs(V[:700]), kr)

print("WYNIK:", "wszystko OK" if bledy == 0 else f"{bledy} błędów")
sys.exit(1 if bledy else 0)
