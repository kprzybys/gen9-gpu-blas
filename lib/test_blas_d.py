"""Test funkcji D (FP64) warstwy BLAS/CBLAS (libgen9blas_blas.so) względem NumPy -- wersja test_blas.py dla float64.

Uruchom: python3 lib/test_blas.py
Wymusza GPU (GEN9BLAS_BLAS=gpu) i sprawdza z dziennika (GEN9BLAS_BLAS_INFO), że każda funkcja S poszła na GPU. Testy: CBLAS
w układzie wierszowym i kolumnowym, interfejs Fortran (kolumnowy), wszystkie kombinacje trans/uplo/side/diag,
lda większe od minimum, kroki wektorów (także ujemne), poziom 1 na pamięci USM (gen9blas.empty z tej samej biblioteki),
funkcje przekazywane do BLAS na CPU (D, Z). Kryterium: błąd względem skali wzoru ≤ 1e-12 (TRSV/TRSM: względny ≤ 1e-11).
"""
import ctypes
import os
import sys
import tempfile

TU = os.path.dirname(os.path.realpath(__file__))
os.environ.setdefault("GEN9BLAS_BLAS", "gpu")
os.environ["GEN9BLAS_BLAS_INFO"] = "1"
os.environ["GEN9BLAS_LIB"] = os.path.join(TU, "libgen9blas_blas.so")   # gen9blas.empty z tego samego kontekstu OpenCL
import numpy as np  # noqa: E402

sys.path.insert(0, TU)
import gen9blas  # noqa: E402

B = gen9blas._lib
dziennik = tempfile.TemporaryFile()
stderr_kopia = os.dup(2)
os.dup2(dziennik.fileno(), 2)

rng = np.random.default_rng(5)
bledy = 0
RM, CM = 101, 102
TR = {False: 111, True: 112}
UP = {"U": 121, "L": 122}
SD = {"L": 141, "R": 142}
DG = {False: 131, True: 132}
f, i, p, d = ctypes.c_double, ctypes.c_int, ctypes.c_void_p, ctypes.c_double
for nazwa, typy in {
    "cblas_dgemm": [i] * 6 + [f, p, i, p, i, f, p, i], "cblas_dsymm": [i] * 5 + [f, p, i, p, i, f, p, i],
    "cblas_dsyrk": [i] * 5 + [f, p, i, f, p, i], "cblas_dsyr2k": [i] * 5 + [f, p, i, p, i, f, p, i],
    "cblas_dtrmm": [i] * 7 + [f, p, i, p, i], "cblas_dtrsm": [i] * 7 + [f, p, i, p, i],
    "cblas_dgemv": [i] * 4 + [f, p, i, p, i, f, p, i], "cblas_dger": [i] * 3 + [f, p, i, p, i, p, i],
    "cblas_dsymv": [i] * 3 + [f, p, i, p, i, f, p, i], "cblas_dtrmv": [i] * 5 + [p, i, p, i],
    "cblas_dtrsv": [i] * 5 + [p, i, p, i], "cblas_dsyr": [i] * 3 + [f, p, i, p, i], "cblas_dsyr2": [i] * 3 + [f, p, i, p, i, p, i],
    "cblas_daxpy": [i, f, p, i, p, i], "cblas_dscal": [i, f, p, i], "cblas_dcopy": [i, p, i, p, i], "cblas_dswap": [i, p, i, p, i],
    "cblas_drot": [i, p, i, p, i, f, f], "cblas_drotm": [i, p, i, p, i, p], "cblas_ddot": [i, p, i, p, i], "cblas_dasum": [i, p, i], "cblas_dnrm2": [i, p, i], "cblas_idamax": [i, p, i],
}.items():
    getattr(B, nazwa).argtypes = typy
for nazwa in ("cblas_ddot", "cblas_dasum", "cblas_dnrm2"):
    getattr(B, nazwa).restype = f
B.cblas_idamax.restype = ctypes.c_size_t


def ocen(nazwa, W, ref, skala, kr=1e-12):
    global bledy
    e = float(np.max(np.abs(np.asarray(W, np.float64) - ref) / np.maximum(skala, 1e-30)))
    if not e <= kr:
        bledy += 1
        print(f"  BLAD {nazwa}: {e:.1e}", file=sys.stdout)


def mac(r, c, uklad, zapas=3, typ=np.float64):
    """(widok r×c, wskaźnik, ld) w układzie wierszowym (RM) albo kolumnowym (CM), z ld większym od minimum."""
    if uklad == RM:
        baza = rng.uniform(-1, 1, (r, c + zapas)).astype(typ); v = baza[:, :c]; return v, v.ctypes.data, c + zapas
    baza = np.asfortranarray(rng.uniform(-1, 1, (r + zapas, c)).astype(typ)); v = baza[:r, :]; return v, v.ctypes.data, r + zapas


def wek(n, krok):
    """(widok, wskaźnik na najniższy adres, krok BLAS): krok < 0 -- element 0 na najwyższym adresie."""
    baza = rng.uniform(-1, 1, (n - 1) * abs(krok) + 1).astype(np.float64)
    v = baza[::krok] if krok > 0 else baza[::krok]
    return v, baza.ctypes.data, krok


def op(X, t):
    return X.T if t else X


for uklad in (RM, CM):
    u_n = "wiersz" if uklad == RM else "kolumna"
    M, N, K = 67, 45, 38
    for ta in (False, True):
        for tb in (False, True):
            A, pa, lda = mac(*((K, M) if ta else (M, K)), uklad)
            Bm, pb, ldb = mac(*((N, K) if tb else (K, N)), uklad)
            C, pc, ldc = mac(M, N, uklad)
            Ad, Bd, Cd = (X.astype(np.float64) for X in (A, Bm, C))
            B.cblas_dgemm(uklad, TR[ta], TR[tb], M, N, K, 1.5, pa, lda, pb, ldb, -0.5, pc, ldc)
            ocen(f"sgemm {u_n} {ta}{tb}", C, 1.5 * op(Ad, ta) @ op(Bd, tb) - 0.5 * Cd, 1.5 * abs(op(Ad, ta)) @ abs(op(Bd, tb)) + 0.5 * abs(Cd))
    for strona in "LR":
        n = M if strona == "L" else N
        for uplo in "UL":
            A, pa, lda = mac(n, n, uklad); Ad = A.astype(np.float64)
            T = np.triu(Ad) if uplo == "U" else np.tril(Ad); S = T + T.T - np.diag(np.diag(Ad))
            Bm, pb, ldb = mac(M, N, uklad); C, pc, ldc = mac(M, N, uklad); Bd, Cd = Bm.astype(np.float64), C.astype(np.float64)
            B.cblas_dsymm(uklad, SD[strona], UP[uplo], M, N, 0.7, pa, lda, pb, ldb, 0.3, pc, ldc)
            ref = 0.7 * (S @ Bd if strona == "L" else Bd @ S) + 0.3 * Cd
            ocen(f"ssymm {u_n} {strona}{uplo}", C, ref, 0.7 * (abs(S) @ abs(Bd) if strona == "L" else abs(Bd) @ abs(S)) + 0.3 * abs(Cd))
            for t in (False, True):
                for dg in (False, True):
                    Tm = T.copy()
                    if dg:
                        np.fill_diagonal(Tm, 1)
                    Bm, pb, ldb = mac(M, N, uklad); Bd = Bm.astype(np.float64)
                    B.cblas_dtrmm(uklad, SD[strona], UP[uplo], TR[t], DG[dg], M, N, 2.0, pa, lda, pb, ldb)
                    ref = 2 * (op(Tm, t) @ Bd if strona == "L" else Bd @ op(Tm, t))
                    ocen(f"strmm {u_n} {strona}{uplo}{t}{dg}", Bm, ref, 2 * (abs(op(Tm, t)) @ abs(Bd) if strona == "L" else abs(Bd) @ abs(op(Tm, t))))
                    # TRSM: dobrze uwarunkowana trójkątna
                    A2, pa2, lda2 = mac(n, n, uklad); A2[:] /= n; A2[np.diag_indices(n)] = 1 + rng.uniform(0, 1, n)
                    T2 = (np.triu if uplo == "U" else np.tril)(A2.astype(np.float64))
                    if dg:
                        np.fill_diagonal(T2, 1)
                    Bm, pb, ldb = mac(M, N, uklad); Bd = Bm.astype(np.float64)
                    B.cblas_dtrsm(uklad, SD[strona], UP[uplo], TR[t], DG[dg], M, N, 1.0, pa2, lda2, pb, ldb)
                    ref = np.linalg.solve(op(T2, t), Bd) if strona == "L" else np.linalg.solve(op(T2, t).T, Bd.T).T
                    ocen(f"strsm {u_n} {strona}{uplo}{t}{dg}", Bm, ref, np.max(abs(ref)), 1e-11)
    for uplo in "UL":
        for t in (False, True):
            A, pa, lda = mac(*((K, N) if t else (N, K)), uklad); Bm, pb, ldb = mac(*((K, N) if t else (N, K)), uklad)
            C, pc, ldc = mac(N, N, uklad); Ad, Bd, Cd = (X.astype(np.float64) for X in (A, Bm, C))
            maska = np.triu(np.ones((N, N), bool)) if uplo == "U" else np.tril(np.ones((N, N), bool))
            C0 = C.copy()
            B.cblas_dsyrk(uklad, UP[uplo], TR[t], N, K, 0.5, pa, lda, 2.0, pc, ldc)
            ref = np.where(maska, 0.5 * op(Ad, t) @ op(Ad, t).T + 2 * Cd, Cd)
            ocen(f"ssyrk {u_n} {uplo}{t}", C, ref, 0.5 * abs(op(Ad, t)) @ abs(op(Ad, t)).T + 2 * abs(Cd))
            C[:] = C0
            B.cblas_dsyr2k(uklad, UP[uplo], TR[t], N, K, 0.5, pa, lda, pb, ldb, 2.0, pc, ldc)
            ref = np.where(maska, 0.5 * (op(Ad, t) @ op(Bd, t).T + op(Bd, t) @ op(Ad, t).T) + 2 * Cd, Cd)
            ocen(f"ssyr2k {u_n} {uplo}{t}", C, ref, abs(op(Ad, t)) @ abs(op(Bd, t)).T + 2 * abs(Cd))
    # poziom 2
    M, N = 300, 210
    for t in (False, True):
        for kx, ky in ((1, 1), (2, -3), (-1, 2)):
            A, pa, lda = mac(M, N, uklad); Ad = A.astype(np.float64)
            x, px, ix = wek(M if t else N, kx); y, py, iy = wek(N if t else M, ky)
            X, Y = x.astype(np.float64), y.astype(np.float64)
            B.cblas_dgemv(uklad, TR[t], M, N, 0.5, pa, lda, px, ix, 2.0, py, iy)
            ocen(f"sgemv {u_n} t={t} kroki {kx},{ky}", y, 0.5 * op(Ad, t) @ X + 2 * Y, 0.5 * abs(op(Ad, t)) @ abs(X) + 2 * abs(Y))
    x, px, ix = wek(M, -2); y, py, iy = wek(N, 3); A, pa, lda = mac(M, N, uklad)
    X, Y, Ad = x.astype(np.float64), y.astype(np.float64), A.astype(np.float64)
    B.cblas_dger(uklad, M, N, -1.5, px, ix, py, iy, pa, lda)
    ocen(f"sger {u_n}", A, Ad - 1.5 * np.outer(X, Y), abs(Ad) + 1.5 * np.outer(abs(X), abs(Y)))
    n = 260
    for uplo in "UL":
        A, pa, lda = mac(n, n, uklad); Ad = A.astype(np.float64)
        T = np.triu(Ad) if uplo == "U" else np.tril(Ad); S = T + T.T - np.diag(np.diag(Ad))
        maska = np.triu(np.ones((n, n), bool)) if uplo == "U" else np.tril(np.ones((n, n), bool))
        x, px, ix = wek(n, -2); y, py, iy = wek(n, 1); X, Y = x.astype(np.float64), y.astype(np.float64)
        B.cblas_dsymv(uklad, UP[uplo], n, 0.5, pa, lda, px, ix, 1.0, py, iy)
        ocen(f"ssymv {u_n} {uplo}", y, 0.5 * S @ X + Y, 0.5 * abs(S) @ abs(X) + abs(Y))
        B.cblas_dsyr(uklad, UP[uplo], n, 0.5, px, ix, pa, lda)
        ocen(f"ssyr {u_n} {uplo}", A, np.where(maska, Ad + 0.5 * np.outer(X, X), Ad), abs(Ad) + 0.5 * np.outer(abs(X), abs(X)))
        A[:] = Ad
        B.cblas_dsyr2(uklad, UP[uplo], n, 0.5, px, ix, py, iy, pa, lda)
        Y2 = y.astype(np.float64)
        ocen(f"ssyr2 {u_n} {uplo}", A, np.where(maska, Ad + 0.5 * (np.outer(X, Y2) + np.outer(Y2, X)), Ad),
             abs(Ad) + 0.5 * (np.outer(abs(X), abs(Y2)) + np.outer(abs(Y2), abs(X))))
        for t in (False, True):
            for dg in (False, True):
                A3, pa3, lda3 = mac(n, n, uklad)   # świeża macierz (A zmieniły ssyr/ssyr2)
                Tm = (np.triu if uplo == "U" else np.tril)(A3.astype(np.float64))
                if dg:
                    np.fill_diagonal(Tm, 1)
                x, px, ix = wek(n, 2); X = x.astype(np.float64)
                B.cblas_dtrmv(uklad, UP[uplo], TR[t], DG[dg], n, pa3, lda3, px, ix)
                ocen(f"strmv {u_n} {uplo}{t}{dg}", x, op(Tm, t) @ X, abs(op(Tm, t)) @ abs(X))
                A2, pa2, lda2 = mac(n, n, uklad); A2[:] /= n; A2[np.diag_indices(n)] = 1 + rng.uniform(0, 1, n)
                T2 = (np.triu if uplo == "U" else np.tril)(A2.astype(np.float64))
                if dg:
                    np.fill_diagonal(T2, 1)
                x, px, ix = wek(n, -1); X = x.astype(np.float64)
                B.cblas_dtrsv(uklad, UP[uplo], TR[t], DG[dg], n, pa2, lda2, px, ix)
                ref = np.linalg.solve(op(T2, t), X)
                ocen(f"strsv {u_n} {uplo}{t}{dg}", x, ref, np.max(abs(ref)), 1e-11)

# Fortran (kolumnowy): wybrane funkcje, opcje znakami (także małymi literami)
c, ir, fr = ctypes.c_char_p, ctypes.byref, ctypes.byref
I = lambda v: ctypes.byref(ctypes.c_int(v))  # noqa: E731
F = lambda v: ctypes.byref(ctypes.c_double(v))  # noqa: E731
M, N, K = 50, 40, 30
A, pa, lda = mac(M, K, CM); Bm, pb, ldb = mac(K, N, CM); C, pc, ldc = mac(M, N, CM)
Ad, Bd, Cd = (X.astype(np.float64) for X in (A, Bm, C))
B.dgemm_(b"N", b"n", I(M), I(N), I(K), F(1.0), p(pa), I(lda), p(pb), I(ldb), F(1.0), p(pc), I(ldc))
ocen("sgemm_", C, Ad @ Bd + Cd, abs(Ad) @ abs(Bd) + abs(Cd))
A, pa, lda = mac(M, N, CM); Ad = A.astype(np.float64); x, px, ix = wek(M, -1); y, py, iy = wek(N, 2)
X, Y = x.astype(np.float64), y.astype(np.float64)
B.dgemv_(b"T", I(M), I(N), F(2.0), p(pa), I(lda), p(px), I(ix), F(0.0), p(py), I(iy))
ocen("sgemv_ T", y, 2 * Ad.T @ X, 2 * abs(Ad.T) @ abs(X))
n = 70
A2, pa2, lda2 = mac(n, n, CM); A2[:] /= n; A2[np.diag_indices(n)] = 2
T2 = np.triu(A2.astype(np.float64)); Bm, pb, ldb = mac(n, 20, CM); Bd = Bm.astype(np.float64)
B.dtrsm_(b"L", b"u", b"T", b"N", I(n), I(20), F(1.0), p(pa2), I(lda2), p(pb), I(ldb))
ocen("strsm_", Bm, np.linalg.solve(T2.T, Bd), np.max(abs(Bd)), 1e-11)
C, pc, ldc = mac(n, n, CM); Cd = C.astype(np.float64); A, pa, lda = mac(n, 25, CM); Ad = A.astype(np.float64)
B.dsyrk_(b"L", b"N", I(n), I(25), F(1.0), p(pa), I(lda), F(0.0), p(pc), I(ldc))
ocen("ssyrk_", np.tril(C), np.tril(Ad @ Ad.T), abs(Ad) @ abs(Ad).T)

# poziom 1 CBLAS na USM
n = 100003
for kx, ky in ((1, 1), (3, 2)):
    xb = gen9blas.empty((n * kx,), np.float64); yb = gen9blas.empty((n * ky,), np.float64)
    xb[:] = rng.uniform(-1, 1, xb.shape); yb[:] = rng.uniform(-1, 1, yb.shape)
    x, y = xb[::kx], yb[::ky]; X, Y = x.astype(np.float64), y.astype(np.float64)
    px, py = xb.ctypes.data, yb.ctypes.data
    ocen(f"ddot {kx},{ky}", B.cblas_ddot(n, px, kx, py, ky), X @ Y, abs(X) @ abs(Y))
    ocen("dasum", B.cblas_dasum(n, px, kx), abs(X).sum(), abs(X).sum())
    ocen("dnrm2", B.cblas_dnrm2(n, px, kx), np.linalg.norm(X), np.linalg.norm(X))
    if B.cblas_idamax(n, px, kx) != np.argmax(abs(X)):
        bledy += 1; print("  BLAD isamax")
    B.cblas_daxpy(n, 0.5, px, kx, py, ky); ocen("daxpy", y, 0.5 * X + Y, 0.5 * abs(X) + abs(Y))
    y[:] = Y; B.cblas_dscal(n, -2.0, py, ky); ocen("dscal", y, -2 * Y, 2 * abs(Y))
    y[:] = Y; B.cblas_dcopy(n, px, kx, py, ky); ocen("dcopy", y, X, 1, 0)
    y[:] = Y; B.cblas_dswap(n, px, kx, py, ky); ocen("dswap", np.concatenate([x, y]), np.concatenate([Y, X]), 1, 0)
    x[:], y[:] = X, Y; B.cblas_drot(n, px, kx, py, ky, 0.6, 0.8)
    ocen("drot", np.concatenate([x, y]), np.concatenate([0.6 * X + 0.8 * Y, 0.6 * Y - 0.8 * X]), 2, 1e-13)
    x[:], y[:] = X, Y; P = np.array([-1.0, 0.5, -0.3, 0.2, 1.5], np.float64)
    B.cblas_drotm(n, px, kx, py, ky, P.ctypes.data)
    ocen("drotm", np.concatenate([x, y]), np.concatenate([0.5 * X + 0.2 * Y, -0.3 * X + 1.5 * Y]), 3, 1e-13)

# dziennik: każda funkcja S na GPU
os.dup2(stderr_kopia, 2)
dziennik.seek(0)
wpisy = dziennik.read().decode().splitlines()
gpu = {w.split()[1] for w in wpisy if w.endswith("-> GPU")}
cpu = {w.split()[1] for w in wpisy if w.endswith("-> CPU")}
oczekiwane = {"dgemm", "dsymm", "dsyrk", "dsyr2k", "dtrmm", "dtrsm", "dgemv", "dger", "dsymv", "dtrmv", "dtrsv", "dsyr",
              "dsyr2", "daxpy", "dscal", "dcopy", "dswap", "drot", "drotm", "ddot", "dasum", "dnrm2", "idamax"}
if oczekiwane - gpu:
    bledy += 1; print("  BLAD: nie na GPU:", sorted(oczekiwane - gpu))
if cpu:
    bledy += 1; print("  BLAD: na CPU:", sorted(cpu))
inne = [w for w in wpisy if "->" not in w]
if inne:
    print("  stderr:", inne[:5])
print(f"wywołań na GPU: {sum(w.endswith('-> GPU') for w in wpisy)}")
print("WYNIK:", "wszystko OK" if bledy == 0 else f"{bledy} błędów")
sys.exit(1 if bledy else 0)
