"""Test różnicowy pasmowych i spakowanych funkcji BLAS w libgen9blas_blas.so (GPU wymuszone) względem OpenBLAS.

Uruchom: python3 lib/test_blas_banded.py
Te same dane podawane przez CBLAS (RowMajor, ColMajor) i Fortran do libgen9blas_blas.so oraz wprost do libopenblas.so.0;
kryterium: różnica / skala ≤ 1e-5 (rozwiązania TBSV/TPSV: względna ≤ 1e-4). Sprawdza też z dziennika, że funkcje poszły
na GPU. Formaty generuje OpenBLAS-owa definicja (dane losowe w całej tablicy pasma / wektora spakowanego).
"""
import ctypes
import os
import sys
import tempfile

TU = os.path.dirname(os.path.realpath(__file__))
os.environ["GEN9BLAS_BLAS"] = "gpu"
os.environ["GEN9BLAS_BLAS_INFO"] = "1"
import numpy as np  # noqa: E402

K_ = ctypes.CDLL(os.path.join(TU, "libgen9blas_blas.so"))
O_ = ctypes.CDLL("libopenblas.so.0")
dziennik = tempfile.TemporaryFile(); kopia = os.dup(2); os.dup2(dziennik.fileno(), 2)
rng = np.random.default_rng(9)
bledy = 0
I = lambda v: ctypes.byref(ctypes.c_int(v))  # noqa: E731
F = lambda v: ctypes.byref(ctypes.c_float(v))  # noqa: E731
f, i_, p = ctypes.c_float, ctypes.c_int, ctypes.c_void_p


def por(nazwa, wynik, ref, skala, kr=1e-5):
    global bledy
    e = float(np.max(np.abs(wynik.astype(np.float64) - ref.astype(np.float64))) / max(skala, 1e-30))
    if not e <= kr:
        bledy += 1; print(f"  BLAD {nazwa}: {e:.1e}")


def oba(nazwa, wywolaj, wyjscie, kr=1e-5):
    """wywolaj(lib, dane) na kopiach danych w obu bibliotekach; porównanie tablicy wyjscie."""
    dane = {k: v.copy(order="K") for k, v in wyjscie.items()}   # z zachowaniem układu (Fortran!)
    d2 = {k: v.copy(order="K") for k, v in wyjscie.items()}
    wywolaj(K_, dane); wywolaj(O_, d2)
    for k in wyjscie:
        if k.startswith("_"):
            continue
        if np.isnan(dane[k]).any() or np.isnan(d2[k]).any():
            print(f"  NaN w {nazwa} {k}: gen9blas {np.isnan(dane[k]).sum()}, OpenBLAS {np.isnan(d2[k]).sum()}, wejście z NaN: "
                  f"{[n for n, v in wyjscie.items() if np.isnan(v).any()]}")
        por(nazwa + " " + k, dane[k], d2[k], float(np.max(np.abs(d2[k]))) + 1, kr)


RM, CM = 101, 102
for (M, N, KL, KU) in ((60, 50, 3, 5), (200, 230, 0, 9), (130, 100, 20, 0)):
    for uklad in (RM, CM):
        for t in (111, 112):
            wiersze = M if uklad == RM else N   # RowMajor: M wierszy pasma; ColMajor: N kolumn pasma
            A = rng.uniform(-1, 1, (wiersze, KL + KU + 3)).astype(np.float32); lda = KL + KU + 3
            nx, ny = (M, N) if t == 112 else (N, M)
            dane = {"_A": A, "x": rng.uniform(-1, 1, nx * 2).astype(np.float32), "y": rng.uniform(-1, 1, ny * 3).astype(np.float32)}
            oba(f"cblas_sgbmv {uklad} {t} {M}x{N}", lambda L, d: L.cblas_sgbmv(uklad, t, M, N, KL, KU, f(0.5), p(d["_A"].ctypes.data), lda,
                p(d["x"].ctypes.data), -2, f(1.5), p(d["y"].ctypes.data), 3), dane)
    A = np.asfortranarray(rng.uniform(-1, 1, (KL + KU + 2, N)).astype(np.float32))
    dane = {"_A": A, "x": rng.uniform(-1, 1, N).astype(np.float32), "y": rng.uniform(-1, 1, M).astype(np.float32)}
    oba("sgbmv_", lambda L, d: L.sgbmv_(b"N", I(M), I(N), I(KL), I(KU), F(2.0), p(d["_A"].ctypes.data), I(KL + KU + 2),
        p(d["x"].ctypes.data), I(1), F(0.0), p(d["y"].ctypes.data), I(1)), dane)
for n, K in ((70, 4), (300, 30), (50, 49)):
    for uklad in (RM, CM):
        for u in (121, 122):
            A = rng.uniform(-1, 1, (n, K + 2)).astype(np.float32); lda = K + 2
            dane = {"_A": A, "x": rng.uniform(-1, 1, n).astype(np.float32), "y": rng.uniform(-1, 1, 2 * n).astype(np.float32)}
            oba(f"cblas_ssbmv {uklad} {u} n={n}", lambda L, d: L.cblas_ssbmv(uklad, u, n, K, f(0.7), p(d["_A"].ctypes.data), lda,
                p(d["x"].ctypes.data), 1, f(-1.0), p(d["y"].ctypes.data), -2), dane)
            Ap = rng.uniform(-1, 1, n * (n + 1) // 2).astype(np.float32)
            dane = {"_Ap": Ap, "x": rng.uniform(-1, 1, 3 * n).astype(np.float32), "y": rng.uniform(-1, 1, n).astype(np.float32)}
            oba(f"cblas_sspmv {uklad} {u} n={n}", lambda L, d: L.cblas_sspmv(uklad, u, n, f(1.0), p(d["_Ap"].ctypes.data),
                p(d["x"].ctypes.data), 3, f(0.5), p(d["y"].ctypes.data), 1), dane)
            dane = {"Ap": Ap, "x": rng.uniform(-1, 1, n).astype(np.float32), "_y": rng.uniform(-1, 1, n).astype(np.float32)}
            oba(f"cblas_sspr {uklad} {u} n={n}", lambda L, d: L.cblas_sspr(uklad, u, n, f(0.3), p(d["x"].ctypes.data), 1,
                p(d["Ap"].ctypes.data)), {"Ap": Ap, "_x": dane["x"]} | {"x": dane["x"]})
            oba(f"cblas_sspr2 {uklad} {u} n={n}", lambda L, d: L.cblas_sspr2(uklad, u, n, f(0.3), p(d["x"].ctypes.data), 1,
                p(d["_y"].ctypes.data), -1, p(d["Ap"].ctypes.data)), dane)
            for t in (111, 112):
                for dg in (131, 132):
                    A = rng.uniform(-1, 1, (n, K + 2)).astype(np.float32) / max(K, 1) / 4
                    kol = K if (u == 122) == (uklad == RM) else 0   # przekątna: dolna RowMajor / górna ColMajor -> kolumna K
                    A[:, kol] = 1 + rng.uniform(0, 1, n)
                    dane = {"_A": A, "x": rng.uniform(-1, 1, 2 * n).astype(np.float32)}
                    oba(f"cblas_stbmv {uklad} {u} {t} {dg} n={n}", lambda L, d: L.cblas_stbmv(uklad, u, t, dg, n, K,
                        p(d["_A"].ctypes.data), K + 2, p(d["x"].ctypes.data), -2), dane)
                    oba(f"cblas_stbsv {uklad} {u} {t} {dg} n={n}", lambda L, d: L.cblas_stbsv(uklad, u, t, dg, n, K,
                        p(d["_A"].ctypes.data), K + 2, p(d["x"].ctypes.data), 2), dane, 1e-4)
                    Ap = (rng.uniform(-1, 1, n * (n + 1) // 2) / n).astype(np.float32)
                    gorny = (u == 121) == (uklad == RM)   # w RowMajor górna = wiersze od przekątnej; w ColMajor -- odwrotnie
                    przek = [r * n - r * (r - 1) // 2 for r in range(n)] if gorny else [r * (r + 1) // 2 + r for r in range(n)]
                    Ap[przek] = 1 + rng.uniform(0, 1, n)
                    dane = {"_Ap": Ap, "x": rng.uniform(-1, 1, n).astype(np.float32)}
                    oba(f"cblas_stpmv {uklad} {u} {t} {dg} n={n}", lambda L, d: L.cblas_stpmv(uklad, u, t, dg, n,
                        p(d["_Ap"].ctypes.data), p(d["x"].ctypes.data), 1), dane)
                    oba(f"cblas_stpsv {uklad} {u} {t} {dg} n={n}", lambda L, d: L.cblas_stpsv(uklad, u, t, dg, n,
                        p(d["_Ap"].ctypes.data), p(d["x"].ctypes.data), -1), dane, 1e-4)
    # Fortran
    A = np.asfortranarray((rng.uniform(-1, 1, (K + 1, n)) / max(K, 1) / 4).astype(np.float32))
    A[K, :] = 2   # Fortran, pasmo górne: przekątna w wierszu K
    dane = {"_A": A, "x": rng.uniform(-1, 1, n).astype(np.float32)}
    for nazwa_f in ("stbmv_", "stbsv_"):
        oba(nazwa_f, lambda L, d: getattr(L, nazwa_f)(b"U", b"T", b"N", I(n), I(K), p(d["_A"].ctypes.data), I(K + 1), p(d["x"].ctypes.data), I(1)), dane, 1e-4)
    Ap = (rng.uniform(-1, 1, n * (n + 1) // 2) / n).astype(np.float32); Ap[[j * (j + 1) // 2 + j for j in range(n)]] = 2   # górna kolumnowo
    dane = {"_Ap": Ap, "x": rng.uniform(-1, 1, n).astype(np.float32), "y": rng.uniform(-1, 1, n).astype(np.float32)}
    oba("sspmv_", lambda L, d: L.sspmv_(b"U", I(n), F(1.0), p(d["_Ap"].ctypes.data), p(d["x"].ctypes.data), I(1), F(1.0), p(d["y"].ctypes.data), I(1)), dane)
    oba("stpsv_", lambda L, d: L.stpsv_(b"u", b"n", b"n", I(n), p(d["_Ap"].ctypes.data), p(d["x"].ctypes.data), I(1)), dane, 1e-4)
    oba("ssbmv_", lambda L, d: L.ssbmv_(b"L", I(n), I(K), F(1.0), p(A.ctypes.data), I(K + 1), p(d["x"].ctypes.data), I(1), F(0.0), p(d["y"].ctypes.data), I(1)), dane)
    dane = {"Ap": Ap.copy(), "_x": rng.uniform(-1, 1, n).astype(np.float32), "_y": rng.uniform(-1, 1, n).astype(np.float32)}
    oba("sspr2_", lambda L, d: L.sspr2_(b"L", I(n), F(0.5), p(d["_x"].ctypes.data), I(1), p(d["_y"].ctypes.data), I(1), p(d["Ap"].ctypes.data)), dane)

os.dup2(kopia, 2); dziennik.seek(0)
wpisy = dziennik.read().decode().splitlines()
gpu = {w.split()[1] for w in wpisy if w.endswith("-> GPU")}
ocz = {"sgbmv", "ssbmv", "sspmv", "sspr", "sspr2", "stbmv", "stbsv", "stpmv", "stpsv"}
if ocz - gpu:
    bledy += 1; print("  BLAD: nie na GPU:", sorted(ocz - gpu))
cpu = [w for w in wpisy if w.endswith("-> CPU")]
if cpu:
    bledy += 1; print("  BLAD: na CPU:", cpu[:5])
print(f"wywołań na GPU: {sum(w.endswith('-> GPU') for w in wpisy)}")
print("WYNIK:", "wszystko OK" if bledy == 0 else f"{bledy} błędów")
sys.exit(1 if bledy else 0)
