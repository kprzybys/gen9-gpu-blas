"""Test chudego GEMM (małe M albo N, bez pakowania; PLAN_CHUDY_GEMM.md) wobec NumPy w FP64.

Uruchom: python3 lib/test_skinny.py   (wymusza chude jądro dla min(M, N) ≤ 64 przez GEN9BLAS_SKINNY=64, niezależnie od progu
wyboru w gen9blas.c; dawną ścieżkę sprawdzają test_basic i test_level3)
Przypadki: oba układy wag (A·Wᵀ i A·B), transpozycje A i B (widoki .T), małe N (zamiana ról), alpha/beta, C jako wycinek
(krok wierszy), rozmiary nieparzyste i ogony K (K mod 8, K mod 128), M wokół kafli (1, 3, 4, 5, 8, 9, 16, 17, 33, 64),
pamięć USM (gen9blas.empty) i zwykła (NumPy). FP16 i FP32.
"""
import os
import sys

import numpy as np

os.environ.setdefault("GEN9BLAS_SKINNY", "64")   # przed załadowaniem biblioteki (czytane raz)
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import gen9blas  # noqa: E402

rng = np.random.default_rng(7)
TOL = {np.float16: 4e-3, np.float32: 2e-6}
zle = n_test = 0


def tab(shape, typ, usm):
    X = gen9blas.empty(shape, typ) if usm else np.empty(shape, typ)
    X[...] = rng.uniform(-1, 1, shape).astype(typ)
    return X


def sprawdz(opis, typ, wynik, ref):
    global zle, n_test
    n_test += 1
    e = np.max(np.abs(wynik.astype(np.float64) - ref)) / max(1e-30, np.max(np.abs(ref)))
    if not np.isfinite(e) or e > TOL[typ]:
        zle += 1
        print(f"ZLE {opis}: błąd {e:.2e}")


for typ in (np.float16, np.float32):
    fn = gen9blas.hgemm if typ == np.float16 else gen9blas.sgemm
    nz = np.dtype(typ).name
    for usm in (True, False):
        pm = "usm" if usm else "zwykla"
        for M, N, K in [(1, 1000, 777), (3, 517, 1029), (4, 2048, 128), (5, 64, 9), (8, 333, 4096), (9, 1500, 1500),
                        (16, 4096, 1000), (17, 130, 257), (33, 700, 300), (64, 256, 513), (1, 8, 3), (2, 7, 2000)]:
            for tA in (False, True):
                for tB in (False, True):
                    A = tab((K, M), typ, usm).T if tA else tab((M, K), typ, usm)
                    B = tab((N, K), typ, usm).T if tB else tab((K, N), typ, usm)
                    Ad, Bd = A.astype(np.float64), B.astype(np.float64)
                    opis = f"{nz} {pm} M={M} N={N} K={K} tA={tA} tB={tB}"
                    sprawdz(opis, typ, fn(A, B), Ad @ Bd)
                    # małe N: zamiana ról (C = Bᵀ-strona), te same operandy w odwrotnej kolejności
                    sprawdz(opis + " (małe N)", typ, fn(B.T, A.T), Bd.T @ Ad.T)
            # alpha, beta, C jako wycinek większej tablicy (krok wierszy)
            A = tab((M, K), typ, usm); W = tab((N, K), typ, usm)
            duza = tab((M + 2, N + 5), typ, usm); C = duza[1:M + 1, 2:N + 2]
            C0 = C.astype(np.float64); D0 = duza.copy()
            fn(A, W.T, alpha=0.75, beta=-1.5, C=C)
            sprawdz(f"{nz} {pm} M={M} N={N} K={K} alpha/beta, C wycinek", typ, C,
                    0.75 * A.astype(np.float64) @ W.astype(np.float64).T - 1.5 * C0)
            maska = np.ones(duza.shape, bool); maska[1:M + 1, 2:N + 2] = False
            if not np.array_equal(duza[maska], D0[maska]):   # poza wycinkiem C nic nie może się zmienić
                zle += 1; print(f"ZLE {nz} {pm} M={M} N={N}: zmiana poza wycinkiem C")
print(f"testów: {n_test}")
print("WYNIK: wszystko OK" if not zle else f"WYNIK: {zle} błędów")
sys.exit(1 if zle else 0)
