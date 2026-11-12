"""Test libgen9blas: poprawność (rodzaje, transpozycje, zero-copy, alpha/beta, krzywe rozmiary) i szybkość.

Uruchom: make -C lib test
Błąd względny e = |C - ref| / (|alpha|·Σ|a·b| + |beta·c|). Kryteria (średnia / maksimum):
  hgemm:          ≤ 4,9e-4 (1 zaokrąglenie FP16) / ≤ 5e-3
  hgemm dokładny: ≤ 4,9e-4 / ≤ 5e-3 dla K < 256 (krótka suma mieści się w jednej porcji FP16),
                  dla K ≥ 256: ≤ 1e-4 / ≤ 1e-3, czyli wyraźnie lepiej niż hgemm (1,4e-4 / 1,7–2,4e-3)
  sgemm:          ≤ 1e-6 / ≤ 1e-5
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import gen9blas  # noqa: E402

RODZAJE = [("hgemm", np.float16, dict(), 4.9e-4, 5e-3), ("sgemm", np.float32, dict(), 1e-6, 1e-5)]
gen9blas._init()
if gen9blas._lib.gen9blas_available(gen9blas.GEN9BLAS_HS):
    RODZAJE.insert(1, ("hgemm dokl.", np.float16, dict(accurate=True), 4.9e-4, 5e-3))
else:
    print("UWAGA: wariant dokładny niezbudowany (asm/hgemm_kc_build.sh), pomijam")


def wywolaj(typ, opcje, A, B, al, be, C):
    f = gen9blas.sgemm if typ == np.float32 else gen9blas.hgemm
    return f(A, B, al, be, C, **opcje)


def nowa_tablica(shape, typ, zc):
    """zc=1: gen9blas.empty (USM); zc=0: zwykła tablica NumPy, celowo niewyrównana (userptr albo kopia)."""
    if zc:
        return gen9blas.empty(shape, typ)
    t = np.zeros(shape[0] * shape[1] + 64, typ)
    off = 32 if (t.ctypes.data % 4096) == 0 else 0
    return t[off:off + shape[0] * shape[1]].reshape(shape)


def przypadek(typ, opcje, M, N, K, al, be, ta, tb, zc, seed=0):
    rng = np.random.default_rng(seed)
    a = nowa_tablica((K, M) if ta else (M, K), typ, zc); a[...] = rng.uniform(-1, 1, a.shape)
    b = nowa_tablica((N, K) if tb else (K, N), typ, zc); b[...] = rng.uniform(-1, 1, b.shape)
    A, B = (a.T if ta else a), (b.T if tb else b)
    C0 = rng.uniform(-1, 1, (M, N)).astype(typ)
    C = nowa_tablica((M, N), typ, zc)
    C[...] = C0 if be else 0
    W = wywolaj(typ, opcje, A, B, al, be, C)
    a64, b64 = A.astype(np.float64), B.astype(np.float64)
    ref = al * a64 @ b64 + (be * C0.astype(np.float64) if be else 0)
    skala = abs(al) * np.abs(a64) @ np.abs(b64) + (abs(be) * np.abs(C0.astype(np.float64)) if be else 0)
    e = np.abs(W.astype(np.float64) - ref) / skala
    return e.mean(), e.max(), gen9blas.last_zero_copy()


# Bez NEO zwykła pamięć idzie przez userptr (bez kopii); GEN9BLAS_NO_USERPTR=1 albo libgen9blas_neo.so: kopia przez CPU.
USERPTR = not os.environ.get("GEN9BLAS_NO_USERPTR") and "neo" not in os.path.basename(gen9blas._lib._name)
bledy = 0
for nazwa, typ, opcje, kr_sr, kr_mx in RODZAJE:
    print(f"\n== {nazwa}")
    for (M, N, K, al, be) in [(1, 1, 1, 1, 0), (17, 33, 15, 1, 0), (100, 70, 300, 2.0, 0), (64, 96, 48, 0.5, 1.0),
                              (333, 129, 1000, 1, -1.0), (1000, 1024, 777, 1, 0.5)]:
        for ta, tb in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            for zc in (0, 1):
                sr, mx, z = przypadek(typ, opcje, M, N, K, al, be, ta, tb, zc)
                if opcje.get("accurate") and K >= 256:
                    ok = sr <= 1e-4 and mx <= 1e-3
                else:
                    ok = sr <= kr_sr and mx <= kr_mx
                ok = ok and (z == 7 if zc or USERPTR else z == 0)   # bez NEO: zwykłe tablice też bez kopii (userptr)
                bledy += not ok
                if not ok or (ta, tb, zc) == (0, 0, 0):
                    print(f"{'OK ' if ok else 'BLAD'} M={M:5d} N={N:5d} K={K:5d} a={al:4} b={be:4} tA={ta} tB={tb} "
                          f"zero-copy={z:03b}  e: sr={sr:.1e} maks={mx:.1e}")
    print("   (sprawdzone wszystkie 4 kombinacje transpozycji × kopia / zero-copy dla każdego rozmiaru)")

print("\n== Szybkość: GFLOPS całego wywołania (kopia / zero-copy) i samego kernela")
for nazwa, typ, opcje, _, _ in RODZAJE:
    for n in (1024, 2048, 4096):
        wyn = []
        for zc in (0, 1):
            A = nowa_tablica((n, n), typ, zc); A[...] = np.random.default_rng(1).uniform(-1, 1, (n, n))
            B = nowa_tablica((n, n), typ, zc); B[...] = np.random.default_rng(2).uniform(-1, 1, (n, n))
            C = nowa_tablica((n, n), typ, zc)
            wywolaj(typ, opcje, A, B, 1.0, 0.0, C)
            best = 1e9
            for _ in range(3):
                t = time.perf_counter(); wywolaj(typ, opcje, A, B, 1.0, 0.0, C); best = min(best, time.perf_counter() - t)
            wyn.append(2 * n**3 / best / 1e9)
        print(f"  {nazwa:12s} N={n:5d}: kopia {wyn[0]:6.1f}, zero-copy {wyn[1]:6.1f}, kernel {2 * n**3 / gen9blas.last_time_ms() / 1e6:6.1f}")
print("\nWYNIK:", "wszystko OK" if not bledy else f"{bledy} błędów")
sys.exit(1 if bledy else 0)
