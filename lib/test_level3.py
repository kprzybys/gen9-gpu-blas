"""Test BLAS poziomu 3 i GEMM wsadowego w libgen9blas względem NumPy (float64).

Uruchom: python3 lib/test_level3.py
Błąd e = |wynik − ref| / skala, skala = |alpha|·(|A|·|B|) (+ |beta·C|) dla danego wzoru. Kryteria (średnia / maks.):
  float16 (hgemm, dokładny): ≤ 4,9e-4 / ≤ 5e-3;  float32: ≤ 1e-6 / ≤ 1e-5;  float64: ≤ 1e-14 / ≤ 1e-13.
TRSM: błąd względny rozwiązania max|X − X_ref| / max|X_ref| ≤ 1e-4 (float32), 1e-12 (float64); A dobrze uwarunkowana.
SYRK/SYR2K: sprawdzany trójkąt uplo; drugi trójkąt C musi zostać bez zmian.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import gen9blas  # noqa: E402

TYPY = [("fp16", np.float16, False, 4.9e-4, 5e-3), ("fp16 dokl.", np.float16, True, 4.9e-4, 5e-3),
        ("fp32", np.float32, False, 1e-6, 1e-5), ("fp64", np.float64, False, 1e-14, 1e-13)]
rng = np.random.default_rng(7)
bledy = 0


def ocen(nazwa, W, ref, skala, kr, maska=None):
    global bledy
    e = np.abs(W.astype(np.float64) - ref) / np.maximum(skala, 1e-30)
    if maska is not None:
        e = e[maska]
    ok = e.mean() <= kr[0] and e.max() <= kr[1]
    bledy += not ok
    if not ok:
        print(f"  BLAD {nazwa}: sr={e.mean():.1e} maks={e.max():.1e}")
    return ok


def los(shape, typ):
    return rng.uniform(-1, 1, shape).astype(typ)


for opis, typ, dokl, ksr, kmx in TYPY:
    kr = (ksr, kmx)
    f64 = lambda x: x.astype(np.float64)
    n_ok = n_all = 0
    # GEMM wsadowy
    for cnt, M, N, K in [(1, 64, 64, 64), (7, 100, 50, 30), (64, 16, 32, 16), (33, 65, 47, 129)]:
        A, B, C0 = los((cnt, M, K), typ), los((cnt, K, N), typ), los((cnt, M, N), typ)
        W = gen9blas.gemm_batched(A, B, 1.5, 0.5, C0.copy(), accurate=dokl)
        ref = 1.5 * np.einsum("bmk,bkn->bmn", f64(A), f64(B)) + 0.5 * f64(C0)
        sk = 1.5 * np.einsum("bmk,bkn->bmn", np.abs(f64(A)), np.abs(f64(B))) + 0.5 * np.abs(f64(C0))
        n_all += 1; n_ok += ocen(f"wsadowy cnt={cnt} {M}x{N}x{K}", W, ref, sk, kr)
    # SYMM
    for side in "LR":
        for uplo in "LU":
            M, N = 70, 45
            n = M if side == "L" else N
            S = los((n, n), typ)
            Ssym = np.tril(f64(S)) + np.tril(f64(S), -1).T if uplo == "L" else np.triu(f64(S)) + np.triu(f64(S), 1).T
            B, C0 = los((M, N), typ), los((M, N), typ)
            W = gen9blas.symm(S, B, side, uplo, 2.0, -1.0, C0.copy(), accurate=dokl)
            P = Ssym @ f64(B) if side == "L" else f64(B) @ Ssym
            Pa = np.abs(Ssym) @ np.abs(f64(B)) if side == "L" else np.abs(f64(B)) @ np.abs(Ssym)
            n_all += 1; n_ok += ocen(f"symm {side}{uplo}", W, 2.0 * P - f64(C0), 2.0 * Pa + np.abs(f64(C0)), kr)
    # SYRK / SYR2K
    for uplo in "LU":
        for tr in (False, True):
            N, K = 60, 90
            A, B, C0 = los((K, N) if tr else (N, K), typ), los((K, N) if tr else (N, K), typ), los((N, N), typ)
            oA, oB = (f64(A).T, f64(B).T) if tr else (f64(A), f64(B))
            maska = np.tril(np.ones((N, N), bool)) if uplo == "L" else np.triu(np.ones((N, N), bool))
            W = gen9blas.syrk(A, uplo, tr, 0.7, 0.3, C0.copy(), accurate=dokl)
            ok = ocen(f"syrk {uplo} t={tr}", W, 0.7 * oA @ oA.T + 0.3 * f64(C0),
                      0.7 * np.abs(oA) @ np.abs(oA).T + 0.3 * np.abs(f64(C0)), kr, maska)
            ok &= np.array_equal(W[~maska], C0[~maska])
            W = gen9blas.syr2k(A, B, uplo, tr, 0.7, 0.3, C0.copy(), accurate=dokl)
            ok &= ocen(f"syr2k {uplo} t={tr}", W, 0.7 * (oA @ oB.T + oB @ oA.T) + 0.3 * f64(C0),
                       0.7 * (np.abs(oA) @ np.abs(oB).T + np.abs(oB) @ np.abs(oA).T) + 0.3 * np.abs(f64(C0)), kr, maska)
            ok &= np.array_equal(W[~maska], C0[~maska])
            if not ok:
                print(f"  BLAD syrk/syr2k {uplo} t={tr} (także: drugi trójkąt zmieniony?)")
            bledy += 0 if ok else 0   # błędy liczone w ocen; tu tylko trójkąt
            n_all += 1; n_ok += ok
    # TRMM
    for side in "LR":
        for uplo in "LU":
            for ta in (False, True):
                for dg in (False, True):
                    M, N = 50, 70
                    n = M if side == "L" else N
                    A, B = los((n, n), typ), los((M, N), typ)
                    T = np.tril(f64(A)) if uplo == "L" else np.triu(f64(A))
                    if dg:
                        np.fill_diagonal(T, 1.0)
                    T = T.T if ta else T
                    W = gen9blas.trmm(A, B, side, uplo, ta, dg, 1.5, accurate=dokl)
                    P = T @ f64(B) if side == "L" else f64(B) @ T
                    Pa = np.abs(T) @ np.abs(f64(B)) if side == "L" else np.abs(f64(B)) @ np.abs(T)
                    n_all += 1; n_ok += ocen(f"trmm {side}{uplo} tA={ta} diag={dg}", W, 1.5 * P, 1.5 * Pa, kr)
    print(f"{opis:11s}: {n_ok}/{n_all} grup przypadków OK")

# TRSM: A = trójkątna, przekątna ≥ n (dobrze uwarunkowana). float32: ≤ 1e-4; float16: ≤ 5e-3 (rozwiązanie
# przechowywane w FP16 między blokami; odniesienie liczone z danych zaokrąglonych do FP16)
for t_opis, t_typ, t_dokl, t_kr in (("trsm fp64", np.float64, False, 1e-12), ("trsm fp32", np.float32, False, 1e-4), ("trsm fp16", np.float16, False, 5e-3),
                                    ("trsm fp16d", np.float16, True, 5e-3)):
  n_ok = n_all = 0
  for side in "LR":
      for uplo in "LU":
          for ta in (False, True):
              for dg in (False, True):
                  for (M, N) in [(40, 30), (300, 200)]:
                      n = M if side == "L" else N
                      A = rng.uniform(-1, 1, (n, n)).astype(np.float32)
                      A[np.diag_indices(n)] = n if not dg else 0.5
                      if dg:   # diag=1: przekątna nie jest czytana; dla uwarunkowania skalujemy resztę
                          A *= 1.0 / n
                          A[np.diag_indices(n)] = 123.0
                      B = rng.uniform(-1, 1, (M, N)).astype(np.float32)
                      A, B = A.astype(t_typ), B.astype(t_typ)
                      T = np.tril(A.astype(np.float64)) if uplo == "L" else np.triu(A.astype(np.float64))
                      if dg:
                          np.fill_diagonal(T, 1.0)
                      T = T.T if ta else T
                      Xr = np.linalg.solve(T, 2.0 * B.astype(np.float64)) if side == "L" else np.linalg.solve(T.T, 2.0 * B.astype(np.float64).T).T
                      X = gen9blas.trsm(A, B, side, uplo, ta, dg, 2.0, accurate=t_dokl).astype(np.float64)
                      e = np.abs(X - Xr).max() / np.abs(Xr).max()
                      ok = e <= t_kr
                      bledy += not ok
                      n_all += 1; n_ok += ok
                      if not ok:
                          print(f"  BLAD {t_opis} {side}{uplo} tA={ta} diag={dg} {M}x{N}: {e:.1e}")
  print(f"{t_opis:11s}: {n_ok}/{n_all} przypadków OK")

print("\n== GEMM wsadowy: 256 × (64×64×64), FP16 / FP32")
for typ in (np.float16, np.float32):
    A, B = los((256, 64, 64), typ), los((256, 64, 64), typ)
    gen9blas.gemm_batched(A, B)
    t = time.perf_counter(); gen9blas.gemm_batched(A, B); tw = time.perf_counter() - t
    t = time.perf_counter()
    for i in range(256):
        gen9blas.gemm(A[i], B[i])
    tp = time.perf_counter() - t
    t = time.perf_counter(); np.matmul(A.astype(np.float32), B.astype(np.float32)); tn = time.perf_counter() - t
    fl = 2 * 256 * 64 ** 3
    print(f"  {np.dtype(typ).name}: wsadowo {fl / tw / 1e9:6.1f} GFLOPS ({tw * 1e3:.1f} ms), pojedynczo {fl / tp / 1e9:6.1f} "
          f"({tp * 1e3:.1f} ms), NumPy fp32 {fl / tn / 1e9:6.1f}")

print("\n== Widoki bez kopii: wycinki (krok wierszy), A[::2], widoki .T, C jako wycinek (FP32, FP16)")
kr32 = (1e-6, 1e-5)
for nazwa, typ in (("fp32", np.float32), ("fp16", np.float16)):
    kr = kr32 if typ == np.float32 else (4.9e-4, 5e-3)
    D = los((300, 400), typ); E = los((400, 300), typ); P = los((260, 260), typ)
    widoki = {"wycinek": lambda X: X[5:105, 7:107], "co drugi": lambda X: X[::2, 3:103][:100],
              "T wycinka": lambda X: X[10:110, 20:120].T}
    for wa, fa in widoki.items():
        for wb, fb in widoki.items():
            A, B = fa(D), fb(E)
            A64, B64 = A.astype(np.float64), B.astype(np.float64)
            ref, sk = A64 @ B64, np.abs(A64) @ np.abs(B64)
            ok = ocen(f"gemm {nazwa} {wa}·{wb}", gen9blas.gemm(A, B), ref, sk, kr)
            Cb = np.zeros((120, 130), typ); Cv = Cb[3:103, 9:109]; Cv[...] = 1
            gen9blas.gemm(A, B, beta=0.5, C=Cv)
            ok &= ocen(f"gemm C-wycinek {nazwa} {wa}·{wb}", Cv, ref + 0.5, sk + 0.5, kr)
            reszta = Cb.copy(); reszta[3:103, 9:109] = 0
            if np.any(reszta):
                bledy += 1; print(f"  BLAD gemm {nazwa}: zapis poza wycinkiem C")
    # A kwadratowa jako widok (wycinek i .T): symm, trmm, trsm, syrk, syr2k
    S = P[30:130, 40:140]; St = P[30:130, 40:140].T; B = E[:100, :50]; B64 = B.astype(np.float64)
    for wa, A in (("wycinek", S), ("T", St)):
        A64 = A.astype(np.float64)
        for uplo in "LU":
            M = np.tril(A64) if uplo == "L" else np.triu(A64)
            Sym = M + M.T - np.diag(np.diag(A64))
            ocen(f"symm {nazwa} {wa} {uplo}", gen9blas.symm(A, B, uplo=uplo), Sym @ B64, np.abs(Sym) @ np.abs(B64), kr)
            ocen(f"trmm {nazwa} {wa} {uplo}", gen9blas.trmm(A, B, uplo=uplo), M @ B64, np.abs(M) @ np.abs(B64), kr)
            ocen(f"trmm T {nazwa} {wa} {uplo}", gen9blas.trmm(A, B, uplo=uplo, transA=True), M.T @ B64,
                 np.abs(M.T) @ np.abs(B64), kr)
            if typ == np.float32:
                Ad = A.copy(); Ad[np.diag_indices(100)] = 100
                Ad = Ad if wa == "wycinek" else np.ascontiguousarray(Ad.T).T   # ten sam rodzaj widoku
                Md = np.tril(Ad.astype(np.float64)) if uplo == "L" else np.triu(Ad.astype(np.float64))
                X = gen9blas.trsm(Ad, B, uplo=uplo); Xr = np.linalg.solve(Md, B64)
                e = np.abs(X - Xr).max() / np.abs(Xr).max()
                if e > 1e-4:
                    bledy += 1; print(f"  BLAD trsm {nazwa} {wa} {uplo}: {e:.1e}")
    for wa, A in (("wycinek", D[5:105, 7:57]), ("T", D[5:55, 7:107].T)):
        A64 = A.astype(np.float64); Bk = (E[:100, :50] if wa == "wycinek" else E[:50, :100].T)
        for uplo in "LU":
            mask = np.tril(np.ones((100, 100), bool)) if uplo == "L" else np.triu(np.ones((100, 100), bool))
            ocen(f"syrk {nazwa} {wa} {uplo}", gen9blas.syrk(A, uplo=uplo), A64 @ A64.T, np.abs(A64) @ np.abs(A64.T), kr, mask)
            B2 = Bk.astype(np.float64)
            for wb, Bx in (("ten sam", Bk), ("inny", np.ascontiguousarray(Bk) if wa == "T" else Bk.T.copy().T)):
                ocen(f"syr2k {nazwa} {wa}/{wb} {uplo}", gen9blas.syr2k(A, Bx, uplo=uplo), A64 @ B2.T + B2 @ A64.T,
                     np.abs(A64) @ np.abs(B2.T) + np.abs(B2) @ np.abs(A64.T), kr, mask)
print("  sprawdzone" if not bledy else f"  błędów: {bledy}")

print("\n== Szybkość poziomu 3 (FP32, n = 2048): GFLOPS liczone wzorem standardowym")
n = 2048
A, B = los((n, n), np.float32), los((n, n), np.float32)
A[np.diag_indices(n)] = n
for nazwa, f, fl in [("gemm", lambda: gen9blas.sgemm(A, B), 2 * n ** 3), ("symm", lambda: gen9blas.symm(A, B), 2 * n ** 3),
                     ("syrk", lambda: gen9blas.syrk(A), n ** 3), ("trmm", lambda: gen9blas.trmm(A, B), n ** 3),
                     ("trsm", lambda: gen9blas.trsm(A, B), n ** 3)]:
    f()
    t = time.perf_counter(); f(); t = time.perf_counter() - t
    print(f"  {nazwa}: {fl / t / 1e9:6.1f} GFLOPS ({t * 1e3:.0f} ms)")
print("\nWYNIK:", "wszystko OK" if not bledy else f"{bledy} błędów")
sys.exit(1 if bledy else 0)
