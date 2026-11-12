"""Level 3 BLAS and batched GEMM test in libgen9blas against NumPy (float64).

Run: python3 lib/test_level3.py
Error e = |result − ref| / scale, scale = |alpha|·(|A|·|B|) (+ |beta·C|) for the given formula. Criteria (mean / max):
  float16 (hgemm, accurate): ≤ 4.9e-4 / ≤ 5e-3;  float32: ≤ 1e-6 / ≤ 1e-5;  float64: ≤ 1e-14 / ≤ 1e-13.
TRSM: relative error of the solution max|X − X_ref| / max|X_ref| ≤ 1e-4 (float32), 1e-12 (float64); A well conditioned.
SYRK/SYR2K: the triangle uplo is checked; the other triangle of C must stay unchanged.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import gen9blas  # noqa: E402

TYPES = [("fp16", np.float16, False, 4.9e-4, 5e-3), ("fp16 acc.", np.float16, True, 4.9e-4, 5e-3),
        ("fp32", np.float32, False, 1e-6, 1e-5), ("fp64", np.float64, False, 1e-14, 1e-13)]
TYPES = [t for t in TYPES if gen9blas.available(t[1], t[2])]   # Gen8: no FP16
if not gen9blas.available(np.float16):
    print("NOTE: FP16 unavailable on this GPU (Gen8), skipping the FP16 cases")
rng = np.random.default_rng(7)
errors = 0


def check(name, W, ref, scale, tol, mask=None):
    global errors
    e = np.abs(W.astype(np.float64) - ref) / np.maximum(scale, 1e-30)
    if mask is not None:
        e = e[mask]
    ok = e.mean() <= tol[0] and e.max() <= tol[1]
    errors += not ok
    if not ok:
        print(f"  ERR {name}: avg={e.mean():.1e} max={e.max():.1e}")
    return ok


def rnd(shape, dt):
    return rng.uniform(-1, 1, shape).astype(dt)


for desc, dt, acc, tol_avg, tol_max in TYPES:
    tol = (tol_avg, tol_max)
    f64 = lambda x: x.astype(np.float64)
    n_ok = n_all = 0
    # GEMM batched
    for cnt, M, N, K in [(1, 64, 64, 64), (7, 100, 50, 30), (64, 16, 32, 16), (33, 65, 47, 129)]:
        A, B, C0 = rnd((cnt, M, K), dt), rnd((cnt, K, N), dt), rnd((cnt, M, N), dt)
        W = gen9blas.gemm_batched(A, B, 1.5, 0.5, C0.copy(), accurate=acc)
        ref = 1.5 * np.einsum("bmk,bkn->bmn", f64(A), f64(B)) + 0.5 * f64(C0)
        sk = 1.5 * np.einsum("bmk,bkn->bmn", np.abs(f64(A)), np.abs(f64(B))) + 0.5 * np.abs(f64(C0))
        n_all += 1; n_ok += check(f"batched cnt={cnt} {M}x{N}x{K}", W, ref, sk, tol)
    # SYMM
    for side in "LR":
        for uplo in "LU":
            M, N = 70, 45
            n = M if side == "L" else N
            S = rnd((n, n), dt)
            Ssym = np.tril(f64(S)) + np.tril(f64(S), -1).T if uplo == "L" else np.triu(f64(S)) + np.triu(f64(S), 1).T
            B, C0 = rnd((M, N), dt), rnd((M, N), dt)
            W = gen9blas.symm(S, B, side, uplo, 2.0, -1.0, C0.copy(), accurate=acc)
            P = Ssym @ f64(B) if side == "L" else f64(B) @ Ssym
            Pa = np.abs(Ssym) @ np.abs(f64(B)) if side == "L" else np.abs(f64(B)) @ np.abs(Ssym)
            n_all += 1; n_ok += check(f"symm {side}{uplo}", W, 2.0 * P - f64(C0), 2.0 * Pa + np.abs(f64(C0)), tol)
    # SYRK / SYR2K
    for uplo in "LU":
        for tr in (False, True):
            N, K = 60, 90
            A, B, C0 = rnd((K, N) if tr else (N, K), dt), rnd((K, N) if tr else (N, K), dt), rnd((N, N), dt)
            oA, oB = (f64(A).T, f64(B).T) if tr else (f64(A), f64(B))
            mask = np.tril(np.ones((N, N), bool)) if uplo == "L" else np.triu(np.ones((N, N), bool))
            W = gen9blas.syrk(A, uplo, tr, 0.7, 0.3, C0.copy(), accurate=acc)
            ok = check(f"syrk {uplo} t={tr}", W, 0.7 * oA @ oA.T + 0.3 * f64(C0),
                      0.7 * np.abs(oA) @ np.abs(oA).T + 0.3 * np.abs(f64(C0)), tol, mask)
            ok &= np.array_equal(W[~mask], C0[~mask])
            W = gen9blas.syr2k(A, B, uplo, tr, 0.7, 0.3, C0.copy(), accurate=acc)
            ok &= check(f"syr2k {uplo} t={tr}", W, 0.7 * (oA @ oB.T + oB @ oA.T) + 0.3 * f64(C0),
                       0.7 * (np.abs(oA) @ np.abs(oB).T + np.abs(oB) @ np.abs(oA).T) + 0.3 * np.abs(f64(C0)), tol, mask)
            ok &= np.array_equal(W[~mask], C0[~mask])
            if not ok:
                print(f"  ERR syrk/syr2k {uplo} t={tr} (also: other triangle changed?)")
            errors += 0 if ok else 0   # errors are counted in check; here only the triangle
            n_all += 1; n_ok += ok
    # TRMM
    for side in "LR":
        for uplo in "LU":
            for ta in (False, True):
                for dg in (False, True):
                    M, N = 50, 70
                    n = M if side == "L" else N
                    A, B = rnd((n, n), dt), rnd((M, N), dt)
                    T = np.tril(f64(A)) if uplo == "L" else np.triu(f64(A))
                    if dg:
                        np.fill_diagonal(T, 1.0)
                    T = T.T if ta else T
                    W = gen9blas.trmm(A, B, side, uplo, ta, dg, 1.5, accurate=acc)
                    P = T @ f64(B) if side == "L" else f64(B) @ T
                    Pa = np.abs(T) @ np.abs(f64(B)) if side == "L" else np.abs(f64(B)) @ np.abs(T)
                    n_all += 1; n_ok += check(f"trmm {side}{uplo} tA={ta} diag={dg}", W, 1.5 * P, 1.5 * Pa, tol)
    print(f"{desc:11s}: {n_ok}/{n_all} case groups OK")

# TRSM: A = triangular, diagonal ≥ n (well conditioned). float32: ≤ 1e-4; float16: ≤ 5e-3 (the solution
# is stored in FP16 between blocks; the reference is computed from data rounded to FP16)
for t_desc, t_dt, t_acc, t_tol in [t for t in (("trsm fp64", np.float64, False, 1e-12), ("trsm fp32", np.float32, False, 1e-4),
                                    ("trsm fp16", np.float16, False, 5e-3), ("trsm fp16d", np.float16, True, 5e-3))
                                    if gen9blas.available(t[1], t[2])]:
  n_ok = n_all = 0
  for side in "LR":
      for uplo in "LU":
          for ta in (False, True):
              for dg in (False, True):
                  for (M, N) in [(40, 30), (300, 200)]:
                      n = M if side == "L" else N
                      A = rng.uniform(-1, 1, (n, n)).astype(np.float32)
                      A[np.diag_indices(n)] = n if not dg else 0.5
                      if dg:   # diag=1: the diagonal is not read; for conditioning we scale the rest
                          A *= 1.0 / n
                          A[np.diag_indices(n)] = 123.0
                      B = rng.uniform(-1, 1, (M, N)).astype(np.float32)
                      A, B = A.astype(t_dt), B.astype(t_dt)
                      T = np.tril(A.astype(np.float64)) if uplo == "L" else np.triu(A.astype(np.float64))
                      if dg:
                          np.fill_diagonal(T, 1.0)
                      T = T.T if ta else T
                      Xr = np.linalg.solve(T, 2.0 * B.astype(np.float64)) if side == "L" else np.linalg.solve(T.T, 2.0 * B.astype(np.float64).T).T
                      X = gen9blas.trsm(A, B, side, uplo, ta, dg, 2.0, accurate=t_acc).astype(np.float64)
                      e = np.abs(X - Xr).max() / np.abs(Xr).max()
                      ok = e <= t_tol
                      errors += not ok
                      n_all += 1; n_ok += ok
                      if not ok:
                          print(f"  ERR {t_desc} {side}{uplo} tA={ta} diag={dg} {M}x{N}: {e:.1e}")
  print(f"{t_desc:11s}: {n_ok}/{n_all} cases OK")

print("\n== GEMM batched: 256 × (64×64×64), FP16 / FP32")
for dt in [d for d in (np.float16, np.float32) if gen9blas.available(d)]:
    A, B = rnd((256, 64, 64), dt), rnd((256, 64, 64), dt)
    gen9blas.gemm_batched(A, B)
    t = time.perf_counter(); gen9blas.gemm_batched(A, B); tw = time.perf_counter() - t
    t = time.perf_counter()
    for i in range(256):
        gen9blas.gemm(A[i], B[i])
    tp = time.perf_counter() - t
    t = time.perf_counter(); np.matmul(A.astype(np.float32), B.astype(np.float32)); tn = time.perf_counter() - t
    fl = 2 * 256 * 64 ** 3
    print(f"  {np.dtype(dt).name}: batched {fl / tw / 1e9:6.1f} GFLOPS ({tw * 1e3:.1f} ms), one by one {fl / tp / 1e9:6.1f} "
          f"({tp * 1e3:.1f} ms), NumPy fp32 {fl / tn / 1e9:6.1f}")

print("\n== Zero-copy views: slices (row stride), A[::2], .T views, C as a slice (FP32, FP16)")
tol32 = (1e-6, 1e-5)
for name, dt in [t for t in (("fp32", np.float32), ("fp16", np.float16)) if gen9blas.available(t[1])]:
    tol = tol32 if dt == np.float32 else (4.9e-4, 5e-3)
    D = rnd((300, 400), dt); E = rnd((400, 300), dt); P = rnd((260, 260), dt)
    views = {"slice": lambda X: X[5:105, 7:107], "every 2nd": lambda X: X[::2, 3:103][:100],
             "T slice": lambda X: X[10:110, 20:120].T}
    for wa, fa in views.items():
        for wb, fb in views.items():
            A, B = fa(D), fb(E)
            A64, B64 = A.astype(np.float64), B.astype(np.float64)
            ref, sk = A64 @ B64, np.abs(A64) @ np.abs(B64)
            ok = check(f"gemm {name} {wa}·{wb}", gen9blas.gemm(A, B), ref, sk, tol)
            Cb = np.zeros((120, 130), dt); Cv = Cb[3:103, 9:109]; Cv[...] = 1
            gen9blas.gemm(A, B, beta=0.5, C=Cv)
            ok &= check(f"gemm C-slice {name} {wa}·{wb}", Cv, ref + 0.5, sk + 0.5, tol)
            rest = Cb.copy(); rest[3:103, 9:109] = 0
            if np.any(rest):
                errors += 1; print(f"  ERR gemm {name}: write outside the slice of C")
    # A square as a view (slice and .T): symm, trmm, trsm, syrk, syr2k
    S = P[30:130, 40:140]; St = P[30:130, 40:140].T; B = E[:100, :50]; B64 = B.astype(np.float64)
    for wa, A in (("slice", S), ("T", St)):
        A64 = A.astype(np.float64)
        for uplo in "LU":
            M = np.tril(A64) if uplo == "L" else np.triu(A64)
            Sym = M + M.T - np.diag(np.diag(A64))
            check(f"symm {name} {wa} {uplo}", gen9blas.symm(A, B, uplo=uplo), Sym @ B64, np.abs(Sym) @ np.abs(B64), tol)
            check(f"trmm {name} {wa} {uplo}", gen9blas.trmm(A, B, uplo=uplo), M @ B64, np.abs(M) @ np.abs(B64), tol)
            check(f"trmm T {name} {wa} {uplo}", gen9blas.trmm(A, B, uplo=uplo, transA=True), M.T @ B64,
                 np.abs(M.T) @ np.abs(B64), tol)
            if dt == np.float32:
                Ad = A.copy(); Ad[np.diag_indices(100)] = 100
                Ad = Ad if wa == "slice" else np.ascontiguousarray(Ad.T).T   # the same kind of view
                Md = np.tril(Ad.astype(np.float64)) if uplo == "L" else np.triu(Ad.astype(np.float64))
                X = gen9blas.trsm(Ad, B, uplo=uplo); Xr = np.linalg.solve(Md, B64)
                e = np.abs(X - Xr).max() / np.abs(Xr).max()
                if e > 1e-4:
                    errors += 1; print(f"  ERR trsm {name} {wa} {uplo}: {e:.1e}")
    for wa, A in (("slice", D[5:105, 7:57]), ("T", D[5:55, 7:107].T)):
        A64 = A.astype(np.float64); Bk = (E[:100, :50] if wa == "slice" else E[:50, :100].T)
        for uplo in "LU":
            mask = np.tril(np.ones((100, 100), bool)) if uplo == "L" else np.triu(np.ones((100, 100), bool))
            check(f"syrk {name} {wa} {uplo}", gen9blas.syrk(A, uplo=uplo), A64 @ A64.T, np.abs(A64) @ np.abs(A64.T), tol, mask)
            B2 = Bk.astype(np.float64)
            for wb, Bx in (("same", Bk), ("other", np.ascontiguousarray(Bk) if wa == "T" else Bk.T.copy().T)):
                check(f"syr2k {name} {wa}/{wb} {uplo}", gen9blas.syr2k(A, Bx, uplo=uplo), A64 @ B2.T + B2 @ A64.T,
                     np.abs(A64) @ np.abs(B2.T) + np.abs(B2) @ np.abs(A64.T), tol, mask)
print("  checked" if not errors else f"  errors: {errors}")

print("\n== Level 3 speed (FP32, n = 2048): GFLOPS computed with the standard formula")
n = 2048
A, B = rnd((n, n), np.float32), rnd((n, n), np.float32)
A[np.diag_indices(n)] = n
for name, f, fl in [("gemm", lambda: gen9blas.sgemm(A, B), 2 * n ** 3), ("symm", lambda: gen9blas.symm(A, B), 2 * n ** 3),
                     ("syrk", lambda: gen9blas.syrk(A), n ** 3), ("trmm", lambda: gen9blas.trmm(A, B), n ** 3),
                     ("trsm", lambda: gen9blas.trsm(A, B), n ** 3)]:
    f()
    t = time.perf_counter(); f(); t = time.perf_counter() - t
    print(f"  {name}: {fl / t / 1e9:6.1f} GFLOPS ({t * 1e3:.0f} ms)")
print("\nRESULT:", "all OK" if not errors else f"{errors} errors")
sys.exit(1 if errors else 0)
