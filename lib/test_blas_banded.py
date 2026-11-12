"""Differential test of the banded and packed BLAS functions in libgen9blas_blas.so (GPU forced) against OpenBLAS.

Run: python3 lib/test_blas_banded.py
The same data passed through CBLAS (RowMajor, ColMajor) and Fortran to libgen9blas_blas.so and directly to libopenblas.so.0;
criterion: difference / scale ≤ 1e-5 (TBSV/TPSV solutions: relative ≤ 1e-4). Also checks in the log that the functions went
to the GPU. The formats follow the OpenBLAS definition (random data in the whole band array / packed vector).
"""
import ctypes
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.realpath(__file__))
os.environ["GEN9BLAS_BLAS"] = "gpu"
os.environ["GEN9BLAS_BLAS_INFO"] = "1"
import numpy as np  # noqa: E402

K_ = ctypes.CDLL(os.path.join(HERE, "libgen9blas_blas.so"))
O_ = ctypes.CDLL("libopenblas.so.0")
log = tempfile.TemporaryFile(); saved_stderr = os.dup(2); os.dup2(log.fileno(), 2)
rng = np.random.default_rng(9)
errors = 0
I = lambda v: ctypes.byref(ctypes.c_int(v))  # noqa: E731
F = lambda v: ctypes.byref(ctypes.c_float(v))  # noqa: E731
f, i_, p = ctypes.c_float, ctypes.c_int, ctypes.c_void_p


def cmp(name, result, ref, scale, tol=1e-5):
    global errors
    e = float(np.max(np.abs(result.astype(np.float64) - ref.astype(np.float64))) / max(scale, 1e-30))
    if not e <= tol:
        errors += 1; print(f"  ERR {name}: {e:.1e}")


def both(name, call, out, tol=1e-5):
    """call(lib, data) on copies of the data in both libraries; comparison of the array out."""
    data = {k: v.copy(order="K") for k, v in out.items()}   # preserving the layout (Fortran!)
    d2 = {k: v.copy(order="K") for k, v in out.items()}
    call(K_, data); call(O_, d2)
    for k in out:
        if k.startswith("_"):
            continue
        if np.isnan(data[k]).any() or np.isnan(d2[k]).any():
            print(f"  NaN in {name} {k}: gen9blas {np.isnan(data[k]).sum()}, OpenBLAS {np.isnan(d2[k]).sum()}, input with NaN: "
                  f"{[n for n, v in out.items() if np.isnan(v).any()]}")
        cmp(name + " " + k, data[k], d2[k], float(np.max(np.abs(d2[k]))) + 1, tol)


RM, CM = 101, 102
for (M, N, KL, KU) in ((60, 50, 3, 5), (200, 230, 0, 9), (130, 100, 20, 0)):
    for layout in (RM, CM):
        for t in (111, 112):
            rows = M if layout == RM else N   # RowMajor: M band rows; ColMajor: N band columns
            A = rng.uniform(-1, 1, (rows, KL + KU + 3)).astype(np.float32); lda = KL + KU + 3
            nx, ny = (M, N) if t == 112 else (N, M)
            data = {"_A": A, "x": rng.uniform(-1, 1, nx * 2).astype(np.float32), "y": rng.uniform(-1, 1, ny * 3).astype(np.float32)}
            both(f"cblas_sgbmv {layout} {t} {M}x{N}", lambda L, d: L.cblas_sgbmv(layout, t, M, N, KL, KU, f(0.5), p(d["_A"].ctypes.data), lda,
                p(d["x"].ctypes.data), -2, f(1.5), p(d["y"].ctypes.data), 3), data)
    A = np.asfortranarray(rng.uniform(-1, 1, (KL + KU + 2, N)).astype(np.float32))
    data = {"_A": A, "x": rng.uniform(-1, 1, N).astype(np.float32), "y": rng.uniform(-1, 1, M).astype(np.float32)}
    both("sgbmv_", lambda L, d: L.sgbmv_(b"N", I(M), I(N), I(KL), I(KU), F(2.0), p(d["_A"].ctypes.data), I(KL + KU + 2),
        p(d["x"].ctypes.data), I(1), F(0.0), p(d["y"].ctypes.data), I(1)), data)
for n, K in ((70, 4), (300, 30), (50, 49)):
    for layout in (RM, CM):
        for u in (121, 122):
            A = rng.uniform(-1, 1, (n, K + 2)).astype(np.float32); lda = K + 2
            data = {"_A": A, "x": rng.uniform(-1, 1, n).astype(np.float32), "y": rng.uniform(-1, 1, 2 * n).astype(np.float32)}
            both(f"cblas_ssbmv {layout} {u} n={n}", lambda L, d: L.cblas_ssbmv(layout, u, n, K, f(0.7), p(d["_A"].ctypes.data), lda,
                p(d["x"].ctypes.data), 1, f(-1.0), p(d["y"].ctypes.data), -2), data)
            Ap = rng.uniform(-1, 1, n * (n + 1) // 2).astype(np.float32)
            data = {"_Ap": Ap, "x": rng.uniform(-1, 1, 3 * n).astype(np.float32), "y": rng.uniform(-1, 1, n).astype(np.float32)}
            both(f"cblas_sspmv {layout} {u} n={n}", lambda L, d: L.cblas_sspmv(layout, u, n, f(1.0), p(d["_Ap"].ctypes.data),
                p(d["x"].ctypes.data), 3, f(0.5), p(d["y"].ctypes.data), 1), data)
            data = {"Ap": Ap, "x": rng.uniform(-1, 1, n).astype(np.float32), "_y": rng.uniform(-1, 1, n).astype(np.float32)}
            both(f"cblas_sspr {layout} {u} n={n}", lambda L, d: L.cblas_sspr(layout, u, n, f(0.3), p(d["x"].ctypes.data), 1,
                p(d["Ap"].ctypes.data)), {"Ap": Ap, "_x": data["x"]} | {"x": data["x"]})
            both(f"cblas_sspr2 {layout} {u} n={n}", lambda L, d: L.cblas_sspr2(layout, u, n, f(0.3), p(d["x"].ctypes.data), 1,
                p(d["_y"].ctypes.data), -1, p(d["Ap"].ctypes.data)), data)
            for t in (111, 112):
                for dg in (131, 132):
                    A = rng.uniform(-1, 1, (n, K + 2)).astype(np.float32) / max(K, 1) / 4
                    col = K if (u == 122) == (layout == RM) else 0   # diagonal: lower RowMajor / upper ColMajor -> column K
                    A[:, col] = 1 + rng.uniform(0, 1, n)
                    data = {"_A": A, "x": rng.uniform(-1, 1, 2 * n).astype(np.float32)}
                    both(f"cblas_stbmv {layout} {u} {t} {dg} n={n}", lambda L, d: L.cblas_stbmv(layout, u, t, dg, n, K,
                        p(d["_A"].ctypes.data), K + 2, p(d["x"].ctypes.data), -2), data)
                    both(f"cblas_stbsv {layout} {u} {t} {dg} n={n}", lambda L, d: L.cblas_stbsv(layout, u, t, dg, n, K,
                        p(d["_A"].ctypes.data), K + 2, p(d["x"].ctypes.data), 2), data, 1e-4)
                    Ap = (rng.uniform(-1, 1, n * (n + 1) // 2) / n).astype(np.float32)
                    upper = (u == 121) == (layout == RM)   # in RowMajor upper = rows from the diagonal; in ColMajor -- the opposite
                    diag_idx = [r * n - r * (r - 1) // 2 for r in range(n)] if upper else [r * (r + 1) // 2 + r for r in range(n)]
                    Ap[diag_idx] = 1 + rng.uniform(0, 1, n)
                    data = {"_Ap": Ap, "x": rng.uniform(-1, 1, n).astype(np.float32)}
                    both(f"cblas_stpmv {layout} {u} {t} {dg} n={n}", lambda L, d: L.cblas_stpmv(layout, u, t, dg, n,
                        p(d["_Ap"].ctypes.data), p(d["x"].ctypes.data), 1), data)
                    both(f"cblas_stpsv {layout} {u} {t} {dg} n={n}", lambda L, d: L.cblas_stpsv(layout, u, t, dg, n,
                        p(d["_Ap"].ctypes.data), p(d["x"].ctypes.data), -1), data, 1e-4)
    # Fortran
    A = np.asfortranarray((rng.uniform(-1, 1, (K + 1, n)) / max(K, 1) / 4).astype(np.float32))
    A[K, :] = 2   # Fortran, upper band: diagonal in row K
    data = {"_A": A, "x": rng.uniform(-1, 1, n).astype(np.float32)}
    for fname in ("stbmv_", "stbsv_"):
        both(fname, lambda L, d: getattr(L, fname)(b"U", b"T", b"N", I(n), I(K), p(d["_A"].ctypes.data), I(K + 1), p(d["x"].ctypes.data), I(1)), data, 1e-4)
    Ap = (rng.uniform(-1, 1, n * (n + 1) // 2) / n).astype(np.float32); Ap[[j * (j + 1) // 2 + j for j in range(n)]] = 2   # upper, column-major
    data = {"_Ap": Ap, "x": rng.uniform(-1, 1, n).astype(np.float32), "y": rng.uniform(-1, 1, n).astype(np.float32)}
    both("sspmv_", lambda L, d: L.sspmv_(b"U", I(n), F(1.0), p(d["_Ap"].ctypes.data), p(d["x"].ctypes.data), I(1), F(1.0), p(d["y"].ctypes.data), I(1)), data)
    both("stpsv_", lambda L, d: L.stpsv_(b"u", b"n", b"n", I(n), p(d["_Ap"].ctypes.data), p(d["x"].ctypes.data), I(1)), data, 1e-4)
    both("ssbmv_", lambda L, d: L.ssbmv_(b"L", I(n), I(K), F(1.0), p(A.ctypes.data), I(K + 1), p(d["x"].ctypes.data), I(1), F(0.0), p(d["y"].ctypes.data), I(1)), data)
    data = {"Ap": Ap.copy(), "_x": rng.uniform(-1, 1, n).astype(np.float32), "_y": rng.uniform(-1, 1, n).astype(np.float32)}
    both("sspr2_", lambda L, d: L.sspr2_(b"L", I(n), F(0.5), p(d["_x"].ctypes.data), I(1), p(d["_y"].ctypes.data), I(1), p(d["Ap"].ctypes.data)), data)

os.dup2(saved_stderr, 2); log.seek(0)
entries = log.read().decode().splitlines()
gpu = {w.split()[1] for w in entries if w.endswith("-> GPU")}
expected = {"sgbmv", "ssbmv", "sspmv", "sspr", "sspr2", "stbmv", "stbsv", "stpmv", "stpsv"}
if expected - gpu:
    errors += 1; print("  ERR: not on the GPU:", sorted(expected - gpu))
cpu = [w for w in entries if w.endswith("-> CPU")]
if cpu:
    errors += 1; print("  ERR: on the CPU:", cpu[:5])
print(f"calls on the GPU: {sum(w.endswith('-> GPU') for w in entries)}")
print("RESULT:", "all OK" if errors == 0 else f"{errors} errors")
sys.exit(1 if errors else 0)
