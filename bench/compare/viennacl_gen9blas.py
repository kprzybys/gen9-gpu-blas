"""Test ViennaCL 1.7.1 dense_blas (profil OpenBenchmarking pts/viennacl) powtórzony na libgen9blas: te same operacje, rozmiary,
miary i metoda pomiaru. Dane odniesienia: results/zewnetrzne/openbenchmarking-2103278-IB-KABYLAKET61/.

Uruchom: python3 bench/compare/viennacl_gen9blas.py [serie]   (domyślnie 3 serie, jak PTS)
Metoda jak examples/benchmarks/dense_blas.cpp: operacja raz na rozgrzewkę, potem powtarzana, aż łączny czas ≥ 1 s;
wynik = średni czas jednego wywołania (z czekaniem na koniec). Rozmiary: BLAS1 N = 10⁷, GEMV 3840 × 3840,
GEMM 1976³. Miary: COPY 2N·e, AXPY 3N·e, DOT 2N·e bajtów; GEMV (M + N + M·N)·e bajtów; GEMM 2·M·N·K flop.
Dane w pamięci USM (gen9blas.empty) -- odpowiednik buforów OpenCL w ViennaCL (dane już na GPU).
ViennaCL: macierze kolumnowe (column_major). Kolumnowa X (m×n) = wierszowa Xᵀ w tej samej pamięci, więc:
  GEMV-N  y = A·x       -> gen9blas.gemv(a, x, trans=True)   (a = zapis wierszowy, czyli Aᵀ)
  GEMV-T  x = Aᵀ·y      -> gen9blas.gemv(a, y)
  GEMM    C = op1(A)·op2(B) kolumnowo  ->  Cᵀ = op2(B)ᵀ·op1(A)ᵀ wierszowo; NN: gemm(b, a), NT: gemm(btᵀ, a),
          TN: gemm(b, atᵀ), TT: gemm(btᵀ, atᵀ)   (b, a, bt, at -- zapisy wierszowe; ᵀ = widok .T, bez kopii)
AXPY w ViennaCL: x = y + α·x (ten sam ruch pamięci co axpy BLAS); tu gen9blas.axpy(α, y, x) = x + α·y, ruch 3N·e jak tam.
"""
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "lib"))
import gen9blas  # noqa: E402

N1, M2, N2, M3 = 10_000_000, 3840, 3840, 1976
rng = np.random.default_rng(0)


def los(shape, typ):
    X = gen9blas.empty(shape, typ)
    X[...] = rng.random(shape).astype(typ)
    return X


def bench(f):
    f()
    n, t = 0, 0.0
    while t < 1.0:
        t0 = time.perf_counter(); f(); t += time.perf_counter() - t0; n += 1
    return t / n


def seria(typ, pref):
    e = np.dtype(typ).itemsize
    w = {}
    x, y = los(N1, typ), los(N1, typ)
    w["COPY"] = 2 * N1 * e / bench(lambda: gen9blas.copy(y, x)) * 1e-9
    w["AXPY"] = 3 * N1 * e / bench(lambda: gen9blas.axpy(2.4, y, x)) * 1e-9
    w["DOT"] = 2 * N1 * e / bench(lambda: gen9blas.dot(x, y)) * 1e-9
    del x, y
    a, xv, yv = los((N2, M2), typ), los(N2, typ), los(M2, typ)   # a = zapis A (M2 × N2 kolumnowo)
    b2 = (M2 + N2 + M2 * N2) * e
    w["GEMV-N"] = b2 / bench(lambda: gen9blas.gemv(a, xv, y=yv, trans=True)) * 1e-9
    w["GEMV-T"] = b2 / bench(lambda: gen9blas.gemv(a, yv, y=xv)) * 1e-9
    del a
    fl = 2.0 * M3 ** 3
    A, B, AT, BT = (los((M3, M3), typ) for _ in range(4))
    C = gen9blas.empty((M3, M3), typ)
    fn = gen9blas.hgemm if typ == np.float16 else gen9blas.gemm
    for nazwa, (p, q) in {"NN": (B, A), "NT": (BT.T, A), "TN": (B, AT.T), "TT": (BT.T, AT.T)}.items():
        w["GEMM-" + nazwa] = fl / bench(lambda: fn(p, q, C=C)) * 1e-9
    for k, v in w.items():
        print(f"{pref}{k} : {v:.3g} {'GFLOPs/s' if 'GEMM' in k else 'GB/s'}", flush=True)


serie = int(sys.argv[1]) if len(sys.argv) > 1 else 3
for s in range(serie):
    print(f"# seria {s + 1}", flush=True)
    seria(np.float32, "s")
    seria(np.float64, "d")
    seria(np.float16, "h")   # poza testem ViennaCL (brak FP16) -- dla porównania z naszą najmocniejszą stroną
