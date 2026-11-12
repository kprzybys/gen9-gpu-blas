"""ViennaCL 1.7.1 dense_blas test (OpenBenchmarking profile pts/viennacl) repeated on libgen9blas: same operations, sizes,
metrics and measurement method. Reference data: https://openbenchmarking.org/result/2103278-IB-KABYLAKET61 (not redistributed here).

Run: python3 bench/compare/viennacl_gen9blas.py [series]   (default 3 series, as in PTS)
Method as in examples/benchmarks/dense_blas.cpp: the operation once as warm-up, then repeated until the total time is ≥ 1 s;
result = mean time of one call (waiting for completion). Sizes: BLAS1 N = 10⁷, GEMV 3840 × 3840,
GEMM 1976³. Metrics: COPY 2N·e, AXPY 3N·e, DOT 2N·e bytes; GEMV (M + N + M·N)·e bytes; GEMM 2·M·N·K flop.
Data in USM memory (gen9blas.empty) -- the equivalent of ViennaCL OpenCL buffers (data already on the GPU).
ViennaCL: column-major matrices. A column-major X (m×n) = a row-major Xᵀ in the same memory, so:
  GEMV-N  y = A·x       -> gen9blas.gemv(a, x, trans=True)   (a = row-major storage, i.e. Aᵀ)
  GEMV-T  x = Aᵀ·y      -> gen9blas.gemv(a, y)
  GEMM    C = op1(A)·op2(B) column-major  ->  Cᵀ = op2(B)ᵀ·op1(A)ᵀ row-major; NN: gemm(b, a), NT: gemm(btᵀ, a),
          TN: gemm(b, atᵀ), TT: gemm(btᵀ, atᵀ)   (b, a, bt, at -- row-major storage; ᵀ = the .T view, zero-copy)
AXPY in ViennaCL: x = y + α·x (the same memory traffic as BLAS axpy); here gen9blas.axpy(α, y, x) = x + α·y, traffic 3N·e as there.
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


def rnd(shape, dtype):
    X = gen9blas.empty(shape, dtype)
    X[...] = rng.random(shape).astype(dtype)
    return X


def bench(f):
    f()
    n, t = 0, 0.0
    while t < 1.0:
        t0 = time.perf_counter(); f(); t += time.perf_counter() - t0; n += 1
    return t / n


def run_series(dtype, pref):
    e = np.dtype(dtype).itemsize
    w = {}
    x, y = rnd(N1, dtype), rnd(N1, dtype)
    w["COPY"] = 2 * N1 * e / bench(lambda: gen9blas.copy(y, x)) * 1e-9
    w["AXPY"] = 3 * N1 * e / bench(lambda: gen9blas.axpy(2.4, y, x)) * 1e-9
    w["DOT"] = 2 * N1 * e / bench(lambda: gen9blas.dot(x, y)) * 1e-9
    del x, y
    a, xv, yv = rnd((N2, M2), dtype), rnd(N2, dtype), rnd(M2, dtype)   # a = storage of A (M2 × N2 column-major)
    b2 = (M2 + N2 + M2 * N2) * e
    w["GEMV-N"] = b2 / bench(lambda: gen9blas.gemv(a, xv, y=yv, trans=True)) * 1e-9
    w["GEMV-T"] = b2 / bench(lambda: gen9blas.gemv(a, yv, y=xv)) * 1e-9
    del a
    fl = 2.0 * M3 ** 3
    A, B, AT, BT = (rnd((M3, M3), dtype) for _ in range(4))
    C = gen9blas.empty((M3, M3), dtype)
    fn = gen9blas.hgemm if dtype == np.float16 else gen9blas.gemm
    for name, (p, q) in {"NN": (B, A), "NT": (BT.T, A), "TN": (B, AT.T), "TT": (BT.T, AT.T)}.items():
        w["GEMM-" + name] = fl / bench(lambda: fn(p, q, C=C)) * 1e-9
    for k, v in w.items():
        print(f"{pref}{k} : {v:.3g} {'GFLOPs/s' if 'GEMM' in k else 'GB/s'}", flush=True)


series = int(sys.argv[1]) if len(sys.argv) > 1 else 3
for s in range(series):
    print(f"# series {s + 1}", flush=True)
    run_series(np.float32, "s")
    run_series(np.float64, "d")
    run_series(np.float16, "h")   # outside the ViennaCL test (no FP16 there) -- for comparison with our strongest side
