"""Skinny GEMM (small M): time, weight GB/s, GFLOPS, CPU and error -- libgen9blas (FP16, FP32) and OpenBLAS SGEMM (CPU reference).
Plan and model: docs/skinny-gemm.md. Results: results/skinny-gemm-2026-11-12.txt.

Run: python3 bench/skinny_gemm.py [usm|plain] [label]   (GEN9BLAS_SKINNY=0 disables the skinny kernel -- the old path)
C (M×N) = A (M×K) · Wᵀ (weights N×K, "W") or A · B (weights K×N, "B"). GB/s = weight bytes / time (weights read once).
Error: max |C − C_fp64| / max |C_fp64| (the FP16 inputs are exactly the same in both).
"""
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # before NumPy: OpenBLAS threads (the FP64 reference) inflated the CPU
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "lib"))
import gen9blas  # noqa: E402

MEM = sys.argv[1] if len(sys.argv) > 1 else "usm"
LABEL = sys.argv[2] if len(sys.argv) > 2 else ""
SHAPES = [(4096, 4096), (1024, 1024), (2048, 2048), (8192, 8192), (11008, 4096), (4096, 11008), (5120, 1280), (51866, 1280)]
MS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
rng = np.random.default_rng(1)


def make_array(shape, dtype):
    X = gen9blas.empty(shape, dtype) if MEM == "usm" else np.empty(shape, dtype)
    X[...] = rng.uniform(-1, 1, shape).astype(dtype)
    return X


def measure(f, budget=0.25):
    for _ in range(2):
        f()
    t0 = time.perf_counter(); f(); t1 = time.perf_counter() - t0
    R = int(min(200, max(3, budget / max(t1, 1e-5))))
    c, t = time.process_time(), time.perf_counter()
    for _ in range(R):
        f()
    return (time.perf_counter() - t) / R * 1e3, (time.process_time() - c) / R * 1e3


print(f"# memory {MEM} {LABEL}; time [ms] / weight GB/s / GFLOPS / CPU [ms] / error")
for dtype, name in ((np.float16, "H"), (np.float32, "S")):
    e = np.dtype(dtype).itemsize
    fn = gen9blas.hgemm if dtype == np.float16 else gen9blas.sgemm
    for N, K in SHAPES:
        if N * K * e > 400e6:
            continue
        W = make_array((N, K), dtype); B = make_array((K, N), dtype)
        Wd, Bd = W.astype(np.float64), B.astype(np.float64)
        for M in MS:
            A = make_array((M, K), dtype); Ad = A.astype(np.float64)
            for layout, f, ref in (("W", lambda: fn(A, W.T), lambda: Ad @ Wd.T), ("B", lambda: fn(A, B), lambda: Ad @ Bd)):
                t, c = measure(f)
                C = f().astype(np.float64); R = ref()
                err = np.max(np.abs(C - R)) / np.max(np.abs(R))
                print(f"{name} {layout} N={N:6d} K={K:6d} M={M:4d}  {t:8.3f} ms  {N*K*e/t/1e6:6.1f} GB/s  "
                      f"{2*M*N*K/t/1e6:7.1f} GFLOPS  CPU {c:6.3f} ms  err {err:.1e}", flush=True)
        del W, B, Wd, Bd
