"""Chudy GEMM (małe M): czas, GB/s wag, GFLOPS, CPU i błąd -- libgen9blas (FP16, FP32) i OpenBLAS SGEMM (odniesienie CPU).
Plan i model: PLAN_CHUDY_GEMM.md. Wyniki: results/skinny-gemm-2026-11-12.txt.

Uruchom: python3 bench/skinny_gemm.py [usm|zwykla] [etykieta]   (GEN9BLAS_SKINNY=0 wyłącza chude jądro -- dawna ścieżka)
C (M×N) = A (M×K) · Wᵀ (wagi N×K, „W”) albo A · B (wagi K×N, „B”). GB/s = bajty wag / czas (wagi czytane raz).
Błąd: max |C − C_fp64| / max |C_fp64| (wejścia FP16 dokładnie te same w obu).
"""
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # przed NumPy: wątki OpenBLAS (odniesienie FP64) zawyżały CPU
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "lib"))
import gen9blas  # noqa: E402

PAM = sys.argv[1] if len(sys.argv) > 1 else "usm"
ETYK = sys.argv[2] if len(sys.argv) > 2 else ""
KSZT = [(4096, 4096), (1024, 1024), (2048, 2048), (8192, 8192), (11008, 4096), (4096, 11008), (5120, 1280), (51866, 1280)]
MS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
rng = np.random.default_rng(1)


def tab(shape, typ):
    X = gen9blas.empty(shape, typ) if PAM == "usm" else np.empty(shape, typ)
    X[...] = rng.uniform(-1, 1, shape).astype(typ)
    return X


def mierz(f, budzet=0.25):
    for _ in range(2):
        f()
    t0 = time.perf_counter(); f(); t1 = time.perf_counter() - t0
    R = int(min(200, max(3, budzet / max(t1, 1e-5))))
    c, t = time.process_time(), time.perf_counter()
    for _ in range(R):
        f()
    return (time.perf_counter() - t) / R * 1e3, (time.process_time() - c) / R * 1e3


print(f"# pamięć {PAM} {ETYK}; czas [ms] / GB/s wag / GFLOPS / CPU [ms] / błąd")
for typ, nazwa in ((np.float16, "H"), (np.float32, "S")):
    e = np.dtype(typ).itemsize
    fn = gen9blas.hgemm if typ == np.float16 else gen9blas.sgemm
    for N, K in KSZT:
        if N * K * e > 400e6:
            continue
        W = tab((N, K), typ); B = tab((K, N), typ)
        Wd, Bd = W.astype(np.float64), B.astype(np.float64)
        for M in MS:
            A = tab((M, K), typ); Ad = A.astype(np.float64)
            for uk, f, ref in (("W", lambda: fn(A, W.T), lambda: Ad @ Wd.T), ("B", lambda: fn(A, B), lambda: Ad @ Bd)):
                t, c = mierz(f)
                C = f().astype(np.float64); R = ref()
                bl = np.max(np.abs(C - R)) / np.max(np.abs(R))
                print(f"{nazwa} {uk} N={N:6d} K={K:6d} M={M:4d}  {t:8.3f} ms  {N*K*e/t/1e6:6.1f} GB/s  "
                      f"{2*M*N*K/t/1e6:7.1f} GFLOPS  CPU {c:6.3f} ms  błąd {bl:.1e}", flush=True)
        del W, B, Wd, Bd
