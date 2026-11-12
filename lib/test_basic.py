"""libgen9blas test: correctness (types, transposes, zero-copy, alpha/beta, odd sizes) and speed.

Run: make -C lib test
Relative error e = |C - ref| / (|alpha|·Σ|a·b| + |beta·c|). Criteria (mean / maximum):
  hgemm:          ≤ 4.9e-4 (1 FP16 rounding) / ≤ 5e-3
  hgemm accurate: ≤ 4.9e-4 / ≤ 5e-3 for K < 256 (a short sum fits in one FP16 chunk),
                  for K ≥ 256: ≤ 1e-4 / ≤ 1e-3, i.e. clearly better than hgemm (1.4e-4 / 1.7-2.4e-3)
  sgemm:          ≤ 1e-6 / ≤ 1e-5
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import gen9blas  # noqa: E402

KINDS = [("hgemm", np.float16, dict(), 4.9e-4, 5e-3), ("sgemm", np.float32, dict(), 1e-6, 1e-5)]
gen9blas._init()
if gen9blas._lib.gen9blas_available(gen9blas.GEN9BLAS_HS):
    KINDS.insert(1, ("hgemm acc.", np.float16, dict(accurate=True), 4.9e-4, 5e-3))
else:
    print("NOTE: accurate variant not built (asm/hgemm_kc_build.sh), skipping")


def call(dt, opts, A, B, al, be, C):
    f = gen9blas.sgemm if dt == np.float32 else gen9blas.hgemm
    return f(A, B, al, be, C, **opts)


def new_array(shape, dt, zc):
    """zc=1: gen9blas.empty (USM); zc=0: plain NumPy array, deliberately unaligned (userptr or copy)."""
    if zc:
        return gen9blas.empty(shape, dt)
    t = np.zeros(shape[0] * shape[1] + 64, dt)
    off = 32 if (t.ctypes.data % 4096) == 0 else 0
    return t[off:off + shape[0] * shape[1]].reshape(shape)


def case(dt, opts, M, N, K, al, be, ta, tb, zc, seed=0):
    rng = np.random.default_rng(seed)
    a = new_array((K, M) if ta else (M, K), dt, zc); a[...] = rng.uniform(-1, 1, a.shape)
    b = new_array((N, K) if tb else (K, N), dt, zc); b[...] = rng.uniform(-1, 1, b.shape)
    A, B = (a.T if ta else a), (b.T if tb else b)
    C0 = rng.uniform(-1, 1, (M, N)).astype(dt)
    C = new_array((M, N), dt, zc)
    C[...] = C0 if be else 0
    W = call(dt, opts, A, B, al, be, C)
    a64, b64 = A.astype(np.float64), B.astype(np.float64)
    ref = al * a64 @ b64 + (be * C0.astype(np.float64) if be else 0)
    scale = abs(al) * np.abs(a64) @ np.abs(b64) + (abs(be) * np.abs(C0.astype(np.float64)) if be else 0)
    e = np.abs(W.astype(np.float64) - ref) / scale
    return e.mean(), e.max(), gen9blas.last_zero_copy()


# Without NEO plain memory goes through userptr (zero-copy); GEN9BLAS_NO_USERPTR=1 or libgen9blas_neo.so: copy by the CPU.
USERPTR = not os.environ.get("GEN9BLAS_NO_USERPTR") and "neo" not in os.path.basename(gen9blas._lib._name)
errors = 0
for name, dt, opts, tol_avg, tol_max in KINDS:
    print(f"\n== {name}")
    for (M, N, K, al, be) in [(1, 1, 1, 1, 0), (17, 33, 15, 1, 0), (100, 70, 300, 2.0, 0), (64, 96, 48, 0.5, 1.0),
                              (333, 129, 1000, 1, -1.0), (1000, 1024, 777, 1, 0.5)]:
        for ta, tb in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            for zc in (0, 1):
                avg, mx, z = case(dt, opts, M, N, K, al, be, ta, tb, zc)
                if opts.get("accurate") and K >= 256:
                    ok = avg <= 1e-4 and mx <= 1e-3
                else:
                    ok = avg <= tol_avg and mx <= tol_max
                ok = ok and (z == 7 if zc or USERPTR else z == 0)   # without NEO: plain arrays are zero-copy too (userptr)
                errors += not ok
                if not ok or (ta, tb, zc) == (0, 0, 0):
                    print(f"{'OK ' if ok else 'ERR '} M={M:5d} N={N:5d} K={K:5d} a={al:4} b={be:4} tA={ta} tB={tb} "
                          f"zero-copy={z:03b}  e: avg={avg:.1e} max={mx:.1e}")
    print("   (checked all 4 transpose combinations × copy / zero-copy for each size)")

print("\n== Speed: GFLOPS of the whole call (copy / zero-copy) and of the kernel alone")
for name, dt, opts, _, _ in KINDS:
    for n in (1024, 2048, 4096):
        res = []
        for zc in (0, 1):
            A = new_array((n, n), dt, zc); A[...] = np.random.default_rng(1).uniform(-1, 1, (n, n))
            B = new_array((n, n), dt, zc); B[...] = np.random.default_rng(2).uniform(-1, 1, (n, n))
            C = new_array((n, n), dt, zc)
            call(dt, opts, A, B, 1.0, 0.0, C)
            best = 1e9
            for _ in range(3):
                t = time.perf_counter(); call(dt, opts, A, B, 1.0, 0.0, C); best = min(best, time.perf_counter() - t)
            res.append(2 * n**3 / best / 1e9)
        print(f"  {name:12s} N={n:5d}: copy {res[0]:6.1f}, zero-copy {res[1]:6.1f}, kernel {2 * n**3 / gen9blas.last_time_ms() / 1e6:6.1f}")
print("\nRESULT:", "all OK" if not errors else f"{errors} errors")
sys.exit(1 if errors else 0)
