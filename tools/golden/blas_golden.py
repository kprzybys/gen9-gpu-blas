#!/usr/bin/env python3
"""Golden outputs of the BLAS layer: sha256 of the result bytes for a fixed set of calls with seeded inputs.

A refactor that does not change kernels or host arithmetic must reproduce every hash bit for bit. Each call runs twice;
a call whose two results differ is marked NONDET (its hash is then not a valid golden value).
usage: python3 tools/golden/blas_golden.py [LIBDIR=lib]   ->  lines "name  sha256[:16]  [NONDET]"
"""
import hashlib, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, sys.argv[1] if len(sys.argv) > 1 else "lib"))
import gen9blas as g  # noqa: E402


def data(seed, shape, dt):
    return np.random.default_rng(seed).standard_normal(shape).astype(dt)


def spd(n, dt, seed):
    a = data(seed, (n, n), np.float64)
    return (a @ a.T / n + np.eye(n)).astype(dt)


def tri(n, dt, seed):
    return (np.tril(data(seed, (n, n), np.float64)) * 0.1 + np.eye(n) * 2).astype(dt)


CASES = []
for dt, tag in ((np.float16, "h"), (np.float32, "s")):
    for (m, n, k) in ((64, 64, 64), (129, 77, 300), (512, 384, 1000), (1024, 1024, 1024), (8, 4096, 1024), (2048, 16, 512)):
        CASES.append((f"{tag}gemm {m}x{n}x{k}",
                      lambda m=m, n=n, k=k, dt=dt: g.gemm(data(1, (m, k), dt), data(2, (k, n), dt), 1.0, 0.0)))
        CASES.append((f"{tag}gemm a/b {m}x{n}x{k}",
                      lambda m=m, n=n, k=k, dt=dt: g.gemm(data(1, (m, k), dt), data(2, (k, n), dt), 0.5, 2.0,
                                                          data(3, (m, n), dt))))
    CASES += [
        (f"{tag}symm 300", lambda dt=dt: g.symm(spd(300, dt, 4), data(5, (300, 200), dt))),
        (f"{tag}syrk 300", lambda dt=dt: g.syrk(data(6, (300, 250), dt))),
        (f"{tag}syr2k 300", lambda dt=dt: g.syr2k(data(6, (300, 250), dt), data(7, (300, 250), dt))),
        (f"{tag}trmm 300", lambda dt=dt: g.trmm(tri(300, dt, 8), data(9, (300, 200), dt))),
        (f"{tag}trsm 300", lambda dt=dt: g.trsm(tri(300, dt, 8), data(9, (300, 200), dt))),
        (f"{tag}gemm_batched 4x96", lambda dt=dt: g.gemm_batched(data(10, (4, 96, 80), dt), data(11, (4, 80, 112), dt))),
    ]
CASES.append(("hgemm accurate 512x384x1000",
              lambda: g.gemm(data(1, (512, 1000), np.float16), data(2, (1000, 384), np.float16), accurate=True)))
for dt, tag in ((np.float32, "s"), (np.float64, "d")):
    n = 4097
    CASES += [
        (f"{tag}axpy", lambda dt=dt: g.axpy(0.7, data(20, n, dt), data(21, n, dt).copy())),
        (f"{tag}dot", lambda dt=dt: np.asarray(g.dot(data(20, n, dt), data(21, n, dt)))),
        (f"{tag}nrm2", lambda dt=dt: np.asarray(g.nrm2(data(20, n, dt)))),
        (f"{tag}asum", lambda dt=dt: np.asarray(g.asum(data(20, n, dt)))),
        (f"{tag}iamax", lambda dt=dt: np.asarray(g.iamax(data(20, n, dt)))),
        (f"{tag}gemv", lambda dt=dt: g.gemv(data(22, (700, 900), dt), data(23, 900, dt))),
        (f"{tag}gemv T", lambda dt=dt: g.gemv(data(22, (700, 900), dt), data(24, 700, dt), trans=True)),
        (f"{tag}symv", lambda dt=dt: g.symv(spd(500, dt, 25), data(26, 500, dt))),
        (f"{tag}trmv", lambda dt=dt: g.trmv(tri(500, dt, 27), data(26, 500, dt).copy())),
        (f"{tag}trsv", lambda dt=dt: g.trsv(tri(500, dt, 27), data(26, 500, dt).copy())),
        (f"{tag}ger", lambda dt=dt: g.ger(0.3, data(28, 600, dt), data(29, 500, dt), data(30, (600, 500), dt).copy())),
    ]


def digest(x):
    a = np.ascontiguousarray(np.asarray(x))
    return hashlib.sha256(a.dtype.str.encode() + str(a.shape).encode() + a.tobytes()).hexdigest()[:16]


def main():
    g._init()
    bad = 0
    for name, fn in CASES:
        if name.startswith("h") and not g.available(np.float16):
            print(f"{name:32s} SKIP (no FP16)"); continue
        try:
            h1, h2 = digest(fn()), digest(fn())
        except Exception as e:  # an API change shows up here, not as a silent pass
            print(f"{name:32s} ERROR {type(e).__name__}: {e}"); bad += 1; continue
        print(f"{name:32s} {h1}" + ("" if h1 == h2 else f"  NONDET {h2}"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
