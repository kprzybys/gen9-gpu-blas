"""Step 3 (2026-11-12): GEMV N bandwidth (row-major y = A·x) in libgen9blas, grouped variant vs the old one (GEN9BLAS_GEMV_N=old).
Run: python3 bench/gemv_n.py [usm|plain]   (both variants alternating in separate processes, 3 rounds, median GB/s).
usm: gen9blas.empty arrays; plain: NumPy arrays (userptr, 4 KB pages -- the old variant lost 2× here, docs/no-neo.md, docs/pitfalls.md)."""
import os
import subprocess
import sys
import time

import numpy as np

SIZES = [(1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192), (256, 65536), (65536, 256), (16384, 512),
        (8192, 1024), (4096, 2048), (51866, 1280), (1280, 5120)]
MEM = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("usm", "plain") else "usm"


def child(mem):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "lib"))
    import gen9blas
    for M, N in SIZES:
        alloc = gen9blas.empty if mem == "usm" else np.empty
        A = alloc((M, N), np.float32); A[:] = 1e-3
        x = alloc(N, np.float32); x[:] = 1; y = alloc(M, np.float32)
        for _ in range(3):
            gen9blas.gemv(A, x, y=y)
        R = max(5, int(0.3 / max(1e-5, M * N * 4 / 15e9)))
        R = min(R, 500)
        t = time.perf_counter()
        for _ in range(R):
            gen9blas.gemv(A, x, y=y)
        t = (time.perf_counter() - t) / R
        assert abs(float(y[0]) - N * 1e-3) < 1e-2 * N * 1e-3, (M, N, float(y[0]))
        print(M, N, (M * N + M + N) * 4 / t / 1e9, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "-":
        child(sys.argv[2]); sys.exit()
    w = {}
    for r in range(3):
        for variant in (["grouped", "old"] if r % 2 == 0 else ["old", "grouped"]):
            env = dict(os.environ, GEN9BLAS_GEMV_N=variant, OPENBLAS_NUM_THREADS="1")
            for line in subprocess.run([sys.executable, __file__, "-", MEM], env=env, capture_output=True, text=True, check=True).stdout.split("\n"):
                if line:
                    M, N, g = line.split()
                    w.setdefault((int(M), int(N)), {}).setdefault(variant, []).append(float(g))
    print(f"memory: {MEM}\n{'M':>6s} {'N':>6s}  old GB/s  grouped GB/s")
    for (M, N), d in w.items():
        print(f"{M:6d} {N:6d}  {np.median(d['old']):8.1f}  {np.median(d['grouped']):12.1f}")
