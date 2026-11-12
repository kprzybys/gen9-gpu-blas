"""Krok 3 (2026-11-12): przepustowość GEMV N (row-major y = A·x) w libgen9blas, wariant grupowy wobec dawnego (GEN9BLAS_GEMV_N=old).
Uruchom: python3 bench/gemv_n.py [usm|zwykla]   (oba warianty na przemian w osobnych procesach, 3 rundy, mediana GB/s).
usm: tablice gen9blas.empty; zwykla: tablice NumPy (userptr, strony 4 KB -- tu dawny wariant tracił 2×, BEZNEO/NOTATKI)."""
import os
import subprocess
import sys
import time

import numpy as np

ROZM = [(1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192), (256, 65536), (65536, 256), (16384, 512),
        (8192, 1024), (4096, 2048), (51866, 1280), (1280, 5120)]
PAM = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("usm", "zwykla") else "usm"


def dziecko(pam):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "lib"))
    import gen9blas
    for M, N in ROZM:
        nowa = gen9blas.empty if pam == "usm" else np.empty
        A = nowa((M, N), np.float32); A[:] = 1e-3
        x = nowa(N, np.float32); x[:] = 1; y = nowa(M, np.float32)
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
        dziecko(sys.argv[2]); sys.exit()
    w = {}
    for r in range(3):
        for wariant in (["grouped", "old"] if r % 2 == 0 else ["old", "grouped"]):
            env = dict(os.environ, GEN9BLAS_GEMV_N=wariant, OPENBLAS_NUM_THREADS="1")
            for linia in subprocess.run([sys.executable, __file__, "-", PAM], env=env, capture_output=True, text=True, check=True).stdout.split("\n"):
                if linia:
                    M, N, g = linia.split()
                    w.setdefault((int(M), int(N)), {}).setdefault(wariant, []).append(float(g))
    print(f"pamięć: {PAM}\n{'M':>6s} {'N':>6s}  stary GB/s  grupowy GB/s")
    for (M, N), d in w.items():
        print(f"{M:6d} {N:6d}  {np.median(d['old']):10.1f}  {np.median(d['grouped']):12.1f}")
