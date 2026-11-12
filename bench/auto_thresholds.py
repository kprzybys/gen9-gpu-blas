"""Progi trybu auto w libgen9blas_blas (poziomy 1–2 na zwykłej pamięci po userptr): czas CPU procesu i czas ściany
na wywołanie, GPU wobec OpenBLAS (1 i 8 wątków). Wyniki i przewidywanie: results/auto-thresholds-2026-11-12.txt.

Uruchom: python3 bench/auto_thresholds.py [rundy] [typ: s|d]   (rodzic uruchamia tryby na przemian w osobnych procesach)
Tryby: gpu (GEN9BLAS_BLAS=gpu, OPENBLAS_NUM_THREADS=1), cpu1, cpu8 (GEN9BLAS_BLAS=cpu, OPENBLAS_NUM_THREADS=1 / 8).
Przypadki: „te”: te same tablice w kolejnych wywołaniach; „nowe”: nowe tablice NumPy (np.empty + wypełnienie)
przed każdym wywołaniem. CPU wywołania = (CPU procesu w pętli − CPU wątku głównego na przygotowanie danych) / R --
wliczone jest wirowanie wątków OpenBLAS po wywołaniu i przypięcie stron userptr.
"""
import ctypes as C
import json
import os
import subprocess
import sys
import time

import numpy as np

KAT = os.path.dirname(os.path.realpath(__file__))
WIERSZ, BEZ, TR, GORNY, ZWYKLA = 101, 111, 112, 121, 131
L1 = [1 << k for k in range(10, 25, 2)]
L2 = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
NOWE_MAX = 1 << 22   # „nowe” tylko do 16 MB na tablicę (dłużej i tak przeważa wypełnianie)


def dziecko(tryb, t):
    # GEN9BLAS_BLAS i OPENBLAS_NUM_THREADS ustawia rodzic w środowisku procesu: NumPy przy imporcie ładuje systemowy
    # OpenBLAS, który od razu czyta liczbę wątków (zmiana os.environ tutaj byłaby za późno)
    lib = C.CDLL(os.path.join(KAT, "..", "lib", "libgen9blas_blas.so"))
    typ, ct = (np.float32, C.c_float) if t == "s" else (np.float64, C.c_double)
    f = lambda nazwa: getattr(lib, nazwa.replace("?", t))  # noqa: E731
    for nazwa, rt in [("cblas_?dot", ct), ("cblas_?nrm2", ct), ("cblas_?asum", ct), ("cblas_i?amax", C.c_size_t)]:
        f(nazwa).restype = rt
    p = lambda a: C.c_void_p(a.ctypes.data)  # noqa: E731
    rng = np.random.default_rng(1)

    zrodlo = (rng.uniform(-1, 1, 1 << 20) * 1e-3).astype(typ)

    def nowa(*shape):   # wypełnienie kopiowaniem (tanie i stałe; losowanie zaszumiało pomiar „nowe”)
        a = np.empty(shape, typ)
        fl, S = a.reshape(-1), zrodlo.size
        for i in range(0, fl.size, S):
            fl[i:i + S] = zrodlo[:min(S, fl.size - i)]
        return a

    def tri(n):
        A = nowa(n, n)
        A[np.diag_indices(n)] = 2.0   # dobrze uwarunkowana trójkątna (trsv)
        return A

    a = ct(0.5)
    przyp = {
        "axpy": (L1, lambda n: (nowa(n), nowa(n)), lambda n, x, y: f("cblas_?axpy")(n, a, p(x), 1, p(y), 1)),
        "dot": (L1, lambda n: (nowa(n), nowa(n)), lambda n, x, y: f("cblas_?dot")(n, p(x), 1, p(y), 1)),
        "nrm2": (L1, lambda n: (nowa(n),), lambda n, x: f("cblas_?nrm2")(n, p(x), 1)),
        "scal": (L1, lambda n: (nowa(n),), lambda n, x: f("cblas_?scal")(n, ct(0.999), p(x), 1)),
        "copy": (L1, lambda n: (nowa(n), nowa(n)), lambda n, x, y: f("cblas_?copy")(n, p(x), 1, p(y), 1)),
        "iamax": (L1, lambda n: (nowa(n),), lambda n, x: f("cblas_i?amax")(n, p(x), 1)),
        "gemv N": (L2, lambda n: (nowa(n, n), nowa(n), nowa(n)),
                   lambda n, A, x, y: f("cblas_?gemv")(WIERSZ, BEZ, n, n, a, p(A), n, p(x), 1, ct(0.0), p(y), 1)),
        "gemv T": (L2, lambda n: (nowa(n, n), nowa(n), nowa(n)),
                   lambda n, A, x, y: f("cblas_?gemv")(WIERSZ, TR, n, n, a, p(A), n, p(x), 1, ct(0.0), p(y), 1)),
        "ger": (L2, lambda n: (nowa(n, n), nowa(n), nowa(n)),
                lambda n, A, x, y: f("cblas_?ger")(WIERSZ, n, n, ct(1e-3), p(x), 1, p(y), 1, p(A), n)),
        "symv": (L2, lambda n: (nowa(n, n), nowa(n), nowa(n)),
                 lambda n, A, x, y: f("cblas_?symv")(WIERSZ, GORNY, n, a, p(A), n, p(x), 1, ct(0.0), p(y), 1)),
        "trmv": (L2, lambda n: (tri(n), nowa(n)), lambda n, A, x: f("cblas_?trmv")(WIERSZ, GORNY, BEZ, ZWYKLA, n, p(A), n, p(x), 1)),
        "trsv": (L2, lambda n: (tri(n), nowa(n)), lambda n, A, x: f("cblas_?trsv")(WIERSZ, GORNY, BEZ, ZWYKLA, n, p(A), n, p(x), 1)),
        "syr": (L2, lambda n: (nowa(n, n), nowa(n)), lambda n, A, x: f("cblas_?syr")(WIERSZ, GORNY, n, ct(1e-3), p(x), 1, p(A), n)),
    }
    for nazwa, (rozm, prz, wyw) in przyp.items():
        for n in rozm:
            el = n if rozm is L1 else n * n
            dane = prz(n)
            for _ in range(2):
                wyw(n, *dane)
            t0 = time.perf_counter(); wyw(n, *dane); t1 = time.perf_counter() - t0
            R = int(min(2000, max(5, 0.08 / max(t1, 1e-6))))
            time.sleep(0.2)   # wątki OpenBLAS po poprzednim przypadku przestają wirować (inaczej ich CPU trafia tutaj)
            # te same tablice: pętla z wywołaniem (pętla bez wywołania nic nie kosztuje)
            c0, s0 = time.process_time(), time.perf_counter()
            for _ in range(R):
                wyw(n, *dane)
            wynik = {"f": nazwa, "n": n, "p": "te", "t": (time.perf_counter() - s0) / R * 1e6,
                     "c": (time.process_time() - c0) / R * 1e6}
            print(json.dumps(wynik), flush=True)
            if el > NOWE_MAX:
                continue
            del dane
            Rn = max(5, min(R, 200))
            # nowe tablice: CPU procesu w pętli minus CPU wątku głównego na przygotowanie (i zwolnienie) danych --
            # wątki OpenBLAS (także wirujące po wywołaniu) liczą się w całości; czas ściany -- samo wywołanie
            c0, ts, baza = time.process_time(), 0.0, 0.0
            for _ in range(Rn):
                b0 = time.thread_time(); d = prz(n); baza += time.thread_time() - b0
                s0 = time.perf_counter(); wyw(n, *d); ts += time.perf_counter() - s0
                b0 = time.thread_time(); del d; baza += time.thread_time() - b0
            wynik = {"f": nazwa, "n": n, "p": "nowe", "t": ts / Rn * 1e6, "c": (time.process_time() - c0 - baza) / Rn * 1e6}
            print(json.dumps(wynik), flush=True)


def rodzic(rundy, t):
    tryby = ["gpu", "cpu1", "cpu8"]
    w = {}
    for r in range(rundy):
        for tryb in (tryby if r % 2 == 0 else tryby[::-1]):
            print(f"# runda {r + 1}/{rundy}: {tryb}", file=sys.stderr, flush=True)
            env = dict(os.environ, GEN9BLAS_BLAS="gpu" if tryb == "gpu" else "cpu",
                       OPENBLAS_NUM_THREADS="8" if tryb == "cpu8" else "1")
            wy = subprocess.run([sys.executable, __file__, "--dziecko", tryb, t], capture_output=True, text=True,
                                check=True, env=env).stdout
            for linia in wy.splitlines():
                d = json.loads(linia)
                w.setdefault((d["f"], d["n"], d["p"]), {}).setdefault(tryb, []).append((d["t"], d["c"]))
    print(f"typ {t}; czas ściany / CPU procesu na wywołanie [µs], mediany {rundy} rund; "
          "GPU/cpu8 i GPU/cpu1: iloraz CPU (< 1: GPU odciąża)")
    print(f"{'funkcja':8s} {'n':>9s} {'przyp':5s} | {'gpu':>17s} | {'cpu1':>17s} | {'cpu8':>17s} | {'CPU g/8':>7s} {'g/1':>6s} {'ściana g/8':>10s}")
    for (fn, n, pr), m in w.items():
        med = {k: np.median(np.array(v), axis=0) for k, v in m.items()}
        g, c1, c8 = med["gpu"], med["cpu1"], med["cpu8"]
        k = lambda x: f"{x[0]:8.1f} /{x[1]:8.1f}"  # noqa: E731
        print(f"{fn:8s} {n:9d} {pr:5s} | {k(g)} | {k(c1)} | {k(c8)} | {g[1] / max(c8[1], 1e-9):7.2f} "
              f"{g[1] / max(c1[1], 1e-9):6.2f} {g[0] / max(c8[0], 1e-9):10.2f}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--dziecko":
        dziecko(sys.argv[2], sys.argv[3])
    else:
        rodzic(int(sys.argv[1]) if len(sys.argv) > 1 else 3, sys.argv[2] if len(sys.argv) > 2 else "s")
