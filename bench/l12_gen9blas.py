"""BLAS poziomów 1–2 w libgen9blas: przepustowość (GB/s) i czas CPU procesu, wobec NumPy/SciPy (OpenBLAS) na CPU.
GB/s liczone z rozmiaru danych wzoru (trójkąt: pół macierzy; SYMV: cała macierz, choć czytany jest trójkąt dwa razy).

Uruchom: python3 bench/l12_gen9blas.py [float32|float16]
GPU „usm”: tablice gen9blas.empty (bez kopii); „kopia”: zwykłe tablice NumPy (CPU kopiuje do bufora roboczego).
CPU%: czas CPU procesu / czas ściany (100% = jeden rdzeń). Najlepszy czas z 5 powtórzeń, CPU -- średnia.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "lib"))
import gen9blas  # noqa: E402

typ = np.dtype(sys.argv[1] if len(sys.argv) > 1 else "float32").type
e = np.dtype(typ).itemsize
rng = np.random.default_rng(1)


def tab(shape, usm):
    X = gen9blas.empty(shape, typ) if usm else np.empty(shape, typ)
    X[:] = rng.uniform(-1, 1, shape)
    return X


def _b(nazwa):
    """Funkcja BLAS z SciPy dla typu (FP16 na CPU: brak -- liczone w float32 na kopii nie ma sensu, zwracamy None)."""
    import scipy.linalg.blas as b
    return getattr(b, "s" + nazwa)


def mierz(f, powt=5):
    f()
    sc, cp = [], []
    for _ in range(powt):
        t, c = time.perf_counter(), time.process_time()
        f()
        sc.append(time.perf_counter() - t); cp.append(time.process_time() - c)
    return min(sc), sum(cp) / len(cp) / np.mean(sc)


n1, n2 = 1 << 24, 8192
print(f"== {np.dtype(typ).name}: wektory n = {n1}, macierze {n2} x {n2}")
print(f"  {'':10s} {'GPU usm':>18s} {'GPU kopia':>18s} {'CPU (NumPy)':>18s}")
for nazwa, bajty, gpu, cpu in [
    ("axpy", 3 * n1 * e, lambda x, y, A: gen9blas.axpy(0.5, x, y), lambda x, y, A: y.__iadd__(0.5 * x)),
    ("dot", 2 * n1 * e, lambda x, y, A: gen9blas.dot(x, y), lambda x, y, A: float(x @ y)),
    ("nrm2", n1 * e, lambda x, y, A: gen9blas.nrm2(x), lambda x, y, A: float(np.linalg.norm(x))),
    ("iamax", n1 * e, lambda x, y, A: gen9blas.iamax(x), lambda x, y, A: int(np.argmax(np.abs(x)))),
    ("gemv N", n2 * n2 * e, lambda x, y, A: gen9blas.gemv(A, x[:n2], y=y[:n2]), lambda x, y, A: A @ x[:n2]),
    ("gemv T", n2 * n2 * e, lambda x, y, A: gen9blas.gemv(A, x[:n2], y=y[:n2], trans=True), lambda x, y, A: A.T @ x[:n2]),
    ("ger", 2 * n2 * n2 * e, lambda x, y, A: gen9blas.ger(0.5, x[:n2], y[:n2], A), lambda x, y, A: A.__iadd__(0.5 * np.outer(x[:n2], y[:n2]))),
    # etap B: macierz liczona raz (pół macierzy dla trójkąta, cała dla SYMV); CPU: SciPy (BLAS, A.T bez kopii)
    ("symv", n2 * n2 * e, lambda x, y, A: gen9blas.symv(A, x[:n2], y=y[:n2]), lambda x, y, A: _b("symv")(1.0, A.T, x[:n2])),
    ("trmv N", n2 * n2 * e / 2, lambda x, y, A: gen9blas.trmv(A, x[:n2]), lambda x, y, A: _b("trmv")(A.T, x[:n2], lower=0)),
    ("trmv T", n2 * n2 * e / 2, lambda x, y, A: gen9blas.trmv(A, x[:n2], trans=True), lambda x, y, A: _b("trmv")(A.T, x[:n2], lower=0, trans=1)),
    ("trsv N", n2 * n2 * e / 2, lambda x, y, A: gen9blas.trsv(A, x[:n2]), lambda x, y, A: _b("trsv")(A.T, x[:n2], lower=0)),
    ("trsv T", n2 * n2 * e / 2, lambda x, y, A: gen9blas.trsv(A, x[:n2], trans=True), lambda x, y, A: _b("trsv")(A.T, x[:n2], lower=0, trans=1)),
    ("syr", n2 * n2 * e, lambda x, y, A: gen9blas.syr(0.5, x[:n2], A), lambda x, y, A: _b("syr")(0.5, x[:n2], a=A.T, overwrite_a=1)),
    ("syr2", n2 * n2 * e, lambda x, y, A: gen9blas.syr2(0.5, x[:n2], y[:n2], A), lambda x, y, A: _b("syr2")(0.5, x[:n2], y[:n2], a=A.T, overwrite_a=1)),
]:
    wyniki = []
    for usm in (True, False):
        x, y = tab((n1,), usm), tab((n1,), usm)
        A = tab((n2, n2), usm) if nazwa not in ("axpy", "dot", "nrm2", "iamax") else None
        if A is not None and nazwa.startswith("trsv"):
            A[np.diag_indices(n2)] = n2   # dobrze uwarunkowana
        t, c = mierz(lambda: gpu(x, y, A))
        wyniki.append(f"{bajty / t / 1e9:5.1f} GB/s {100 * c:4.0f}%")
    x, y = tab((n1,), False), tab((n1,), False)
    A = tab((n2, n2), False) if nazwa not in ("axpy", "dot", "nrm2", "iamax") else None
    if A is not None and nazwa.startswith("trsv"):
        A[np.diag_indices(n2)] = n2
    if typ == np.float16 and nazwa in ("symv", "syr", "syr2") or typ == np.float16 and nazwa[:4] in ("trmv", "trsv"):
        wyniki.append("-")   # brak BLAS FP16 na CPU
    else:
        t, c = mierz(lambda: cpu(x, y, A))
        wyniki.append(f"{bajty / t / 1e9:5.1f} GB/s {100 * c:4.0f}%")
    print(f"  {nazwa:10s} " + " ".join(f"{w:>18s}" for w in wyniki))
