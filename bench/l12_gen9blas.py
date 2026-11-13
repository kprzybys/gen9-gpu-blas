"""BLAS levels 1–2 in libgen9blas: bandwidth (GB/s) and process CPU time, against NumPy/SciPy (OpenBLAS) on the CPU.
GB/s computed from the data size of the formula (triangle: half the matrix; SYMV: the whole matrix, although the triangle is read twice).

Run: python3 bench/l12_gen9blas.py [float32|float16]
GPU "usm": gen9blas.empty arrays (zero-copy); "copy": plain NumPy arrays (the CPU copies into a work buffer).
CPU%: process CPU time / wall time (100% = one core). Best time of 5 repetitions, CPU -- mean.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "lib"))
import gen9blas  # noqa: E402

dtype = np.dtype(sys.argv[1] if len(sys.argv) > 1 else "float32").type
e = np.dtype(dtype).itemsize
rng = np.random.default_rng(1)


def make_array(shape, usm):
    X = gen9blas.empty(shape, dtype) if usm else np.empty(shape, dtype)
    X[:] = rng.uniform(-1, 1, shape)
    return X


def _blas(name):
    """SciPy BLAS function for the type (FP16 on the CPU: none -- computing in float32 on a copy makes no sense, we return None)."""
    import scipy.linalg.blas as b
    return getattr(b, "s" + name)


def measure(f, reps=5):
    f()
    sc, cp = [], []
    for _ in range(reps):
        t, c = time.perf_counter(), time.process_time()
        f()
        sc.append(time.perf_counter() - t); cp.append(time.process_time() - c)
    return min(sc), sum(cp) / len(cp) / np.mean(sc)


n1, n2 = 1 << 24, 8192
print(f"== {np.dtype(dtype).name}: vectors n = {n1}, matrices {n2} x {n2}")
print(f"  {'':10s} {'GPU usm':>18s} {'GPU copy':>18s} {'CPU (NumPy)':>18s}")
for name, nbytes, gpu, cpu in [
    ("axpy", 3 * n1 * e, lambda x, y, A: gen9blas.axpy(0.5, x, y), lambda x, y, A: y.__iadd__(0.5 * x)),
    ("dot", 2 * n1 * e, lambda x, y, A: gen9blas.dot(x, y), lambda x, y, A: float(x @ y)),
    ("nrm2", n1 * e, lambda x, y, A: gen9blas.nrm2(x), lambda x, y, A: float(np.linalg.norm(x))),
    ("iamax", n1 * e, lambda x, y, A: gen9blas.iamax(x), lambda x, y, A: int(np.argmax(np.abs(x)))),
    ("gemv N", n2 * n2 * e, lambda x, y, A: gen9blas.gemv(A, x[:n2], y=y[:n2]), lambda x, y, A: A @ x[:n2]),
    ("gemv T", n2 * n2 * e, lambda x, y, A: gen9blas.gemv(A, x[:n2], y=y[:n2], trans=True), lambda x, y, A: A.T @ x[:n2]),
    ("ger", 2 * n2 * n2 * e, lambda x, y, A: gen9blas.ger(0.5, x[:n2], y[:n2], A), lambda x, y, A: A.__iadd__(0.5 * np.outer(x[:n2], y[:n2]))),
    # stage B: matrix counted once (half the matrix for a triangle, the whole for SYMV); CPU: SciPy (BLAS, A.T without copy)
    ("symv", n2 * n2 * e, lambda x, y, A: gen9blas.symv(A, x[:n2], y=y[:n2]), lambda x, y, A: _blas("symv")(1.0, A.T, x[:n2])),
    ("trmv N", n2 * n2 * e / 2, lambda x, y, A: gen9blas.trmv(A, x[:n2]), lambda x, y, A: _blas("trmv")(A.T, x[:n2], lower=0)),
    ("trmv T", n2 * n2 * e / 2, lambda x, y, A: gen9blas.trmv(A, x[:n2], trans=True), lambda x, y, A: _blas("trmv")(A.T, x[:n2], lower=0, trans=1)),
    ("trsv N", n2 * n2 * e / 2, lambda x, y, A: gen9blas.trsv(A, x[:n2]), lambda x, y, A: _blas("trsv")(A.T, x[:n2], lower=0)),
    ("trsv T", n2 * n2 * e / 2, lambda x, y, A: gen9blas.trsv(A, x[:n2], trans=True), lambda x, y, A: _blas("trsv")(A.T, x[:n2], lower=0, trans=1)),
    ("syr", n2 * n2 * e, lambda x, y, A: gen9blas.syr(0.5, x[:n2], A), lambda x, y, A: _blas("syr")(0.5, x[:n2], a=A.T, overwrite_a=1)),
    ("syr2", n2 * n2 * e, lambda x, y, A: gen9blas.syr2(0.5, x[:n2], y[:n2], A), lambda x, y, A: _blas("syr2")(0.5, x[:n2], y[:n2], a=A.T, overwrite_a=1)),
]:
    results = []
    for usm in (True, False):
        x, y = make_array((n1,), usm), make_array((n1,), usm)
        A = make_array((n2, n2), usm) if name not in ("axpy", "dot", "nrm2", "iamax") else None
        if A is not None and name.startswith("trsv"):
            A[np.diag_indices(n2)] = n2   # well conditioned
        t, c = measure(lambda: gpu(x, y, A))
        results.append(f"{nbytes / t / 1e9:5.1f} GB/s {100 * c:4.0f}%")
    x, y = make_array((n1,), False), make_array((n1,), False)
    A = make_array((n2, n2), False) if name not in ("axpy", "dot", "nrm2", "iamax") else None
    if A is not None and name.startswith("trsv"):
        A[np.diag_indices(n2)] = n2
    if dtype == np.float16 and name in ("symv", "syr", "syr2") or dtype == np.float16 and name[:4] in ("trmv", "trsv"):
        results.append("-")   # no FP16 BLAS on the CPU
    else:
        try:
            t, c = measure(lambda: cpu(x, y, A))
            results.append(f"{nbytes / t / 1e9:5.1f} GB/s {100 * c:4.0f}%")
        except ImportError:   # no SciPy (CPU BLAS column), e.g. a machine with only the GPU library
            results.append("-")
    print(f"  {name:10s} " + " ".join(f"{w:>18s}" for w in results))
