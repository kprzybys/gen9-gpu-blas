"""Fixed cost of one gen9blas call from Python: wall time and process CPU time per call, split into driver and wrapper.

Run: python3 bench/call_overhead.py [n] [calls] [rounds]     (default n = 4096 elements, 2000 calls, 5 rounds)
  wrapper+driver : gen9blas.axpy(a, x, y) on gen9blas.empty arrays (the whole Python call)
  driver (ctypes): the same C function called through ctypes directly with prepared pointers (no argument checks of the wrapper)
  wrapper part   : difference
Median over rounds of the mean per call. The C-only cost (no Python at all) is in results/overhead-2026-11-12.txt.
"""
import ctypes as C
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "lib"))
import gen9blas  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
calls = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 5
x = gen9blas.empty((n,), np.float32)
y = gen9blas.empty((n,), np.float32)
x[:] = 1
y[:] = 1
gen9blas.axpy(0.5, x, y)
lib = gen9blas._lib
fn = lib.gen9blas_axpy
kind = gen9blas._kind_basic(np.float32)
px, py = x.ctypes.data, y.ctypes.data


def full():
    gen9blas.axpy(0.5, x, y)


def raw():
    fn(kind, n, 0.5, px, 1, py, 1)


def measure(f):
    for _ in range(50):
        f()
    t, c = time.perf_counter(), time.process_time()
    for _ in range(calls):
        f()
    return (time.perf_counter() - t) / calls * 1e6, (time.process_time() - c) / calls * 1e6


res = {"full": [], "raw": []}
for _ in range(rounds):
    for k, f in (("full", full), ("raw", raw)):
        res[k].append(measure(f))
med = {k: (statistics.median(w for w, _ in v), statistics.median(c for _, c in v)) for k, v in res.items()}
for k, label in (("full", "gen9blas.axpy (Python wrapper + ctypes + library)"), ("raw", "ctypes call of gen9blas_axpy, prepared pointers")):
    w, c = med[k]
    allw = [a for a, _ in res[k]]
    print(f"  {label:52s} wall {w:6.1f} us [{min(allw):.1f}-{max(allw):.1f}]   CPU {c:6.1f} us")
print(f"  {'Python wrapper part (difference)':52s} wall {med['full'][0] - med['raw'][0]:6.1f} us            CPU {med['full'][1] - med['raw'][1]:6.1f} us")
print(f"n = {n} float32, {calls} calls per round, {rounds} rounds (alternating), median")
