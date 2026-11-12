"""Bring-up steps on a Gen8 Cherryview GPU (docs/gen8.md), one per run, each checked against the CPU:
  init   kcl/gen9blas initialisation only (no GPU work): generation and available types
  copy   FP32 copy (kernel copy16_f: no DW x DW multiply, only width-1 64-bit regions)
  axpy   FP32 axpy (axpy_f: adds vector DW x DW multiplies in index code)
  gemm   small SGEMM and DGEMM (packing and epilogue kernels: DW x DW multiplies; dgemm_v: FP64)
Run one step: KCL_PROFILE_KERNELS=1 python3 test_gen8_smoke.py <step>   (prints which kernels ran)
"""
import sys

import numpy as np

import gen9blas as g

step = sys.argv[1] if len(sys.argv) > 1 else "init"
rng = np.random.default_rng(1)
g._init()
print("kcl_gen8 =", g._lib.kcl_gen8(), "| types available (H, S, HS, D):", [g._lib.gen9blas_available(t) for t in range(4)])
ok = True
if step == "copy":
    for n in (64, 4096, 1 << 20):
        x = rng.standard_normal(n).astype(np.float32)
        y = np.zeros(n, np.float32)
        g.copy(x, y)
        good = np.array_equal(x, y)
        ok &= good
        print(f"copy n={n}: {'OK' if good else 'WRONG'}")
elif step == "axpy":
    for n in (64, 4096, 1 << 20):
        x = rng.standard_normal(n).astype(np.float32)
        y = rng.standard_normal(n).astype(np.float32)
        ref = 0.5 * x + y
        g.axpy(0.5, x, y)
        err = float(np.max(np.abs(y - ref)))
        ok &= err < 1e-5
        print(f"axpy n={n}: max abs err {err:.2e} {'OK' if err < 1e-5 else 'WRONG'}")
elif step == "gemm":
    for dt, tol in ((np.float32, 1e-4), (np.float64, 1e-12)):
        for n in (16, 64, 256):
            A = rng.standard_normal((n, n)).astype(dt)
            B = rng.standard_normal((n, n)).astype(dt)
            C = g.gemm(A, B)
            ref = A.astype(np.float64) @ B.astype(np.float64)
            err = float(np.max(np.abs(C - ref)) / np.max(np.abs(ref)))
            ok &= err < tol
            print(f"gemm {np.dtype(dt).name} n={n}: max rel err {err:.1e} {'OK' if err < tol else 'WRONG'}")
elif step != "init":
    sys.exit(f"unknown step {step}")
print("RESULT:", "OK" if ok else "WRONG")
sys.exit(0 if ok else 1)
