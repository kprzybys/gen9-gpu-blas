"""Bring-up steps on a Gen8 Cherryview GPU (docs/gen8.md), one per run, each checked against the CPU:
  init   kcl/gen9blas initialisation only (no GPU work): generation and available types
  copy   FP32 copy (kernel copy16_f: no DW x DW multiply, only width-1 64-bit regions)
  axpy   FP32 axpy (axpy_f: adds vector DW x DW multiplies in index code)
  sgemm  SGEMM n = 16, 64, 100, 256 and batched (2 matrices): packing pkf_a/pkf_b, sgemm_v, epilogue
  dcopy  FP64 copy, n from the command line (default 64): copy16_d, A64 qword gather/scatter + DF moves, no multiplies
  daxpy  FP64 axpy (axpy_d), n = 64, 4096, 2^20
  dtrsm  FP64 TRSM n = 64 (trsm_inv*_d: emulated FP64 division on Gen8)
  ddiv   FP64 division as the library does it (TRSV 1x1 -> trsv_block_d, emulated division on Gen8): special and
         extreme pairs and random pairs over the whole FP64 range, bit-compared with IEEE a / b
  dtrsm_scaled  DTRSM n = 64 with diagonals 1e50, 1e-50, 1e-300 (scaling of the emulated division)
  dgemm  small DGEMM (pkd_a hung on the first run, docs/gen8.md 6a -- run only on purpose)
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
elif step == "sgemm":
    for n in (16, 64, 100, 256):
        A = rng.standard_normal((n, n)).astype(np.float32)
        B = rng.standard_normal((n, n)).astype(np.float32)
        C = g.gemm(A, B)
        ref = A.astype(np.float64) @ B.astype(np.float64)
        err = float(np.max(np.abs(C - ref)) / np.max(np.abs(ref)))
        ok &= err < 1e-4
        print(f"sgemm n={n}: max rel err {err:.1e} {'OK' if err < 1e-4 else 'WRONG'}")
    A = rng.standard_normal((2, 64, 64)).astype(np.float32)
    B = rng.standard_normal((2, 64, 64)).astype(np.float32)
    C = g.gemm_batched(A, B)
    ref = np.matmul(A.astype(np.float64), B.astype(np.float64))
    err = float(np.max(np.abs(C - ref)) / np.max(np.abs(ref)))
    ok &= err < 1e-4
    print(f"sgemm batched 2 x 64: max rel err {err:.1e} {'OK' if err < 1e-4 else 'WRONG'}")
elif step == "dcopy":
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    x = rng.standard_normal(n)
    y = np.zeros(n)
    g.copy(x, y)
    good = np.array_equal(x, y)
    ok &= good
    print(f"dcopy n={n}: {'OK' if good else 'WRONG'}")
elif step == "daxpy":
    for n in (64, 4096, 1 << 20):
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        ref = 0.5 * x + y
        g.axpy(0.5, x, y)
        err = float(np.max(np.abs(y - ref)))
        ok &= err < 1e-14
        print(f"daxpy n={n}: max abs err {err:.2e} {'OK' if err < 1e-14 else 'WRONG'}")
elif step == "dtrsm":
    n = 64
    A = np.tril(rng.standard_normal((n, n))) + n * np.eye(n)
    B = rng.standard_normal((n, n))
    X = g.trsm(A, B.copy())
    err = float(np.max(np.abs(A @ X - B)) / np.max(np.abs(B)))
    ok &= err < 1e-12
    print(f"dtrsm n={n}: max rel residual {err:.1e} {'OK' if err < 1e-12 else 'WRONG'}")
elif step == "ddiv":
    import math
    def same(x, y):
        return (math.isnan(x) and math.isnan(y)) or (x == y and math.copysign(1, x) == math.copysign(1, y))
    special = [0.0, -0.0, 1.0, -3.0, math.inf, -math.inf, math.nan, 1e300, 1e-300, 5e-324, 2.2e-308, 1e-310,
               1.7976931348623157e308, 1e50, 1e-50, 7.0, -1e-320]
    pairs = [(a, b) for a in special for b in special]
    pairs += [(float(rng.standard_normal() * 10.0 ** rng.integers(-320, 308)),
               float(rng.standard_normal() * 10.0 ** rng.integers(-320, 308))) for _ in range(2000)]
    exact = ulp1 = 0
    for a, b in pairs:
        x = np.array([a]); g.trsv(np.array([[b]]), x)
        with np.errstate(all="ignore"):
            want = float(np.float64(a) / np.float64(b))
        got = float(x[0])
        if same(got, want):
            exact += 1
        elif math.isfinite(want) and want != 0 and abs(got - want) <= math.ulp(want) and abs(want) < 2.3e-308:
            ulp1 += 1   # subnormal result: 1 ulp allowed (double rounding, docs/gen8.md)
        else:
            ok = False
            print(f"  WRONG {a!r} / {b!r}: {got!r}, IEEE {want!r}")
    print(f"ddiv: {exact} of {len(pairs)} bit-exact, {ulp1} subnormal results within 1 ulp")
elif step == "dtrsm_scaled":
    n = 64
    for scale in (1e50, 1e-50, 1e-300):
        A = np.tril(rng.standard_normal((n, n))) * scale + n * scale * np.eye(n)
        B = rng.standard_normal((n, n))
        X = g.trsm(A, B.copy())
        err = float(np.max(np.abs(A @ X - B)) / np.max(np.abs(B)))
        ok &= err < 1e-12
        print(f"dtrsm diagonal ~{scale:g}: max rel residual {err:.1e} {'OK' if err < 1e-12 else 'WRONG'}")
elif step == "dgemm":
    for dt, tol in ((np.float64, 1e-12),):
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
