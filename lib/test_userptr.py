"""userptr test (without NEO): plain NumPy arrays without a CPU copy, with the kcl object cache.

Run: python3 lib/test_userptr.py   (also with KCL_USERPTR_MB=8: cache eviction and the fallback copy)
Checks cases in which the cache could hand the GPU stale or foreign pages:
  1. the CPU changes data between calls (the same arrays) -> new result;
  2. arrays freed and recreated in different sizes, interleaved with gen9blas.empty (mmap of GEM objects takes the
     freed addresses) -> every result correct, no execbuffer error (EFAULT);
  3. a file mapped read-only as input (read-only userptr);
  4. several views of the same array as different operands; small arrays on a shared page; writing to two arrays
     whose ends lie in one memory line (swap/rot: two objects on the same pages lost a write);
  5. in-place result (axpy, scal, trmm) on plain arrays.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import gen9blas  # noqa: E402

rng = np.random.default_rng(3)
errors = 0


def chk(name, W, ref, tol):
    global errors
    e = np.abs(np.asarray(W, np.float64) - ref).max() / max(np.abs(ref).max(), 1e-30)
    if not e <= tol:
        errors += 1
        print(f"  ERR {name}: rel. error {e:.1e}")


def gemm_ref(A, B):
    return A.astype(np.float64) @ B.astype(np.float64)


print("== 1. changes by the CPU between calls")
A, B = rng.uniform(-1, 1, (300, 200)).astype(np.float32), rng.uniform(-1, 1, (200, 100)).astype(np.float32)
C = np.zeros((300, 100), np.float32)
for r in range(6):
    A[r::7] += 1.0; B[:, r] *= -2
    gen9blas.sgemm(A, B, C=C)
    chk(f"sgemm round {r}", C, gemm_ref(A, B), 1e-5)
x, y = rng.uniform(-1, 1, 10001).astype(np.float32), np.zeros(10001, np.float32)
for r in range(5):
    x[r] = 100 + r
    y0 = y.astype(np.float64).copy(); gen9blas.axpy(0.5, x, y)
    chk(f"axpy round {r}", y, y0 + 0.5 * x, 1e-6)
    chk(f"dot round {r}", gen9blas.dot(x, y), x.astype(np.float64) @ y.astype(np.float64), 1e-5)

print("== 2. freeing and reallocating (also gen9blas.empty at freed addresses)")
kept = []
for r in range(40):
    n = int(rng.choice([16, 100, 513, 1024, 1500, 2048]))
    k = int(rng.choice([8, 64, 300, 1024]))
    A = rng.uniform(-1, 1, (n, k)).astype(np.float32); B = rng.uniform(-1, 1, (k, 64)).astype(np.float32)
    if r % 3 == 0:
        E = gen9blas.empty((int(rng.integers(1, 3000)), 1024), np.float32); E[0, 0] = 1; kept.append(E)
    if r % 5 == 0 and kept:
        kept.pop(int(rng.integers(0, len(kept))))
    chk(f"sgemm r{r} n={n} k={k}", gen9blas.sgemm(A, B), gemm_ref(A, B), 1e-5)
    v = rng.uniform(-1, 1, n * k).astype(np.float32)
    chk(f"nrm2 r{r}", gen9blas.nrm2(v), np.linalg.norm(v.astype(np.float64)), 1e-5)
    del A, B, v

print("== 3. file mapped read-only")
with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "a.bin")
    D = rng.uniform(-1, 1, (512, 256)).astype(np.float32); D.tofile(path)
    M = np.memmap(path, np.float32, "r", shape=(512, 256))
    B = rng.uniform(-1, 1, (256, 128)).astype(np.float32)
    for r in range(2):
        chk(f"sgemm z memmap r{r}", gen9blas.sgemm(M, B), gemm_ref(D, B), 1e-5)
    chk("dot z memmap", gen9blas.dot(M[0], M[1]), D[0].astype(np.float64) @ D[1].astype(np.float64), 1e-5)
    y = np.zeros(256, np.float32)
    gen9blas.gemv(M.T, M[:, 0].copy(), y=y)
    chk("gemv z memmap", y, D.T.astype(np.float64) @ D[:, 0].astype(np.float64), 1e-5)
    del M

print("== 4. views of the same array; small arrays on a shared page")
X = rng.uniform(-1, 1, (600, 600)).astype(np.float32)
for r in range(3):
    A, B = X[:400, 5:305], X[100:400, 200:500]
    chk(f"sgemm views X r{r}", gen9blas.sgemm(A, B), gemm_ref(A, B), 1e-5)
    chk(f"sgemm X·X.T r{r}", gen9blas.sgemm(X[:300], X[:300].T), gemm_ref(X[:300], X[:300].T), 1e-5)
small = [rng.uniform(-1, 1, (8, 8)).astype(np.float32) for _ in range(30)]
for r in range(3):
    for i in range(0, 28, 3):
        chk(f"small {i} r{r}", gen9blas.sgemm(small[i], small[i + 1]), gemm_ref(small[i], small[i + 1]), 1e-5)

for dt in (np.float32, np.float64):
    for n in (5, 255, 1000, 30000, 100003):
        for r in range(4):
            bufs = [rng.uniform(-1, 1, n + 3).astype(dt) for _ in range(6)]   # neighbours on the heap
            for i in range(0, 6, 2):
                x, y = bufs[i][1:n + 1], bufs[i + 1][::-1][:n]
                X, Y = x.copy(), y.copy()
                gen9blas.dot(x, y); gen9blas.axpy(0.0, x, y)   # earlier objects: read-only / write
                gen9blas.swap(x, y)
                if not (np.array_equal(x, Y) and np.array_equal(y, X)):
                    errors += 1; print(f"  ERR swap adjacent {np.dtype(dt).name} n={n} r{r}: x {np.sum(x != Y)}, y {np.sum(y != X)}")
                gen9blas.rot(x, y, 0.6, 0.8)
                chk(f"rot adjacent n={n}", np.concatenate([x, y]), np.concatenate([0.6 * Y + 0.8 * X, 0.6 * X - 0.8 * Y]), 1e-6)

print("== 5. in-place result on plain arrays")
for dt, tol in ((np.float32, 1e-5), (np.float64, 1e-12)):
    A = np.tril(rng.uniform(-1, 1, (300, 300))).astype(dt) + 300 * np.eye(300, dtype=dt)
    B = rng.uniform(-1, 1, (300, 70)).astype(dt)
    for r in range(2):
        chk(f"trmm {np.dtype(dt).name} r{r}", gen9blas.trmm(A, B), A.astype(np.float64) @ B, tol)
        X2 = gen9blas.trsm(A, B)
        chk(f"trsm {np.dtype(dt).name} r{r}", A.astype(np.float64) @ X2, B, tol * 10)
        B[r] += 1
    z = rng.uniform(-1, 1, 7777).astype(dt); z0 = z.astype(np.float64)
    gen9blas.scal(3.0, z); chk(f"scal {np.dtype(dt).name}", z, 3 * z0, tol)

print("\nRESULT:", "all OK" if not errors else f"{errors} errors")
sys.exit(1 if errors else 0)
