"""Test userptr (bez NEO): zwykłe tablice NumPy bez kopii przez CPU, z pamięcią podręczną obiektów w kcl.

Uruchom: python3 lib/test_userptr.py   (także z KCL_USERPTR_MB=8: usuwanie z pamięci podręcznej i kopia zastępcza)
Sprawdza przypadki, w których pamięć podręczna mogłaby podać GPU nieaktualne albo cudze strony:
  1. CPU zmienia dane między wywołaniami (te same tablice) -> wynik nowy;
  2. tablice zwalniane i tworzone na nowo w różnych rozmiarach, przeplatane z gen9blas.empty (mmap obiektów GEM zajmuje
     zwolnione adresy) -> każdy wynik poprawny, bez błędu execbuffer (EFAULT);
  3. plik zmapowany tylko do odczytu jako wejście (userptr tylko do odczytu);
  4. kilka widoków tej samej tablicy jako różne operandy; małe tablice na wspólnej stronie; zapis do dwóch tablic,
     których końce leżą w jednej linii pamięci (swap/rot: dwa obiekty na tych samych stronach gubiły zapis);
  5. wynik w miejscu (axpy, scal, trmm) na zwykłych tablicach.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import gen9blas  # noqa: E402

rng = np.random.default_rng(3)
bledy = 0


def spr(nazwa, W, ref, tol):
    global bledy
    e = np.abs(np.asarray(W, np.float64) - ref).max() / max(np.abs(ref).max(), 1e-30)
    if not e <= tol:
        bledy += 1
        print(f"  BLAD {nazwa}: blad wzgl. {e:.1e}")


def gemm_ref(A, B):
    return A.astype(np.float64) @ B.astype(np.float64)


print("== 1. zmiany przez CPU między wywołaniami")
A, B = rng.uniform(-1, 1, (300, 200)).astype(np.float32), rng.uniform(-1, 1, (200, 100)).astype(np.float32)
C = np.zeros((300, 100), np.float32)
for r in range(6):
    A[r::7] += 1.0; B[:, r] *= -2
    gen9blas.sgemm(A, B, C=C)
    spr(f"sgemm runda {r}", C, gemm_ref(A, B), 1e-5)
x, y = rng.uniform(-1, 1, 10001).astype(np.float32), np.zeros(10001, np.float32)
for r in range(5):
    x[r] = 100 + r
    y0 = y.astype(np.float64).copy(); gen9blas.axpy(0.5, x, y)
    spr(f"axpy runda {r}", y, y0 + 0.5 * x, 1e-6)
    spr(f"dot runda {r}", gen9blas.dot(x, y), x.astype(np.float64) @ y.astype(np.float64), 1e-5)

print("== 2. zwalnianie i ponowne przydzielanie (także gen9blas.empty na zwolnionych adresach)")
trzymane = []
for r in range(40):
    n = int(rng.choice([16, 100, 513, 1024, 1500, 2048]))
    k = int(rng.choice([8, 64, 300, 1024]))
    A = rng.uniform(-1, 1, (n, k)).astype(np.float32); B = rng.uniform(-1, 1, (k, 64)).astype(np.float32)
    if r % 3 == 0:
        E = gen9blas.empty((int(rng.integers(1, 3000)), 1024), np.float32); E[0, 0] = 1; trzymane.append(E)
    if r % 5 == 0 and trzymane:
        trzymane.pop(int(rng.integers(0, len(trzymane))))
    spr(f"sgemm r{r} n={n} k={k}", gen9blas.sgemm(A, B), gemm_ref(A, B), 1e-5)
    v = rng.uniform(-1, 1, n * k).astype(np.float32)
    spr(f"nrm2 r{r}", gen9blas.nrm2(v), np.linalg.norm(v.astype(np.float64)), 1e-5)
    del A, B, v

print("== 3. plik zmapowany tylko do odczytu")
with tempfile.TemporaryDirectory() as kat:
    sciezka = os.path.join(kat, "a.bin")
    D = rng.uniform(-1, 1, (512, 256)).astype(np.float32); D.tofile(sciezka)
    M = np.memmap(sciezka, np.float32, "r", shape=(512, 256))
    B = rng.uniform(-1, 1, (256, 128)).astype(np.float32)
    for r in range(2):
        spr(f"sgemm z memmap r{r}", gen9blas.sgemm(M, B), gemm_ref(D, B), 1e-5)
    spr("dot z memmap", gen9blas.dot(M[0], M[1]), D[0].astype(np.float64) @ D[1].astype(np.float64), 1e-5)
    y = np.zeros(256, np.float32)
    gen9blas.gemv(M.T, M[:, 0].copy(), y=y)
    spr("gemv z memmap", y, D.T.astype(np.float64) @ D[:, 0].astype(np.float64), 1e-5)
    del M

print("== 4. widoki tej samej tablicy; małe tablice na wspólnej stronie")
X = rng.uniform(-1, 1, (600, 600)).astype(np.float32)
for r in range(3):
    A, B = X[:400, 5:305], X[100:400, 200:500]
    spr(f"sgemm widoki X r{r}", gen9blas.sgemm(A, B), gemm_ref(A, B), 1e-5)
    spr(f"sgemm X·X.T r{r}", gen9blas.sgemm(X[:300], X[:300].T), gemm_ref(X[:300], X[:300].T), 1e-5)
male = [rng.uniform(-1, 1, (8, 8)).astype(np.float32) for _ in range(30)]
for r in range(3):
    for i in range(0, 28, 3):
        spr(f"male {i} r{r}", gen9blas.sgemm(male[i], male[i + 1]), gemm_ref(male[i], male[i + 1]), 1e-5)

for typ in (np.float32, np.float64):
    for n in (5, 255, 1000, 30000, 100003):
        for r in range(4):
            bufory = [rng.uniform(-1, 1, n + 3).astype(typ) for _ in range(6)]   # sąsiednie na stercie
            for i in range(0, 6, 2):
                x, y = bufory[i][1:n + 1], bufory[i + 1][::-1][:n]
                X, Y = x.copy(), y.copy()
                gen9blas.dot(x, y); gen9blas.axpy(0.0, x, y)   # wcześniejsze obiekty: tylko do odczytu / do zapisu
                gen9blas.swap(x, y)
                if not (np.array_equal(x, Y) and np.array_equal(y, X)):
                    bledy += 1; print(f"  BLAD swap sasiednie {np.dtype(typ).name} n={n} r{r}: x {np.sum(x != Y)}, y {np.sum(y != X)}")
                gen9blas.rot(x, y, 0.6, 0.8)
                spr(f"rot sasiednie n={n}", np.concatenate([x, y]), np.concatenate([0.6 * Y + 0.8 * X, 0.6 * X - 0.8 * Y]), 1e-6)

print("== 5. wynik w miejscu na zwykłych tablicach")
for typ, tol in ((np.float32, 1e-5), (np.float64, 1e-12)):
    A = np.tril(rng.uniform(-1, 1, (300, 300))).astype(typ) + 300 * np.eye(300, dtype=typ)
    B = rng.uniform(-1, 1, (300, 70)).astype(typ)
    for r in range(2):
        spr(f"trmm {np.dtype(typ).name} r{r}", gen9blas.trmm(A, B), A.astype(np.float64) @ B, tol)
        X2 = gen9blas.trsm(A, B)
        spr(f"trsm {np.dtype(typ).name} r{r}", A.astype(np.float64) @ X2, B, tol * 10)
        B[r] += 1
    z = rng.uniform(-1, 1, 7777).astype(typ); z0 = z.astype(np.float64)
    gen9blas.scal(3.0, z); spr(f"scal {np.dtype(typ).name}", z, 3 * z0, tol)

print("\nWYNIK:", "wszystko OK" if not bledy else f"{bledy} błędów")
sys.exit(1 if bledy else 0)
