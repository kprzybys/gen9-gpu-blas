"""Nakładka Pythona (ctypes) na libgen9blas: BLAS (poziomy 1–3) na Intel UHD 620 z pętlami GEMM w asemblerze.

    import numpy as np, gen9blas
    C = gen9blas.hgemm(A, B)                        # float16, suma FP16 (najszybszy); accurate=True: suma porcjami do FP32
    C = gen9blas.sgemm(A.T, B, alpha=2.0)           # float32; widoki transponowane bez kopii
    C = gen9blas.gemm_batched(A3, B3)               # paczka: A3 (count, M, K), B3 (count, K, N) -> (count, M, N)
    C = gen9blas.symm(A, B, side="L", uplo="L")     # A symetryczna
    C = gen9blas.syrk(A, uplo="L")                  # trójkąt uplo z A·Aᵀ
    C = gen9blas.syr2k(A, B, uplo="L")              # trójkąt uplo z A·Bᵀ + B·Aᵀ
    B = gen9blas.trmm(A, B, side="L", uplo="L")     # op(A)·B, A trójkątna (w miejscu, na kopii B)
    X = gen9blas.trsm(A, B, side="L", uplo="L")     # rozwiązanie op(A)·X = B (float32 / float16)
    X = gen9blas.empty((n, n), np.float16)          # pamięć USM; zwykłe tablice też bez kopii (userptr, BEZNEO.md 6g)
    C = gen9blas.sgemm(A[:, :k], B[:k].T)           # wycinki i widoki .T poziomów 1–3 bez kopii (ld z kroków NumPy)
    gen9blas.axpy(0.5, x, y); d = gen9blas.dot(x, y)     # poziom 1 (także scal, copy, swap, rot, asum, nrm2, iamax)
    y = gen9blas.gemv(A, x); gen9blas.ger(1.0, x, y, A)  # poziom 2

Typ (float16 / float32) wynika z typu danych. Błąd względny (do sumy |a·b|): hgemm ~1e-4, dokładny ~1e-5, sgemm ~1e-8.
"""
import ctypes
import os
import weakref

import numpy as np

_TU = os.path.dirname(os.path.realpath(__file__))
_lib = ctypes.CDLL(os.environ.get("GEN9BLAS_LIB", os.path.join(_TU, "libgen9blas.so")))   # GEN9BLAS_LIB: inna wersja biblioteki
_i, _f, _p, _l, _d = ctypes.c_int, ctypes.c_float, ctypes.c_void_p, ctypes.c_long, ctypes.c_double
_lib.gen9blas_init.argtypes = [ctypes.c_char_p]
_lib.gen9blas_gemm.argtypes = [_i, _i, _i, _i, _i, _i, _d, _p, _i, _p, _i, _d, _p, _i]
_lib.gen9blas_gemm_batched.argtypes = [_i, _i, _i, _i, _i, _i, _d, _p, _i, _l, _p, _i, _l, _d, _p, _i, _l, _i]
_lib.gen9blas_symm.argtypes = [_i, _i, _i, _i, _i, _d, _p, _i, _p, _i, _d, _p, _i]
_lib.gen9blas_syrk.argtypes = [_i, _i, _i, _i, _i, _d, _p, _i, _d, _p, _i]
_lib.gen9blas_syr2k.argtypes = [_i, _i, _i, _i, _i, _d, _p, _i, _p, _i, _d, _p, _i]
_lib.gen9blas_trmm.argtypes = [_i, _i, _i, _i, _i, _i, _i, _d, _p, _i, _p, _i]
_lib.gen9blas_trsm.argtypes = [_i, _i, _i, _i, _i, _i, _i, _d, _p, _i, _p, _i]
_lib.gen9blas_last_time_ms.restype = ctypes.c_double
_lib.gen9blas_alloc.argtypes = [ctypes.c_size_t]
_lib.gen9blas_alloc.restype = ctypes.c_void_p
_lib.gen9blas_free.argtypes = [ctypes.c_void_p]
_lib.gen9blas_error.restype = ctypes.c_char_p
GEN9BLAS_H, GEN9BLAS_S, GEN9BLAS_HS, GEN9BLAS_D = 0, 1, 2, 3
_STRONA = {"L": 0, "R": 1}
_TROJKAT = {"L": 0, "U": 1}
_gotowe = False


def _init():
    global _gotowe
    if not _gotowe:
        if _lib.gen9blas_init(os.path.join(_TU, "..", "bench").encode()) != 0:
            raise RuntimeError(_lib.gen9blas_error().decode())
        _gotowe = True


def _sprawdz(w):
    if w != 0:
        raise RuntimeError(_lib.gen9blas_error().decode())


# Pula pamięci USM: przydział w sterowniku kosztuje ~0,5 ms CPU na MB (strony przygotowywane od razu; 64 MB = 33 ms),
# więc zwolnione bloki czekają w puli na kolejne gen9blas.empty tego samego (zaokrąglonego) rozmiaru.
_pula = {}
_pula_bajty = 0
PULA_MAX = 1 << 30


def _rozmiar_bloku(n):
    return max(4096, (n + 65535) // 65536 * 65536 if n > 65536 else (n + 4095) // 4096 * 4096)


def _oddaj(p, n):
    global _pula_bajty
    if _pula_bajty + n <= PULA_MAX:
        _pula.setdefault(n, []).append(p); _pula_bajty += n
    else:
        _lib.gen9blas_free(p)


def free_pool():
    """Oddaje sterownikowi bloki pamięci czekające w puli (tablice w użyciu zostają)."""
    global _pula_bajty
    for lista in _pula.values():
        for p in lista:
            _lib.gen9blas_free(p)
    _pula.clear(); _pula_bajty = 0


def empty(shape, dtype=np.float16):
    """Tablica C-contiguous w pamięci USM host: GPU czyta i pisze ją bez kopii (także jej widoki i wycinki).
    Zwykłe tablice NumPy też idą bez kopii (userptr: pierwsze użycie przypina strony, ~tyle co kopia); gen9blas.empty
    oszczędza to pierwsze przypięcie. Gdy znikną wszystkie widoki tablicy, blok
    wraca do puli (do PULA_MAX bajtów) i jest używany ponownie przez kolejne empty; free_pool() oddaje pulę."""
    global _pula_bajty
    _init()
    dtype = np.dtype(dtype)
    n = int(np.prod(shape)) * dtype.itemsize
    nb = _rozmiar_bloku(n)
    if _pula.get(nb):
        p = _pula[nb].pop(); _pula_bajty -= nb
    else:
        p = _lib.gen9blas_alloc(nb)
        if not p:
            raise MemoryError(_lib.gen9blas_error().decode())
    surowe = (ctypes.c_char * nb).from_address(p)
    f = weakref.finalize(surowe, _oddaj, p, nb)
    f.atexit = False   # przy wyjściu z programu sterownik i tak zwalnia całą pamięć
    return np.frombuffer(surowe, dtype=np.uint8, count=n).view(dtype).reshape(shape)


def _rodzaj(typ, accurate):
    if np.dtype(typ) == np.float32:
        return GEN9BLAS_S
    if np.dtype(typ) == np.float64:
        return GEN9BLAS_D
    if np.dtype(typ) == np.float16:
        return GEN9BLAS_HS if accurate else GEN9BLAS_H
    raise TypeError("obsługiwane typy: float16, float32, float64")


def _m3(X, typ, zapis=False):
    """Operand poziomu 3: (tablica, wskaźnik, ld, trans) dla macierzy 2-D. Wiersze z krokiem (wycinki, np. A[:, :k],
    A[::2]) i widoki transponowane idą bez kopii (ld z kroków NumPy); pozostałe -- kopia przez CPU (zapis: błąd)."""
    X, p, _, _, ld, tr = _mac(X, typ, zapis)
    return X, p, ld, tr


def _c(X, typ):
    return np.ascontiguousarray(X, dtype=typ)


def _kopia(B, typ):
    """Kopia B (wynik w miejscu) w pamięci USM; B ciągłe danego typu kopiuje GPU (bez pracy CPU dla B z gen9blas.empty)."""
    b = empty(B.shape, typ)
    if B.dtype == typ and B.flags.c_contiguous and B.size:
        _sprawdz(_lib.gen9blas_copy(_rodzaj(typ, False), B.size, B.ctypes.data, 1, b.ctypes.data, 1))
    else:
        b[...] = B
    return b


def _wyjscie(C, shape, typ):
    """(C, czy_podane, ldc): nowe C z gen9blas.empty albo podane C o ciągłych wierszach (krok wierszy dowolny)."""
    if C is None:
        return empty(shape, typ), False, shape[-1]
    if C.shape != tuple(shape) or C.dtype != typ or C.strides[-1] != C.itemsize:
        raise ValueError(f"C musi mieć kształt {tuple(shape)}, typ {np.dtype(typ).name} i ciągłe wiersze")
    if C.ndim == 2:
        return C, True, _mac(C, typ, True)[4]
    if not C.flags.c_contiguous:
        raise ValueError("C (paczka) musi być C-contiguous")
    return C, True, shape[-1]


def gemm(A, B, alpha=1.0, beta=0.0, C=None, accurate=False):
    """C = alpha·A·B + beta·C (float16 lub float32 wg A)."""
    _init()
    typ = A.dtype if A.dtype in (np.float16, np.float32, np.float64) else np.float32
    (M, K), (K2, N) = A.shape, B.shape
    if K != K2:
        raise ValueError(f"niezgodne wymiary: {A.shape} · {B.shape}")
    a, pa, lda, ta = _m3(A, typ)
    b, pb, ldb, tb = _m3(B, typ)
    C, jest, ldc = _wyjscie(C, (M, N), typ)
    _sprawdz(_lib.gen9blas_gemm(_rodzaj(typ, accurate), ta, tb, M, N, K, alpha, pa, lda, pb, ldb,
                           beta if jest else 0.0, C.ctypes.data, ldc))
    return C


def hgemm(A, B, alpha=1.0, beta=0.0, C=None, accurate=False):
    return gemm(A if A.dtype == np.float16 else A.astype(np.float16),
                B if B.dtype == np.float16 else B.astype(np.float16), alpha, beta, C, accurate)


def sgemm(A, B, alpha=1.0, beta=0.0, C=None):
    return gemm(A if A.dtype == np.float32 else A.astype(np.float32),
                B if B.dtype == np.float32 else B.astype(np.float32), alpha, beta, C)


def gemm_batched(A, B, alpha=1.0, beta=0.0, C=None, accurate=False):
    """Paczka mnożeń: A (count, M, K), B (count, K, N) -> C (count, M, N), jedno uruchomienie kernela."""
    _init()
    typ = A.dtype if A.dtype in (np.float16, np.float32, np.float64) else np.float32
    a, b = _c(A, typ), _c(B, typ)
    (cnt, M, K), (cnt2, K2, N) = a.shape, b.shape
    if K != K2 or cnt != cnt2:
        raise ValueError(f"niezgodne wymiary: {a.shape} · {b.shape}")
    C, jest, _ = _wyjscie(C, (cnt, M, N), typ)
    _sprawdz(_lib.gen9blas_gemm_batched(_rodzaj(typ, accurate), 0, 0, M, N, K, alpha, a.ctypes.data, K, M * K,
                                   b.ctypes.data, N, K * N, beta if jest else 0.0, C.ctypes.data, N, M * N, cnt))
    return C


def symm(A, B, side="L", uplo="L", alpha=1.0, beta=0.0, C=None, accurate=False):
    _init()
    typ = B.dtype
    a, pa, n, lda, _, uplo = _kwadrat(A, typ, uplo)   # widok A.T: ta sama macierz, trójkąt zamieniony
    b, pb, ldb, tb = _m3(B, typ)
    if tb:
        b, pb, ldb, tb = _m3(_c(B, typ), typ)
    M, N = B.shape
    C, jest, ldc = _wyjscie(C, (M, N), typ)
    _sprawdz(_lib.gen9blas_symm(_rodzaj(typ, accurate), _STRONA[side], _TROJKAT[uplo], M, N, alpha, pa, lda,
                           pb, ldb, beta if jest else 0.0, C.ctypes.data, ldc))
    return C


def syrk(A, uplo="L", trans=False, alpha=1.0, beta=0.0, C=None, accurate=False):
    _init()
    typ = A.dtype
    N, K = (A.shape[1], A.shape[0]) if trans else A.shape
    a, pa, lda, ta = _m3(A, typ)   # widok A.T: zamiana trans
    C, jest, ldc = _wyjscie(C, (N, N), typ)
    if not jest:
        C[...] = 0
    _sprawdz(_lib.gen9blas_syrk(_rodzaj(typ, accurate), _TROJKAT[uplo], int(trans) ^ ta, N, K, alpha, pa, lda,
                           beta if jest else 0.0, C.ctypes.data, ldc))
    return C


def syr2k(A, B, uplo="L", trans=False, alpha=1.0, beta=0.0, C=None, accurate=False):
    _init()
    typ = A.dtype
    N, K = (A.shape[1], A.shape[0]) if trans else A.shape
    a, pa, lda, ta = _m3(A, typ)
    b, pb, ldb, tb = _m3(B, typ)
    if ta != tb:   # jedna flaga trans dla obu: kopia operandu B do układu A
        b, pb, ldb, tb = _m3(np.ascontiguousarray(B.T, dtype=typ).T if ta else _c(B, typ), typ)
    C, jest, ldc = _wyjscie(C, (N, N), typ)
    if not jest:
        C[...] = 0
    _sprawdz(_lib.gen9blas_syr2k(_rodzaj(typ, accurate), _TROJKAT[uplo], int(trans) ^ ta, N, K, alpha, pa, lda,
                            pb, ldb, beta if jest else 0.0, C.ctypes.data, ldc))
    return C


def trmm(A, B, side="L", uplo="L", transA=False, diag=False, alpha=1.0, accurate=False):
    """Zwraca alpha·op(A)·B (side="L") albo alpha·B·op(A) (side="R"); A trójkątna. B nie jest zmieniane."""
    _init()
    typ = B.dtype
    a, pa, _, lda, ta, uplo = _kwadrat(A, typ, uplo)   # widok A.T: zamiana transA i trójkąta
    b = _kopia(B, typ)
    M, N = b.shape
    _sprawdz(_lib.gen9blas_trmm(_rodzaj(typ, accurate), _STRONA[side], _TROJKAT[uplo], int(transA) ^ ta, int(diag), M, N,
                           alpha, pa, lda, b.ctypes.data, N))
    return b


def trsm(A, B, side="L", uplo="L", transA=False, diag=False, alpha=1.0, accurate=False):
    """Zwraca X: op(A)·X = alpha·B (side="L") albo X·op(A) = alpha·B (side="R").
    float32: dokładność FP32; float16: GEMM w FP16 (accurate=True: sumy porcjami do FP32), odwrotności bloków w FP64 (na GPU)."""
    _init()
    typ = B.dtype if B.dtype in (np.float16, np.float32, np.float64) else np.float32
    a, pa, _, lda, ta, uplo = _kwadrat(A, typ, uplo)   # widok A.T: zamiana transA i trójkąta
    b = _kopia(B, typ)
    M, N = b.shape
    _sprawdz(_lib.gen9blas_trsm(_rodzaj(typ, accurate), _STRONA[side], _TROJKAT[uplo], int(transA) ^ ta, int(diag), M, N,
                           alpha, pa, lda, b.ctypes.data, N))
    return b


def last_time_ms():
    return _lib.gen9blas_last_time_ms()


def last_zero_copy():
    return _lib.gen9blas_last_zero_copy()


# ---------------- poziomy 1–2 (float32 / float16; FP16 liczone w FP32, redukcje w FP64) ----------------
# Tablice (i ich widoki, także z krokiem i ujemnym krokiem) idą bez kopii: gen9blas.empty (USM) albo zwykłe (userptr).
# Operacje „w miejscu” wymagają typu float32/float16 w tablicy wynikowej; wejścia innego typu są konwertowane.
_lib.gen9blas_axpy.argtypes = [_i, _i, _d, _p, _i, _p, _i]
_lib.gen9blas_scal.argtypes = [_i, _i, _d, _p, _i]
_lib.gen9blas_copy.argtypes = [_i, _i, _p, _i, _p, _i]
_lib.gen9blas_swap.argtypes = [_i, _i, _p, _i, _p, _i]
_lib.gen9blas_rot.argtypes = [_i, _i, _p, _i, _p, _i, _d, _d]
_lib.gen9blas_dot.argtypes = [_i, _i, _p, _i, _p, _i, ctypes.POINTER(_d)]
_lib.gen9blas_asum.argtypes = [_i, _i, _p, _i, ctypes.POINTER(_d)]
_lib.gen9blas_nrm2.argtypes = [_i, _i, _p, _i, ctypes.POINTER(_d)]
_lib.gen9blas_iamax.argtypes = [_i, _i, _p, _i, ctypes.POINTER(_l)]
_lib.gen9blas_gemv.argtypes = [_i, _i, _i, _i, _d, _p, _i, _p, _i, _d, _p, _i]
_lib.gen9blas_ger.argtypes = [_i, _i, _i, _d, _p, _i, _p, _i, _p, _i]


def _typ(*X):
    """Typ obliczeń z typu pierwszej tablicy: float16, float64 albo (pozostałe) float32."""
    return np.float16 if X[0].dtype == np.float16 else np.float64 if X[0].dtype == np.float64 else np.float32


def _wek(x, typ, zapis=False):
    """(tablica, wskaźnik na najniższy adres, n, inc) dla wektora 1-D; inc < 0 jak w BLAS."""
    if x.ndim != 1:
        raise ValueError("oczekiwany wektor 1-D")
    if x.dtype != typ or x.strides[0] % x.itemsize or x.strides[0] == 0:
        if zapis:
            raise TypeError(f"wynik musi być wektorem {np.dtype(typ).name} z krokiem będącym wielokrotnością elementu")
        x = np.ascontiguousarray(x, dtype=typ)
    n, inc = x.shape[0], x.strides[0] // x.itemsize
    return x, x.ctypes.data + (min(inc, 0) * (n - 1) * x.itemsize if n else 0), n, inc


def _mac(A, typ, zapis=False):
    """(tablica, wskaźnik, wiersze, kolumny, ld, trans) dla macierzy row-major z ld; widok transponowany bez kopii."""
    if A.ndim != 2:
        raise ValueError("oczekiwana macierz 2-D")
    e = A.itemsize
    if A.dtype == typ and A.strides[1] == e and A.strides[0] >= A.shape[1] * e and A.strides[0] % e == 0:
        return A, A.ctypes.data, A.shape[0], A.shape[1], A.strides[0] // e, 0
    if not zapis and A.dtype == typ and A.strides[0] == e and A.strides[1] >= A.shape[0] * e and A.strides[1] % e == 0:
        return A, A.ctypes.data, A.shape[1], A.shape[0], A.strides[1] // e, 1
    if zapis:
        raise TypeError(f"wynik musi być macierzą {np.dtype(typ).name} z ciągłymi wierszami")
    A = np.ascontiguousarray(A, dtype=typ)
    return A, A.ctypes.data, A.shape[0], A.shape[1], A.shape[1], 0


def _rodz(typ):
    return GEN9BLAS_S if typ == np.float32 else GEN9BLAS_D if typ == np.float64 else GEN9BLAS_H


def axpy(a, x, y):
    """y = a·x + y (w miejscu); zwraca y."""
    _init(); t = _typ(y); y, py, n, iy = _wek(y, t, True); x, px, nx, ix = _wek(x, t)
    if nx != n:
        raise ValueError("różne długości")
    _sprawdz(_lib.gen9blas_axpy(_rodz(t), n, a, px, ix, py, iy)); return y


def scal(a, x):
    """x = a·x (w miejscu); zwraca x."""
    _init(); t = _typ(x); x, px, n, ix = _wek(x, t, True)
    _sprawdz(_lib.gen9blas_scal(_rodz(t), n, a, px, ix)); return x


def copy(x, y=None):
    """y = x; bez y: nowa tablica (gen9blas.empty) typu x."""
    _init(); t = _typ(y if y is not None else x)
    if y is None:
        y = empty(x.shape, t)
    y, py, n, iy = _wek(y, t, True); x, px, nx, ix = _wek(x, t)
    if nx != n:
        raise ValueError("różne długości")
    _sprawdz(_lib.gen9blas_copy(_rodz(t), n, px, ix, py, iy)); return y


def swap(x, y):
    _init(); t = _typ(x); x, px, n, ix = _wek(x, t, True); y, py, ny, iy = _wek(y, t, True)
    if ny != n:
        raise ValueError("różne długości")
    _sprawdz(_lib.gen9blas_swap(_rodz(t), n, px, ix, py, iy)); return x, y


def rot(x, y, c, s):
    """x, y = c·x + s·y, c·y − s·x (w miejscu)."""
    _init(); t = _typ(x); x, px, n, ix = _wek(x, t, True); y, py, ny, iy = _wek(y, t, True)
    if ny != n:
        raise ValueError("różne długości")
    _sprawdz(_lib.gen9blas_rot(_rodz(t), n, px, ix, py, iy, c, s)); return x, y


def dot(x, y):
    _init(); t = _typ(x); x, px, n, ix = _wek(x, t); y, py, ny, iy = _wek(y, t)
    if ny != n:
        raise ValueError("różne długości")
    w = _d(); _sprawdz(_lib.gen9blas_dot(_rodz(t), n, px, ix, py, iy, ctypes.byref(w))); return w.value


def asum(x):
    _init(); t = _typ(x); x, px, n, ix = _wek(x, t)
    w = _d(); _sprawdz(_lib.gen9blas_asum(_rodz(t), n, px, ix, ctypes.byref(w))); return w.value


def nrm2(x):
    _init(); t = _typ(x); x, px, n, ix = _wek(x, t)
    w = _d(); _sprawdz(_lib.gen9blas_nrm2(_rodz(t), n, px, ix, ctypes.byref(w))); return w.value


def iamax(x):
    """Indeks (od 0) pierwszego największego |x|; -1 dla pustego wektora."""
    _init(); t = _typ(x); x, px, n, ix = _wek(x, t)
    w = _l(); _sprawdz(_lib.gen9blas_iamax(_rodz(t), n, px, ix, ctypes.byref(w))); return w.value


def gemv(A, x, alpha=1.0, beta=0.0, y=None, trans=False):
    """y = alpha·op(A)·x + beta·y; bez y: nowa tablica. Widok A.T idzie bez kopii."""
    _init(); t = _typ(A); A, pa, M, N, lda, tr = _mac(A, t); tr ^= int(trans)
    x, px, nx, ix = _wek(x, t)
    ny = N if tr else M
    if nx != (M if tr else N):
        raise ValueError("długość x nie pasuje do A")
    if y is None:
        y, beta = empty((ny,), t), 0.0
    y, py, n, iy = _wek(y, t, True)
    if n != ny:
        raise ValueError("długość y nie pasuje do A")
    _sprawdz(_lib.gen9blas_gemv(_rodz(t), tr, M, N, alpha, pa, lda, px, ix, beta, py, iy)); return y


def ger(alpha, x, y, A):
    """A = alpha·x·yᵀ + A (w miejscu); zwraca A."""
    _init(); t = _typ(A); A, pa, M, N, lda, _ = _mac(A, t, True)
    x, px, nx, ix = _wek(x, t); y, py, ny, iy = _wek(y, t)
    if nx != M or ny != N:
        raise ValueError("długości x, y nie pasują do A")
    _sprawdz(_lib.gen9blas_ger(_rodz(t), M, N, alpha, px, ix, py, iy, pa, lda)); return A



_lib.gen9blas_symv.argtypes = [_i, _i, _i, _d, _p, _i, _p, _i, _d, _p, _i]
_lib.gen9blas_trmv.argtypes = [_i, _i, _i, _i, _i, _p, _i, _p, _i]
_lib.gen9blas_trsv.argtypes = [_i, _i, _i, _i, _i, _p, _i, _p, _i]
_lib.gen9blas_syr.argtypes = [_i, _i, _i, _d, _p, _i, _p, _i]
_lib.gen9blas_syr2.argtypes = [_i, _i, _i, _d, _p, _i, _p, _i, _p, _i]
_ODWROTNY = {"L": "U", "U": "L"}


def _kwadrat(A, t, uplo):
    """(tablica, wskaźnik, n, ld, transponowana, uplo w pamięci): widok A.T -- bez kopii, trójkąt zamieniony."""
    A, pa, M, N, lda, tr = _mac(A, t)
    if M != N:
        raise ValueError("oczekiwana macierz kwadratowa")
    return A, pa, N, lda, tr, _ODWROTNY[uplo] if tr else uplo


def symv(A, x, uplo="L", alpha=1.0, beta=0.0, y=None):
    """y = alpha·A·x + beta·y, A symetryczna (używany trójkąt uplo); bez y: nowa tablica."""
    _init(); t = _typ(A); A, pa, n, lda, _, u = _kwadrat(A, t, uplo)
    x, px, nx, ix = _wek(x, t)
    if nx != n:
        raise ValueError("długość x nie pasuje do A")
    if y is None:
        y, beta = empty((n,), t), 0.0
    y, py, ny, iy = _wek(y, t, True)
    if ny != n:
        raise ValueError("długość y nie pasuje do A")
    _sprawdz(_lib.gen9blas_symv(_rodz(t), _TROJKAT[u], n, alpha, pa, lda, px, ix, beta, py, iy)); return y


def trmv(A, x, uplo="L", trans=False, diag=False):
    """x = op(A)·x (w miejscu), A trójkątna; diag=True: jedynki na przekątnej. Zwraca x."""
    _init(); t = _typ(x); A, pa, n, lda, tr, u = _kwadrat(A, t, uplo)
    x, px, nx, ix = _wek(x, t, True)
    if nx != n:
        raise ValueError("długość x nie pasuje do A")
    _sprawdz(_lib.gen9blas_trmv(_rodz(t), _TROJKAT[u], int(trans) ^ tr, int(diag), n, pa, lda, px, ix)); return x


def trsv(A, x, uplo="L", trans=False, diag=False):
    """Rozwiązuje op(A)·z = x i zapisuje z w x (w miejscu); A trójkątna. Wektor roboczy w FP32. Zwraca x."""
    _init(); t = _typ(x); A, pa, n, lda, tr, u = _kwadrat(A, t, uplo)
    x, px, nx, ix = _wek(x, t, True)
    if nx != n:
        raise ValueError("długość x nie pasuje do A")
    _sprawdz(_lib.gen9blas_trsv(_rodz(t), _TROJKAT[u], int(trans) ^ tr, int(diag), n, pa, lda, px, ix)); return x


def syr(alpha, x, A, uplo="L"):
    """A = alpha·x·xᵀ + A na trójkącie uplo (w miejscu); zwraca A."""
    _init(); t = _typ(A); A, pa, M, N, lda, _ = _mac(A, t, True)
    x, px, n, ix = _wek(x, t)
    if not M == N == n:
        raise ValueError("wymiary nie pasują")
    _sprawdz(_lib.gen9blas_syr(_rodz(t), _TROJKAT[uplo], n, alpha, px, ix, pa, lda)); return A


def syr2(alpha, x, y, A, uplo="L"):
    """A = alpha·(x·yᵀ + y·xᵀ) + A na trójkącie uplo (w miejscu); zwraca A."""
    _init(); t = _typ(A); A, pa, M, N, lda, _ = _mac(A, t, True)
    x, px, n, ix = _wek(x, t); y, py, ny, iy = _wek(y, t)
    if not M == N == n == ny:
        raise ValueError("wymiary nie pasują")
    _sprawdz(_lib.gen9blas_syr2(_rodz(t), _TROJKAT[uplo], n, alpha, px, ix, py, iy, pa, lda)); return A


_lib.gen9blas_rotm.argtypes = [_i, _i, _p, _i, _p, _i, ctypes.POINTER(_d)]
_fp = ctypes.POINTER(_f)
_lib.gen9blas_rotg.argtypes = [_fp, _fp, _fp, _fp]
_lib.gen9blas_rotmg.argtypes = [_fp, _fp, _fp, _f, _fp]


def rotm(x, y, param):
    """x, y = h11·x + h12·y, h21·x + h22·y (w miejscu); param: 5 liczb jak w BLAS (flaga, h11, h21, h12, h22)."""
    _init(); t = _typ(x); x, px, n, ix = _wek(x, t, True); y, py, ny, iy = _wek(y, t, True)
    if ny != n:
        raise ValueError("różne długości")
    pr = (_d * 5)(*[float(v) for v in param])
    _sprawdz(_lib.gen9blas_rotm(_rodz(t), n, px, ix, py, iy, pr)); return x, y


def rotg(a, b):
    """(r, z, c, s): obrót Givensa zerujący b (CPU)."""
    w = [_f(a), _f(b), _f(), _f()]
    _lib.gen9blas_rotg(*[ctypes.byref(v) for v in w]); return tuple(v.value for v in w)


def rotmg(d1, d2, x1, y1):
    """(d1, d2, x1, param): zmodyfikowany obrót Givensa (CPU); param jak w BLAS (5 liczb)."""
    w = [_f(d1), _f(d2), _f(x1)]; pr = (_f * 5)()
    _lib.gen9blas_rotmg(*[ctypes.byref(v) for v in w], _f(y1), pr)
    return w[0].value, w[1].value, w[2].value, np.array(pr[:], np.float32)


# ---------------- poziom 2: macierze pasmowe i spakowane (układ wierszowy jak CBLAS RowMajor) ----------------
# Pasmo: tablica (wiersze, KL+KU+1), A[i, j] w wierszu i, kolumnie KL + j − i (symetryczna/trójkątna: górna KL = 0,
# KU = K; dolna KL = K, KU = 0). Spakowana: wektor n(n+1)/2 -- kolejne wiersze trójkąta (górna: j = i..n−1, dolna: j = 0..i).
for _n, _t in {"gen9blas_gbmv": [_i] * 6 + [_d, _p, _i, _p, _i, _d, _p, _i], "gen9blas_sbmv": [_i] * 4 + [_d, _p, _i, _p, _i, _d, _p, _i],
               "gen9blas_tbmv": [_i] * 6 + [_p, _i, _p, _i], "gen9blas_tbsv": [_i] * 6 + [_p, _i, _p, _i],
               "gen9blas_spmv": [_i] * 3 + [_d, _p, _p, _i, _d, _p, _i], "gen9blas_tpmv": [_i] * 5 + [_p, _p, _i],
               "gen9blas_tpsv": [_i] * 5 + [_p, _p, _i], "gen9blas_spr": [_i] * 3 + [_d, _p, _i, _p],
               "gen9blas_spr2": [_i] * 3 + [_d, _p, _i, _p, _i, _p]}.items():
    getattr(_lib, _n).argtypes = _t


def _pasmo(A, t, szer, zapis=False):
    A, pa, M, W, lda, tr = _mac(A, t, zapis)
    if tr or W != szer:
        raise ValueError(f"pasmo: oczekiwana tablica wierszowa o {szer} kolumnach")
    return A, pa, M, lda


def _spak(Ap, t, n, zapis=False):
    Ap, pp, m, inc = _wek(Ap, t, zapis)
    if inc != 1 or m != n * (n + 1) // 2:
        raise ValueError(f"spakowana: oczekiwany ciągły wektor długości {n * (n + 1) // 2}")
    return Ap, pp


def gbmv(A, x, M, N, KL, KU, alpha=1.0, beta=0.0, y=None, trans=False):
    """y = alpha·op(A)·x + beta·y, A: M×N pasmowa (tablica (M, KL+KU+1))."""
    _init(); t = _typ(A); A, pa, m, lda = _pasmo(A, t, KL + KU + 1)
    if m != M:
        raise ValueError("liczba wierszy pasma != M")
    x, px, nx, ix = _wek(x, t)
    if y is None:
        y, beta = empty((N if trans else M,), t), 0.0
    y, py, ny, iy = _wek(y, t, True)
    if nx != (M if trans else N) or ny != (N if trans else M):
        raise ValueError("długości x, y")
    _sprawdz(_lib.gen9blas_gbmv(_rodz(t), int(trans), M, N, KL, KU, alpha, pa, lda, px, ix, beta, py, iy)); return y


def sbmv(A, x, K, uplo="L", alpha=1.0, beta=0.0, y=None):
    """y = alpha·A·x + beta·y, A symetryczna pasmowa (tablica (n, K+1), trójkąt uplo)."""
    _init(); t = _typ(A); A, pa, n, lda = _pasmo(A, t, K + 1)
    x, px, nx, ix = _wek(x, t)
    if y is None:
        y, beta = empty((n,), t), 0.0
    y, py, ny, iy = _wek(y, t, True)
    if nx != n or ny != n:
        raise ValueError("długości x, y")
    _sprawdz(_lib.gen9blas_sbmv(_rodz(t), _TROJKAT[uplo], n, K, alpha, pa, lda, px, ix, beta, py, iy)); return y


def tbmv(A, x, K, uplo="L", trans=False, diag=False):
    """x = op(A)·x (w miejscu), A trójkątna pasmowa (tablica (n, K+1))."""
    _init(); t = _typ(x); A, pa, n, lda = _pasmo(A, t, K + 1); x, px, nx, ix = _wek(x, t, True)
    _sprawdz(_lib.gen9blas_tbmv(_rodz(t), _TROJKAT[uplo], int(trans), int(diag), n, K, pa, lda, px, ix)); return x


def tbsv(A, x, K, uplo="L", trans=False, diag=False):
    """Rozwiązuje op(A)·z = x (w miejscu), A trójkątna pasmowa (tablica (n, K+1))."""
    _init(); t = _typ(x); A, pa, n, lda = _pasmo(A, t, K + 1); x, px, nx, ix = _wek(x, t, True)
    _sprawdz(_lib.gen9blas_tbsv(_rodz(t), _TROJKAT[uplo], int(trans), int(diag), n, K, pa, lda, px, ix)); return x


def spmv(Ap, x, uplo="L", alpha=1.0, beta=0.0, y=None):
    """y = alpha·A·x + beta·y, A symetryczna spakowana (wektor n(n+1)/2)."""
    _init(); t = _typ(Ap); x, px, n, ix = _wek(x, t); Ap, pp = _spak(Ap, t, n)
    if y is None:
        y, beta = empty((n,), t), 0.0
    y, py, ny, iy = _wek(y, t, True)
    _sprawdz(_lib.gen9blas_spmv(_rodz(t), _TROJKAT[uplo], n, alpha, pp, px, ix, beta, py, iy)); return y


def tpmv(Ap, x, uplo="L", trans=False, diag=False):
    """x = op(A)·x (w miejscu), A trójkątna spakowana."""
    _init(); t = _typ(x); x, px, n, ix = _wek(x, t, True); Ap, pp = _spak(Ap, t, n)
    _sprawdz(_lib.gen9blas_tpmv(_rodz(t), _TROJKAT[uplo], int(trans), int(diag), n, pp, px, ix)); return x


def tpsv(Ap, x, uplo="L", trans=False, diag=False):
    """Rozwiązuje op(A)·z = x (w miejscu), A trójkątna spakowana."""
    _init(); t = _typ(x); x, px, n, ix = _wek(x, t, True); Ap, pp = _spak(Ap, t, n)
    _sprawdz(_lib.gen9blas_tpsv(_rodz(t), _TROJKAT[uplo], int(trans), int(diag), n, pp, px, ix)); return x


def spr(alpha, x, Ap, uplo="L"):
    """A = alpha·x·xᵀ + A (spakowana, w miejscu)."""
    _init(); t = _typ(Ap); x, px, n, ix = _wek(x, t); Ap, pp = _spak(Ap, t, n, True)
    _sprawdz(_lib.gen9blas_spr(_rodz(t), _TROJKAT[uplo], n, alpha, px, ix, pp)); return Ap


def spr2(alpha, x, y, Ap, uplo="L"):
    """A = alpha·(x·yᵀ + y·xᵀ) + A (spakowana, w miejscu)."""
    _init(); t = _typ(Ap); x, px, n, ix = _wek(x, t); y, py, ny, iy = _wek(y, t); Ap, pp = _spak(Ap, t, n, True)
    _sprawdz(_lib.gen9blas_spr2(_rodz(t), _TROJKAT[uplo], n, alpha, px, ix, py, iy, pp)); return Ap
