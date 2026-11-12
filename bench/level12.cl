// libgen9blas: BLAS poziomów 1–2. Kompilowany trzy razy: -DPOL (FP16 w pamięci, liczenie w FP32), -DDBL (FP64) i bez (FP32).
// R -- typ obliczeń (float dla FP16/FP32, double dla FP64), także skalarów (alpha, beta, c, s, ...) i buforów roboczych.
// Wektor: bufor + przesunięcie o + krok inc (w elementach, inc może być ujemny); macierz: bufor + przesunięcie + ld
// (row-major). Operandy mogą być tym samym buforem (wspólne strony pamięci hosta), stąd brak restrict.
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#ifdef POL
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef half T;
typedef float R;
typedef float4 R4;
#define F(n) n##_h
#define V4(p) convert_float4(vload4(0, p))
#define VS4(v, p) vstore4(convert_half4(v), 0, p)
typedef half4 TV4;
#define CONV_R4(v) convert_float4(v)
// GW elementów na lane w gemv_nw/copy16. FP32/FP64: 16 B (pełne linie 64 B w komunikacie SIMD16). FP16 zostaje przy 4
// (8 B): przy 8 kompilator robi SIMD8 i bajtowe odczyty po 4 B -- gemv FP16 15,4 -> 11,1 GB/s (results/komunikat-16B-...);
// kopię FP16 robi jądro FP32 (gen9blas_copy).
#define GW 4
typedef float4 RG;
#define VG(p) convert_float4(vload4(0, p))
#define VLR(p) vload4(0, p)
#define KOPIUJG(p, q) vstore4(vload4(0, p), 0, q)
#define DOTG(a, b) dot(a, b)
#elif defined(DBL)
typedef double T;
typedef double R;
typedef double4 R4;
#define F(n) n##_d
#define V4(p) vload4(0, p)
#define VS4(v, p) vstore4(v, 0, p)
typedef double4 TV4;
#define CONV_R4(v) convert_double4(v)
#define GW 2
typedef double2 RG;
#define VG(p) vload2(0, p)
#define VLR(p) vload2(0, p)
#define KOPIUJG(p, q) vstore2(vload2(0, p), 0, q)
#define DOTG(a, b) dot(a, b)
#else
typedef float T;
typedef float R;
typedef float4 R4;
#define F(n) n##_f
#define V4(p) vload4(0, p)
#define VS4(v, p) vstore4(v, 0, p)
typedef float4 TV4;
#define CONV_R4(v) convert_float4(v)
#define GW 4
typedef float4 RG;
#define VG(p) vload4(0, p)
#define VLR(p) vload4(0, p)
#define KOPIUJG(p, q) vstore4(vload4(0, p), 0, q)
#define DOTG(a, b) dot(a, b)
#endif
#define WE(p, i) ((R)(p)[i])
#define WY(p, i, v) ((p)[i] = (T)(v))
typedef long L;

// ---- poziom 1: work-item = 4 kolejne elementy; przy kroku 1 odczyt/zapis wektorowy (vload4/vstore4) ----
// Wcześniej element na work-item: lane czytał 4 B (FP32) / 2 B (FP16) na instrukcję, redukcje FP32 13–17 GB/s,
// FP16 7–8 GB/s (FP64 24 GB/s -- ograniczenie liczbą instrukcji odczytu, nie bajtów).
#define CZTERY(n, ...) const int i0 = get_global_id(0) * 4; if (i0 >= n) return; \
    const int pelne = i0 + 3 < n && (__VA_ARGS__)
__kernel void F(axpy)(int n, R a, __global const T *x, L ox, L ix, __global T *y, L oy, L iy) {
    CZTERY(n, ix == 1 && iy == 1);
    if (pelne) { VS4(fma((R4)a, V4(x + ox + i0), V4(y + oy + i0)), y + oy + i0); return; }
    for (int i = i0; i < min(n, i0 + 4); i++) { const L jy = oy + i * iy; WY(y, jy, fma(a, WE(x, ox + i * ix), WE(y, jy))); }
}
__kernel void F(scal)(int n, R a, __global T *x, L ox, L ix) {
    CZTERY(n, ix == 1);
    if (pelne) { VS4((R4)a * V4(x + ox + i0), x + ox + i0); return; }
    for (int i = i0; i < min(n, i0 + 4); i++) { const L j = ox + i * ix; WY(x, j, a * WE(x, j)); }
}
__kernel void F(copy)(int n, __global const T *x, L ox, L ix, __global T *y, L oy, L iy) {
    CZTERY(n, ix == 1 && iy == 1);
    if (pelne) { vstore4(vload4(0, x + ox + i0), 0, y + oy + i0); return; }
    for (int i = i0; i < min(n, i0 + 4); i++) y[oy + i * iy] = x[ox + i * ix];
}
// COPY ciągłe (krok 1): 16 B na lane w każdym typie (GW elementów). Wersja z 4 elementami: FP64 32 B na lane -- każdy
// z dwóch zapisów subgrupy pokrywał połowę linii 64 B (cel nieczytany -> doczytanie linii), 17 GB/s; FP16 8 B, 13 GB/s.
__kernel void F(copy16)(int n, __global const T *x, L ox, __global T *y, L oy) {
    const int i0 = (int)get_global_id(0) * GW;
    if (i0 >= n) return;
    if (i0 + GW <= n) { KOPIUJG(x + ox + i0, y + oy + i0); return; }
    for (int i = i0; i < n; i++) y[oy + i] = x[ox + i];
}
// rot: x' = c·x + s·y, y' = c·y − s·x; zamiana = 1: swap.
__kernel void F(rot)(int n, __global T *x, L ox, L ix, __global T *y, L oy, L iy, R c, R s, int zamiana) {
    CZTERY(n, ix == 1 && iy == 1);
    if (pelne) {
        __global T *px = x + ox + i0, *py = y + oy + i0;
        if (zamiana) { const TV4 t = vload4(0, px); vstore4(vload4(0, py), 0, px); vstore4(t, 0, py); return; }
        const R4 a = V4(px), b = V4(py);
        VS4(fma((R4)c, a, (R4)s * b), px); VS4(fma((R4)c, b, -(R4)s * a), py); return;
    }
    for (int i = i0; i < min(n, i0 + 4); i++) {
        const L jx = ox + i * ix, jy = oy + i * iy;
        if (zamiana) { const T t = x[jx]; x[jx] = y[jy]; y[jy] = t; continue; }
        const R a = WE(x, jx), b = WE(y, jy);
        WY(x, jx, fma(c, a, s * b)); WY(y, jy, fma(c, b, -s * a));
    }
}
// rotm: x' = h11·x + h12·y, y' = h21·x + h22·y (macierz H wyliczona na CPU z param[0..4]).
__kernel void F(rotm)(int n, __global T *x, L ox, L ix, __global T *y, L oy, L iy, R h11, R h12, R h21, R h22) {
    CZTERY(n, ix == 1 && iy == 1);
    if (pelne) {
        __global T *px = x + ox + i0, *py = y + oy + i0;
        const R4 a = V4(px), b = V4(py);
        VS4(fma((R4)h11, a, (R4)h12 * b), px); VS4(fma((R4)h21, a, (R4)h22 * b), py); return;
    }
    for (int i = i0; i < min(n, i0 + 4); i++) {
        const L jx = ox + i * ix, jy = oy + i * iy;
        const R a = WE(x, jx), b = WE(y, jy);
        WY(x, jx, fma(h11, a, h12 * b)); WY(y, jy, fma(h21, a, h22 * b));
    }
}

// ---- redukcje: suma w FP64, część na grupę roboczą (256), sumowanie części na CPU (≤ 96 liczb) ----
// Pętla po porcjach 4 elementów (krok siatki 4·rozmiar siatki); przy kroku 1 odczyt wektorowy.
// tryb 0: x·y, 1: Σ|x|, 2: Σx² (nrm2 = sqrt na CPU)
#define D4(p) convert_double4(V4(p))
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void F(red)(int n, int tryb, __global const T *x, L ox, L ix, __global const T *y, L oy, L iy, __global double *cz) {
    double s = 0.0;
    const int wekt = ix == 1 && (tryb || iy == 1);
    for (int i0 = get_global_id(0) * 4; i0 < n; i0 += get_global_size(0) * 4) {
        if (wekt && i0 + 3 < n) {
            const double4 v = D4(x + ox + i0);
            if (tryb == 0) s += dot(v, D4(y + oy + i0));
            else if (tryb == 1) s += dot(fabs(v), (double4)1.0);
            else s += dot(v, v);
            continue;
        }
        for (int i = i0; i < min(n, i0 + 4); i++) {
            const double v = WE(x, ox + (L)i * ix);
            if (tryb == 0) s = fma(v, (double)WE(y, oy + (L)i * iy), s);
            else if (tryb == 1) s += fabs(v);
            else s = fma(v, v, s);
        }
    }
    s = work_group_reduce_add(s);
    if (get_local_id(0) == 0) cz[get_group_id(0)] = s;
}
// iamax: pierwszy indeks największego |x|; część na grupę: (wartość, indeks); NaN pomijane. Grup ≤ 96 (GRUPY w gen9blas.c).
// Work-item przegląda swoje elementy rosnąco (porcje po 4, kolejno), więc przy remisie zostaje mniejszy indeks.
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void F(iamax)(int n, __global const T *x, L ox, L ix, __global R *cv) {   // wartości [0, 96), indeksy [96, 192)
    __global int *ci = (__global int *)(cv + 96);
    R bv = -1.0f; int bi = 0x7fffffff;
    for (int i0 = get_global_id(0) * 4; i0 < n; i0 += get_global_size(0) * 4) {
        if (ix == 1 && i0 + 3 < n) {
            const R4 v = fabs(V4(x + ox + i0));
            if (v.x > bv) { bv = v.x; bi = i0; }
            if (v.y > bv) { bv = v.y; bi = i0 + 1; }
            if (v.z > bv) { bv = v.z; bi = i0 + 2; }
            if (v.w > bv) { bv = v.w; bi = i0 + 3; }
            continue;
        }
        for (int i = i0; i < min(n, i0 + 4); i++) { const R v = fabs(WE(x, ox + (L)i * ix)); if (v > bv) { bv = v; bi = i; } }
    }
    const R m = work_group_reduce_max(bv);
    const int k = work_group_reduce_min(bv == m ? bi : 0x7fffffff);
    if (get_local_id(0) == 0) { cv[get_group_id(0)] = m; ci[get_group_id(0)] = k; }
}

// ---- poziom 2 ----
// Maska trójkąta dla elementu (wiersz i, kolumna j): 0 całość, 1 j <= i (dolny), 2 j >= i (górny), 3 j < i, 4 j > i.
// jedn = 1: do sumy dochodzi x z indeksem przekątnej (jedynki na przekątnej przy masce ścisłej 3/4; TRMV diag = U).
inline int w_trojkacie(int i, int j, int tryb) {
    return tryb == 0 || (tryb == 1 && j <= i) || (tryb == 2 && j >= i) || (tryb == 3 && j < i) || (tryb == 4 && j > i);
}
// GEMV N: y = alpha·A·x + beta·y, A: M×N (z maską trójkąta). Subgrupa 16 liczy 4 wiersze (x czytany raz na 4 wiersze),
// każdy lane po 4 kolejne elementy (vload4), krok 64. beta = 0: y nie jest czytany.
#define RW 4
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void F(gemv_n)(int M, int N, R alpha, __global const T *A, L oa, int lda, __global const T *x, L ox, L ix,
                        R beta, __global T *y, L oy, L iy, int tryb, int jedn) {
    const int r0 = (int)(get_global_id(0) / 16) * RW, l = get_sub_group_local_id();
    if (r0 >= M) return;
    const int nr = min(RW, M - r0);
    int a0[RW], a1[RW];   // zakres kolumn [a0, a1) wiersza r0 + r
#pragma unroll
    for (int r = 0; r < RW; r++) {
        const int i = r0 + r;
        a0[r] = tryb == 2 ? i : tryb == 4 ? i + 1 : 0; a1[r] = tryb == 1 ? i + 1 : tryb == 3 ? i : N;
        a0[r] = clamp(a0[r], 0, N); a1[r] = clamp(a1[r], 0, N);
    }
    const int lo = a0[0] & ~3, hi = a1[nr - 1] > a1[0] ? a1[nr - 1] : a1[0];
    const __global T *a = A + oa + (L)r0 * lda;
    R s[RW] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (int j = lo + 4 * l; j < hi; j += 64) {
        R4 xv;
        if (ix == 1 && j + 3 < N) xv = V4(x + ox + j);
        else xv = (R4)(WE(x, ox + j * ix), j + 1 < N ? WE(x, ox + (j + 1) * ix) : 0.0f,
                           j + 2 < N ? WE(x, ox + (j + 2) * ix) : 0.0f, j + 3 < N ? WE(x, ox + (j + 3) * ix) : 0.0f);
#pragma unroll
        for (int r = 0; r < RW; r++) {
            if (r >= nr) continue;
            if (j >= a0[r] && j + 3 < a1[r]) s[r] += dot(V4(a + (L)r * lda + j), xv);
            else for (int t = 0; t < 4; t++)
                if (j + t >= a0[r] && j + t < a1[r]) s[r] = fma(WE(a, (L)r * lda + j + t), ((__private R *)&xv)[t], s[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < RW; r++) {
        R t = sub_group_reduce_add(s[r]);
        if (l == 0 && r < nr) {
            if (jedn) t += WE(x, ox + (L)(r0 + r) * ix);
            const L k = oy + (L)(r0 + r) * iy; WY(y, k, beta == 0.0f ? alpha * t : fma(alpha, t, beta * WE(y, k)));
        }
    }
}
// GEMV N, wariant grupowy (krok 3, 2026-11-12): grupa robocza 16 subgrup liczy te same RW wierszy -- subgrupa sg czyta
// odcinki 256 B od kolumny 64·sg co 1024 elementów, więc grupa dotyka naraz jednej strony 4 KB na wiersz (wersja
// gemv_n: każda subgrupa inne 4 wiersze, w całym GPU ~700 stron naraz -- hipoteza: chybienia TLB). Sumy częściowe
// subgrup składane w pamięci lokalnej. Ta sama maska trójkąta co gemv_n.
#define SGW 16
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16 * SGW, 1, 1)))
__kernel void F(gemv_nw)(int M, int N, R alpha, __global const T *A, L oa, int lda, __global const T *x, L ox, L ix,
                         R beta, __global T *y, L oy, L iy, int tryb, int jedn) {
    __local R czesc[RW][SGW];
    const int r0 = (int)get_group_id(0) * RW, l = get_sub_group_local_id(), sg = get_sub_group_id();
    const int nr = min(RW, M - r0);
    int a0[RW], a1[RW];
#pragma unroll
    for (int r = 0; r < RW; r++) {
        const int i = r0 + r;
        a0[r] = tryb == 2 ? i : tryb == 4 ? i + 1 : 0; a1[r] = tryb == 1 ? i + 1 : tryb == 3 ? i : N;
        a0[r] = clamp(a0[r], 0, N); a1[r] = clamp(a1[r], 0, N);
    }
    // GW elementów (16 B) na lane, subgrupa 16·GW ciągiem -- pełne linie 64 B w każdym komunikacie (wcześniej 4 elementy:
    // FP64 32 B na lane, FP16 8 B; results/message-16B-2026-11-12.txt)
    const int lo = a0[0] & ~(GW - 1), hi = a1[nr - 1] > a1[0] ? a1[nr - 1] : a1[0];
    const __global T *a = A + oa + (L)r0 * lda;
    R s[RW] = { 0.0f, 0.0f, 0.0f, 0.0f };
#ifdef POL
    // FP16, pełna macierz, adresy wyrównane do 16 B: odczyty blokowe (subgrupa = ciągłe 256 B jednym komunikatem; vload
    // half kompilator rozbija na bajtowe odczyty po 4 B). Lane dostaje elementy co 16 -- dla iloczynu bez znaczenia.
    const int blok = tryb == 0 && ix == 1 && (lda & 7) == 0 && ((((ulong)a) | ((ulong)(x + ox))) & 15) == 0;
    if (blok) {
        const int jb = N & ~127;
        for (int j0 = 128 * sg; j0 < jb; j0 += 128 * SGW) {
            const float8 xv = convert_float8(as_half8(intel_sub_group_block_read_us8((const __global ushort *)(x + ox + j0))));
#pragma unroll
            for (int r = 0; r < RW; r++) {
                const __global T *ar = a + (L)min(r, nr - 1) * lda + j0;
                const float8 av = convert_float8(as_half8(intel_sub_group_block_read_us8((const __global ushort *)ar)));
                if (r < nr) s[r] += dot(av.lo, xv.lo) + dot(av.hi, xv.hi);
            }
        }
        for (int j = jb + 16 * sg + l; j < N; j += 16 * SGW) {   // ogon < 128 elementów
            const R xj = WE(x, ox + j);
#pragma unroll
            for (int r = 0; r < RW; r++) if (r < nr) s[r] = fma(WE(a, (L)r * lda + j), xj, s[r]);
        }
    } else
#endif
    for (int j = lo + 16 * GW * sg + GW * l; j < hi; j += 16 * GW * SGW) {
        RG xv;
        if (ix == 1 && j + GW - 1 < N) xv = VG(x + ox + j);
        else {
            R xe[GW];
            for (int t = 0; t < GW; t++) xe[t] = j + t < N ? WE(x, ox + (j + t) * ix) : 0.0f;
            xv = VLR(xe);
        }
#pragma unroll
        for (int r = 0; r < RW; r++) {
            if (r >= nr) continue;
            if (j >= a0[r] && j + GW - 1 < a1[r]) s[r] += DOTG(VG(a + (L)r * lda + j), xv);
            else for (int t = 0; t < GW; t++)
                if (j + t >= a0[r] && j + t < a1[r]) s[r] = fma(WE(a, (L)r * lda + j + t), ((__private R *)&xv)[t], s[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < RW; r++) { R t = sub_group_reduce_add(s[r]); if (l == 0) czesc[r][sg] = t; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg == 0 && l < nr) {
        R t = 0.0f;
        for (int g = 0; g < SGW; g++) t += czesc[l][g];
        if (jedn) t += WE(x, ox + (L)(r0 + l) * ix);
        const L k = oy + (L)(r0 + l) * iy; WY(y, k, beta == 0.0f ? alpha * t : fma(alpha, t, beta * WE(y, k)));
    }
}
// GEMV T: y = alpha·Aᵀ·x + beta·y (y długości N; maska trójkąta jak w gemv_n dla elementu (i, kolumna)). Etap 1:
// work-item = 4 kolumny × część wierszy (wpc), sumy częściowe do P[część·N + kolumna]. Etap 2: y = alpha·Σ P + beta·y.
__kernel void F(gemv_t1)(int M, int N, int wpc, __global const T *A, L oa, int lda, __global const T *x, L ox, L ix,
                         __global R *P, int tryb, int jedn) {
    const int c = get_global_id(0) * 4, r = get_global_id(1);
    if (c >= N) return;
    int i0 = r * wpc, i1 = min(M, i0 + wpc);
    // wiersze, w których któraś z 4 kolumn jest w trójkącie
    if (tryb == 1) i0 = max(i0, c); else if (tryb == 3) i0 = max(i0, c + 1);
    else if (tryb == 2) i1 = min(i1, c + 4); else if (tryb == 4) i1 = min(i1, c + 3);
    const __global T *a = A + oa + c;
    R4 s = 0.0f;
    const int pelne = c + 3 < N;
    for (int i = i0; i < i1; i++) {
        const R xv = WE(x, ox + (L)i * ix);
        R4 v;
        if (pelne) v = V4(a + (L)i * lda);
        else v = (R4)(WE(a, (L)i * lda), c + 1 < N ? WE(a, (L)i * lda + 1) : 0.0f, c + 2 < N ? WE(a, (L)i * lda + 2) : 0.0f, 0.0f);
        if (tryb) v *= (R4)(w_trojkacie(i, c, tryb), w_trojkacie(i, c + 1, tryb), w_trojkacie(i, c + 2, tryb), w_trojkacie(i, c + 3, tryb));
        s = fma(v, (R4)xv, s);
    }
    if (jedn) for (int t = 0; t < 4; t++)   // przekątna (i = c + t) w tej części wierszy
        if (c + t >= r * wpc && c + t < min(M, r * wpc + wpc)) ((__private R *)&s)[t] += WE(x, ox + (L)(c + t) * ix);
    __global R *p = P + (L)r * N + c;
    p[0] = s.x; if (c + 1 < N) p[1] = s.y; if (c + 2 < N) p[2] = s.z; if (c + 3 < N) p[3] = s.w;
}
// GEMV T dla FP16 i pełnej macierzy (tryb 0, bez jedynek): odczyt blokowy -- subgrupa czyta 64 kolejne kolumny wiersza
// (128 B) jednym komunikatem, lane dostaje kolumny c0 + l + 16·t (vload4 half kompilator rozbija na bajtowe odczyty).
// Bez wyrównania do 16 B albo na brzegu (mniej niż 64 kolumny) -- ścieżka zwykła jak gemv_t1. Dla FP32/FP64: zaślepka.
#ifdef POL
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void F(gemv_t1b)(int M, int N, int wpc, __global const T *A, L oa, int lda, __global const T *x, L ox, L ix,
                          __global R *P) {
    const int l = get_sub_group_local_id(), r = get_global_id(1);
    const int c0 = ((int)get_global_id(0) - l) * 4;
    if (c0 >= N) return;   // jednakowe dla całej subgrupy
    const int i0 = r * wpc, i1 = min(M, i0 + wpc);
    __global R *p = P + (L)r * N;
    if (c0 + 64 <= N && (lda & 7) == 0 && (((ulong)(A + oa)) & 15) == 0) {
        const __global ushort *a = (const __global ushort *)(A + oa + c0);
        float4 s = 0.0f;
        for (int i = i0; i < i1; i++)
            s = fma(convert_float4(as_half4(intel_sub_group_block_read_us4(a + (L)i * lda))), (float4)WE(x, ox + (L)i * ix), s);
        p += c0 + l; p[0] = s.x; p[16] = s.y; p[32] = s.z; p[48] = s.w;
        return;
    }
    const int c = c0 + 4 * l;
    if (c >= N) return;
    const __global T *a = A + oa + c;
    float4 s = 0.0f;
    for (int i = i0; i < i1; i++) {
        const R xv = WE(x, ox + (L)i * ix);
        R4 v;
        if (c + 3 < N) v = V4(a + (L)i * lda);
        else v = (R4)(WE(a, (L)i * lda), c + 1 < N ? WE(a, (L)i * lda + 1) : 0.0f, c + 2 < N ? WE(a, (L)i * lda + 2) : 0.0f, 0.0f);
        s = fma(v, (R4)xv, s);
    }
    p += c; p[0] = s.x; if (c + 1 < N) p[1] = s.y; if (c + 2 < N) p[2] = s.z; if (c + 3 < N) p[3] = s.w;
}
#else
__kernel void F(gemv_t1b)(int M, int N, int wpc, __global const T *A, L oa, int lda, __global const T *x, L ox, L ix,
                          __global R *P) { }
#endif
__kernel void F(gemv_t2)(int N, int czesci, __global const R *P, R alpha, R beta, __global T *y, L oy, L iy) {
    const int j = get_global_id(0); if (j >= N) return;
    R t = 0.0f;
    for (int r = 0; r < czesci; r++) t += P[(L)r * N + j];
    const L k = oy + (L)j * iy; WY(y, k, beta == 0.0f ? alpha * t : fma(alpha, t, beta * WE(y, k)));
}
// SYMV jednym przejściem po zapisanym trójkącie (gorny = 0: j <= i, 1: j >= i). Element A[i][j] czytany raz daje
// wkład do y_i (część wierszowa, z przekątną) i y_j (część kolumnowa, bez przekątnej). Work-item = 4 kolumny c..c+3
// × pas wierszy (wpc); subgrupa 16 = grupa 64 kolumn g. Wiersz: suma po 64 kolumnach (sub_group_reduce) -> Q[g·n + i];
// kolumna: sumy po pasie -> P[pas·n + c]. symv2: y = alpha·(Σ P + Σ Q) + beta·y. Wcześniej dwa przejścia (N i T,
// trójkąt czytany dwa razy): 8,2 GB/s FP32 liczone od połowy macierzy.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void F(symv1)(int n, int wpc, int gorny, __global const T *A, L oa, int lda, __global const T *x, L ox, L ix,
                       __global R *P, __global R *Q) {
    const int c = get_global_id(0) * 4, r = get_global_id(1), l = get_sub_group_local_id(), g = get_global_id(0) / 16, cg = g * 64;
    if (cg >= n) return;   // cała subgrupa poza macierzą
    int i0 = r * wpc, i1 = min(n, i0 + wpc);
    if (gorny) i1 = min(i1, cg + 64); else i0 = max(i0, cg);
    const int pelne = c + 3 < n;
    const R4 xc = pelne && ix == 1 ? V4(x + ox + c)
        : (R4)(c < n ? WE(x, ox + (L)c * ix) : 0.0f, c + 1 < n ? WE(x, ox + (L)(c + 1) * ix) : 0.0f,
               c + 2 < n ? WE(x, ox + (L)(c + 2) * ix) : 0.0f, c + 3 < n ? WE(x, ox + (L)(c + 3) * ix) : 0.0f);
    const int4 j = (int4)(c, c + 1, c + 2, c + 3);
    R4 s = 0.0f;
    for (int i = i0; i < i1; i++) {
        const __global T *a = A + oa + (L)i * lda + c;
        const R4 v = pelne ? V4(a) : (R4)(c < n ? WE(a, 0) : 0.0f, c + 1 < n ? WE(a, 1) : 0.0f, c + 2 < n ? WE(a, 2) : 0.0f, 0.0f);
        const R4 vw = v * CONV_R4(-(gorny ? j >= i : j <= i));   // w trójkącie (z przekątną)
        const R4 vk = vw * CONV_R4(-(j != i));                  // bez przekątnej
        s = fma(vk, (R4)WE(x, ox + (L)i * ix), s);
        const R w = sub_group_reduce_add(dot(vw, xc));
        if (l == 0) Q[(L)g * n + i] = w;
    }
    if (c < n) {
        __global R *p = P + (L)r * n + c;
        p[0] = s.x; if (c + 1 < n) p[1] = s.y; if (c + 2 < n) p[2] = s.z; if (c + 3 < n) p[3] = s.w;
    }
}
__kernel void F(symv2)(int n, int czesci, int gorny, __global const R *P, __global const R *Q, R alpha, R beta,
                       __global T *y, L oy, L iy) {
    const int i = get_global_id(0); if (i >= n) return;
    R t = 0.0f;
    for (int r = 0; r < czesci; r++) t += P[(L)r * n + i];
    const int g0 = gorny ? i / 64 : 0, g1 = gorny ? (n + 63) / 64 : i / 64 + 1;   // grupy kolumn z wpisem dla wiersza i
    for (int g = g0; g < g1; g++) t += Q[(L)g * n + i];
    const L k = oy + (L)i * iy; WY(y, k, beta == 0.0f ? alpha * t : fma(alpha, t, beta * WE(y, k)));
}
// SYR / SYR2 na trójkącie (tryb 1 dolny, 2 górny): A += alpha·x·yᵀ (dwa = 0; SYR: y = x) albo alpha·(x·yᵀ + y·xᵀ).
__kernel void F(syr2)(int n, R alpha, __global const T *x, L ox, L ix, __global const T *y, L oy, L iy,
                      __global T *A, L oa, int lda, int tryb, int dwa) {
    const int j = get_global_id(0), i = get_global_id(1);
    if (j >= n || i >= n || !w_trojkacie(i, j, tryb)) return;
    const L k = oa + (L)i * lda + j;
    const R xi = WE(x, ox + (L)i * ix), yj = WE(y, oy + (L)j * iy);
    R v = fma(alpha * xi, yj, WE(A, k));
    if (dwa) v = fma(alpha * WE(y, oy + (L)i * iy), WE(x, ox + (L)j * ix), v);
    WY(A, k, v);
}
// TRSV blokowo, wektor roboczy xf w FP32. op(A)[r][c] = tr ? A[c][r] : A[r][c].
// Konwersje: xf = x (do_f) i x = xf (z_f).
__kernel void F(do_f)(int n, __global const T *x, L ox, L ix, __global R *xf) {
    const int i = get_global_id(0); if (i < n) xf[i] = WE(x, ox + (L)i * ix);
}
__kernel void F(z_f)(int n, __global const R *xf, __global T *x, L ox, L ix) {
    const int i = get_global_id(0); if (i < n) WY(x, ox + (L)i * ix, xf[i]);
}
// TRSV blokowo (bloki 64): blok przekątny w jednej grupie roboczej (pamięć lokalna), potem aktualizacja reszty
// wektora. Wektor roboczy xf w FP32. op(A)[r][c] = tr ? A[c][r] : A[r][c]. Wersja jednym uruchomieniem (flagi
// gotowości bloków, bez globalnej synchronizacji) okazała się 3–8× wolniejsza -- NOTATKI.md (TRSV).
// Blok przekątny [p, p+b), b <= NBT (256), jedna grupa robocza 256: 4 podbloki po 64 -- podblok wczytany do pamięci
// lokalnej i podstawiany (dolna = op(A) dolna: w przód), potem aktualizacja pozostałych wierszy bloku; wektor bloku
// w pamięci lokalnej. Wcześniej blok 64 (2 uruchomienia na 64 wiersze; n = 8192: 256 uruchomień, CPU 40–51%).
#define NBT 256
#define OPA(r, c) (tr ? WE(A, oa + (L)(c) * lda + (r)) : WE(A, oa + (L)(r) * lda + (c)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void F(trsv_blok)(int p, int b, __global const T *A, L oa, int lda, int tr, int dolna, int diag, __global R *xf) {
    __local R Lb[64][65];
    __local R xl[NBT];
    const int t = get_local_id(0), ns = (b + 63) / 64;
    for (int i = t; i < b; i += 256) xl[i] = xf[p + i];
    for (int q = 0; q < ns; q++) {
        const int s0 = (dolna ? q : ns - 1 - q) * 64, bs = min(64, b - s0);
#pragma unroll
        for (int e = t; e < 64 * 64; e += 256) {   // 16 niezależnych odczytów na wątek (rozwinięte)
            const int r = e / 64, c = e % 64;
            Lb[r][c] = r < bs && c < bs ? OPA(p + s0 + r, p + s0 + c) : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // podblok 64 w porcjach po 16 wierszy: porcję rozwiązuje pierwsza subgrupa samymi rozgłoszeniami (bez barier),
        // potem cała grupa aktualizuje pozostałe wiersze podbloku -- 2 bariery na 16 wierszy zamiast 16
        const int ng = (bs + 15) / 16;
        for (int gq = 0; gq < ng; gq++) {
            const int g0 = (dolna ? gq : ng - 1 - gq) * 16, gs = min(16, bs - g0);
            if (t < 16) {
                R v = t < gs ? xl[s0 + g0 + t] : 0.0f;
                for (int i = 0; i < gs; i++) {
                    const int k = dolna ? i : gs - 1 - i;
                    R vk = sub_group_broadcast(v, k);
                    if (!diag) vk /= Lb[g0 + k][g0 + k];
                    if (t == k) v = vk;
                    else if (t < gs && (dolna ? t > k : t < k)) v = fma(-Lb[g0 + t][g0 + k], vk, v);
                }
                if (t < gs) xl[s0 + g0 + t] = v;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            const int u0 = dolna ? g0 + gs : 0, u1 = dolna ? bs : g0;   // pozostałe wiersze podbloku
            if (t >= u0 && t < u1) {
                R s = 0.0f;
                for (int c = 0; c < gs; c++) s = fma(Lb[t][g0 + c], xl[s0 + g0 + c], s);
                xl[s0 + t] -= s;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        // pozostałe wiersze bloku w kierunku rozwiązywania: dolna [s0+bs, b), górna [0, s0)
        const int r0 = dolna ? s0 + bs : 0, r1 = dolna ? b : s0;
        for (int r = r0 + t; r < r1; r += 256) {
            R s = 0.0f;
            if (bs == 64) {   // odczyty rozwinięte: bez czekania na każdy (wersja z pętlą: ~260 µs na blok 256)
#pragma unroll 16
                for (int c = 0; c < 64; c++) s = fma(OPA(p + r, p + s0 + c), xl[s0 + c], s);
            } else for (int c = 0; c < bs; c++) s = fma(OPA(p + r, p + s0 + c), xl[s0 + c], s);
            xl[r] -= s;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int i = t; i < b; i += 256) xf[p + i] = xl[i];
}
// Aktualizacja reszty po bloku [p, p+b): xf[r] -= Σ_{c w bloku} op(A)[r][c]·xf[c] dla r w [r0, r1).
// Bez transpozycji: wiersz r ciągły w pamięci -- subgrupa na wiersz, lane po 4 kolejne elementy w porcjach po 64.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void F(trsv_akt_n)(int p, int b, int r0, int r1, __global const T *A, L oa, int lda, __global R *xf) {
    const int r = r0 + (int)(get_global_id(0) / 16), l = get_sub_group_local_id();
    if (r >= r1) return;
    R s = 0.0f;
#pragma unroll
    for (int c0 = 0; c0 < NBT; c0 += 64)
        if (c0 + 4 * l + 3 < b) s += dot(V4(A + oa + (L)r * lda + p + c0 + 4 * l), vload4(0, xf + p + c0 + 4 * l));
        else for (int t = 0; t < 4; t++) { const int c = c0 + 4 * l + t; if (c < b) s = fma(WE(A, oa + (L)r * lda + p + c), xf[p + c], s); }
    s = sub_group_reduce_add(s);
    if (l == 0) xf[r] -= s;
}
// Z transpozycją: op(A)[r][c] = A[c][r] -- work-item na r (kolumna A), pętla po wierszach bloku (odczyt ciągły w grupie).
__kernel void F(trsv_akt_t)(int p, int b, int r0, int r1, __global const T *A, L oa, int lda, __global R *xf) {
    const int r = r0 + get_global_id(0);
    if (r >= r1) return;
    R s = 0.0f;
#pragma unroll 16
    for (int c = 0; c < b; c++) s = fma(WE(A, oa + (L)(p + c) * lda + r), xf[p + c], s);
    xf[r] -= s;
}
// GER: A += alpha·x·yᵀ, A: M×N; work-item = element.
__kernel void F(ger)(int M, int N, R alpha, __global const T *x, L ox, L ix, __global const T *y, L oy, L iy,
                     __global T *A, L oa, int lda) {
    const int j = get_global_id(0), i = get_global_id(1);
    if (j >= N || i >= M) return;
    const L k = oa + (L)i * lda + j;
    WY(A, k, fma(alpha * WE(x, ox + (L)i * ix), WE(y, oy + (L)j * iy), WE(A, k)));
}

// ---- poziom 2: macierze pasmowe i spakowane (układ wierszowy jak CBLAS RowMajor) ----
// Format fmt: 0 pasmo (KL pod, KU nad przekątną; A(r,c) = a[r·lda + KL + c − r]), 1 spakowana górna (wiersz r:
// c = r..n−1 od r·n − r(r−1)/2), 2 spakowana dolna (wiersz r: c = 0..r od r(r+1)/2). n -- liczba kolumn A.
inline R el(int fmt, __global const T *A, L oa, int lda, int KL, int KU, int n, int r, int c) {
    if (fmt == 0) return (c - r < -KL || c - r > KU) ? 0.0f : WE(A, oa + (L)r * lda + KL + c - r);
    if (fmt == 1) return c < r ? 0.0f : WE(A, oa + (L)r * n - (L)r * (r - 1) / 2 + c - r);
    return c > r ? 0.0f : WE(A, oa + (L)r * (r + 1) / 2 + c);
}
// Element op(A)(i, j): sym -- symetryczna z zapisanego trójkąta (gorny); inaczej tr ? A(j, i) : A(i, j); jedn -- jedynki
// na przekątnej (trójkątna).
inline R elop(int fmt, __global const T *A, L oa, int lda, int KL, int KU, int n, int i, int j, int tr, int sym, int gorny, int jedn) {
    if (jedn && i == j) return 1.0f;
    if (sym) return gorny ? el(fmt, A, oa, lda, KL, KU, n, min(i, j), max(i, j)) : el(fmt, A, oa, lda, KL, KU, n, max(i, j), min(i, j));
    return tr ? el(fmt, A, oa, lda, KL, KU, n, j, i) : el(fmt, A, oa, lda, KL, KU, n, i, j);
}
// y = alpha·op(A)·x + beta·y; work-item na element wyniku, j w [j0(i), j1(i)) -- zakres niezerowy op(A) w wierszu i.
// lo, hi: op(A)(i, j) może być niezerowe dla j w [i − lo, i + hi]; nx -- długość x.
__kernel void F(pmv)(int ny, int nx, int fmt, int KL, int KU, int n, __global const T *A, L oa, int lda,
                     __global const T *x, L ox, L ix, R alpha, R beta, __global T *y, L oy, L iy,
                     int tr, int sym, int gorny, int jedn, int lo, int hi) {
    const int i = get_global_id(0); if (i >= ny) return;
    const int j0 = max(0, i - lo), j1 = min(nx, i + hi + 1);
    R s = 0.0f;
    for (int j = j0; j < j1; j++) s = fma(elop(fmt, A, oa, lda, KL, KU, n, i, j, tr, sym, gorny, jedn), WE(x, ox + (L)j * ix), s);
    const L k = oy + (L)i * iy; WY(y, k, beta == 0.0f ? alpha * s : fma(alpha, s, beta * WE(y, k)));
}
// Rozwiązanie op(A)·x = b dla trójkątnej pasmowej / spakowanej: jedna grupa robocza 256, wiersz po wierszu
// (dolna = op(A) dolna: w przód), iloczyn po zakresie (pasmo: K, spakowana: cały wiersz) równolegle + redukcja.
// xf: wektor roboczy FP32 (b na wejściu, x na wyjściu).
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void F(psv)(int n, int fmt, int KL, int KU, __global const T *A, L oa, int lda, __global R *xf,
            int tr, int dolna, int jedn, int K) {
    const int t = get_local_id(0);
    for (int q = 0; q < n; q++) {
        const int i = dolna ? q : n - 1 - q;
        const int j0 = dolna ? max(0, i - K) : i + 1, j1 = dolna ? i : min(n, i + K + 1);
        R s = 0.0f;
        for (int j = j0 + t; j < j1; j += 256) s = fma(elop(fmt, A, oa, lda, KL, KU, n, i, j, tr, 0, 0, 0), xf[j], s);
        s = work_group_reduce_add(s);
        if (t == 0) { R v = xf[i] - s; if (!jedn) v /= elop(fmt, A, oa, lda, KL, KU, n, i, i, tr, 0, 0, 0); xf[i] = v; }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}
// SPR / SPR2 na macierzy spakowanej (fmt 1 górna, 2 dolna): A += alpha·x·yᵀ (dwa = 0, y = x) albo alpha·(x·yᵀ + y·xᵀ).
__kernel void F(spr2)(int n, int fmt, R alpha, __global const T *x, L ox, L ix, __global const T *y, L oy, L iy,
                      __global T *A, L oa, int dwa) {
    const int j = get_global_id(0), i = get_global_id(1);
    if (j >= n || i >= n || (fmt == 1 ? j < i : j > i)) return;
    const L k = oa + (fmt == 1 ? (L)i * n - (L)i * (i - 1) / 2 + j - i : (L)i * (i + 1) / 2 + j);
    R v = fma(alpha * WE(x, ox + (L)i * ix), WE(y, oy + (L)j * iy), WE(A, k));
    if (dwa) v = fma(alpha * WE(y, oy + (L)i * iy), WE(x, ox + (L)j * ix), v);
    WY(A, k, v);
}

// ---- chudy GEMM (małe M), 2026-11-12: PLAN_CHUDY_GEMM.md ----
// C (M×N) = alpha·A·op(W) + beta·C bez pakowania; wagi czytane raz na kafel MT wierszy A; akumulacja w FP32 (R).
// Dowolne kroki: A[m][k] = A[m·sam + k·sak], C[m][n] = C[m·scm + n·scn] (zamiana ról przy małym N: gen9blas.c).
// FP64 pominięte (pamięć lokalna i rejestry; dla D zostaje dawna ścieżka).
#ifndef DBL
#ifdef POL
#define CHUDY_BLOK 1
#define BLOK4(p) convert_float4(as_half4(intel_sub_group_block_read_us4((const __global ushort *)(p))))
#define V8C(p) convert_float8(vload8(0, p))
#else
#define CHUDY_BLOK 0
#define BLOK4(p) vload4(0, p)
#define V8C(p) vload8(0, p)
#endif
// (1) Wagi ciągłe wzdłuż K: element (k, n) = W[n·ldw + k] (C = A·Wᵀ). Układ jak gemv_nw (24–25 GB/s): grupa = 16 subgrup
// na tych samych RW wierszach wag (mało stron naraz -- lekcja z GEMV N na userptr); lane czyta 4 kolejne k, subgrupa 64 k
// ciągiem, subgrupy co 64, krok 1024. A czytane tak samo (ciągle, każdy odczyt dla RW wierszy wag). Wersja 1
// (A wspólne dla subgrupy, lane = wiersz wag) była ograniczona instrukcjami: ~60 GFLOPS, FP16 M = 1 7,5 GB/s.
// Wymiar 0 siatki = kafel M (najszybszy): kolejne grupy czytają te same wiersze wag, gdy są jeszcze w pamięci podręcznej.
#define CHUDY_K(MT, RW) \
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(256, 1, 1))) \
__kernel void F(chudy_k##MT)(int M, int N, int K, R alpha, R beta, __global const T *A, L sam, L sak, \
                             __global const T *W, L ldw, __global T *C, L scm, L scn) { \
    __local float czesc[16][RW * MT]; \
    const int l = get_sub_group_local_id(), sg = get_sub_group_id(); \
    const int m0 = (int)get_group_id(0) * MT, r0 = (int)get_group_id(1) * RW, mt = min(MT, M - m0); \
    /* wskaźniki wierszy raz przed pętlą (64-bitowe mnożenie na Gen9 jest emulowane); wiersze poza zakresem -- */ \
    /* kopia ostatniego ważnego (liczone, nie zapisywane), więc pętla bez warunków */ \
    const __global T *wr[RW], *ar[MT]; \
    __attribute__((opencl_unroll_hint)) for (int r = 0; r < RW; r++) wr[r] = W + (L)min(r0 + r, N - 1) * ldw; \
    __attribute__((opencl_unroll_hint)) for (int m = 0; m < MT; m++) ar[m] = A + (L)(m0 + min(m, mt - 1)) * sam; \
    float s[RW][MT]; \
    __attribute__((opencl_unroll_hint)) for (int r = 0; r < RW; r++) \
        __attribute__((opencl_unroll_hint)) for (int m = 0; m < MT; m++) s[r][m] = 0.0f; \
    int j = 64 * sg + 4 * l; \
    if (CHUDY_BLOK && sak == 1 && (K & 63) == 0 && ((ldw | sam) & 7) == 0 && \
        ((((ulong)wr[0]) | ((ulong)ar[0])) & 15) == 0) {   /* FP16: odczyty blokowe (lane = k co 16) */ \
        for (int j0 = 64 * sg; j0 < K; j0 += 1024) { \
            float4 wv[RW]; \
            __attribute__((opencl_unroll_hint)) for (int r = 0; r < RW; r++) wv[r] = BLOK4(wr[r] + j0); \
            __attribute__((opencl_unroll_hint)) for (int m = 0; m < MT; m++) { \
                const float4 av = BLOK4(ar[m] + j0); \
                __attribute__((opencl_unroll_hint)) for (int r = 0; r < RW; r++) s[r][m] += dot(wv[r], av); \
            } \
        } \
        j = K; \
    } else if (sak == 1) { \
        for (; j + 3 < K; j += 1024) { \
            float4 wv[RW]; \
            __attribute__((opencl_unroll_hint)) for (int r = 0; r < RW; r++) wv[r] = V4(wr[r] + j); \
            __attribute__((opencl_unroll_hint)) for (int m = 0; m < MT; m++) { \
                const float4 av = V4(ar[m] + j); \
                __attribute__((opencl_unroll_hint)) for (int r = 0; r < RW; r++) s[r][m] += dot(wv[r], av); \
            } \
        } \
    } else { \
        for (; j + 3 < K; j += 1024) { \
            float4 wv[RW]; \
            __attribute__((opencl_unroll_hint)) for (int r = 0; r < RW; r++) wv[r] = V4(wr[r] + j); \
            const L o = (L)j * sak; \
            __attribute__((opencl_unroll_hint)) for (int m = 0; m < MT; m++) { \
                const __global T *am = ar[m] + o; \
                const float4 av = (float4)((float)am[0], (float)am[sak], (float)am[2 * sak], (float)am[3 * sak]); \
                __attribute__((opencl_unroll_hint)) for (int r = 0; r < RW; r++) s[r][m] += dot(wv[r], av); \
            } \
        } \
    } \
    if (j < K) {   /* ogon: 1–3 elementy K (jeden przebieg) */ \
        float4 wv[RW]; \
        __attribute__((opencl_unroll_hint)) for (int r = 0; r < RW; r++) \
            wv[r] = (float4)((float)wr[r][j], j + 1 < K ? (float)wr[r][j + 1] : 0.0f, j + 2 < K ? (float)wr[r][j + 2] : 0.0f, 0.0f); \
        __attribute__((opencl_unroll_hint)) for (int m = 0; m < MT; m++) { \
            const __global T *am = ar[m] + (L)j * sak; \
            const float4 av = (float4)((float)am[0], j + 1 < K ? (float)am[sak] : 0.0f, j + 2 < K ? (float)am[2 * sak] : 0.0f, 0.0f); \
            __attribute__((opencl_unroll_hint)) for (int r = 0; r < RW; r++) s[r][m] += dot(wv[r], av); \
        } \
    } \
    __attribute__((opencl_unroll_hint)) for (int r = 0; r < RW; r++) \
        __attribute__((opencl_unroll_hint)) for (int m = 0; m < MT; m++) { \
            const float t = sub_group_reduce_add(s[r][m]); \
            if (l == 0) czesc[sg][r * MT + m] = t; \
        } \
    barrier(CLK_LOCAL_MEM_FENCE); \
    const int o = sg * 16 + l; \
    if (o < RW * MT) { \
        const int r = o / MT, m = o % MT, n = r0 + r; \
        if (m < mt && n < N) { \
            float t = 0.0f; \
            for (int g = 0; g < 16; g++) t += czesc[g][o]; \
            const L kc = (L)(m0 + m) * scm + (L)n * scn; \
            WY(C, kc, beta == 0.0f ? alpha * t : fma(alpha, t, beta * WE(C, kc))); \
        } \
    } \
}
CHUDY_K(1, 4)
CHUDY_K(4, 4)
CHUDY_K(8, 2)   // MT = 8 przy RW = 4 (32 akumulatory) -> spill rejestrów (spill_size 448–640 B), kcl tego nie obsługuje

// (1b) Przebudowa 2026-11-12 (results/skinny-gemm-rework-2026-11-12.txt): wagi ciągłe wzdłuż K, A ciągłe wzdłuż K (sak = 1),
// K % (8·KL) = 0, A, W i kroki wyrównane do 16 B (sprawdza gen9blas.c; ścieżka zapasowa w jądrze przelewała rejestry). SIMD8 (float na lane = 1 GRF -> 8·MT akumulatorów bez spilla), RW = 8 wierszy wag,
// S = 4 subgrupy co 8·KL k (krok 32·KL -> długa pętla, tani epilog). Lane czyta 8 B (FP16: KL = 4, FP32: KL = 2) --
// odczyt blokowy jest przeplatany (element c lane'u l = p[l + 8c]); ten sam układ dla A i wag, więc iloczyn się zgadza.
// Akumulacja jawnym fma, konwersja składowa po składowej.
// Surowe dane trzymane jako dwordy (ushort4 IGC trzymał po 1 GRF na składową). FP16: KL = 2, jeden dword na lane
// (para k, k+1; komunikat 32 B) -- przy KL = 4 (uint2) r8 przelewał rejestry (spill 1056 B: 64 akumulatory + surowe A i W
// + konwersje wyciągnięte przed pętlę). FP32: KL = 2, uint2 na lane (komunikat 64 B).
#ifdef POL
#define KL 4
#define RRAW uint2
#define RREAD(p) intel_sub_group_block_read2((const __global uint *)(p))
#define RCONV(x, c) ({ const half2 h_ = as_half2(((c) >> 1) ? (x).s1 : (x).s0); (float)(((c) & 1) ? h_.s1 : h_.s0); })
#else
#define KL 2
#define RRAW uint2
#define RREAD(p) intel_sub_group_block_read2((const __global uint *)(p))
#define RCONV(x, c) as_float((c) ? (x).s1 : (x).s0)
#endif
// Akumulatory: osobny wektor na wiersz wag (s0..s7, składowa = m). Tablica float s[8][8] (256 B na lane) trafiała do
// pamięci prywatnej (scratch, kcl jej nie konfiguruje -> złe wiersze wyniku; przez NEO dobrze; 2026-11-12) --
// IGC przenosi do rejestrów tylko mniejsze tablice (s[8][4] = 128 B przechodziło), osobne zmienne zawsze.
#ifndef CHUDY_S
#define CHUDY_S 16   // subgrupy na grupę (gen9blas.c: CHUDY_S)
#endif
#define CHUDY_W8(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7)
#define CR_ZERO(r) vr_t s##r = 0.0f;
#define CR_FMA(r) s##r = fma((vr_t)RCONV(rw[r], c), av, s##r);
// Krok redukcji transpozycyjnej: para wektorów (x, y) -> x; lane z bitem b zatrzymuje y, partner x.
#define CR_PARA(x, y, b) { const int g_ = (l & (b)) != 0; const vr_t wy_ = g_ ? x : y, zo_ = g_ ? y : x; \
    x = zo_ + intel_sub_group_shuffle_xor(wy_, (b)); }
#define CHUDY_PETLA(CZYTAJ) \
    for (int j = 8 * KL * sg; j < K; j += 8 * KL * CHUDY_S) { \
        RRAW rw[8], ra[MTc]; \
        __attribute__((opencl_unroll_hint(8))) for (int r = 0; r < 8; r++) rw[r] = CZYTAJ(wr[r] + j); \
        __attribute__((opencl_unroll_hint(MTc))) for (int m = 0; m < MTc; m++) ra[m] = CZYTAJ(ar[m] + j); \
        __attribute__((opencl_unroll_hint(KL))) for (int c = 0; c < KL; c++) { \
            vr_t av; \
            __attribute__((opencl_unroll_hint(MTc))) for (int m = 0; m < MTc; m++) \
                ((__private float *)&av)[m] = RCONV(ra[m], c); \
            CHUDY_W8(CR_FMA) \
        } \
    }
#define CHUDY_R(MT, VR) \
__attribute__((intel_reqd_sub_group_size(8))) __attribute__((reqd_work_group_size(8 * CHUDY_S, 1, 1))) \
__kernel void F(chudy_r##MT)(int M, int N, int K, R alpha, R beta, __global const T *A, L sam, \
                             __global const T *W, L ldw, __global T *C, L scm, L scn) { \
    typedef VR vr_t; enum { MTc = MT }; \
    __local float czesc[CHUDY_S][8 * MT]; \
    const int l = get_sub_group_local_id(), sg = get_sub_group_id(); \
    const int m0 = (int)get_group_id(0) * MT, r0 = (int)get_group_id(1) * 8, mt = min(MT, M - m0); \
    const __global T *wr[8], *ar[MT]; \
    __attribute__((opencl_unroll_hint(8))) for (int r = 0; r < 8; r++) wr[r] = W + (L)min(r0 + r, N - 1) * ldw; \
    __attribute__((opencl_unroll_hint(MT))) for (int m = 0; m < MT; m++) ar[m] = A + (L)(m0 + min(m, mt - 1)) * sam; \
    CHUDY_W8(CR_ZERO) \
    CHUDY_PETLA(RREAD) \
    CR_PARA(s0, s4, 4) CR_PARA(s1, s5, 4) CR_PARA(s2, s6, 4) CR_PARA(s3, s7, 4) \
    CR_PARA(s0, s2, 2) CR_PARA(s1, s3, 2) CR_PARA(s0, s1, 1)   /* lane l: s0 = sumy dla wiersza wag l (składowa = m) */ \
    __attribute__((opencl_unroll_hint(MT))) for (int i = 0; i < MT; i++) czesc[sg][l * MT + i] = ((__private float *)&s0)[i]; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    for (int o = sg * 8 + l; o < 8 * MT; o += 8 * CHUDY_S) { \
        const int r = o / MT, m = o % MT, n = r0 + r; \
        if (m < mt && n < N) { \
            float t = 0.0f; \
            for (int g = 0; g < CHUDY_S; g++) t += czesc[g][o]; \
            const L kc = (L)(m0 + m) * scm + (L)n * scn; \
            WY(C, kc, beta == 0.0f ? alpha * t : fma(alpha, t, beta * WE(C, kc))); \
        } \
    } \
}
CHUDY_R(4, float4)
CHUDY_R(8, float8)

// (2) Wagi ciągłe wzdłuż N: element (k, n) = W[k·ldw + n] (C = A·B). Work-item = 4 kolejne kolumny, wymiar 1 siatki =
// część K × kafel M (kafel szybszy: kolejne kafle czytają ten sam fragment wag, póki jest w LLC). MT = 4 (MT = 8: spill). A wczytywane raz na 16 k rozłożone po lane'ach i rozgłaszane
// (sub_group_broadcast) -- wersja 1 czytała A[m][k] osobno dla każdego k (komunikat na FMA). Sumy częściowe
// P[część][m][n] (FP32), drugi etap chudy_n2 sumuje części i robi epilog.
#define CHUDY_N(MT) \
__attribute__((intel_reqd_sub_group_size(16))) \
__kernel void F(chudy_n##MT)(int M, int N, int K, int kpc, __global const T *A, L sam, L sak, \
                             __global const T *W, L ldw, __global float *P) { \
    const int kafle = (M + MT - 1) / MT, cz = (int)get_global_id(1) / kafle, m0 = (int)get_global_id(1) % kafle * MT; \
    const int n = (int)get_global_id(0) * 4; \
    const int l = get_sub_group_local_id(), mt = min(MT, M - m0), k0 = cz * kpc, k1 = min(K, k0 + kpc); \
    const int nn = min(n, ((N - 1) / 4) * 4);   /* lane poza N: czyta ostatnią czwórkę, nie zapisuje */ \
    const int pelne = nn + 3 < N; \
    const __global T *ap[MT]; \
    __attribute__((opencl_unroll_hint)) for (int m = 0; m < MT; m++) ap[m] = A + (L)(m0 + min(m, mt - 1)) * sam; \
    float4 acc[MT]; \
    __attribute__((opencl_unroll_hint)) for (int m = 0; m < MT; m++) acc[m] = 0.0f; \
    const __global T *wr = W + (L)k0 * ldw + nn;   /* wiersz k wag; przesuwany o ldw (bez mnożenia w pętli) */ \
    for (int kb = k0; kb < k1; kb += 16) { \
        float ar[MT]; \
        const int kl = kb + l; \
        __attribute__((opencl_unroll_hint)) for (int m = 0; m < MT; m++) \
            ar[m] = kl < k1 ? (float)(sak == 1 ? ap[m][kl] : ap[m][(L)kl * sak]) : 0.0f; \
        const int kn = min(16, k1 - kb); \
        if (kn == 16 && pelne) { \
            __attribute__((opencl_unroll_hint)) for (int t = 0; t < 16; t++) { \
                const float4 wv = V4(wr); wr += ldw; \
                __attribute__((opencl_unroll_hint)) for (int m = 0; m < MT; m++) \
                    acc[m] = fma((float4)(sub_group_broadcast(ar[m], t)), wv, acc[m]); \
            } \
        } else { \
            for (int t = 0; t < 16; t++) { \
                float4 wv = 0.0f; \
                if (t < kn) { \
                    wv = pelne ? V4(wr) : (float4)((float)wr[0], nn + 1 < N ? (float)wr[1] : 0.0f, nn + 2 < N ? (float)wr[2] : 0.0f, 0.0f); \
                    wr += ldw; \
                } \
                __attribute__((opencl_unroll_hint)) for (int m = 0; m < MT; m++) \
                    acc[m] = fma((float4)(sub_group_broadcast(ar[m], t)), wv, acc[m]); \
            } \
        } \
    } \
    if (n >= N) return; \
    __global float *p = P + ((L)cz * M + m0) * N + n; \
    __attribute__((opencl_unroll_hint)) for (int m = 0; m < MT; m++) if (m < mt) { \
        if (n + 3 < N) vstore4(acc[m], 0, p + (L)m * N); \
        else for (int t = 0; t < 4; t++) if (n + t < N) p[(L)m * N + t] = ((__private float *)&acc[m])[t]; \
    } \
}
CHUDY_N(4)
// Etap 2: C[m][n] = alpha·Σ_cz P[cz][m][n] + beta·C (work-item = element).
__kernel void F(chudy_n2)(int M, int N, int czesci, __global const float *P, R alpha, R beta,
                          __global T *C, L scm, L scn) {
    const int n = (int)get_global_id(0), m = (int)get_global_id(1);
    if (n >= N || m >= M) return;
    float s = 0.0f;
    for (int c = 0; c < czesci; c++) s += P[((L)c * M + m) * N + n];
    const L kc = (L)m * scm + (L)n * scn;
    WY(C, kc, beta == 0.0f ? alpha * s : fma(alpha, s, beta * WE(C, kc)));
}
#endif
