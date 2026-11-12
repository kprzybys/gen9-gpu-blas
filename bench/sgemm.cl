// SGEMM C = A·B (FP32, row-major) dla UHD 620, ta sama metoda co hgemm.cl.
// Wątek SIMD16 liczy kafelek C 16×16 (16 akumulatorów float = 32 GRF). Krok k = 16.
// Dane spakowane (ciągłe kafelki, OWord block read z bufora):
//   Apf: [m/16][k/16][r 0..15][kk 0..15]: blok A 16×16 = 1 KB; block_read8 (uint8) daje 8 wierszy, linia j = kolumna k+j.
//   Bpf: [n/16][k][n%16]: blok B (8 wierszy k × 16 kolumn) = 512 B ciągłe; linia j = kolumna n0+j.
// Mnożenia: acc[r] += A[m0+r][k+kk] (rozgłoszone z linii kk) * B[k+kk][n0+lane], 256 mad (16) :f na krok.
#ifndef GM
#define GM 16
#endif
#define BCF(v, i) as_float(sub_group_broadcast(v, i))

inline uint s8(uint8 v, int i) {
    switch (i) { case 0: return v.s0; case 1: return v.s1; case 2: return v.s2; case 3: return v.s3;
                 case 4: return v.s4; case 5: return v.s5; case 6: return v.s6; default: return v.s7; }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void sgemm_16x16p(__global const uint *Ap, __global const uint *Bp, __global float *C,
                           int M, int ldc, int K) {
    const int pid = get_group_id(0) * get_num_sub_groups() + get_sub_group_id();
    const int npm = M / 16, npn = ldc / 16;
    const int grupa = pid / (GM * npn), m_pierwszy = grupa * GM;
    const int gsz = min(npm - m_pierwszy, GM);
    const int pm = m_pierwszy + (pid % (GM * npn)) % gsz, pn = (pid % (GM * npn)) / gsz;
    float acc[16];
    for (int r = 0; r < 16; r++) acc[r] = 0;
    __global const uint *pa = Ap + (size_t)pm * (K / 16) * 256;
    __global const uint *pb = Bp + (size_t)pn * K * 16;
    for (int k = 0; k < K; k += 16) {
        uint8 a_lo = intel_sub_group_block_read8(pa), a_hi = intel_sub_group_block_read8(pa + 128);
        uint8 b_lo = intel_sub_group_block_read8(pb), b_hi = intel_sub_group_block_read8(pb + 128);
        pa += 256; pb += 256;
        __attribute__((opencl_unroll_hint))
        for (int r = 0; r < 16; r++) {
            uint av = r < 8 ? s8(a_lo, r) : s8(a_hi, r - 8);
            __attribute__((opencl_unroll_hint))
            for (int q = 0; q < 8; q++) acc[r] = fma(BCF(av, q), as_float(s8(b_lo, q)), acc[r]);
            __attribute__((opencl_unroll_hint))
            for (int q = 0; q < 8; q++) acc[r] = fma(BCF(av, q + 8), as_float(s8(b_hi, q)), acc[r]);
        }
    }
    for (int r = 0; r < 16; r++)
        intel_sub_group_block_write((__global uint *)(C + (size_t)(pm * 16 + r) * ldc + pn * 16), as_uint(acc[r]));
}

__kernel void pack_af(__global const float *A, __global float *Ap, int K, int lda) {
    const int k = get_global_id(0), m = get_global_id(1);
    Ap[(((size_t)(m / 16) * (K / 16) + k / 16) * 16 + m % 16) * 16 + k % 16] = A[(size_t)m * lda + k];
}
__kernel void pack_bf(__global const float *B, __global float *Bp, int K, int ldb) {
    const int n = get_global_id(0), k = get_global_id(1);
    Bp[((size_t)(n / 16) * K + k) * 16 + n % 16] = B[(size_t)k * ldb + n];
}

// Układy dla ręcznej pętli asm (asm/sgemm_asm.py), bez konfliktów banków. Krok k16 dzielony na 4 ćwiartki (kl 0..3).
// A2f: ćwiartka = 16 wierszy × 4 kl = 64 floaty = 8 GRF (256 B, offset 256*q w bloku 1 KB).
//      GRF g: wiersze 4*(g/2)..+3; g nieparzysty -> kl parzyste (0,2), g parzysty -> kl nieparzyste; element (r%4)*2 + kl/2.
__kernel void pack_af2(__global const float *A, __global float *Ap, int K, int lda) {
    const int k = get_global_id(0), m = get_global_id(1);
    const int r = m % 16, kk = k % 16, q = kk / 4, kl = kk % 4;
    const int g = 2 * (r / 4) + ((kl % 2 == 0) ? 1 : 0);
    Ap[((size_t)(m / 16) * (K / 16) + k / 16) * 256 + q * 64 + g * 8 + (r % 4) * 2 + kl / 2] = A[(size_t)m * lda + k];
}
// B2f: jak Bpf, ale w każdym bloku 4 wierszy kolejność 0,2,1,3 (parzyste kl i nieparzyste osobno, po 128 B).
__kernel void pack_bf2(__global const float *B, __global float *Bp, int K, int ldb) {
    const int n = get_global_id(0), k = get_global_id(1), kl = k % 4;
    Bp[((size_t)(n / 16) * K + (k - kl) + (kl % 2) * 2 + kl / 2) * 16 + n % 16] = B[(size_t)k * ldb + n];
}

// Pakowanie z brzegami (zera poza M/N/K) dla biblioteki; układy jak pack_af2 / pack_bf2. Siatki: (Kp, Mp) i (Np, Kp).
__kernel void pack_af2_pad(__global const float *A, __global float *Ap, int M, int K, int lda, int Kp) {
    const int k = get_global_id(0), m = get_global_id(1);
    const int r = m % 16, kk = k % 16, q = kk / 4, kl = kk % 4;
    const int g = 2 * (r / 4) + ((kl % 2 == 0) ? 1 : 0);
    Ap[((size_t)(m / 16) * (Kp / 16) + k / 16) * 256 + q * 64 + g * 8 + (r % 4) * 2 + kl / 2] =
        (m < M && k < K) ? A[(size_t)m * lda + k] : 0.0f;
}
__kernel void pack_bf2_pad(__global const float *B, __global float *Bp, int K, int N, int ldb, int Kp) {
    const int n = get_global_id(0), k = get_global_id(1), kl = k % 4;
    Bp[((size_t)(n / 16) * Kp + (k - kl) + (kl % 2) * 2 + kl / 2) * 16 + n % 16] = (n < N && k < K) ? B[(size_t)k * ldb + n] : 0.0f;
}
__kernel void kpg_epilog_f(__global const float *Cp, int ldp, __global float *C, int ldc, float alpha, float beta) {
    const int j = get_global_id(0), i = get_global_id(1);
    float v = alpha * Cp[(size_t)i * ldp + j];
    if (beta != 0.0f) v += beta * C[(size_t)i * ldc + j];
    C[(size_t)i * ldc + j] = v;
}

__kernel void pack_af2_pad_t(__global const float *A, __global float *Ap, int M, int K, int lda, int Kp) {
    const int m = get_global_id(0), k = get_global_id(1);
    const int r = m % 16, kk = k % 16, q = kk / 4, kl = kk % 4;
    const int g = 2 * (r / 4) + ((kl % 2 == 0) ? 1 : 0);
    Ap[((size_t)(m / 16) * (Kp / 16) + k / 16) * 256 + q * 64 + g * 8 + (r % 4) * 2 + kl / 2] =
        (m < M && k < K) ? A[(size_t)k * lda + m] : 0.0f;
}
__kernel void pack_bf2_pad_t(__global const float *B, __global float *Bp, int K, int N, int ldb, int Kp) {
    const int k = get_global_id(0), n = get_global_id(1), kl = k % 4;
    Bp[((size_t)(n / 16) * Kp + (k - kl) + (kl % 2) * 2 + kl / 2) * 16 + n % 16] = (n < N && k < K) ? B[(size_t)n * ldb + k] : 0.0f;
}

// ---------- Wsadowe: kernel z numerem macierzy, pakowanie i epilog z 3. wymiarem siatki ----------
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void sgemm_16x16p_b(__global const uint *Ap, __global const uint *Bp, __global float *C, int M, int ldc, int K) {
    const int pid0 = get_group_id(0) * get_num_sub_groups() + get_sub_group_id();
    const int npm = M / 16, npn = ldc / 16;
    const int b = pid0 / (npm * npn), pid = pid0 % (npm * npn);
    const int grupa = pid / (GM * npn), m_pierwszy = grupa * GM;
    const int gsz = min(npm - m_pierwszy, GM);
    const int pm = m_pierwszy + (pid % (GM * npn)) % gsz, pn = (pid % (GM * npn)) / gsz;
    Ap += (size_t)b * M * K; Bp += (size_t)b * ldc * K; C += (size_t)b * M * ldc;
    float acc[16];
    for (int r = 0; r < 16; r++) acc[r] = 0;
    __global const uint *pa = Ap + (size_t)pm * (K / 16) * 256;
    __global const uint *pb = Bp + (size_t)pn * K * 16;
    for (int k = 0; k < K; k += 16) {
        uint8 a_lo = intel_sub_group_block_read8(pa), a_hi = intel_sub_group_block_read8(pa + 128);
        uint8 b_lo = intel_sub_group_block_read8(pb), b_hi = intel_sub_group_block_read8(pb + 128);
        pa += 256; pb += 256;
        __attribute__((opencl_unroll_hint))
        for (int r = 0; r < 16; r++) {
            uint av = r < 8 ? s8(a_lo, r) : s8(a_hi, r - 8);
            __attribute__((opencl_unroll_hint))
            for (int q = 0; q < 8; q++) acc[r] = fma(BCF(av, q), as_float(s8(b_lo, q)), acc[r]);
            __attribute__((opencl_unroll_hint))
            for (int q = 0; q < 8; q++) acc[r] = fma(BCF(av, q + 8), as_float(s8(b_hi, q)), acc[r]);
        }
    }
    for (int r = 0; r < 16; r++)
        intel_sub_group_block_write((__global uint *)(C + (size_t)(pm * 16 + r) * ldc + pn * 16), as_uint(acc[r]));
}
#define PKF_IDX_A(m, k, Kp) ((((size_t)((m) / 16) * ((Kp) / 16) + (k) / 16) * 256 + ((k) % 16 / 4) * 64 + \
    (2 * ((m) % 16 / 4) + (((k) % 4) % 2 == 0)) * 8 + ((m) % 4) * 2 + ((k) % 4) / 2))
#define PKF_IDX_B(n, k, Kp) ((((size_t)((n) / 16) * (Kp) + ((k) - (k) % 4) + ((k) % 4 % 2) * 2 + ((k) % 4) / 2)) * 16 + (n) % 16)
__kernel void pkf_a(__global const float *A, __global float *Ap, int M, int K, int lda, int Kp, long sA, int Mp) {
    const int k = get_global_id(0), m = get_global_id(1), b = get_global_id(2);
    Ap[(size_t)b * Mp * Kp + PKF_IDX_A(m, k, Kp)] = (m < M && k < K) ? A[b * sA + (size_t)m * lda + k] : 0.0f; }
__kernel void pkf_a_t(__global const float *A, __global float *Ap, int M, int K, int lda, int Kp, long sA, int Mp) {
    const int m = get_global_id(0), k = get_global_id(1), b = get_global_id(2);
    Ap[(size_t)b * Mp * Kp + PKF_IDX_A(m, k, Kp)] = (m < M && k < K) ? A[b * sA + (size_t)k * lda + m] : 0.0f; }
__kernel void pkf_b(__global const float *B, __global float *Bp, int K, int N, int ldb, int Kp, long sB, int Np) {
    const int n = get_global_id(0), k = get_global_id(1), b = get_global_id(2);
    Bp[(size_t)b * Np * Kp + PKF_IDX_B(n, k, Kp)] = (n < N && k < K) ? B[b * sB + (size_t)k * ldb + n] : 0.0f; }
__kernel void pkf_b_t(__global const float *B, __global float *Bp, int K, int N, int ldb, int Kp, long sB, int Np) {
    const int k = get_global_id(0), n = get_global_id(1), b = get_global_id(2);
    Bp[(size_t)b * Np * Kp + PKF_IDX_B(n, k, Kp)] = (n < N && k < K) ? B[b * sB + (size_t)n * ldb + k] : 0.0f; }
__kernel void kpg_epilog_fb(__global const float *Cp, int ldp, int Mp, __global float *C, int ldc, long sC, float alpha, float beta) {
    const int j = get_global_id(0), i = get_global_id(1), b = get_global_id(2);
    float v = alpha * Cp[(size_t)b * Mp * ldp + (size_t)i * ldp + j];
    __global float *c = C + b * sC + (size_t)i * ldc + j;
    if (beta != 0.0f) v += beta * c[0];
    c[0] = v;
}

// Wersja „v” (zob. hgemm_v): (Ap, Bp, C, M, N, ldc, K, Kfull).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void sgemm_v(__global const uint *Ap, __global const uint *Bp, __global float *C,
                      int M, int N, int ldc, int K, int Kfull) {
    const int pid0 = get_group_id(0) * get_num_sub_groups() + get_sub_group_id();
    const int npm = M / 16, npn = N / 16;
    const int b = pid0 / (npm * npn), pid = pid0 % (npm * npn);
    const int grupa = pid / (GM * npn), m_pierwszy = grupa * GM;
    const int gsz = min(npm - m_pierwszy, GM);
    const int pm = m_pierwszy + (pid % (GM * npn)) % gsz, pn = (pid % (GM * npn)) / gsz;
    Ap += (size_t)b * M * Kfull; Bp += (size_t)b * N * Kfull; C += (size_t)b * M * ldc;
    float acc[16];
    for (int r = 0; r < 16; r++) acc[r] = 0;
    __global const uint *pa = Ap + (size_t)pm * (Kfull / 16) * 256;
    __global const uint *pb = Bp + (size_t)pn * Kfull * 16;
    for (int k = 0; k < K; k += 16) {
        uint8 a_lo = intel_sub_group_block_read8(pa), a_hi = intel_sub_group_block_read8(pa + 128);
        uint8 b_lo = intel_sub_group_block_read8(pb), b_hi = intel_sub_group_block_read8(pb + 128);
        pa += 256; pb += 256;
        __attribute__((opencl_unroll_hint))
        for (int r = 0; r < 16; r++) {
            uint av = r < 8 ? s8(a_lo, r) : s8(a_hi, r - 8);
            __attribute__((opencl_unroll_hint))
            for (int q = 0; q < 8; q++) acc[r] = fma(BCF(av, q), as_float(s8(b_lo, q)), acc[r]);
            __attribute__((opencl_unroll_hint))
            for (int q = 0; q < 8; q++) acc[r] = fma(BCF(av, q + 8), as_float(s8(b_hi, q)), acc[r]);
        }
    }
    for (int r = 0; r < 16; r++)
        intel_sub_group_block_write((__global uint *)(C + (size_t)(pm * 16 + r) * ldc + pn * 16), as_uint(acc[r]));
}
__kernel void kpg_epilog_trif(__global const float *Cp, int ldp, __global float *C, int ldc, float alpha, float beta, int uplo) {
    const int j = get_global_id(0), i = get_global_id(1);
    if ((uplo == 0 && j > i) || (uplo == 1 && j < i)) return;
    float v = alpha * Cp[(size_t)i * ldp + j];
    __global float *c = C + (size_t)i * ldc + j;
    if (beta != 0.0f) v += beta * c[0];
    c[0] = v;
}

// ---------- TRSM: przygotowanie i zapis na GPU (float) ----------
// L (n×n, dolna) = T po ewentualnym odwróceniu kolejności indeksów; T = op(A) (lewa) albo op(A)ᵀ (prawa), tr: T = Aᵀ.
__kernel void trsm_L_f(__global const float *A, int lda, __global float *L, int n, int uplo, int tr, int diag, int odwr) {
    const int j = get_global_id(0), i = get_global_id(1);
    const int si = odwr ? n - 1 - i : i, sj = odwr ? n - 1 - j : j;
    const int ai = tr ? sj : si, aj = tr ? si : sj;
    const int w = uplo == 0 ? aj <= ai : aj >= ai;
    float v = w ? A[(size_t)ai * lda + aj] : 0.0f;
    if (diag && si == sj) v = 1.0f;
    L[(size_t)i * n + j] = v;
}
// Y (np × mp) = alpha·R, R = B (lewa) albo Bᵀ (prawa), wiersze w kolejności odwr; dopełnienie zerami.
__kernel void trsm_Y_f(__global const float *B, int ldb, __global float *Y, int n, int m, int mp, int lewa, int odwr, float alpha) {
    const int j = get_global_id(0), i = get_global_id(1);
    float v = 0.0f;
    if (i < n && j < m) { const int si = odwr ? n - 1 - i : i; v = alpha * (float)(lewa ? B[(size_t)si * ldb + j] : B[(size_t)j * ldb + si]); }
    Y[(size_t)i * mp + j] = (float)v;
}
// Wynik z powrotem do B (odwrotnie niż trsm_Y, bez alpha).
__kernel void trsm_zapisz_f(__global const float *Y, int mp, __global float *B, int ldb, int n, int lewa, int odwr) {
    const int j = get_global_id(0), i = get_global_id(1);
    const int si = odwr ? n - 1 - i : i;
    if (lewa) B[(size_t)si * ldb + j] = Y[(size_t)i * mp + j]; else B[(size_t)j * ldb + si] = Y[(size_t)i * mp + j];
}

// ---------- Przygotowanie danych poziomu 3 na GPU (float) ----------
// Pełna macierz n×n z trójkąta uplo macierzy A: sym=1 odbicie (symetryczna), sym=0 trójkątna z zerami
// (diag=1: jedynki na przekątnej); tr=1: wynik transponowany. Kafelki 16×16: oba potrzebne kafelki A (D = A[bi.., bj..]
// i odbity E = A[bj.., bi..]) czytane wierszami do pamięci lokalnej, zapis T wierszami (bez dostępu z krokiem).
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void gen9blas_pelna_f(__global const float *A, int lda, __global float *T, int n, int uplo, int sym, int diag, int tr) {
    __local float D[16][17], E[16][17];
    const int lj = get_local_id(0), li = get_local_id(1), bj = get_group_id(0) * 16, bi = get_group_id(1) * 16;
    const int r = bi + li, c = bj + lj;
    D[li][lj] = r < n && c < n ? A[(size_t)r * lda + c] : 0.0f;
    if (sym || tr) E[li][lj] = bj + li < n && bi + lj < n ? A[(size_t)(bj + li) * lda + bi + lj] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (r >= n || c >= n) return;
    // element wyniku T(r, c) = F(i, j), (i, j) = tr ? (c, r) : (r, c); A[r][c] = D[li][lj], A[c][r] = E[lj][li]
    const int i = tr ? c : r, j = tr ? r : c;
    const float aij = tr ? E[lj][li] : D[li][lj], aji = tr ? D[li][lj] : E[lj][li];
    const int w = uplo == 0 ? j <= i : j >= i;
    float v = w ? aij : (sym ? aji : 0.0f);
    if (!sym && diag && i == j) v = 1.0f;
    T[(size_t)r * n + c] = v;
}
// SYR2K: X = [op(A) | op(B)], Y = [op(B) | op(A)] (N × 2K, ld 2K); trans: A, B zapisane K×N.
__kernel void gen9blas_sklej_f(__global const float *A, int lda, __global const float *B, int ldb, __global float *X, __global float *Y,
                         int K, int trans) {
    const int k = get_global_id(0), i = get_global_id(1);
    const float a = trans ? A[(size_t)k * lda + i] : A[(size_t)i * lda + k], b = trans ? B[(size_t)k * ldb + i] : B[(size_t)i * ldb + k];
    const size_t x = (size_t)i * 2 * K + k;
    X[x] = a; X[x + K] = b; Y[x] = b; Y[x + K] = a;
}
// TRSM: odwrotności bloków diagonalnych L (n×n, ld n, dolna) w FP64, rekurencyjnie. Indeksy „globalne” g = 0..nbl·nb
// (dopełnienie powyżej n: L = I). X (double): blok bi od bi·nb·nb, ld nb; trójkąt górny X nie jest nigdy czytany.
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define INV_L(i, k) ((i) < n && (k) < n ? (double)L[(size_t)(i) * n + (k)] : (double)((i) == (k)))
#define INV_X(X, i, k) X[(size_t)((i) / nb) * nb * nb + (size_t)((i) % nb) * nb + (k) % nb]
// Bloki bazowe 32×32: work-item = kolumna j bloku g, podstawianie w przód.
__kernel void trsm_inv0_f(__global const float *L, int n, int nb, __global double *X) {
    const int j = get_global_id(0), g0 = get_global_id(1) * 32;
    for (int i = j; i < 32; i++) {
        double s = i == j ? 1.0 : 0.0;
        for (int k = j; k < i; k++) s -= INV_L(g0 + i, g0 + k) * INV_X(X, g0 + k, g0 + j);
        INV_X(X, g0 + i, g0 + j) = s / INV_L(g0 + i, g0 + i);
    }
}
// Poziom h (pary bloków h×h w blok 2h): krok 1: Tm = L21·X11 (Tm na miejscu X21 w buforze Tm),
// krok 2: X21 = −X22·Tm. Siatka (h, h, liczba par).
__kernel void trsm_inv1_f(__global const float *L, int n, int nb, __global double *X, __global double *Tm, int h) {
    const int j = get_global_id(0), i = get_global_id(1), r0 = get_global_id(2) * 2 * h;
    double s = 0.0;
    for (int k = j; k < h; k++) s += INV_L(r0 + h + i, r0 + k) * INV_X(X, r0 + k, r0 + j);
    INV_X(Tm, r0 + h + i, r0 + j) = s;
}
__kernel void trsm_inv2_f(__global const float *L, int n, int nb, __global double *X, __global double *Tm, int h) {
    const int j = get_global_id(0), i = get_global_id(1), r0 = get_global_id(2) * 2 * h;
    double s = 0.0;
    for (int k = 0; k <= i; k++) s += INV_X(X, r0 + h + i, r0 + h + k) * INV_X(Tm, r0 + h + k, r0 + j);
    INV_X(X, r0 + h + i, r0 + j) = -s;
}
// Wynik: out (blok bi od bi·nb·nb, ld b = min(nb, n − bi·nb)), trójkąt górny = 0. Siatka (nb, nb, nbl).
__kernel void trsm_inv3_f(int n, int nb, __global const double *X, __global float *out) {
    const int j = get_global_id(0), i = get_global_id(1), bi = get_global_id(2), b = min(nb, n - bi * nb);
    if (i >= b || j >= b) return;
    out[(size_t)bi * nb * nb + (size_t)i * b + j] = j <= i ? (float)X[(size_t)bi * nb * nb + (size_t)i * nb + j] : 0.0f;
}
