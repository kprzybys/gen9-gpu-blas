// HGEMM C = A·B (row-major, FP16, sumowanie w FP16) dla UHD 620. Wersja szablonowa w OpenCL C.
// Jeden wątek SIMD16 liczy kafelek C: 16 wierszy × 32 kolumny (2 grupy po 16 linii).
// A, B czytane jako obrazy 2D (CL_R, CL_HALF_FLOAT) przez media block read (intel_sub_group_block_read_us8):
//   blok 32 B × 8 wierszy = 16 linii × 8 wierszy halfów, jeden komunikat send.
// Krok k = 16: A: 2 bloki (wiersze m0..m0+7, m0+8..m0+15; linia j = kolumna k+j),
//              B: 4 bloki (wiersze k..k+7, k+8..k+15 × kolumny n0.., n0+16..; linia j = kolumna n).
// Mnożenia: acc[r][c] += A[m0+r][k+kk] (rozgłoszone z linii kk) * B[k+kk][n0+16c+lane], 512 mad(16):hf na krok.
// Rozmiary: M % 16 == 0, N % 32 == 0, K % 16 == 0.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define BC(v, i) as_half(sub_group_broadcast(v, i))

inline void tile_16x32(__read_only image2d_t A, __read_only image2d_t B, __global half *C,
                       int N, int K, int ldc, int m0, int n0) {

    half acc0[16], acc1[16];
    for (int r = 0; r < 16; r++) { acc0[r] = 0; acc1[r] = 0; }

    for (int k = 0; k < K; k += 16) {
        // A: lane j = A[m0+r][k+j]; a_lo.sr = wiersz m0+r, a_hi.sr = wiersz m0+8+r
        ushort8 a_lo = intel_sub_group_block_read_us8(A, (int2)(k * 2, m0));
        ushort8 a_hi = intel_sub_group_block_read_us8(A, (int2)(k * 2, m0 + 8));
        // B: lane j = B[k+r][n0+j] (b0*) i B[k+r][n0+16+j] (b1*)
        ushort8 b0l = intel_sub_group_block_read_us8(B, (int2)(n0 * 2, k));
        ushort8 b0h = intel_sub_group_block_read_us8(B, (int2)(n0 * 2, k + 8));
        ushort8 b1l = intel_sub_group_block_read_us8(B, (int2)(n0 * 2 + 32, k));
        ushort8 b1h = intel_sub_group_block_read_us8(B, (int2)(n0 * 2 + 32, k + 8));

        #define ROW(r, AV, AI) { \
            acc0[r] = fma(BC(AV, 0), as_half(b0l.s0), acc0[r]); acc1[r] = fma(BC(AV, 0), as_half(b1l.s0), acc1[r]); \
            acc0[r] = fma(BC(AV, 1), as_half(b0l.s1), acc0[r]); acc1[r] = fma(BC(AV, 1), as_half(b1l.s1), acc1[r]); \
            acc0[r] = fma(BC(AV, 2), as_half(b0l.s2), acc0[r]); acc1[r] = fma(BC(AV, 2), as_half(b1l.s2), acc1[r]); \
            acc0[r] = fma(BC(AV, 3), as_half(b0l.s3), acc0[r]); acc1[r] = fma(BC(AV, 3), as_half(b1l.s3), acc1[r]); \
            acc0[r] = fma(BC(AV, 4), as_half(b0l.s4), acc0[r]); acc1[r] = fma(BC(AV, 4), as_half(b1l.s4), acc1[r]); \
            acc0[r] = fma(BC(AV, 5), as_half(b0l.s5), acc0[r]); acc1[r] = fma(BC(AV, 5), as_half(b1l.s5), acc1[r]); \
            acc0[r] = fma(BC(AV, 6), as_half(b0l.s6), acc0[r]); acc1[r] = fma(BC(AV, 6), as_half(b1l.s6), acc1[r]); \
            acc0[r] = fma(BC(AV, 7), as_half(b0l.s7), acc0[r]); acc1[r] = fma(BC(AV, 7), as_half(b1l.s7), acc1[r]); \
            acc0[r] = fma(BC(AV, 8), as_half(b0h.s0), acc0[r]); acc1[r] = fma(BC(AV, 8), as_half(b1h.s0), acc1[r]); \
            acc0[r] = fma(BC(AV, 9), as_half(b0h.s1), acc0[r]); acc1[r] = fma(BC(AV, 9), as_half(b1h.s1), acc1[r]); \
            acc0[r] = fma(BC(AV,10), as_half(b0h.s2), acc0[r]); acc1[r] = fma(BC(AV,10), as_half(b1h.s2), acc1[r]); \
            acc0[r] = fma(BC(AV,11), as_half(b0h.s3), acc0[r]); acc1[r] = fma(BC(AV,11), as_half(b1h.s3), acc1[r]); \
            acc0[r] = fma(BC(AV,12), as_half(b0h.s4), acc0[r]); acc1[r] = fma(BC(AV,12), as_half(b1h.s4), acc1[r]); \
            acc0[r] = fma(BC(AV,13), as_half(b0h.s5), acc0[r]); acc1[r] = fma(BC(AV,13), as_half(b1h.s5), acc1[r]); \
            acc0[r] = fma(BC(AV,14), as_half(b0h.s6), acc0[r]); acc1[r] = fma(BC(AV,14), as_half(b1h.s6), acc1[r]); \
            acc0[r] = fma(BC(AV,15), as_half(b0h.s7), acc0[r]); acc1[r] = fma(BC(AV,15), as_half(b1h.s7), acc1[r]); }
        ROW(0, a_lo.s0, 0) ROW(1, a_lo.s1, 0) ROW(2, a_lo.s2, 0) ROW(3, a_lo.s3, 0)
        ROW(4, a_lo.s4, 0) ROW(5, a_lo.s5, 0) ROW(6, a_lo.s6, 0) ROW(7, a_lo.s7, 0)
        ROW(8, a_hi.s0, 0) ROW(9, a_hi.s1, 0) ROW(10, a_hi.s2, 0) ROW(11, a_hi.s3, 0)
        ROW(12, a_hi.s4, 0) ROW(13, a_hi.s5, 0) ROW(14, a_hi.s6, 0) ROW(15, a_hi.s7, 0)
        #undef ROW
    }
    // zapis: wiersz po wierszu, 2 × 32 B (OWord block write)
    for (int r = 0; r < 16; r++) {
        __global ushort *row = (__global ushort *)(C + (size_t)(m0 + r) * ldc + n0);
        intel_sub_group_block_write_us(row, as_ushort(acc0[r]));
        intel_sub_group_block_write_us(row + 16, as_ushort(acc1[r]));
    }
}

// Siatka 2D: (N/32 × 16, M/16), kolejność kafelków taka, jak sprzęt rozdziela grupy (wiersz po wierszu).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void hgemm_16x32(__read_only image2d_t A, __read_only image2d_t B, __global half *C,
                          int M, int ldc, int K) {
    tile_16x32(A, B, C, ldc, K, ldc, get_global_id(1) * 16, get_group_id(0) * 32);
}

// Siatka 1D: jeden kafelek na subgrupę, kolejność grupowana (jak w Tritonie): kolejne kafelki idą w dół
// pasa GM wierszy kafelków, potem w prawo. Jednocześnie aktywne wątki liczą zwarty fragment C,
// więc dzielą wiersze A i kolumny B w L3/LLC.
#ifndef GM
#define GM 16
#endif
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void hgemm_16x32g(__read_only image2d_t A, __read_only image2d_t B, __global half *C,
                           int M, int ldc, int K) {
    const int pid = get_group_id(0) * get_num_sub_groups() + get_sub_group_id();
    const int npm = M / 16, npn = get_image_width(B) / 32;
    const int grupa = pid / (GM * npn), m_pierwszy = grupa * GM;
    const int gsz = min(npm - m_pierwszy, GM);
    const int pm = m_pierwszy + (pid % (GM * npn)) % gsz;
    const int pn = (pid % (GM * npn)) / gsz;
    tile_16x32(A, B, C, ldc, K, ldc, pm * 16, pn * 32);
}

// ---------- Wariant z macierzami spakowanymi (dane kafelka ciągłe w pamięci, odczyt z bufora) ----------
// Bp: panele po 16 kolumn: Bp[n/16][k][n%16]. Blok B (k..k+7, 16 kolumn) = 8×16 halfów = 256 B ciągłe:
//     intel_sub_group_block_read_us8(Bp + ...) daje linii j wartości [i*16 + j], i = 0..7, czyli wiersze k+i.
// Ap: panele po 16 wierszy, bloki k po 16: Ap[m/16][k/16][m%16][k%16]. Blok A (8 wierszy, 16 k) = 256 B ciągłe:
//     linia j = kolumna k+j, element i = wiersz m0+8h+i.
// Odczyt z bufora to OWord block read (2 komunikaty po 128 B na blok 256 B).
inline void tile_16x32p(__global const ushort *Ap, __global const ushort *Bp, __global half *C,
                        int K, int ldc, int pm, int pn) {
    const int m0 = pm * 16, n0 = pn * 32;
    half acc0[16], acc1[16];
    for (int r = 0; r < 16; r++) { acc0[r] = 0; acc1[r] = 0; }
    __global const ushort *pa = Ap + (size_t)pm * (K / 16) * 256;
    __global const ushort *pb0 = Bp + (size_t)(2 * pn) * K * 16, *pb1 = pb0 + (size_t)K * 16;
    for (int k = 0; k < K; k += 16) {
        ushort8 a_lo = intel_sub_group_block_read_us8(pa);
        ushort8 a_hi = intel_sub_group_block_read_us8(pa + 128);
        ushort8 b0l = intel_sub_group_block_read_us8(pb0);
        ushort8 b0h = intel_sub_group_block_read_us8(pb0 + 128);
        ushort8 b1l = intel_sub_group_block_read_us8(pb1);
        ushort8 b1h = intel_sub_group_block_read_us8(pb1 + 128);
        pa += 256; pb0 += 256; pb1 += 256;
        #define ROW(r, AV) { \
            acc0[r] = fma(BC(AV, 0), as_half(b0l.s0), acc0[r]); acc1[r] = fma(BC(AV, 0), as_half(b1l.s0), acc1[r]); \
            acc0[r] = fma(BC(AV, 1), as_half(b0l.s1), acc0[r]); acc1[r] = fma(BC(AV, 1), as_half(b1l.s1), acc1[r]); \
            acc0[r] = fma(BC(AV, 2), as_half(b0l.s2), acc0[r]); acc1[r] = fma(BC(AV, 2), as_half(b1l.s2), acc1[r]); \
            acc0[r] = fma(BC(AV, 3), as_half(b0l.s3), acc0[r]); acc1[r] = fma(BC(AV, 3), as_half(b1l.s3), acc1[r]); \
            acc0[r] = fma(BC(AV, 4), as_half(b0l.s4), acc0[r]); acc1[r] = fma(BC(AV, 4), as_half(b1l.s4), acc1[r]); \
            acc0[r] = fma(BC(AV, 5), as_half(b0l.s5), acc0[r]); acc1[r] = fma(BC(AV, 5), as_half(b1l.s5), acc1[r]); \
            acc0[r] = fma(BC(AV, 6), as_half(b0l.s6), acc0[r]); acc1[r] = fma(BC(AV, 6), as_half(b1l.s6), acc1[r]); \
            acc0[r] = fma(BC(AV, 7), as_half(b0l.s7), acc0[r]); acc1[r] = fma(BC(AV, 7), as_half(b1l.s7), acc1[r]); \
            acc0[r] = fma(BC(AV, 8), as_half(b0h.s0), acc0[r]); acc1[r] = fma(BC(AV, 8), as_half(b1h.s0), acc1[r]); \
            acc0[r] = fma(BC(AV, 9), as_half(b0h.s1), acc0[r]); acc1[r] = fma(BC(AV, 9), as_half(b1h.s1), acc1[r]); \
            acc0[r] = fma(BC(AV,10), as_half(b0h.s2), acc0[r]); acc1[r] = fma(BC(AV,10), as_half(b1h.s2), acc1[r]); \
            acc0[r] = fma(BC(AV,11), as_half(b0h.s3), acc0[r]); acc1[r] = fma(BC(AV,11), as_half(b1h.s3), acc1[r]); \
            acc0[r] = fma(BC(AV,12), as_half(b0h.s4), acc0[r]); acc1[r] = fma(BC(AV,12), as_half(b1h.s4), acc1[r]); \
            acc0[r] = fma(BC(AV,13), as_half(b0h.s5), acc0[r]); acc1[r] = fma(BC(AV,13), as_half(b1h.s5), acc1[r]); \
            acc0[r] = fma(BC(AV,14), as_half(b0h.s6), acc0[r]); acc1[r] = fma(BC(AV,14), as_half(b1h.s6), acc1[r]); \
            acc0[r] = fma(BC(AV,15), as_half(b0h.s7), acc0[r]); acc1[r] = fma(BC(AV,15), as_half(b1h.s7), acc1[r]); }
        ROW(0, a_lo.s0) ROW(1, a_lo.s1) ROW(2, a_lo.s2) ROW(3, a_lo.s3)
        ROW(4, a_lo.s4) ROW(5, a_lo.s5) ROW(6, a_lo.s6) ROW(7, a_lo.s7)
        ROW(8, a_hi.s0) ROW(9, a_hi.s1) ROW(10, a_hi.s2) ROW(11, a_hi.s3)
        ROW(12, a_hi.s4) ROW(13, a_hi.s5) ROW(14, a_hi.s6) ROW(15, a_hi.s7)
        #undef ROW
    }
    for (int r = 0; r < 16; r++) {
        __global ushort *row = (__global ushort *)(C + (size_t)(m0 + r) * ldc + n0);
        intel_sub_group_block_write_us(row, as_ushort(acc0[r]));
        intel_sub_group_block_write_us(row + 16, as_ushort(acc1[r]));
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void hgemm_16x32p(__global const ushort *Ap, __global const ushort *Bp, __global half *C,
                           int M, int ldc, int K) {
    const int pid = get_group_id(0) * get_num_sub_groups() + get_sub_group_id();
    const int npm = M / 16, npn = ldc / 32;
    const int grupa = pid / (GM * npn), m_pierwszy = grupa * GM;
    const int gsz = min(npm - m_pierwszy, GM);
    tile_16x32p(Ap, Bp, C, K, ldc, m_pierwszy + (pid % (GM * npn)) % gsz, (pid % (GM * npn)) / gsz);
}

// Pakowanie (row-major, ld) -> Ap / Bp. Jeden work-item = jeden element.
__kernel void pack_a(__global const half *A, __global half *Ap, int K, int lda) {
    const int k = get_global_id(0), m = get_global_id(1);
    Ap[(((size_t)(m / 16) * (K / 16) + k / 16) * 16 + m % 16) * 16 + k % 16] = A[(size_t)m * lda + k];
}
__kernel void pack_b(__global const half *B, __global half *Bp, int K, int ldb) {
    const int n = get_global_id(0), k = get_global_id(1);
    Bp[((size_t)(n / 16) * K + k) * 16 + n % 16] = B[(size_t)k * ldb + n];
}

// Układ A2 dla ręcznej pętli w asemblerze (bez konfliktów banków GRF):
// blok k po 16, w nim 2 połówki (kk 0..7, 8..15) po 256 B = 8 rejestrów GRF.
// GRF g połówki h: wiersze 4*(g/2)..+3; g nieparzysty -> parzyste kl (0,2,4,6), g parzysty -> nieparzyste kl.
// Element e = (r%4)*4 + kl/2. Gdy rejestry B (wiersz kl) mają parzystość kl, rejestr A ma parzystość przeciwną.
__kernel void pack_a2(__global const half *A, __global half *Ap, int K, int lda) {
    const int k = get_global_id(0), m = get_global_id(1);
    const int r = m % 16, kk = k % 16, h = kk / 8, kl = kk % 8;
    const int g = 2 * (r / 4) + ((kl % 2 == 0) ? 1 : 0);
    const size_t blok = (size_t)(m / 16) * (K / 16) + k / 16;
    Ap[blok * 256 + h * 128 + g * 16 + (r % 4) * 4 + kl / 2] = A[(size_t)m * lda + k];
}

// ---------- Warianty dokładniejsze (dane FP16, wynik FP16, suma pośrednia w FP32) ----------
// hgemm_16x32kc: sumowanie FP16 w porcjach po KC (16 lub 32) iloczynów, porcje dodawane do akumulatora FP32.
//                Arytmetyka głównie w tempie FP16; błąd ogranicza się do błędu krótkiej sumy FP16.
// hgemm_16x32mx: każdy iloczyn FP16·FP16 dodawany do FP32 (tempo FP32, dokładność ~FP32).
// Dane spakowane jak w hgemm_16x32p (PACK=1).
#ifndef KC
#define KC 32
#endif
#define ROWH(r, AV, ACC0, ACC1) { \
    for (int q = 0; q < 8; q++) { \
        ACC0[r] = fma(BC(AV, q), as_half(sub8(b0l, q)), ACC0[r]); ACC1[r] = fma(BC(AV, q), as_half(sub8(b1l, q)), ACC1[r]); } \
    for (int q = 0; q < 8; q++) { \
        ACC0[r] = fma(BC(AV, q + 8), as_half(sub8(b0h, q)), ACC0[r]); ACC1[r] = fma(BC(AV, q + 8), as_half(sub8(b1h, q)), ACC1[r]); } }
inline ushort sub8(ushort8 v, int i) {
    switch (i) { case 0: return v.s0; case 1: return v.s1; case 2: return v.s2; case 3: return v.s3;
                 case 4: return v.s4; case 5: return v.s5; case 6: return v.s6; default: return v.s7; }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void hgemm_16x32kc(__global const ushort *Ap, __global const ushort *Bp, __global half *C,
                            int M, int ldc, int K) {
    const int pid = get_group_id(0) * get_num_sub_groups() + get_sub_group_id();
    const int npm = M / 16, npn = ldc / 32;
    const int grupa = pid / (GM * npn), m_pierwszy = grupa * GM;
    const int gsz = min(npm - m_pierwszy, GM);
    const int pm = m_pierwszy + (pid % (GM * npn)) % gsz, pn = (pid % (GM * npn)) / gsz;
    const int m0 = pm * 16, n0 = pn * 32;
    float f0[16], f1[16]; half h0[16], h1[16];
    for (int r = 0; r < 16; r++) { f0[r] = 0; f1[r] = 0; h0[r] = 0; h1[r] = 0; }
    __global const ushort *pa = Ap + (size_t)pm * (K / 16) * 256;
    __global const ushort *pb0 = Bp + (size_t)(2 * pn) * K * 16, *pb1 = pb0 + (size_t)K * 16;
    for (int k = 0; k < K; k += 16) {
        ushort8 a_lo = intel_sub_group_block_read_us8(pa), a_hi = intel_sub_group_block_read_us8(pa + 128);
        ushort8 b0l = intel_sub_group_block_read_us8(pb0), b0h = intel_sub_group_block_read_us8(pb0 + 128);
        ushort8 b1l = intel_sub_group_block_read_us8(pb1), b1h = intel_sub_group_block_read_us8(pb1 + 128);
        pa += 256; pb0 += 256; pb1 += 256;
        __attribute__((opencl_unroll_hint))
        for (int r = 0; r < 8; r++) ROWH(r, sub8(a_lo, r), h0, h1)
        __attribute__((opencl_unroll_hint))
        for (int r = 0; r < 8; r++) ROWH(r + 8, sub8(a_hi, r), h0, h1)
        if ((k + 16) % KC == 0) {
            for (int r = 0; r < 16; r++) { f0[r] += (float)h0[r]; f1[r] += (float)h1[r]; h0[r] = 0; h1[r] = 0; }
        }
    }
    for (int r = 0; r < 16; r++) {
        __global ushort *row = (__global ushort *)(C + (size_t)(m0 + r) * ldc + n0);
        intel_sub_group_block_write_us(row, as_ushort((half)(f0[r] + (float)h0[r])));
        intel_sub_group_block_write_us(row + 16, as_ushort((half)(f1[r] + (float)h1[r])));
    }
}

#define ROWM(r, AV) { \
    for (int q = 0; q < 8; q++) { float av = (float)BC(AV, q); \
        f0[r] = fma(av, (float)as_half(sub8(b0l, q)), f0[r]); f1[r] = fma(av, (float)as_half(sub8(b1l, q)), f1[r]); } \
    for (int q = 0; q < 8; q++) { float av = (float)BC(AV, q + 8); \
        f0[r] = fma(av, (float)as_half(sub8(b0h, q)), f0[r]); f1[r] = fma(av, (float)as_half(sub8(b1h, q)), f1[r]); } }
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void hgemm_16x32mx(__global const ushort *Ap, __global const ushort *Bp, __global half *C,
                            int M, int ldc, int K) {
    const int pid = get_group_id(0) * get_num_sub_groups() + get_sub_group_id();
    const int npm = M / 16, npn = ldc / 32;
    const int grupa = pid / (GM * npn), m_pierwszy = grupa * GM;
    const int gsz = min(npm - m_pierwszy, GM);
    const int pm = m_pierwszy + (pid % (GM * npn)) % gsz, pn = (pid % (GM * npn)) / gsz;
    const int m0 = pm * 16, n0 = pn * 32;
    float f0[16], f1[16];
    for (int r = 0; r < 16; r++) { f0[r] = 0; f1[r] = 0; }
    __global const ushort *pa = Ap + (size_t)pm * (K / 16) * 256;
    __global const ushort *pb0 = Bp + (size_t)(2 * pn) * K * 16, *pb1 = pb0 + (size_t)K * 16;
    for (int k = 0; k < K; k += 16) {
        ushort8 a_lo = intel_sub_group_block_read_us8(pa), a_hi = intel_sub_group_block_read_us8(pa + 128);
        ushort8 b0l = intel_sub_group_block_read_us8(pb0), b0h = intel_sub_group_block_read_us8(pb0 + 128);
        ushort8 b1l = intel_sub_group_block_read_us8(pb1), b1h = intel_sub_group_block_read_us8(pb1 + 128);
        pa += 256; pb0 += 256; pb1 += 256;
        __attribute__((opencl_unroll_hint))
        for (int r = 0; r < 8; r++) ROWM(r, sub8(a_lo, r))
        __attribute__((opencl_unroll_hint))
        for (int r = 0; r < 8; r++) ROWM(r + 8, sub8(a_hi, r))
    }
    for (int r = 0; r < 16; r++) {
        __global ushort *row = (__global ushort *)(C + (size_t)(m0 + r) * ldc + n0);
        intel_sub_group_block_write_us(row, as_ushort((half)f0[r]));
        intel_sub_group_block_write_us(row + 16, as_ushort((half)f1[r]));
    }
}

// ---------- Szybsze pakowanie: jeden work-item przenosi 16 kolejnych elementów wiersza (vload16) ----------
// Wymiary wejścia dowolne: brakujące wiersze/kolumny do wielokrotności kafelka (M%16, N%32, K%16) wypełniane zerami.
// Siatka: (ceil(K/16) lub ceil(N/16), Mp lub Kp) — patrz program hosta.
inline half16 wiersz16(__global const half *X, int ld, int rows, int cols, int r, int c0) {
    if (r >= rows) return (half16)0;
    if (c0 + 16 <= cols && ((ld | c0) & 1) == 0) return vload16(0, X + (size_t)r * ld + c0);
    half16 v = 0; __attribute__((opencl_unroll_hint))
    for (int j = 0; j < 16; j++) if (c0 + j < cols) ((half *)&v)[j] = X[(size_t)r * ld + c0 + j];
    return v;
}
// Bp[n/16][k][n%16], Kp = K zaokrąglone w górę do 16. gid0 = blok 16 kolumn, gid1 = wiersz k (0..Kp-1).
__kernel void pack_b16(__global const half *B, __global half *Bp, int K, int N, int ldb, int Kp) {
    const int nb = get_global_id(0), k = get_global_id(1);
    vstore16(wiersz16(B, ldb, K, N, k, nb * 16), 0, Bp + ((size_t)nb * Kp + k) * 16);
}
// A2 (patrz pack_a2), Kp jak wyżej. gid0 = blok 16 kolumn k, gid1 = wiersz m (0..Mp-1).
__kernel void pack_a2_16(__global const half *A, __global half *Ap, int M, int K, int lda, int Kp) {
    const int kb = get_global_id(0), m = get_global_id(1), r = m % 16;
    half16 v = wiersz16(A, lda, M, K, m, kb * 16);
    __global half *blok = Ap + ((size_t)(m / 16) * (Kp / 16) + kb) * 256;
    // połówka h, kl parzyste -> GRF nieparzysty g = 2*(r/4)+1, kl nieparzyste -> g = 2*(r/4); element (r%4)*4 + kl/2
    __global half *p0 = blok + (2 * (r / 4) + 1) * 16 + (r % 4) * 4, *p1 = blok + (2 * (r / 4)) * 16 + (r % 4) * 4;
    vstore4(v.s02468ace.lo, 0, p0);         vstore4(v.s13579bdf.lo, 0, p1);
    vstore4(v.s02468ace.hi, 0, p0 + 128);   vstore4(v.s13579bdf.hi, 0, p1 + 128);
}
