// Warianty pakowania B (FP16) do paneli szerokości 16: Bp[(n/16)·Kp + k][n % 16]. Wzorzec: pack_b_pad (lib, element
// na work-item). Pomiar: bench/packing.c. Sygnatura wszystkich: (B, Bp, K, N, ldb, Kp); siatka w komentarzu.

// Wzorzec (kopia z hgemm.cl). Siatka (Np, Kp).
__kernel void pack_b_pad(__global const half *B, __global half *Bp, int K, int N, int ldb, int Kp) {
    const int n = get_global_id(0), k = get_global_id(1);
    Bp[((size_t)(n / 16) * Kp + k) * 16 + n % 16] = (n < N && k < K) ? B[(size_t)k * ldb + n] : (half)0;
}

// 2 / 4 / 8 kolejne elementy wiersza na work-item: subgrupa (16) pisze do 2 / 4 / 8 paneli. Siatka (Np/W, Kp).
#define PACK_BW(W, VT, VLOAD, VSTORE)                                                                        \
__kernel void pack_b_w##W(__global const half *B, __global half *Bp, int K, int N, int ldb, int Kp) {      \
    const int n0 = get_global_id(0) * W, k = get_global_id(1);                                             \
    VT v;                                                                                                    \
    if (k < K && n0 + W <= N) v = VLOAD(0, B + (size_t)k * ldb + n0);                                       \
    else for (int j = 0; j < W; j++) ((half *)&v)[j] = (k < K && n0 + j < N) ? B[(size_t)k * ldb + n0 + j] : (half)0; \
    VSTORE(v, 0, Bp + ((size_t)(n0 / 16) * Kp + k) * 16 + n0 % 16);                                        \
}
PACK_BW(2, half2, vload2, vstore2)
PACK_BW(4, half4, vload4, vstore4)
PACK_BW(8, half8, vload8, vstore8)

// Wiersz panelu (16 elementów, 32 B) na work-item, sąsiednie work-itemy po k: subgrupa pisze 512 B ciągiem do jednego
// panelu, czyta 16 wierszy B. Siatka (Kp, Np/16).
__kernel void pack_b_k16(__global const half *B, __global half *Bp, int K, int N, int ldb, int Kp) {
    const int k = get_global_id(0), n0 = get_global_id(1) * 16;
    half16 v;
    if (k < K && n0 + 16 <= N) v = vload16(0, B + (size_t)k * ldb + n0);
    else for (int j = 0; j < 16; j++) ((half *)&v)[j] = (k < K && n0 + j < N) ? B[(size_t)k * ldb + n0 + j] : (half)0;
    vstore16(v, 0, Bp + ((size_t)(n0 / 16) * Kp + k) * 16);
}

// Jak pack_b_w4, ale bloki subgrupowe: 16 lanes × 4 = 64 elementy wiersza = 4 panele; odczyt 128 B jednym blokiem.
// Siatka (Np/4, Kp), lokalna 16 (wymaga N % 64 == 0 i K pełne; inaczej wzorzec).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void pack_b_blk4(__global const half *B, __global half *Bp, int K, int N, int ldb, int Kp) {
    const int k = get_global_id(1), nb = get_group_id(0) * 64, l = get_sub_group_local_id();
    // blok: 64 halfy = 32 uinty; każdy lane dostaje 2 uinty (lane l: uinty l i l+16 -> elementy 2l,2l+1 i 32+2l,32+2l+1)
    __global const uint *src = (__global const uint *)(B + (size_t)k * ldb + nb);
    uint2 d = intel_sub_group_block_read2(src);
    // elementy 2l, 2l+1 -> panel (nb + 2l)/16, pozycja (2l)%16; 32+2l -> panel +2
    __global uint *dst0 = (__global uint *)(Bp + ((size_t)((nb + 2 * l) / 16) * Kp + k) * 16 + (2 * l) % 16);
    __global uint *dst1 = (__global uint *)(Bp + ((size_t)((nb + 32 + 2 * l) / 16) * Kp + k) * 16 + (2 * l) % 16);
    *dst0 = d.x; *dst1 = d.y;
}

// ---- Kernele biblioteki (kopie z hgemm.cl: układ BQ dla GEN9BLAS_H, A2) i ich warianty. Sygnatura (X, Xp, R1, R2, ld, Kp, long sX, int Rp).
#define PK_IDX_A2(m, k, Kp) ((((size_t)((m) / 16) * ((Kp) / 16) + (k) / 16) * 256 + ((k) % 16 / 8) * 128 + \
    (2 * ((m) % 16 / 4) + (((k) % 8) % 2 == 0)) * 16 + ((m) % 4) * 4 + ((k) % 8) / 2))
#define PK_IDX_BQ(n, k, Kp) (((((size_t)((n) / 32) * ((Kp) / 16) + (k) / 16) * 2 + ((n) % 32) / 16) * 16 + (k) % 16) * 16 + (n) % 16)
__kernel void pk_a2(__global const half *A, __global half *Ap, int M, int K, int lda, int Kp, long sA, int Mp) {
    const int k = get_global_id(0), m = get_global_id(1), b = get_global_id(2);
    Ap[(size_t)b * Mp * Kp + PK_IDX_A2(m, k, Kp)] = (m < M && k < K) ? A[b * sA + (size_t)m * lda + k] : (half)0; }
__kernel void pk_bq(__global const half *B, __global half *Bp, int K, int N, int ldb, int Kp, long sB, int Np) {
    const int n = get_global_id(0), k = get_global_id(1), b = get_global_id(2);
    Bp[(size_t)b * Np * Kp + PK_IDX_BQ(n, k, Kp)] = (n < N && k < K) ? B[b * sB + (size_t)k * ldb + n] : (half)0; }
// BQ, 2 elementy na work-item (siatka Np/2, Kp, count)
__kernel void pk_bq_w2(__global const half *B, __global half *Bp, int K, int N, int ldb, int Kp, long sB, int Np) {
    const int n = get_global_id(0) * 2, k = get_global_id(1), b = get_global_id(2);
    __global const half *s = B + b * sB + (size_t)k * ldb + n;
    half2 v;
    if (k < K && n + 2 <= N) v = vload2(0, s);
    else { v.x = (k < K && n < N) ? s[0] : (half)0; v.y = (k < K && n + 1 < N) ? s[1] : (half)0; }
    vstore2(v, 0, Bp + (size_t)b * Np * Kp + PK_IDX_BQ(n, k, Kp)); }
// A2: w panelu 16×16 elementy k i k+1 (ta sama m) leżą o 16 pozycji od siebie (para nieparzysta/parzysta), więc
// element na work-item, ale wiersz A czytany po 2: work-item (k parzyste, m) czyta half2 i pisze 2 elementy. Siatka (Kp/2, Mp).
__kernel void pk_a2_w2(__global const half *A, __global half *Ap, int M, int K, int lda, int Kp, long sA, int Mp) {
    const int k = get_global_id(0) * 2, m = get_global_id(1), b = get_global_id(2);
    __global const half *s = A + b * sA + (size_t)m * lda + k;
    half2 v;
    if (m < M && k + 2 <= K) v = vload2(0, s);
    else { v.x = (m < M && k < K) ? s[0] : (half)0; v.y = (m < M && k + 1 < K) ? s[1] : (half)0; }
    __global half *d = Ap + (size_t)b * Mp * Kp;
    d[PK_IDX_A2(m, k, Kp)] = v.x; d[PK_IDX_A2(m, k + 1, Kp)] = v.y; }
// A2, work-item = 4 wiersze (ćwiartka kafelka 16) × 8 kolumn k: czyta 4 × half8, pisze 32 elementy ciągiem
// (grupa g = 2q: k nieparzyste, g = 2q+1: k parzyste; w grupie (m%4)*4 + (k%8)/2). Sąsiednie work-itemy: ćwiartki q,
// potem połówki h kafelka, potem kolejne kafelki k -- subgrupa pisze 1 KB ciągiem. Siatka (Kp/8 * 4, Mp/16).
__kernel void pk_a2_q(__global const half *A, __global half *Ap, int M, int K, int lda, int Kp, long sA, int Mp) {
    const int i = get_global_id(0), q = i % 4, k0 = (i / 4) * 8, m0 = get_global_id(1) * 16 + q * 4, b = get_global_id(2);
    __global const half *s = A + b * sA;
    half16 nie, par;
    for (int r = 0; r < 4; r++) {
        const int m = m0 + r;
        half8 v;
        if (m < M && k0 + 8 <= K) v = vload8(0, s + (size_t)m * lda + k0);
        else for (int j = 0; j < 8; j++) ((half *)&v)[j] = (m < M && k0 + j < K) ? s[(size_t)m * lda + k0 + j] : (half)0;
        ((half *)&nie)[r * 4 + 0] = v.s1; ((half *)&nie)[r * 4 + 1] = v.s3; ((half *)&nie)[r * 4 + 2] = v.s5; ((half *)&nie)[r * 4 + 3] = v.s7;
        ((half *)&par)[r * 4 + 0] = v.s0; ((half *)&par)[r * 4 + 1] = v.s2; ((half *)&par)[r * 4 + 2] = v.s4; ((half *)&par)[r * 4 + 3] = v.s6;
    }
    __global half *d = Ap + (size_t)b * Mp * Kp + ((size_t)(m0 / 16) * (Kp / 16) + k0 / 16) * 256 + (k0 % 16 / 8) * 128 + 2 * q * 16;
    vstore16(nie, 0, d); vstore16(par, 0, d + 16);
}
