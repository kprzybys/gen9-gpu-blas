// Variants of packing B (FP16) into panels of width 16: Bp[(n/16)·Kp + k][n % 16]. Reference: pack_b_pad (lib, one element
// per work-item). Measurement: bench/packing.c. Signature of all: (B, Bp, K, N, ldb, Kp); grid in the comment.

// Reference (copy from hgemm.cl). Grid (Np, Kp).
__kernel void pack_b_pad(__global const half *B, __global half *Bp, int K, int N, int ldb, int Kp) {
    const int n = get_global_id(0), k = get_global_id(1);
    Bp[((size_t)(n / 16) * Kp + k) * 16 + n % 16] = (n < N && k < K) ? B[(size_t)k * ldb + n] : (half)0;
}

// 2 / 4 / 8 consecutive row elements per work-item: a subgroup (16) writes to 2 / 4 / 8 panels. Grid (Np/W, Kp).
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

// Panel row (16 elements, 32 B) per work-item, neighbouring work-items along k: a subgroup writes 512 B contiguously to one
// panel, reads 16 rows of B. Grid (Kp, Np/16).
__kernel void pack_b_k16(__global const half *B, __global half *Bp, int K, int N, int ldb, int Kp) {
    const int k = get_global_id(0), n0 = get_global_id(1) * 16;
    half16 v;
    if (k < K && n0 + 16 <= N) v = vload16(0, B + (size_t)k * ldb + n0);
    else for (int j = 0; j < 16; j++) ((half *)&v)[j] = (k < K && n0 + j < N) ? B[(size_t)k * ldb + n0 + j] : (half)0;
    vstore16(v, 0, Bp + ((size_t)(n0 / 16) * Kp + k) * 16);
}

// Like pack_b_w4, but subgroup blocks: 16 lanes × 4 = 64 row elements = 4 panels; 128 B read as one block.
// Grid (Np/4, Kp), local 16 (requires N % 64 == 0 and full K; otherwise the reference).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void pack_b_blk4(__global const half *B, __global half *Bp, int K, int N, int ldb, int Kp) {
    const int k = get_global_id(1), nb = get_group_id(0) * 64, l = get_sub_group_local_id();
    // block: 64 halves = 32 uints; each lane gets 2 uints (lane l: uints l and l+16 -> elements 2l,2l+1 and 32+2l,32+2l+1)
    __global const uint *src = (__global const uint *)(B + (size_t)k * ldb + nb);
    uint2 d = intel_sub_group_block_read2(src);
    // elements 2l, 2l+1 -> panel (nb + 2l)/16, position (2l)%16; 32+2l -> panel +2
    __global uint *dst0 = (__global uint *)(Bp + ((size_t)((nb + 2 * l) / 16) * Kp + k) * 16 + (2 * l) % 16);
    __global uint *dst1 = (__global uint *)(Bp + ((size_t)((nb + 32 + 2 * l) / 16) * Kp + k) * 16 + (2 * l) % 16);
    *dst0 = d.x; *dst1 = d.y;
}

// ---- Library kernels (copies from hgemm.cl: BQ layout for GEN9BLAS_H, A2) and their variants. Signature (X, Xp, R1, R2, ld, Kp, long sX, int Rp).
#define PK_IDX_A2(m, k, Kp) ((((size_t)((m) / 16) * ((Kp) / 16) + (k) / 16) * 256 + ((k) % 16 / 8) * 128 + \
    (2 * ((m) % 16 / 4) + (((k) % 8) % 2 == 0)) * 16 + ((m) % 4) * 4 + ((k) % 8) / 2))
#define PK_IDX_BQ(n, k, Kp) (((((size_t)((n) / 32) * ((Kp) / 16) + (k) / 16) * 2 + ((n) % 32) / 16) * 16 + (k) % 16) * 16 + (n) % 16)
__kernel void pk_a2(__global const half *A, __global half *Ap, int M, int K, int lda, int Kp, long sA, int Mp) {
    const int k = get_global_id(0), m = get_global_id(1), b = get_global_id(2);
    Ap[(size_t)b * Mp * Kp + PK_IDX_A2(m, k, Kp)] = (m < M && k < K) ? A[b * sA + (size_t)m * lda + k] : (half)0; }
__kernel void pk_bq(__global const half *B, __global half *Bp, int K, int N, int ldb, int Kp, long sB, int Np) {
    const int n = get_global_id(0), k = get_global_id(1), b = get_global_id(2);
    Bp[(size_t)b * Np * Kp + PK_IDX_BQ(n, k, Kp)] = (n < N && k < K) ? B[b * sB + (size_t)k * ldb + n] : (half)0; }
// BQ, 2 elements per work-item (grid Np/2, Kp, count)
__kernel void pk_bq_w2(__global const half *B, __global half *Bp, int K, int N, int ldb, int Kp, long sB, int Np) {
    const int n = get_global_id(0) * 2, k = get_global_id(1), b = get_global_id(2);
    __global const half *s = B + b * sB + (size_t)k * ldb + n;
    half2 v;
    if (k < K && n + 2 <= N) v = vload2(0, s);
    else { v.x = (k < K && n < N) ? s[0] : (half)0; v.y = (k < K && n + 1 < N) ? s[1] : (half)0; }
    vstore2(v, 0, Bp + (size_t)b * Np * Kp + PK_IDX_BQ(n, k, Kp)); }
// A2: in a 16×16 panel elements k and k+1 (same m) lie 16 positions apart (odd/even pair), so
// one element per work-item, but the A row is read 2 at a time: a work-item (even k, m) reads half2 and writes 2 elements. Grid (Kp/2, Mp).
__kernel void pk_a2_w2(__global const half *A, __global half *Ap, int M, int K, int lda, int Kp, long sA, int Mp) {
    const int k = get_global_id(0) * 2, m = get_global_id(1), b = get_global_id(2);
    __global const half *s = A + b * sA + (size_t)m * lda + k;
    half2 v;
    if (m < M && k + 2 <= K) v = vload2(0, s);
    else { v.x = (m < M && k < K) ? s[0] : (half)0; v.y = (m < M && k + 1 < K) ? s[1] : (half)0; }
    __global half *d = Ap + (size_t)b * Mp * Kp;
    d[PK_IDX_A2(m, k, Kp)] = v.x; d[PK_IDX_A2(m, k + 1, Kp)] = v.y; }
// A2, work-item = 4 rows (quarter of a 16 tile) × 8 columns k: reads 4 × half8, writes 32 elements contiguously
// (group g = 2q: odd k, g = 2q+1: even k; within the group (m%4)*4 + (k%8)/2). Neighbouring work-items: quarters q,
// then halves h of the tile, then consecutive k tiles -- a subgroup writes 1 KB contiguously. Grid (Kp/8 * 4, Mp/16).
__kernel void pk_a2_q(__global const half *A, __global half *Ap, int M, int K, int lda, int Kp, long sA, int Mp) {
    const int i = get_global_id(0), q = i % 4, k0 = (i / 4) * 8, m0 = get_global_id(1) * 16 + q * 4, b = get_global_id(2);
    __global const half *s = A + b * sA;
    half16 odd, even;
    for (int r = 0; r < 4; r++) {
        const int m = m0 + r;
        half8 v;
        if (m < M && k0 + 8 <= K) v = vload8(0, s + (size_t)m * lda + k0);
        else for (int j = 0; j < 8; j++) ((half *)&v)[j] = (m < M && k0 + j < K) ? s[(size_t)m * lda + k0 + j] : (half)0;
        ((half *)&odd)[r * 4 + 0] = v.s1; ((half *)&odd)[r * 4 + 1] = v.s3; ((half *)&odd)[r * 4 + 2] = v.s5; ((half *)&odd)[r * 4 + 3] = v.s7;
        ((half *)&even)[r * 4 + 0] = v.s0; ((half *)&even)[r * 4 + 1] = v.s2; ((half *)&even)[r * 4 + 2] = v.s4; ((half *)&even)[r * 4 + 3] = v.s6;
    }
    __global half *d = Ap + (size_t)b * Mp * Kp + ((size_t)(m0 / 16) * (Kp / 16) + k0 / 16) * 256 + (k0 % 16 / 8) * 128 + 2 * q * 16;
    vstore16(odd, 0, d); vstore16(even, 0, d + 16);
}
