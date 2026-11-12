// HGEMM C = A·B (row-major, FP16, FP16 accumulation) for UHD 620. Template version in OpenCL C.
// One SIMD16 thread computes a C tile: 16 rows × 32 columns (2 groups of 16 lanes).
// A, B are read as 2D images (CL_R, CL_HALF_FLOAT) via media block read (intel_sub_group_block_read_us8):
//   block of 32 B × 8 rows = 16 lanes × 8 rows of halves, one send message.
// Step k = 16: A: 2 blocks (rows m0..m0+7, m0+8..m0+15; lane j = column k+j),
//              B: 4 blocks (rows k..k+7, k+8..k+15 × columns n0.., n0+16..; lane j = column n).
// Multiplications: acc[r][c] += A[m0+r][k+kk] (broadcast from lane kk) * B[k+kk][n0+16c+lane], 512 mad(16):hf per step.
// Sizes: M % 16 == 0, N % 32 == 0, K % 16 == 0.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define BC(v, i) as_half(sub_group_broadcast(v, i))

inline void tile_16x32(__read_only image2d_t A, __read_only image2d_t B, __global half *C,
                       int N, int K, int ldc, int m0, int n0) {

    half acc0[16], acc1[16];
    for (int r = 0; r < 16; r++) { acc0[r] = 0; acc1[r] = 0; }

    for (int k = 0; k < K; k += 16) {
        // A: lane j = A[m0+r][k+j]; a_lo.sr = row m0+r, a_hi.sr = row m0+8+r
        ushort8 a_lo = intel_sub_group_block_read_us8(A, (int2)(k * 2, m0));
        ushort8 a_hi = intel_sub_group_block_read_us8(A, (int2)(k * 2, m0 + 8));
        // B: lane j = B[k+r][n0+j] (b0*) and B[k+r][n0+16+j] (b1*)
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
    // write: row by row, 2 × 32 B (OWord block write)
    for (int r = 0; r < 16; r++) {
        __global ushort *row = (__global ushort *)(C + (size_t)(m0 + r) * ldc + n0);
        intel_sub_group_block_write_us(row, as_ushort(acc0[r]));
        intel_sub_group_block_write_us(row + 16, as_ushort(acc1[r]));
    }
}

// 2D grid: (N/32 × 16, M/16), tile order matches how the hardware dispatches groups (row by row).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void hgemm_16x32(__read_only image2d_t A, __read_only image2d_t B, __global half *C,
                          int M, int ldc, int K) {
    tile_16x32(A, B, C, ldc, K, ldc, get_global_id(1) * 16, get_group_id(0) * 32);
}

// 1D grid: one tile per subgroup, grouped order (as in Triton): consecutive tiles go down a band
// of GM tile rows, then to the right. Simultaneously active threads compute a compact piece of C,
// so they share rows of A and columns of B in L3/LLC.
#ifndef GM
#define GM 16
#endif
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void hgemm_16x32g(__read_only image2d_t A, __read_only image2d_t B, __global half *C,
                           int M, int ldc, int K) {
    const int pid = get_group_id(0) * get_num_sub_groups() + get_sub_group_id();
    const int npm = M / 16, npn = get_image_width(B) / 32;
    const int group = pid / (GM * npn), m_first = group * GM;
    const int gsz = min(npm - m_first, GM);
    const int pm = m_first + (pid % (GM * npn)) % gsz;
    const int pn = (pid % (GM * npn)) / gsz;
    tile_16x32(A, B, C, ldc, K, ldc, pm * 16, pn * 32);
}

// ---------- Packed-matrix variant (tile data contiguous in memory, read from a buffer) ----------
// Bp: panels of 16 columns: Bp[n/16][k][n%16]. B block (k..k+7, 16 columns) = 8×16 halves = 256 B contiguous:
//     intel_sub_group_block_read_us8(Bp + ...) gives lane j the values [i*16 + j], i = 0..7, i.e. rows k+i.
// Ap: panels of 16 rows, blocks of 16 k: Ap[m/16][k/16][m%16][k%16]. A block (8 rows, 16 k) = 256 B contiguous:
//     lane j = column k+j, element i = row m0+8h+i.
// A buffer read is an OWord block read (2 messages of 128 B per 256 B block).
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
    const int group = pid / (GM * npn), m_first = group * GM;
    const int gsz = min(npm - m_first, GM);
    tile_16x32p(Ap, Bp, C, K, ldc, m_first + (pid % (GM * npn)) % gsz, (pid % (GM * npn)) / gsz);
}

// Packing (row-major, ld) -> Ap / Bp. One work-item = one element.
__kernel void pack_a(__global const half *A, __global half *Ap, int K, int lda) {
    const int k = get_global_id(0), m = get_global_id(1);
    Ap[(((size_t)(m / 16) * (K / 16) + k / 16) * 16 + m % 16) * 16 + k % 16] = A[(size_t)m * lda + k];
}
__kernel void pack_b(__global const half *B, __global half *Bp, int K, int ldb) {
    const int n = get_global_id(0), k = get_global_id(1);
    Bp[((size_t)(n / 16) * K + k) * 16 + n % 16] = B[(size_t)k * ldb + n];
}

// A2 layout for a hand-written asm loop (no GRF bank conflicts):
// block of 16 k, inside it 2 halves (kk 0..7, 8..15) of 256 B = 8 GRF registers.
// GRF g of half h: rows 4*(g/2)..+3; odd g -> even kl (0,2,4,6), even g -> odd kl.
// Element e = (r%4)*4 + kl/2. When the B registers (row kl) have kl parity, the A register has the opposite parity.
__kernel void pack_a2(__global const half *A, __global half *Ap, int K, int lda) {
    const int k = get_global_id(0), m = get_global_id(1);
    const int r = m % 16, kk = k % 16, h = kk / 8, kl = kk % 8;
    const int g = 2 * (r / 4) + ((kl % 2 == 0) ? 1 : 0);
    const size_t blk = (size_t)(m / 16) * (K / 16) + k / 16;
    Ap[blk * 256 + h * 128 + g * 16 + (r % 4) * 4 + kl / 2] = A[(size_t)m * lda + k];
}

// ---------- More accurate variants (FP16 data, FP16 result, FP32 partial sums) ----------
// hgemm_16x32kc: FP16 accumulation in chunks of KC (16 or 32) products, chunks added to an FP32 accumulator.
//                Arithmetic mostly at FP16 rate; the error is limited to that of a short FP16 sum.
// hgemm_16x32mx: every FP16·FP16 product added to FP32 (FP32 rate, ~FP32 accuracy).
// Data packed as in hgemm_16x32p (PACK=1).
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
    const int group = pid / (GM * npn), m_first = group * GM;
    const int gsz = min(npm - m_first, GM);
    const int pm = m_first + (pid % (GM * npn)) % gsz, pn = (pid % (GM * npn)) / gsz;
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
    const int group = pid / (GM * npn), m_first = group * GM;
    const int gsz = min(npm - m_first, GM);
    const int pm = m_first + (pid % (GM * npn)) % gsz, pn = (pid % (GM * npn)) / gsz;
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

// ---------- Faster packing: one work-item moves 16 consecutive elements of a row (vload16) ----------
// Input dimensions arbitrary: rows/columns missing up to a multiple of the tile (M%16, N%32, K%16) are zero-filled.
// Grid: (ceil(K/16) or ceil(N/16), Mp or Kp) -- see the host program.
inline half16 row16(__global const half *X, int ld, int rows, int cols, int r, int c0) {
    if (r >= rows) return (half16)0;
    if (c0 + 16 <= cols && ((ld | c0) & 1) == 0) return vload16(0, X + (size_t)r * ld + c0);
    half16 v = 0; __attribute__((opencl_unroll_hint))
    for (int j = 0; j < 16; j++) if (c0 + j < cols) ((half *)&v)[j] = X[(size_t)r * ld + c0 + j];
    return v;
}
// Bp[n/16][k][n%16], Kp = K rounded up to 16. gid0 = block of 16 columns, gid1 = row k (0..Kp-1).
__kernel void pack_b16(__global const half *B, __global half *Bp, int K, int N, int ldb, int Kp) {
    const int nb = get_global_id(0), k = get_global_id(1);
    vstore16(row16(B, ldb, K, N, k, nb * 16), 0, Bp + ((size_t)nb * Kp + k) * 16);
}
// A2 (see pack_a2), Kp as above. gid0 = block of 16 columns k, gid1 = row m (0..Mp-1).
__kernel void pack_a2_16(__global const half *A, __global half *Ap, int M, int K, int lda, int Kp) {
    const int kb = get_global_id(0), m = get_global_id(1), r = m % 16;
    half16 v = row16(A, lda, M, K, m, kb * 16);
    __global half *blk = Ap + ((size_t)(m / 16) * (Kp / 16) + kb) * 256;
    // half h, even kl -> odd GRF g = 2*(r/4)+1, odd kl -> g = 2*(r/4); element (r%4)*4 + kl/2
    __global half *p0 = blk + (2 * (r / 4) + 1) * 16 + (r % 4) * 4, *p1 = blk + (2 * (r / 4)) * 16 + (r % 4) * 4;
    vstore4(v.s02468ace.lo, 0, p0);         vstore4(v.s13579bdf.lo, 0, p1);
    vstore4(v.s02468ace.hi, 0, p0 + 128);   vstore4(v.s13579bdf.hi, 0, p1 + 128);
}

// pack_a2_g: A2 layout, one work-item = one 32-byte output row (GRF register) written contiguously (vstore16).
// Output as an array of rows of 16 halves: o -> blk = o/16 (pm, kb), h = (o%16)/8, g = o%8.
// GRF g: rows r = 4*(g/2) + ri (ri 0..3), kl = 2*j + (g odd ? 0 : 1) (j 0..3), element ri*4 + j.
// Edges (m >= M, k >= K) zero-filled. Grid: Mp*Kp/16 work-items.
__kernel void pack_a2_g(__global const half *A, __global half *Ap, int M, int K, int lda, int Kp) {
    const int o = get_global_id(0), blk = o / 16, h = (o % 16) / 8, g = o % 8;
    const int pm = blk / (Kp / 16), kb = blk % (Kp / 16);
    const int k0 = kb * 16 + 8 * h, odd = (g & 1) ? 0 : 1;
    half16 w;
    for (int ri = 0; ri < 4; ri++) {
        const int m = pm * 16 + 4 * (g / 2) + ri;
        half8 v = 0;
        if (m < M) {
            if (k0 + 8 <= K && ((lda | k0) & 7) == 0) v = vload8(0, A + (size_t)m * lda + k0);
            else for (int j = 0; j < 8; j++) if (k0 + j < K) ((half *)&v)[j] = A[(size_t)m * lda + k0 + j];
        }
        half4 s = odd ? v.s1357 : v.s0246;
        ((half *)&w)[ri * 4 + 0] = s.x; ((half *)&w)[ri * 4 + 1] = s.y;
        ((half *)&w)[ri * 4 + 2] = s.z; ((half *)&w)[ri * 4 + 3] = s.w;
    }
    vstore16(w, o, Ap);
}

// Packing with edge handling, one element per work-item (fastest of those tested: neighbouring work-items read
// a contiguous row and write to at most 2 panels; the "16 elements per work-item" variants were slower because of the TLB).
// pack_a2_pad: grid (Kp, Mp); pack_b_pad: grid (Np, Kp). Out of range (m >= M, k >= K, n >= N) gives zeros.
__kernel void pack_a2_pad(__global const half *A, __global half *Ap, int M, int K, int lda, int Kp) {
    const int k = get_global_id(0), m = get_global_id(1);
    const int r = m % 16, kk = k % 16, h = kk / 8, kl = kk % 8;
    const int g = 2 * (r / 4) + ((kl % 2 == 0) ? 1 : 0);
    Ap[((size_t)(m / 16) * (Kp / 16) + k / 16) * 256 + h * 128 + g * 16 + (r % 4) * 4 + kl / 2] =
        (m < M && k < K) ? A[(size_t)m * lda + k] : (half)0;
}
__kernel void pack_b_pad(__global const half *B, __global half *Bp, int K, int N, int ldb, int Kp) {
    const int n = get_global_id(0), k = get_global_id(1);
    Bp[((size_t)(n / 16) * Kp + k) * 16 + n % 16] = (n < N && k < K) ? B[(size_t)k * ldb + n] : (half)0;
}

// Versions with transposed input (op(A) = Aᵀ: A stored as K×M; op(B) = Bᵀ: B stored as N×K).
// Grid swapped so that neighbouring work-items read contiguous memory: pack_a2_pad_t (Mp, Kp), pack_b_pad_t (Kp, Np).
__kernel void pack_a2_pad_t(__global const half *A, __global half *Ap, int M, int K, int lda, int Kp) {
    const int m = get_global_id(0), k = get_global_id(1);
    const int r = m % 16, kk = k % 16, h = kk / 8, kl = kk % 8;
    const int g = 2 * (r / 4) + ((kl % 2 == 0) ? 1 : 0);
    Ap[((size_t)(m / 16) * (Kp / 16) + k / 16) * 256 + h * 128 + g * 16 + (r % 4) * 4 + kl / 2] =
        (m < M && k < K) ? A[(size_t)k * lda + m] : (half)0;
}
__kernel void pack_b_pad_t(__global const half *B, __global half *Bp, int K, int N, int ldb, int Kp) {
    const int k = get_global_id(0), n = get_global_id(1);
    Bp[((size_t)(n / 16) * Kp + k) * 16 + n % 16] = (n < N && k < K) ? B[(size_t)n * ldb + k] : (half)0;
}

// ---------- Accurate variant for a hand-written asm loop: 16×16 tile, FP16 sum in chunks of KC2 products -> FP32 ----------
// A3: k16 block = 4 quarters (kl 0..3) of 64 halves = 4 GRF (128 B). GRF g: rows 8*(g/2)..+7;
//     odd g -> even kl (0,2), even g -> odd kl; element (r%8)*2 + kl/2.
// B: plain panels of 16 columns (pack_b_pad); a quarter = 4 rows × 32 B = 128 B.
#ifndef KC2
#define KC2 32
#endif
inline ushort s4(ushort4 v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void hgemm_16x16kc(__global const ushort *Ap, __global const ushort *Bp, __global half *C,
                            int M, int ldc, int K) {
    const int pid = get_group_id(0) * get_num_sub_groups() + get_sub_group_id();
    const int npm = M / 16, npn = ldc / 16;
    const int group = pid / (GM * npn), m_first = group * GM;
    const int gsz = min(npm - m_first, GM);
    const int pm = m_first + (pid % (GM * npn)) % gsz, pn = (pid % (GM * npn)) / gsz;
    float f[16]; half h[16];
    for (int r = 0; r < 16; r++) { f[r] = 0; h[r] = 0; }
    __global const ushort *pa = Ap + (size_t)pm * (K / 16) * 256;
    __global const ushort *pb = Bp + (size_t)pn * K * 16;
    for (int k = 0; k < K; k += 16) {
        __attribute__((opencl_unroll_hint))
        for (int q = 0; q < 4; q++) {
            ushort4 a = intel_sub_group_block_read_us4(pa + q * 64);
            ushort4 b = intel_sub_group_block_read_us4(pb + q * 64);
            __attribute__((opencl_unroll_hint))
            for (int kl = 0; kl < 4; kl++)
                __attribute__((opencl_unroll_hint))
                for (int r = 0; r < 16; r++) {
                    const int g = 2 * (r / 8) + ((kl % 2 == 0) ? 1 : 0);
                    h[r] = fma(as_half(sub_group_broadcast(s4(a, g), (r % 8) * 2 + kl / 2)), as_half(s4(b, kl)), h[r]);
                }
        }
        pa += 256; pb += 256;
        if ((k + 16) % KC2 == 0)
            for (int r = 0; r < 16; r++) { f[r] += (float)h[r]; h[r] = 0; }
    }
    for (int r = 0; r < 16; r++)
        intel_sub_group_block_write_us((__global ushort *)(C + (size_t)(pm * 16 + r) * ldc + pn * 16),
                                       as_ushort((half)(f[r] + (float)h[r])));
}
// A3 packing with edges: grid (Kp, Mp); transposed version (A stored K×M): grid (Mp, Kp).
__kernel void pack_a3_pad(__global const half *A, __global half *Ap, int M, int K, int lda, int Kp) {
    const int k = get_global_id(0), m = get_global_id(1);
    const int r = m % 16, kk = k % 16, q = kk / 4, kl = kk % 4, g = 2 * (r / 8) + ((kl % 2 == 0) ? 1 : 0);
    Ap[((size_t)(m / 16) * (Kp / 16) + k / 16) * 256 + q * 64 + g * 16 + (r % 8) * 2 + kl / 2] =
        (m < M && k < K) ? A[(size_t)m * lda + k] : (half)0;
}
__kernel void pack_a3_pad_t(__global const half *A, __global half *Ap, int M, int K, int lda, int Kp) {
    const int m = get_global_id(0), k = get_global_id(1);
    const int r = m % 16, kk = k % 16, q = kk / 4, kl = kk % 4, g = 2 * (r / 8) + ((kl % 2 == 0) ? 1 : 0);
    Ap[((size_t)(m / 16) * (Kp / 16) + k / 16) * 256 + q * 64 + g * 16 + (r % 8) * 2 + kl / 2] =
        (m < M && k < K) ? A[(size_t)k * lda + m] : (half)0;
}

// ---------- HGEMM with a single B pointer (Bq layout) -- template for the generator with automatic analysis ----------
// Bq: [n/32][k/16][c 0..1][kk 0..15][16]: in a k16 block first the 16 rows of column group c=0 (512 B), then c=1 (512 B).
// Half-step h (kk 8h..8h+7): B0 = 256 B from pb + 256h, B1 = 256 B from pb + 512 + 256h. Step k16: pb += 1 KB, pa += 512 B.
inline void tile_16x32q(__global const ushort *Ap, __global const ushort *Bp, __global half *C,
                        int K, int ldc, int pm, int pn) {
    const int m0 = pm * 16, n0 = pn * 32;
    half acc0[16], acc1[16];
    for (int r = 0; r < 16; r++) { acc0[r] = 0; acc1[r] = 0; }
    __global const ushort *pa = Ap + (size_t)pm * (K / 16) * 256;
    __global const ushort *pb = Bp + (size_t)pn * (K / 16) * 512;
    for (int k = 0; k < K; k += 16) {
        ushort8 a_lo = intel_sub_group_block_read_us8(pa);
        ushort8 a_hi = intel_sub_group_block_read_us8(pa + 128);
        ushort8 b0l = intel_sub_group_block_read_us8(pb);
        ushort8 b0h = intel_sub_group_block_read_us8(pb + 128);
        ushort8 b1l = intel_sub_group_block_read_us8(pb + 256);
        ushort8 b1h = intel_sub_group_block_read_us8(pb + 384);
        pa += 256; pb += 512;
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

// Common (grouped) tile order and matrix index in the batch: pid -> (b, pm, pn).
// Batched kernels (*_b): count matrices, each Mp×ldc; pointers offset by b * (packed size / result size).
#define TILE(npm, npn) \
    const int pid0 = get_group_id(0) * get_num_sub_groups() + get_sub_group_id(); \
    const int b = pid0 / ((npm) * (npn)), pid = pid0 % ((npm) * (npn)); \
    const int group = pid / (GM * (npn)), m_first = group * GM; \
    const int gsz = min((npm) - m_first, GM); \
    const int pm = m_first + (pid % (GM * (npn))) % gsz, pn = (pid % (GM * (npn))) / gsz;

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void hgemm_16x32q(__global const ushort *Ap, __global const ushort *Bp, __global half *C, int M, int ldc, int K) {
    TILE(M / 16, ldc / 32)
    tile_16x32q(Ap + (size_t)b * M * K, Bp + (size_t)b * ldc * K, C + (size_t)b * M * ldc, K, ldc, pm, pn);
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void hgemm_16x16kc_b(__global const ushort *Ap, __global const ushort *Bp, __global half *C, int M, int ldc, int K) {
    TILE(M / 16, ldc / 16)
    Ap += (size_t)b * M * K; Bp += (size_t)b * ldc * K; C += (size_t)b * M * ldc;
    float f[16]; half h[16];
    for (int r = 0; r < 16; r++) { f[r] = 0; h[r] = 0; }
    __global const ushort *pa = Ap + (size_t)pm * (K / 16) * 256;
    __global const ushort *pb = Bp + (size_t)pn * K * 16;
    for (int k = 0; k < K; k += 16) {
        __attribute__((opencl_unroll_hint))
        for (int q = 0; q < 4; q++) {
            ushort4 a = intel_sub_group_block_read_us4(pa + q * 64);
            ushort4 bb = intel_sub_group_block_read_us4(pb + q * 64);
            __attribute__((opencl_unroll_hint))
            for (int kl = 0; kl < 4; kl++)
                __attribute__((opencl_unroll_hint))
                for (int r = 0; r < 16; r++) {
                    const int g = 2 * (r / 8) + ((kl % 2 == 0) ? 1 : 0);
                    h[r] = fma(as_half(sub_group_broadcast(s4(a, g), (r % 8) * 2 + kl / 2)), as_half(s4(bb, kl)), h[r]);
                }
        }
        pa += 256; pb += 256;
        if ((k + 16) % KC2 == 0)
            for (int r = 0; r < 16; r++) { f[r] += (float)h[r]; h[r] = 0; }
    }
    for (int r = 0; r < 16; r++)
        intel_sub_group_block_write_us((__global ushort *)(C + (size_t)(pm * 16 + r) * ldc + pn * 16),
                                       as_ushort((half)(f[r] + (float)h[r])));
}

// Batched packing (3rd grid dimension = matrix index b; sX = stride of the input matrices in elements).
// Output: matrix b occupies Mp*Kp (A) / Np*Kp (B) elements. The _t versions: transposed input.
#define PK_IDX_A2(m, k, Kp) ((((size_t)((m) / 16) * ((Kp) / 16) + (k) / 16) * 256 + ((k) % 16 / 8) * 128 + \
    (2 * ((m) % 16 / 4) + (((k) % 8) % 2 == 0)) * 16 + ((m) % 4) * 4 + ((k) % 8) / 2))
#define PK_IDX_A3(m, k, Kp) ((((size_t)((m) / 16) * ((Kp) / 16) + (k) / 16) * 256 + ((k) % 16 / 4) * 64 + \
    (2 * ((m) % 16 / 8) + (((k) % 4) % 2 == 0)) * 16 + ((m) % 8) * 2 + ((k) % 4) / 2))
#define PK_IDX_BQ(n, k, Kp) (((((size_t)((n) / 32) * ((Kp) / 16) + (k) / 16) * 2 + ((n) % 32) / 16) * 16 + (k) % 16) * 16 + (n) % 16)
#define PK_IDX_B16(n, k, Kp) (((size_t)((n) / 16) * (Kp) + (k)) * 16 + (n) % 16)
#define PACK_A(name, IDX) \
__kernel void name(__global const half *A, __global half *Ap, int M, int K, int lda, int Kp, long sA, int Mp) { \
    const int k = get_global_id(0), m = get_global_id(1), b = get_global_id(2); \
    Ap[(size_t)b * Mp * Kp + IDX(m, k, Kp)] = (m < M && k < K) ? A[b * sA + (size_t)m * lda + k] : (half)0; } \
__kernel void name##_t(__global const half *A, __global half *Ap, int M, int K, int lda, int Kp, long sA, int Mp) { \
    const int m = get_global_id(0), k = get_global_id(1), b = get_global_id(2); \
    Ap[(size_t)b * Mp * Kp + IDX(m, k, Kp)] = (m < M && k < K) ? A[b * sA + (size_t)k * lda + m] : (half)0; }
#define PACK_B(name, IDX) \
__kernel void name(__global const half *B, __global half *Bp, int K, int N, int ldb, int Kp, long sB, int Np) { \
    const int n = get_global_id(0), k = get_global_id(1), b = get_global_id(2); \
    Bp[(size_t)b * Np * Kp + IDX(n, k, Kp)] = (n < N && k < K) ? B[b * sB + (size_t)k * ldb + n] : (half)0; } \
__kernel void name##_t(__global const half *B, __global half *Bp, int K, int N, int ldb, int Kp, long sB, int Np) { \
    const int k = get_global_id(0), n = get_global_id(1), b = get_global_id(2); \
    Bp[(size_t)b * Np * Kp + IDX(n, k, Kp)] = (n < N && k < K) ? B[b * sB + (size_t)n * ldb + k] : (half)0; }
PACK_A(pk_a2, PK_IDX_A2)
PACK_A(pk_a3, PK_IDX_A3)
PACK_B(pk_bq, PK_IDX_BQ)
PACK_B(pk_b16, PK_IDX_B16)
// Faster packing without transposition (bench/packing.c; n = 1024: B 0.45 -> 0.24 ms, A 0.54 -> 0.31 ms).
// BQ, 2 elements per work-item (grid Np/2, Kp, count)
__kernel void pk_bq_w2(__global const half *B, __global half *Bp, int K, int N, int ldb, int Kp, long sB, int Np) {
    const int n = get_global_id(0) * 2, k = get_global_id(1), b = get_global_id(2);
    __global const half *s = B + b * sB + (size_t)k * ldb + n;
    half2 v;
    if (k < K && n + 2 <= N) v = vload2(0, s);
    else { v.x = (k < K && n < N) ? s[0] : (half)0; v.y = (k < K && n + 1 < N) ? s[1] : (half)0; }
    vstore2(v, 0, Bp + (size_t)b * Np * Kp + PK_IDX_BQ(n, k, Kp)); }
// A3 (accurate variant), work-item = 8 rows (half hh of the tile) × 4 columns k: groups g = 2hh (odd k) and
// 2hh+1 (even k) of 16 elements each ((m%8)*2 + (k%4)/2), 32 elements in a row in total. Grid (Kp/4 * 2, Mp/16).
__kernel void pk_a3_q(__global const half *A, __global half *Ap, int M, int K, int lda, int Kp, long sA, int Mp) {
    const int i = get_global_id(0), hh = i % 2, k0 = (i / 2) * 4, m0 = get_global_id(1) * 16 + hh * 8, b = get_global_id(2);
    __global const half *s = A + b * sA;
    half16 odd, even;
    for (int r = 0; r < 8; r++) {
        const int m = m0 + r;
        half4 v;
        if (m < M && k0 + 4 <= K) v = vload4(0, s + (size_t)m * lda + k0);
        else for (int j = 0; j < 4; j++) ((half *)&v)[j] = (m < M && k0 + j < K) ? s[(size_t)m * lda + k0 + j] : (half)0;
        ((half *)&odd)[r * 2] = v.s1; ((half *)&odd)[r * 2 + 1] = v.s3;
        ((half *)&even)[r * 2] = v.s0; ((half *)&even)[r * 2 + 1] = v.s2;
    }
    __global half *d = Ap + (size_t)b * Mp * Kp + ((size_t)(m0 / 16) * (Kp / 16) + k0 / 16) * 256 + (k0 % 16 / 4) * 64 + 2 * hh * 16;
    vstore16(odd, 0, d); vstore16(even, 0, d + 16);
}
// B16 (accurate variant), 2 elements per work-item. Grid (Np/2, Kp).
__kernel void pk_b16_w2(__global const half *B, __global half *Bp, int K, int N, int ldb, int Kp, long sB, int Np) {
    const int n = get_global_id(0) * 2, k = get_global_id(1), b = get_global_id(2);
    __global const half *s = B + b * sB + (size_t)k * ldb + n;
    half2 v;
    if (k < K && n + 2 <= N) v = vload2(0, s);
    else { v.x = (k < K && n < N) ? s[0] : (half)0; v.y = (k < K && n + 1 < N) ? s[1] : (half)0; }
    vstore2(v, 0, Bp + (size_t)b * Np * Kp + PK_IDX_B16(n, k, Kp)); }
// A2, work-item = 4 rows (quarter of a 16 tile) × 8 columns k: reads 4 × half8, writes 32 elements in a row
// (group g = 2q: odd k, g = 2q+1: even k; within the group (m%4)*4 + (k%8)/2). Neighbouring work-items: quarters q,
// then halves h of the tile, then consecutive k tiles -- the subgroup writes 1 KB contiguously. Grid (Kp/8 * 4, Mp/16).
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
// Batched epilog: C_b = alpha * Cp_b + beta * C_b
__kernel void gen9blas_epilog_b(__global const half *Cp, int ldp, int Mp, __global half *C, int ldc, long sC, float alpha, float beta) {
    const int j = get_global_id(0), i = get_global_id(1), b = get_global_id(2);
    float v = alpha * vload_half(0, Cp + (size_t)b * Mp * ldp + (size_t)i * ldp + j);
    __global half *c = C + b * sC + (size_t)i * ldc + j;
    if (beta != 0.0f) v += beta * vload_half(0, c);
    vstore_half(v, 0, c);
}

// ---------- "v" versions: separate N (columns to compute), ldc (row stride of C), K (loop length) and Kfull
// (stride of packed data panels). Allows computing a piece of packed data once (sub-buffers with offset).
// Batched: matrix b starts at Ap + b·M·Kfull, Bp + b·N·Kfull, C + b·M·ldc.
#define TILE_V(npm, npn) TILE(npm, npn)
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void hgemm_v(__global const ushort *Ap, __global const ushort *Bp, __global half *C,
                      int M, int N, int ldc, int K, int Kfull) {
    TILE_V(M / 16, N / 32)
    Ap += (size_t)b * M * Kfull; Bp += (size_t)b * N * Kfull; C += (size_t)b * M * ldc;
    const int m0 = pm * 16, n0 = pn * 32;
    half acc0[16], acc1[16];
    for (int r = 0; r < 16; r++) { acc0[r] = 0; acc1[r] = 0; }
    __global const ushort *pa = Ap + (size_t)pm * (Kfull / 16) * 256;
    __global const ushort *pb = Bp + (size_t)pn * (Kfull / 16) * 512;
    for (int k = 0; k < K; k += 16) {
        ushort8 a_lo = intel_sub_group_block_read_us8(pa), a_hi = intel_sub_group_block_read_us8(pa + 128);
        ushort8 b0l = intel_sub_group_block_read_us8(pb), b0h = intel_sub_group_block_read_us8(pb + 128);
        ushort8 b1l = intel_sub_group_block_read_us8(pb + 256), b1h = intel_sub_group_block_read_us8(pb + 384);
        pa += 256; pb += 512;
        __attribute__((opencl_unroll_hint))
        for (int r = 0; r < 16; r++) {
            ushort av = r < 8 ? sub8(a_lo, r) : sub8(a_hi, r - 8);
            __attribute__((opencl_unroll_hint))
            for (int q = 0; q < 8; q++) {
                acc0[r] = fma(BC(av, q), as_half(sub8(b0l, q)), acc0[r]); acc1[r] = fma(BC(av, q), as_half(sub8(b1l, q)), acc1[r]); }
            __attribute__((opencl_unroll_hint))
            for (int q = 0; q < 8; q++) {
                acc0[r] = fma(BC(av, q + 8), as_half(sub8(b0h, q)), acc0[r]); acc1[r] = fma(BC(av, q + 8), as_half(sub8(b1h, q)), acc1[r]); }
        }
    }
    for (int r = 0; r < 16; r++) {
        __global ushort *row = (__global ushort *)(C + (size_t)(m0 + r) * ldc + n0);
        intel_sub_group_block_write_us(row, as_ushort(acc0[r]));
        intel_sub_group_block_write_us(row + 16, as_ushort(acc1[r]));
    }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void hgemm_kc_v(__global const ushort *Ap, __global const ushort *Bp, __global half *C,
                         int M, int N, int ldc, int K, int Kfull) {
    TILE_V(M / 16, N / 16)
    Ap += (size_t)b * M * Kfull; Bp += (size_t)b * N * Kfull; C += (size_t)b * M * ldc;
    float f[16]; half h[16];
    for (int r = 0; r < 16; r++) { f[r] = 0; h[r] = 0; }
    __global const ushort *pa = Ap + (size_t)pm * (Kfull / 16) * 256;
    __global const ushort *pb = Bp + (size_t)pn * Kfull * 16;
    for (int k = 0; k < K; k += 16) {
        __attribute__((opencl_unroll_hint))
        for (int q = 0; q < 4; q++) {
            ushort4 a = intel_sub_group_block_read_us4(pa + q * 64);
            ushort4 bb = intel_sub_group_block_read_us4(pb + q * 64);
            __attribute__((opencl_unroll_hint))
            for (int kl = 0; kl < 4; kl++)
                __attribute__((opencl_unroll_hint))
                for (int r = 0; r < 16; r++) {
                    const int g = 2 * (r / 8) + ((kl % 2 == 0) ? 1 : 0);
                    h[r] = fma(as_half(sub_group_broadcast(s4(a, g), (r % 8) * 2 + kl / 2)), as_half(s4(bb, kl)), h[r]);
                }
        }
        pa += 256; pb += 256;
        if ((k + 16) % KC2 == 0)
            for (int r = 0; r < 16; r++) { f[r] += (float)h[r]; h[r] = 0; }
    }
    for (int r = 0; r < 16; r++)
        intel_sub_group_block_write_us((__global ushort *)(C + (size_t)(pm * 16 + r) * ldc + pn * 16),
                                       as_ushort((half)(f[r] + (float)h[r])));
}
// Epilog with triangle: uplo 0 = lower (j <= i), 1 = upper (j >= i), 2 = whole. Out of range C unchanged.
__kernel void gen9blas_epilog_tri(__global const half *Cp, int ldp, __global half *C, int ldc, float alpha, float beta, int uplo) {
    const int j = get_global_id(0), i = get_global_id(1);
    if ((uplo == 0 && j > i) || (uplo == 1 && j < i)) return;
    float v = alpha * vload_half(0, Cp + (size_t)i * ldp + j);
    __global half *c = C + (size_t)i * ldc + j;
    if (beta != 0.0f) v += beta * vload_half(0, c);
    vstore_half(v, 0, c);
}

// ---------- TRSM: preparation and store on the GPU (half) ----------
// L (n×n, lower) = T after optionally reversing the index order; T = op(A) (left) or op(A)ᵀ (right), tr: T = Aᵀ.
__kernel void trsm_L_h(__global const half *A, int lda, __global half *L, int n, int uplo, int tr, int diag, int rev) {
    const int j = get_global_id(0), i = get_global_id(1);
    const int si = rev ? n - 1 - i : i, sj = rev ? n - 1 - j : j;
    const int ai = tr ? sj : si, aj = tr ? si : sj;
    const int w = uplo == 0 ? aj <= ai : aj >= ai;
    half v = w ? A[(size_t)ai * lda + aj] : (half)0;
    if (diag && si == sj) v = (half)1;
    L[(size_t)i * n + j] = v;
}
// Y (np × mp) = alpha·R, R = B (left) or Bᵀ (right), rows in rev order; zero-padded.
__kernel void trsm_Y_h(__global const half *B, int ldb, __global half *Y, int n, int m, int mp, int left, int rev, float alpha) {
    const int j = get_global_id(0), i = get_global_id(1);
    float v = 0.0f;
    if (i < n && j < m) { const int si = rev ? n - 1 - i : i; v = alpha * (float)(left ? B[(size_t)si * ldb + j] : B[(size_t)j * ldb + si]); }
    Y[(size_t)i * mp + j] = (half)v;
}
// Result back to B (inverse of trsm_Y, without alpha).
__kernel void trsm_store_h(__global const half *Y, int mp, __global half *B, int ldb, int n, int left, int rev) {
    const int j = get_global_id(0), i = get_global_id(1);
    const int si = rev ? n - 1 - i : i;
    if (left) B[(size_t)si * ldb + j] = Y[(size_t)i * mp + j]; else B[(size_t)j * ldb + si] = Y[(size_t)i * mp + j];
}

// ---------- Level 3 data preparation on the GPU (half) ----------
// Full n×n matrix from the uplo triangle of A: sym=1 mirror (symmetric), sym=0 triangular with zeros
// (diag=1: ones on the diagonal); tr=1: transposed result. 16×16 tiles: both needed tiles of A (D = A[bi.., bj..]
// and mirrored E = A[bj.., bi..]) read row-wise into local memory, T written row-wise (no strided access).
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void gen9blas_expand_h(__global const half *A, int lda, __global half *T, int n, int uplo, int sym, int diag, int tr) {
    __local half D[16][17], E[16][17];
    const int lj = get_local_id(0), li = get_local_id(1), bj = get_group_id(0) * 16, bi = get_group_id(1) * 16;
    const int r = bi + li, c = bj + lj;
    D[li][lj] = r < n && c < n ? A[(size_t)r * lda + c] : (half)0;
    if (sym || tr) E[li][lj] = bj + li < n && bi + lj < n ? A[(size_t)(bj + li) * lda + bi + lj] : (half)0;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (r >= n || c >= n) return;
    // result element T(r, c) = F(i, j), (i, j) = tr ? (c, r) : (r, c); A[r][c] = D[li][lj], A[c][r] = E[lj][li]
    const int i = tr ? c : r, j = tr ? r : c;
    const half aij = tr ? E[lj][li] : D[li][lj], aji = tr ? D[li][lj] : E[lj][li];
    const int w = uplo == 0 ? j <= i : j >= i;
    half v = w ? aij : (sym ? aji : (half)0);
    if (!sym && diag && i == j) v = (half)1;
    T[(size_t)r * n + c] = v;
}
// SYR2K: X = [op(A) | op(B)], Y = [op(B) | op(A)] (N × 2K, ld 2K); trans: A, B stored K×N.
__kernel void gen9blas_syr2k_merge_h(__global const half *A, int lda, __global const half *B, int ldb, __global half *X, __global half *Y,
                         int K, int trans) {
    const int k = get_global_id(0), i = get_global_id(1);
    const half a = trans ? A[(size_t)k * lda + i] : A[(size_t)i * lda + k], b = trans ? B[(size_t)k * ldb + i] : B[(size_t)i * ldb + k];
    const size_t x = (size_t)i * 2 * K + k;
    X[x] = a; X[x + K] = b; Y[x] = b; Y[x + K] = a;
}
// TRSM: inverses of the diagonal blocks of L (n×n, ld n, lower) in FP64, recursive. "Global" indices g = 0..nbl·nb
// (padding above n: L = I). X (double): block bi from bi·nb·nb, ld nb; the upper triangle of X is never read.
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define INV_L(i, k) ((i) < n && (k) < n ? (double)L[(size_t)(i) * n + (k)] : (double)((i) == (k)))
#define INV_X(X, i, k) X[(size_t)((i) / nb) * nb * nb + (size_t)((i) % nb) * nb + (k) % nb]
// Base blocks 32×32: work-item = column j of block g, forward substitution.
__kernel void trsm_inv0_h(__global const half *L, int n, int nb, __global double *X) {
    const int j = get_global_id(0), g0 = get_global_id(1) * 32;
    for (int i = j; i < 32; i++) {
        double s = i == j ? 1.0 : 0.0;
        for (int k = j; k < i; k++) s -= INV_L(g0 + i, g0 + k) * INV_X(X, g0 + k, g0 + j);
        INV_X(X, g0 + i, g0 + j) = s / INV_L(g0 + i, g0 + i);
    }
}
// Level h (pairs of h×h blocks into a 2h block): step 1: Tm = L21·X11 (Tm in place of X21 in the Tm buffer),
// step 2: X21 = −X22·Tm. Grid (h, h, number of pairs).
__kernel void trsm_inv1_h(__global const half *L, int n, int nb, __global double *X, __global double *Tm, int h) {
    const int j = get_global_id(0), i = get_global_id(1), r0 = get_global_id(2) * 2 * h;
    double s = 0.0;
    for (int k = j; k < h; k++) s += INV_L(r0 + h + i, r0 + k) * INV_X(X, r0 + k, r0 + j);
    INV_X(Tm, r0 + h + i, r0 + j) = s;
}
__kernel void trsm_inv2_h(__global const half *L, int n, int nb, __global double *X, __global double *Tm, int h) {
    const int j = get_global_id(0), i = get_global_id(1), r0 = get_global_id(2) * 2 * h;
    double s = 0.0;
    for (int k = 0; k <= i; k++) s += INV_X(X, r0 + h + i, r0 + h + k) * INV_X(Tm, r0 + h + k, r0 + j);
    INV_X(X, r0 + h + i, r0 + j) = -s;
}
// Result: out (block bi from bi·nb·nb, ld b = min(nb, n − bi·nb)), upper triangle = 0. Grid (nb, nb, nbl).
__kernel void trsm_inv3_h(int n, int nb, __global const double *X, __global half *out) {
    const int j = get_global_id(0), i = get_global_id(1), bi = get_global_id(2), b = min(nb, n - bi * nb);
    if (i >= b || j >= b) return;
    out[(size_t)bi * nb * nb + (size_t)i * b + j] = j <= i ? (half)X[(size_t)bi * nb * nb + (size_t)i * nb + j] : (half)0;
}
