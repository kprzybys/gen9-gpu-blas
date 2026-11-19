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
#ifdef PLAIN_PACK_A2   // Gen12LP build (lib/gen_kernels.py tgllp): the main kernel is the OpenCL C hgemm_v, which reads
                       // A as a plain 16 x 16 tile (row m % 16, column k % 16), not the A2 layout of gen_h.bin
#define PK_IDX_A2(m, k, Kp) (((((size_t)((m) / 16) * ((Kp) / 16) + (k) / 16) * 16 + (m) % 16) * 16) + (k) % 16)
#else
#define PK_IDX_A2(m, k, Kp) ((((size_t)((m) / 16) * ((Kp) / 16) + (k) / 16) * 256 + ((k) % 16 / 8) * 128 + \
    (2 * ((m) % 16 / 4) + (((k) % 8) % 2 == 0)) * 16 + ((m) % 4) * 4 + ((k) % 8) / 2))
#endif
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
#ifndef PLAIN_PACK_A2   // writes the A2 layout directly
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
#endif
// Batched epilog: C_b = alpha * Cp_b + beta * C_b
__kernel void gen9blas_epilog_b(__global const half *Cp, int ldp, int Mp, __global half *C, int ldc, long sC, float alpha, float beta) {
    const int j = get_global_id(0), i = get_global_id(1), b = get_global_id(2);
    float v = alpha * vload_half(0, Cp + (size_t)b * Mp * ldp + (size_t)i * ldp + j);
    __global half *c = C + b * sC + (size_t)i * ldc + j;
    if (beta != 0.0f) v += beta * vload_half(0, c);
    vstore_half(v, 0, c);
}

// ggml backend (F16 weights x F32 activations, docs/ggml-backend-design.md section 4):
// F32 activations packed straight into the A3 layout of the accurate FP16 GEMM, rounded to nearest. Grid (Kp, Mp, count).
__kernel void pk_a3_f32(__global const float *A, __global half *Ap, int M, int K, int lda, int Kp, long sA, int Mp) {
    const int k = get_global_id(0), m = get_global_id(1), b = get_global_id(2);
    vstore_half_rte((m < M && k < K) ? A[b * sA + (size_t)m * lda + k] : 0.0f, 0, Ap + (size_t)b * Mp * Kp + PK_IDX_A3(m, k, Kp)); }
// pk_a3_f32 with 16 B per lane (same values, same RTE rounding): one 16-lane work-group per 16 x 16 tile of A3. Lane L
// reads row m0 + L (4 x float4 when the 16 k are inside K, else element by element with zeros outside), the tile goes
// through SLM, and lane L writes the 8-half chunks L and L + 16 of the tile's 256 contiguous halves (one vstore_half8).
// Chunk c: kq = c / 8, h = c / 4 % 2, even = c / 2 % 2, mq = c % 2; element 2 i + t = (m = 8 h + 4 mq + i,
// k = 4 kq + (even ? 0 : 1) + 2 t)  [PK_IDX_A3: ((k % 4) % 2 == 0) selects the second 16-run of the pair].
// Grid (Kp / 16 * 16, Mp / 16, batch), local (16, 1, 1).
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void pk_a3_f32v(__global const float *A, __global half *Ap, int M, int K, int lda, int Kp, long sA, int Mp) {
    __local float tl[16][17];
    const int L = get_local_id(0), k0 = get_group_id(0) * 16, m0 = get_global_id(1) * 16, b = get_global_id(2);
    const int m = m0 + L;
    __global const float *row = A + b * sA + (size_t)m * lda + k0;
    if (m < M && k0 + 16 <= K) {
        const float4 a = vload4(0, row), c = vload4(1, row), d = vload4(2, row), e = vload4(3, row);
        tl[L][0] = a.s0; tl[L][1] = a.s1; tl[L][2] = a.s2; tl[L][3] = a.s3; tl[L][4] = c.s0; tl[L][5] = c.s1;
        tl[L][6] = c.s2; tl[L][7] = c.s3; tl[L][8] = d.s0; tl[L][9] = d.s1; tl[L][10] = d.s2; tl[L][11] = d.s3;
        tl[L][12] = e.s0; tl[L][13] = e.s1; tl[L][14] = e.s2; tl[L][15] = e.s3;
    } else {
        for (int kk = 0; kk < 16; kk++) tl[L][kk] = (m < M && k0 + kk < K) ? row[kk] : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    __global half *tile = Ap + (size_t)b * Mp * Kp + ((size_t)(m0 / 16) * (Kp / 16) + k0 / 16) * 256;
    for (int cc = 0; cc < 2; cc++) {
        const int c = L + 16 * cc, kq = c / 8, h = c / 4 % 2, ev = c / 2 % 2, mq = c % 2;   // ev: run of even k
        const int mb = 8 * h + 4 * mq, kb = 4 * kq + (ev ? 0 : 1);
        const float8 v = (float8)(tl[mb][kb], tl[mb][kb + 2], tl[mb + 1][kb], tl[mb + 1][kb + 2],
                                  tl[mb + 2][kb], tl[mb + 2][kb + 2], tl[mb + 3][kb], tl[mb + 3][kb + 2]);
        vstore4(as_uint4(convert_half8_rte(v)), 0, (__global uint *)(tile + c * 8));   // 16 B: one xyzw store
    }
}
// gen9blas_epilog_hf with 4 consecutive j per work-item: 8 B FP16 read, 16 B F32 write, so a SIMD16 store covers
// whole 64 B lines (8 per lane = 32 B split every line between two messages and was slower than 1 per lane, 2026-11-13).
// The per-element expression is the same as in gen9blas_epilog_hf. Grid ((N + 3) / 4, rows, batch).
__kernel void gen9blas_epilog_hf8(__global const half *Cp, int ldp, int Mp, __global float *C, int ldc, long sC,
                                  float alpha, float beta, int N, __global int *nonfinite) {
    const int j0 = get_global_id(0) * 4, i = get_global_id(1), b = get_global_id(2);
    __global const half *cp = Cp + (size_t)b * Mp * ldp + (size_t)i * ldp + j0;
    __global float *c = C + b * sC + (size_t)i * ldc + j0;
    int bad = 0;
    if (j0 + 4 <= N && beta == 0.0f) {
        const float4 v = alpha * convert_float4(as_half4(vload2(0, (__global const uint *)cp)));
        bad = any(!isfinite(v));
        vstore4(v, 0, c);
    } else {
        for (int t = 0; t < 4 && j0 + t < N; t++) {
            float v = alpha * vload_half(0, cp + t);
            if (beta != 0.0f) v += beta * c[t];
            bad |= !isfinite(v);
            c[t] = v;
        }
    }
    if (bad) atomic_or(nonfinite, 1);
}
// gen9blas_epilog_hf8 with the ops that follow a MUL_MAT in whisper's encoder fused in (one pass over C instead of
// separate ADD / GELU kernels): v = fp32(Cp); flags & 1: v += bias[j]; flags & 2: v = gelu(v) (gen9blas_gelu's formula);
// flags & 4: v += R[i * ldr + j]. Same F32 operations in the same order as the separate kernels -> bit-identical.
// R may be C itself (in-place residual: each work-item reads its 4 elements before writing them). The nonfinite check
// is on the raw GEMM value, as gen9blas_epilog_hf8 (FP16 overflow -> the caller recomputes unfused in F32).
// One GELU for gen9blas_gelu and the fused epilogue: FP contraction off, so the result does not depend on the kernel
// it is inlined into (with contraction on, IGC fused differently in the vector epilogue: 0.5 % of values 1 ulp apart).
static inline float epi_gelu(float v) {
    #pragma OPENCL FP_CONTRACT OFF
    if (v <= -10.0f) return 0.0f;
    if (v >= 10.0f) return v;
    const float h = convert_float(convert_half_rte(v));
    const float g = 0.5f * h * (1.0f + tanh(0.79788456080286535587989211986876f * h * (1.0f + 0.044715f * h * h)));
    return convert_float(convert_half_rte(g));
}
__kernel void gen9blas_epilog_hf8x(__global const half *Cp, int ldp, int Mp, __global float *C, int ldc, int N,
                                   __global int *nonfinite, __global const float *bias, __global const float *R, int ldr,
                                   int flags) {
    const int j0 = get_global_id(0) * 4, i = get_global_id(1);
    __global const half *cp = Cp + (size_t)i * ldp + j0;
    __global float *c = C + (size_t)i * ldc + j0;
    __global const float *r = R + (size_t)i * ldr + j0;
    int bad = 0;
    if (j0 + 4 <= N) {
        float4 v = convert_float4(as_half4(vload2(0, (__global const uint *)cp)));
        bad = any(!isfinite(v));
        if (flags & 1) v += vload4(0, bias + j0);
        if (flags & 2) v = (float4)(epi_gelu(v.s0), epi_gelu(v.s1), epi_gelu(v.s2), epi_gelu(v.s3));
        if (flags & 4) v += vload4(0, r);
        vstore4(v, 0, c);
    } else {
        for (int t = 0; t < 4 && j0 + t < N; t++) {
            float v = vload_half(0, cp + t);
            bad |= !isfinite(v);
            if (flags & 1) v += bias[j0 + t];
            if (flags & 2) v = epi_gelu(v);
            if (flags & 4) v += r[t];
            c[t] = v;
        }
    }
    if (bad) atomic_or(nonfinite, 1);
}
// Epilogue FP16 work buffer -> F32 C; *nonfinite |= 1 if any result is inf or NaN (FP16 overflow: recompute in F32).
__kernel void gen9blas_epilog_hf(__global const half *Cp, int ldp, int Mp, __global float *C, int ldc, long sC, float alpha,
                                 float beta, __global int *nonfinite) {
    const int j = get_global_id(0), i = get_global_id(1), b = get_global_id(2);
    float v = alpha * vload_half(0, Cp + (size_t)b * Mp * ldp + (size_t)i * ldp + j);
    __global float *c = C + b * sC + (size_t)i * ldc + j;
    if (beta != 0.0f) v += beta * c[0];
    if (!isfinite(v)) atomic_or(nonfinite, 1);
    c[0] = v;
}

// ggml backend FFN fusion: h = silu(g) * u (ggml SWIGLU, silu(x) = x / (1 + e^-x)) from two FP16 GEMM results
// (row-major, row stride ldp), computed in F32 and written as FP16 straight into the A3 packed layout of the next GEMM
// (down projection, Kp = its padded K). *nonfinite |= 1 on inf / NaN in g, u or h, or |h| beyond FP16. Grid (Kp, Mp).
__kernel void gen9blas_swiglu_pack_a3(__global const half *G, __global const half *U, int ldp, int M, int N,
                                      __global half *Ap, int Kp, __global int *nonfinite) {
    const int k = get_global_id(0), m = get_global_id(1);
    float h = 0.0f;
    if (m < M && k < N) {
        const float g = vload_half(0, G + (size_t)m * ldp + k), u = vload_half(0, U + (size_t)m * ldp + k);
        h = g / (1.0f + exp(-g)) * u;
        if (!isfinite(g) || !isfinite(u) || !(fabs(h) <= 65504.0f)) atomic_or(nonfinite, 1);
    }
    vstore_half_rte(h, 0, Ap + PK_IDX_A3(m, k, Kp));
}

// ggml backend: quantized weights (ggml block layouts, row n at W + n * row_bytes) dequantized straight into the B16 packed
// layout of the accurate FP16 GEMM (GEN9BLAS_HS, Kp = K). qtype 0 = Q8_0 (32-value blocks of 34 B), 1 = Q4_K, 2 = Q6_K
// (256-value blocks of 144 / 210 B). Same arithmetic order as ggml's dequantize_row_* (no FMA contraction) and FP16
// rounding to nearest even, so the result equals to_float + ggml_fp32_to_fp16_row bit for bit. Work-item = one block of
// row n; neighbouring work-items = neighbouring rows, so a SIMD thread stores 32 contiguous bytes per k. Rows n >= N are
// zeros. *oor |= 1 if any value is beyond the FP16 range or not finite. Grid (rows, K / block), global offset = first row.
#pragma OPENCL FP_CONTRACT OFF
#define DQ_PUT(k, v) do { const float v_ = (v); bad |= !(fabs(v_) <= 65504.0f); vstore_half_rte(v_, 0, d + (size_t)(k) * 16); } while (0)
__kernel void gen9blas_dequant_b16(__global const uchar *W, ulong row_bytes, int qtype, int N, int Kp,
                                   __global half *Bp, __global int *oor) {
    const int n = get_global_id(0), j = get_global_id(1), QB = qtype == 0 ? 32 : 256;
    __global half *d = Bp + ((size_t)(n / 16) * Kp + (size_t)j * QB) * 16 + n % 16;
    if (n >= N) { for (int k = 0; k < QB; k++) d[(size_t)k * 16] = 0; return; }
    __global const uchar *b = W + (size_t)n * row_bytes + (size_t)j * (qtype == 0 ? 34 : qtype == 1 ? 144 : 210);
    int bad = 0;
    if (qtype == 0) {
        const float dd = vload_half(0, (__global const half *)b);
        for (int k = 0; k < 32; k++) DQ_PUT(k, (float)((__global const char *)b)[2 + k] * dd);
    } else if (qtype == 1) {
        const float dd = vload_half(0, (__global const half *)b), mn = vload_half(1, (__global const half *)b);
        __global const uchar *s = b + 4, *q = b + 16;
        for (int jj = 0, is = 0; jj < 256; jj += 64, is += 2, q += 32) {
            // ggml get_scale_min_k4 for is and is + 1, in scalars. A first version kept them in private arrays
            // (uchar sc[2], m[2]): IGC put them in a 1 KB scratch buffer addressed through r0.5, which kcl does not set
            // up -> threads overwrote each other, wrong values from N ~ 1024 under kcl, exact through NEO (2026-11-13).
            // lib/gen_kernels.py now rejects such kernels (per_thread_memory_buffers: scratch).
            const int i0 = is, i1 = is + 1;
            const uint sc0 = i0 < 4 ? s[i0] & 63 : (s[i0 + 4] & 0xF) | ((s[i0 - 4] >> 6) << 4);
            const uint mm0 = i0 < 4 ? s[i0 + 4] & 63 : (s[i0 + 4] >> 4) | ((s[i0] >> 6) << 4);
            const uint sc1 = i1 < 4 ? s[i1] & 63 : (s[i1 + 4] & 0xF) | ((s[i1 - 4] >> 6) << 4);
            const uint mm1 = i1 < 4 ? s[i1 + 4] & 63 : (s[i1 + 4] >> 4) | ((s[i1] >> 6) << 4);
            const float d1 = dd * (float)sc0, m1 = mn * (float)mm0, d2 = dd * (float)sc1, m2 = mn * (float)mm1;
            for (int l = 0; l < 32; l++) DQ_PUT(jj + l, d1 * (float)(q[l] & 0xF) - m1);
            for (int l = 0; l < 32; l++) DQ_PUT(jj + 32 + l, d2 * (float)(q[l] >> 4) - m2);
        }
    } else {
        const float dd = vload_half(0, (__global const half *)(b + 208));
        __global const uchar *ql = b, *qh = b + 128;
        __global const char *sc = (__global const char *)(b + 192);
        for (int nn = 0; nn < 256; nn += 128, ql += 64, qh += 32, sc += 8)
            for (int l = 0; l < 32; l++) {
                const int is = l / 16;
                const int q1 = (int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                DQ_PUT(nn + l, dd * (float)sc[is] * (float)q1);
                DQ_PUT(nn + l + 32, dd * (float)sc[is + 2] * (float)q2);
                DQ_PUT(nn + l + 64, dd * (float)sc[is + 4] * (float)q3);
                DQ_PUT(nn + l + 96, dd * (float)sc[is + 6] * (float)q4);
            }
    }
    if (bad) atomic_or(oor, 1);
}

// v2 of the above (same arithmetic, bit-identical): work-group of 128 = 16 rows (one B16 row group g) x one 256-value
// chunk j (Q8_0: 8 blocks; needs K % 256 == 0, and W + n * row_bytes 16 B aligned for Q8_0 / Q4_K). Phase 1 copies the
// 16 rows' chunk bytes to SLM (16 B loads; Q6_K 2 B); phase 2 computes the sub-block scales / mins once; phase 3: lane t
// computes rows 2 (t % 8), + 1 at k = t / 8 + 16 i and stores them as one half2, so 8 lanes write one k (32 B) and the
// 8 KB output tile is written contiguously. No private arrays (kcl has no scratch space: see v1's Q4_K note).
// Grid ((rows / 16) * 128, K / 256), local (128, 1); n0 = first row of the launch (multiple of 16).
__kernel __attribute__((reqd_work_group_size(128, 1, 1)))
void gen9blas_dequant_b16s(__global const uchar *W, ulong row_bytes, int qtype, int N, int Kp, int n0,
                           __global half *Bp, __global int *oor) {
    // SLM rows RS bytes apart: Q8_0 / Q4_K chunk + 4 B so that the 8 even rows read by 8 lanes in phase 3 fall in 8
    // different banks (2 x 144 B = 72 dwords hit only banks 0 and 8: Q4_K was 1.6x slower than Q6_K with RS = 210)
    __local uint sm1[16 * 276 / 4];
    __local uchar *sm = (__local uchar *)sm1;
    __local float pd[16 * 17], pm[16 * 9];   // per row (stride 17 / 9: no bank conflicts) and sub-block: scale
                                             // (Q4_K d1, Q6_K dd*sc, Q8_0 d), Q4_K min m1
    const int t = get_local_id(0), g = n0 / 16 + get_group_id(0), j = get_global_id(1);
    const int CB = qtype == 0 ? 272 : qtype == 1 ? 144 : 210, RS = qtype == 2 ? 210 : CB + 4;
    if (qtype != 2) {   // Q8_0 / Q4_K: chunks of 272 / 144 B, rows and W 16 B aligned -> 16 B loads
        const int C16 = CB / 16;
        for (int i = t; i < 16 * C16; i += 128) {
            const int r = i / C16, n = g * 16 + r;
            const uint4 v = n < N ? *(__global const uint4 *)(W + (size_t)n * row_bytes + (size_t)j * CB + (i - r * C16) * 16)
                                  : (uint4)(0);
            __local uint *d = sm1 + (r * RS + (i - r * C16) * 16) / 4;
            d[0] = v.s0; d[1] = v.s1; d[2] = v.s2; d[3] = v.s3;
        }
    } else {            // Q6_K: 210 B chunks, only 2 B alignment -> 2 B loads
        __local ushort *sm2 = (__local ushort *)sm1;
        for (int i = t; i < 16 * 105; i += 128) {
            const int r = i / 105, n = g * 16 + r;
            sm2[i] = n < N ? *(__global const ushort *)(W + (size_t)n * row_bytes + (size_t)j * 210 + (i - r * 105) * 2) : 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    {   // phase 2: sub-block parameters, same products as ggml (no FMA)
        const int r = t / 8, sb = t % 8;
        __local const uchar *b = sm + r * RS;
        if (qtype == 0) {
            pd[r * 17 + sb] = vload_half(0, (__local const half *)(b + sb * 34));
        } else if (qtype == 1) {
            const float dd = vload_half(0, (__local const half *)b), mn = vload_half(1, (__local const half *)b);
            __local const uchar *sc = b + 4;
            const int is = sb;
            const uint scv = is < 4 ? sc[is] & 63 : (sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4);
            const uint mv = is < 4 ? sc[is + 4] & 63 : (sc[is + 4] >> 4) | ((sc[is] >> 6) << 4);
            pd[r * 17 + sb] = dd * (float)scv; pm[r * 9 + sb] = mn * (float)mv;
        } else {
            const float dd = vload_half(0, (__local const half *)(b + 208));
            __local const char *sc = (__local const char *)(b + 192);
            pd[r * 17 + sb] = dd * (float)sc[sb]; pd[r * 17 + 8 + sb] = dd * (float)sc[8 + sb];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // phase 3: lane t computes rows r0 = 2 (t % 8), r0 + 1 at k = t / 8 + 16 i and stores them as one half2, so 8 lanes
    // write one k (32 B) and a SIMD32 thread two full 64 B lines per store.
    int bad = 0;
    const int r0 = 2 * (t % 8);
    const bool z0 = g * 16 + r0 >= N, z1 = g * 16 + r0 + 1 >= N;   // rows n >= N: zeros (not -0)
    __global half *dst = Bp + ((size_t)g * Kp + (size_t)j * 256) * 16 + r0;
    for (int i = 0; i < 16; i++) {
        const int k = t / 8 + 16 * i;
        float2 v;
        if (qtype == 0) {
            const int o = (k / 32) * 34 + 2 + k % 32, p = k / 32;
            v.x = (float)(char)sm[r0 * RS + o] * pd[r0 * 17 + p];
            v.y = (float)(char)sm[(r0 + 1) * RS + o] * pd[(r0 + 1) * 17 + p];
        } else if (qtype == 1) {
            const int grp = k / 64, hi = (k % 64) >= 32, o = 16 + grp * 32 + k % 32, p = 2 * grp + hi, sh = hi ? 4 : 0;
            v.x = pd[r0 * 17 + p] * (float)((sm[r0 * RS + o] >> sh) & 0xF) - pm[r0 * 9 + p];
            v.y = pd[(r0 + 1) * 17 + p] * (float)((sm[(r0 + 1) * RS + o] >> sh) & 0xF) - pm[(r0 + 1) * 9 + p];
        } else {
            const int nn = k / 128, kr = k % 128, qu = kr / 32, l = kr % 32;
            const int ol = nn * 64 + (qu & 1) * 32 + l, oh = 128 + nn * 32 + l, sl = qu >= 2 ? 4 : 0, shh = 2 * qu, p = k / 16;
            v.x = pd[r0 * 17 + p] * (float)((int)(((sm[r0 * RS + ol] >> sl) & 0xF) | (((sm[r0 * RS + oh] >> shh) & 3) << 4)) - 32);
            v.y = pd[(r0 + 1) * 17 + p] *
                  (float)((int)(((sm[(r0 + 1) * RS + ol] >> sl) & 0xF) | (((sm[(r0 + 1) * RS + oh] >> shh) & 3) << 4)) - 32);
        }
        if (z0) v.x = 0;
        if (z1) v.y = 0;
        bad |= !(fabs(v.x) <= 65504.0f) | !(fabs(v.y) <= 65504.0f);
        vstore_half2_rte(v, 0, dst + (size_t)k * 16);
    }
    if (bad) atomic_or(oor, 1);
}
#pragma OPENCL FP_CONTRACT ON

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
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define INV_T double
#else
#define INV_T float   // no FP64 (Gen12LP, docs/gen12lp.md section 2): the block inverses in FP32
#endif
#define INV_L(i, k) ((i) < n && (k) < n ? (INV_T)L[(size_t)(i) * n + (k)] : (INV_T)((i) == (k)))
#define INV_X(X, i, k) X[(size_t)((i) / nb) * nb * nb + (size_t)((i) % nb) * nb + (k) % nb]
// Base blocks 32×32: work-item = column j of block g, forward substitution.
__kernel void trsm_inv0_h(__global const half *L, int n, int nb, __global INV_T *X) {
    const int j = get_global_id(0), g0 = get_global_id(1) * 32;
    for (int i = j; i < 32; i++) {
        INV_T s = i == j ? 1.0 : 0.0;
        for (int k = j; k < i; k++) s -= INV_L(g0 + i, g0 + k) * INV_X(X, g0 + k, g0 + j);
        INV_X(X, g0 + i, g0 + j) = s / INV_L(g0 + i, g0 + i);
    }
}
// Level h (pairs of h×h blocks into a 2h block): step 1: Tm = L21·X11 (Tm in place of X21 in the Tm buffer),
// step 2: X21 = −X22·Tm. Grid (h, h, number of pairs).
__kernel void trsm_inv1_h(__global const half *L, int n, int nb, __global INV_T *X, __global INV_T *Tm, int h) {
    const int j = get_global_id(0), i = get_global_id(1), r0 = get_global_id(2) * 2 * h;
    INV_T s = 0.0;
    for (int k = j; k < h; k++) s += INV_L(r0 + h + i, r0 + k) * INV_X(X, r0 + k, r0 + j);
    INV_X(Tm, r0 + h + i, r0 + j) = s;
}
__kernel void trsm_inv2_h(__global const half *L, int n, int nb, __global INV_T *X, __global INV_T *Tm, int h) {
    const int j = get_global_id(0), i = get_global_id(1), r0 = get_global_id(2) * 2 * h;
    INV_T s = 0.0;
    for (int k = 0; k <= i; k++) s += INV_X(X, r0 + h + i, r0 + h + k) * INV_X(Tm, r0 + h + k, r0 + j);
    INV_X(X, r0 + h + i, r0 + j) = -s;
}
// Result: out (block bi from bi·nb·nb, ld b = min(nb, n − bi·nb)), upper triangle = 0. Grid (nb, nb, nbl).
__kernel void trsm_inv3_h(int n, int nb, __global const INV_T *X, __global half *out) {
    const int j = get_global_id(0), i = get_global_id(1), bi = get_global_id(2), b = min(nb, n - bi * nb);
    if (i >= b || j >= b) return;
    out[(size_t)bi * nb * nb + (size_t)i * b + j] = j <= i ? (half)X[(size_t)bi * nb * nb + (size_t)i * nb + j] : (half)0;
}

// ---------------- ggml backend: per-layer elementwise ops (RMS_NORM(+MUL), ADD / MUL by a row, ROPE) ----------------
// F32 in and out, host memory (userptr); formulas and operation order as ggml-cpu (ops.cpp) so results differ only by
// summation order. No private arrays (kcl has no scratch space: lib/gen_kernels.py rejects such kernels).
#pragma OPENCL FP_CONTRACT OFF
// y = x / sqrt(mean(x^2) + eps) [* w]: one work-group of 128 per row (grid (128, M)); w = 0: no weight.
// ggml-cpu: sum in double, mean and scale in float, y = (x * scale) * w.
__kernel __attribute__((reqd_work_group_size(128, 1, 1)))
void gen9blas_rms_norm(__global const float *x, int ldx, __global const float *w, int has_w, float eps,
                       __global float *y, int ldy, int K) {
    __local float part[128];
    const int t = get_local_id(0);
    const size_t r = get_global_id(1);
    __global const float *xr = x + r * ldx;
    float s = 0.0f;
    for (int i = t; i < K; i += 128) { const float v = xr[i]; s += v * v; }
    part[t] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int h = 64; h > 0; h >>= 1) {
        if (t < h) part[t] += part[t + h];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float mean = part[0] / (float)K, scale = 1.0f / sqrt(mean + eps);
    __global float *yr = y + r * ldy;
    if (has_w) for (int i = t; i < K; i += 128) yr[i] = xr[i] * scale * w[i];
    else       for (int i = t; i < K; i += 128) yr[i] = xr[i] * scale;
}
// c = a + b (op 0) or a * b (op 1), M rows of K; b is one row (ldb = 0, ggml broadcast) or M rows. Grid (K, M).
// bs0: element stride of b within a row (1; 0 = one value per row, ggml broadcast of a column, e.g. a conv bias).
__kernel void gen9blas_binop_rows(__global const float *a, int lda, __global const float *b, int ldb,
                                  __global float *c, int ldc, int op, int bs0) {
    const size_t i = get_global_id(0), r = get_global_id(1);
    const float u = a[r * lda + i], v = b[r * ldb + i * bs0];
    c[r * ldc + i] = op ? u * v : u + v;
}
// ggml ROPE (modes NORMAL 0 and NEOX 2, n_dims = ne0, no offset), forward. Element (i, h, t) at t * s2 + h * s1 + i
// (floats). One work-item per (pair j < n_dims/2, token t); the angle is computed once and applied to all nh heads.
// Angle as ggml_rope_cache_init: theta = pos, multiplied j times by theta_scale (float), divided by the freq factor,
// then rope_yarn (ext_factor, corr_dims, mscale = attn_factor). Grid (n_dims/2, n_tok).
__kernel void gen9blas_rope(__global const float *x, long xs1, long xs2, __global float *y, long ys1, long ys2,
                            __global const int *pos, __global const float *ff, int has_ff, int nh, int n_dims, int neox,
                            float theta_scale, float freq_scale, float ext_factor, float attn_factor,
                            float corr0, float corr1) {
    const int j = get_global_id(0), t = get_global_id(1);
    float theta = (float)pos[t];
    for (int k = 0; k < j; k++) theta *= theta_scale;
    const float theta_extrap = has_ff ? theta / ff[j] : theta;
    const float theta_interp = freq_scale * theta_extrap;
    float th = theta_interp, mscale = attn_factor;
    if (ext_factor != 0.0f) {
        const float yv = ((float)j - corr0) / fmax(0.001f, corr1 - corr0);   // rope_yarn_ramp with i0 = 2j
        const float ramp_mix = (1.0f - fmin(1.0f, fmax(0.0f, yv))) * ext_factor;
        th = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    const float c = cos(th) * mscale, s = sin(th) * mscale;
    const int i0 = neox ? j : 2 * j, i1 = neox ? j + n_dims / 2 : 2 * j + 1;
    for (int h = 0; h < nh; h++) {
        __global const float *xp = x + t * xs2 + h * xs1;
        __global float *yp = y + t * ys2 + h * ys1;
        const float x0 = xp[i0], x1 = xp[i1];
        yp[i0] = x0 * c - x1 * s;
        yp[i1] = x0 * s + x1 * c;
    }
}
// ggml ROPE, mode GGML_ROPE_TYPE_MROPE (ggml_rope_multi; PaddleOCR-VL / ERNIE text model): as gen9blas_rope in NEOX pair
// order (i, i + n_dims/2), but pair j takes its position from sector j % (s0+s1+s2+s3): t (pos[t]) for sector < s0, h
// (pos[t + n_tok]) below s0+s1, w (pos[t + 2 n_tok]) below s0+s1+s2, e (pos[t + 3 n_tok]) after; theta = p * theta_scale^j
// by repeated multiplication as ggml_mrope_cache_init (no reset per section: indep_sects is false for MROPE).
// Mode VISION (ggml_rope_multi with GGML_ROPE_TYPE_VISION; mtmd's Qwen2-VL-style encoders, PaddleOCR-VL): indep = 1, the
// running thetas are reset at section starts exactly as ggml's else-if chain does (empty sections included), pairs (j, j + pairoff)
// with pairoff = n_dims = ne0 / 2, so all ne0 elements are rotated. The grid's first dimension is the number of pairs.
__kernel void gen9blas_rope_multi(__global const float *x, long xs1, long xs2, __global float *y, long ys1, long ys2,
                                  __global const int *pos, __global const float *ff, int has_ff, int nh, int n_dims, int n_tok,
                                  int s0, int s1, int s2, int s3, float theta_scale, float freq_scale, float ext_factor,
                                  float attn_factor, float corr0, float corr1, int pairoff, int indep) {
    const int j = get_global_id(0), t = get_global_id(1);
    const int sd = s0 + s1 + s2 + s3, sector = j % sd;
    const int sec = sector < s0 ? 0 : sector < s0 + s1 ? 1 : sector < s0 + s1 + s2 ? 2 : 3;
    // the four running thetas of ggml_mrope_cache_init, replayed up to pair j (with its else-if reset chain when indep)
    const float p0 = (float)pos[t], p1 = (float)pos[t + n_tok], p2 = (float)pos[t + 2 * n_tok], p3 = (float)pos[t + 3 * n_tok];
    float th0 = p0, th1 = p1, th2 = p2, th3 = p3;
    for (int k = 0; k < j; k++) {
        if (indep) {
            const int sk = k % sd;
            if (sk == 0) th0 = p0; else if (sk == s0) th1 = p1; else if (sk == s0 + s1) th2 = p2; else if (sk == s0 + s1 + s2) th3 = p3;
        }
        th0 *= theta_scale; th1 *= theta_scale; th2 *= theta_scale; th3 *= theta_scale;
    }
    if (indep) {
        if (sector == 0) th0 = p0; else if (sector == s0) th1 = p1; else if (sector == s0 + s1) th2 = p2; else if (sector == s0 + s1 + s2) th3 = p3;
    }
    const float theta = sec == 0 ? th0 : sec == 1 ? th1 : sec == 2 ? th2 : th3;
    const float theta_extrap = has_ff ? theta / ff[j] : theta;
    const float theta_interp = freq_scale * theta_extrap;
    float th = theta_interp, mscale = attn_factor;
    if (ext_factor != 0.0f) {
        const float yv = ((float)j - corr0) / fmax(0.001f, corr1 - corr0);
        const float ramp_mix = (1.0f - fmin(1.0f, fmax(0.0f, yv))) * ext_factor;
        th = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    const float c = cos(th) * mscale, s = sin(th) * mscale;
    const int i0 = j, i1 = j + pairoff;
    for (int h = 0; h < nh; h++) {
        __global const float *xp = x + t * xs2 + h * xs1;
        __global float *yp = y + t * ys2 + h * ys1;
        const float x0 = xp[i0], x1 = xp[i1];
        yp[i0] = x0 * c - x1 * s;
        yp[i1] = x0 * s + x1 * c;
    }
}
// ggml CPY / CONT / DUP between F32 and F16 (tin, tout: 0 = F32, 1 = F16) with any byte strides (views, permutes) and
// different shapes of the same element count (ggml_cont_Nd): dst element (i0, i1, i2, i3) takes the source element with
// the same row-major linear index. F32 -> F16 rounds to nearest even (vstore_half_rte, = ggml_cpu_fp32_to_fp16);
// same type: bit copy. same: the shapes are equal (no index division). Grid (dne0, dne1, dne2 * dne3).
__kernel void gen9blas_cpy_any(__global const uchar *src, long sb0, long sb1, long sb2, long sb3, int sne0, int sne1, int sne2,
                               __global uchar *dst, long db0, long db1, long db2, long db3, int dne0, int dne1, int dne2,
                               int tin, int tout, int same) {
    const int i0 = get_global_id(0), i1 = get_global_id(1), p = get_global_id(2);
    const int i2 = p % dne2, i3 = p / dne2;
    int j0 = i0, j1 = i1, j2 = i2, j3 = i3;
    if (!same) {
        long l = (((long)i3 * dne2 + i2) * dne1 + i1) * dne0 + i0;
        j0 = (int)(l % sne0); l /= sne0; j1 = (int)(l % sne1); l /= sne1; j2 = (int)(l % sne2); j3 = (int)(l / sne2);
    }
    __global const uchar *sp = src + j0 * sb0 + j1 * sb1 + j2 * sb2 + j3 * sb3;
    __global uchar *dp = dst + i0 * db0 + i1 * db1 + i2 * db2 + i3 * db3;
    if (tin == tout) {
        if (tin) *(__global ushort *)dp = *(__global const ushort *)sp;
        else *(__global uint *)dp = *(__global const uint *)sp;
    } else if (tout) vstore_half_rte(*(__global const float *)sp, 0, (__global half *)dp);
    else *(__global float *)dp = vload_half(0, (__global const half *)sp);
}
// ggml UPSCALE / ggml_interpolate, F32, modes as ggml-cpu: 0 nearest, 1 bilinear, 1 + antialias (aa) (triangle filter,
// PyTorch antialias=True), align corners (ac). Byte strides for source and destination. Grid (ne0, ne1, ne2 * ne3).
__kernel void gen9blas_upscale_f32(__global const uchar *src, long sb0, long sb1, long sb2, long sb3, int sne0, int sne1,
                                   __global uchar *dst, long db0, long db1, long db2, long db3, int ne2,
                                   float sf0, float sf1, float sf2, float sf3, int mode, int aa, float pixel_offset) {
    const int i0 = get_global_id(0), i1 = get_global_id(1), p = get_global_id(2);
    const int i2 = p % ne2, i3 = p / ne2;
    const long i02 = (long)(i2 / sf2), i03 = (long)(i3 / sf3);
    __global const uchar *s23 = src + i02 * sb2 + i03 * sb3;
    float val;
    if (mode == 0) {
        val = *(__global const float *)(s23 + (long)(i0 / sf0) * sb0 + (long)(i1 / sf1) * sb1);
    } else if (aa) {
        const float support1 = fmax(1.0f, 1.0f / sf1), invscale1 = 1.0f / support1;
        const float support0 = fmax(1.0f, 1.0f / sf0), invscale0 = 1.0f / support0;
        const float yy = ((float)i1 + pixel_offset) / sf1, xx = ((float)i0 + pixel_offset) / sf0;
        const long x_min = max((long)(xx - support0 + pixel_offset), 0L), x_max = min((long)(xx + support0 + pixel_offset), (long)sne0);
        const long y_min = max((long)(yy - support1 + pixel_offset), 0L), y_max = min((long)(yy + support1 + pixel_offset), (long)sne1);
        float acc = 0.0f, tw = 0.0f;
        for (long sy = y_min; sy < y_max; sy++) {
            const float wy = fmax(1.0f - fabs(((float)sy - yy + pixel_offset) * invscale1), 0.0f);
            for (long sx = x_min; sx < x_max; sx++) {
                const float wx = fmax(1.0f - fabs(((float)sx - xx + pixel_offset) * invscale0), 0.0f);
                const float w = wx * wy;
                if (w <= 0.0f) continue;
                acc += *(__global const float *)(s23 + sx * sb0 + sy * sb1) * w;
                tw += w;
            }
        }
        val = tw > 0.0f ? acc / tw : acc;
    } else {
        const float yy = ((float)i1 + pixel_offset) / sf1 - pixel_offset, xx = ((float)i0 + pixel_offset) / sf0 - pixel_offset;
        long y0 = (long)floor(yy), x0 = (long)floor(xx), y1 = y0 + 1, x1 = x0 + 1;
        y0 = max(0L, min(y0, (long)sne1 - 1)); y1 = max(0L, min(y1, (long)sne1 - 1));
        x0 = max(0L, min(x0, (long)sne0 - 1)); x1 = max(0L, min(x1, (long)sne0 - 1));
        const float dy = fmax(0.0f, fmin(yy - (float)y0, 1.0f)), dx = fmax(0.0f, fmin(xx - (float)x0, 1.0f));
        const float a = *(__global const float *)(s23 + x0 * sb0 + y0 * sb1), b = *(__global const float *)(s23 + x1 * sb0 + y0 * sb1);
        const float c = *(__global const float *)(s23 + x0 * sb0 + y1 * sb1), d = *(__global const float *)(s23 + x1 * sb0 + y1 * sb1);
        val = a * (1 - dx) * (1 - dy) + b * dx * (1 - dy) + c * (1 - dx) * dy + d * dx * dy;
    }
    *(__global float *)(dst + i0 * db0 + i1 * db1 + i2 * db2 + i3 * db3) = val;
}
// Attention for decoding several KV streams (llama-server -np N: one or a few new tokens per stream), D = 128, G <= 8
// query heads per KV head. K / V are read in place from llama.cpp's cache layout (row = key, [HKV][D] halves per row, any
// row / head / stream strides): no transposed copy (the streams kernel of attn.cl transposes all of K in every step).
// Split over keys (flash-decoding): work-group = one SIMD16 subgroup for (stream s, token t, KV head g, head group hg of
// DEC_HG query heads, chunk c of `chunk` keys); writes per query head (m, l, acc[D]) in the exp2 domain to part[];
// gen9blas_attn_dec_comb merges the chunks. 2 heads per work-group keep the accumulators in registers (8 and 4 spilled
// 8 / 2 KB to scratch); the head groups of one chunk run side by side and share its K/V rows through L3.
// Q K^T: lane = 8 dims, each K row read whole (16 lanes x 16 B, one 256 B line run; lane = key with per-lane rows was
// 3.5 ms for 12 streams at 2000 keys: 16 lines per message), dot products reduced over the subgroup; Q in registers,
// scaled by qscale = scale * log2 e. P V: lane = 8 output dims, p broadcast from the key lanes. FP32 accumulation.
// Token t of stream s sees keys [0, vis[s] + dvis * t) (clamped to nk); the keys past it are never read.
#define DEC_D 128
#define DEC_HG 2
__kernel __attribute__((reqd_work_group_size(16, 1, 1))) __attribute__((intel_reqd_sub_group_size(16)))
void gen9blas_attn_dec_part(__global const float *Q, long qt, long qh, long qs, __global const half *K, long kk, long kh, long ks,
                            __global const half *V, long vk, long vh, long vs, __global const int *vis, int dvis, int nq,
                            int HKV, int G, int nk, int chunk, int nchunk, float qscale, __global float *part) {
    const int ngr = (G + DEC_HG - 1) / DEC_HG;   // head groups of one chunk are neighbours in the grid (shared K/V in L3)
    const int lane = get_local_id(0), hg = get_group_id(1) % ngr, c = get_group_id(1) / ngr, idx = get_group_id(2);
    const int g = idx % HKV, t = (idx / HKV) % nq, s = idx / (HKV * nq);
    const int h0 = g * G + hg * DEC_HG, nh = min(DEC_HG, G - hg * DEC_HG);
    int vt = vis[s] + dvis * t; vt = vt < nk ? vt : nk;
    const int k0 = c * chunk, k1 = min(k0 + chunk, vt);
    // part: [s][t][h (all H heads of the stream's token)][chunk][2 + D]
    __global float *pp = part + (((long)(s * nq + t) * (HKV * G) + h0) * nchunk + c) * (2 + DEC_D);
    const long hstride = (long)nchunk * (2 + DEC_D);
    if (k0 >= k1) {   // chunk past the visible keys: l = 0 marks it empty for the merge
        if (lane < nh) { pp[lane * hstride] = -INFINITY; pp[lane * hstride + 1] = 0.0f; }
        return;
    }
    float8 qv[DEC_HG];   // lane = 8 dims of each head's query, scaled
    for (int hh = 0; hh < DEC_HG; hh++)
        qv[hh] = hh < nh ? vload8(0, Q + (long)s * qs + (long)t * qt + (long)(h0 + hh) * qh + lane * 8) * qscale : (float8)(0.0f);
    float m[DEC_HG], l[DEC_HG];
    float8 acc[DEC_HG];
    for (int hh = 0; hh < DEC_HG; hh++) { m[hh] = -INFINITY; l[hh] = 0.0f; acc[hh] = (float8)(0.0f); }
    __global const half *Kb = K + (long)s * ks + (long)g * kh + lane * 8, *Vb = V + (long)s * vs + (long)g * vh + lane * 8;
    for (int kb = k0; kb < k1; kb += 16) {
        const int nv = min(16, k1 - kb);
        float sc[DEC_HG];   // lane j: score of key kb + j
        for (int hh = 0; hh < DEC_HG; hh++) sc[hh] = 0.0f;
        for (int j = 0; j < 16; j++) {   // K row j read whole (16 lanes x 16 B), dot per head reduced over the lanes
            if (j >= nv) break;
            const float8 kr = vload_half8(0, Kb + (long)(kb + j) * kk);
            for (int hh = 0; hh < DEC_HG; hh++) {
                const float r = sub_group_reduce_add(dot(kr.lo, qv[hh].lo) + dot(kr.hi, qv[hh].hi));
                if (lane == j) sc[hh] = r;
            }
        }
        float p[DEC_HG];
        for (int hh = 0; hh < DEC_HG; hh++) {
            const float x = lane < nv ? sc[hh] : -INFINITY;
            const float mn = fmax(m[hh], sub_group_reduce_max(x));
            const float a = exp2(m[hh] - mn);
            p[hh] = lane < nv ? exp2(x - mn) : 0.0f;
            l[hh] = l[hh] * a + sub_group_reduce_add(p[hh]);
            acc[hh] *= a;
            m[hh] = mn;
        }
        for (int j = 0; j < 16; j++) {
            if (j >= nv) break;
            const float8 vv = vload_half8(0, Vb + (long)(kb + j) * vk);
            for (int hh = 0; hh < DEC_HG; hh++) acc[hh] += sub_group_broadcast(p[hh], j) * vv;
        }
    }
    for (int hh = 0; hh < nh; hh++) {
        __global float *ph = pp + hh * hstride;
        if (lane == 0) { ph[0] = m[hh]; ph[1] = l[hh]; }
        vstore8(acc[hh], 0, ph + 2 + lane * 8);
    }
}
// Merge of gen9blas_attn_dec_part's chunks: O[s][t][h][d] (llama.cpp's FLASH_ATTN_EXT output [D, H, nq, ns], contiguous)
// = sum_c acc_c 2^(m_c - M) / sum_c l_c 2^(m_c - M) over the non-empty chunks. Grid (D, H, ns * nq).
__kernel void gen9blas_attn_dec_comb(__global const float *part, __global float *O, int H, int nchunk) {
    const int d = get_global_id(0), h = get_global_id(1), st = get_global_id(2);
    __global const float *pb = part + ((long)st * H + h) * nchunk * (2 + DEC_D);
    float M = -INFINITY;
    for (int c = 0; c < nchunk; c++) { __global const float *pc = pb + (long)c * (2 + DEC_D); if (pc[1] > 0.0f) M = fmax(M, pc[0]); }
    float L = 0.0f, a = 0.0f;
    for (int c = 0; c < nchunk; c++) {
        __global const float *pc = pb + (long)c * (2 + DEC_D);
        if (pc[1] > 0.0f) { const float w = exp2(pc[0] - M); L += pc[1] * w; a += pc[2 + d] * w; }
    }
    O[((long)st * H + h) * DEC_D + d] = a / L;
}
// ggml SET_ROWS, F32 rows -> F16 rows (the KV-cache writes of llama.cpp): dst row idx[i] of plane (i02, i03) = src row i,
// rounded to nearest even (vstore_half_rte, = ggml_cpu_fp32_to_fp16). Strides in elements (src floats, dst halves), index
// strides in bytes; idx64: indices are int64 (else int32); indices broadcast over planes as ggml-cpu (i11 = i02 % ne11,
// i12 = i03 % ne12). An index outside [0, ne1) sets *bad and writes nothing. Grid (nc, nr, ne02 * ne03).
__kernel void gen9blas_set_rows_f16(__global const float *src, long s1, long s2, long s3, __global const uchar *idx,
                                    long b10, long b11, long b12, int idx64, int ne11, int ne12, int ne02,
                                    __global half *dst, long d1, long d2, long d3, long ne1, __global int *bad) {
    const int c = get_global_id(0), i = get_global_id(1), p = get_global_id(2);
    const int i02 = p % ne02, i03 = p / ne02;
    __global const uchar *ip = idx + i * b10 + (i02 % ne11) * b11 + (i03 % ne12) * b12;
    const long i1 = idx64 ? *(__global const long *)ip : (long)*(__global const int *)ip;
    if (i1 < 0 || i1 >= ne1) { if (c == 0) atomic_or(bad, 1); return; }
    vstore_half_rte(src[i * s1 + i02 * s2 + i03 * s3 + c], 0, dst + i1 * d1 + i02 * d2 + i03 * d3 + c);
}
// ggml NORM (LayerNorm without affine) [* w] [+ b], one work-group of 128 per row (grid (128, M)), as ggml-cpu:
// mean = sum / K, variance = sum((x - mean)^2) / K, y = (x - mean) * (1 / sqrt(variance + eps)), then * w, then + b.
__kernel __attribute__((reqd_work_group_size(128, 1, 1)))
void gen9blas_layer_norm(__global const float *x, int ldx, __global const float *w, int has_w, __global const float *b,
                         int has_b, float eps, __global float *y, int ldy, int K) {
    __local float part[128];
    const int t = get_local_id(0);
    const size_t r = get_global_id(1);
    __global const float *xr = x + r * ldx;
    float s = 0.0f;
    for (int i = t; i < K; i += 128) s += xr[i];
    part[t] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int h = 64; h > 0; h >>= 1) { if (t < h) part[t] += part[t + h]; barrier(CLK_LOCAL_MEM_FENCE); }
    const float mean = part[0] / (float)K;
    barrier(CLK_LOCAL_MEM_FENCE);
    s = 0.0f;
    for (int i = t; i < K; i += 128) { const float d = xr[i] - mean; s += d * d; }
    part[t] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int h = 64; h > 0; h >>= 1) { if (t < h) part[t] += part[t + h]; barrier(CLK_LOCAL_MEM_FENCE); }
    const float scale = 1.0f / sqrt(part[0] / (float)K + eps);
    __global float *yr = y + r * ldy;
    for (int i = t; i < K; i += 128) {
        float v = (xr[i] - mean) * scale;
        if (has_w) v = v * w[i];
        if (has_b) v = v + b[i];
        yr[i] = v;
    }
}
// gen9blas_layer_norm that also writes its output as the FP16 A3 tiles of the next GEMM (what pk_a3_f32v would make of
// y): the GEMMs reading this norm skip their own packing. Same arithmetic for y; FP16 by convert_half_rte as
// pk_a3_f32v; rows M..Mp-1 and columns K..Kp-1 of the tiles are zero. Grid (128, Mp).
// A3 tile (16 x 16, 256 halves, tile (m / 16, k / 16) at ((m / 16) * (Kp / 16) + k / 16) * 256): element (mi, ki) of the
// tile sits in 8-half chunk c = (ki / 4) * 8 + (mi / 8) * 4 + (ki % 2 == 0) * 2 + (mi / 4) % 2, position
// (mi % 4) * 2 + (ki % 4) / 2 (the inverse of pk_a3_f32v's store loop).
static inline size_t a3_index(int m, int k, int Kp) {
    const int mi = m & 15, ki = k & 15;
    const int c = (ki >> 2) * 8 + (mi >> 3) * 4 + ((ki & 1) == 0) * 2 + ((mi >> 2) & 1);
    return ((size_t)(m >> 4) * (Kp >> 4) + (k >> 4)) * 256 + c * 8 + (mi & 3) * 2 + ((ki & 3) >> 1);
}
__kernel __attribute__((reqd_work_group_size(128, 1, 1)))
void gen9blas_layer_norm_pack(__global const float *x, int ldx, __global const float *w, int has_w, __global const float *b,
                              int has_b, float eps, __global float *y, int ldy, int K, __global half *Ap, int Kp, int M) {
    __local float part[128];
    const int t = get_local_id(0);
    const int r = get_global_id(1);
    if (r >= M) {   // padding row of the last 16-row band
        for (int i = t; i < Kp; i += 128) Ap[a3_index(r, i, Kp)] = (half)0.0f;
        return;     // the whole work-group (one row) takes this branch: no barrier is skipped unevenly
    }
    __global const float *xr = x + (size_t)r * ldx;
    float s = 0.0f;
    for (int i = t; i < K; i += 128) s += xr[i];
    part[t] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int h = 64; h > 0; h >>= 1) { if (t < h) part[t] += part[t + h]; barrier(CLK_LOCAL_MEM_FENCE); }
    const float mean = part[0] / (float)K;
    barrier(CLK_LOCAL_MEM_FENCE);
    s = 0.0f;
    for (int i = t; i < K; i += 128) { const float d = xr[i] - mean; s += d * d; }
    part[t] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int h = 64; h > 0; h >>= 1) { if (t < h) part[t] += part[t + h]; barrier(CLK_LOCAL_MEM_FENCE); }
    const float scale = 1.0f / sqrt(part[0] / (float)K + eps);
    __global float *yr = y + (size_t)r * ldy;
    for (int i = t; i < Kp; i += 128) {
        float v = 0.0f;
        if (i < K) {
            v = (xr[i] - mean) * scale;
            if (has_w) v = v * w[i];
            if (has_b) v = v + b[i];
            yr[i] = v;
        }
        Ap[a3_index(r, i, Kp)] = convert_half_rte(v);
    }
}
// ggml GELU (tanh form) as ggml-cpu with GGML_GELU_FP16: x <= -10 -> 0, x >= 10 -> x, otherwise the table value
// fp16(gelu(fp16(x))). n elements, contiguous. Grid (n).
__kernel void gen9blas_gelu(__global const float *x, __global float *y) {
    const size_t i = get_global_id(0);
    y[i] = epi_gelu(x[i]);   // shared with gen9blas_epilog_hf8x (bit-identical fused and unfused paths)
}
// F32 rows of H heads x D (row r, head h at x + r * s_row + h * s_head, strides in floats) -> FP16 [r][h][d] contiguous,
// round to nearest even, saturated to +-65504 (no Inf: a key/value beyond FP16 range is clamped, not turned into NaN).
// Grid (D, H, rows). For attention inputs given as F32 views (NeMo-Speech.cpp: RoPE output / fused QKV projection).
__kernel void gen9blas_cvt_f16_heads(__global const float *x, long s_row, long s_head, __global half *y) {
    const size_t d = get_global_id(0), h = get_global_id(1), r = get_global_id(2), D = get_global_size(0), H = get_global_size(1);
    const float v = clamp(x[r * s_row + h * s_head + d], -65504.0f, 65504.0f);
    vstore_half_rte(v, (r * H + h) * D + d, y);
}
// ggml IM2COL as ggml-cpu (ggml_compute_forward_im2col_f32/_f16): src [N, IC, IH, IW] (rows of IW contiguous; image n at
// x + n * ofs0, channel c at + c * ofs1, offsets in floats) -> dst [N, OH, OW, IC*KH*KW], zero padding; f16: dst in FP16
// (RTE, as GGML_CPU_FP32_TO_FP16). 1-D: IH = KH = OH = 1. One work-item per output element, grid (IC*KH*KW, OW, N*OH).
__kernel void gen9blas_im2col(__global const float *x, long ofs0, long ofs1, int IH, int IW, int KH, int KW, int OH,
                              int s0, int s1, int p0, int p1, int d0, int d1, int f16, __global float *yf) {
    const int k = get_global_id(0), ow = get_global_id(1), nh = get_global_id(2);
    const int K = get_global_size(0), OW = get_global_size(1);
    const int n = nh / OH, oh = nh % OH, KK = KH * KW, c = k / KK, kh = (k % KK) / KW, kw = k % KW;
    const int iw = ow * s0 + kw * d0 - p0, ih = oh * s1 + kh * d1 - p1;
    const float v = (ih < 0 || ih >= IH || iw < 0 || iw >= IW) ? 0.0f : x[n * ofs0 + c * ofs1 + (long)ih * IW + iw];
    const size_t o = ((size_t)nh * OW + ow) * K + k;
    if (f16) vstore_half_rte(v, o, (__global half *)yf); else yf[o] = v;
}
// ggml RELU: y = max(x, 0) (x > 0 ? x : 0, as ggml_vec_relu_f32). n elements, contiguous. Grid (n).
__kernel void gen9blas_relu(__global const float *x, __global float *y) {
    const size_t i = get_global_id(0);
    const float v = x[i];
    y[i] = v > 0.0f ? v : 0.0f;
}
// ggml GELU_ERF as ggml-cpu (ggml_vec_gelu_erf_f32): 0.5 x (1 + erf(x / sqrt 2)), F32. n elements, contiguous. Grid (n).
__kernel void gen9blas_gelu_erf(__global const float *x, __global float *y) {
    const size_t i = get_global_id(0);
    const float v = x[i];
    y[i] = 0.5f * v * (1.0f + erf(v * 0.70710678118654752440f));
}
#pragma OPENCL FP_CONTRACT ON

// ggml backend, small batches (several sequences decoded together, kp-docsearch PaddleOCR-VL): C[m][n] = sum_k A[m][k] *
// B[k][n] for M <= MT rows of F32 activations against GEN9BLAS_HS packed weights (PK_IDX_B16: panels of 16 columns, for
// every k the 16 halves of the panel contiguous), FP32 products and sums, F32 result written straight into C (no packing
// of A, no FP16 intermediate, no epilogue kernel). Work-group = SKP_S SIMD16 subgroups on one panel, lane = column; subgroup
// s takes the 16-k steps s, s + S, ... (two 256 B block reads per step, S x 512 B contiguous per group step: the
// contiguity lesson of docs/skinny-gemm.md (e)); partial sums reduced in local memory by subgroup m for row m.
// Rows of A past K read as 0 (the packed B is zero there anyway). *nonfinite |= 1 for an inf / NaN result.
// Grid: (Np / 16) x (16 * SKP_S), local 16 * SKP_S.
#define SKP_S 16
#define SKP_ROW(m) if ((m) < M) { const float a = k + l < K ? A[(size_t)(m) * lda + k + l] : 0.0f; \
    _Pragma("unroll") for (int j = 0; j < 8; j++) acc##m = fma(sub_group_broadcast(a, j), w0[j], acc##m); \
    _Pragma("unroll") for (int j = 0; j < 8; j++) acc##m = fma(sub_group_broadcast(a, 8 + j), w1[j], acc##m); }
#define SKP_DECL(m) float acc##m = 0.0f;
#define SKP_PART(m) part[s][m][l] = acc##m;
#define SKP_R4(X) X(0) X(1) X(2) X(3)
#define SKP_R8(X) SKP_R4(X) X(4) X(5) X(6) X(7)
#define SKP_R16(X) SKP_R8(X) X(8) X(9) X(10) X(11) X(12) X(13) X(14) X(15)
#define SKINNY_PB16(MT) \
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16 * SKP_S, 1, 1))) \
__kernel void gen9blas_skinny_pb16_##MT(__global const float *A, int lda, int M, int K, __global const ushort *Bp, int Kp, \
                                         __global float *C, int ldc, int N, __global int *nonfinite) { \
    __local float part[SKP_S][MT][16]; \
    const int p = get_group_id(0), s = get_sub_group_id(), l = get_sub_group_local_id(); \
    const __global ushort *bp = Bp + (size_t)p * Kp * 16; \
    SKP_R##MT(SKP_DECL) \
    for (int k = s * 16; k < Kp; k += SKP_S * 16) { \
        const float8 w0 = convert_float8(as_half8(intel_sub_group_block_read_us8(bp + (size_t)k * 16))); \
        const float8 w1 = convert_float8(as_half8(intel_sub_group_block_read_us8(bp + (size_t)k * 16 + 128))); \
        SKP_R##MT(SKP_ROW) \
    } \
    SKP_R##MT(SKP_PART) \
    barrier(CLK_LOCAL_MEM_FENCE); \
    const int n = p * 16 + l; \
    if (s < MT && s < M && n < N) { \
        float v = 0.0f; \
        _Pragma("unroll") for (int i = 0; i < SKP_S; i++) v += part[i][s][l]; \
        if (!isfinite(v)) atomic_or(nonfinite, 1); \
        C[(size_t)s * ldc + n] = v; \
    } \
}
SKINNY_PB16(4)
SKINNY_PB16(8)
SKINNY_PB16(16)
