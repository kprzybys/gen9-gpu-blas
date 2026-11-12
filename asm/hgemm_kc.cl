// HGEMM C = A·B (row-major, FP16, FP16 accumulation) for UHD 620. Template version in OpenCL C.
// One SIMD16 thread computes a C tile: 16 rows × 32 columns (2 groups of 16 lanes).
// A, B are read as 2D images (CL_R, CL_HALF_FLOAT) via media block read (intel_sub_group_block_read_us8):
//   block of 32 B × 8 rows = 16 lanes × 8 rows of halfs, one send message.
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

// 2D grid: (N/32 × 16, M/16), tile order is the one in which the hardware dispatches work-groups (row by row).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void hgemm_16x32(__read_only image2d_t A, __read_only image2d_t B, __global half *C,
                          int M, int ldc, int K) {
    tile_16x32(A, B, C, ldc, K, ldc, get_global_id(1) * 16, get_group_id(0) * 32);
}

// 1D grid: one tile per subgroup, grouped order (as in Triton): consecutive tiles go down a band
// of GM tile rows, then to the right. Concurrently active threads compute a compact piece of C,
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

// ---------- Variant with packed matrices (tile data contiguous in memory, read from a buffer) ----------
// Bp: panels of 16 columns: Bp[n/16][k][n%16]. B block (k..k+7, 16 columns) = 8×16 halfs = 256 B contiguous:
//     intel_sub_group_block_read_us8(Bp + ...) gives lane j the values [i*16 + j], i = 0..7, i.e. rows k+i.
// Ap: panels of 16 rows, k blocks of 16: Ap[m/16][k/16][m%16][k%16]. A block (8 rows, 16 k) = 256 B contiguous:
//     lane j = column k+j, element i = row m0+8h+i.
// Reading from a buffer is an OWord block read (2 messages of 128 B per 256 B block).
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

// Layout A2 for the hand-written assembly loop (no GRF bank conflicts):
// k block of 16, in it 2 halves (kk 0..7, 8..15) of 256 B = 8 GRF registers each.
// GRF g of half h: rows 4*(g/2)..+3; odd g -> even kl (0,2,4,6), even g -> odd kl.
// Element e = (r%4)*4 + kl/2. When the B registers (row kl) have the parity of kl, the A register has the opposite parity.
__kernel void pack_a2(__global const half *A, __global half *Ap, int K, int lda) {
    const int k = get_global_id(0), m = get_global_id(1);
    const int r = m % 16, kk = k % 16, h = kk / 8, kl = kk % 8;
    const int g = 2 * (r / 4) + ((kl % 2 == 0) ? 1 : 0);
    const size_t block = (size_t)(m / 16) * (K / 16) + k / 16;
    Ap[block * 256 + h * 128 + g * 16 + (r % 4) * 4 + kl / 2] = A[(size_t)m * lda + k];
}

// ---------- More accurate variants (FP16 data, FP16 result, intermediate sum in FP32) ----------
// hgemm_16x32kc: FP16 accumulation in chunks of KC (16 or 32) products, chunks added to an FP32 accumulator.
//                Arithmetic mostly at FP16 rate; the error is limited to the error of a short FP16 sum.
// hgemm_16x32mx: each FP16·FP16 product is added to FP32 (FP32 rate, ~FP32 accuracy).
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

// ---------- Faster packing: one work-item moves 16 consecutive row elements (vload16) ----------
// Input dimensions arbitrary: missing rows/columns up to a multiple of the tile (M%16, N%32, K%16) are filled with zeros.
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
    __global half *block = Ap + ((size_t)(m / 16) * (Kp / 16) + kb) * 256;
    // half h, even kl -> odd GRF g = 2*(r/4)+1, odd kl -> g = 2*(r/4); element (r%4)*4 + kl/2
    __global half *p0 = block + (2 * (r / 4) + 1) * 16 + (r % 4) * 4, *p1 = block + (2 * (r / 4)) * 16 + (r % 4) * 4;
    vstore4(v.s02468ace.lo, 0, p0);         vstore4(v.s13579bdf.lo, 0, p1);
    vstore4(v.s02468ace.hi, 0, p0 + 128);   vstore4(v.s13579bdf.hi, 0, p1 + 128);
}

// pack_a2_g: layout A2, one work-item = one 32-byte output row (GRF register) written contiguously (vstore16).
// Output as an array of rows of 16 halfs: o -> block = o/16 (pm, kb), h = (o%16)/8, g = o%8.
// GRF g: rows r = 4*(g/2) + ri (ri 0..3), kl = 2*j + (g odd ? 0 : 1) (j 0..3), element ri*4 + j.
// Edges (m >= M, k >= K) are filled with zeros. Grid: Mp*Kp/16 work-items.
__kernel void pack_a2_g(__global const half *A, __global half *Ap, int M, int K, int lda, int Kp) {
    const int o = get_global_id(0), block = o / 16, h = (o % 16) / 8, g = o % 8;
    const int pm = block / (Kp / 16), kb = block % (Kp / 16);
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
// pack_a2_pad: grid (Kp, Mp); pack_b_pad: grid (Np, Kp). Out of range (m >= M, k >= K, n >= N) zeros.
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

// ---------- Accurate variant for the hand-written asm loop: 16×16 tile, FP16 sum in chunks of KC2 products -> FP32 ----------
// A3: k16 block = 4 quarters (kl 0..3) of 64 halfs = 4 GRF (128 B). GRF g: rows 8*(g/2)..+7;
//     odd g -> even kl (0,2), even g -> odd kl; element (r%8)*2 + kl/2.
// B: ordinary panels of 16 columns (pack_b_pad); quarter = 4 rows × 32 B = 128 B.
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
// Packing A3 with edges: grid (Kp, Mp); transposed version (A stored K×M): grid (Mp, Kp).
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
