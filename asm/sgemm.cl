// SGEMM C = A·B (FP32, row-major) for UHD 620, same method as hgemm.cl.
// A SIMD16 thread computes a 16×16 C tile (16 float accumulators = 32 GRF). Step k = 16.
// Packed data (contiguous tiles, OWord block read from a buffer):
//   Apf: [m/16][k/16][r 0..15][kk 0..15]: A block 16×16 = 1 KB; block_read8 (uint8) gives 8 rows, lane j = column k+j.
//   Bpf: [n/16][k][n%16]: B block (8 rows k × 16 columns) = 512 B contiguous; lane j = column n0+j.
// Multiplications: acc[r] += A[m0+r][k+kk] (broadcast from lane kk) * B[k+kk][n0+lane], 256 mad (16) :f per step.
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
    const int group = pid / (GM * npn), m_first = group * GM;
    const int gsz = min(npm - m_first, GM);
    const int pm = m_first + (pid % (GM * npn)) % gsz, pn = (pid % (GM * npn)) / gsz;
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

// Layouts for the hand-written asm loop (asm/sgemm_asm.py), no bank conflicts. Step k16 split into 4 quarters (kl 0..3).
// A2f: quarter = 16 rows × 4 kl = 64 floats = 8 GRF (256 B, offset 256*q in the 1 KB block).
//      GRF g: rows 4*(g/2)..+3; odd g -> even kl (0,2), even g -> odd kl; element (r%4)*2 + kl/2.
__kernel void pack_af2(__global const float *A, __global float *Ap, int K, int lda) {
    const int k = get_global_id(0), m = get_global_id(1);
    const int r = m % 16, kk = k % 16, q = kk / 4, kl = kk % 4;
    const int g = 2 * (r / 4) + ((kl % 2 == 0) ? 1 : 0);
    Ap[((size_t)(m / 16) * (K / 16) + k / 16) * 256 + q * 64 + g * 8 + (r % 4) * 2 + kl / 2] = A[(size_t)m * lda + k];
}
// B2f: like Bpf, but in each block of 4 rows the order is 0,2,1,3 (even and odd kl separately, 128 B each).
__kernel void pack_bf2(__global const float *B, __global float *Bp, int K, int ldb) {
    const int n = get_global_id(0), k = get_global_id(1), kl = k % 4;
    Bp[((size_t)(n / 16) * K + (k - kl) + (kl % 2) * 2 + kl / 2) * 16 + n % 16] = B[(size_t)k * ldb + n];
}

// Packing with edges (zeros outside M/N/K) for the library; layouts as pack_af2 / pack_bf2. Grids: (Kp, Mp) and (Np, Kp).
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
__kernel void gen9blas_epilog_f(__global const float *Cp, int ldp, __global float *C, int ldc, float alpha, float beta) {
    const int j = get_global_id(0), i = get_global_id(1);
    float v = alpha * Cp[(size_t)i * ldp + j];
    if (beta != 0.0f) v += beta * C[(size_t)i * ldc + j];
    C[(size_t)i * ldc + j] = v;
}
