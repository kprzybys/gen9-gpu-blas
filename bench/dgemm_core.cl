// DGEMM (FP64) for libgen9blas: packing and the main kernel. The remaining FP64 kernels (epilogs, TRSM, expansion, merging)
// are derived from sgemm.cl by type substitution -- lib/gen_dgemm.py assembles bench/dgemm.cl from it.
// Panel layout (A: r = row m, B: r = column n): P[(r/16)·Kp·16 + k·16 + r%16] -- in a 16-row panel element
// (k, r) is at k·16 + r. Offsets used by the library (a piece starting at row i0 and column k0: i0·Kp + k0·16
// for A, k0·16 for B) are the same as in the FP32 asm layout.
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// Element access (also used by the kernels copied from sgemm.cl below): plain indexing, except on Gen8 Cherryview
// (GEN8_CHV, docs/gen8.md 6a): no A64 QWord messages (doubles as uint pairs), no math.invm/madm (division emulated).
#ifdef GEN8_CHV
#define RD(p, i) as_double(vload2((i), (const __global uint *)(p)))
#define WR(p, i, v) vstore2(as_uint2((double)(v)), (i), (__global uint *)(p))
inline double div_chv(double a, double b) {   // reciprocal from FP32 + 3 Newton steps + quotient correction (DF mul/fma)
    double r = (double)native_recip((float)b);
    r = fma(r, fma(-b, r, 1.0), r); r = fma(r, fma(-b, r, 1.0), r); r = fma(r, fma(-b, r, 1.0), r);
    const double q = a * r;
    return fma(r, fma(-b, q, a), q);
}
#define DIVD(a, b) div_chv(a, b)
#else
#define RD(p, i) ((p)[i])
#define WR(p, i, v) ((p)[i] = (v))
#define DIVD(a, b) ((a) / (b))
#endif
#define RDD(p, i) RD(p, i)
#define WRD(p, i, v) WR(p, i, v)
#define GMD 16
#define PD(r, k, Kp) ((size_t)((r) / 16) * (Kp) * 16 + (size_t)(k) * 16 + (r) % 16)
__kernel void pkd_a(__global const double *A, __global double *Ap, int M, int K, int lda, int Kp, long sA, int Mp) {
    const int k = get_global_id(0), m = get_global_id(1), b = get_global_id(2);
    WR(Ap, (size_t)b * Mp * Kp + PD(m, k, Kp), (m < M && k < K) ? RD(A, b * sA + (size_t)m * lda + k) : 0.0); }
__kernel void pkd_a_t(__global const double *A, __global double *Ap, int M, int K, int lda, int Kp, long sA, int Mp) {
    const int m = get_global_id(0), k = get_global_id(1), b = get_global_id(2);
    WR(Ap, (size_t)b * Mp * Kp + PD(m, k, Kp), (m < M && k < K) ? RD(A, b * sA + (size_t)k * lda + m) : 0.0); }
__kernel void pkd_b(__global const double *B, __global double *Bp, int K, int N, int ldb, int Kp, long sB, int Np) {
    const int n = get_global_id(0), k = get_global_id(1), b = get_global_id(2);
    WR(Bp, (size_t)b * Np * Kp + PD(n, k, Kp), (n < N && k < K) ? RD(B, b * sB + (size_t)k * ldb + n) : 0.0); }
__kernel void pkd_b_t(__global const double *B, __global double *Bp, int K, int N, int ldb, int Kp, long sB, int Np) {
    const int k = get_global_id(0), n = get_global_id(1), b = get_global_id(2);
    WR(Bp, (size_t)b * Np * Kp + PD(n, k, Kp), (n < N && k < K) ? RD(B, b * sB + (size_t)n * ldb + k) : 0.0); }

// Main kernel, signature as in asm "v": (Ap, Bp, C, M, N, ldc, K, Kfull); M, N multiples of 16, K -- number of steps
// k from the start of the piece, Kfull -- panel length. Work-group 16 = a 16×16 tile of C from two subgroups of 8 (native
// FP64 width on Gen9): subgroup h computes rows 8h..8h+7, lane l columns l and l+8 -- one broadcast of A per two
// FMAs (version with subgroup 16 and one FMA per broadcast: 26 GFLOPS). Tile order: groups of GMD tile rows.
__attribute__((intel_reqd_sub_group_size(8)))
__kernel __attribute__((reqd_work_group_size(16, 1, 1)))
void dgemm_v(__global const double *Ap, __global const double *Bp, __global double *C, int M, int N, int ldc, int K, int Kfull) {
    const int pid0 = get_group_id(0);
    const int npm = M / 16, npn = N / 16;
    const int b = pid0 / (npm * npn), pid = pid0 % (npm * npn);
    const int group = pid / (GMD * npn), m_first = group * GMD, gsz = min(npm - m_first, GMD);
    const int pm = m_first + (pid % (GMD * npn)) % gsz, pn = (pid % (GMD * npn)) / gsz;
    const int l = get_sub_group_local_id(), h = get_sub_group_id();
    const __global double *pa = Ap + (size_t)b * M * Kfull + (size_t)pm * Kfull * 16 + 8 * h + l;
    const __global double *pb = Bp + (size_t)b * N * Kfull + (size_t)pn * Kfull * 16 + l;
    double acc0[8], acc1[8];
#pragma unroll
    for (int r = 0; r < 8; r++) { acc0[r] = 0.0; acc1[r] = 0.0; }
    for (int k = 0; k < K; k += 4) {
        double a[4], b0[4], b1[4];
#pragma unroll
        for (int q = 0; q < 4; q++) { a[q] = RD(pa, (size_t)(k + q) * 16); b0[q] = RD(pb, (size_t)(k + q) * 16); b1[q] = RD(pb, (size_t)(k + q) * 16 + 8); }
#pragma unroll
        for (int q = 0; q < 4; q++)
#pragma unroll
            for (int r = 0; r < 8; r++) {
                const double ar = sub_group_broadcast(a[q], r);
                acc0[r] = fma(ar, b0[q], acc0[r]); acc1[r] = fma(ar, b1[q], acc1[r]);
            }
    }
    __global double *c = C + (size_t)b * M * ldc + (size_t)(pm * 16 + 8 * h) * ldc + pn * 16 + l;
#pragma unroll
    for (int r = 0; r < 8; r++) { WR(c, (size_t)r * ldc, acc0[r]); WR(c, (size_t)r * ldc + 8, acc1[r]); }
}
