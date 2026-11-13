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
inline double div_chv(double a, double b) {   // a / b with DF mul/fma and dword integer ops only (Gen8 Cherryview)
    const uint2 ua = as_uint2(a), ub = as_uint2(b);
    const uint ea = (ua.s1 >> 20) & 0x7ffu, eb = (ub.s1 >> 20) & 0x7ffu, sign = (ua.s1 ^ ub.s1) & 0x80000000u;
    const int a_zero = ((ua.s1 & 0x7fffffffu) | ua.s0) == 0, b_zero = ((ub.s1 & 0x7fffffffu) | ub.s0) == 0;
    const int a_inf = ea == 0x7ffu && ((ua.s1 & 0xfffffu) | ua.s0) == 0, b_inf = eb == 0x7ffu && ((ub.s1 & 0xfffffu) | ub.s0) == 0;
    const int a_nan = ea == 0x7ffu && !a_inf, b_nan = eb == 0x7ffu && !b_inf;
    // both operands brought into [1, 2) (a subnormal first scaled by 2^54), quotient in (0.5, 2), then scaled by 2^(ka - kb)
    const uint2 na = as_uint2(ea == 0 ? a * 0x1p54 : a), nb = as_uint2(eb == 0 ? b * 0x1p54 : b);
    int k = ((int)((na.s1 >> 20) & 0x7ffu) - (ea == 0 ? 54 : 0)) - ((int)((nb.s1 >> 20) & 0x7ffu) - (eb == 0 ? 54 : 0));
    const double am = as_double((uint2)(na.s0, (na.s1 & 0x800fffffu) | 0x3ff00000u));
    const double bm = as_double((uint2)(nb.s0, (nb.s1 & 0x800fffffu) | 0x3ff00000u));
    double r = (double)native_recip((float)bm);
    r = fma(r, fma(-bm, r, 1.0), r); r = fma(r, fma(-bm, r, 1.0), r); r = fma(r, fma(-bm, r, 1.0), r);
    double q = am * r;
    q = fma(r, fma(-bm, q, am), q);
    for (int t = 0; t < 3; t++) {   // 2^k as up to three normal factors (a result beyond the range overflows / underflows here)
        const int kt = clamp(k, -1022, 1023);
        q *= as_double((uint2)(0u, (uint)(kt + 1023) << 20)); k -= kt;
    }
    const uint2 nan = (uint2)(0u, 0x7ff80000u), inf = (uint2)(0u, sign | 0x7ff00000u), zero = (uint2)(0u, sign);
    uint2 res = as_uint2(q);
    res = (b_inf || a_zero) ? zero : res;
    res = (a_inf || b_zero) ? inf : res;
    res = (a_nan || b_nan || (a_inf && b_inf) || (a_zero && b_zero)) ? nan : res;
    return as_double(res);
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
