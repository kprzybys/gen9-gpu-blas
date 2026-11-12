// libgen9blas: BLAS levels 1-2. Compiled three times: -DHALF (FP16 in memory, FP32 arithmetic), -DDBL (FP64) and without (FP32).
// R -- computation type (float for FP16/FP32, double for FP64), also for scalars (alpha, beta, c, s, ...) and work buffers.
// Vector: buffer + offset o + stride inc (in elements, inc may be negative); matrix: buffer + offset + ld
// (row-major). Operands may be the same buffer (shared host memory pages), hence no restrict.
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#ifdef HALF
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef half T;
typedef float R;
typedef float4 R4;
#define F(n) n##_h
#define V4(p) convert_float4(vload4(0, p))
#define VS4(v, p) vstore4(convert_half4(v), 0, p)
typedef half4 TV4;
#define CONV_R4(v) convert_float4(v)
// GW elements per lane in gemv_nw/copy16. FP32/FP64: 16 B (full 64 B lines in a SIMD16 message). FP16 stays at 4
// (8 B): with 8 the compiler makes SIMD8 and byte-wise 4 B reads -- FP16 gemv 15.4 -> 11.1 GB/s (results/message-16B-...);
// the FP16 copy is done by the FP32 kernel (gen9blas_copy).
#define GW 4
typedef float4 RG;
#define VG(p) convert_float4(vload4(0, p))
#define VLR(p) vload4(0, p)
#define COPYG(p, q) vstore4(vload4(0, p), 0, q)
#define DOTG(a, b) dot(a, b)
#elif defined(DBL)
typedef double T;
typedef double R;
typedef double4 R4;
#define F(n) n##_d
#define V4(p) vload4(0, p)
#define VS4(v, p) vstore4(v, 0, p)
typedef double4 TV4;
#define CONV_R4(v) convert_double4(v)
#define GW 2
typedef double2 RG;
#define VG(p) vload2(0, p)
#define VLR(p) vload2(0, p)
#define COPYG(p, q) vstore2(vload2(0, p), 0, q)
#define DOTG(a, b) dot(a, b)
#else
typedef float T;
typedef float R;
typedef float4 R4;
#define F(n) n##_f
#define V4(p) vload4(0, p)
#define VS4(v, p) vstore4(v, 0, p)
typedef float4 TV4;
#define CONV_R4(v) convert_float4(v)
#define GW 4
typedef float4 RG;
#define VG(p) vload4(0, p)
#define VLR(p) vload4(0, p)
#define COPYG(p, q) vstore4(vload4(0, p), 0, q)
#define DOTG(a, b) dot(a, b)
#endif
#define LD(p, i) ((R)(p)[i])
#define ST(p, i, v) ((p)[i] = (T)(v))
typedef long L;

// ---- level 1: work-item = 4 consecutive elements; with stride 1 vector read/write (vload4/vstore4) ----
// Previously one element per work-item: a lane read 4 B (FP32) / 2 B (FP16) per instruction, FP32 reductions 13-17 GB/s,
// FP16 7-8 GB/s (FP64 24 GB/s -- limited by the number of read instructions, not bytes).
#define FOUR(n, ...) const int i0 = get_global_id(0) * 4; if (i0 >= n) return; \
    const int full = i0 + 3 < n && (__VA_ARGS__)
__kernel void F(axpy)(int n, R a, __global const T *x, L ox, L ix, __global T *y, L oy, L iy) {
    FOUR(n, ix == 1 && iy == 1);
    if (full) { VS4(fma((R4)a, V4(x + ox + i0), V4(y + oy + i0)), y + oy + i0); return; }
    for (int i = i0; i < min(n, i0 + 4); i++) { const L jy = oy + i * iy; ST(y, jy, fma(a, LD(x, ox + i * ix), LD(y, jy))); }
}
__kernel void F(scal)(int n, R a, __global T *x, L ox, L ix) {
    FOUR(n, ix == 1);
    if (full) { VS4((R4)a * V4(x + ox + i0), x + ox + i0); return; }
    for (int i = i0; i < min(n, i0 + 4); i++) { const L j = ox + i * ix; ST(x, j, a * LD(x, j)); }
}
__kernel void F(copy)(int n, __global const T *x, L ox, L ix, __global T *y, L oy, L iy) {
    FOUR(n, ix == 1 && iy == 1);
    if (full) { vstore4(vload4(0, x + ox + i0), 0, y + oy + i0); return; }
    for (int i = i0; i < min(n, i0 + 4); i++) y[oy + i * iy] = x[ox + i * ix];
}
// COPY contiguous (stride 1): 16 B per lane in every type (GW elements). The 4-element version: FP64 32 B per lane -- each
// of the subgroup's two writes covered half of a 64 B line (target not read -> line fill), 17 GB/s; FP16 8 B, 13 GB/s.
__kernel void F(copy16)(int n, __global const T *x, L ox, __global T *y, L oy) {
    const int i0 = (int)get_global_id(0) * GW;
    if (i0 >= n) return;
    if (i0 + GW <= n) { COPYG(x + ox + i0, y + oy + i0); return; }
    for (int i = i0; i < n; i++) y[oy + i] = x[ox + i];
}
// rot: x' = c·x + s·y, y' = c·y − s·x; swap = 1: swap.
__kernel void F(rot)(int n, __global T *x, L ox, L ix, __global T *y, L oy, L iy, R c, R s, int swap) {
    FOUR(n, ix == 1 && iy == 1);
    if (full) {
        __global T *px = x + ox + i0, *py = y + oy + i0;
        if (swap) { const TV4 t = vload4(0, px); vstore4(vload4(0, py), 0, px); vstore4(t, 0, py); return; }
        const R4 a = V4(px), b = V4(py);
        VS4(fma((R4)c, a, (R4)s * b), px); VS4(fma((R4)c, b, -(R4)s * a), py); return;
    }
    for (int i = i0; i < min(n, i0 + 4); i++) {
        const L jx = ox + i * ix, jy = oy + i * iy;
        if (swap) { const T t = x[jx]; x[jx] = y[jy]; y[jy] = t; continue; }
        const R a = LD(x, jx), b = LD(y, jy);
        ST(x, jx, fma(c, a, s * b)); ST(y, jy, fma(c, b, -s * a));
    }
}
// rotm: x' = h11·x + h12·y, y' = h21·x + h22·y (matrix H computed on the CPU from param[0..4]).
__kernel void F(rotm)(int n, __global T *x, L ox, L ix, __global T *y, L oy, L iy, R h11, R h12, R h21, R h22) {
    FOUR(n, ix == 1 && iy == 1);
    if (full) {
        __global T *px = x + ox + i0, *py = y + oy + i0;
        const R4 a = V4(px), b = V4(py);
        VS4(fma((R4)h11, a, (R4)h12 * b), px); VS4(fma((R4)h21, a, (R4)h22 * b), py); return;
    }
    for (int i = i0; i < min(n, i0 + 4); i++) {
        const L jx = ox + i * ix, jy = oy + i * iy;
        const R a = LD(x, jx), b = LD(y, jy);
        ST(x, jx, fma(h11, a, h12 * b)); ST(y, jy, fma(h21, a, h22 * b));
    }
}

// ---- reductions: FP64 sum, one part per work-group (256), parts accumulated on the CPU (<= 96 numbers) ----
// Loop over chunks of 4 elements (grid step 4·grid size); with stride 1 vector read.
// mode 0: x·y, 1: Σ|x|, 2: Σx² (nrm2 = sqrt on the CPU)
#define D4(p) convert_double4(V4(p))
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void F(red)(int n, int mode, __global const T *x, L ox, L ix, __global const T *y, L oy, L iy, __global double *part) {
    double s = 0.0;
    const int vec = ix == 1 && (mode || iy == 1);
    for (int i0 = get_global_id(0) * 4; i0 < n; i0 += get_global_size(0) * 4) {
        if (vec && i0 + 3 < n) {
            const double4 v = D4(x + ox + i0);
            if (mode == 0) s += dot(v, D4(y + oy + i0));
            else if (mode == 1) s += dot(fabs(v), (double4)1.0);
            else s += dot(v, v);
            continue;
        }
        for (int i = i0; i < min(n, i0 + 4); i++) {
            const double v = LD(x, ox + (L)i * ix);
            if (mode == 0) s = fma(v, (double)LD(y, oy + (L)i * iy), s);
            else if (mode == 1) s += fabs(v);
            else s = fma(v, v, s);
        }
    }
    s = work_group_reduce_add(s);
    if (get_local_id(0) == 0) part[get_group_id(0)] = s;
}
// iamax: first index of the largest |x|; one part per group: (value, index); NaN skipped. Groups <= 96 (GROUPS in gen9blas.c).
// A work-item scans its elements in ascending order (chunks of 4, in sequence), so on a tie the smaller index stays.
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void F(iamax)(int n, __global const T *x, L ox, L ix, __global R *cv) {   // values [0, 96), indices [96, 192)
    __global int *ci = (__global int *)(cv + 96);
    R bv = -1.0f; int bi = 0x7fffffff;
    for (int i0 = get_global_id(0) * 4; i0 < n; i0 += get_global_size(0) * 4) {
        if (ix == 1 && i0 + 3 < n) {
            const R4 v = fabs(V4(x + ox + i0));
            if (v.x > bv) { bv = v.x; bi = i0; }
            if (v.y > bv) { bv = v.y; bi = i0 + 1; }
            if (v.z > bv) { bv = v.z; bi = i0 + 2; }
            if (v.w > bv) { bv = v.w; bi = i0 + 3; }
            continue;
        }
        for (int i = i0; i < min(n, i0 + 4); i++) { const R v = fabs(LD(x, ox + (L)i * ix)); if (v > bv) { bv = v; bi = i; } }
    }
    const R m = work_group_reduce_max(bv);
    const int k = work_group_reduce_min(bv == m ? bi : 0x7fffffff);
    if (get_local_id(0) == 0) { cv[get_group_id(0)] = m; ci[get_group_id(0)] = k; }
}

// ---- level 2 ----
// Triangle mask for element (row i, column j): 0 whole, 1 j <= i (lower), 2 j >= i (upper), 3 j < i, 4 j > i.
// unit = 1: x at the diagonal index is added to the sum (ones on the diagonal with strict mask 3/4; TRMV diag = U).
inline int in_triangle(int i, int j, int mode) {
    return mode == 0 || (mode == 1 && j <= i) || (mode == 2 && j >= i) || (mode == 3 && j < i) || (mode == 4 && j > i);
}
// GEMV N: y = alpha·A·x + beta·y, A: M×N (with triangle mask). A subgroup of 16 computes 4 rows (x read once per 4 rows),
// each lane 4 consecutive elements (vload4), step 64. beta = 0: y is not read.
#define RW 4
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void F(gemv_n)(int M, int N, R alpha, __global const T *A, L oa, int lda, __global const T *x, L ox, L ix,
                        R beta, __global T *y, L oy, L iy, int mode, int unit) {
    const int r0 = (int)(get_global_id(0) / 16) * RW, l = get_sub_group_local_id();
    if (r0 >= M) return;
    const int nr = min(RW, M - r0);
    int a0[RW], a1[RW];   // column range [a0, a1) of row r0 + r
#pragma unroll
    for (int r = 0; r < RW; r++) {
        const int i = r0 + r;
        a0[r] = mode == 2 ? i : mode == 4 ? i + 1 : 0; a1[r] = mode == 1 ? i + 1 : mode == 3 ? i : N;
        a0[r] = clamp(a0[r], 0, N); a1[r] = clamp(a1[r], 0, N);
    }
    const int lo = a0[0] & ~3, hi = a1[nr - 1] > a1[0] ? a1[nr - 1] : a1[0];
    const __global T *a = A + oa + (L)r0 * lda;
    R s[RW] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (int j = lo + 4 * l; j < hi; j += 64) {
        R4 xv;
        if (ix == 1 && j + 3 < N) xv = V4(x + ox + j);
        else xv = (R4)(LD(x, ox + j * ix), j + 1 < N ? LD(x, ox + (j + 1) * ix) : 0.0f,
                           j + 2 < N ? LD(x, ox + (j + 2) * ix) : 0.0f, j + 3 < N ? LD(x, ox + (j + 3) * ix) : 0.0f);
#pragma unroll
        for (int r = 0; r < RW; r++) {
            if (r >= nr) continue;
            if (j >= a0[r] && j + 3 < a1[r]) s[r] += dot(V4(a + (L)r * lda + j), xv);
            else for (int t = 0; t < 4; t++)
                if (j + t >= a0[r] && j + t < a1[r]) s[r] = fma(LD(a, (L)r * lda + j + t), ((__private R *)&xv)[t], s[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < RW; r++) {
        R t = sub_group_reduce_add(s[r]);
        if (l == 0 && r < nr) {
            if (unit) t += LD(x, ox + (L)(r0 + r) * ix);
            const L k = oy + (L)(r0 + r) * iy; ST(y, k, beta == 0.0f ? alpha * t : fma(alpha, t, beta * LD(y, k)));
        }
    }
}
// GEMV N, group variant (step 3, 2026-11-12): a work-group of 16 subgroups computes the same RW rows -- subgroup sg reads
// 256 B segments from column 64·sg every 1024 elements, so the group touches one 4 KB page per row at a time (the
// gemv_n version: each subgroup other 4 rows, ~700 pages at once across the whole GPU -- hypothesis: TLB misses). Partial sums
// of the subgroups are combined in local memory. Same triangle mask as gemv_n.
#define SGW 16
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16 * SGW, 1, 1)))
__kernel void F(gemv_nw)(int M, int N, R alpha, __global const T *A, L oa, int lda, __global const T *x, L ox, L ix,
                         R beta, __global T *y, L oy, L iy, int mode, int unit) {
    __local R part[RW][SGW];
    const int r0 = (int)get_group_id(0) * RW, l = get_sub_group_local_id(), sg = get_sub_group_id();
    const int nr = min(RW, M - r0);
    int a0[RW], a1[RW];
#pragma unroll
    for (int r = 0; r < RW; r++) {
        const int i = r0 + r;
        a0[r] = mode == 2 ? i : mode == 4 ? i + 1 : 0; a1[r] = mode == 1 ? i + 1 : mode == 3 ? i : N;
        a0[r] = clamp(a0[r], 0, N); a1[r] = clamp(a1[r], 0, N);
    }
    // GW elements (16 B) per lane, subgroup 16·GW contiguous -- full 64 B lines in every message (previously 4 elements:
    // FP64 32 B per lane, FP16 8 B; results/message-16B-2026-11-12.txt)
    const int lo = a0[0] & ~(GW - 1), hi = a1[nr - 1] > a1[0] ? a1[nr - 1] : a1[0];
    const __global T *a = A + oa + (L)r0 * lda;
    R s[RW] = { 0.0f, 0.0f, 0.0f, 0.0f };
#ifdef HALF
    // FP16, full matrix, addresses aligned to 16 B: block reads (subgroup = contiguous 256 B in one message; the compiler
    // splits vload half into byte-wise 4 B reads). A lane gets elements every 16 -- irrelevant for the dot product.
    const int blk = mode == 0 && ix == 1 && (lda & 7) == 0 && ((((ulong)a) | ((ulong)(x + ox))) & 15) == 0;
    if (blk) {
        const int jb = N & ~127;
        for (int j0 = 128 * sg; j0 < jb; j0 += 128 * SGW) {
            const float8 xv = convert_float8(as_half8(intel_sub_group_block_read_us8((const __global ushort *)(x + ox + j0))));
#pragma unroll
            for (int r = 0; r < RW; r++) {
                const __global T *ar = a + (L)min(r, nr - 1) * lda + j0;
                const float8 av = convert_float8(as_half8(intel_sub_group_block_read_us8((const __global ushort *)ar)));
                if (r < nr) s[r] += dot(av.lo, xv.lo) + dot(av.hi, xv.hi);
            }
        }
        for (int j = jb + 16 * sg + l; j < N; j += 16 * SGW) {   // tail < 128 elements
            const R xj = LD(x, ox + j);
#pragma unroll
            for (int r = 0; r < RW; r++) if (r < nr) s[r] = fma(LD(a, (L)r * lda + j), xj, s[r]);
        }
    } else
#endif
    for (int j = lo + 16 * GW * sg + GW * l; j < hi; j += 16 * GW * SGW) {
        RG xv;
        if (ix == 1 && j + GW - 1 < N) xv = VG(x + ox + j);
        else {
            R xe[GW];
            for (int t = 0; t < GW; t++) xe[t] = j + t < N ? LD(x, ox + (j + t) * ix) : 0.0f;
            xv = VLR(xe);
        }
#pragma unroll
        for (int r = 0; r < RW; r++) {
            if (r >= nr) continue;
            if (j >= a0[r] && j + GW - 1 < a1[r]) s[r] += DOTG(VG(a + (L)r * lda + j), xv);
            else for (int t = 0; t < GW; t++)
                if (j + t >= a0[r] && j + t < a1[r]) s[r] = fma(LD(a, (L)r * lda + j + t), ((__private R *)&xv)[t], s[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < RW; r++) { R t = sub_group_reduce_add(s[r]); if (l == 0) part[r][sg] = t; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg == 0 && l < nr) {
        R t = 0.0f;
        for (int g = 0; g < SGW; g++) t += part[l][g];
        if (unit) t += LD(x, ox + (L)(r0 + l) * ix);
        const L k = oy + (L)(r0 + l) * iy; ST(y, k, beta == 0.0f ? alpha * t : fma(alpha, t, beta * LD(y, k)));
    }
}
// GEMV T: y = alpha·Aᵀ·x + beta·y (y of length N; triangle mask as in gemv_n for element (i, column)). Stage 1:
// work-item = 4 columns × part of the rows (rpp), partial sums to P[part·N + column]. Stage 2: y = alpha·Σ P + beta·y.
__kernel void F(gemv_t1)(int M, int N, int rpp, __global const T *A, L oa, int lda, __global const T *x, L ox, L ix,
                         __global R *P, int mode, int unit) {
    const int c = get_global_id(0) * 4, r = get_global_id(1);
    if (c >= N) return;
    int i0 = r * rpp, i1 = min(M, i0 + rpp);
    // rows in which any of the 4 columns is in the triangle
    if (mode == 1) i0 = max(i0, c); else if (mode == 3) i0 = max(i0, c + 1);
    else if (mode == 2) i1 = min(i1, c + 4); else if (mode == 4) i1 = min(i1, c + 3);
    const __global T *a = A + oa + c;
    R4 s = 0.0f;
    const int full = c + 3 < N;
    for (int i = i0; i < i1; i++) {
        const R xv = LD(x, ox + (L)i * ix);
        R4 v;
        if (full) v = V4(a + (L)i * lda);
        else v = (R4)(LD(a, (L)i * lda), c + 1 < N ? LD(a, (L)i * lda + 1) : 0.0f, c + 2 < N ? LD(a, (L)i * lda + 2) : 0.0f, 0.0f);
        if (mode) v *= (R4)(in_triangle(i, c, mode), in_triangle(i, c + 1, mode), in_triangle(i, c + 2, mode), in_triangle(i, c + 3, mode));
        s = fma(v, (R4)xv, s);
    }
    if (unit) for (int t = 0; t < 4; t++)   // diagonal (i = c + t) in this part of the rows
        if (c + t >= r * rpp && c + t < min(M, r * rpp + rpp)) ((__private R *)&s)[t] += LD(x, ox + (L)(c + t) * ix);
    __global R *p = P + (L)r * N + c;
    p[0] = s.x; if (c + 1 < N) p[1] = s.y; if (c + 2 < N) p[2] = s.z; if (c + 3 < N) p[3] = s.w;
}
// GEMV T for FP16 and a full matrix (mode 0, no unit diagonal): block read -- a subgroup reads 64 consecutive columns of a row
// (128 B) in one message, a lane gets columns c0 + l + 16·t (the compiler splits vload4 half into byte-wise reads).
// Without 16 B alignment or at the edge (fewer than 64 columns) -- the ordinary path as in gemv_t1. For FP32/FP64: stub.
#ifdef HALF
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void F(gemv_t1b)(int M, int N, int rpp, __global const T *A, L oa, int lda, __global const T *x, L ox, L ix,
                          __global R *P) {
    const int l = get_sub_group_local_id(), r = get_global_id(1);
    const int c0 = ((int)get_global_id(0) - l) * 4;
    if (c0 >= N) return;   // uniform across the subgroup
    const int i0 = r * rpp, i1 = min(M, i0 + rpp);
    __global R *p = P + (L)r * N;
    if (c0 + 64 <= N && (lda & 7) == 0 && (((ulong)(A + oa)) & 15) == 0) {
        const __global ushort *a = (const __global ushort *)(A + oa + c0);
        float4 s = 0.0f;
        for (int i = i0; i < i1; i++)
            s = fma(convert_float4(as_half4(intel_sub_group_block_read_us4(a + (L)i * lda))), (float4)LD(x, ox + (L)i * ix), s);
        p += c0 + l; p[0] = s.x; p[16] = s.y; p[32] = s.z; p[48] = s.w;
        return;
    }
    const int c = c0 + 4 * l;
    if (c >= N) return;
    const __global T *a = A + oa + c;
    float4 s = 0.0f;
    for (int i = i0; i < i1; i++) {
        const R xv = LD(x, ox + (L)i * ix);
        R4 v;
        if (c + 3 < N) v = V4(a + (L)i * lda);
        else v = (R4)(LD(a, (L)i * lda), c + 1 < N ? LD(a, (L)i * lda + 1) : 0.0f, c + 2 < N ? LD(a, (L)i * lda + 2) : 0.0f, 0.0f);
        s = fma(v, (R4)xv, s);
    }
    p += c; p[0] = s.x; if (c + 1 < N) p[1] = s.y; if (c + 2 < N) p[2] = s.z; if (c + 3 < N) p[3] = s.w;
}
#else
__kernel void F(gemv_t1b)(int M, int N, int rpp, __global const T *A, L oa, int lda, __global const T *x, L ox, L ix,
                          __global R *P) { }
#endif
__kernel void F(gemv_t2)(int N, int nparts, __global const R *P, R alpha, R beta, __global T *y, L oy, L iy) {
    const int j = get_global_id(0); if (j >= N) return;
    R t = 0.0f;
    for (int r = 0; r < nparts; r++) t += P[(L)r * N + j];
    const L k = oy + (L)j * iy; ST(y, k, beta == 0.0f ? alpha * t : fma(alpha, t, beta * LD(y, k)));
}
// SYMV in one pass over the stored triangle (upper = 0: j <= i, 1: j >= i). Element A[i][j], read once, contributes
// to y_i (row part, with the diagonal) and y_j (column part, without the diagonal). Work-item = 4 columns c..c+3
// × band of rows (rpp); subgroup of 16 = group of 64 columns g. Row: sum over 64 columns (sub_group_reduce) -> Q[g·n + i];
// column: sums over the band -> P[band·n + c]. symv2: y = alpha·(Σ P + Σ Q) + beta·y. Previously two passes (N and T,
// triangle read twice): 8.2 GB/s FP32 counted from half the matrix.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void F(symv1)(int n, int rpp, int upper, __global const T *A, L oa, int lda, __global const T *x, L ox, L ix,
                       __global R *P, __global R *Q) {
    const int c = get_global_id(0) * 4, r = get_global_id(1), l = get_sub_group_local_id(), g = get_global_id(0) / 16, cg = g * 64;
    if (cg >= n) return;   // whole subgroup outside the matrix
    int i0 = r * rpp, i1 = min(n, i0 + rpp);
    if (upper) i1 = min(i1, cg + 64); else i0 = max(i0, cg);
    const int full = c + 3 < n;
    const R4 xc = full && ix == 1 ? V4(x + ox + c)
        : (R4)(c < n ? LD(x, ox + (L)c * ix) : 0.0f, c + 1 < n ? LD(x, ox + (L)(c + 1) * ix) : 0.0f,
               c + 2 < n ? LD(x, ox + (L)(c + 2) * ix) : 0.0f, c + 3 < n ? LD(x, ox + (L)(c + 3) * ix) : 0.0f);
    const int4 j = (int4)(c, c + 1, c + 2, c + 3);
    R4 s = 0.0f;
    for (int i = i0; i < i1; i++) {
        const __global T *a = A + oa + (L)i * lda + c;
        const R4 v = full ? V4(a) : (R4)(c < n ? LD(a, 0) : 0.0f, c + 1 < n ? LD(a, 1) : 0.0f, c + 2 < n ? LD(a, 2) : 0.0f, 0.0f);
        const R4 vw = v * CONV_R4(-(upper ? j >= i : j <= i));   // in the triangle (with the diagonal)
        const R4 vk = vw * CONV_R4(-(j != i));                  // without the diagonal
        s = fma(vk, (R4)LD(x, ox + (L)i * ix), s);
        const R w = sub_group_reduce_add(dot(vw, xc));
        if (l == 0) Q[(L)g * n + i] = w;
    }
    if (c < n) {
        __global R *p = P + (L)r * n + c;
        p[0] = s.x; if (c + 1 < n) p[1] = s.y; if (c + 2 < n) p[2] = s.z; if (c + 3 < n) p[3] = s.w;
    }
}
__kernel void F(symv2)(int n, int nparts, int upper, __global const R *P, __global const R *Q, R alpha, R beta,
                       __global T *y, L oy, L iy) {
    const int i = get_global_id(0); if (i >= n) return;
    R t = 0.0f;
    for (int r = 0; r < nparts; r++) t += P[(L)r * n + i];
    const int g0 = upper ? i / 64 : 0, g1 = upper ? (n + 63) / 64 : i / 64 + 1;   // column groups with an entry for row i
    for (int g = g0; g < g1; g++) t += Q[(L)g * n + i];
    const L k = oy + (L)i * iy; ST(y, k, beta == 0.0f ? alpha * t : fma(alpha, t, beta * LD(y, k)));
}
// SYR / SYR2 on a triangle (mode 1 lower, 2 upper): A += alpha·x·yᵀ (rank2 = 0; SYR: y = x) or alpha·(x·yᵀ + y·xᵀ).
__kernel void F(syr2)(int n, R alpha, __global const T *x, L ox, L ix, __global const T *y, L oy, L iy,
                      __global T *A, L oa, int lda, int mode, int rank2) {
    const int j = get_global_id(0), i = get_global_id(1);
    if (j >= n || i >= n || !in_triangle(i, j, mode)) return;
    const L k = oa + (L)i * lda + j;
    const R xi = LD(x, ox + (L)i * ix), yj = LD(y, oy + (L)j * iy);
    R v = fma(alpha * xi, yj, LD(A, k));
    if (rank2) v = fma(alpha * LD(y, oy + (L)i * iy), LD(x, ox + (L)j * ix), v);
    ST(A, k, v);
}
// Blocked TRSV, work vector xf in FP32. op(A)[r][c] = tr ? A[c][r] : A[r][c].
// Conversions: xf = x (to_f) and x = xf (from_f).
__kernel void F(to_f)(int n, __global const T *x, L ox, L ix, __global R *xf) {
    const int i = get_global_id(0); if (i < n) xf[i] = LD(x, ox + (L)i * ix);
}
__kernel void F(from_f)(int n, __global const R *xf, __global T *x, L ox, L ix) {
    const int i = get_global_id(0); if (i < n) ST(x, ox + (L)i * ix, xf[i]);
}
// Blocked TRSV (blocks of 64): the diagonal block in one work-group (local memory), then update of the rest of the
// vector. Work vector xf in FP32. op(A)[r][c] = tr ? A[c][r] : A[r][c]. A single-launch version (block ready
// flags, no global synchronization) turned out 3-8x slower -- docs/pitfalls.md (TRSV).
// Diagonal block [p, p+b), b <= NBT (256), one work-group of 256: 4 sub-blocks of 64 -- a sub-block is loaded into local
// memory and substituted (lower = op(A) lower: forward), then the remaining rows of the block are updated; the block vector
// in local memory. Previously block 64 (2 launches per 64 rows; n = 8192: 256 launches, CPU 40-51%).
#define NBT 256
#define OPA(r, c) (tr ? LD(A, oa + (L)(c) * lda + (r)) : LD(A, oa + (L)(r) * lda + (c)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void F(trsv_block)(int p, int b, __global const T *A, L oa, int lda, int tr, int lower, int diag, __global R *xf) {
    __local R Lb[64][65];
    __local R xl[NBT];
    const int t = get_local_id(0), ns = (b + 63) / 64;
    for (int i = t; i < b; i += 256) xl[i] = xf[p + i];
    for (int q = 0; q < ns; q++) {
        const int s0 = (lower ? q : ns - 1 - q) * 64, bs = min(64, b - s0);
#pragma unroll
        for (int e = t; e < 64 * 64; e += 256) {   // 16 independent reads per thread (unrolled)
            const int r = e / 64, c = e % 64;
            Lb[r][c] = r < bs && c < bs ? OPA(p + s0 + r, p + s0 + c) : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // sub-block of 64 in chunks of 16 rows: the first subgroup solves a chunk with broadcasts only (no barriers),
        // then the whole group updates the remaining rows of the sub-block -- 2 barriers per 16 rows instead of 16
        const int ng = (bs + 15) / 16;
        for (int gq = 0; gq < ng; gq++) {
            const int g0 = (lower ? gq : ng - 1 - gq) * 16, gs = min(16, bs - g0);
            if (t < 16) {
                R v = t < gs ? xl[s0 + g0 + t] : 0.0f;
                for (int i = 0; i < gs; i++) {
                    const int k = lower ? i : gs - 1 - i;
                    R vk = sub_group_broadcast(v, k);
                    if (!diag) vk /= Lb[g0 + k][g0 + k];
                    if (t == k) v = vk;
                    else if (t < gs && (lower ? t > k : t < k)) v = fma(-Lb[g0 + t][g0 + k], vk, v);
                }
                if (t < gs) xl[s0 + g0 + t] = v;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            const int u0 = lower ? g0 + gs : 0, u1 = lower ? bs : g0;   // remaining rows of the sub-block
            if (t >= u0 && t < u1) {
                R s = 0.0f;
                for (int c = 0; c < gs; c++) s = fma(Lb[t][g0 + c], xl[s0 + g0 + c], s);
                xl[s0 + t] -= s;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        // remaining rows of the block in the solving direction: lower [s0+bs, b), upper [0, s0)
        const int r0 = lower ? s0 + bs : 0, r1 = lower ? b : s0;
        for (int r = r0 + t; r < r1; r += 256) {
            R s = 0.0f;
            if (bs == 64) {   // reads unrolled: no waiting for each one (loop version: ~260 µs per block of 256)
#pragma unroll 16
                for (int c = 0; c < 64; c++) s = fma(OPA(p + r, p + s0 + c), xl[s0 + c], s);
            } else for (int c = 0; c < bs; c++) s = fma(OPA(p + r, p + s0 + c), xl[s0 + c], s);
            xl[r] -= s;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int i = t; i < b; i += 256) xf[p + i] = xl[i];
}
// Update of the rest after block [p, p+b): xf[r] -= Σ_{c in block} op(A)[r][c]·xf[c] for r in [r0, r1).
// Without transpose: row r contiguous in memory -- one subgroup per row, a lane takes 4 consecutive elements in chunks of 64.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void F(trsv_upd_n)(int p, int b, int r0, int r1, __global const T *A, L oa, int lda, __global R *xf) {
    const int r = r0 + (int)(get_global_id(0) / 16), l = get_sub_group_local_id();
    if (r >= r1) return;
    R s = 0.0f;
#pragma unroll
    for (int c0 = 0; c0 < NBT; c0 += 64)
        if (c0 + 4 * l + 3 < b) s += dot(V4(A + oa + (L)r * lda + p + c0 + 4 * l), vload4(0, xf + p + c0 + 4 * l));
        else for (int t = 0; t < 4; t++) { const int c = c0 + 4 * l + t; if (c < b) s = fma(LD(A, oa + (L)r * lda + p + c), xf[p + c], s); }
    s = sub_group_reduce_add(s);
    if (l == 0) xf[r] -= s;
}
// With transpose: op(A)[r][c] = A[c][r] -- one work-item per r (column of A), loop over the block rows (contiguous read within the group).
__kernel void F(trsv_upd_t)(int p, int b, int r0, int r1, __global const T *A, L oa, int lda, __global R *xf) {
    const int r = r0 + get_global_id(0);
    if (r >= r1) return;
    R s = 0.0f;
#pragma unroll 16
    for (int c = 0; c < b; c++) s = fma(LD(A, oa + (L)(p + c) * lda + r), xf[p + c], s);
    xf[r] -= s;
}
// GER: A += alpha·x·yᵀ, A: M×N; work-item = element.
__kernel void F(ger)(int M, int N, R alpha, __global const T *x, L ox, L ix, __global const T *y, L oy, L iy,
                     __global T *A, L oa, int lda) {
    const int j = get_global_id(0), i = get_global_id(1);
    if (j >= N || i >= M) return;
    const L k = oa + (L)i * lda + j;
    ST(A, k, fma(alpha * LD(x, ox + (L)i * ix), LD(y, oy + (L)j * iy), LD(A, k)));
}

// ---- level 2: banded and packed matrices (row layout as CBLAS RowMajor) ----
// Format fmt: 0 band (KL below, KU above the diagonal; A(r,c) = a[r·lda + KL + c − r]), 1 packed upper (row r:
// c = r..n−1 from r·n − r(r−1)/2), 2 packed lower (row r: c = 0..r from r(r+1)/2). n -- number of columns of A.
inline R el(int fmt, __global const T *A, L oa, int lda, int KL, int KU, int n, int r, int c) {
    if (fmt == 0) return (c - r < -KL || c - r > KU) ? 0.0f : LD(A, oa + (L)r * lda + KL + c - r);
    if (fmt == 1) return c < r ? 0.0f : LD(A, oa + (L)r * n - (L)r * (r - 1) / 2 + c - r);
    return c > r ? 0.0f : LD(A, oa + (L)r * (r + 1) / 2 + c);
}
// Element op(A)(i, j): sym -- symmetric from the stored triangle (upper); otherwise tr ? A(j, i) : A(i, j); unit -- ones
// on the diagonal (triangular).
inline R elop(int fmt, __global const T *A, L oa, int lda, int KL, int KU, int n, int i, int j, int tr, int sym, int upper, int unit) {
    if (unit && i == j) return 1.0f;
    if (sym) return upper ? el(fmt, A, oa, lda, KL, KU, n, min(i, j), max(i, j)) : el(fmt, A, oa, lda, KL, KU, n, max(i, j), min(i, j));
    return tr ? el(fmt, A, oa, lda, KL, KU, n, j, i) : el(fmt, A, oa, lda, KL, KU, n, i, j);
}
// y = alpha·op(A)·x + beta·y; one work-item per result element, j in [j0(i), j1(i)) -- the nonzero range of op(A) in row i.
// lo, hi: op(A)(i, j) may be nonzero for j in [i − lo, i + hi]; nx -- length of x.
__kernel void F(pmv)(int ny, int nx, int fmt, int KL, int KU, int n, __global const T *A, L oa, int lda,
                     __global const T *x, L ox, L ix, R alpha, R beta, __global T *y, L oy, L iy,
                     int tr, int sym, int upper, int unit, int lo, int hi) {
    const int i = get_global_id(0); if (i >= ny) return;
    const int j0 = max(0, i - lo), j1 = min(nx, i + hi + 1);
    R s = 0.0f;
    for (int j = j0; j < j1; j++) s = fma(elop(fmt, A, oa, lda, KL, KU, n, i, j, tr, sym, upper, unit), LD(x, ox + (L)j * ix), s);
    const L k = oy + (L)i * iy; ST(y, k, beta == 0.0f ? alpha * s : fma(alpha, s, beta * LD(y, k)));
}
// Solve op(A)·x = b for triangular banded / packed: one work-group of 256, row by row
// (lower = op(A) lower: forward), dot product over the range (band: K, packed: whole row) in parallel + reduction.
// xf: FP32 work vector (b on input, x on output).
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void F(psv)(int n, int fmt, int KL, int KU, __global const T *A, L oa, int lda, __global R *xf,
            int tr, int lower, int unit, int K) {
    const int t = get_local_id(0);
    for (int q = 0; q < n; q++) {
        const int i = lower ? q : n - 1 - q;
        const int j0 = lower ? max(0, i - K) : i + 1, j1 = lower ? i : min(n, i + K + 1);
        R s = 0.0f;
        for (int j = j0 + t; j < j1; j += 256) s = fma(elop(fmt, A, oa, lda, KL, KU, n, i, j, tr, 0, 0, 0), xf[j], s);
        s = work_group_reduce_add(s);
        if (t == 0) { R v = xf[i] - s; if (!unit) v /= elop(fmt, A, oa, lda, KL, KU, n, i, i, tr, 0, 0, 0); xf[i] = v; }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}
// SPR / SPR2 on a packed matrix (fmt 1 upper, 2 lower): A += alpha·x·yᵀ (rank2 = 0, y = x) or alpha·(x·yᵀ + y·xᵀ).
__kernel void F(spr2)(int n, int fmt, R alpha, __global const T *x, L ox, L ix, __global const T *y, L oy, L iy,
                      __global T *A, L oa, int rank2) {
    const int j = get_global_id(0), i = get_global_id(1);
    if (j >= n || i >= n || (fmt == 1 ? j < i : j > i)) return;
    const L k = oa + (fmt == 1 ? (L)i * n - (L)i * (i - 1) / 2 + j - i : (L)i * (i + 1) / 2 + j);
    R v = fma(alpha * LD(x, ox + (L)i * ix), LD(y, oy + (L)j * iy), LD(A, k));
    if (rank2) v = fma(alpha * LD(y, oy + (L)i * iy), LD(x, ox + (L)j * ix), v);
    ST(A, k, v);
}

// ---- skinny GEMM (small M), 2026-11-12: docs/skinny-gemm.md ----
// C (M×N) = alpha·A·op(W) + beta·C without packing; weights read once per tile of MT rows of A; FP32 accumulation (R).
// Arbitrary strides: A[m][k] = A[m·sam + k·sak], C[m][n] = C[m·scm + n·scn] (swapping roles for small N: gen9blas.c).
// FP64 omitted (local memory and registers; D keeps the old path).
#ifndef DBL
#ifdef HALF
#define SKINNY_BLOCK 1
#define BLOCK4(p) convert_float4(as_half4(intel_sub_group_block_read_us4((const __global ushort *)(p))))
#define V8C(p) convert_float8(vload8(0, p))
#else
#define SKINNY_BLOCK 0
#define BLOCK4(p) vload4(0, p)
#define V8C(p) vload8(0, p)
#endif
// (1) Weights contiguous along K: element (k, n) = W[n·ldw + k] (C = A·Wᵀ). Layout as gemv_nw (24-25 GB/s): group = 16 subgroups
// on the same RW weight rows (few pages at once -- the lesson from GEMV N on userptr); a lane reads 4 consecutive k, a subgroup 64 k
// contiguous, subgroups every 64, step 1024. A is read the same way (contiguous, each read for RW weight rows). Version 1
// (A shared by the subgroup, lane = weight row) was instruction-bound: ~60 GFLOPS, FP16 M = 1 7.5 GB/s.
// Grid dimension 0 = tile of M (fastest): consecutive groups read the same weight rows while they are still in the cache.
#define SKINNY_K(MT, RW) \
__attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(256, 1, 1))) \
__kernel void F(skinny_k##MT)(int M, int N, int K, R alpha, R beta, __global const T *A, L sam, L sak, \
                             __global const T *W, L ldw, __global T *C, L scm, L scn) { \
    __local float part[16][RW * MT]; \
    const int l = get_sub_group_local_id(), sg = get_sub_group_id(); \
    const int m0 = (int)get_group_id(0) * MT, r0 = (int)get_group_id(1) * RW, mt = min(MT, M - m0); \
    /* row pointers once before the loop (64-bit multiplication on Gen9 is emulated); rows out of range -- */ \
    /* copy of the last valid one (computed, not written), so the loop has no conditions */ \
    const __global T *wr[RW], *ar[MT]; \
    __attribute__((opencl_unroll_hint)) for (int r = 0; r < RW; r++) wr[r] = W + (L)min(r0 + r, N - 1) * ldw; \
    __attribute__((opencl_unroll_hint)) for (int m = 0; m < MT; m++) ar[m] = A + (L)(m0 + min(m, mt - 1)) * sam; \
    float s[RW][MT]; \
    __attribute__((opencl_unroll_hint)) for (int r = 0; r < RW; r++) \
        __attribute__((opencl_unroll_hint)) for (int m = 0; m < MT; m++) s[r][m] = 0.0f; \
    int j = 64 * sg + 4 * l; \
    if (SKINNY_BLOCK && sak == 1 && (K & 63) == 0 && ((ldw | sam) & 7) == 0 && \
        ((((ulong)wr[0]) | ((ulong)ar[0])) & 15) == 0) {   /* FP16: block reads (lane = k every 16) */ \
        for (int j0 = 64 * sg; j0 < K; j0 += 1024) { \
            float4 wv[RW]; \
            __attribute__((opencl_unroll_hint)) for (int r = 0; r < RW; r++) wv[r] = BLOCK4(wr[r] + j0); \
            __attribute__((opencl_unroll_hint)) for (int m = 0; m < MT; m++) { \
                const float4 av = BLOCK4(ar[m] + j0); \
                __attribute__((opencl_unroll_hint)) for (int r = 0; r < RW; r++) s[r][m] += dot(wv[r], av); \
            } \
        } \
        j = K; \
    } else if (sak == 1) { \
        for (; j + 3 < K; j += 1024) { \
            float4 wv[RW]; \
            __attribute__((opencl_unroll_hint)) for (int r = 0; r < RW; r++) wv[r] = V4(wr[r] + j); \
            __attribute__((opencl_unroll_hint)) for (int m = 0; m < MT; m++) { \
                const float4 av = V4(ar[m] + j); \
                __attribute__((opencl_unroll_hint)) for (int r = 0; r < RW; r++) s[r][m] += dot(wv[r], av); \
            } \
        } \
    } else { \
        for (; j + 3 < K; j += 1024) { \
            float4 wv[RW]; \
            __attribute__((opencl_unroll_hint)) for (int r = 0; r < RW; r++) wv[r] = V4(wr[r] + j); \
            const L o = (L)j * sak; \
            __attribute__((opencl_unroll_hint)) for (int m = 0; m < MT; m++) { \
                const __global T *am = ar[m] + o; \
                const float4 av = (float4)((float)am[0], (float)am[sak], (float)am[2 * sak], (float)am[3 * sak]); \
                __attribute__((opencl_unroll_hint)) for (int r = 0; r < RW; r++) s[r][m] += dot(wv[r], av); \
            } \
        } \
    } \
    if (j < K) {   /* tail: 1-3 elements of K (one pass) */ \
        float4 wv[RW]; \
        __attribute__((opencl_unroll_hint)) for (int r = 0; r < RW; r++) \
            wv[r] = (float4)((float)wr[r][j], j + 1 < K ? (float)wr[r][j + 1] : 0.0f, j + 2 < K ? (float)wr[r][j + 2] : 0.0f, 0.0f); \
        __attribute__((opencl_unroll_hint)) for (int m = 0; m < MT; m++) { \
            const __global T *am = ar[m] + (L)j * sak; \
            const float4 av = (float4)((float)am[0], j + 1 < K ? (float)am[sak] : 0.0f, j + 2 < K ? (float)am[2 * sak] : 0.0f, 0.0f); \
            __attribute__((opencl_unroll_hint)) for (int r = 0; r < RW; r++) s[r][m] += dot(wv[r], av); \
        } \
    } \
    __attribute__((opencl_unroll_hint)) for (int r = 0; r < RW; r++) \
        __attribute__((opencl_unroll_hint)) for (int m = 0; m < MT; m++) { \
            const float t = sub_group_reduce_add(s[r][m]); \
            if (l == 0) part[sg][r * MT + m] = t; \
        } \
    barrier(CLK_LOCAL_MEM_FENCE); \
    const int o = sg * 16 + l; \
    if (o < RW * MT) { \
        const int r = o / MT, m = o % MT, n = r0 + r; \
        if (m < mt && n < N) { \
            float t = 0.0f; \
            for (int g = 0; g < 16; g++) t += part[g][o]; \
            const L kc = (L)(m0 + m) * scm + (L)n * scn; \
            ST(C, kc, beta == 0.0f ? alpha * t : fma(alpha, t, beta * LD(C, kc))); \
        } \
    } \
}
SKINNY_K(1, 4)
SKINNY_K(4, 4)
SKINNY_K(8, 2)   // MT = 8 with RW = 4 (32 accumulators) -> register spill (spill_size 448-640 B), kcl does not support this

// (1b) Rework 2026-11-12 (results/skinny-gemm-rework-2026-11-12.txt): weights contiguous along K, A contiguous along K (sak = 1),
// K % (8·KL) = 0, A, W and strides aligned to 16 B (checked by gen9blas.c; the fallback path in the kernel spilled registers). SIMD8 (float per lane = 1 GRF -> 8·MT accumulators without spill), RW = 8 weight rows,
// S = 4 subgroups every 8·KL k (step 32·KL -> long loop, cheap epilogue). A lane reads 8 B (FP16: KL = 4, FP32: KL = 2) --
// the block read is interleaved (element c of lane l = p[l + 8c]); the same layout for A and weights, so the dot product matches.
// Accumulation with explicit fma, component-wise conversion.
// Raw data kept as dwords (ushort4 IGC kept 1 GRF per component). FP16: KL = 2, one dword per lane
// (pair k, k+1; 32 B message) -- with KL = 4 (uint2) r8 spilled registers (spill 1056 B: 64 accumulators + raw A and W
// + conversions hoisted before the loop). FP32: KL = 2, uint2 per lane (64 B message).
#ifdef HALF
#define KL 4
#define RRAW uint2
#define RREAD(p) intel_sub_group_block_read2((const __global uint *)(p))
#define RCONV(x, c) ({ const half2 h_ = as_half2(((c) >> 1) ? (x).s1 : (x).s0); (float)(((c) & 1) ? h_.s1 : h_.s0); })
#else
#define KL 2
#define RRAW uint2
#define RREAD(p) intel_sub_group_block_read2((const __global uint *)(p))
#define RCONV(x, c) as_float((c) ? (x).s1 : (x).s0)
#endif
// Accumulators: a separate vector per weight row (s0..s7, component = m). The array float s[8][8] (256 B per lane) landed in
// private memory (scratch, kcl does not configure it -> wrong result rows; fine through NEO; 2026-11-12) --
// IGC moves only smaller arrays to registers (s[8][4] = 128 B passed), separate variables always.
#ifndef SKINNY_S
#define SKINNY_S 16   // subgroups per group (gen9blas.c: SKINNY_S)
#endif
#define SKINNY_W8(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7)
#define CR_ZERO(r) vr_t s##r = 0.0f;
#define CR_FMA(r) s##r = fma((vr_t)RCONV(rw[r], c), av, s##r);
// Transposing reduction step: pair of vectors (x, y) -> x; the lane with bit b keeps y, its partner x.
#define CR_PAIR(x, y, b) { const int g_ = (l & (b)) != 0; const vr_t wy_ = g_ ? x : y, zo_ = g_ ? y : x; \
    x = zo_ + intel_sub_group_shuffle_xor(wy_, (b)); }
#define SKINNY_LOOP(READ) \
    for (int j = 8 * KL * sg; j < K; j += 8 * KL * SKINNY_S) { \
        RRAW rw[8], ra[MTc]; \
        __attribute__((opencl_unroll_hint(8))) for (int r = 0; r < 8; r++) rw[r] = READ(wr[r] + j); \
        __attribute__((opencl_unroll_hint(MTc))) for (int m = 0; m < MTc; m++) ra[m] = READ(ar[m] + j); \
        __attribute__((opencl_unroll_hint(KL))) for (int c = 0; c < KL; c++) { \
            vr_t av; \
            __attribute__((opencl_unroll_hint(MTc))) for (int m = 0; m < MTc; m++) \
                ((__private float *)&av)[m] = RCONV(ra[m], c); \
            SKINNY_W8(CR_FMA) \
        } \
    }
#define SKINNY_R(MT, VR) \
__attribute__((intel_reqd_sub_group_size(8))) __attribute__((reqd_work_group_size(8 * SKINNY_S, 1, 1))) \
__kernel void F(skinny_r##MT)(int M, int N, int K, R alpha, R beta, __global const T *A, L sam, \
                             __global const T *W, L ldw, __global T *C, L scm, L scn) { \
    typedef VR vr_t; enum { MTc = MT }; \
    __local float part[SKINNY_S][8 * MT]; \
    const int l = get_sub_group_local_id(), sg = get_sub_group_id(); \
    const int m0 = (int)get_group_id(0) * MT, r0 = (int)get_group_id(1) * 8, mt = min(MT, M - m0); \
    const __global T *wr[8], *ar[MT]; \
    __attribute__((opencl_unroll_hint(8))) for (int r = 0; r < 8; r++) wr[r] = W + (L)min(r0 + r, N - 1) * ldw; \
    __attribute__((opencl_unroll_hint(MT))) for (int m = 0; m < MT; m++) ar[m] = A + (L)(m0 + min(m, mt - 1)) * sam; \
    SKINNY_W8(CR_ZERO) \
    SKINNY_LOOP(RREAD) \
    CR_PAIR(s0, s4, 4) CR_PAIR(s1, s5, 4) CR_PAIR(s2, s6, 4) CR_PAIR(s3, s7, 4) \
    CR_PAIR(s0, s2, 2) CR_PAIR(s1, s3, 2) CR_PAIR(s0, s1, 1)   /* lane l: s0 = sums for weight row l (component = m) */ \
    __attribute__((opencl_unroll_hint(MT))) for (int i = 0; i < MT; i++) part[sg][l * MT + i] = ((__private float *)&s0)[i]; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    for (int o = sg * 8 + l; o < 8 * MT; o += 8 * SKINNY_S) { \
        const int r = o / MT, m = o % MT, n = r0 + r; \
        if (m < mt && n < N) { \
            float t = 0.0f; \
            for (int g = 0; g < SKINNY_S; g++) t += part[g][o]; \
            const L kc = (L)(m0 + m) * scm + (L)n * scn; \
            ST(C, kc, beta == 0.0f ? alpha * t : fma(alpha, t, beta * LD(C, kc))); \
        } \
    } \
}
SKINNY_R(4, float4)
SKINNY_R(8, float8)

// (2) Weights contiguous along N: element (k, n) = W[k·ldw + n] (C = A·B). Work-item = 4 consecutive columns, grid dimension 1 =
// part of K × tile of M (tile faster: consecutive tiles read the same chunk of weights while it is in the LLC). MT = 4 (MT = 8: spill). A loaded once per 16 k spread across the lanes and broadcast
// (sub_group_broadcast) -- version 1 read A[m][k] separately for each k (a message per FMA). Partial sums
// P[part][m][n] (FP32), the second stage skinny_n2 sums the parts and does the epilogue.
#define SKINNY_N(MT) \
__attribute__((intel_reqd_sub_group_size(16))) \
__kernel void F(skinny_n##MT)(int M, int N, int K, int kpc, __global const T *A, L sam, L sak, \
                             __global const T *W, L ldw, __global float *P) { \
    const int tiles = (M + MT - 1) / MT, part = (int)get_global_id(1) / tiles, m0 = (int)get_global_id(1) % tiles * MT; \
    const int n = (int)get_global_id(0) * 4; \
    const int l = get_sub_group_local_id(), mt = min(MT, M - m0), k0 = part * kpc, k1 = min(K, k0 + kpc); \
    const int nn = min(n, ((N - 1) / 4) * 4);   /* lane beyond N: reads the last four, does not write */ \
    const int full = nn + 3 < N; \
    const __global T *ap[MT]; \
    __attribute__((opencl_unroll_hint)) for (int m = 0; m < MT; m++) ap[m] = A + (L)(m0 + min(m, mt - 1)) * sam; \
    float4 acc[MT]; \
    __attribute__((opencl_unroll_hint)) for (int m = 0; m < MT; m++) acc[m] = 0.0f; \
    const __global T *wr = W + (L)k0 * ldw + nn;   /* weight row k; advanced by ldw (no multiplication in the loop) */ \
    for (int kb = k0; kb < k1; kb += 16) { \
        float ar[MT]; \
        const int kl = kb + l; \
        __attribute__((opencl_unroll_hint)) for (int m = 0; m < MT; m++) \
            ar[m] = kl < k1 ? (float)(sak == 1 ? ap[m][kl] : ap[m][(L)kl * sak]) : 0.0f; \
        const int kn = min(16, k1 - kb); \
        if (kn == 16 && full) { \
            __attribute__((opencl_unroll_hint)) for (int t = 0; t < 16; t++) { \
                const float4 wv = V4(wr); wr += ldw; \
                __attribute__((opencl_unroll_hint)) for (int m = 0; m < MT; m++) \
                    acc[m] = fma((float4)(sub_group_broadcast(ar[m], t)), wv, acc[m]); \
            } \
        } else { \
            for (int t = 0; t < 16; t++) { \
                float4 wv = 0.0f; \
                if (t < kn) { \
                    wv = full ? V4(wr) : (float4)((float)wr[0], nn + 1 < N ? (float)wr[1] : 0.0f, nn + 2 < N ? (float)wr[2] : 0.0f, 0.0f); \
                    wr += ldw; \
                } \
                __attribute__((opencl_unroll_hint)) for (int m = 0; m < MT; m++) \
                    acc[m] = fma((float4)(sub_group_broadcast(ar[m], t)), wv, acc[m]); \
            } \
        } \
    } \
    if (n >= N) return; \
    __global float *p = P + ((L)part * M + m0) * N + n; \
    __attribute__((opencl_unroll_hint)) for (int m = 0; m < MT; m++) if (m < mt) { \
        if (n + 3 < N) vstore4(acc[m], 0, p + (L)m * N); \
        else for (int t = 0; t < 4; t++) if (n + t < N) p[(L)m * N + t] = ((__private float *)&acc[m])[t]; \
    } \
}
SKINNY_N(4)
// Stage 2: C[m][n] = alpha·Σ_part P[part][m][n] + beta·C (work-item = element).
__kernel void F(skinny_n2)(int M, int N, int nparts, __global const float *P, R alpha, R beta,
                          __global T *C, L scm, L scn) {
    const int n = (int)get_global_id(0), m = (int)get_global_id(1);
    if (n >= N || m >= M) return;
    float s = 0.0f;
    for (int c = 0; c < nparts; c++) s += P[((L)c * M + m) * N + n];
    const L kc = (L)m * scm + (L)n * scn;
    ST(C, kc, beta == 0.0f ? alpha * s : fma(alpha, s, beta * LD(C, kc)));
}
#endif
