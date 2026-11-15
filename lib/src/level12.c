// Levels 1-2 BLAS (vectors, GEMV, triangular solves) and level 2 banded / packed matrices.
// Split from lib/gen9blas.c (refactor 2026-11-15); shared state in internal.h.
#include "internal.h"

// ---------------- levels 1–2 (host memory: USM zero-copy or a copy through a work buffer, see get_obj()) ----------------
static cl_mem dP;        // GEMV T partial sums
static size_t sP;
static double *red_out;      // reduction results (USM host: the GPU writes, the CPU reads after wait_gpu())

int l12_type(int type, l12_t **k, int *es) {
    NEW_OPERATION();
    int i = type == GEN9BLAS_S ? 1 : type == GEN9BLAS_D ? 2 : type == GEN9BLAS_H || type == GEN9BLAS_HS ? 0 : -1;
    if (i < 0 || !L12[i].axpy) FAIL("type %d unavailable", type);
    *k = &L12[i]; *es = L12[i].es; return 0;
}
// Scalar for a kernel in the compute type R (double for FP64, float for FP16/FP32).
cl_int argr(cl_kernel K, int i, double v, int es) {
    if (es == 8) return clSetKernelArg(K, i, sizeof v, &v);
    float f = (float)v; return clSetKernelArg(K, i, sizeof f, &f);
}
// Operand: region [p, p+b) of host memory; g -- pointer for the kernel (p or a work buffer), zc: zero-copy.
static work_t *const work_l12[3] = { &uA, &uB, &uC };
static int load_operands(op_t *w, int n, const int *copy_in, const int *written) {
    for (int i = 0; i < n; i++)
        if (get_obj(w[i].p, w[i].b, work_l12[i], copy_in[i], written ? written[i] : 0, &w[i].g, &w[i].zc)) return -1;
    return 0;
}
// Vector region: n elements spaced inc apart. inc < 0 as in BLAS: pointer to the lowest address, element 0 at the end.
static size_t vec_bytes(int n, int inc, int es) { return ((size_t)(n - 1) * (size_t)(inc < 0 ? -inc : inc) + 1) * es; }
static cl_long vec_off(int n, int inc) { return inc < 0 ? (cl_long)(n - 1) * -inc : 0; }
// End: wait, copy the written non-USM operands back to host memory.
static int finish(op_t *w, int n, const int *written) {
    if (wait_gpu()) return -1;
    for (int i = 0; i < n; i++) if (written[i] && !w[i].zc) memcpy((void *)w[i].p, w[i].g.p, w[i].b);
    return 0;
}
static int start1(cl_kernel k, int n);
// Element-wise kernels (axpy, scal, copy, rot, rotm): work-item = 4 elements.
static int start4(cl_kernel k, int n) { return start1(k, (n + 3) / 4); }
static int start1(cl_kernel k, int n) {
    size_t g = ((size_t)n + 255) / 256 * 256, l = 256;
    CHECK(clEnqueueNDRangeKernel(q, k, 1, NULL, &g, &l, 0, NULL, NULL)); return 0;
}

int gen9blas_axpy(int type, int n, double a, const void *x, int incx, void *y, int incy) {
    BEGIN(n, !incx || !incy);
    op_t w[2] = { { x, vec_bytes(n, incx, es) }, { y, vec_bytes(n, incy, es) } }; const int cin[2] = { 1, 1 }, cout[2] = { 0, 1 };
    if (load_operands(w, 2, cin, cout)) return -1;
    cl_long ox = vec_off(n, incx), oy = vec_off(n, incy), ix = incx, iy = incy; cl_kernel K = k->axpy;
    ARG(K, 0, n); ARGR(K, 1, a); ARGB(K, 2, w[0].g); ARG(K, 3, ox); ARG(K, 4, ix); ARGB(K, 5, w[1].g); ARG(K, 6, oy); ARG(K, 7, iy);
    return start4(K, n) || finish(w, 2, cout) ? -1 : 0;
}

int gen9blas_scal(int type, int n, double a, void *x, int incx) {
    BEGIN(n, !incx);
    op_t w[1] = { { x, vec_bytes(n, incx, es) } }; const int cin[1] = { 1 };
    if (load_operands(w, 1, cin, cin)) return -1;
    cl_long ox = vec_off(n, incx), ix = incx; cl_kernel K = k->scal;
    ARG(K, 0, n); ARGR(K, 1, a); ARGB(K, 2, w[0].g); ARG(K, 3, ox); ARG(K, 4, ix);
    return start4(K, n) || finish(w, 1, cin) ? -1 : 0;
}

int gen9blas_copy(int type, int n, const void *x, int incx, void *y, int incy) {
    BEGIN(n, !incx || !incy);
    // y is also copied into the work buffer for COPY: with increment > 1 the gaps between elements must stay unchanged
    op_t w[2] = { { x, vec_bytes(n, incx, es) }, { y, vec_bytes(n, incy, es) } }; const int cin[2] = { 1, incy != 1 && incy != -1 }, cout[2] = { 0, 1 };
    if (load_operands(w, 2, cin, cout)) return -1;
    cl_long ox = vec_off(n, incx), oy = vec_off(n, incy), ix = incx, iy = incy;
    if (incx == 1 && incy == 1) {   // 16 B per lane in every type (copy16): FP64 17 -> 24 GB/s
        // FP16: the compiler turns vload8/vstore8 of half into SIMD8 with 4 B byte reads (16 GB/s) -- copying does not depend
        // on the type, so an even number of elements aligned to 4 B goes through the FP32 kernel (n/2 32-bit words).
        l12_t *kk = k; int nn = n, e = es;
        if (es == 2 && n % 2 == 0 && ((uintptr_t)x | (uintptr_t)y) % 4 == 0) { kk = &L12[1]; nn = n / 2; e = 4; }
        cl_kernel K = kk->copy16; const int gw = e == 2 ? 4 : 16 / e;   // = GW in level12.cl (FP16: 4, FP32: 4, FP64: 2)
        ARG(K, 0, nn); ARGB(K, 1, w[0].g); ARG(K, 2, ox); ARGB(K, 3, w[1].g); ARG(K, 4, oy);
        return start1(K, (nn + gw - 1) / gw) || finish(w, 2, cout) ? -1 : 0;
    }
    cl_kernel K = k->copy;
    ARG(K, 0, n); ARGB(K, 1, w[0].g); ARG(K, 2, ox); ARG(K, 3, ix); ARGB(K, 4, w[1].g); ARG(K, 5, oy); ARG(K, 6, iy);
    return start4(K, n) || finish(w, 2, cout) ? -1 : 0;
}

static int rot(int type, int n, void *x, int incx, void *y, int incy, double c, double s, int swap_flag) {
    BEGIN(n, !incx || !incy);
    op_t w[2] = { { x, vec_bytes(n, incx, es) }, { y, vec_bytes(n, incy, es) } }; const int cin[2] = { 1, 1 };
    if (load_operands(w, 2, cin, cin)) return -1;
    cl_long ox = vec_off(n, incx), oy = vec_off(n, incy), ix = incx, iy = incy; cl_kernel K = k->rot;
    ARG(K, 0, n); ARGB(K, 1, w[0].g); ARG(K, 2, ox); ARG(K, 3, ix); ARGB(K, 4, w[1].g); ARG(K, 5, oy); ARG(K, 6, iy);
    ARGR(K, 7, c); ARGR(K, 8, s); ARG(K, 9, swap_flag);
    return start4(K, n) || finish(w, 2, cin) ? -1 : 0;
}
// ROTM: H from param (BLAS): param[0] = flag: -1 full [h11 h12; h21 h22], 0 [1 h12; h21 1], 1 [h11 1; -1 h22],
// -2 identity (nothing to do); param[1..4] = h11, h21, h12, h22.
int gen9blas_rotm(int type, int n, void *x, int incx, void *y, int incy, const double *param) {
    double fl = param[0], h11, h12, h21, h22;
    if (fl == -2.0) return 0;
    if (fl < 0) { h11 = param[1]; h21 = param[2]; h12 = param[3]; h22 = param[4]; }
    else if (fl == 0) { h11 = 1; h21 = param[2]; h12 = param[3]; h22 = 1; }
    else { h11 = param[1]; h21 = -1; h12 = 1; h22 = param[4]; }
    BEGIN(n, !incx || !incy);
    op_t w[2] = { { x, vec_bytes(n, incx, es) }, { y, vec_bytes(n, incy, es) } }; const int cin[2] = { 1, 1 };
    if (load_operands(w, 2, cin, cin)) return -1;
    cl_long ox = vec_off(n, incx), oy = vec_off(n, incy), ix = incx, iy = incy; cl_kernel K = k->rotm;
    ARG(K, 0, n); ARGB(K, 1, w[0].g); ARG(K, 2, ox); ARG(K, 3, ix); ARGB(K, 4, w[1].g); ARG(K, 5, oy); ARG(K, 6, iy);
    ARGR(K, 7, h11); ARGR(K, 8, h12); ARGR(K, 9, h21); ARGR(K, 10, h22);
    return start4(K, n) || finish(w, 2, cin) ? -1 : 0;
}

// ROTG and ROTMG: computations on a few numbers -- on the CPU, reference algorithms (Netlib BLAS 3.10+).
// ROTG: [c s; -s c]·[a; b] = [r; 0]; *a = r, *b = z (c, s encoded in a single number), overflow-safe.
void gen9blas_rotg(float *a, float *b, float *c, float *s) {
    const float safmin = 1.17549435e-38f, safmax = 1.0f / safmin;
    float an = fabsf(*a), bn = fabsf(*b);
    if (bn == 0) { *c = 1; *s = 0; *b = 0; return; }
    if (an == 0) { *c = 0; *s = 1; *a = *b; *b = 1; return; }
    float scl = fminf(safmax, fmaxf(safmin, fmaxf(an, bn)));
    float sigma = an > bn ? copysignf(1, *a) : copysignf(1, *b);
    float r = sigma * scl * sqrtf((*a / scl) * (*a / scl) + (*b / scl) * (*b / scl));
    *c = *a / r; *s = *b / r;
    float z = an > bn ? *s : *c != 0 ? 1 / *c : 1;
    *a = r; *b = z;
}
// ROTMG: modified Givens rotation (d1, d2 -- squared scales, x1, y1); result in param (see gen9blas_rotm).
void gen9blas_rotmg(float *d1, float *d2, float *x1, float y1, float *param) {
    const float gam = 4096.0f, gamsq = 16777216.0f, rgamsq = 5.9604645e-8f;
    float fl, h11 = 0, h12 = 0, h21 = 0, h22 = 0;
    if (*d1 < 0) { fl = -1; *d1 = *d2 = *x1 = 0; }
    else {
        float p2 = *d2 * y1;
        if (p2 == 0) { param[0] = -2; return; }
        float p1 = *d1 * *x1, q2 = p2 * y1, q1 = p1 * *x1;
        if (fabsf(q1) > fabsf(q2)) {
            h21 = -y1 / *x1; h12 = p2 / p1;
            float u = 1 - h12 * h21;
            if (u > 0) { fl = 0; *d1 /= u; *d2 /= u; *x1 *= u; }
            else { fl = -1; h11 = h12 = h21 = h22 = 0; *d1 = *d2 = *x1 = 0; }
        } else if (q2 < 0) { fl = -1; h11 = h12 = h21 = h22 = 0; *d1 = *d2 = *x1 = 0; }
        else {
            fl = 1; h11 = p1 / p2; h22 = *x1 / y1;
            float u = 1 + h11 * h22, t = *d2 / u;
            *d2 = *d1 / u; *d1 = t; *x1 = y1 * u;
        }
        if (*d1 != 0)
            while (*d1 <= rgamsq || *d1 >= gamsq) {
                if (fl == 0) { h11 = 1; h22 = 1; } else if (fl > 0) { h21 = -1; h12 = 1; }
                fl = -1;
                if (*d1 <= rgamsq) { *d1 *= gam * gam; *x1 /= gam; h11 /= gam; h12 /= gam; }
                else { *d1 /= gam * gam; *x1 *= gam; h11 *= gam; h12 *= gam; }
            }
        if (*d2 != 0)
            while (fabsf(*d2) <= rgamsq || fabsf(*d2) >= gamsq) {
                if (fl == 0) { h11 = 1; h22 = 1; } else if (fl > 0) { h21 = -1; h12 = 1; }
                fl = -1;
                if (fabsf(*d2) <= rgamsq) { *d2 *= gam * gam; h21 /= gam; h22 /= gam; }
                else { *d2 /= gam * gam; h21 *= gam; h22 *= gam; }
            }
    }
    if (fl < 0) { param[1] = h11; param[2] = h21; param[3] = h12; param[4] = h22; }
    else if (fl == 0) { param[2] = h21; param[3] = h12; }
    else { param[1] = h11; param[4] = h22; }
    param[0] = fl;
}

int gen9blas_swap(int type, int n, void *x, int incx, void *y, int incy) { return rot(type, n, x, incx, y, incy, 1, 0, 1); }
int gen9blas_rot(int type, int n, void *x, int incx, void *y, int incy, double c, double s) { return rot(type, n, x, incx, y, incy, c, s, 0); }

static int alloc_red_out(void) {
    if (!red_out && !(red_out = gen9blas_alloc(2 * GROUPS * sizeof(double)))) return -1;
    return 0;
}
static int reduction(int type, int mode, int n, const void *x, int incx, const void *y, int incy, double *result) {
    *result = 0;
    BEGIN(n, !incx || (mode == 0 && !incy));
    op_t w[2] = { { x, vec_bytes(n, incx, es) }, { y, mode == 0 ? vec_bytes(n, incy, es) : 0 } }; const int cin[2] = { 1, 1 };
    if (load_operands(w, mode == 0 ? 2 : 1, cin, NULL) || alloc_red_out()) return -1;
    if (mode) { w[1] = w[0]; incy = incx; }
    int g = (n + 255) / 256 < GROUPS ? (n + 255) / 256 : GROUPS;
    cl_long ox = vec_off(n, incx), oy = vec_off(n, incy), ix = incx, iy = incy; cl_kernel K = k->red;
    ARG(K, 0, n); ARG(K, 1, mode); ARGB(K, 2, w[0].g); ARG(K, 3, ox); ARG(K, 4, ix); ARGB(K, 5, w[1].g); ARG(K, 6, oy); ARG(K, 7, iy);
    CHECK(usm_arg(K, 8, red_out));
    size_t G = (size_t)g * 256, Lw = 256;
    CHECK(clEnqueueNDRangeKernel(q, K, 1, NULL, &G, &Lw, 0, NULL, NULL));
    if (wait_gpu()) return -1;
    double s = 0; for (int i = 0; i < g; i++) s += red_out[i];
    *result = mode == 2 ? sqrt(s) : s;
    return 0;
}
int gen9blas_dot(int type, int n, const void *x, int incx, const void *y, int incy, double *result) { return reduction(type, 0, n, x, incx, y, incy, result); }
int gen9blas_asum(int type, int n, const void *x, int incx, double *result) { return reduction(type, 1, n, x, incx, NULL, 0, result); }
int gen9blas_nrm2(int type, int n, const void *x, int incx, double *result) { return reduction(type, 2, n, x, incx, NULL, 0, result); }

int gen9blas_iamax(int type, int n, const void *x, int incx, long *result) {
    *result = -1;
    BEGIN(n, !incx);
    op_t w[1] = { { x, vec_bytes(n, incx, es) } }; const int cin[1] = { 1 };
    if (load_operands(w, 1, cin, NULL) || alloc_red_out()) return -1;
    int g = (n + 255) / 256 < GROUPS ? (n + 255) / 256 : GROUPS;
    cl_long ox = vec_off(n, incx), ix = incx; cl_kernel K = k->iamax;
    ARG(K, 0, n); ARGB(K, 1, w[0].g); ARG(K, 2, ox); ARG(K, 3, ix); CHECK(usm_arg(K, 4, red_out));
    size_t G = (size_t)g * 256, Lw = 256;
    CHECK(clEnqueueNDRangeKernel(q, K, 1, NULL, &G, &Lw, 0, NULL, NULL));
    if (wait_gpu()) return -1;
    // values in the compute type (float or double), int indices after the 96 values
    const int *ci = (const int *)((const char *)red_out + GROUPS * RS(k));
    double cv[GROUPS];
    for (int i = 0; i < g; i++) cv[i] = k->es == 8 ? ((const double *)red_out)[i] : ((const float *)red_out)[i];
    int b = 0;
    for (int i = 1; i < g; i++) if (cv[i] > cv[b] || (cv[i] == cv[b] && ci[i] < ci[b])) b = i;
    *result = ci[b] == 0x7fffffff ? 0 : ci[b];   // all NaN: index 0
    return 0;
}

// GEMV on buffers (no waiting): y = alpha·op(A)·x + beta·y with the triangle mask mode (0 whole, 1 j <= i, 2 j >= i,
// 3 j < i, 4 j > i for element A[i][j]); unit = 1: x at the diagonal index is added to the sum (unit diagonal).
static int gemv_inner(l12_t *k, int trans, int M, int N, double alpha, bf_t A, int lda, bf_t x, cl_long ox, cl_long ix,
                    double beta, bf_t y, cl_long oy, cl_long iy, int mode, int unit) {
    cl_long oa = 0;
    if (!trans) {
        // Rows of 1024 columns and more: the grouped variant (16 subgroups on the same 4 rows, few pages at once). On ordinary memory
        // (userptr, 4 KB pages) the old one lost 2× (n = 4096: 10.5 -> 18.9 GB/s), on USM -5..+20%; with N <= 512
        // grouped is slower (too few columns for 16 subgroups). Measurement: bench/gemv_n.py, results/gemv-n-2026-11-12.txt.
        // GEN9BLAS_GEMV_N=old|grouped forces a variant (comparisons).
        static int forced = -1;
        if (forced < 0) { const char *e = getenv("GEN9BLAS_GEMV_N"); forced = !e ? 0 : !strcmp(e, "old") ? 1 : !strcmp(e, "grouped") ? 2 : 0; }
        const int grouped = !k->gemv_n || forced == 2 || (forced == 0 && N >= 1024);
        cl_kernel K = grouped ? k->gemv_nw : k->gemv_n;
        ARG(K, 0, M); ARG(K, 1, N); ARGR(K, 2, alpha); ARGB(K, 3, A); ARG(K, 4, oa); ARG(K, 5, lda); ARGB(K, 6, x);
        ARG(K, 7, ox); ARG(K, 8, ix); ARGR(K, 9, beta); ARGB(K, 10, y); ARG(K, 11, oy); ARG(K, 12, iy); ARG(K, 13, mode); ARG(K, 14, unit);
        size_t g = ((size_t)(M + 3) / 4 * 16 + 63) / 64 * 64, l = 64;
        if (grouped) { l = 256; g = (size_t)(M + 3) / 4 * 256; }
        CHECK(clEnqueueNDRangeKernel(q, K, 1, NULL, &g, &l, 0, NULL, NULL));
        return 0;
    }
    int cols4 = (N + 3) / 4, parts = (16384 + cols4 - 1) / cols4;   // ~16 thousand work-items
    if (parts > M) parts = M;
    int rpp = (M + parts - 1) / parts; parts = (M + rpp - 1) / rpp;
    if (ensure_buf(&dP, &sP, (size_t)parts * N * RS(k))) return -1;
    cl_kernel K;
    if (k->es == 2 && !mode && !unit) {   // FP16, full matrix: block reads (gemv_t1b; results/message-16B-...)
        K = k->gemv_t1b;
        ARG(K, 0, M); ARG(K, 1, N); ARG(K, 2, rpp); ARGB(K, 3, A); ARG(K, 4, oa); ARG(K, 5, lda); ARGB(K, 6, x);
        ARG(K, 7, ox); ARG(K, 8, ix); ARG(K, 9, dP);
    } else {
        K = k->gemv_t1;
        ARG(K, 0, M); ARG(K, 1, N); ARG(K, 2, rpp); ARGB(K, 3, A); ARG(K, 4, oa); ARG(K, 5, lda); ARGB(K, 6, x);
        ARG(K, 7, ox); ARG(K, 8, ix); ARG(K, 9, dP); ARG(K, 10, mode); ARG(K, 11, unit);
    }
    size_t g[2] = { ((size_t)cols4 + 63) / 64 * 64, (size_t)parts }, l[2] = { 64, 1 };
    CHECK(clEnqueueNDRangeKernel(q, K, 2, NULL, g, l, 0, NULL, NULL));
    K = k->gemv_t2;
    ARG(K, 0, N); ARG(K, 1, parts); ARG(K, 2, dP); ARGR(K, 3, alpha); ARGR(K, 4, beta); ARGB(K, 5, y); ARG(K, 6, oy); ARG(K, 7, iy);
    return start1(K, N);
}

int gen9blas_gemv(int type, int trans, int M, int N, double alpha, const void *A, int lda, const void *x, int incx,
             double beta, void *y, int incy) {
    BEGIN(M < N ? M : N, !incx || !incy);
    if (lda < N) FAIL("lda < N");
    int nx = trans ? M : N, ny = trans ? N : M;
    op_t w[3] = { { A, ((size_t)(M - 1) * lda + N) * es }, { x, vec_bytes(nx, incx, es) }, { y, vec_bytes(ny, incy, es) } };
    const int cin[3] = { 1, 1, beta != 0.0f || (incy != 1 && incy != -1) }, cout[3] = { 0, 0, 1 };
    if (load_operands(w, 3, cin, cout)) return -1;
    if (gemv_inner(k, trans, M, N, alpha, w[0].g, lda, w[1].g, vec_off(nx, incx), incx, beta, w[2].g, vec_off(ny, incy), incy, 0, 0)) return -1;
    return finish(w, 3, cout);
}

// SYMV: y = alpha·A·x + beta·y, A symmetric n×n, the uplo triangle is used. One pass over the stored triangle
// (level12.cl: symv1 -- partial sums of rows into Q and of columns into P, symv2 -- combination).
cl_mem dQ;
size_t sQ;
int gen9blas_symv(int type, int uplo, int n, double alpha, const void *A, int lda, const void *x, int incx,
             double beta, void *y, int incy) {
    BEGIN(n, !incx || !incy);
    if (lda < n) FAIL("lda < n");
    op_t w[3] = { { A, ((size_t)(n - 1) * lda + n) * es }, { x, vec_bytes(n, incx, es) }, { y, vec_bytes(n, incy, es) } };
    const int cin[3] = { 1, 1, beta != 0.0f || (incy != 1 && incy != -1) }, cout[3] = { 0, 0, 1 };
    if (load_operands(w, 3, cin, cout)) return -1;
    int upper = uplo == GEN9BLAS_UPPER, cols4 = (n + 3) / 4, parts = (16384 + cols4 - 1) / cols4, ng = (n + 63) / 64;
    if (parts > n) parts = n;
    int rpp = (n + parts - 1) / parts; parts = (n + rpp - 1) / rpp;
    if (ensure_buf(&dP, &sP, (size_t)parts * n * RS(k)) || ensure_buf(&dQ, &sQ, (size_t)ng * n * RS(k))) return -1;
    cl_long oa = 0, ox = vec_off(n, incx), oy = vec_off(n, incy), ix = incx, iy = incy;
    cl_kernel K = k->symv1;
    ARG(K, 0, n); ARG(K, 1, rpp); ARG(K, 2, upper); ARGB(K, 3, w[0].g); ARG(K, 4, oa); ARG(K, 5, lda); ARGB(K, 6, w[1].g);
    ARG(K, 7, ox); ARG(K, 8, ix); ARG(K, 9, dP); ARG(K, 10, dQ);
    size_t g[2] = { (size_t)ng * 16, (size_t)parts }, l[2] = { 64, 1 };
    if (g[0] % 64) g[0] = (g[0] + 63) / 64 * 64;
    CHECK(clEnqueueNDRangeKernel(q, K, 2, NULL, g, l, 0, NULL, NULL));
    K = k->symv2;
    ARG(K, 0, n); ARG(K, 1, parts); ARG(K, 2, upper); ARG(K, 3, dP); ARG(K, 4, dQ); ARGR(K, 5, alpha); ARGR(K, 6, beta);
    ARGB(K, 7, w[2].g); ARG(K, 8, oy); ARG(K, 9, iy);
    if (start1(K, n)) return -1;
    return finish(w, 3, cout);
}

static cl_mem dTv, dXf;   // TRMV: intermediate result (element type); TRSV: FP32 work vector
static size_t sTv, sXf;

// TRMV: x = op(A)·x, A triangular n×n (uplo; diag = 1: unit diagonal). The result goes to a buffer, then is copied to x.
int gen9blas_trmv(int type, int uplo, int trans, int diag, int n, const void *A, int lda, void *x, int incx) {
    BEGIN(n, !incx);
    if (lda < n) FAIL("lda < n");
    op_t w[2] = { { A, ((size_t)(n - 1) * lda + n) * es }, { x, vec_bytes(n, incx, es) } };
    const int cin[2] = { 1, 1 }, cout[2] = { 0, 1 };
    if (load_operands(w, 2, cin, cout) || ensure_buf(&dTv, &sTv, (size_t)n * es)) return -1;
    int upper = uplo == GEN9BLAS_UPPER, mode = diag ? (upper ? 4 : 3) : (upper ? 2 : 1);
    cl_long ox = vec_off(n, incx), ix = incx, z = 0, j = 1;
    if (gemv_inner(k, trans, n, n, 1.0f, w[0].g, lda, w[1].g, ox, incx, 0.0f, BM(dTv), 0, 1, mode, diag)) return -1;
    cl_kernel K = k->copy;
    ARG(K, 0, n); ARG(K, 1, dTv); ARG(K, 2, z); ARG(K, 3, j); ARGB(K, 4, w[1].g); ARG(K, 5, ox); ARG(K, 6, ix);
    return start4(K, n) || finish(w, 2, cout) ? -1 : 0;
}

// TRSV: solve op(A)·x = b (b in x, result in x). In blocks of 256: the diagonal block in a single work-group (sub-blocks of 64
// in local memory), then an update of the rest of the vector; 2 launches per block. FP32 work vector (also for FP16).
// op(A) lower: forward. (A single-launch version without global synchronization: slower, docs/pitfalls.md.)
int gen9blas_trsv(int type, int uplo, int trans, int diag, int n, const void *A, int lda, void *x, int incx) {
    BEGIN(n, !incx);
    if (lda < n) FAIL("lda < n");
    op_t w[2] = { { A, ((size_t)(n - 1) * lda + n) * es }, { x, vec_bytes(n, incx, es) } };
    const int cin[2] = { 1, 1 }, cout[2] = { 0, 1 };
    if (load_operands(w, 2, cin, cout) || ensure_buf(&dXf, &sXf, (size_t)n * RS(k))) return -1;
    cl_long ox = vec_off(n, incx), ix = incx, oa = 0;
    int lower = (uplo == GEN9BLAS_LOWER) != (trans != 0), tr = trans != 0, dg = diag != 0;
    cl_kernel K = k->to_f;
    ARG(K, 0, n); ARGB(K, 1, w[1].g); ARG(K, 2, ox); ARG(K, 3, ix); ARG(K, 4, dXf);
    if (start1(K, n)) return -1;
    const int nb = 256, nbl = (n + nb - 1) / nb;   // = NBT in level12.cl
    static int prof = -1; if (prof < 0) prof = getenv("GEN9BLAS_PROFILE") != NULL;
    cl_event ev[2 * 1024]; int ne = 0;
    if (prof && 2 * nbl > 2 * 1024) prof = 0;
    for (int s = 0; s < nbl; s++) {
        int p = (lower ? s : nbl - 1 - s) * nb, b = n - p < nb ? n - p : nb;
        K = k->trsv_block;
        ARG(K, 0, p); ARG(K, 1, b); ARGB(K, 2, w[0].g); ARG(K, 3, oa); ARG(K, 4, lda); ARG(K, 5, tr); ARG(K, 6, lower);
        ARG(K, 7, dg); ARG(K, 8, dXf);
        size_t g1 = 256, l1 = 256;
        CHECK(clEnqueueNDRangeKernel(q, K, 1, NULL, &g1, &l1, 0, NULL, prof ? &ev[ne++] : NULL));
        int r0 = lower ? p + b : 0, r1 = lower ? n : p;
        if (r1 <= r0) continue;
        K = tr ? k->trsv_upd_t : k->trsv_upd_n;
        ARG(K, 0, p); ARG(K, 1, b); ARG(K, 2, r0); ARG(K, 3, r1); ARGB(K, 4, w[0].g); ARG(K, 5, oa); ARG(K, 6, lda); ARG(K, 7, dXf);
        size_t g = ((size_t)(r1 - r0) * (tr ? 1 : 16) + 63) / 64 * 64, l = 64;
        CHECK(clEnqueueNDRangeKernel(q, K, 1, NULL, &g, &l, 0, NULL, prof ? &ev[ne++] : NULL));
    }
    if (prof) {   // GEN9BLAS_PROFILE=1: kernel times (diagonal blocks / updates) and the gaps between them
        if (wait_gpu()) return -1;
        double tb = 0, ta = 0; cl_ulong first_t = 0, last_t = 0;
        for (int i = 0; i < ne; i++) {
            cl_ulong t0, t1; clGetEventProfilingInfo(ev[i], CL_PROFILING_COMMAND_START, 8, &t0, NULL);
            clGetEventProfilingInfo(ev[i], CL_PROFILING_COMMAND_END, 8, &t1, NULL);
            if (!i) first_t = t0;
            last_t = t1; *(i % 2 ? &ta : &tb) += (t1 - t0) * 1e-6; clReleaseEvent(ev[i]);
        }
        fprintf(stderr, "trsv n=%d: blocks %.2f ms, updates %.2f ms, first to last %.2f ms (%d kernels)\n",
                n, tb, ta, (last_t - first_t) * 1e-6, ne);
    }
    K = k->from_f;
    ARG(K, 0, n); ARG(K, 1, dXf); ARGB(K, 2, w[1].g); ARG(K, 3, ox); ARG(K, 4, ix);
    return start1(K, n) || finish(w, 2, cout) ? -1 : 0;
}

// SYR / SYR2 on the uplo triangle: A = alpha·x·xᵀ + A or A = alpha·(x·yᵀ + y·xᵀ) + A.
static int syr2(int type, int uplo, int n, double alpha, const void *x, int incx, const void *y, int incy, void *A, int lda) {
    BEGIN(n, !incx || (y && !incy));
    if (lda < n) FAIL("lda < n");
    int two = y != NULL;
    op_t w[3] = { { A, ((size_t)(n - 1) * lda + n) * es }, { x, vec_bytes(n, incx, es) }, { two ? y : x, vec_bytes(n, two ? incy : incx, es) } };
    const int cin[3] = { 1, 1, 1 }, cout[3] = { 1, 0, 0 };
    if (load_operands(w, two ? 3 : 2, cin, cout)) return -1;
    if (!two) { w[2] = w[1]; incy = incx; }
    cl_long ox = vec_off(n, incx), oy = vec_off(n, incy), oa = 0, ix = incx, iy = incy;
    int mode = uplo == GEN9BLAS_UPPER ? 2 : 1; cl_kernel K = k->syr2;
    ARG(K, 0, n); ARGR(K, 1, alpha); ARGB(K, 2, w[1].g); ARG(K, 3, ox); ARG(K, 4, ix); ARGB(K, 5, w[2].g); ARG(K, 6, oy);
    ARG(K, 7, iy); ARGB(K, 8, w[0].g); ARG(K, 9, oa); ARG(K, 10, lda); ARG(K, 11, mode); ARG(K, 12, two);
    size_t g[2] = { ((size_t)n + 63) / 64 * 64, (size_t)n }, l[2] = { 64, 1 };
    CHECK(clEnqueueNDRangeKernel(q, K, 2, NULL, g, l, 0, NULL, NULL));
    return finish(w, 1, cout);
}
int gen9blas_syr(int type, int uplo, int n, double alpha, const void *x, int incx, void *A, int lda) {
    return syr2(type, uplo, n, alpha, x, incx, NULL, 0, A, lda);
}
int gen9blas_syr2(int type, int uplo, int n, double alpha, const void *x, int incx, const void *y, int incy, void *A, int lda) {
    return syr2(type, uplo, n, alpha, x, incx, y, incy, A, lda);
}

int gen9blas_ger(int type, int M, int N, double alpha, const void *x, int incx, const void *y, int incy, void *A, int lda) {
    BEGIN(M < N ? M : N, !incx || !incy);
    if (lda < N) FAIL("lda < N");
    op_t w[3] = { { x, vec_bytes(M, incx, es) }, { y, vec_bytes(N, incy, es) }, { A, ((size_t)(M - 1) * lda + N) * es } };
    const int cin[3] = { 1, 1, 1 }, cout[3] = { 0, 0, 1 };
    if (load_operands(w, 3, cin, cout)) return -1;
    cl_long ox = vec_off(M, incx), oy = vec_off(N, incy), oa = 0, ix = incx, iy = incy; cl_kernel K = k->ger;
    ARG(K, 0, M); ARG(K, 1, N); ARGR(K, 2, alpha); ARGB(K, 3, w[0].g); ARG(K, 4, ox); ARG(K, 5, ix); ARGB(K, 6, w[1].g);
    ARG(K, 7, oy); ARG(K, 8, iy); ARGB(K, 9, w[2].g); ARG(K, 10, oa); ARG(K, 11, lda);
    size_t g[2] = { ((size_t)N + 63) / 64 * 64, (size_t)M }, l[2] = { 64, 1 };
    CHECK(clEnqueueNDRangeKernel(q, K, 2, NULL, g, l, 0, NULL, NULL));
    return finish(w, 3, cout);
}

// ---------------- level 2: banded and packed matrices (row-major layout as in CBLAS RowMajor; see level12.cl) ----------------
enum { BAND = 0, PACK_U = 1, PACK_L = 2 };
static int elem_size(int type) { return type == GEN9BLAS_S ? 4 : type == GEN9BLAS_D ? 8 : 2; }
static size_t band_bytes(int rows, int lda, int KL, int KU, int es) { return ((size_t)(rows - 1) * lda + KL + KU + 1) * es; }
static size_t packed_bytes(int n, int es) { return (size_t)n * (n + 1) / 2 * es; }

// y = alpha·op(A)·x + beta·y (A in buffer a, format description) -- no waiting; y may be an intermediate buffer.
static int pmv_inner(l12_t *k, int ny, int nx, int fmt, int KL, int KU, int n, bf_t a, int lda, bf_t x, cl_long ox, cl_long ix,
                   double alpha, double beta, bf_t y, cl_long oy, cl_long iy, int tr, int sym, int upper, int unit, int lo, int hi) {
    cl_long oa = 0; cl_kernel K = k->pmv;
    ARG(K, 0, ny); ARG(K, 1, nx); ARG(K, 2, fmt); ARG(K, 3, KL); ARG(K, 4, KU); ARG(K, 5, n); ARGB(K, 6, a); ARG(K, 7, oa);
    ARG(K, 8, lda); ARGB(K, 9, x); ARG(K, 10, ox); ARG(K, 11, ix); ARGR(K, 12, alpha); ARGR(K, 13, beta); ARGB(K, 14, y);
    ARG(K, 15, oy); ARG(K, 16, iy); ARG(K, 17, tr); ARG(K, 18, sym); ARG(K, 19, upper); ARG(K, 20, unit); ARG(K, 21, lo); ARG(K, 22, hi);
    return start1(K, ny);
}
// Product with the result in y (GBMV, SBMV, SPMV).
static int pmv_y(int type, int M, int N, int fmt, int KL, int KU, int tr, int sym, int upper, int lo, int hi,
                 double alpha, const void *A, size_t bA, int lda, const void *x, int incx, double beta, void *y, int incy) {
    int nx = tr ? M : N, ny = tr ? N : M;
    BEGIN(nx < ny ? nx : ny, !incx || !incy);
    op_t w[3] = { { A, bA }, { x, vec_bytes(nx, incx, es) }, { y, vec_bytes(ny, incy, es) } };
    const int cin[3] = { 1, 1, beta != 0.0f || (incy != 1 && incy != -1) }, cout[3] = { 0, 0, 1 };
    if (load_operands(w, 3, cin, cout)) return -1;
    if (pmv_inner(k, ny, nx, fmt, KL, KU, N, w[0].g, lda, w[1].g, vec_off(nx, incx), incx, alpha, beta, w[2].g, vec_off(ny, incy), incy,
                tr, sym, upper, 0, lo, hi)) return -1;
    return finish(w, 3, cout);
}
int gen9blas_gbmv(int type, int trans, int M, int N, int KL, int KU, double alpha, const void *A, int lda, const void *x, int incx,
             double beta, void *y, int incy) {
    if (KL < 0 || KU < 0 || lda < KL + KU + 1) FAIL("gbmv: KL, KU, lda");
    int es = elem_size(type);
    return pmv_y(type, M, N, BAND, KL, KU, trans != 0, 0, 0, trans ? KU : KL, trans ? KL : KU, alpha, A,
                 M > 0 ? band_bytes(M, lda, KL, KU, es) : 0, lda, x, incx, beta, y, incy);
}
int gen9blas_sbmv(int type, int uplo, int N, int K, double alpha, const void *A, int lda, const void *x, int incx,
             double beta, void *y, int incy) {
    if (K < 0 || lda < K + 1) FAIL("sbmv: K, lda");
    int es = elem_size(type), g = uplo == GEN9BLAS_UPPER;
    return pmv_y(type, N, N, BAND, g ? 0 : K, g ? K : 0, 0, 1, g, K, K, alpha, A, N > 0 ? band_bytes(N, lda, 0, K, es) : 0,
                 lda, x, incx, beta, y, incy);
}
int gen9blas_spmv(int type, int uplo, int N, double alpha, const void *Ap, const void *x, int incx, double beta, void *y, int incy) {
    int es = elem_size(type), g = uplo == GEN9BLAS_UPPER;
    return pmv_y(type, N, N, g ? PACK_U : PACK_L, 0, 0, 0, 1, g, N, N, alpha, Ap, packed_bytes(N, es), 0, x, incx, beta, y, incy);
}

// x = op(A)·x (TBMV, TPMV): the result goes to an intermediate buffer, then is copied to x.
static int pmv_x(int type, int N, int fmt, int KL, int KU, int tr, int unit, int lo, int hi, const void *A, size_t bA, int lda,
                 void *x, int incx) {
    BEGIN(N, !incx);
    op_t w[2] = { { A, bA }, { x, vec_bytes(N, incx, es) } };
    const int cin[2] = { 1, 1 }, cout[2] = { 0, 1 };
    if (load_operands(w, 2, cin, cout) || ensure_buf(&dTv, &sTv, (size_t)N * es)) return -1;
    cl_long ox = vec_off(N, incx), ix = incx, z = 0, j = 1;
    if (pmv_inner(k, N, N, fmt, KL, KU, N, w[0].g, lda, w[1].g, ox, ix, 1.0f, 0.0f, BM(dTv), 0, 1, tr, 0, 0, unit, lo, hi)) return -1;
    cl_kernel K = k->copy;
    ARG(K, 0, N); ARG(K, 1, dTv); ARG(K, 2, z); ARG(K, 3, j); ARGB(K, 4, w[1].g); ARG(K, 5, ox); ARG(K, 6, ix);
    return start4(K, N) || finish(w, 2, cout) ? -1 : 0;
}
// op(A)·x = b (TBSV, TPSV): a single work-group, row by row (inherently sequential), FP32 work vector.
static int psv(int type, int N, int fmt, int KL, int KU, int tr, int lower, int unit, int K_, const void *A, size_t bA, int lda,
               void *x, int incx) {
    BEGIN(N, !incx);
    op_t w[2] = { { A, bA }, { x, vec_bytes(N, incx, es) } };
    const int cin[2] = { 1, 1 }, cout[2] = { 0, 1 };
    if (load_operands(w, 2, cin, cout) || ensure_buf(&dXf, &sXf, (size_t)N * RS(k))) return -1;
    cl_long ox = vec_off(N, incx), ix = incx, oa = 0;
    cl_kernel K = k->to_f;
    ARG(K, 0, N); ARGB(K, 1, w[1].g); ARG(K, 2, ox); ARG(K, 3, ix); ARG(K, 4, dXf);
    if (start1(K, N)) return -1;
    K = k->psv;
    ARG(K, 0, N); ARG(K, 1, fmt); ARG(K, 2, KL); ARG(K, 3, KU); ARGB(K, 4, w[0].g); ARG(K, 5, oa); ARG(K, 6, lda); ARG(K, 7, dXf);
    ARG(K, 8, tr); ARG(K, 9, lower); ARG(K, 10, unit); ARG(K, 11, K_);
    size_t g = 256, l = 256;
    CHECK(clEnqueueNDRangeKernel(q, K, 1, NULL, &g, &l, 0, NULL, NULL));
    K = k->from_f;
    ARG(K, 0, N); ARG(K, 1, dXf); ARGB(K, 2, w[1].g); ARG(K, 3, ox); ARG(K, 4, ix);
    return start1(K, N) || finish(w, 2, cout) ? -1 : 0;
}
// op(A) lower <=> (uplo lower) xor trans. Nonzero range of op(A) in a row: lower [i − K, i], upper [i, i + K].
int gen9blas_tbmv(int type, int uplo, int trans, int diag, int N, int K, const void *A, int lda, void *x, int incx) {
    if (K < 0 || lda < K + 1) FAIL("tbmv: K, lda");
    int es = elem_size(type), g = uplo == GEN9BLAS_UPPER, low = !g != (trans != 0);
    return pmv_x(type, N, BAND, g ? 0 : K, g ? K : 0, trans != 0, diag != 0, low ? K : 0, low ? 0 : K, A,
                 N > 0 ? band_bytes(N, lda, 0, K, es) : 0, lda, x, incx);
}
int gen9blas_tpmv(int type, int uplo, int trans, int diag, int N, const void *Ap, void *x, int incx) {
    int es = elem_size(type), g = uplo == GEN9BLAS_UPPER, low = !g != (trans != 0);
    return pmv_x(type, N, g ? PACK_U : PACK_L, 0, 0, trans != 0, diag != 0, low ? N : 0, low ? 0 : N, Ap, packed_bytes(N, es), 0, x, incx);
}
int gen9blas_tbsv(int type, int uplo, int trans, int diag, int N, int K, const void *A, int lda, void *x, int incx) {
    if (K < 0 || lda < K + 1) FAIL("tbsv: K, lda");
    int es = elem_size(type), g = uplo == GEN9BLAS_UPPER;
    return psv(type, N, BAND, g ? 0 : K, g ? K : 0, trans != 0, !g != (trans != 0), diag != 0, K, A,
               N > 0 ? band_bytes(N, lda, 0, K, es) : 0, lda, x, incx);
}
int gen9blas_tpsv(int type, int uplo, int trans, int diag, int N, const void *Ap, void *x, int incx) {
    int es = elem_size(type), g = uplo == GEN9BLAS_UPPER;
    return psv(type, N, g ? PACK_U : PACK_L, 0, 0, trans != 0, !g != (trans != 0), diag != 0, N, Ap, packed_bytes(N, es), 0, x, incx);
}
// SPR / SPR2: A = alpha·x·xᵀ + A, A = alpha·(x·yᵀ + y·xᵀ) + A (packed, uplo triangle).
static int spr2(int type, int uplo, int n, double alpha, const void *x, int incx, const void *y, int incy, void *Ap) {
    BEGIN(n, !incx || (y && !incy));
    int two = y != NULL, fmt = uplo == GEN9BLAS_UPPER ? PACK_U : PACK_L;
    op_t w[3] = { { Ap, packed_bytes(n, es) }, { x, vec_bytes(n, incx, es) }, { two ? y : x, vec_bytes(n, two ? incy : incx, es) } };
    const int cin[3] = { 1, 1, 1 }, cout[3] = { 1, 0, 0 };
    if (load_operands(w, two ? 3 : 2, cin, cout)) return -1;
    if (!two) { w[2] = w[1]; incy = incx; }
    cl_long ox = vec_off(n, incx), oy = vec_off(n, incy), oa = 0, ix = incx, iy = incy; cl_kernel K = k->spr2;
    ARG(K, 0, n); ARG(K, 1, fmt); ARGR(K, 2, alpha); ARGB(K, 3, w[1].g); ARG(K, 4, ox); ARG(K, 5, ix); ARGB(K, 6, w[2].g);
    ARG(K, 7, oy); ARG(K, 8, iy); ARGB(K, 9, w[0].g); ARG(K, 10, oa); ARG(K, 11, two);
    size_t g[2] = { ((size_t)n + 63) / 64 * 64, (size_t)n }, l[2] = { 64, 1 };
    CHECK(clEnqueueNDRangeKernel(q, K, 2, NULL, g, l, 0, NULL, NULL));
    return finish(w, 1, cout);
}
int gen9blas_spr(int type, int uplo, int n, double alpha, const void *x, int incx, void *Ap) { return spr2(type, uplo, n, alpha, x, incx, NULL, 0, Ap); }
int gen9blas_spr2(int type, int uplo, int n, double alpha, const void *x, int incx, const void *y, int incy, void *Ap) {
    return spr2(type, uplo, n, alpha, x, incx, y, incy, Ap);
}
