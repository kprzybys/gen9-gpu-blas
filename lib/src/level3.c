// Level 3 BLAS built on GEMM (SYMM, SYRK, SYR2K, TRMM, TRSM), entirely on the GPU.
// Split from lib/gen9blas.c (refactor 2026-11-15); shared state in internal.h.
#include "internal.h"

// ---------------- level 3 built on GEMM ----------------

static cl_mem dT, dX, dY;   // work buffers: full matrix (SYMM, TRMM), X and Y (SYR2K)
static size_t sT, sX, sY;
static int num_blocks(int n) { const char *e = getenv("GEN9BLAS_BLOCKS"); if (e) return atoi(e); return n >= 512 ? 4 : 1; }

// dT = full n×n matrix from the uplo triangle of A (host memory), on the GPU: sym=1 symmetric (mirrored),
// sym=0 triangular with zeros (diag=1: unit diagonal); tr=1: transposed result.
static int full_dev(type_t *r, const void *A, int lda, int n, int uplo, int sym, int diag, int tr) {
    size_t bA = ((size_t)(n - 1) * lda + n) * r->es; bf_t xA; int z;
    if (ensure_buf(&dT, &sT, (size_t)n * n * r->es) || get_obj(A, bA, &uA, 1, 0, &xA, &z)) return -1;
    cl_kernel k = r->expand;
    ARGB(k, 0, xA); ARG(k, 1, lda); ARG(k, 2, dT); ARG(k, 3, n); ARG(k, 4, uplo); ARG(k, 5, sym); ARG(k, 6, diag); ARG(k, 7, tr);
    size_t g[2] = { (size_t)(n + 15) / 16 * 16, (size_t)(n + 15) / 16 * 16 }, l[2] = { 16, 16 };
    CHECK(clEnqueueNDRangeKernel(q, k, 2, NULL, g, l, 0, NULL, NULL));
    return 0;
}

// C = alpha·F·B + beta·C (left) or alpha·B·F + beta·C (right), F = dT (n×n, n = M or N), B and C in host memory.
// shape F: 0 full, 1 lower, 2 upper -- then p bands (GEN9BLAS_BLOCKS), each only over the nonzero range of k (~62%
// of the work). C may be the same region as B (TRMM, beta = 0): B is packed before the epilogue (queue order).
static int f_times_b(type_t *r, int left, int shape, int M, int N, const void *B, int ldb, double alpha, double beta,
                    void *C, int ldc) {
    double t_in = now_s();
    int es = r->es, n = left ? M : N, w;
    int Mp = (M + 63) / 64 * 64, Np = (N + 63) / 64 * 64, Kp = left ? Mp : Np;
    if (ensure_buf(&dAp, &sAp, (size_t)Mp * Kp * es + 4096) || ensure_buf(&dBp, &sBp, (size_t)Np * Kp * es + 4096) ||
        ensure_buf(&dCp, &sCp, (size_t)Mp * Np * es))
        return -1;
    if (left) w = pack_dev(r, 0, 0, BM(dT), n, M, M, Mp, Kp, dAp, 0, 0) || pack_host(r, 1, 0, B, ldb, N, M, Np, Kp, dBp);
    else      w = pack_host(r, 0, 0, B, ldb, M, N, Mp, Kp, dAp) || pack_dev(r, 1, 0, BM(dT), n, N, N, Np, Kp, dBp, 0, 0);
    if (w) return -1;
    int p = shape ? num_blocks(n) : 1, dim = left ? Mp : Np, nb = ((dim + p - 1) / p + 63) / 64 * 64;
    nev = 0;
    for (int i0 = 0; i0 < dim && !w; i0 += nb) {
        int b = dim - i0 < nb ? dim - i0 : nb;
        if (left) {   // rows i0..i0+b: k in [0, i0+b) (lower) or [i0, Kp) (upper)
            int k0 = shape == 2 ? i0 : 0, k1 = shape == 1 ? i0 + b : Kp;
            w = launch(r, dAp, ((size_t)i0 * Kp + (size_t)k0 * 16) * es, dBp, (size_t)k0 * r->tn * es, dCp, (size_t)i0 * Np * es,
                        b, Np, Np, k1 - k0, Kp, 1);
        } else {      // columns i0..i0+b: k in [i0, Kp) (lower) or [0, i0+b) (upper)
            int k0 = shape == 1 ? i0 : 0, k1 = shape == 2 ? i0 + b : Kp;
            w = launch(r, dAp, (size_t)k0 * 16 * es, dBp, ((size_t)i0 * Kp + (size_t)k0 * r->tn) * es, dCp, (size_t)i0 * es,
                        Mp, b, Np, k1 - k0, Kp, 1);
        }
    }
    if (w || epilogue_host(r, dCp, Np, C, M, N, ldc, alpha, beta, 2)) return -1;
    { double tz = event_times(); last_ms = tz >= 0 ? tz : (now_s() - t_in) * 1e3; }
    return 0;
}

int gen9blas_symm(int type, int side, int uplo, int M, int N, double alpha, const void *A, int lda,
             const void *B, int ldb, double beta, void *C, int ldc) {
    CHECK_TYPE(type);
    type_t *r = &types[type]; int left = side == GEN9BLAS_LEFT;
    if (full_dev(r, A, lda, left ? M : N, uplo, 1, 0, 0)) return -1;
    return f_times_b(r, left, 0, M, N, B, ldb, alpha, beta, C, ldc);
}

int gen9blas_trmm(int type, int side, int uplo, int transA, int diag, int M, int N, double alpha,
             const void *A, int lda, void *B, int ldb) {
    // B = alpha·op(A)·B (left) or alpha·B·op(A) (right). F = op(A) explicitly (lower when uplo=L xor transA).
    CHECK_TYPE(type);
    type_t *r = &types[type]; int left = side == GEN9BLAS_LEFT, lower = (uplo == GEN9BLAS_LOWER) != (transA != 0);
    if (full_dev(r, A, lda, left ? M : N, uplo, 0, diag, transA)) return -1;
    return f_times_b(r, left, lower ? 1 : 2, M, N, B, ldb, alpha, 0.0f, B, ldb);
}

// Triangle (uplo) of alpha·op(X)·op(Y)ᵀ + beta·C, op(X), op(Y): N×K (tX: X, Y stored as K×N), X and Y on the GPU.
// First packing (tri_pack), then p kernel launches
// on row bands: lower triangle -> columns 0..i0+b, upper -> i0..N (~(p+1)/(2p) of the work), epilogue on the triangle.
static int tri_pack(type_t *r, int tX, int N, int K, bf_t X, int ldx, bf_t Y, int ldy) {
    int es = r->es, Np = (N + 63) / 64 * 64, Kp = (K + r->km - 1) / r->km * r->km;
    if (ensure_buf(&dAp, &sAp, (size_t)Np * Kp * es + 4096) || ensure_buf(&dBp, &sBp, (size_t)Np * Kp * es + 4096) ||
        ensure_buf(&dCp, &sCp, (size_t)Np * Np * es))
        return -1;
    return pack_dev(r, 0, tX, X, ldx, N, K, Np, Kp, dAp, 0, 0) || pack_dev(r, 1, !tX, Y, ldy, N, K, Np, Kp, dBp, 0, 0) ? -1 : 0;
}
static int tri_compute(type_t *r, int uplo, int N, int K, double alpha, double beta, void *C, int ldc) {
    double t_in = now_s();
    int es = r->es, Np = (N + 63) / 64 * 64, Kp = (K + r->km - 1) / r->km * r->km;
    int p = num_blocks(N), nb = ((Np + p - 1) / p + 63) / 64 * 64;
    nev = 0;
    for (int i0 = 0; i0 < Np; i0 += nb) {
        int b = Np - i0 < nb ? Np - i0 : nb;
        int c0 = uplo == GEN9BLAS_LOWER ? 0 : i0, c1 = uplo == GEN9BLAS_LOWER ? i0 + b : Np;
        if (launch(r, dAp, (size_t)i0 * Kp * es, dBp, (size_t)c0 * Kp * es, dCp, ((size_t)i0 * Np + c0) * es,
                    b, c1 - c0, Np, Kp, Kp, 1)) return -1;
    }
    if (epilogue_host(r, dCp, Np, C, N, N, ldc, alpha, beta, uplo)) return -1;
    { double tz = event_times(); last_ms = tz >= 0 ? tz : (now_s() - t_in) * 1e3; }
    return 0;
}

int gen9blas_syrk(int type, int uplo, int trans, int N, int K, double alpha, const void *A, int lda,
             double beta, void *C, int ldc) {
    CHECK_TYPE(type);
    type_t *r = &types[type];
    size_t bA = (trans ? (size_t)(K - 1) * lda + N : (size_t)(N - 1) * lda + K) * r->es; bf_t xA; int z;
    if (get_obj(A, bA, &uA, 1, 0, &xA, &z) || tri_pack(r, trans, N, K, xA, lda, xA, lda)) return -1;
    return tri_compute(r, uplo, N, K, alpha, beta, C, ldc);
}

int gen9blas_syr2k(int type, int uplo, int trans, int N, int K, double alpha, const void *A, int lda,
              const void *B, int ldb, double beta, void *C, int ldc) {
    // alpha·(op(A)·op(B)ᵀ + op(B)·op(A)ᵀ) = alpha·X·Yᵀ, X = [op(A) | op(B)], Y = [op(B) | op(A)] (N × 2K, joined on the GPU)
    CHECK_TYPE(type);
    type_t *r = &types[type]; int es = r->es, K2 = 2 * K;
    size_t bA = (trans ? (size_t)(K - 1) * lda + N : (size_t)(N - 1) * lda + K) * es;
    size_t bB = (trans ? (size_t)(K - 1) * ldb + N : (size_t)(N - 1) * ldb + K) * es;
    int zA, zB; bf_t xA, xB;
    if (ensure_buf(&dX, &sX, (size_t)N * K2 * es) || ensure_buf(&dY, &sY, (size_t)N * K2 * es) ||
        get_obj(A, bA, &uA, 1, 0, &xA, &zA) || get_obj(B, bB, &uB, 1, 0, &xB, &zB))
        return -1;
    cl_kernel k = r->syr2k_merge;
    ARGB(k, 0, xA); ARG(k, 1, lda); ARGB(k, 2, xB); ARG(k, 3, ldb); ARG(k, 4, dX); ARG(k, 5, dY); ARG(k, 6, K); ARG(k, 7, trans);
    size_t g[2] = { (size_t)K, (size_t)N };
    CHECK(clEnqueueNDRangeKernel(q, k, 2, NULL, g, NULL, 0, NULL, NULL));
    if (tri_pack(r, 0, N, K2, BM(dX), K2, BM(dY), K2)) return -1;
    return tri_compute(r, uplo, N, K2, alpha, beta, C, ldc);
}

int gen9blas_trsm(int type, int side, int uplo, int transA, int diag, int M, int N, double alpha,
             const void *A, int lda, void *B, int ldb) {
    // Solves op(A)·X = alpha·B (left) or X·op(A) = alpha·B (right); X overwrites B. FP32 and FP16 (GEN9BLAS_H, GEN9BLAS_HS).
    // Form L·Y = alpha·R with L lower (an upper one is reduced by reversing the index order, the right side by transposition).
    // Everything on the GPU: building L and Y from A, B (trsm_L/trsm_Y), inverses of the diagonal blocks of L in FP64 (trsm_inv),
    // L packed once, block loop (nb = 256): Y_i -= L[i, <i]·Y[<i]; Y_i = inv(L_ii)·Y_i; packing Y_i afterwards;
    // finally the write to B (trsm_zapisz).
    CHECK_TYPE(type);
    type_t *r = &types[type]; int es = r->es;
    int left = side == GEN9BLAS_LEFT, n = left ? M : N, m = left ? N : M;
    int tr = left ? transA : !transA, lower = (uplo == GEN9BLAS_LOWER) != tr, rev = !lower;
    const int nb = 256;
    int np = (n + 63) / 64 * 64, mp = (m + 63) / 64 * 64, w = 0, nbl = (n + nb - 1) / nb;
    double tp0 = now_s();
    // A and B for the GPU (B: input and output)
    size_t bA = ((size_t)(n - 1) * lda + n) * es, bB = ((size_t)(M - 1) * ldb + N) * es;
    bf_t xA, xB; int zA, zB; cl_event evi = NULL, evi2 = NULL;
    if (get_obj(A, bA, &uA, 1, 0, &xA, &zA) || get_obj(B, bB, &uB, 1, 1, &xB, &zB)) return -1;
    // TRSM work buffers (kept between calls): L, Y, Y packed, inv(L_ii) packed, R_i packed,
    // block inverses (element type), X and Tm (FP64, inversion)
    static cl_mem Ldv, Yd, Yp, Ip, Rp, Inv, Xd, Tm;
    static size_t sLdv, sYd, sYp, sIp, sRp, sInv, sXd, sTm;
    size_t blk = (size_t)nbl * nb * nb;
    if (ensure_buf(&Ldv, &sLdv, (size_t)n * n * es) || ensure_buf(&Yd, &sYd, (size_t)np * mp * es) ||
        ensure_buf(&Yp, &sYp, (size_t)mp * np * es + 4096) || ensure_buf(&Ip, &sIp, (size_t)nb * nb * es + 4096) ||
        ensure_buf(&Rp, &sRp, (size_t)mp * nb * es + 4096) || ensure_buf(&Inv, &sInv, blk * es) ||
        ensure_buf(&Xd, &sXd, blk * sizeof(double)) || ensure_buf(&Tm, &sTm, blk * sizeof(double)))
        return -1;
    if (ensure_buf(&dAp, &sAp, (size_t)np * np * es + 4096) || ensure_buf(&dCp, &sCp, (size_t)nb * mp * es)) w = -1;
    if (!w) {   // L = f(A), Y = alpha·R(B), inverses of the L blocks; L packed once
        cl_kernel k = r->tL; int u = uplo;
        ARGB(k, 0, xA); ARG(k, 1, lda); ARG(k, 2, Ldv); ARG(k, 3, n); ARG(k, 4, u); ARG(k, 5, tr); ARG(k, 6, diag); ARG(k, 7, rev);
        size_t g[2] = { (size_t)n, (size_t)n };
        CHECK(clEnqueueNDRangeKernel(q, k, 2, NULL, g, NULL, 0, NULL, NULL));
        k = r->tY;
        ARGB(k, 0, xB); ARG(k, 1, ldb); ARG(k, 2, Yd); ARG(k, 3, n); ARG(k, 4, m); ARG(k, 5, mp); ARG(k, 6, left); ARG(k, 7, rev); ARGS(k, 8, alpha);
        size_t gy[2] = { (size_t)mp, (size_t)np };
        CHECK(clEnqueueNDRangeKernel(q, k, 2, NULL, gy, NULL, 0, NULL, NULL));
        // inverses: 32×32 base blocks, then levels h = 32..nb/2: X21 = −X22·(L21·X11), finally converted to the element type
        k = r->tI[0];
        ARG(k, 0, Ldv); ARG(k, 1, n); ARG(k, 2, nb); ARG(k, 3, Xd);
        size_t g0[2] = { 32, (size_t)nbl * nb / 32 };
        CHECK(clEnqueueNDRangeKernel(q, k, 2, NULL, g0, NULL, 0, NULL, &evi));
        for (int h = 32; h < nb; h *= 2)
            for (int step = 1; step <= 2; step++) {
                k = r->tI[step];
                ARG(k, 0, Ldv); ARG(k, 1, n); ARG(k, 2, nb); ARG(k, 3, Xd); ARG(k, 4, Tm); ARG(k, 5, h);
                size_t gh[3] = { (size_t)h, (size_t)h, (size_t)nbl * nb / (2 * h) };
                CHECK(clEnqueueNDRangeKernel(q, k, 3, NULL, gh, NULL, 0, NULL, NULL));
            }
        k = r->tI[3];
        ARG(k, 0, n); ARG(k, 1, nb); ARG(k, 2, Xd); ARG(k, 3, Inv);
        size_t g3[3] = { (size_t)nb, (size_t)nb, (size_t)nbl };
        CHECK(clEnqueueNDRangeKernel(q, k, 3, NULL, g3, NULL, 0, NULL, &evi2));
        w = pack_dev(r, 0, 0, BM(Ldv), n, n, n, np, np, dAp, 0, 0);
    }
    for (int i0 = 0; i0 < n && !w; i0 += nb) {
        int b = n - i0 < nb ? n - i0 : nb, bp = (b + 63) / 64 * 64;
        cl_mem Yi = subbuffer(Yd, (size_t)i0 * mp * es, (size_t)bp * mp * es);
        cl_mem Ii = subbuffer(Inv, (size_t)i0 * nb * es, (size_t)b * b * es);
        if (!Yi || !Ii) { w = -1; if (Yi) clReleaseMemObject(Yi); break; }
        if (i0 > 0)   // Y_i -= L[i0:i0+b, 0:i0] · Y[0:i0]
            w = launch(r, dAp, (size_t)i0 * np * es, Yp, 0, dCp, 0, bp, mp, mp, i0, np, 0) ||
                epilogue_dev(r, dCp, mp, Yi, mp, b, m, -1.0, 1.0);
        if (!w)       // Y_i = inv(L_ii) · Y_i
            w = pack_dev(r, 0, 0, BM(Ii), b, b, b, bp, bp, Ip, 0, 0) ||
                pack_dev(r, 1, 0, BM(Yi), mp, m, b, mp, bp, Rp, 0, 0) ||
                launch(r, Ip, 0, Rp, 0, dCp, 0, bp, mp, mp, bp, bp, 0) ||
                epilogue_dev(r, dCp, mp, Yi, mp, b, m, 1.0, 0.0);
        if (!w)       // packing the solved block (rows k = i0..i0+b in Yp)
            w = pack_dev(r, 1, 0, BM(Yd), mp, m, n, mp, np, Yp, i0, bp);
        clReleaseMemObject(Yi); clReleaseMemObject(Ii);
    }
    if (!w) {         // write to B on the GPU, then to host memory
        cl_kernel k = r->tZ;
        ARG(k, 0, Yd); ARG(k, 1, mp); ARGB(k, 2, xB); ARG(k, 3, ldb); ARG(k, 4, n); ARG(k, 5, left); ARG(k, 6, rev);
        size_t g[2] = { (size_t)m, (size_t)n };
        CHECK(clEnqueueNDRangeKernel(q, k, 2, NULL, g, NULL, 0, NULL, NULL));
        w = to_host(xB, zB, bB, B);
    }
    last_ms = (now_s() - tp0) * 1e3;   // TRSM: time of the whole call
    if (evi && evi2) {
        if (getenv("GEN9BLAS_PROFILE")) { cl_ulong t0, t1; clGetEventProfilingInfo(evi, CL_PROFILING_COMMAND_START, 8, &t0, NULL);
            clGetEventProfilingInfo(evi2, CL_PROFILING_COMMAND_END, 8, &t1, NULL);
            fprintf(stderr, "trsm n=%d m=%d: %.1f ms (block inverses %.1f ms)\n", n, m, last_ms, (t1 - t0) * 1e-6); }
        clReleaseEvent(evi); clReleaseEvent(evi2);
    }
    return w;
}
