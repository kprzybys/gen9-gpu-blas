// GEMM: batched GEMM on host memory (gen9blas_gemm_batched), the device path that packs once and launches on chunks, skinny GEMM (small M or N).
// Split from lib/gen9blas.c (refactor 2026-11-15); shared state in internal.h.
#include "internal.h"

int gen9blas_gemm_batched(int type, int tA, int tB, int M, int N, int K, double alpha,
                     const void *A, int lda, long sA, const void *B, int ldb, long sB,
                     double beta, void *C, int ldc, long sC, int count) {
    NEW_OPERATION();
    if (type < 0 || type > 3 || !types[type].ready) FAIL("type %d unavailable", type);
    if (M <= 0 || N <= 0 || K <= 0 || count <= 0) return 0;
    if (count == 1 && (M < N ? M : N) <= skinny_threshold(type))   // small M or N: skinny kernel without packing
        return gemm_skinny(type, tA, tB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    double t_start = now_s();   // GEN9BLAS_PROFILE=1: stage times on stderr
    type_t *r = &types[type];
    int es = r->es, Mp = (M + 15) / 16 * 16, Np = (N + r->tn - 1) / r->tn * r->tn, Kp = (K + r->km - 1) / r->km * r->km;
    size_t c1 = count - 1;
    size_t bA = c1 * sA * es + (tA ? (size_t)(K - 1) * lda + M : (size_t)(M - 1) * lda + K) * es;
    size_t bB = c1 * sB * es + (tB ? (size_t)(N - 1) * ldb + K : (size_t)(K - 1) * ldb + N) * es;
    size_t bC = c1 * sC * es + ((size_t)(M - 1) * ldc + N) * es;
    bf_t xA, xB, xC; int zA, zB, zC;
    if (get_obj(A, bA, &uA, 1, 0, &xA, &zA) || get_obj(B, bB, &uB, 1, 0, &xB, &zB) || get_obj(C, bC, &uC, beta != 0.0f, 1, &xC, &zC))
        return -1;
    last_zc = zA | zB << 1 | zC << 2;
    if (ensure_buf(&dAp, &sAp, (size_t)count * Mp * Kp * es + 4096) || ensure_buf(&dBp, &sBp, (size_t)count * Np * Kp * es + 4096) ||
        ensure_buf(&dCp, &sCp, (size_t)count * Mp * Np * es))
        return -1;
    cl_long lA = sA, lB = sB, lC = sC;
    size_t ga[3], gb[3];
    cl_kernel ka = pack_kernel(r, 0, tA, Mp, Kp, count, ga), kb = pack_kernel(r, 1, tB, Np, Kp, count, gb);
    ARGB(ka, 0, xA); ARG(ka, 1, dAp); ARG(ka, 2, M); ARG(ka, 3, K); ARG(ka, 4, lda); ARG(ka, 5, Kp); ARG(ka, 6, lA); ARG(ka, 7, Mp);
    double t_buf = now_s() - t_start; cl_event evp[3];
    CHECK(clEnqueueNDRangeKernel(q, ka, 3, NULL, ga, NULL, 0, NULL, &evp[0]));
    ARGB(kb, 0, xB); ARG(kb, 1, dBp); ARG(kb, 2, K); ARG(kb, 3, N); ARG(kb, 4, ldb); ARG(kb, 5, Kp); ARG(kb, 6, lB); ARG(kb, 7, Np);
    CHECK(clEnqueueNDRangeKernel(q, kb, 3, NULL, gb, NULL, 0, NULL, &evp[1]));
    // Direct write (no Cp and no epilogue): alpha = 1, beta = 0, M and N multiples of the tile, batch contiguous in C.
    double t_one = est_ms(r, Mp, Np, Kp);
    int split = t_one * count > max_launch_ms();   // too long for one launch: bands (launch()) per matrix, epilogue
    int direct = !split && alpha == 1.0 && beta == 0.0 && M == Mp && N == Np && (count == 1 || sC == (long)M * ldc);
    bf_t target = direct ? xC : BM(dCp); int ldtarget = direct ? ldc : Np;
    ARG(r->main, 0, dAp); ARG(r->main, 1, dBp); ARGB(r->main, 2, target); ARG(r->main, 3, Mp); ARG(r->main, 4, Np);
    ARG(r->main, 5, ldtarget); ARG(r->main, 6, Kp); ARG(r->main, 7, Kp);
    size_t gm = (size_t)count * (Np / r->tn) * (Mp / 16) * r->sg, lm = r->sg;
    cl_event ev = NULL;
    if (!split) CHECK(clEnqueueNDRangeKernel(q, r->main, 1, NULL, &gm, &lm, 0, NULL, &ev));
    else {
        nev = 0;
        for (int b = 0; b < count; b++)
            if (launch(r, dAp, (size_t)b * Mp * Kp * es, dBp, (size_t)b * Np * Kp * es, dCp, (size_t)b * Mp * Np * es,
                       Mp, Np, Np, Kp, Kp, 1)) return -1;
    }
    ARG(r->epi, 0, dCp); ARG(r->epi, 1, Np); ARG(r->epi, 2, Mp); ARGB(r->epi, 3, xC); ARG(r->epi, 4, ldc); ARG(r->epi, 5, lC);
    ARGS(r->epi, 6, alpha); ARGS(r->epi, 7, beta);
    size_t ge[3] = { (size_t)N, (size_t)M, (size_t)count };
    if (direct) CHECK(clEnqueueMarkerWithWaitList(q, 0, NULL, &evp[2]));
    else CHECK(clEnqueueNDRangeKernel(q, r->epi, 3, NULL, ge, NULL, 0, NULL, &evp[2]));
    if (to_host(xC, zC, bC, C)) return -1;
    if (split) { double tz = event_times(); last_ms = tz >= 0 ? tz : (now_s() - t_start) * 1e3; }
    else if (profile) {
        cl_ulong t0, t1;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof t0, &t0, NULL);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof t1, &t1, NULL);
        last_ms = (t1 - t0) * 1e-6;
    } else last_ms = (now_s() - t_start) * 1e3;
    if (ev) clReleaseEvent(ev);
    if (getenv("GEN9BLAS_PROFILE")) {
        double tp[3];
        for (int e = 0; e < 3; e++) { cl_ulong a0, a1; clGetEventProfilingInfo(evp[e], CL_PROFILING_COMMAND_START, 8, &a0, NULL);
            clGetEventProfilingInfo(evp[e], CL_PROFILING_COMMAND_END, 8, &a1, NULL); tp[e] = (a1 - a0) * 1e-6; }
        fprintf(stderr, "gen9blas_gemm %dx%dx%d x%d: total %.2f ms (buffers %.2f, pack A %.2f, pack B %.2f, kernel %.2f, epilogue %.2f), zero-copy %d\n",
                M, N, K, count, (now_s() - t_start) * 1e3, t_buf * 1e3, tp[0], tp[1], last_ms, tp[2], last_zc);
    }
    for (int e = 0; e < 3; e++) clReleaseEvent(evp[e]);
    return 0;
}

// ---------------- device path: pack once, many kernel launches on chunks ----------------

// Packs op(X) into the buffer `target`: role A (roleB=0): op(X) R×K, trans: X stored as K×R; role B (roleB=1): op(X) K×R,
// trans: X stored as R×K. Zero padding up to Rp × Kp. Packing from a device buffer; for role B without transpose only the rows k in [k0, k0+kn)
// can be packed (global grid offset), e.g. successive solution blocks in TRSM. kn = 0: everything.
int pack_dev(type_t *r, int roleB, int trans, bf_t xX, int ld, int Rr, int K, int Rp, int Kp, cl_mem target,
                     int k0, int kn) {
    cl_long zero = 0;
    size_t g[3], o[3] = { 0, 0, 0 };
    cl_kernel k = pack_kernel(r, roleB, trans, Rp, Kp, 1, g);
    if (!roleB) { ARGB(k, 0, xX); ARG(k, 1, target); ARG(k, 2, Rr); ARG(k, 3, K); ARG(k, 4, ld); ARG(k, 5, Kp); ARG(k, 6, zero); ARG(k, 7, Rp); }
    else        { ARGB(k, 0, xX); ARG(k, 1, target); ARG(k, 2, K); ARG(k, 3, Rr); ARG(k, 4, ld); ARG(k, 5, Kp); ARG(k, 6, zero); ARG(k, 7, Rp); }
    if (kn && roleB && !trans) { g[1] = kn; o[1] = k0; }   // k chunk (B and B_w2: k in g[1])
    CHECK(clEnqueueNDRangeKernel(q, k, 3, o, g, NULL, 0, NULL, NULL));
    return 0;
}

// Packs op(X) from host memory into the buffer `target` (see pack_dev).
int pack_host(type_t *r, int roleB, int trans, const void *X, int ld, int Rr, int K, int Rp, int Kp, cl_mem target) {
    size_t bX = (roleB ? (trans ? (size_t)(Rr - 1) * ld + K : (size_t)(K - 1) * ld + Rr)
                       : (trans ? (size_t)(K - 1) * ld + Rr : (size_t)(Rr - 1) * ld + K)) * r->es;
    bf_t xX; int z;
    if (get_obj(X, bX, roleB ? &uB : &uA, 1, 0, &xX, &z)) return -1;
    return pack_dev(r, roleB, trans, xX, ld, Rr, K, Rp, Kp, target, 0, 0);
}

cl_mem subbuffer(cl_mem b, size_t off, size_t size) {
    cl_int err; cl_buffer_region reg = { off, size };
    cl_mem s = clCreateSubBuffer(b, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg, &err);
    if (err != CL_SUCCESS) { snprintf(errbuf, sizeof errbuf, "clCreateSubBuffer = %d", err); return NULL; }
    return s;
}

// Epilogue on a device buffer: Cd = alpha·Cp + beta·Cd (M×N, the whole rectangle).
int epilogue_dev(type_t *r, cl_mem Cp, int ldp, cl_mem Cd, int ldc, int M, int N, double alpha, double beta) {
    int uplo = 2;
    ARG(r->epi_tri, 0, Cp); ARG(r->epi_tri, 1, ldp); ARG(r->epi_tri, 2, Cd); ARG(r->epi_tri, 3, ldc);
    ARGS(r->epi_tri, 4, alpha); ARGS(r->epi_tri, 5, beta); ARG(r->epi_tri, 6, uplo);
    size_t g[2] = { (size_t)N, (size_t)M };
    CHECK(clEnqueueNDRangeKernel(q, r->epi_tri, 2, NULL, g, NULL, 0, NULL, NULL));
    return 0;
}

// Longest planned GEMM kernel launch. kcl does not set up mid-thread preemption, so a launch that runs longer than the
// engine's preempt_timeout_ms (640 ms by default) while another context (e.g. the desktop) waits for the GPU gets the
// engine reset and its results lost. Long launches are split into row bands, each estimated below this time from a
// conservative throughput per type. GEN9BLAS_MAX_LAUNCH_MS overrides it (default 100).
double max_launch_ms(void) {
    static double v = 0;
    if (v <= 0) { const char *e = getenv("GEN9BLAS_MAX_LAUNCH_MS"); v = e && atof(e) > 0 ? atof(e) : 100; }
    return v;
}
double est_ms(type_t *r, double M, double N, double K) {   // conservative GFLOPS of the main kernels
    double gf = r == &types[GEN9BLAS_D] ? 45 : r == &types[GEN9BLAS_S] ? 300 : r == &types[GEN9BLAS_HS] ? 500 : 600;
    if (gen8) gf = r == &types[GEN9BLAS_D] ? 8 : 50;   // CHV peaks: FP64 14, FP32 115 GFLOPS (docs/gen8.md); not measured yet
    return 2 * M * N * K / (gf * 1e6);
}

// GEMM kernel on a chunk: sub-buffers with offsets (bytes; aligned to 128 B). measure: event goes to evs[].
static int launch1(type_t *r, cl_mem Ap, size_t oA, cl_mem Bp, size_t oB, cl_mem Cp, size_t oC,
                   int M, int N, int ldc, int K, int Kfull, int measure) {
    cl_int err; cl_mem sub[3] = { Ap, Bp, Cp }; size_t off[3] = { oA, oB, oC }; int sub_made[3] = { 0, 0, 0 };
    for (int i = 0; i < 3; i++) if (off[i]) {
        size_t size; clGetMemObjectInfo(sub[i], CL_MEM_SIZE, sizeof size, &size, NULL);
        cl_buffer_region reg = { off[i], size - off[i] };
        sub[i] = clCreateSubBuffer(sub[i], CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg, &err); CHECK(err); sub_made[i] = 1;
    }
    ARG(r->main, 0, sub[0]); ARG(r->main, 1, sub[1]); ARG(r->main, 2, sub[2]); ARG(r->main, 3, M); ARG(r->main, 4, N);
    ARG(r->main, 5, ldc); ARG(r->main, 6, K); ARG(r->main, 7, Kfull);
    size_t gm = (size_t)(N / r->tn) * (M / 16) * r->sg, lm = r->sg; cl_event ev;
    CHECK(clEnqueueNDRangeKernel(q, r->main, 1, NULL, &gm, &lm, 0, NULL, &ev));
    if (measure && nev < (int)(sizeof evs / sizeof evs[0])) evs[nev++] = ev; else clReleaseEvent(ev);
    for (int i = 0; i < 3; i++) if (sub_made[i]) clReleaseMemObject(sub[i]);
    return 0;
}
// As launch1, split into bands of rows (multiples of 16; packed A: row i0 at i0·Kfull, C: row i0 at i0·ldc) so that each
// launch stays below max_launch_ms().
int launch(type_t *r, cl_mem Ap, size_t oA, cl_mem Bp, size_t oB, cl_mem Cp, size_t oC,
                   int M, int N, int ldc, int K, int Kfull, int measure) {
    int parts = (int)(est_ms(r, M, N, K) / max_launch_ms()) + 1;
    int mb = ((M + parts - 1) / parts + 15) / 16 * 16;
    if (mb >= M) return launch1(r, Ap, oA, Bp, oB, Cp, oC, M, N, ldc, K, Kfull, measure);
    for (int i0 = 0; i0 < M; i0 += mb) {
        int b = M - i0 < mb ? M - i0 : mb;
        if (launch1(r, Ap, oA + (size_t)i0 * Kfull * r->es, Bp, oB, Cp, oC + (size_t)i0 * ldc * r->es, b, N, ldc, K, Kfull, measure))
            return -1;
    }
    return 0;
}

// Epilogue to host memory: C = alpha·Cp + beta·C on the uplo triangle (0 lower, 1 upper, 2 whole).
int epilogue_host(type_t *r, cl_mem Cp, int ldp, void *C, int M, int N, int ldc, double alpha, double beta, int uplo) {
    size_t bC = ((size_t)(M - 1) * ldc + N) * r->es; bf_t xC; int zC;
    if (get_obj(C, bC, &uC, beta != 0.0f || uplo != 2, 1, &xC, &zC)) return -1;
    ARG(r->epi_tri, 0, Cp); ARG(r->epi_tri, 1, ldp); ARGB(r->epi_tri, 2, xC); ARG(r->epi_tri, 3, ldc);
    ARGS(r->epi_tri, 4, alpha); ARGS(r->epi_tri, 5, beta); ARG(r->epi_tri, 6, uplo);
    size_t g[2] = { (size_t)N, (size_t)M };
    CHECK(clEnqueueNDRangeKernel(q, r->epi_tri, 2, NULL, g, NULL, 0, NULL, NULL));
    return to_host(xC, zC, bC, C);
}

int gen9blas_gemm(int type, int tA, int tB, int M, int N, int K, double alpha, const void *A, int lda,
             const void *B, int ldb, double beta, void *C, int ldc) {
    return gen9blas_gemm_batched(type, tA, tB, M, N, K, alpha, A, lda, 0, B, ldb, 0, beta, C, ldc, 0, 1);
}
int gen9blas_hgemm(int M, int N, int K, double alpha, const uint16_t *A, int lda, const uint16_t *B, int ldb,
              double beta, uint16_t *C, int ldc) { return gen9blas_gemm(GEN9BLAS_H, 0, 0, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc); }
int gen9blas_sgemm(int M, int N, int K, double alpha, const float *A, int lda, const float *B, int ldb,
              double beta, float *C, int ldc) { return gen9blas_gemm(GEN9BLAS_S, 0, 0, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc); }

// ---------------- skinny GEMM (small M or N): docs/skinny-gemm.md, results/skinny-gemm-2026-11-12.txt ----------------
// No packing: the weights (the larger operand) are read directly once per tile of rows of the small operand, FP32 accumulation (also
// for FP16 -- accuracy at least as good as GEN9BLAS_HS). The small operand "a" (Mc × K) and the weights "w" (K × Nc) with general strides;
// for small N the roles are swapped: Cᵀ = op(B)ᵀ · op(A)ᵀ (C is written with swapped strides).
// GEN9BLAS_SKINNY=0 disables it (old path), GEN9BLAS_SKINNY=n sets the threshold min(M, N) ≤ n for all types (measurements).
int skinny_threshold(int type) {
    static int forced = -2;
    if (forced == -2) { const char *e = getenv("GEN9BLAS_SKINNY"); forced = e ? atoi(e) : -1; }
    if (type == GEN9BLAS_D) return 0;
    if (forced >= 0) return forced;
    // From measurement (results/skinny-gemm-2026-11-12.txt, 8 shapes, USM and ordinary memory): at M = 16 skinny is faster almost
    // everywhere (FP16 1.0–2.0×, FP32 0.9–2.2×; weaker only for 51866×1280), at M = 32 FP16 0.3–1.0×, FP32 0.8–1.2×.
    return 16;
}
int gemm_skinny(int type, int tA, int tB, int M, int N, int K, double alpha, const void *A, int lda,
                      const void *B, int ldb, double beta, void *C, int ldc) {
    l12_t *k = &L12[type == GEN9BLAS_S ? 1 : 0];
    const int es = k->es;
    size_t bA = (tA ? (size_t)(K - 1) * lda + M : (size_t)(M - 1) * lda + K) * es;
    size_t bB = (tB ? (size_t)(N - 1) * ldb + K : (size_t)(K - 1) * ldb + N) * es;
    size_t bC = ((size_t)(M - 1) * ldc + N) * es;
    bf_t xA, xB, xC; int zA, zB, zC;
    if (get_obj(A, bA, &uA, 1, 0, &xA, &zA) || get_obj(B, bB, &uB, 1, 0, &xB, &zB) || get_obj(C, bC, &uC, beta != 0.0, 1, &xC, &zC))
        return -1;
    last_zc = zA | zB << 1 | zC << 2;
    // a: Mc × K (strides sam, sak); w: element (k, n) = w[n·swn + k·swk]; C[m][n] = C[m·scm + n·scn]
    int Mc, Nc; bf_t xa, xw; cl_long sam, sak, swn, swk, scm, scn;
    if (M <= N) {
        Mc = M; Nc = N; xa = xA; xw = xB; scm = ldc; scn = 1;
        sam = tA ? 1 : lda; sak = tA ? lda : 1;          // op(A)[m][k]
        swn = tB ? ldb : 1; swk = tB ? 1 : ldb;           // op(B)[k][n]
    } else {
        Mc = N; Nc = M; xa = xB; xw = xA; scm = 1; scn = ldc;
        sam = tB ? ldb : 1; sak = tB ? 1 : ldb;           // op(B)ᵀ[n][k] = op(B)[k][n]
        swn = tA ? 1 : lda; swk = tA ? lda : 1;           // op(A)ᵀ[k][m] = op(A)[m][k]
    }
    int iK = K, iM = Mc, iN = Nc;
    static int no_r = -1;   // GEN9BLAS_SKINNY_R=0: old skinny_k4/k8 (comparisons)
    if (no_r < 0) { const char *e = getenv("GEN9BLAS_SKINNY_R"); no_r = e && !atoi(e); }
    const int KL = es == 2 ? 4 : 2;   // k elements per lane (8 B), as in skinny_r in level12.cl
    const uintptr_t align_or = (uintptr_t)(M <= N ? A : B) | (uintptr_t)(M <= N ? B : A) | (uintptr_t)(sam * es) | (uintptr_t)(swn * es);
    // the GPU address has the same offset within a page as on the host (USM, kcl userptr; work buffer aligned)
    // M ≤ 4 with a weight row > 16 KB: old kernel (measured 2026-11-12: N = 4096, K = 11008 FP32 M = 2–4 0.7×, FP16 0.9–1.0×;
    // model gap: 8 long rows at once per group -- pages/TLB?)
    const int r_ok = Mc > 4 || (size_t)K * es <= 16384;
    if (swk == 1 && sak == 1 && Mc >= 2 && K % (8 * KL) == 0 && (align_or & 15) == 0 && r_ok && !no_r) {
        // rework of 2026-11-12 (results/skinny-gemm-rework-2026-11-12.txt): SIMD8, 8 weight rows, SKINNY_S subgroups
        const int MT = Mc <= 4 ? 4 : 8;
        cl_kernel Kk = MT == 4 ? k->skinny_r4 : k->skinny_r8;
        ARG(Kk, 0, iM); ARG(Kk, 1, iN); ARG(Kk, 2, iK); CHECK(argr(Kk, 3, alpha, 4)); CHECK(argr(Kk, 4, beta, 4));
        ARGB(Kk, 5, xa); ARG(Kk, 6, sam); ARGB(Kk, 7, xw); ARG(Kk, 8, swn); ARGB(Kk, 9, xC); ARG(Kk, 10, scm); ARG(Kk, 11, scn);
        size_t g[2] = { (size_t)(Mc + MT - 1) / MT * 8 * SKINNY_S, (size_t)(Nc + 7) / 8 }, l[2] = { 8 * SKINNY_S, 1 };
        CHECK(clEnqueueNDRangeKernel(q, Kk, 2, NULL, g, l, 0, NULL, NULL));
    } else if (swk == 1) {   // weights contiguous along K
        const int MT = Mc == 1 ? 1 : Mc <= 4 ? 4 : 8, RW = MT == 8 ? 2 : 4;   // as SKINNY_K in level12.cl
        cl_kernel Kk = MT == 1 ? k->skinny_k1 : MT == 4 ? k->skinny_k4 : k->skinny_k8;
        ARG(Kk, 0, iM); ARG(Kk, 1, iN); ARG(Kk, 2, iK); CHECK(argr(Kk, 3, alpha, es == 8 ? 8 : 4)); CHECK(argr(Kk, 4, beta, 4));
        ARGB(Kk, 5, xa); ARG(Kk, 6, sam); ARG(Kk, 7, sak); ARGB(Kk, 8, xw); ARG(Kk, 9, swn); ARGB(Kk, 10, xC); ARG(Kk, 11, scm); ARG(Kk, 12, scn);
        size_t g[2] = { (size_t)(Mc + MT - 1) / MT * 256, (size_t)(Nc + RW - 1) / RW }, l[2] = { 256, 1 };
        CHECK(clEnqueueNDRangeKernel(q, Kk, 2, NULL, g, l, 0, NULL, NULL));
    } else {          // weights contiguous along N (swn == 1): parts of K, then the sum and epilogue
        const int MT = 4, cols4 = (Nc + 3) / 4, tiles = (Mc + MT - 1) / MT;
        int parts = (8192 + cols4 - 1) / cols4; if (parts > K) parts = K; if (parts < 1) parts = 1;   // ~8 thousand work-items per M tile
        int kpp = (K + parts - 1) / parts; parts = (K + kpp - 1) / kpp;
        if (ensure_buf(&dQ, &sQ, (size_t)parts * Mc * Nc * sizeof(float))) return -1;
        cl_kernel Kn = k->skinny_n4;
        ARG(Kn, 0, iM); ARG(Kn, 1, iN); ARG(Kn, 2, iK); ARG(Kn, 3, kpp); ARGB(Kn, 4, xa); ARG(Kn, 5, sam); ARG(Kn, 6, sak);
        ARGB(Kn, 7, xw); ARG(Kn, 8, swk); ARG(Kn, 9, dQ);
        size_t g[2] = { ((size_t)cols4 + 63) / 64 * 64, (size_t)parts * tiles }, l[2] = { 64, 1 };
        CHECK(clEnqueueNDRangeKernel(q, Kn, 2, NULL, g, l, 0, NULL, NULL));
        cl_kernel K2 = k->skinny_n2;
        ARG(K2, 0, iM); ARG(K2, 1, iN); ARG(K2, 2, parts); ARG(K2, 3, dQ); CHECK(argr(K2, 4, alpha, 4)); CHECK(argr(K2, 5, beta, 4));
        ARGB(K2, 6, xC); ARG(K2, 7, scm); ARG(K2, 8, scn);
        size_t g2[2] = { ((size_t)Nc + 63) / 64 * 64, (size_t)Mc }, l2[2] = { 64, 1 };
        CHECK(clEnqueueNDRangeKernel(q, K2, 2, NULL, g2, l2, 0, NULL, NULL));
    }
    return to_host(xC, zC, bC, C);
}
