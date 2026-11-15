// Packed weights kept between calls (gen9blas_pack_b / _q, gemm_pb, F16 weights x F32 activations, multi-GEMM, FFN) for the ggml backend.
// Split from lib/gen9blas.c (refactor 2026-11-15); shared state in internal.h.
#include "internal.h"

// ---------------- packed B kept between calls (constant weights; ggml backend, docs/ggml-backend-design.md) ----------------


// Spare packed buffers: gen9blas_packed_b_free keeps up to SPARES freed buffers (GEN9BLAS_SPARE_MB in all, default 512)
// and the next packing takes the smallest one that fits. A new buffer costs about as much as the dequantization
// itself (i915 zeroes its pages: ~3.7 GB/s with userptr, 2026-11-13); this matters for weights packed on every use.
static struct { cl_mem m; size_t sz; } spare[SPARES];
static size_t spare_cap(void) {
    static size_t v = (size_t)-1;
    if (v == (size_t)-1) { const char *e = getenv("GEN9BLAS_SPARE_MB"); v = (size_t)(e ? atof(e) : 512) << 20; }
    return v;
}
static cl_mem packed_buf(size_t need, size_t *got, cl_int *err) {
    int best = -1;
    for (int i = 0; i < SPARES; i++)
        if (spare[i].m && spare[i].sz >= need && (best < 0 || spare[i].sz < spare[best].sz)) best = i;
    if (best >= 0) { cl_mem m = spare[best].m; *got = spare[best].sz; spare[best].m = NULL; *err = CL_SUCCESS; return m; }
    *got = need;
    return clCreateBuffer(ctx, CL_MEM_READ_WRITE, need, NULL, err);
}
static void packed_buf_release(cl_mem m, size_t sz) {
    size_t total = sz; int freei = -1, smallest = -1;
    for (int i = 0; i < SPARES; i++) {
        if (!spare[i].m) { if (freei < 0) freei = i; continue; }
        total += spare[i].sz;
        if (smallest < 0 || spare[i].sz < spare[smallest].sz) smallest = i;
    }
    if (freei < 0 && smallest >= 0 && spare[smallest].sz < sz) {   // full: replace the smallest by a larger one
        total -= spare[smallest].sz; clReleaseMemObject(spare[smallest].m); spare[smallest].m = NULL; freei = smallest;
    }
    if (freei < 0 || total > spare_cap()) { clReleaseMemObject(m); return; }
    spare[freei].m = m; spare[freei].sz = sz;
}

static size_t pack_band_bytes(void) {
    static size_t v = 0;
    if (!v) { const char *e = getenv("GEN9BLAS_PACK_BAND_MB"); v = (size_t)(e && atof(e) > 0 ? atof(e) : 128) << 20; }
    return v;
}

// Packs op(B) into P->buf in bands of whole 32-column groups (n), each launch at most pack_band_bytes() of output.
static int pack_b_bands(type_t *r, int tB, const void *B, int ldb, gen9blas_packed_b *P) {
    size_t bB = (tB ? (size_t)(P->N - 1) * ldb + P->K : (size_t)(P->K - 1) * ldb + P->N) * r->es;
    bf_t xB; int z;
    if (get_obj(B, bB, &uB, 1, 0, &xB, &z)) return -1;
    size_t g[3], o[3] = { 0, 0, 0 };
    cl_kernel k = pack_kernel(r, 1, tB, P->Np, P->Kp, 1, g);
    int d = tB ? 1 : 0, per = (!tB && k == r->pb_w2) ? 2 : 1;   // grid dimension of n; work-items per 2 columns (w2)
    cl_long zero = 0;
    ARGB(k, 0, xB); ARG(k, 1, P->buf); ARG(k, 2, P->K); ARG(k, 3, P->N); ARG(k, 4, ldb); ARG(k, 5, P->Kp); ARG(k, 6, zero); ARG(k, 7, P->Np);
    size_t nb = pack_band_bytes() / ((size_t)P->Kp * r->es) / 32 * 32;
    if (nb < 32) nb = 32;
    for (size_t n0 = 0; n0 < (size_t)P->Np; n0 += nb) {
        size_t n = (size_t)P->Np - n0 < nb ? (size_t)P->Np - n0 : nb;
        o[d] = n0 / per; g[d] = n / per;
        CHECK(clEnqueueNDRangeKernel(q, k, 3, o, g, NULL, 0, NULL, NULL));
        CHECK(clFlush(q));   // one band per submission (kcl: each flush is one batch buffer)
    }
    if (wait_gpu()) return -1;
#ifndef GEN9BLAS_NEO
    if (z) kcl_userptr_release(B, bB);   // the original is not needed on the GPU any more: unpin it
#endif
    return 0;
}

gen9blas_packed_b *gen9blas_pack_b(int type, int tB, int K, int N, const void *B, int ldb) {
    NEW_OPERATION();
    if (type < 0 || type > 3 || !types[type].ready) { snprintf(errbuf, sizeof errbuf, "type %d unavailable", type); return NULL; }
    if (K <= 0 || N <= 0) { snprintf(errbuf, sizeof errbuf, "pack_b: K = %d, N = %d", K, N); return NULL; }
    type_t *r = &types[type];
    gen9blas_packed_b *P = calloc(1, sizeof *P);
    if (!P) return NULL;
    P->type = type; P->K = K; P->N = N;
    P->Np = (N + r->tn - 1) / r->tn * r->tn; P->Kp = (K + r->km - 1) / r->km * r->km;
    cl_int err;
    P->buf = packed_buf((size_t)P->Np * P->Kp * r->es + 4096, &P->bufsize, &err);
    if (err != CL_SUCCESS) { snprintf(errbuf, sizeof errbuf, "pack_b: clCreateBuffer = %d", err); free(P); return NULL; }
    if (pack_b_bands(r, tB, B, ldb, P)) {   // userptr failures show up late (argument / submission): retry with a copy
        force_copy = 1; NEW_OPERATION();
        int rc = pack_b_bands(r, tB, B, ldb, P);
        force_copy = 0;
        if (rc) { packed_buf_release(P->buf, P->bufsize); free(P); return NULL; }
    }
    return P;
}

// Quantized weights (ggml block layouts) dequantized on the GPU straight into a GEN9BLAS_HS packed B (see gen9blas.h).
static int pack_b_q_bands(int qt, size_t rb, const void *W, gen9blas_packed_b *P, int *nf) {
    size_t bW = (size_t)P->N * rb;
    bf_t xW; int z;
    if (get_obj(W, bW, &uB, 1, 0, &xW, &z)) return -1;
    bf_t xnf = { NULL, nf };
    cl_ulong rbl = rb;
    // v2 (SLM staging, contiguous 32 B writes; needs K % 256 == 0); GEN9BLAS_DEQ=v1 keeps the first kernel for comparisons
    static int v1 = -1;
    if (v1 < 0) { const char *e = getenv("GEN9BLAS_DEQ"); v1 = e && !strcmp(e, "v1"); }
    // v2 loads 16 B for Q8_0 / Q4_K: every row start W + n * rb must be 16 B aligned (always so for ggml tensors: 32 B
    // aligned data, rows of 272 / 144 B multiples); anything else goes to v1. Q6_K loads 2 B.
    const int aligned = qt == 2 ? !(((uintptr_t)W | rb) & 1) : !(((uintptr_t)W | rb) & 15);
    if (k_dequant2 && !v1 && P->K % 256 == 0 && aligned) {
        cl_kernel k2 = k_dequant2;
        ARGB(k2, 0, xW); ARG(k2, 1, rbl); ARG(k2, 2, qt); ARG(k2, 3, P->N); ARG(k2, 4, P->Kp); ARGB(k2, 7, xnf);
        ARG(k2, 6, P->buf);
        size_t nb = pack_band_bytes() / ((size_t)P->Kp * 2) / 16 * 16;
        if (nb < 16) nb = 16;
        for (size_t n0 = 0; n0 < (size_t)P->Np; n0 += nb) {
            const size_t rows = (size_t)P->Np - n0 < nb ? (size_t)P->Np - n0 : nb;
            int n0i = (int)n0;
            ARG(k2, 5, n0i);
            size_t g[2] = { rows / 16 * 128, (size_t)P->K / 256 }, l[2] = { 128, 1 };
            CHECK(clEnqueueNDRangeKernel(q, k2, 2, NULL, g, l, 0, NULL, NULL));
            CHECK(clFlush(q));
        }
        goto done;
    }
    ARGB(k_dequant, 0, xW); ARG(k_dequant, 1, rbl); ARG(k_dequant, 2, qt); ARG(k_dequant, 3, P->N); ARG(k_dequant, 4, P->Kp);
    ARG(k_dequant, 5, P->buf); ARGB(k_dequant, 6, xnf);
    size_t nb = pack_band_bytes() / ((size_t)P->Kp * 2) / 16 * 16;
    if (nb < 16) nb = 16;
    for (size_t n0 = 0; n0 < (size_t)P->Np; n0 += nb) {
        size_t o[2] = { n0, 0 }, g[2] = { (size_t)P->Np - n0 < nb ? (size_t)P->Np - n0 : nb, (size_t)P->K / (qt == 0 ? 32 : 256) };
        CHECK(clEnqueueNDRangeKernel(q, k_dequant, 2, o, g, NULL, 0, NULL, NULL));
        CHECK(clFlush(q));
    }
done:
    if (wait_gpu()) return -1;
#ifndef GEN9BLAS_NEO
    if (z) kcl_userptr_release(W, bW);
#endif
    return 0;
}

gen9blas_packed_b *gen9blas_pack_b_q(int qtype, int K, int N, const void *W, size_t row_bytes, int *out_of_range) {
    static int *nf;
    NEW_OPERATION();
    *out_of_range = 0;
    if (!k_dequant || !types[GEN9BLAS_HS].ready) { snprintf(errbuf, sizeof errbuf, "pack_b_q: dequant kernel unavailable"); return NULL; }
    const int qb = qtype == GEN9BLAS_Q8_0 ? 32 : 256;
    const size_t bb = qtype == GEN9BLAS_Q8_0 ? 34 : qtype == GEN9BLAS_Q4_K ? 144 : qtype == GEN9BLAS_Q6_K ? 210 : 0;
    if (!bb || K <= 0 || N <= 0 || K % qb || row_bytes < (size_t)K / qb * bb) {
        snprintf(errbuf, sizeof errbuf, "pack_b_q: qtype %d, K = %d, N = %d, row_bytes = %zu", qtype, K, N, row_bytes); return NULL;
    }
    if (!nf && !(nf = gen9blas_alloc(64))) return NULL;
    type_t *r = &types[GEN9BLAS_HS];
    gen9blas_packed_b *P = calloc(1, sizeof *P);
    if (!P) return NULL;
    P->type = GEN9BLAS_HS; P->K = K; P->N = N;
    P->Np = (N + r->tn - 1) / r->tn * r->tn; P->Kp = (K + r->km - 1) / r->km * r->km;
    if (P->Kp != K) { snprintf(errbuf, sizeof errbuf, "pack_b_q: K = %d not a multiple of %d", K, r->km); free(P); return NULL; }
    cl_int err;
    P->buf = packed_buf((size_t)P->Np * P->Kp * 2 + 4096, &P->bufsize, &err);
    if (err != CL_SUCCESS) { snprintf(errbuf, sizeof errbuf, "pack_b_q: clCreateBuffer = %d", err); free(P); return NULL; }
    *nf = 0;
    if (pack_b_q_bands(qtype, row_bytes, W, P, nf)) {   // userptr failures show up late: retry with a copy
        force_copy = 1; NEW_OPERATION(); *nf = 0;
        int rc = pack_b_q_bands(qtype, row_bytes, W, P, nf);
        force_copy = 0;
        if (rc) { packed_buf_release(P->buf, P->bufsize); free(P); return NULL; }
    }
    *out_of_range = *nf != 0;
    return P;
}

void gen9blas_packed_b_free(gen9blas_packed_b *P) {
    if (!P) return;
    wait_gpu();
    packed_buf_release(P->buf, P->bufsize); free(P);
}

int gen9blas_gemm_pb(int tA, int M, double alpha, const void *A, int lda, const gen9blas_packed_b *P,
                     double beta, void *C, int ldc) {
    NEW_OPERATION();
    if (!P) FAIL("gemm_pb: no packed B");
    const int type = P->type;
    if (!types[type].ready) FAIL("gemm_pb: type %d unavailable", type);
    if (M <= 0) return 0;
    type_t *r = &types[type];
    int es = r->es, N = P->N, K = P->K, Np = P->Np, Kp = P->Kp, Mp = (M + 15) / 16 * 16;
    size_t bA = (tA ? (size_t)(K - 1) * lda + M : (size_t)(M - 1) * lda + K) * es, bC = ((size_t)(M - 1) * ldc + N) * es;
    bf_t xA, xC; int zA, zC;
    if (get_obj(A, bA, &uA, 1, 0, &xA, &zA) || get_obj(C, bC, &uC, beta != 0.0, 1, &xC, &zC)) return -1;
    last_zc = zA | 2 | zC << 2;
    if (ensure_buf(&dAp, &sAp, (size_t)Mp * Kp * es + 4096) || ensure_buf(&dCp, &sCp, (size_t)Mp * Np * es)) return -1;
    size_t ga[3]; cl_long zero = 0;
    cl_kernel ka = pack_kernel(r, 0, tA, Mp, Kp, 1, ga);
    ARGB(ka, 0, xA); ARG(ka, 1, dAp); ARG(ka, 2, M); ARG(ka, 3, K); ARG(ka, 4, lda); ARG(ka, 5, Kp); ARG(ka, 6, zero); ARG(ka, 7, Mp);
    CHECK(clEnqueueNDRangeKernel(q, ka, 3, NULL, ga, NULL, 0, NULL, NULL));
    int split = est_ms(r, Mp, Np, Kp) > max_launch_ms();
    int direct = !split && alpha == 1.0 && beta == 0.0 && M == Mp && N == Np;
    if (!split) {
        bf_t target = direct ? xC : BM(dCp); int ldt = direct ? ldc : Np;
        ARG(r->main, 0, dAp); ARG(r->main, 1, P->buf); ARGB(r->main, 2, target); ARG(r->main, 3, Mp); ARG(r->main, 4, Np);
        ARG(r->main, 5, ldt); ARG(r->main, 6, Kp); ARG(r->main, 7, Kp);
        size_t gm = (size_t)(Np / r->tn) * (Mp / 16) * 16, lm = 16;
        CHECK(clEnqueueNDRangeKernel(q, r->main, 1, NULL, &gm, &lm, 0, NULL, NULL));
    } else if (launch(r, dAp, 0, P->buf, 0, dCp, 0, Mp, Np, Np, Kp, Kp, 0)) return -1;
    if (!direct) {   // epilogue in row bands of at most pack_band_bytes() of C each
        cl_long lC = 0;
        ARG(r->epi, 0, dCp); ARG(r->epi, 1, Np); ARG(r->epi, 2, Mp); ARGB(r->epi, 3, xC); ARG(r->epi, 4, ldc); ARG(r->epi, 5, lC);
        ARGS(r->epi, 6, alpha); ARGS(r->epi, 7, beta);
        size_t mb = pack_band_bytes() / ((size_t)N * es * 2);
        if (mb < 1) mb = 1;
        for (size_t i0 = 0; i0 < (size_t)M; i0 += mb) {
            size_t o[3] = { 0, i0, 0 }, ge[3] = { (size_t)N, (size_t)M - i0 < mb ? (size_t)M - i0 : mb, 1 };
            CHECK(clEnqueueNDRangeKernel(q, r->epi, 3, o, ge, NULL, 0, NULL, NULL));
        }
    }
    return to_host(xC, zC, bC, C);
}

// C (F32, M x N) = A (F32, M x K) . Bp, Bp packed with type GEN9BLAS_HS (FP16 weights). A is rounded to FP16 while packing,
// the accurate FP16 GEMM writes FP16 into a work buffer, the epilogue converts to F32 and sets *nonfinite when any result
// is inf / NaN (then the caller recomputes in F32). Epilogue in row bands as in gemm_pb.
// F32 activations -> A3 packed FP16 (16 x 16 tiles) and FP16 GEMM result -> F32 C: the 16 B-per-lane kernels
// (pk_a3_f32v, gen9blas_epilog_hf8, same values bit for bit) unless GEN9BLAS_PKV=0 or the binary lacks them.
static int pkv(void) { static int v = -1; if (v < 0) { const char *e = getenv("GEN9BLAS_PKV"); v = !(e && !strcmp(e, "0")); } return v; }
static int pack_a3_f32(bf_t xA, cl_mem dAp, int M, int K, int lda, int Kp, int Mp) {
    cl_long zero = 0;
    cl_kernel k = pkv() && k_pa3_v ? k_pa3_v : k_pa3_f32;
    ARGB(k, 0, xA); ARG(k, 1, dAp); ARG(k, 2, M); ARG(k, 3, K); ARG(k, 4, lda); ARG(k, 5, Kp); ARG(k, 6, zero); ARG(k, 7, Mp);
    if (k == k_pa3_v) {
        size_t g[3] = { (size_t)Kp, (size_t)Mp / 16, 1 }, l[3] = { 16, 1, 1 };
        CHECK(clEnqueueNDRangeKernel(q, k, 3, NULL, g, l, 0, NULL, NULL));
    } else {
        size_t g[3] = { (size_t)Kp, (size_t)Mp, 1 };
        CHECK(clEnqueueNDRangeKernel(q, k, 3, NULL, g, NULL, 0, NULL, NULL));
    }
    return 0;
}
static int epilog_hf(cl_mem dCp, int ldp, int Mp, bf_t xC, int ldc, int N, int M, bf_t xnf) {
    float one = 1.0f, zf = 0.0f; cl_long lC = 0;
    const int v8 = pkv() && k_epi_8;
    cl_kernel k = v8 ? k_epi_8 : k_epi_hf;
    ARG(k, 0, dCp); ARG(k, 1, ldp); ARG(k, 2, Mp); ARGB(k, 3, xC); ARG(k, 4, ldc); ARG(k, 5, lC); ARG(k, 6, one); ARG(k, 7, zf);
    if (v8) { ARG(k, 8, N); ARGB(k, 9, xnf); } else { ARGB(k, 8, xnf); }
    size_t mb = pack_band_bytes() / ((size_t)N * 6);
    if (mb < 1) mb = 1;
    for (size_t i0 = 0; i0 < (size_t)M; i0 += mb) {
        size_t o[3] = { 0, i0, 0 }, ge[3] = { v8 ? (size_t)(N + 3) / 4 : (size_t)N, (size_t)M - i0 < mb ? (size_t)M - i0 : mb, 1 };
        CHECK(clEnqueueNDRangeKernel(q, k, 3, o, ge, NULL, 0, NULL, NULL));
    }
    return 0;
}

// Epilogue with bias / GELU / residual fused in (gen9blas_epilog_hf8x), banded like epilog_hf.
static int epilog_hfx(cl_mem dCp, int ldp, int Mp, bf_t xC, int ldc, int N, int M, bf_t xnf, bf_t xb, bf_t xR, int ldr, int flags) {
    cl_kernel k = k_epi_8x;
    ARG(k, 0, dCp); ARG(k, 1, ldp); ARG(k, 2, Mp); ARGB(k, 3, xC); ARG(k, 4, ldc); ARG(k, 5, N); ARGB(k, 6, xnf);
    ARGB(k, 7, xb); ARGB(k, 8, xR); ARG(k, 9, ldr); ARG(k, 10, flags);
    size_t mb = pack_band_bytes() / ((size_t)N * 6);
    if (mb < 1) mb = 1;
    for (size_t i0 = 0; i0 < (size_t)M; i0 += mb) {
        size_t o[3] = { 0, i0, 0 }, ge[3] = { (size_t)(N + 3) / 4, (size_t)M - i0 < mb ? (size_t)M - i0 : mb, 1 };
        CHECK(clEnqueueNDRangeKernel(q, k, 3, o, ge, NULL, 0, NULL, NULL));
    }
    return 0;
}
// No-wait mode (gen9blas_nowait, gen9blas.h): products that would wait for their FP16-overflow flag instead take a flag
// slot from one USM array and return without waiting; gen9blas_nowait_pending reads and clears the slots (GPU idle).
static int nw_mode, *nw_flags, nw_cap, nw_used;
static int *nw_take(void) {
    if (!nw_mode) return NULL;
    if (!nw_flags) { nw_cap = 4096; if (!(nw_flags = gen9blas_alloc((size_t)nw_cap * 4))) { nw_cap = 0; return NULL; } memset(nw_flags, 0, (size_t)nw_cap * 4); }
    return nw_used < nw_cap ? nw_flags + nw_used++ : NULL;
}
void gen9blas_set_nowait(int on) { nw_mode = on; }
int gen9blas_nowait_pending(void) {
    int n = 0;
    for (int i = 0; i < nw_used; i++) n += nw_flags[i] != 0;
    if (nw_used) memset(nw_flags, 0, (size_t)nw_used * 4);
    nw_used = 0;
    return n;
}
static int gemm_pb_f16w_any(int ahalf, int M, const void *A, int lda, const gen9blas_packed_b *P, float *C, int ldc, int *nonfinite,
                            const epi_ext_t *ext, cl_mem preA, int *nw);
// Packed GEMM input made once and reused (gen9blas_layer_norm_pack -> gen9blas_gemm_pa_f16w_ex).
gen9blas_packed_a *gen9blas_packed_a_create(void) { return calloc(1, sizeof(gen9blas_packed_a)); }
void gen9blas_packed_a_free(gen9blas_packed_a *pa) { if (!pa) return; if (pa->buf) clReleaseMemObject(pa->buf); free(pa); }
int gen9blas_gemm_pa_f16w_ex(const gen9blas_packed_a *pa, const gen9blas_packed_b *P, float *C, int ldc,
                             const float *bias, int gelu, const float *R, int ldr, int *nonfinite) {
    if (!pa || !pa->buf || !P || pa->K != P->K || pa->Kp != P->Kp) FAIL("gemm_pa_f16w_ex: packed A does not match B (K %d/%d, Kp %d/%d)",
                                                                      pa ? pa->K : -1, P ? P->K : -1, pa ? pa->Kp : -1, P ? P->Kp : -1);
    const epi_ext_t e = { bias, gelu, R, ldr };
    const int any = bias || gelu || R;
    if (any && !k_epi_8x) FAIL("gemm_pa_f16w_ex: the binary lacks gen9blas_epilog_hf8x");
    return gemm_pb_f16w_any(0, pa->M, NULL, 0, P, C, ldc, nonfinite, any ? &e : NULL, pa->buf, NULL);
}

int gen9blas_gemm_pb_f16w_ex(int M, const float *A, int lda, const gen9blas_packed_b *P, float *C, int ldc,
                             const float *bias, int gelu, const float *R, int ldr, int *nonfinite) {
    if (!k_epi_8x) FAIL("gemm_pb_f16w_ex: the binary lacks gen9blas_epilog_hf8x");
    const epi_ext_t e = { bias, gelu, R, ldr };
    return gemm_pb_f16w_any(0, M, A, lda, P, C, ldc, nonfinite, &e, NULL, NULL);
}
int gen9blas_gemm_pb_f16w(int M, const float *A, int lda, const gen9blas_packed_b *P, float *C, int ldc, int *nonfinite) {
    return gemm_pb_f16w_any(0, M, A, lda, P, C, ldc, nonfinite, NULL, NULL, NULL);
}
// The same with FP16 activations (ggml conv_1d: im2col result as src0, F16 kernel as src1): packed by the HS pa kernel.
int gen9blas_gemm_pb_f16w_h(int M, const uint16_t *A, int lda, const gen9blas_packed_b *P, float *C, int ldc, int *nonfinite) {
    return gemm_pb_f16w_any(1, M, A, lda, P, C, ldc, nonfinite, NULL, NULL, NULL);
}

// Small batches (M <= 16) on HS-packed weights without packing A: gen9blas_skinny_pb16_<MT> (lib/kernels/hgemm.cl), F32 products
// and sums, F32 result straight into C. Returns 1 (nothing done) when the kernel is missing or M is out of range.
int gen9blas_gemm_pb_skinny(int M, const float *A, int lda, const gen9blas_packed_b *P, float *C, int ldc, int *nonfinite) {
    static int *nf;
    NEW_OPERATION();
    if (!P || P->type != GEN9BLAS_HS) FAIL("gemm_pb_skinny: needs B packed as GEN9BLAS_HS");
    if (M < 1 || M > 16 || !gen9blas_has(GEN9BLAS_FEAT_SKINNY_PB)) return 1;
    if (!nf && !(nf = gen9blas_alloc(64))) return -1;
    *nonfinite = 0; *nf = 0;
    const int N = P->N, K = P->K, Kp = P->Kp, Np = P->Np;
    const size_t bA = ((size_t)(M - 1) * lda + K) * 4, bC = ((size_t)(M - 1) * ldc + N) * 4;
    bf_t xA, xC; int zA, zC;
    if (get_obj(A, bA, &uA, 1, 0, &xA, &zA) || get_obj(C, bC, &uC, 0, 1, &xC, &zC)) return -1;
    cl_kernel k = k_skp[M <= 4 ? 0 : M <= 8 ? 1 : 2];
    bf_t xnf = { NULL, nf };
    ARGB(k, 0, xA); ARG(k, 1, lda); ARG(k, 2, M); ARG(k, 3, K); ARG(k, 4, P->buf); ARG(k, 5, Kp);
    ARGB(k, 6, xC); ARG(k, 7, ldc); ARG(k, 8, N); ARGB(k, 9, xnf);
    size_t g = (size_t)(Np / 16) * 256, l = 256;
    CHECK(clEnqueueNDRangeKernel(q, k, 1, NULL, &g, &l, 0, NULL, NULL));
    if (to_host(xC, zC, bC, C)) return -1;
    *nonfinite = *nf != 0;
    return 0;
}
static int gemm_pb_f16w_any(int ahalf, int M, const void *A, int lda, const gen9blas_packed_b *P, float *C, int ldc, int *nonfinite,
                            const epi_ext_t *ext, cl_mem preA, int *nw) {
    static int *nf;   // USM flag word (coherent with the GPU through the LLC)
    NEW_OPERATION();
    if (!P || P->type != GEN9BLAS_HS || !types[GEN9BLAS_HS].ready || !k_pa3_f32) FAIL("gemm_pb_f16w: needs B packed as GEN9BLAS_HS");
    if (!nf && !(nf = gen9blas_alloc(64))) return -1;
    *nonfinite = 0;
    if (M <= 0) return 0;
    type_t *r = &types[GEN9BLAS_HS];
    int N = P->N, K = P->K, Np = P->Np, Kp = P->Kp, Mp = (M + 15) / 16 * 16;
    size_t bA = preA ? 0 : ((size_t)(M - 1) * lda + K) * (ahalf ? 2 : 4), bC = ((size_t)(M - 1) * ldc + N) * 4;
    bf_t xA = BM(NULL), xC; int zA = 1, zC;
    if ((!preA && get_obj(A, bA, &uA, 1, 0, &xA, &zA)) || get_obj(C, bC, &uC, 0, 1, &xC, &zC)) return -1;
    if (!nw && zC) nw = nw_take();   // no-wait mode: only when the result lands in place (no copy-out after a wait)
    if (ensure_buf(&dAp, &sAp, (size_t)Mp * Kp * 2 + 4096) || ensure_buf(&dCp, &sCp, (size_t)Mp * Np * 2)) return -1;
    if (!nw) *nf = 0;
    cl_mem Ap = preA ? preA : dAp;   // preA: already packed (gen9blas_layer_norm_pack)
    if (preA) {
    } else if (ahalf) {   // FP16 rows -> A3 tiles: the HS type's own packing kernel (as gemm_pb)
        size_t ga[3]; cl_long zero = 0;
        cl_kernel ka = pack_kernel(r, 0, 0, Mp, Kp, 1, ga);
        ARGB(ka, 0, xA); ARG(ka, 1, dAp); ARG(ka, 2, M); ARG(ka, 3, K); ARG(ka, 4, lda); ARG(ka, 5, Kp); ARG(ka, 6, zero); ARG(ka, 7, Mp);
        CHECK(clEnqueueNDRangeKernel(q, ka, 3, NULL, ga, NULL, 0, NULL, NULL));
    } else if (pack_a3_f32(xA, dAp, M, K, lda, Kp, Mp)) return -1;
    if (est_ms(r, Mp, Np, Kp) <= max_launch_ms()) {
        ARG(r->main, 0, Ap); ARG(r->main, 1, P->buf); ARG(r->main, 2, dCp); ARG(r->main, 3, Mp); ARG(r->main, 4, Np);
        ARG(r->main, 5, Np); ARG(r->main, 6, Kp); ARG(r->main, 7, Kp);
        size_t gm = (size_t)(Np / r->tn) * (Mp / 16) * 16, lm = 16;
        CHECK(clEnqueueNDRangeKernel(q, r->main, 1, NULL, &gm, &lm, 0, NULL, NULL));
    } else if (launch(r, Ap, 0, P->buf, 0, dCp, 0, Mp, Np, Np, Kp, Kp, 0)) return -1;
    bf_t xnf = { NULL, nw ? nw : nf };
    if (ext) {   // bias (row vector, N floats) / GELU / residual R (M x N, row stride ldr; may be C itself)
        static work_t uBias, uR;
        static float dummy[4];
        bf_t xb, xR; int zb, zR;
        const int flags = (ext->bias ? 1 : 0) | (ext->gelu ? 2 : 0) | (ext->R ? 4 : 0);
        if (get_obj(ext->bias ? ext->bias : dummy, ext->bias ? (size_t)N * 4 : sizeof dummy, &uBias, 1, 0, &xb, &zb)) return -1;
        if (ext->R && ext->R == C && ext->ldr == ldc) {   // in place: the same object as C
            xR = xC;
            if (!zC) memcpy(xC.p, C, bC);                  // work-buffer path: C's current values are the residual
        }
        else if (get_obj(ext->R ? ext->R : dummy, ext->R ? ((size_t)(M - 1) * ext->ldr + N) * 4 : sizeof dummy, &uR, 1, 0, &xR, &zR)) return -1;
        if (epilog_hfx(dCp, Np, Mp, xC, ldc, N, M, xnf, xb, xR, ext->R ? ext->ldr : 0, flags)) return -1;
    } else if (epilog_hf(dCp, Np, Mp, xC, ldc, N, M, xnf)) return -1;
    if (nw) return 0;   // queued: the caller waits later (gen9blas_flush) and reads its flag slot
    if (to_host(xC, zC, bC, C)) return -1;
    *nonfinite = *nf != 0;
    return 0;
}



// GEMM of a packed A (device) with a packed B into a device buffer C (row stride Np), banded when long.
static int gemm_dev(type_t *r, cl_mem Ap, cl_mem Bp, cl_mem C, int Mp, int Np, int Kp) {
    if (est_ms(r, Mp, Np, Kp) > max_launch_ms()) return launch(r, Ap, 0, Bp, 0, C, 0, Mp, Np, Np, Kp, Kp, 0);
    ARG(r->main, 0, Ap); ARG(r->main, 1, Bp); ARG(r->main, 2, C); ARG(r->main, 3, Mp); ARG(r->main, 4, Np);
    ARG(r->main, 5, Np); ARG(r->main, 6, Kp); ARG(r->main, 7, Kp);
    size_t gm = (size_t)(Np / r->tn) * (Mp / 16) * 16, lm = 16;
    CHECK(clEnqueueNDRangeKernel(q, r->main, 1, NULL, &gm, &lm, 0, NULL, NULL));
    return 0;
}

// n (1..8) products of one F32 input with HS-packed weights (ggml: Q/K/V MUL_MATs on the same normed input): A packed to
// FP16 once, then GEMM + epilogue per weight, one wait. nonfinite[i]: output i had inf / NaN (caller recomputes it in F32).
// Returns 1 (nothing done) when an output is not zero-copy (the copy path has one work buffer for C).
int gen9blas_gemm_multi_f16w(int M, const float *A, int lda, int n, const gen9blas_packed_b *const *P, float *const *C,
                             const int *ldc, int *nonfinite) {
    static int *nf;   // one USM flag word per output
    NEW_OPERATION();
    if (n < 1 || n > 8 || !types[GEN9BLAS_HS].ready || !k_pa3_f32) FAIL("gemm_multi_f16w: bad arguments");
    if (!nf && !(nf = gen9blas_alloc(64))) return -1;
    const int K = P[0]->K, Kp = P[0]->Kp, Mp = (M + 15) / 16 * 16;
    int Npmax = 0;
    for (int i = 0; i < n; i++) {
        if (!P[i] || P[i]->type != GEN9BLAS_HS || P[i]->K != K || P[i]->Kp != Kp) FAIL("gemm_multi_f16w: B %d not HS-packed with K = %d", i, K);
        if (P[i]->Np > Npmax) Npmax = P[i]->Np;
        nonfinite[i] = 0;
    }
    if (M <= 0) return 0;
    type_t *r = &types[GEN9BLAS_HS];
    bf_t xA, xC[8]; int zA, zC[8];
    if (get_obj(A, ((size_t)(M - 1) * lda + K) * 4, &uA, 1, 0, &xA, &zA)) return -1;
    for (int i = 0; i < n; i++) {
        if (get_obj(C[i], ((size_t)(M - 1) * ldc[i] + P[i]->N) * 4, &uC, 0, 1, &xC[i], &zC[i])) return -1;
        if (!zC[i]) return 1;
    }
    if (ensure_buf(&dAp, &sAp, (size_t)Mp * Kp * 2 + 4096) || ensure_buf(&dCp, &sCp, (size_t)Mp * Npmax * 2)) return -1;
    int *nws[8] = { 0 };
    for (int i = 0; i < n; i++) { nws[i] = nw_take(); if (!nws[i]) nf[i] = 0; }   // no-wait mode: own flag slots (outputs are zero-copy here)
    if (pack_a3_f32(xA, dAp, M, K, lda, Kp, Mp)) return -1;
    for (int i = 0; i < n; i++) {
        if (gemm_dev(r, dAp, P[i]->buf, dCp, Mp, P[i]->Np, Kp)) return -1;
        bf_t xnf = { NULL, nws[i] ? nws[i] : nf + i };
        if (epilog_hf(dCp, P[i]->Np, Mp, xC[i], ldc[i], P[i]->N, M, xnf)) return -1;
    }
    int all = 1;
    for (int i = 0; i < n; i++) all &= nws[i] != NULL;
    if (all) return 0;          // no-wait mode
    if (wait_gpu()) return -1;
    for (int i = 0; i < n; i++) nonfinite[i] = !nws[i] && nf[i] != 0;   // calls with a slot report through gen9blas_nowait_pending
    return 0;
}


// Gated FFN with FP16 weights (ggml: down(swiglu(gate(x), up(x)))), all intermediates on the GPU:
// Y (F32, M x Nd) = (silu(X Wg) * (X Wu)) Wd, with Pg, Pu (K -> N) and Pd (N -> Nd) packed as GEN9BLAS_HS.
// X is packed to FP16 once; gate and up stay FP16; h is computed in F32 and rounded to FP16 into down's packed A;
// down's result goes through the FP16 -> F32 epilogue. *nonfinite = 1 on any inf / NaN / FP16 overflow on the way.
static cl_mem dG, dU, dH; static size_t sG, sU, sH;
int gen9blas_ffn_f16w(int M, const float *X, int ldx, const gen9blas_packed_b *Pg, const gen9blas_packed_b *Pu,
                      const gen9blas_packed_b *Pd, float *Y, int ldy, int *nonfinite) {
    static int *nf;
    NEW_OPERATION();
    if (!Pg || !Pu || !Pd || Pg->type != GEN9BLAS_HS || Pu->type != GEN9BLAS_HS || Pd->type != GEN9BLAS_HS || !k_swiglu)
        FAIL("ffn_f16w: needs weights packed as GEN9BLAS_HS");
    if (Pg->K != Pu->K || Pg->N != Pu->N || Pd->K != Pg->N || Pg->Np != Pu->Np)
        FAIL("ffn_f16w: shapes do not chain (gate %dx%d, up %dx%d, down %dx%d)", Pg->K, Pg->N, Pu->K, Pu->N, Pd->K, Pd->N);
    if (!nf && !(nf = gen9blas_alloc(64))) return -1;
    *nonfinite = 0;
    if (M <= 0) return 0;
    type_t *r = &types[GEN9BLAS_HS];
    int K = Pg->K, Kp = Pg->Kp, N = Pg->N, Np = Pg->Np, Nd = Pd->N, Npd = Pd->Np, Mp = (M + 15) / 16 * 16;
    size_t bX = ((size_t)(M - 1) * ldx + K) * 4, bY = ((size_t)(M - 1) * ldy + Nd) * 4;
    bf_t xX, xY; int zX, zY;
    if (get_obj(X, bX, &uA, 1, 0, &xX, &zX) || get_obj(Y, bY, &uC, 0, 1, &xY, &zY)) return -1;
    int *nw = zY ? nw_take() : NULL;   // no-wait mode
    if (ensure_buf(&dAp, &sAp, (size_t)Mp * Kp * 2 + 4096) || ensure_buf(&dG, &sG, (size_t)Mp * Np * 2) ||
        ensure_buf(&dU, &sU, (size_t)Mp * Np * 2) || ensure_buf(&dH, &sH, (size_t)Mp * Pd->Kp * 2 + 4096) ||
        ensure_buf(&dCp, &sCp, (size_t)Mp * Npd * 2)) return -1;
    if (!nw) *nf = 0;
    bf_t xnf = { NULL, nw ? nw : nf };
    if (pack_a3_f32(xX, dAp, M, K, ldx, Kp, Mp)) return -1;
    if (gemm_dev(r, dAp, Pg->buf, dG, Mp, Np, Kp) || gemm_dev(r, dAp, Pu->buf, dU, Mp, Np, Kp)) return -1;
    int Kpd = Pd->Kp;
    ARG(k_swiglu, 0, dG); ARG(k_swiglu, 1, dU); ARG(k_swiglu, 2, Np); ARG(k_swiglu, 3, M); ARG(k_swiglu, 4, N);
    ARG(k_swiglu, 5, dH); ARG(k_swiglu, 6, Kpd); ARGB(k_swiglu, 7, xnf);
    size_t gs[2] = { (size_t)Kpd, (size_t)Mp };
    CHECK(clEnqueueNDRangeKernel(q, k_swiglu, 2, NULL, gs, NULL, 0, NULL, NULL));
    if (gemm_dev(r, dH, Pd->buf, dCp, Mp, Npd, Kpd)) return -1;
    if (epilog_hf(dCp, Npd, Mp, xY, ldy, Nd, M, xnf)) return -1;
    if (nw) return 0;
    if (to_host(xY, zY, bY, Y)) return -1;
    *nonfinite = *nf != 0;
    return 0;
}
