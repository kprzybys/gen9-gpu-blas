// ggml backend elementwise ops queued without a wait (norms, RoPE, GELU, copies, upscale, im2col) and decoding attention over KV streams.
// Split from lib/gen9blas.c (refactor 2026-11-15); shared state in internal.h.
#include "internal.h"

// ---------------- ggml backend: per-layer elementwise ops (queued without a wait) ----------------
// Each call is its own operation (userptr regions); without a wait between operations the host buffers must be registered
// as spans (gen9blas_span), so that a tensor written by one operation and read by the next has one GPU address
// (docs/no-neo.md 6g.5). An output that is not zero-copy (copied through a work buffer) forces the wait.
int gen9blas_span(const void *p, size_t bytes, int writes) {
#ifndef GEN9BLAS_NEO
    return kcl_span_add(p, bytes, writes);
#else
    (void)p; (void)bytes; (void)writes; return -1;
#endif
}
void gen9blas_span_clear(void) {
#ifndef GEN9BLAS_NEO
    kcl_span_clear();
#endif
}
int gen9blas_flush(void) { return wait_gpu(); }
static work_t uD;
static int ew_done(int sync, bf_t xY, int zY, size_t bY, void *Y) { return sync || !zY ? to_host(xY, zY, bY, Y) : 0; }

int gen9blas_rms_norm(int M, int K, const float *X, int ldx, const float *w, float eps, float *Y, int ldy, int sync) {
    NEW_OPERATION();
    if (!k_rms) FAIL("rms_norm: kernel not in this build");
    if (M <= 0 || K <= 0) return 0;
    size_t bX = ((size_t)(M - 1) * ldx + K) * 4, bY = ((size_t)(M - 1) * ldy + K) * 4;
    bf_t xX, xW = BM(NULL), xY; int zX, zW, zY, hw = w != NULL;
    if (get_obj(X, bX, &uA, 1, 0, &xX, &zX) || (hw && get_obj(w, (size_t)K * 4, &uB, 1, 0, &xW, &zW)) ||
        get_obj(Y, bY, &uC, 0, 1, &xY, &zY)) return -1;
    if (!hw) xW = xX;   // any valid buffer; not read
    ARGB(k_rms, 0, xX); ARG(k_rms, 1, ldx); ARGB(k_rms, 2, xW); ARG(k_rms, 3, hw); ARG(k_rms, 4, eps); ARGB(k_rms, 5, xY);
    ARG(k_rms, 6, ldy); ARG(k_rms, 7, K);
    size_t g[2] = { 128, (size_t)M }, l[2] = { 128, 1 };
    CHECK(clEnqueueNDRangeKernel(q, k_rms, 2, NULL, g, l, 0, NULL, NULL));
    return ew_done(sync, xY, zY, bY, Y);
}

int gen9blas_binop_rows(int op, int M, int K, const float *A, int lda, const float *B, int ldb, int bs0, float *C, int ldc, int sync) {
    NEW_OPERATION();
    if (!k_binop) FAIL("binop_rows: kernel not in this build");
    if (M <= 0 || K <= 0) return 0;
    size_t bA = ((size_t)(M - 1) * lda + K) * 4, bB = ((size_t)(ldb ? M - 1 : 0) * ldb + (bs0 ? K : 1)) * 4, bC = ((size_t)(M - 1) * ldc + K) * 4;
    bf_t xA, xB, xC; int zA, zB, zC;
    if (get_obj(A, bA, &uA, 1, 0, &xA, &zA) || get_obj(B, bB, &uB, 1, 0, &xB, &zB) || get_obj(C, bC, &uC, 0, 1, &xC, &zC)) return -1;
    ARGB(k_binop, 0, xA); ARG(k_binop, 1, lda); ARGB(k_binop, 2, xB); ARG(k_binop, 3, ldb); ARGB(k_binop, 4, xC);
    ARG(k_binop, 5, ldc); ARG(k_binop, 6, op); ARG(k_binop, 7, bs0);
    size_t g[2] = { (size_t)K, (size_t)M };
    CHECK(clEnqueueNDRangeKernel(q, k_binop, 2, NULL, g, NULL, 0, NULL, NULL));
    return ew_done(sync, xC, zC, bC, C);
}

int gen9blas_layer_norm(int M, int K, const float *X, int ldx, const float *w, const float *b, float eps, float *Y, int ldy, int sync) {
    NEW_OPERATION();
    if (!k_lnorm) FAIL("layer_norm: kernel not in this build");
    if (M <= 0 || K <= 0) return 0;
    size_t bX = ((size_t)(M - 1) * ldx + K) * 4, bY = ((size_t)(M - 1) * ldy + K) * 4;
    bf_t xX, xW = BM(NULL), xB = BM(NULL), xY; int zX, zW, zB, zY, hw = w != NULL, hb = b != NULL;
    if (get_obj(X, bX, &uA, 1, 0, &xX, &zX) || (hw && get_obj(w, (size_t)K * 4, &uB, 1, 0, &xW, &zW)) ||
        (hb && get_obj(b, (size_t)K * 4, &uD, 1, 0, &xB, &zB)) || get_obj(Y, bY, &uC, 0, 1, &xY, &zY)) return -1;
    if (!hw) xW = xX;
    if (!hb) xB = xX;
    ARGB(k_lnorm, 0, xX); ARG(k_lnorm, 1, ldx); ARGB(k_lnorm, 2, xW); ARG(k_lnorm, 3, hw); ARGB(k_lnorm, 4, xB); ARG(k_lnorm, 5, hb);
    ARG(k_lnorm, 6, eps); ARGB(k_lnorm, 7, xY); ARG(k_lnorm, 8, ldy); ARG(k_lnorm, 9, K);
    size_t g[2] = { 128, (size_t)M }, l[2] = { 128, 1 };
    CHECK(clEnqueueNDRangeKernel(q, k_lnorm, 2, NULL, g, l, 0, NULL, NULL));
    return ew_done(sync, xY, zY, bY, Y);
}

// gen9blas_layer_norm + the result packed as the FP16 A3 input of the next GEMMs (pa, reused by gen9blas_gemm_pa_f16w_ex).
int gen9blas_layer_norm_pack(int M, int K, const float *X, int ldx, const float *w, const float *b, float eps, float *Y, int ldy,
                             gen9blas_packed_a *pa, int sync) {
    NEW_OPERATION();
    if (!k_lnorm_pack) FAIL("layer_norm_pack: kernel not in this build");
    if (M <= 0 || K <= 0 || !pa) FAIL("layer_norm_pack: bad arguments");
    const type_t *r = &types[GEN9BLAS_HS];
    const int Kp = (K + r->km - 1) / r->km * r->km, Mp = (M + 15) / 16 * 16;
    if (ensure_buf(&pa->buf, &pa->cap, (size_t)Mp * Kp * 2 + 4096)) return -1;
    pa->M = M; pa->K = K; pa->Kp = Kp;
    size_t bX = ((size_t)(M - 1) * ldx + K) * 4, bY = ((size_t)(M - 1) * ldy + K) * 4;
    bf_t xX, xW = BM(NULL), xB = BM(NULL), xY; int zX, zW, zB, zY, hw = w != NULL, hb = b != NULL;
    if (get_obj(X, bX, &uA, 1, 0, &xX, &zX) || (hw && get_obj(w, (size_t)K * 4, &uB, 1, 0, &xW, &zW)) ||
        (hb && get_obj(b, (size_t)K * 4, &uD, 1, 0, &xB, &zB)) || get_obj(Y, bY, &uC, 0, 1, &xY, &zY)) return -1;
    if (!hw) xW = xX;
    if (!hb) xB = xX;
    cl_kernel k = k_lnorm_pack;
    ARGB(k, 0, xX); ARG(k, 1, ldx); ARGB(k, 2, xW); ARG(k, 3, hw); ARGB(k, 4, xB); ARG(k, 5, hb); ARG(k, 6, eps);
    ARGB(k, 7, xY); ARG(k, 8, ldy); ARG(k, 9, K); ARG(k, 10, pa->buf); ARG(k, 11, Kp); ARG(k, 12, M);
    size_t g[2] = { 128, (size_t)Mp }, l[2] = { 128, 1 };
    CHECK(clEnqueueNDRangeKernel(q, k, 2, NULL, g, l, 0, NULL, NULL));
    return ew_done(sync, xY, zY, bY, Y);
}

int gen9blas_gelu(size_t n, const float *X, float *Y, int sync) {
    NEW_OPERATION();
    if (!k_gelu) FAIL("gelu: kernel not in this build");
    if (!n) return 0;
    bf_t xX, xY; int zX, zY;
    if (get_obj(X, n * 4, &uA, 1, 0, &xX, &zX) || get_obj(Y, n * 4, &uC, 0, 1, &xY, &zY)) return -1;
    ARGB(k_gelu, 0, xX); ARGB(k_gelu, 1, xY);
    CHECK(clEnqueueNDRangeKernel(q, k_gelu, 1, NULL, &n, NULL, 0, NULL, NULL));
    return ew_done(sync, xY, zY, n * 4, Y);
}

int gen9blas_cvt_f16_heads(int rows, int H, int D, const float *X, long s_row, long s_head, uint16_t *Y, int sync) {
    NEW_OPERATION();
    if (!k_cvt_heads) FAIL("cvt_f16_heads: kernel not in this build");
    if (rows <= 0 || H <= 0 || D <= 0) return 0;
    size_t bX = ((size_t)(rows - 1) * s_row + (size_t)(H - 1) * s_head + D) * 4, bY = (size_t)rows * H * D * 2;
    bf_t xX, xY; int zX, zY;
    if (get_obj(X, bX, &uA, 1, 0, &xX, &zX) || get_obj(Y, bY, &uC, 0, 1, &xY, &zY)) return -1;
    ARGB(k_cvt_heads, 0, xX); ARG(k_cvt_heads, 1, s_row); ARG(k_cvt_heads, 2, s_head); ARGB(k_cvt_heads, 3, xY);
    size_t g[3] = { (size_t)D, (size_t)H, (size_t)rows };
    CHECK(clEnqueueNDRangeKernel(q, k_cvt_heads, 3, NULL, g, NULL, 0, NULL, NULL));
    return ew_done(sync, xY, zY, bY, Y);
}

int gen9blas_im2col(int N, int IC, int IH, int IW, const float *X, long ofs0, long ofs1, int KH, int KW, int OH, int OW,
                    int s0, int s1, int p0, int p1, int d0, int d1, void *Y, int f16, int sync) {
    NEW_OPERATION();
    if (!k_im2col) FAIL("im2col: kernel not in this build");
    if (N <= 0 || IC <= 0 || OH <= 0 || OW <= 0 || KH <= 0 || KW <= 0) return 0;
    size_t bX = ((size_t)(N - 1) * ofs0 + (size_t)(IC - 1) * ofs1 + (size_t)IH * IW) * 4;
    size_t bY = (size_t)N * OH * OW * IC * KH * KW * (f16 ? 2 : 4);
    bf_t xX, xY; int zX, zY;
    if (get_obj(X, bX, &uA, 1, 0, &xX, &zX) || get_obj(Y, bY, &uC, 0, 1, &xY, &zY)) return -1;
    ARGB(k_im2col, 0, xX); ARG(k_im2col, 1, ofs0); ARG(k_im2col, 2, ofs1); ARG(k_im2col, 3, IH); ARG(k_im2col, 4, IW);
    ARG(k_im2col, 5, KH); ARG(k_im2col, 6, KW); ARG(k_im2col, 7, OH); ARG(k_im2col, 8, s0); ARG(k_im2col, 9, s1);
    ARG(k_im2col, 10, p0); ARG(k_im2col, 11, p1); ARG(k_im2col, 12, d0); ARG(k_im2col, 13, d1); ARG(k_im2col, 14, f16);
    ARGB(k_im2col, 15, xY);
    size_t g[3] = { (size_t)IC * KH * KW, (size_t)OW, (size_t)N * OH };
    CHECK(clEnqueueNDRangeKernel(q, k_im2col, 3, NULL, g, NULL, 0, NULL, NULL));
    return ew_done(sync, xY, zY, bY, Y);
}

int gen9blas_relu(size_t n, const float *X, float *Y, int sync) {
    NEW_OPERATION();
    if (!k_relu) FAIL("relu: kernel not in this build");
    if (!n) return 0;
    bf_t xX, xY; int zX, zY;
    if (X == Y) {
        if (get_obj(Y, n * 4, &uC, 1, 1, &xY, &zY)) return -1;
        xX = xY;
    } else if (get_obj(X, n * 4, &uA, 1, 0, &xX, &zX) || get_obj(Y, n * 4, &uC, 0, 1, &xY, &zY)) return -1;
    ARGB(k_relu, 0, xX); ARGB(k_relu, 1, xY);
    CHECK(clEnqueueNDRangeKernel(q, k_relu, 1, NULL, &n, NULL, 0, NULL, NULL));
    return ew_done(sync, xY, zY, n * 4, Y);
}

int gen9blas_gelu_erf(size_t n, const float *X, float *Y, int sync) {
    NEW_OPERATION();
    if (!k_gelu_erf) FAIL("gelu_erf: kernel not in this build");
    if (!n) return 0;
    bf_t xX, xY; int zX, zY;
    if (X == Y) {   // in place: one object
        if (get_obj(Y, n * 4, &uC, 1, 1, &xY, &zY)) return -1;
        xX = xY;
    } else if (get_obj(X, n * 4, &uA, 1, 0, &xX, &zX) || get_obj(Y, n * 4, &uC, 0, 1, &xY, &zY)) return -1;
    ARGB(k_gelu_erf, 0, xX); ARGB(k_gelu_erf, 1, xY);
    CHECK(clEnqueueNDRangeKernel(q, k_gelu_erf, 1, NULL, &n, NULL, 0, NULL, NULL));
    return ew_done(sync, xY, zY, n * 4, Y);
}

int gen9blas_rope(int n_dims, int nh, int n_tok, const float *X, long xs1, long xs2, float *Y, long ys1, long ys2,
                  const int *pos, const float *ff, int neox, float freq_base, float freq_scale, float ext_factor,
                  float attn_factor, const float corr[2], int sync) {
    NEW_OPERATION();
    if (!k_rope) FAIL("rope: kernel not in this build");
    if (n_dims <= 0 || nh <= 0 || n_tok <= 0) return 0;
    if (n_dims % 2 || xs1 < n_dims || ys1 < n_dims || xs2 < (long)nh * xs1 - (xs1 - n_dims) || ys2 < (long)nh * ys1 - (ys1 - n_dims))
        FAIL("rope: unsupported layout");
    size_t bX = ((size_t)(n_tok - 1) * xs2 + (size_t)(nh - 1) * xs1 + n_dims) * 4;
    size_t bY = ((size_t)(n_tok - 1) * ys2 + (size_t)(nh - 1) * ys1 + n_dims) * 4;
    bf_t xX, xY, xP, xF = BM(NULL); int zX, zY, zP, zF, hf = ff != NULL;
    if (get_obj(X, bX, &uA, 1, 0, &xX, &zX) || get_obj(pos, (size_t)n_tok * 4, &uB, 1, 0, &xP, &zP) ||
        (hf && get_obj(ff, (size_t)n_dims / 2 * 4, &uD, 1, 0, &xF, &zF)) || get_obj(Y, bY, &uC, 0, 1, &xY, &zY)) return -1;
    if (!hf) xF = xP;
    const float theta_scale = powf(freq_base, -2.0f / n_dims);   // as ggml-cpu
    cl_long l1 = xs1, l2 = xs2, m1 = ys1, m2 = ys2;
    ARGB(k_rope, 0, xX); ARG(k_rope, 1, l1); ARG(k_rope, 2, l2); ARGB(k_rope, 3, xY); ARG(k_rope, 4, m1); ARG(k_rope, 5, m2);
    ARGB(k_rope, 6, xP); ARGB(k_rope, 7, xF); ARG(k_rope, 8, hf); ARG(k_rope, 9, nh); ARG(k_rope, 10, n_dims); ARG(k_rope, 11, neox);
    ARG(k_rope, 12, theta_scale); ARG(k_rope, 13, freq_scale); ARG(k_rope, 14, ext_factor); ARG(k_rope, 15, attn_factor);
    ARG(k_rope, 16, corr[0]); ARG(k_rope, 17, corr[1]);
    size_t g[2] = { (size_t)n_dims / 2, (size_t)n_tok };
    CHECK(clEnqueueNDRangeKernel(q, k_rope, 2, NULL, g, NULL, 0, NULL, NULL));
    return ew_done(sync, xY, zY, bY, Y);
}

// ggml SET_ROWS F32 -> F16 (gen9blas.h). src / dst strides in elements, index strides in bytes; dst_bytes = extent of dst.
int gen9blas_set_rows_f16(int nc, int nr, int ne02, int ne03, const float *src, long s1, long s2, long s3,
                          const void *idx, size_t idx_bytes, long b10, long b11, long b12, int idx64, int ne11, int ne12,
                          uint16_t *dst, size_t dst_bytes, long d1, long d2, long d3, long ne1, int sync, int *bad) {
    static int *flag;
    NEW_OPERATION();
    if (!k_setrows) FAIL("set_rows: kernel not in this build");
    if (nc <= 0 || nr <= 0 || ne02 <= 0 || ne03 <= 0) return 0;
    if (!flag && !(flag = gen9blas_alloc(64))) return -1;
    size_t bS = ((size_t)(nr - 1) * s1 + (size_t)(ne02 - 1) * s2 + (size_t)(ne03 - 1) * s3 + nc) * 4;
    bf_t xS, xI, xD; int zS, zI, zD;
    if (get_obj(src, bS, &uA, 1, 0, &xS, &zS) || get_obj(idx, idx_bytes, &uB, 1, 0, &xI, &zI) ||
        get_obj(dst, dst_bytes, &uC, 1, 1, &xD, &zD)) return -1;
    if (!zD) FAIL("set_rows: destination is not zero-copy");   // a copy-in/copy-out of the whole cache would cost more than the op
    *flag = 0;
    bf_t xF = { NULL, flag };
    cl_long a1 = s1, a2 = s2, a3 = s3, c10 = b10, c11 = b11, c12 = b12, e1 = d1, e2 = d2, e3 = d3, n1 = ne1;
    cl_kernel k = k_setrows;
    ARGB(k, 0, xS); ARG(k, 1, a1); ARG(k, 2, a2); ARG(k, 3, a3); ARGB(k, 4, xI); ARG(k, 5, c10); ARG(k, 6, c11); ARG(k, 7, c12);
    ARG(k, 8, idx64); ARG(k, 9, ne11); ARG(k, 10, ne12); ARG(k, 11, ne02); ARGB(k, 12, xD); ARG(k, 13, e1); ARG(k, 14, e2);
    ARG(k, 15, e3); ARG(k, 16, n1); ARGB(k, 17, xF);
    size_t g[3] = { (size_t)nc, (size_t)nr, (size_t)ne02 * ne03 };
    CHECK(clEnqueueNDRangeKernel(q, k, 3, NULL, g, NULL, 0, NULL, NULL));
    if (sync) { if (wait_gpu()) return -1; if (bad) *bad = *flag; }
    return 0;
}

// ggml_rope_multi, mode MROPE (sections s[4]): pos holds 4 * n_tok positions (t, h, w, e). Layout rules as gen9blas_rope.
int gen9blas_rope_multi(int n_dims, int nh, int n_tok, const float *X, long xs1, long xs2, float *Y, long ys1, long ys2,
                        const int *pos, const float *ff, const int sections[4], float freq_base, float freq_scale,
                        float ext_factor, float attn_factor, const float corr[2], int vision, int sync) {
    NEW_OPERATION();
    if (!k_rope_m) FAIL("rope_multi: kernel not in this build");
    if (n_dims <= 0 || nh <= 0 || n_tok <= 0) return 0;
    const int ne0 = vision ? 2 * n_dims : n_dims;   // elements per head row that are rotated (vision: all, ne0 = 2 n_dims)
    if (n_dims % 2 || xs1 < ne0 || ys1 < ne0 || xs2 < (long)nh * xs1 - (xs1 - ne0) || ys2 < (long)nh * ys1 - (ys1 - ne0))
        FAIL("rope_multi: unsupported layout");
    if (sections[0] + sections[1] + sections[2] + sections[3] <= 0) FAIL("rope_multi: empty sections");
    size_t bX = ((size_t)(n_tok - 1) * xs2 + (size_t)(nh - 1) * xs1 + ne0) * 4;
    size_t bY = ((size_t)(n_tok - 1) * ys2 + (size_t)(nh - 1) * ys1 + ne0) * 4;
    bf_t xX, xY, xP, xF = BM(NULL); int zX, zY, zP, zF, hf = ff != NULL;
    if (get_obj(X, bX, &uA, 1, 0, &xX, &zX) || get_obj(pos, (size_t)n_tok * 16, &uB, 1, 0, &xP, &zP) ||
        (hf && get_obj(ff, (size_t)(vision ? n_dims : n_dims / 2) * 4, &uD, 1, 0, &xF, &zF)) || get_obj(Y, bY, &uC, 0, 1, &xY, &zY)) return -1;
    if (!hf) xF = xP;
    const float theta_scale = powf(freq_base, -2.0f / n_dims);   // as ggml-cpu
    cl_long l1 = xs1, l2 = xs2, m1 = ys1, m2 = ys2;
    cl_kernel k = k_rope_m;
    ARGB(k, 0, xX); ARG(k, 1, l1); ARG(k, 2, l2); ARGB(k, 3, xY); ARG(k, 4, m1); ARG(k, 5, m2);
    ARGB(k, 6, xP); ARGB(k, 7, xF); ARG(k, 8, hf); ARG(k, 9, nh); ARG(k, 10, n_dims); ARG(k, 11, n_tok);
    ARG(k, 12, sections[0]); ARG(k, 13, sections[1]); ARG(k, 14, sections[2]); ARG(k, 15, sections[3]);
    ARG(k, 16, theta_scale); ARG(k, 17, freq_scale); ARG(k, 18, ext_factor); ARG(k, 19, attn_factor);
    ARG(k, 20, corr[0]); ARG(k, 21, corr[1]);
    const int half = vision ? n_dims : n_dims / 2;
    ARG(k, 22, half); ARG(k, 23, vision);
    size_t g[2] = { (size_t)half, (size_t)n_tok };
    CHECK(clEnqueueNDRangeKernel(q, k, 2, NULL, g, NULL, 0, NULL, NULL));
    return ew_done(sync, xY, zY, bY, Y);
}

// ggml CPY / CONT / DUP, F32 / F16 in and out, byte strides, dst shape may differ (same element count): gen9blas.h.
static size_t extent4(const long ne[4], const long nb[4], size_t es) {
    size_t e = es;
    for (int i = 0; i < 4; i++) e += (size_t)(ne[i] - 1) * (size_t)nb[i];
    return e;
}
int gen9blas_cpy_any(const void *src, int tin, const long sne[4], const long snb[4], void *dst, int tout, const long dne[4],
                     const long dnb[4], int sync) {
    NEW_OPERATION();
    if (!k_cpy_any) FAIL("cpy_any: kernel not in this build");
    if ((tin != GEN9BLAS_S && tin != GEN9BLAS_H) || (tout != GEN9BLAS_S && tout != GEN9BLAS_H)) FAIL("cpy_any: types F32 / F16 only");
    tin = tin == GEN9BLAS_H; tout = tout == GEN9BLAS_H;   // kernel codes: 0 F32, 1 F16
    long n = 1, m = 1;
    for (int i = 0; i < 4; i++) {
        if (sne[i] <= 0 || dne[i] <= 0) return 0;
        if (snb[i] < 0 || dnb[i] < 0 || sne[i] > INT_MAX || dne[i] > INT_MAX) FAIL("cpy_any: unsupported layout");
        n *= sne[i]; m *= dne[i];
    }
    if (n != m || (tin | tout) & ~1 || dne[0] * dne[1] > INT_MAX || dne[2] * dne[3] > INT_MAX) FAIL("cpy_any: unsupported shape");
    const int same = sne[0] == dne[0] && sne[1] == dne[1] && sne[2] == dne[2] && sne[3] == dne[3];
    const size_t bS = extent4(sne, snb, tin ? 2 : 4), bD = extent4(dne, dnb, tout ? 2 : 4);
    bf_t xS, xD; int zS, zD;
    if (get_obj(src, bS, &uA, 1, 0, &xS, &zS) || get_obj(dst, bD, &uC, 1, 1, &xD, &zD)) return -1;   // copy-in: dst may be a strided view
    cl_long a0 = snb[0], a1 = snb[1], a2 = snb[2], a3 = snb[3], c0 = dnb[0], c1 = dnb[1], c2 = dnb[2], c3 = dnb[3];
    int s0 = (int)sne[0], s1 = (int)sne[1], s2 = (int)sne[2], d0 = (int)dne[0], d1 = (int)dne[1], d2 = (int)dne[2];
    cl_kernel k = k_cpy_any;
    ARGB(k, 0, xS); ARG(k, 1, a0); ARG(k, 2, a1); ARG(k, 3, a2); ARG(k, 4, a3); ARG(k, 5, s0); ARG(k, 6, s1); ARG(k, 7, s2);
    ARGB(k, 8, xD); ARG(k, 9, c0); ARG(k, 10, c1); ARG(k, 11, c2); ARG(k, 12, c3); ARG(k, 13, d0); ARG(k, 14, d1); ARG(k, 15, d2);
    ARG(k, 16, tin); ARG(k, 17, tout); ARG(k, 18, same);
    size_t g[3] = { (size_t)dne[0], (size_t)dne[1], (size_t)(dne[2] * dne[3]) };
    CHECK(clEnqueueNDRangeKernel(q, k, 3, NULL, g, NULL, 0, NULL, NULL));
    return ew_done(sync, xD, zD, bD, dst);
}

// ggml UPSCALE (ggml_interpolate) F32: mode 0 nearest / 1 bilinear, flags as ggml (bit 8 align corners, bit 9 antialias).
int gen9blas_upscale_f32(const float *src, const long sne[4], const long snb[4], float *dst, const long dne[4], const long dnb[4],
                         int mode_flags, int sync) {
    NEW_OPERATION();
    if (!k_upscale) FAIL("upscale: kernel not in this build");
    for (int i = 0; i < 4; i++) {
        if (sne[i] <= 0 || dne[i] <= 0) return 0;
        if (snb[i] < 0 || dnb[i] < 0 || dne[i] > INT_MAX) FAIL("upscale: unsupported layout");
    }
    const int mode = mode_flags & 0xFF, ac = (mode_flags >> 8) & 1, aa = (mode_flags >> 9) & 1;
    if ((mode != 0 && mode != 1) || (mode_flags & ~0x3FF) || dne[0] * dne[1] > INT_MAX || dne[2] * dne[3] > INT_MAX) FAIL("upscale: unsupported mode");
    float sf0 = (float)dne[0] / sne[0], sf1 = (float)dne[1] / sne[1], sf2 = (float)dne[2] / sne[2], sf3 = (float)dne[3] / sne[3];
    float po = 0.5f;
    if (ac) {   // as ggml-cpu
        po = 0.0f;
        sf0 = dne[0] > 1 && sne[0] > 1 ? (float)(dne[0] - 1) / (sne[0] - 1) : sf0;
        sf1 = dne[1] > 1 && sne[1] > 1 ? (float)(dne[1] - 1) / (sne[1] - 1) : sf1;
    }
    const size_t bS = extent4(sne, snb, 4), bD = extent4(dne, dnb, 4);
    bf_t xS, xD; int zS, zD;
    if (get_obj(src, bS, &uA, 1, 0, &xS, &zS) || get_obj(dst, bD, &uC, 1, 1, &xD, &zD)) return -1;
    cl_long a0 = snb[0], a1 = snb[1], a2 = snb[2], a3 = snb[3], c0 = dnb[0], c1 = dnb[1], c2 = dnb[2], c3 = dnb[3];
    int s0 = (int)sne[0], s1 = (int)sne[1], n2 = (int)dne[2];
    cl_kernel k = k_upscale;
    ARGB(k, 0, xS); ARG(k, 1, a0); ARG(k, 2, a1); ARG(k, 3, a2); ARG(k, 4, a3); ARG(k, 5, s0); ARG(k, 6, s1);
    ARGB(k, 7, xD); ARG(k, 8, c0); ARG(k, 9, c1); ARG(k, 10, c2); ARG(k, 11, c3); ARG(k, 12, n2);
    ARG(k, 13, sf0); ARG(k, 14, sf1); ARG(k, 15, sf2); ARG(k, 16, sf3); ARG(k, 17, mode); ARG(k, 18, aa); ARG(k, 19, po);
    size_t g[3] = { (size_t)dne[0], (size_t)dne[1], (size_t)(dne[2] * dne[3]) };
    CHECK(clEnqueueNDRangeKernel(q, k, 3, NULL, g, NULL, 0, NULL, NULL));
    return ew_done(sync, xD, zD, bD, dst);
}

// Decoding attention over KV streams (gen9blas.h): split-K partial kernel + merge, queued (sync: wait at the end).
int gen9blas_attn_dec_supported(int D, int H, int HKV) {
    return k_dec_part && k_dec_comb && D == 128 && HKV > 0 && H % HKV == 0 && H / HKV >= 1 && H / HKV <= 8;
}
int gen9blas_attn_dec(int D, int H, int HKV, int ns, int nq, int nk, const float *Q, long qt, long qh, long qs,
                      const uint16_t *K, long kk, long kh, long ks, size_t k_bytes, const uint16_t *V, long vk, long vh, long vs,
                      size_t v_bytes, float *O, float scale, const int *vis, int dvis, int chunk, int sync) {
    static float *part; static size_t part_n;
    static int *visring; static int vslot;   // 256 slots of 64 ints: launches stay queued; wait only when the ring wraps
    NEW_OPERATION();
    if (!k_dec_part || !k_dec_comb) FAIL("attn_dec: kernels not in this build");
    const int G = HKV > 0 ? H / HKV : 0;
    if (D != 128 || G < 1 || G > 8 || H != G * HKV || ns < 1 || nq < 1 || nk < 1 || chunk < 16 || chunk % 16) FAIL("attn_dec: unsupported shape");
    if (((uintptr_t)K | (uintptr_t)V | (uintptr_t)(kk * 2) | (uintptr_t)(kh * 2) | (uintptr_t)(vk * 2) | (uintptr_t)(vh * 2)) & 15)
        FAIL("attn_dec: K/V rows not 16-byte aligned");
    const int nchunk = (nk + chunk - 1) / chunk;
    const size_t pn = (size_t)ns * nq * H * nchunk * (2 + 128);
    if (pn > part_n) {
        if (part) { if (wait_gpu()) return -1; usm_free(ctx, part); }
        part = gen9blas_alloc(pn * 4); part_n = part ? pn : 0;
        if (!part) FAIL("attn_dec: scratch of %zu B", pn * 4);
    }
    if (ns > 64) FAIL("attn_dec: more than 64 streams");
    if (!visring && !(visring = gen9blas_alloc((size_t)256 * 64 * 4))) FAIL("attn_dec: vis buffer");
    if (vslot == 256) { if (wait_gpu()) return -1; vslot = 0; }
    int *visb = visring + (size_t)64 * vslot++;
    memcpy(visb, vis, (size_t)ns * 4);
    const size_t bQ = ((size_t)(ns - 1) * qs + (size_t)(nq - 1) * qt + (size_t)(H - 1) * qh + 128) * 4;
    const size_t bO = (size_t)ns * nq * H * 128 * 4;
    bf_t xQ, xK, xV, xO, xP = { NULL, part }, xS = { NULL, visb }; int zQ, zO;
    // K/V zero-copy or nothing, checked before any wait or copy: a copy of the whole cache per call would cost more than the op
    if (get_obj_zc(K, k_bytes, 0, &xK) || get_obj_zc(V, v_bytes, 0, &xV)) { vslot--; FAIL("attn_dec: K/V not zero-copy"); }
    if (get_obj(Q, bQ, &uA, 1, 0, &xQ, &zQ) || get_obj(O, bO, &uC, 0, 1, &xO, &zO)) return -1;
    const float qscale = scale * 1.4426950408889634f;
    cl_long a1 = qt, a2 = qh, a3 = qs, b1 = kk, b2 = kh, b3 = ks, c1 = vk, c2 = vh, c3 = vs;
    cl_kernel k = k_dec_part;
    ARGB(k, 0, xQ); ARG(k, 1, a1); ARG(k, 2, a2); ARG(k, 3, a3); ARGB(k, 4, xK); ARG(k, 5, b1); ARG(k, 6, b2); ARG(k, 7, b3);
    ARGB(k, 8, xV); ARG(k, 9, c1); ARG(k, 10, c2); ARG(k, 11, c3); ARGB(k, 12, xS); ARG(k, 13, dvis); ARG(k, 14, nq);
    ARG(k, 15, HKV); ARG(k, 16, G); ARG(k, 17, nk); ARG(k, 18, chunk); ARG(k, 19, nchunk); ARG(k, 20, qscale); ARGB(k, 21, xP);
    size_t g[3] = { 16, (size_t)nchunk * ((G + 1) / 2), (size_t)ns * nq * HKV }, l[3] = { 16, 1, 1 };   // 2 heads per work-group (DEC_HG)
    CHECK(clEnqueueNDRangeKernel(q, k, 3, NULL, g, l, 0, NULL, NULL));
    k = k_dec_comb;
    ARGB(k, 0, xP); ARGB(k, 1, xO); ARG(k, 2, H); ARG(k, 3, nchunk);
    size_t g2[3] = { 128, (size_t)H, (size_t)ns * nq };
    CHECK(clEnqueueNDRangeKernel(q, k, 3, NULL, g2, NULL, 0, NULL, NULL));
    return ew_done(sync, xO, zO, bO, O);
}
