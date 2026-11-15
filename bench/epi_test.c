// epi_test.c -- gen9blas_gemm_pb_f16w_ex (bias / GELU / residual fused into the GEMM epilogue) must be bit-identical
// to gemm_pb_f16w followed by the separate GPU kernels (binop_rows ADD, gelu, binop_rows ADD), as the ggml backend
// runs them today. Shapes of whisper large-v3-turbo's encoder (K 1280, N 1280 / 5120), M rows of activations.
// Build: gcc -O2 -Wall -I../lib -o epi_test epi_test.c -L../build/lib -lgen9blas -Wl,-rpath,$(realpath ../build/lib) -lm
// Run:   ./epi_test [M=600]       exit 0 = all cases bit-identical
#include "gen9blas_ml.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rs = 12345;
static float rnd(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 65536.0f - 0.5f; }

static int run_case(const char *name, int M, int K, int N, int use_bias, int gelu, int res, int inplace) {
    uint16_t *W = malloc((size_t)N * K * 2);
    float *A = malloc((size_t)M * K * 4), *bias = malloc((size_t)N * 4), *R = malloc((size_t)M * N * 4);
    float *C0 = malloc((size_t)M * N * 4), *C1 = malloc((size_t)M * N * 4);
    for (size_t i = 0; i < (size_t)N * K; i++) { _Float16 h = (_Float16)(0.08f * rnd()); memcpy(&W[i], &h, 2); }
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = 4.0f * rnd();
    for (int j = 0; j < N; j++) bias[j] = rnd();
    for (size_t i = 0; i < (size_t)M * N; i++) R[i] = 2.0f * rnd();
    gen9blas_packed_b *P = gen9blas_pack_b(GEN9BLAS_HS, 1, K, N, W, K);
    if (!P) { printf("%s: pack_b failed: %s\n", name, gen9blas_error()); return 1; }
    int nf0 = 0, nf1 = 0;
    // unfused: as the backend today (each op its own GPU kernel)
    if (gen9blas_gemm_pb_f16w(M, A, K, P, C0, N, &nf0)) { printf("%s: gemm: %s\n", name, gen9blas_error()); return 1; }
    if (use_bias && gen9blas_binop_rows(0, M, N, C0, N, bias, 0, 1, C0, N, 1)) { printf("%s: add: %s\n", name, gen9blas_error()); return 1; }
    if (gelu && gen9blas_gelu((size_t)M * N, C0, C0, 1)) { printf("%s: gelu: %s\n", name, gen9blas_error()); return 1; }
    if (res && gen9blas_binop_rows(0, M, N, C0, N, R, N, 1, C0, N, 1)) { printf("%s: add R: %s\n", name, gen9blas_error()); return 1; }
    // fused
    const float *Rp = NULL;
    if (res && inplace) { memcpy(C1, R, (size_t)M * N * 4); Rp = C1; } else if (res) Rp = R;
    if (gen9blas_gemm_pb_f16w_ex(M, A, K, P, C1, N, use_bias ? bias : NULL, gelu, Rp, N, &nf1)) {
        printf("%s: fused: %s\n", name, gen9blas_error()); return 1;
    }
    double mx = 0; long nd = 0;
    for (size_t i = 0; i < (size_t)M * N; i++) if (memcmp(&C0[i], &C1[i], 4)) { nd++; double d = fabs((double)C0[i] - C1[i]); if (d > mx) mx = d; }
    printf("%-34s M %d K %d N %d: %s (%ld differing, max |diff| %.3e; nonfinite %d/%d)\n", name, M, K, N,
           nd ? "DIFFERENT" : "bit-identical", nd, mx, nf0, nf1);
    gen9blas_packed_b_free(P); free(W); free(A); free(bias); free(R); free(C0); free(C1);
    return nd != 0 || nf0 != nf1;
}

// LayerNorm -> GEMM(s): gen9blas_layer_norm + gemm_pb_f16w_ex (packs Y itself) vs gen9blas_layer_norm_pack +
// gemm_pa_f16w_ex (Y packed by the norm). Both Y and the GEMM outputs must be bit-identical.
static int run_norm_case(const char *name, int M, int K, int N, int gelu) {
    uint16_t *W = malloc((size_t)N * K * 2);
    float *X = malloc((size_t)M * K * 4), *w = malloc((size_t)K * 4), *b = malloc((size_t)K * 4), *bias = malloc((size_t)N * 4);
    float *Y0 = malloc((size_t)M * K * 4), *Y1 = malloc((size_t)M * K * 4), *C0 = malloc((size_t)M * N * 4), *C1 = malloc((size_t)M * N * 4);
    for (size_t i = 0; i < (size_t)N * K; i++) { _Float16 h = (_Float16)(0.08f * rnd()); memcpy(&W[i], &h, 2); }
    for (size_t i = 0; i < (size_t)M * K; i++) X[i] = 3.0f * rnd() + 0.5f;
    for (int i = 0; i < K; i++) { w[i] = 1.0f + rnd(); b[i] = 0.2f * rnd(); }
    for (int j = 0; j < N; j++) bias[j] = rnd();
    gen9blas_packed_b *P = gen9blas_pack_b(GEN9BLAS_HS, 1, K, N, W, K);
    gen9blas_packed_a *pa = gen9blas_packed_a_create();
    int nf0 = 0, nf1 = 0;
    if (!P || !pa) { printf("%s: alloc failed: %s\n", name, gen9blas_error()); return 1; }
    if (gen9blas_layer_norm(M, K, X, K, w, b, 1e-5f, Y0, K, 1) ||
        gen9blas_gemm_pb_f16w_ex(M, Y0, K, P, C0, N, bias, gelu, NULL, 0, &nf0)) { printf("%s: ref: %s\n", name, gen9blas_error()); return 1; }
    if (gen9blas_layer_norm_pack(M, K, X, K, w, b, 1e-5f, Y1, K, pa, 1) ||
        gen9blas_gemm_pa_f16w_ex(pa, P, C1, N, bias, gelu, NULL, 0, &nf1)) { printf("%s: pack: %s\n", name, gen9blas_error()); return 1; }
    long dy = 0, dc = 0;
    for (size_t i = 0; i < (size_t)M * K; i++) dy += memcmp(&Y0[i], &Y1[i], 4) != 0;
    for (size_t i = 0; i < (size_t)M * N; i++) dc += memcmp(&C0[i], &C1[i], 4) != 0;
    printf("%-34s M %d K %d N %d: Y %s (%ld), C %s (%ld)\n", name, M, K, N, dy ? "DIFFERENT" : "bit-identical", dy,
           dc ? "DIFFERENT" : "bit-identical", dc);
    gen9blas_packed_a_free(pa); gen9blas_packed_b_free(P);
    free(W); free(X); free(w); free(b); free(bias); free(Y0); free(Y1); free(C0); free(C1);
    return dy != 0 || dc != 0 || nf0 != nf1;
}

int main(int argc, char **argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 600;
    if (gen9blas_init()) { printf("init: %s\n", gen9blas_error()); return 1; }
    if (!gen9blas_has(GEN9BLAS_FEAT_EPILOG_EXT)) { printf("FAIL: gen9blas_epilog_hf8x not in the library\n"); return 1; }
    int bad = 0;
    bad |= run_case("bias (Q, V)", M, 1280, 1280, 1, 0, 0, 0);
    bad |= run_case("bias + GELU (fc1)", M, 1280, 5120, 1, 1, 0, 0);
    bad |= run_case("bias + residual (out, fc2)", M, 1280, 1280, 1, 0, 1, 0);
    bad |= run_case("bias + residual in place (R = C)", M, 5120, 1280, 1, 0, 1, 1);
    bad |= run_case("odd N (tail of 4)", 37, 1280, 1283, 1, 1, 1, 0);
    if (gen9blas_has(GEN9BLAS_FEAT_NORM_PACK)) {
        bad |= run_norm_case("norm -> pack -> GEMM + bias (Q)", M, 1280, 1280, 0);
        bad |= run_norm_case("norm -> pack -> GEMM + bias + GELU (fc1)", M, 1280, 5120, 1);
        bad |= run_norm_case("norm -> pack, partial 16-row band", 37, 1280, 1280, 0);
        bad |= run_norm_case("norm -> pack, K not a multiple of 32", 40, 1000, 512, 0);
    } else { printf("FAIL: gen9blas_layer_norm_pack not in the library\n"); bad = 1; }
    printf("%s\n", bad ? "FAIL" : "PASS: all cases bit-identical");
    return bad;
}
