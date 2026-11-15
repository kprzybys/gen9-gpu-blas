// decoder_gemm.c -- weight GEMMs of one whisper-large-v3-turbo DECODER step on the GPU for M rows (M = tokens decoded
// together: beams x audio segments), FP16 weights, A/C FP32 as in the ggml backend. Two library paths:
//   packed : weights packed once (GEN9BLAS_HS), gen9blas_gemm_pb_f16w (hgemm_kc_v, the backend's MUL_MAT path)
//   skinny : gen9blas_gemm(GEN9BLAS_H) on row-major FP16 weights (skinny kernels when min(M, N) <= 16; FP16 A/C)
// Per decoder step (per layer: self Q K V O, cross Q O = 6 x 1280x1280, fc1 5120x1280, fc2 1280x5120; x 4 layers; +
// vocabulary 51866x1280; cross K/V are computed once per window and not counted). Each call includes its own wait,
// like the backend today. Prints ms per step, ms per token (/ M) and weight GB/s.
// Build (kp): gcc -O2 -Wall -I../lib -o decoder_gemm decoder_gemm.c -L../lib -lgen9blas -Wl,-rpath,$(realpath ../lib) -lm
// Run:  ./decoder_gemm [reps=5]
#include "gen9blas_ml.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint16_t h(float f) { _Float16 x = (_Float16)f; uint16_t u; memcpy(&u, &x, 2); return u; }

typedef struct { const char *name; int N, K, count; } shape_t;
static const shape_t SH[] = { { "1280x1280 (x6 per layer)", 1280, 1280, 24 }, { "fc1 5120x1280", 5120, 1280, 4 },
                              { "fc2 1280x5120", 1280, 5120, 4 }, { "vocab 51866x1280", 51866, 1280, 1 } };
#define NS 4

int main(int argc, char **argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 5;
    const int Ms[] = { 1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 64 }; const int nM = sizeof Ms / sizeof Ms[0];
    if (gen9blas_init()) { printf("init: %s\n", gen9blas_error()); return 1; }
    srand(1);
    gen9blas_packed_b *P[NS]; uint16_t *W[NS];
    for (int s = 0; s < NS; s++) {
        const size_t n = (size_t)SH[s].N * SH[s].K;
        W[s] = gen9blas_alloc(n * 2);
        for (size_t i = 0; i < n; i++) W[s][i] = h((rand() / (float)RAND_MAX - 0.5f) * 0.1f);
        P[s] = gen9blas_pack_b(GEN9BLAS_HS, 1, SH[s].K, SH[s].N, W[s], SH[s].K);
        if (!P[s]) { printf("pack: %s\n", gen9blas_error()); return 1; }
    }
    const int Kmax = 5120, Nmax = 51866, Mmax = 64;
    float *A = gen9blas_alloc((size_t)Mmax * Kmax * 4), *C = gen9blas_alloc((size_t)Mmax * Nmax * 4);
    uint16_t *Ah = gen9blas_alloc((size_t)Mmax * Kmax * 2), *Ch = gen9blas_alloc((size_t)Mmax * Nmax * 2);
    for (size_t i = 0; i < (size_t)Mmax * Kmax; i++) { A[i] = rand() / (float)RAND_MAX - 0.5f; Ah[i] = h(A[i]); }
    double wbytes = 0;
    for (int s = 0; s < NS; s++) wbytes += 2.0 * SH[s].N * SH[s].K * SH[s].count;
    printf("# whisper-large-v3-turbo decoder step: %.1f MB FP16 weights in %d GEMMs; best of %d; each call waits\n",
           wbytes / 1e6, 24 + 4 + 4 + 1, reps);
    printf("%-7s %4s %10s %10s %8s |", "path", "M", "ms/step", "ms/token", "GB/s");
    for (int s = 0; s < NS; s++) printf(" %24s", SH[s].name);
    printf("   (ms per call)\n");
    for (int path = 0; path < 2; path++)
        for (int mi = 0; mi < nM; mi++) {
            const int M = Ms[mi];
            double per[NS];
            for (int s = 0; s < NS; s++) {
                double best = 1e9; const int N = SH[s].N, K = SH[s].K;
                for (int r = 0; r < reps + 1; r++) {
                    int nf = 0; double t = now(); int rc;
                    if (path == 0) rc = gen9blas_gemm_pb_f16w(M, A, K, P[s], C, N, &nf);
                    else rc = gen9blas_gemm(GEN9BLAS_H, 0, 1, M, N, K, 1.0, Ah, K, W[s], K, 0.0, Ch, N);
                    t = now() - t;
                    if (rc) { printf("gemm: %s\n", gen9blas_error()); return 1; }
                    if (r && t < best) best = t;   // first call: warm-up
                }
                per[s] = best * 1e3;
            }
            double step = 0;
            for (int s = 0; s < NS; s++) step += per[s] * SH[s].count;
            printf("%-7s %4d %10.2f %10.3f %8.1f |", path ? "skinny" : "packed", M, step, step / M, wbytes / step / 1e6);
            for (int s = 0; s < NS; s++) printf(" %24.3f", per[s]);
            printf("\n"); fflush(stdout);
        }
    return 0;
}
