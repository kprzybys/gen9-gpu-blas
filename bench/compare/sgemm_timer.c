// Wall time spent inside cblas_sgemm, per shape, for any BLAS underneath (LD_PRELOAD before it):
//   LD_PRELOAD="$PWD/libsgemm_timer.so <repo>/lib/libgen9blas_blas.so" <program>   (gen9blas)
//   LD_PRELOAD="$PWD/libsgemm_timer.so" <program>                                               (the program's own BLAS)
// At exit prints, per (order, transA, transB, M, N, K): calls, total ms, GFLOPS; then the total and the process wall time.
// Build: gcc -O2 -Wall -shared -fPIC -o libsgemm_timer.so sgemm_timer.c -ldl
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef void (*sgemm_fn)(int, int, int, int, int, int, float, const float *, int, const float *, int, float, float *, int);
static sgemm_fn next;
static struct { int o, ta, tb, M, N, K; long n; double s; } sh[256];
static int nsh;
static double t_start;

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

__attribute__((constructor)) static void start(void) { t_start = now(); }

__attribute__((destructor)) static void report(void) {
    double tot = 0, fl = 0; long n = 0;
    for (int i = 0; i < nsh; i++) {
        double f = 2.0 * sh[i].M * sh[i].N * sh[i].K * sh[i].n;
        fprintf(stderr, "sgemm_timer: o=%d ta=%d tb=%d M=%-6d N=%-6d K=%-6d %6ld calls %10.1f ms %7.1f GFLOPS\n",
                sh[i].o, sh[i].ta, sh[i].tb, sh[i].M, sh[i].N, sh[i].K, sh[i].n, sh[i].s * 1e3, f / sh[i].s / 1e9);
        tot += sh[i].s; fl += f; n += sh[i].n;
    }
    if (n) fprintf(stderr, "sgemm_timer: total %ld calls %.3f s in cblas_sgemm (%.1f GFLOPS), process wall %.3f s\n",
                   n, tot, fl / tot / 1e9, now() - t_start);
}

void cblas_sgemm(int o, int ta, int tb, int M, int N, int K, float alpha, const float *A, int lda,
                 const float *B, int ldb, float beta, float *C, int ldc) {
    if (!next) next = (sgemm_fn)dlsym(RTLD_NEXT, "cblas_sgemm");
    double t0 = now();
    next(o, ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    double dt = now() - t0;
    int i = 0;
    while (i < nsh && !(sh[i].o == o && sh[i].ta == ta && sh[i].tb == tb && sh[i].M == M && sh[i].N == N && sh[i].K == K)) i++;
    if (i == nsh) {
        if (nsh == 256) return;
        sh[i].o = o; sh[i].ta = ta; sh[i].tb = tb; sh[i].M = M; sh[i].N = N; sh[i].K = K; nsh++;
    }
    sh[i].n++; sh[i].s += dt;
}
