// Stały koszt wywołania libgen9blas (krok 2, 2026-11-12): małe axpy w pętli z C, czas ściany i CPU procesu na wywołanie.
// Budowa: gcc -O2 -o bench/overhead bench/overhead.c -Llib -lgen9blas -Wl,-rpath,$PWD/lib
// Uruchom: bench/narzut [n] [R] [usm|zwykla]   (KCL_PHASES=1: rozkład na fazy kcl na stderr)
#include "../lib/gen9blas.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double teraz(clockid_t c) { struct timespec t; clock_gettime(c, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 4096, R = argc > 2 ? atoi(argv[2]) : 5000;
    int usm = argc > 3 && !strcmp(argv[3], "usm");
    if (gen9blas_init(NULL)) { fprintf(stderr, "gen9blas_init: %s\n", gen9blas_error()); return 1; }
    float *x = usm ? gen9blas_alloc(n * 4) : aligned_alloc(64, (n * 4 + 63) / 64 * 64);
    float *y = usm ? gen9blas_alloc(n * 4) : aligned_alloc(64, (n * 4 + 63) / 64 * 64);
    for (int i = 0; i < n; i++) { x[i] = 1; y[i] = 0; }
    for (int i = 0; i < 50; i++) gen9blas_axpy(GEN9BLAS_S, n, 1e-3, x, 1, y, 1);
    double s = teraz(CLOCK_MONOTONIC), c = teraz(CLOCK_PROCESS_CPUTIME_ID);
    for (int i = 0; i < R; i++) if (gen9blas_axpy(GEN9BLAS_S, n, 1e-3, x, 1, y, 1)) { fprintf(stderr, "%s\n", gen9blas_error()); return 1; }
    s = (teraz(CLOCK_MONOTONIC) - s) / R; c = (teraz(CLOCK_PROCESS_CPUTIME_ID) - c) / R;
    printf("axpy n=%d %s: %.1f µs ściany, %.1f µs CPU na wywołanie\n", n, usm ? "usm" : "zwykla", s, c);
    return 0;
}
