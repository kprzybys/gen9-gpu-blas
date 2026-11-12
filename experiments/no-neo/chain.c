// Step 1 without NEO: a chain of K dependent kernels in one batch buffer (store, then K-1 × copy: b[i] = a[i] + 1,
// alternating X -> Y -> X ...). Checks the result v + i + K - 1 and measures time: one buffer with K kernels versus K separate
// submits. Usage: ./chain K N [barrier 1|0]   (N: number of elements, a multiple of 16)
#include <time.h>
#ifdef BTI
#include "kernels_bti.h"   // the same kernels compiled with a binding table (stateful)
#else
#include "kernels.h"
#endif

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int cmp_d(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

static struct g9i g;
static struct g9i_bo X, Y;
static uint64_t sizes[3];
static uint32_t j_store, j_copy, j_reverse, K, N, barrier, cross;

// Buffer contents: kernels from..to-1 of the chain (0 = store). Result buffer after kernel k: X for even k, Y for odd k.
static void build(uint32_t from, uint32_t to) {
    g9i_begin(&g);
    for (uint32_t k = from; k < to; k++) {
        if (k > from && barrier) g9i_barrier(&g);
        if (k == 0) { uint64_t a[2] = { X.addr, 1000 }; g9i_call(&g, j_store, N / 16, 1, 1, a, sizes); }
        else { uint64_t a[3] = { k % 2 ? X.addr : Y.addr, k % 2 ? Y.addr : X.addr, N }; g9i_call(&g, cross ? j_reverse : j_copy, N / 16, 1, 1, a, sizes); }
    }
    struct g9i_bo *bo[2] = { &X, &Y };
    g9i_submit(&g, bo, 2);
}

int main(int argc, char **argv) {
    K = argc > 1 ? atoi(argv[1]) : 8; N = argc > 2 ? atoi(argv[2]) : 1 << 20; barrier = argc > 3 ? atoi(argv[3]) : 1;
    g9i_init(&g);
    j_store = g9i_add_kernel(&g, &store_kernel); j_copy = g9i_add_kernel(&g, &copy_kernel); j_reverse = g9i_add_kernel(&g, &reverse_kernel);
    cross = getenv("CROSS") != NULL;   // reverse steps (b[i] = a[n-1-i] + 1) instead of copy
    g9i_bo_new(&g, &X, N * 4); g9i_bo_new(&g, &Y, N * 4);
    sizes[0] = sizes[1] = (uint64_t)N * 4;

    memset(X.map, 0xff, N * 4); memset(Y.map, 0xff, N * 4);
    build(0, K);
    if (g9i_wait(&g, 5000000000ll)) { fprintf(stderr, "GPU did not finish in 5 s\n"); return 2; }
    const uint32_t *w = (K - 1) % 2 ? Y.map : X.map; uint32_t bad = 0;
    #define EXPECT(i) (1000 + ((cross && (K - 1) % 2) ? N - 1 - (i) : (i)) + K - 1)
    for (uint32_t i = 0; i < N; i++) if (w[i] != EXPECT(i)) { if (bad < 3) printf("  w[%u] = %u (expected %u)\n", i, w[i], EXPECT(i)); bad++; }
    printf("K = %u, N = %u, barrier %u%s: %s (%u bad)\n", K, N, barrier, cross ? ", cross" : "", bad ? "BAD" : "OK", bad);

    if (getenv("RACE")) {   // 50 runs, clearing before each one, checking every result
        uint32_t bad_runs = 0;
        for (int r = 0; r < 50; r++) {
            memset(X.map, 0xff, N * 4); memset(Y.map, 0xff, N * 4);
            build(0, K);
            if (g9i_wait(&g, 5000000000ll)) { fprintf(stderr, "GPU did not finish\n"); return 2; }
            uint32_t z = 0; for (uint32_t i = 0; i < N; i++) z += w[i] != EXPECT(i);
            bad_runs += z != 0;
        }
        printf("  race: %u of 50 runs with a bad result\n", bad_runs);
        return 0;
    }
    enum { R = 200 }; double t1[R], t2[R], c0, cpu1, cpu2;
    c0 = clock() / (double)CLOCKS_PER_SEC;
    for (int r = 0; r < R; r++) { double a = now(); build(0, K); g9i_wait(&g, 5000000000ll); t1[r] = now() - a; }
    cpu1 = (clock() / (double)CLOCKS_PER_SEC - c0) / R;
    c0 = clock() / (double)CLOCKS_PER_SEC;
    for (int r = 0; r < R; r++) { double a = now(); for (uint32_t k = 0; k < K; k++) { build(k, k + 1); g9i_wait(&g, 5000000000ll); } t2[r] = now() - a; }
    cpu2 = (clock() / (double)CLOCKS_PER_SEC - c0) / R;
    qsort(t1, R, sizeof *t1, cmp_d); qsort(t2, R, sizeof *t2, cmp_d);
    printf("  one buffer with %u kernels: median %.1f us, CPU %.1f us\n", K, t1[R / 2] * 1e6, cpu1 * 1e6);
    printf("  %u separate submits:      median %.1f us, CPU %.1f us\n", K, t2[R / 2] * 1e6, cpu2 * 1e6);
    return bad ? 3 : 0;
}
