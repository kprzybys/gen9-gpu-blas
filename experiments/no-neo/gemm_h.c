// HGEMM without NEO: the whole gen9blas path (GEN9BLAS_H, alpha = 1, beta = 0, dimensions a multiple of the tile -> direct write to C)
// in one batch buffer: pk_a2_q + pk_bq_w2 (IGC, binding table), barrier, hgemm_v (our assembler: A, B stateless,
// C through the binding table). Checks 256 random elements against double, measures time (median of 200).
// Usage: ./gemm_h n   (n a multiple of 64)
#include <math.h>
#include <time.h>
#include "kernels_pack.h"
#include "kernels_gemm.h"

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int cmp_d(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static float h2f(uint16_t h) { _Float16 x; memcpy(&x, &h, 2); return (float)x; }
static uint16_t f2h(float f) { _Float16 x = (_Float16)f; uint16_t h; memcpy(&h, &x, 2); return h; }

static struct g9i g;
static struct g9i_bo A, B, C, Ap, Bp;
static uint32_t j_pack_a, j_pack_b, j_gemm, n;

static void build(void) {
    const uint32_t M = n, N = n, K = n, Mp = n, Np = n, Kp = n;   // n a multiple of 64: no padding
    g9i_begin(&g);
    uint64_t ra[8] = { A.size, Ap.size }, rb[8] = { B.size, Bp.size }, rc[8] = { 0, 0, C.size };
    { uint64_t a[8] = { A.addr, Ap.addr, M, K, K /* lda */, Kp, 0 /* sA */, Mp };
      g9i_call(&g, j_pack_a, Kp / 2 / 16, Mp / 16, 1, a, ra); }                 // pk_a2_q: grid (Kp/2, Mp/16)
    { uint64_t a[8] = { B.addr, Bp.addr, K, N, N /* ldb */, Kp, 0 /* sB */, Np };
      g9i_call(&g, j_pack_b, Np / 2 / 32, Kp, 1, a, rb); }                      // pk_bq_w2: grid (Np/2, Kp), SIMD32
    g9i_barrier(&g);
    { uint64_t a[8] = { Ap.addr, Bp.addr, C.addr, Mp, Np, N /* ldc */, Kp, Kp };
      g9i_call(&g, j_gemm, (Np / 32) * (Mp / 16), 1, 1, a, rc); }              // hgemm_v: 16 × 32 tile, SIMD16
    struct g9i_bo *bo[5] = { &A, &B, &Ap, &Bp, &C };
    g9i_submit(&g, bo, 5);
}

int main(int argc, char **argv) {
    n = argc > 1 ? atoi(argv[1]) : 1024;
    if (n % 64) { fprintf(stderr, "n must be a multiple of 64\n"); return 1; }
    g9i_init(&g);
    j_pack_a = g9i_add_kernel(&g, &pk_a2_q_kernel); j_pack_b = g9i_add_kernel(&g, &pk_bq_w2_kernel); j_gemm = g9i_add_kernel(&g, &hgemm_v_kernel);
    const uint64_t nn = (uint64_t)n * n;
    g9i_bo_new(&g, &A, nn * 2); g9i_bo_new(&g, &B, nn * 2); g9i_bo_new(&g, &C, nn * 2);
    g9i_bo_new(&g, &Ap, nn * 2 + 4096); g9i_bo_new(&g, &Bp, nn * 2 + 4096);   // +4 KB: the loop reads one step past the panel
    uint16_t *a = A.map, *b = B.map, *c = C.map;
    srand(1);
    for (uint64_t i = 0; i < nn; i++) { a[i] = f2h(2.0f * rand() / RAND_MAX - 1); b[i] = f2h(2.0f * rand() / RAND_MAX - 1); }
    memset(c, 0xff, nn * 2);

    build();
    if (g9i_wait(&g, 5000000000ll)) { fprintf(stderr, "GPU did not finish in 5 s\n"); return 2; }
    double maxe = 0, sume = 0; srand(99);
    for (int s = 0; s < 256; s++) {
        uint32_t i = rand() % n, j = rand() % n; double ref = 0, mag = 0;
        for (uint32_t k = 0; k < n; k++) { double t = (double)h2f(a[(uint64_t)i * n + k]) * h2f(b[(uint64_t)k * n + j]); ref += t; mag += fabs(t); }
        double e = fabs(h2f(c[(uint64_t)i * n + j]) - ref) / mag; sume += e; if (e > maxe || isnan(e)) maxe = e;
    }
    printf("HGEMM without NEO n = %u: rel. error avg = %.1e, max = %.1e -> %s\n", n, sume / 256, maxe,
           maxe < 1e-2 ? "OK" : "BAD");
    if (!(maxe < 1e-2)) return 3;

    enum { R = 200 }; double t[R]; double c0 = clock() / (double)CLOCKS_PER_SEC, s0 = now();
    int R2 = n >= 4096 ? 20 : R;
    for (int r = 0; r < R2; r++) { double x = now(); build(); g9i_wait(&g, 5000000000ll); t[r] = now() - x; }
    double cpu = clock() / (double)CLOCKS_PER_SEC - c0, wall = now() - s0;
    qsort(t, R2, sizeof *t, cmp_d);
    printf("  median %.3f ms (%.1f GFLOPS), min %.3f ms, CPU %.1f us/call (%.1f%% of a core)\n", t[R2 / 2] * 1e3,
           2.0 * n * n * n / t[R2 / 2] / 1e9, t[0] * 1e3, cpu / R2 * 1e6, 100 * cpu / wall);
    return 0;
}
