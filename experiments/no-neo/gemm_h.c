// HGEMM bez NEO: cała ścieżka gen9blas (GEN9BLAS_H, alpha = 1, beta = 0, wymiary wielokrotnością kafelka -> zapis wprost do C)
// w jednym buforze komend: pk_a2_q + pk_bq_w2 (IGC, binding table), bariera, hgemm_v (nasz asembler: A, B stateless,
// C przez binding table). Sprawdza 256 losowych elementów wobec double, mierzy czas (mediana z 200).
// Użycie: ./gemm_h n   (n wielokrotnością 64)
#include <math.h>
#include <time.h>
#include "kernels_pack.h"
#include "kernels_gemm.h"

static double teraz(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int por(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static float h2f(uint16_t h) { _Float16 x; memcpy(&x, &h, 2); return (float)x; }
static uint16_t f2h(float f) { _Float16 x = (_Float16)f; uint16_t h; memcpy(&h, &x, 2); return h; }

static struct kpgi g;
static struct kpgi_bo A, B, C, Ap, Bp;
static uint32_t ja, jb, jg, n;

static void zbuduj(void) {
    const uint32_t M = n, N = n, K = n, Mp = n, Np = n, Kp = n;   // n wielokrotnością 64: bez dopełnień
    kpgi_poczatek(&g);
    uint64_t ra[8] = { A.size, Ap.size }, rb[8] = { B.size, Bp.size }, rc[8] = { 0, 0, C.size };
    { uint64_t a[8] = { A.adr, Ap.adr, M, K, K /* lda */, Kp, 0 /* sA */, Mp };
      kpgi_wywolaj(&g, ja, Kp / 2 / 16, Mp / 16, 1, a, ra); }                 // pk_a2_q: siatka (Kp/2, Mp/16)
    { uint64_t a[8] = { B.adr, Bp.adr, K, N, N /* ldb */, Kp, 0 /* sB */, Np };
      kpgi_wywolaj(&g, jb, Np / 2 / 32, Kp, 1, a, rb); }                      // pk_bq_w2: siatka (Np/2, Kp), SIMD32
    kpgi_bariera(&g);
    { uint64_t a[8] = { Ap.adr, Bp.adr, C.adr, Mp, Np, N /* ldc */, Kp, Kp };
      kpgi_wywolaj(&g, jg, (Np / 32) * (Mp / 16), 1, 1, a, rc); }              // hgemm_v: kafelek 16 × 32, SIMD16
    struct kpgi_bo *bo[5] = { &A, &B, &Ap, &Bp, &C };
    kpgi_wyslij(&g, bo, 5);
}

int main(int argc, char **argv) {
    n = argc > 1 ? atoi(argv[1]) : 1024;
    if (n % 64) { fprintf(stderr, "n musi byc wielokrotnoscia 64\n"); return 1; }
    kpgi_init(&g);
    ja = kpgi_jadro_dodaj(&g, &pk_a2_q_jadro); jb = kpgi_jadro_dodaj(&g, &pk_bq_w2_jadro); jg = kpgi_jadro_dodaj(&g, &hgemm_v_jadro);
    const uint64_t nn = (uint64_t)n * n;
    kpgi_bo_nowy(&g, &A, nn * 2); kpgi_bo_nowy(&g, &B, nn * 2); kpgi_bo_nowy(&g, &C, nn * 2);
    kpgi_bo_nowy(&g, &Ap, nn * 2 + 4096); kpgi_bo_nowy(&g, &Bp, nn * 2 + 4096);   // +4 KB: pętla czyta krok za panel
    uint16_t *a = A.map, *b = B.map, *c = C.map;
    srand(1);
    for (uint64_t i = 0; i < nn; i++) { a[i] = f2h(2.0f * rand() / RAND_MAX - 1); b[i] = f2h(2.0f * rand() / RAND_MAX - 1); }
    memset(c, 0xff, nn * 2);

    zbuduj();
    if (kpgi_czekaj(&g, 5000000000ll)) { fprintf(stderr, "GPU nie skonczylo w 5 s\n"); return 2; }
    double maxe = 0, sume = 0; srand(99);
    for (int s = 0; s < 256; s++) {
        uint32_t i = rand() % n, j = rand() % n; double ref = 0, mag = 0;
        for (uint32_t k = 0; k < n; k++) { double t = (double)h2f(a[(uint64_t)i * n + k]) * h2f(b[(uint64_t)k * n + j]); ref += t; mag += fabs(t); }
        double e = fabs(h2f(c[(uint64_t)i * n + j]) - ref) / mag; sume += e; if (e > maxe || isnan(e)) maxe = e;
    }
    printf("HGEMM bez NEO n = %u: blad wzgl. sr = %.1e, max = %.1e -> %s\n", n, sume / 256, maxe,
           maxe < 1e-2 ? "OK" : "ZLE");
    if (!(maxe < 1e-2)) return 3;

    enum { R = 200 }; double t[R]; double c0 = clock() / (double)CLOCKS_PER_SEC, s0 = teraz();
    int R2 = n >= 4096 ? 20 : R;
    for (int r = 0; r < R2; r++) { double x = teraz(); zbuduj(); kpgi_czekaj(&g, 5000000000ll); t[r] = teraz() - x; }
    double cpu = clock() / (double)CLOCKS_PER_SEC - c0, sc = teraz() - s0;
    qsort(t, R2, sizeof *t, por);
    printf("  mediana %.3f ms (%.1f GFLOPS), min %.3f ms, CPU %.1f us/wywolanie (%.1f%% rdzenia)\n", t[R2 / 2] * 1e3,
           2.0 * n * n * n / t[R2 / 2] / 1e9, t[0] * 1e3, cpu / R2 * 1e6, 100 * cpu / sc);
    return 0;
}
