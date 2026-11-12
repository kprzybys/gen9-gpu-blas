// Krok 1 bez NEO: łańcuch K zależnych jąder w jednym buforze komend (zapis, potem K-1 × kopiuj: b[i] = a[i] + 1,
// na przemian X -> Y -> X ...). Sprawdza wynik v + i + K - 1 i mierzy czas: jeden bufor z K jądrami wobec K osobnych
// wysłań. Użycie: ./lancuch K N [bariera 1|0]   (N: liczba elementów, wielokrotność 16)
#include <time.h>
#ifdef BTI
#include "kernels_bti.h"   // te same jądra skompilowane z binding table (stateful)
#else
#include "kernels.h"
#endif

static double teraz(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int por(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

static struct kpgi g;
static struct kpgi_bo X, Y;
static uint64_t rozm[3];
static uint32_t jz, jk, jo, K, N, bariera, krzyz;

// Zawartość bufora: od..do-1 jąder łańcucha (0 = zapis). Bufor wyniku po jądrze k: X dla k parzystych, Y dla nieparzystych.
static void zbuduj(uint32_t od, uint32_t do_) {
    kpgi_poczatek(&g);
    for (uint32_t k = od; k < do_; k++) {
        if (k > od && bariera) kpgi_bariera(&g);
        if (k == 0) { uint64_t a[2] = { X.adr, 1000 }; kpgi_wywolaj(&g, jz, N / 16, 1, 1, a, rozm); }
        else { uint64_t a[3] = { k % 2 ? X.adr : Y.adr, k % 2 ? Y.adr : X.adr, N }; kpgi_wywolaj(&g, krzyz ? jo : jk, N / 16, 1, 1, a, rozm); }
    }
    struct kpgi_bo *bo[2] = { &X, &Y };
    kpgi_wyslij(&g, bo, 2);
}

int main(int argc, char **argv) {
    K = argc > 1 ? atoi(argv[1]) : 8; N = argc > 2 ? atoi(argv[2]) : 1 << 20; bariera = argc > 3 ? atoi(argv[3]) : 1;
    kpgi_init(&g);
    jz = kpgi_jadro_dodaj(&g, &zapis_jadro); jk = kpgi_jadro_dodaj(&g, &kopiuj_jadro); jo = kpgi_jadro_dodaj(&g, &odwroc_jadro);
    krzyz = getenv("KRZYZ") != NULL;   // kroki odwroc (b[i] = a[n-1-i] + 1) zamiast kopiuj
    kpgi_bo_nowy(&g, &X, N * 4); kpgi_bo_nowy(&g, &Y, N * 4);
    rozm[0] = rozm[1] = (uint64_t)N * 4;

    memset(X.map, 0xff, N * 4); memset(Y.map, 0xff, N * 4);
    zbuduj(0, K);
    if (kpgi_czekaj(&g, 5000000000ll)) { fprintf(stderr, "GPU nie skonczylo w 5 s\n"); return 2; }
    const uint32_t *w = (K - 1) % 2 ? Y.map : X.map; uint32_t zle = 0;
    #define OCZEK(i) (1000 + ((krzyz && (K - 1) % 2) ? N - 1 - (i) : (i)) + K - 1)
    for (uint32_t i = 0; i < N; i++) if (w[i] != OCZEK(i)) { if (zle < 3) printf("  w[%u] = %u (oczekiwane %u)\n", i, w[i], OCZEK(i)); zle++; }
    printf("K = %u, N = %u, bariera %u%s: %s (%u zlych)\n", K, N, bariera, krzyz ? ", na krzyz" : "", zle ? "ZLE" : "OK", zle);

    if (getenv("WYSCIG")) {   // 50 uruchomień, czyszczenie przed każdym, sprawdzenie każdego wyniku
        uint32_t zlych_przebiegow = 0;
        for (int r = 0; r < 50; r++) {
            memset(X.map, 0xff, N * 4); memset(Y.map, 0xff, N * 4);
            zbuduj(0, K);
            if (kpgi_czekaj(&g, 5000000000ll)) { fprintf(stderr, "GPU nie skonczylo\n"); return 2; }
            uint32_t z = 0; for (uint32_t i = 0; i < N; i++) z += w[i] != OCZEK(i);
            zlych_przebiegow += z != 0;
        }
        printf("  wyscig: %u z 50 przebiegow ze zlym wynikiem\n", zlych_przebiegow);
        return 0;
    }
    enum { R = 200 }; double t1[R], t2[R], c0, cpu1, cpu2;
    c0 = clock() / (double)CLOCKS_PER_SEC;
    for (int r = 0; r < R; r++) { double a = teraz(); zbuduj(0, K); kpgi_czekaj(&g, 5000000000ll); t1[r] = teraz() - a; }
    cpu1 = (clock() / (double)CLOCKS_PER_SEC - c0) / R;
    c0 = clock() / (double)CLOCKS_PER_SEC;
    for (int r = 0; r < R; r++) { double a = teraz(); for (uint32_t k = 0; k < K; k++) { zbuduj(k, k + 1); kpgi_czekaj(&g, 5000000000ll); } t2[r] = teraz() - a; }
    cpu2 = (clock() / (double)CLOCKS_PER_SEC - c0) / R;
    qsort(t1, R, sizeof *t1, por); qsort(t2, R, sizeof *t2, por);
    printf("  jeden bufor z %u jadrami: mediana %.1f us, CPU %.1f us\n", K, t1[R / 2] * 1e6, cpu1 * 1e6);
    printf("  %u osobnych wyslan:       mediana %.1f us, CPU %.1f us\n", K, t2[R / 2] * 1e6, cpu2 * 1e6);
    return zle ? 3 : 0;
}
