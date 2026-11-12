// Mikrotest userptr (DRM_IOCTL_I915_GEM_USERPTR): GPU czyta i pisze zwykłą pamięć procesu (malloc, jak NumPy) bez kopii.
// Porównanie dla rozmiaru S (bajty) jednego jądra kopiuj (b[i] = a[i] + 1, wejście S, wyjście S):
//   kopia  -- jak dziś gen9blas dla tablic spoza USM: memcpy do bufora GEM, jądro, memcpy wyniku z powrotem;
//   up     -- userptr tworzony w każdym wywołaniu (create, execbuf: przypięcie stron, close);
//   up_kesz-- userptr utworzony raz, wielokrotne użycie (strony przypięte; koszt jak bufor GEM?);
//   gem    -- samo jądro na buforach GEM (dolna granica).
// Sprawdza poprawność (także: CPU zmienia dane wejściowe między wywołaniami -> GPU widzi nowe) oraz flagi
// I915_USERPTR_READ_ONLY i PROBE. Użycie: ./userptr [S_MB ...]
#include <time.h>
#include "kernels.h"

static double teraz(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static double cpu(void) { struct timespec t; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int por(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

static struct kpgi g;
static uint32_t jk;

// userptr nad stronami obejmującymi [p, p+n); zwraca obiekt (adres GPU = adr) i przesunięcie p w obiekcie.
static int up_nowy(struct kpgi_bo *b, void *p, uint64_t n, uint64_t adr, uint32_t flagi, uint64_t *przes) {
    const uintptr_t s = (uintptr_t)p & ~4095ul, e = ((uintptr_t)p + n + 4095) & ~4095ul;
    struct drm_i915_gem_userptr u = { .user_ptr = s, .user_size = e - s, .flags = flagi };
    if (ioctl(g.fd, DRM_IOCTL_I915_GEM_USERPTR, &u)) return -errno;
    *b = (struct kpgi_bo){ .h = u.handle, .adr = adr, .size = e - s, .map = NULL, .pisany = 1 };
    if (przes) *przes = (uintptr_t)p - s;
    return 0;
}
static void zamknij(uint32_t h) { struct drm_gem_close c = { .handle = h }; KPGI_SPR(ioctl(g.fd, DRM_IOCTL_GEM_CLOSE, &c)); }

static void uruchom(struct kpgi_bo *we, uint64_t a_we, struct kpgi_bo *wy, uint64_t a_wy, uint64_t S) {
    kpgi_poczatek(&g);
    uint64_t a[2] = { a_we, a_wy };
    kpgi_wywolaj(&g, jk, (uint32_t)(S / 4 / 16), 1, 1, a, NULL);
    struct kpgi_bo *bo[2] = { we, wy }; we->pisany = 0; wy->pisany = 1;
    kpgi_wyslij(&g, bo, 2);
    if (kpgi_czekaj(&g, 10000000000ll)) { fprintf(stderr, "GPU nie skonczylo w 10 s\n"); exit(2); }
}
static uint64_t zle(const uint32_t *we, const uint32_t *wy, uint64_t n) {
    uint64_t z = 0; for (uint64_t i = 0; i < n; i++) z += wy[i] != we[i] + 1; return z;
}

int main(int argc, char **argv) {
    kpgi_init(&g);
    jk = kpgi_jadro_dodaj(&g, &kopiuj_jadro);
    { int v = 0; struct drm_i915_getparam gp = { .param = I915_PARAM_HAS_USERPTR_PROBE, .value = &v };
      int r = ioctl(g.fd, DRM_IOCTL_I915_GETPARAM, &gp);
      printf("I915_PARAM_HAS_USERPTR_PROBE: %s\n", r ? strerror(errno) : v ? "tak" : "nie"); }
    {   // flagi na małym buforze
        void *p = aligned_alloc(4096, 8192); struct kpgi_bo b; int r;
        r = up_nowy(&b, p, 8192, 0, I915_USERPTR_READ_ONLY, NULL);
        printf("I915_USERPTR_READ_ONLY: %s\n", r ? strerror(-r) : "przyjete"); if (!r) zamknij(b.h);
        r = up_nowy(&b, p, 8192, 0, I915_USERPTR_PROBE, NULL);
        printf("I915_USERPTR_PROBE (poprawny adres): %s\n", r ? strerror(-r) : "przyjete"); if (!r) zamknij(b.h);
        r = up_nowy(&b, (void *)0x1000, 8192, 0, I915_USERPTR_PROBE, NULL);
        printf("I915_USERPTR_PROBE (zly adres 0x1000): %s\n", r ? strerror(-r) : "przyjete (!)"); if (!r) zamknij(b.h);
        free(p);
    }
    if (getenv("FAZY")) {   // rozbicie kosztu userptr co wywołanie na fazy (CPU w us), 64 MB wejścia + 64 MB wyjścia
        const uint64_t S = 64ull << 20, n = S / 4; uint32_t *h = malloc(S), *h2 = malloc(S);
        for (uint64_t i = 0; i < n; i++) { h[i] = (uint32_t)i; h2[i] = 0; }
        for (int r = 0; r < 4; r++) {
            struct kpgi_bo uw, uy; uint64_t pw, py; double c0 = cpu(), t0 = teraz();
            KPGI_SPR(up_nowy(&uw, h, S, 0x100000000ull, 0, &pw)); KPGI_SPR(up_nowy(&uy, h2, S, 0x200000000ull, 0, &py));
            double c1 = cpu();
            kpgi_poczatek(&g); uint64_t a[2] = { 0x100000000ull + pw, 0x200000000ull + py };
            kpgi_wywolaj(&g, jk, (uint32_t)(n / 16), 1, 1, a, NULL);
            struct kpgi_bo *bo[2] = { &uw, &uy }; uw.pisany = 0; kpgi_wyslij(&g, bo, 2);
            double c2 = cpu(), t2 = teraz(); kpgi_czekaj(&g, 10000000000ll); double c3 = cpu(), t3 = teraz();
            zamknij(uw.h); zamknij(uy.h); double c4 = cpu();
            printf("FAZY: create %.0f, execbuf %.0f (sciana %.0f), wait %.0f (sciana %.0f), close %.0f us CPU\n",
                   (c1 - c0) * 1e6, (c2 - c1) * 1e6, (t2 - t0) * 1e6, (c3 - c2) * 1e6, (t3 - t2) * 1e6, (c4 - c3) * 1e6);
        }
        return 0;
    }
    int nr = argc > 1 ? argc - 1 : 5; double mbs[] = { 0.0625, 1, 4, 16, 64 };
    const uint64_t A_WE = 0x100000000ull, A_WY = 0x200000000ull;   // stałe adresy GPU dla obiektów userptr
    printf("\n%8s | %-26s | %-26s | %-26s | %-12s | spr.\n", "S", "kopia: czas / CPU [us]", "up co wywolanie", "up raz utworzony", "samo jadro");
    for (int ir = 0; ir < nr; ir++) {
        const uint64_t S = (uint64_t)((argc > 1 ? atof(argv[ir + 1]) : mbs[ir]) * (1 << 20)) & ~63ull, n = S / 4;
        // pamięć jak w NumPy: malloc (duże bloki: mmap, początek 16 B za granicą strony)
        uint32_t *h = malloc(S), *h2 = malloc(S);
        for (uint64_t i = 0; i < n; i++) { h[i] = (uint32_t)(i * 2654435761u); h2[i] = 0; }
        struct kpgi_bo BW, BY; kpgi_bo_nowy(&g, &BW, S); kpgi_bo_nowy(&g, &BY, S);
        const int R = S >= (16 << 20) ? 15 : 60;
        double t[4][60], c[4] = { 0 }; uint64_t z = 0;
        for (int r = 0; r < R; r++) {
            double a = teraz(), ca = cpu();   // kopia
            memcpy(BW.map, h, S); uruchom(&BW, BW.adr, &BY, BY.adr, S); memcpy(h2, BY.map, S);
            t[0][r] = teraz() - a; c[0] += cpu() - ca;
        }
        z += zle(h, h2, n); memset(h2, 0, S);
        for (int r = 0; r < R; r++) {
            double a = teraz(), ca = cpu();   // userptr co wywołanie
            struct kpgi_bo uw, uy; uint64_t pw, py;
            KPGI_SPR(up_nowy(&uw, h, S, A_WE, 0, &pw)); KPGI_SPR(up_nowy(&uy, h2, S, A_WY, 0, &py));
            uruchom(&uw, A_WE + pw, &uy, A_WY + py, S);
            zamknij(uw.h); zamknij(uy.h);
            t[1][r] = teraz() - a; c[1] += cpu() - ca;
        }
        z += zle(h, h2, n); memset(h2, 0, S);
        struct kpgi_bo uw, uy; uint64_t pw, py;
        KPGI_SPR(up_nowy(&uw, h, S, A_WE, 0, &pw)); KPGI_SPR(up_nowy(&uy, h2, S, A_WY, 0, &py));
        uruchom(&uw, A_WE + pw, &uy, A_WY + py, S);   // pierwsze użycie (przypięcie) poza pomiarem
        for (int r = 0; r < R; r++) {
            if (r == R / 2) for (uint64_t i = 0; i < n; i++) h[i] ^= 0x5a5a;   // CPU zmienia wejście: GPU musi to zobaczyć
            double a = teraz(), ca = cpu();
            uruchom(&uw, A_WE + pw, &uy, A_WY + py, S);
            t[2][r] = teraz() - a; c[2] += cpu() - ca;
        }
        z += zle(h, h2, n);
        zamknij(uw.h); zamknij(uy.h);
        for (int r = 0; r < R; r++) {
            double a = teraz(), ca = cpu(); uruchom(&BW, BW.adr, &BY, BY.adr, S); t[3][r] = teraz() - a; c[3] += cpu() - ca;
        }
        for (int k = 0; k < 4; k++) qsort(t[k], R, sizeof(double), por);
        printf("%6.3g MB | %10.0f / %-13.0f | %10.0f / %-13.0f | %10.0f / %-13.0f | %10.0f   | %s (h%%4096 = %lu)\n",
               S / 1048576.0, t[0][R / 2] * 1e6, c[0] / R * 1e6, t[1][R / 2] * 1e6, c[1] / R * 1e6,
               t[2][R / 2] * 1e6, c[2] / R * 1e6, t[3][R / 2] * 1e6, z ? "ZLE" : "OK", (uintptr_t)h % 4096);
        zamknij(BW.h); zamknij(BY.h); munmap(BW.map, BW.size); munmap(BY.map, BY.size);
        free(h); free(h2);
    }
    return 0;
}
