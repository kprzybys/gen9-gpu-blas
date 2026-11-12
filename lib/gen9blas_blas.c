// libgen9blas_blas: pełny BLAS + CBLAS (te same 393 symbole co systemowy libblas.so.3 z OpenBLAS) z funkcjami S poziomów
// 2–3 (i CBLAS poziomu 1 dla pamięci USM) na GPU przez libgen9blas. Pozostałe funkcje (D, C, Z, rotg, pasmowe,
// spakowane, fortranowy poziom 1) skaczą wprost do BLAS na CPU (gen9blas_blas_forward.c, generowane przez gen_forward.py).
//
// Użycie bez przebudowy programów: LD_PRELOAD=.../libgen9blas_blas.so program  (albo jako alternatywa libblas.so.3).
// Zmienne: GEN9BLAS_BLAS=auto (domyślnie) | gpu (każde wywołanie S na GPU) | cpu (wszystko na CPU);
//          GEN9BLAS_BLAS_CPU=ścieżka BLAS na CPU (domyślnie libopenblas.so.0); GEN9BLAS_DIR=katalog kerneli (domyślnie
//          ../bench względem tej biblioteki); GEN9BLAS_BLAS_INFO=1: na stderr każde wywołanie S i gdzie poszło.
// Zasady „auto” (kryterium: czas CPU procesu na wywołanie; pomiar na zasilaniu sieciowym results/auto-thresholds-2026-11-12.txt):
// GPU, gdy praca na CPU kosztowałaby więcej niż stały koszt wywołania GPU (~20–250 µs CPU). Poziom 3 od 4 MFLOP. Poziom 1
// (tylko CBLAS) od 2^20 elementów, poziom 2 od 2^22 (M·N, n = 2048); od 2^16, gdy OpenBLAS ma > 1 wątek, a funkcja jest
// przez niego dzielona na wątki (axpy, gemv, ger, symv, trmv, syr, syr2) -- wątki wirują i zużywają 6–8 × czas ściany.
// Pamięć USM i zwykła tak samo (zwykła bez kopii przez userptr, BEZNEO.md 6g). Na CPU zawsze: alpha = 0 (BLAS nie czyta wtedy A, B), puste wymiary, krok 0,
// błędne argumenty (raportuje je BLAS na CPU). Błąd na GPU: komunikat raz i to wywołanie na CPU.
// Układ kolumnowy = wierszowy dla macierzy transponowanej: zamiana wymiarów, stron (side), trójkątów (uplo)
// i transpozycji -- te same kernele.
#define _GNU_SOURCE
#include "gen9blas.h"
#include <ctype.h>
#include <dlfcn.h>
#include <libgen.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLAS_API __attribute__((visibility("default")))
enum { WIERSZ = 101, KOLUMNA = 102, BEZ = 111, TRANS = 112, TRANS_SPRZ = 113, GORNY = 121, DOLNY = 122,
       ZWYKLA = 131, JEDYNKI = 132, LEWA = 141, PRAWA = 142 };

// ---------------- BLAS na CPU ----------------
extern void *gen9blas_cel[];
extern const char *const gen9blas_cel_nazwy[];
extern const int gen9blas_nprzekaz;
static void *cpu;

static void brak_cpu(void) { fprintf(stderr, "gen9blas_blas: brak BLAS na CPU (GEN9BLAS_BLAS_CPU, libopenblas.so.0)\n"); abort(); }

BLAS_API void cblas_sgemm(int, int, int, int, int, int, float, const float *, int, const float *, int, float, float *, int);

static int (*watki_cpu)(void);   // openblas_get_num_threads (progi poziomów 1–2)
__attribute__((constructor)) static void start(void) {
    const char *kand[] = { getenv("GEN9BLAS_BLAS_CPU"), "libopenblas.so.0", "/usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3" };
    for (size_t i = 0; i < sizeof kand / sizeof kand[0] && !cpu; i++) {
        if (!kand[i]) continue;
        void *h = dlopen(kand[i], RTLD_NOW | RTLD_LOCAL);
        if (h && dlsym(h, "cblas_sgemm") && dlsym(h, "cblas_sgemm") != (void *)cblas_sgemm) cpu = h;   // nie siebie samą
        else if (h) dlclose(h);
    }
    if (cpu) watki_cpu = (int (*)(void))dlsym(cpu, "openblas_get_num_threads");
    for (int i = 0; i < gen9blas_nprzekaz; i++) {
        gen9blas_cel[i] = cpu ? dlsym(cpu, gen9blas_cel_nazwy[i]) : NULL;
        if (!gen9blas_cel[i]) gen9blas_cel[i] = (void *)brak_cpu;
    }
}
static void *cpu_fn(const char *n) { void *f = cpu ? dlsym(cpu, n) : NULL; if (!f) brak_cpu(); return f; }
#define CPU(nazwa, ...) ({ static __typeof__(__VA_ARGS__) f_; if (!f_) f_ = (__typeof__(__VA_ARGS__))cpu_fn(#nazwa); f_; })

// ---------------- GPU: inicjalizacja, zasady, blokada ----------------
static int tryb;   // 0 cpu, 1 auto, 2 gpu
static int info;
static pthread_once_t raz = PTHREAD_ONCE_INIT;
static pthread_mutex_t blokada = PTHREAD_MUTEX_INITIALIZER;

static void init_gpu(void) {
    const char *e = getenv("GEN9BLAS_BLAS");
    tryb = e && !strcmp(e, "cpu") ? 0 : e && !strcmp(e, "gpu") ? 2 : 1;
    info = getenv("GEN9BLAS_BLAS_INFO") != NULL;
    if (!tryb) return;
    const char *k = getenv("GEN9BLAS_DIR"); char kat[PATH_MAX + 16], sciezka[PATH_MAX];
    Dl_info d;
    if (!k && dladdr((void *)init_gpu, &d) && d.dli_fname && realpath(d.dli_fname, sciezka)) {
        snprintf(kat, sizeof kat, "%s/../bench", dirname(sciezka)); k = kat;
    }
    if (!k || gen9blas_init(k)) {
        fprintf(stderr, "gen9blas_blas: GPU niedostępne (%s) -- wszystko na CPU\n", k ? gen9blas_error() : "brak katalogu kerneli");
        tryb = 0;
    }
}
static int pocz(void) { pthread_once(&raz, init_gpu); return tryb; }
// Czy na GPU: duze -- według zasad „auto”.
static int na_gpu(int duze) { return pocz() == 2 || (tryb == 1 && duze); }
// Progi poziomów 1–2 (opis na początku pliku). Liczba wątków czytana przy każdym wywołaniu (program może ją zmienić
// przez openblas_set_num_threads); BLAS na CPU bez tej funkcji: traktowany jak jednowątkowy.
#define PROG_MALY (1 << 16)
#define PROG_L1 (1 << 20)
#define PROG_L2 (1 << 22)
static int duze_l12(double el, double prog, int dzielona) {
    return el >= PROG_MALY && (el >= prog || (dzielona && watki_cpu && watki_cpu() > 1));
}
static void zglos(const char *f, const char *gdzie) { if (info) fprintf(stderr, "gen9blas_blas: %s -> %s\n", f, gdzie); }
// Wywołanie libgen9blas pod blokadą; 0 = zrobione na GPU, 1 = do zrobienia na CPU (błąd: komunikat raz).
static int gpu(const char *f, int w) {
    if (w) {
        static int raz_b;
        if (!raz_b++) fprintf(stderr, "gen9blas_blas: błąd GPU w %s (%s) -- to wywołanie na CPU\n", f, gen9blas_error());
        return 1;
    }
    zglos(f, "GPU");
    return 0;
}
#define GPU(f, wywolanie) ({ pthread_mutex_lock(&blokada); int w_ = (wywolanie); pthread_mutex_unlock(&blokada); gpu(f, w_); })

// ==== POCZĄTEK TYPU: od tego miejsca do końca pliku kod funkcji S; lib/gen_blas_d.py robi z niego wersję D ====
// (float -> double, nazwy s -> d, GEN9BLAS_S -> GEN9BLAS_D, funkcje pomocnicze i typy z przyrostkiem _d), dołączaną na końcu
// (#include "gen9blas_blas_d.inc"). Bloki „tylko S” (funkcje o typach mieszanych) są pomijane.
#define PROG_L3 4e6

// ---------------- rdzeń w układzie wierszowym: 0 = zrobione na GPU, 1 = na CPU ----------------
// Wskaźniki wektorów jak w CBLAS/BLAS i gen9blas: przy inc < 0 wskazują najniższy adres (tam leży element n-1).
static int r_gemm(int ta, int tb, int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb,
                  float beta, float *C, int ldc) {
    if (M <= 0 || N <= 0 || K <= 0 || alpha == 0.0f || lda < (ta ? M : K) || ldb < (tb ? K : N) || ldc < N) return 1;
    if (!na_gpu(2.0 * M * N * K >= PROG_L3)) return 1;
    return GPU("sgemm", gen9blas_gemm(GEN9BLAS_S, ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc));
}
static int r_symm(int lewa, int gorny, int M, int N, float alpha, const float *A, int lda, const float *B, int ldb,
                  float beta, float *C, int ldc) {
    int n = lewa ? M : N;
    if (M <= 0 || N <= 0 || alpha == 0.0f || lda < n || ldb < N || ldc < N) return 1;
    if (!na_gpu(2.0 * M * N * n >= PROG_L3)) return 1;
    return GPU("ssymm", gen9blas_symm(GEN9BLAS_S, lewa ? GEN9BLAS_LEFT : GEN9BLAS_RIGHT, gorny ? GEN9BLAS_UPPER : GEN9BLAS_LOWER, M, N, alpha, A, lda, B, ldb, beta, C, ldc));
}
static int r_syrk(int gorny, int tr, int N, int K, float alpha, const float *A, int lda, float beta, float *C, int ldc) {
    if (N <= 0 || K <= 0 || alpha == 0.0f || lda < (tr ? N : K) || ldc < N) return 1;
    if (!na_gpu(1.0 * N * N * K >= PROG_L3)) return 1;
    return GPU("ssyrk", gen9blas_syrk(GEN9BLAS_S, gorny ? GEN9BLAS_UPPER : GEN9BLAS_LOWER, tr, N, K, alpha, A, lda, beta, C, ldc));
}
static int r_syr2k(int gorny, int tr, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb,
                   float beta, float *C, int ldc) {
    if (N <= 0 || K <= 0 || alpha == 0.0f || lda < (tr ? N : K) || ldb < (tr ? N : K) || ldc < N) return 1;
    if (!na_gpu(2.0 * N * N * K >= PROG_L3)) return 1;
    return GPU("ssyr2k", gen9blas_syr2k(GEN9BLAS_S, gorny ? GEN9BLAS_UPPER : GEN9BLAS_LOWER, tr, N, K, alpha, A, lda, B, ldb, beta, C, ldc));
}
static int r_trxm(int rozw, int lewa, int gorny, int tr, int jedn, int M, int N, float alpha, const float *A, int lda,
                  float *B, int ldb) {
    int n = lewa ? M : N;
    if (M <= 0 || N <= 0 || alpha == 0.0f || lda < n || ldb < N) return 1;
    if (!na_gpu(1.0 * M * N * n >= PROG_L3)) return 1;
    int s = lewa ? GEN9BLAS_LEFT : GEN9BLAS_RIGHT, u = gorny ? GEN9BLAS_UPPER : GEN9BLAS_LOWER;
    return rozw ? GPU("strsm", gen9blas_trsm(GEN9BLAS_S, s, u, tr, jedn, M, N, alpha, A, lda, B, ldb))
                : GPU("strmm", gen9blas_trmm(GEN9BLAS_S, s, u, tr, jedn, M, N, alpha, A, lda, B, ldb));
}
static int duze_l2(int M, int N, int dzielona) { return duze_l12((double)M * N, PROG_L2, dzielona); }
static int r_gemv(int tr, int M, int N, float alpha, const float *A, int lda, const float *x, int incx, float beta,
                  float *y, int incy) {
    if (M <= 0 || N <= 0 || alpha == 0.0f || lda < N || !incx || !incy) return 1;
    if (!na_gpu(duze_l2(M, N, 1))) return 1;
    return GPU("sgemv", gen9blas_gemv(GEN9BLAS_S, tr, M, N, alpha, A, lda, x, incx, beta, y, incy));
}
static int r_ger(int M, int N, float alpha, const float *x, int incx, const float *y, int incy, float *A, int lda) {
    if (M <= 0 || N <= 0 || alpha == 0.0f || lda < N || !incx || !incy) return 1;
    if (!na_gpu(duze_l2(M, N, 1))) return 1;
    return GPU("sger", gen9blas_ger(GEN9BLAS_S, M, N, alpha, x, incx, y, incy, A, lda));
}
static int r_symv(int gorny, int n, float alpha, const float *A, int lda, const float *x, int incx, float beta, float *y, int incy) {
    if (n <= 0 || alpha == 0.0f || lda < n || !incx || !incy) return 1;
    if (!na_gpu(duze_l2(n, n, 1))) return 1;
    return GPU("ssymv", gen9blas_symv(GEN9BLAS_S, gorny ? GEN9BLAS_UPPER : GEN9BLAS_LOWER, n, alpha, A, lda, x, incx, beta, y, incy));
}
static int r_trxv(int rozw, int gorny, int tr, int jedn, int n, const float *A, int lda, float *x, int incx) {
    if (n <= 0 || lda < n || !incx) return 1;
    if (!na_gpu(duze_l2(n, n, !rozw))) return 1;
    int u = gorny ? GEN9BLAS_UPPER : GEN9BLAS_LOWER;
    return rozw ? GPU("strsv", gen9blas_trsv(GEN9BLAS_S, u, tr, jedn, n, A, lda, x, incx))
                : GPU("strmv", gen9blas_trmv(GEN9BLAS_S, u, tr, jedn, n, A, lda, x, incx));
}
static int r_syr2(int gorny, int n, float alpha, const float *x, int incx, const float *y, int incy, float *A, int lda) {
    if (n <= 0 || alpha == 0.0f || lda < n || !incx || (y && !incy)) return 1;
    if (!na_gpu(duze_l2(n, n, 1))) return 1;
    int u = gorny ? GEN9BLAS_UPPER : GEN9BLAS_LOWER;
    return y ? GPU("ssyr2", gen9blas_syr2(GEN9BLAS_S, u, n, alpha, x, incx, y, incy, A, lda))
             : GPU("ssyr", gen9blas_syr(GEN9BLAS_S, u, n, alpha, x, incx, A, lda));
}

// ---------------- CBLAS poziomy 2–3 ----------------
#define PRZEC(x) ((x) == BEZ ? 0 : 1)   // trans: TRANS i TRANS_SPRZ to samo dla liczb rzeczywistych
#define ZLE_T(x) ((x) != BEZ && (x) != TRANS && (x) != TRANS_SPRZ)
#define ZLE_U(x) ((x) != GORNY && (x) != DOLNY)
#define ZLE_S(x) ((x) != LEWA && (x) != PRAWA)
#define ZLE_D(x) ((x) != ZWYKLA && (x) != JEDYNKI)
#define ZLE_O(x) ((x) != WIERSZ && (x) != KOLUMNA)

BLAS_API void cblas_sgemm(int o, int ta, int tb, int M, int N, int K, float alpha, const float *A, int lda,
                          const float *B, int ldb, float beta, float *C, int ldc) {
    int w = 1;
    if (!ZLE_O(o) && !ZLE_T(ta) && !ZLE_T(tb))
        w = o == WIERSZ ? r_gemm(PRZEC(ta), PRZEC(tb), M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
                        : r_gemm(PRZEC(tb), PRZEC(ta), N, M, K, alpha, B, ldb, A, lda, beta, C, ldc);
    if (w) { zglos("sgemm", "CPU"); CPU(cblas_sgemm, void (*)(int, int, int, int, int, int, float, const float *, int, const float *, int, float, float *, int))(o, ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc); }
}
BLAS_API void cblas_ssymm(int o, int s, int u, int M, int N, float alpha, const float *A, int lda, const float *B, int ldb,
                          float beta, float *C, int ldc) {
    int w = 1;
    if (!ZLE_O(o) && !ZLE_S(s) && !ZLE_U(u))
        w = o == WIERSZ ? r_symm(s == LEWA, u == GORNY, M, N, alpha, A, lda, B, ldb, beta, C, ldc)
                        : r_symm(s != LEWA, u != GORNY, N, M, alpha, A, lda, B, ldb, beta, C, ldc);
    if (w) { zglos("ssymm", "CPU"); CPU(cblas_ssymm, void (*)(int, int, int, int, int, float, const float *, int, const float *, int, float, float *, int))(o, s, u, M, N, alpha, A, lda, B, ldb, beta, C, ldc); }
}
BLAS_API void cblas_ssyrk(int o, int u, int t, int N, int K, float alpha, const float *A, int lda, float beta, float *C, int ldc) {
    int w = 1;
    if (!ZLE_O(o) && !ZLE_U(u) && !ZLE_T(t))
        w = o == WIERSZ ? r_syrk(u == GORNY, PRZEC(t), N, K, alpha, A, lda, beta, C, ldc)
                        : r_syrk(u != GORNY, !PRZEC(t), N, K, alpha, A, lda, beta, C, ldc);
    if (w) { zglos("ssyrk", "CPU"); CPU(cblas_ssyrk, void (*)(int, int, int, int, int, float, const float *, int, float, float *, int))(o, u, t, N, K, alpha, A, lda, beta, C, ldc); }
}
BLAS_API void cblas_ssyr2k(int o, int u, int t, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb,
                           float beta, float *C, int ldc) {
    int w = 1;
    if (!ZLE_O(o) && !ZLE_U(u) && !ZLE_T(t))
        w = o == WIERSZ ? r_syr2k(u == GORNY, PRZEC(t), N, K, alpha, A, lda, B, ldb, beta, C, ldc)
                        : r_syr2k(u != GORNY, !PRZEC(t), N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    if (w) { zglos("ssyr2k", "CPU"); CPU(cblas_ssyr2k, void (*)(int, int, int, int, int, float, const float *, int, const float *, int, float, float *, int))(o, u, t, N, K, alpha, A, lda, B, ldb, beta, C, ldc); }
}
static int c_trxm(int rozw, int o, int s, int u, int t, int d, int M, int N, float alpha, const float *A, int lda, float *B, int ldb) {
    if (ZLE_O(o) || ZLE_S(s) || ZLE_U(u) || ZLE_T(t) || ZLE_D(d)) return 1;
    return o == WIERSZ ? r_trxm(rozw, s == LEWA, u == GORNY, PRZEC(t), d == JEDYNKI, M, N, alpha, A, lda, B, ldb)
                       : r_trxm(rozw, s != LEWA, u != GORNY, PRZEC(t), d == JEDYNKI, N, M, alpha, A, lda, B, ldb);
}
typedef void (*trxm_cblas_t)(int, int, int, int, int, int, int, float, const float *, int, float *, int);
BLAS_API void cblas_strmm(int o, int s, int u, int t, int d, int M, int N, float alpha, const float *A, int lda, float *B, int ldb) {
    if (c_trxm(0, o, s, u, t, d, M, N, alpha, A, lda, B, ldb)) { zglos("strmm", "CPU"); CPU(cblas_strmm, trxm_cblas_t)(o, s, u, t, d, M, N, alpha, A, lda, B, ldb); }
}
BLAS_API void cblas_strsm(int o, int s, int u, int t, int d, int M, int N, float alpha, const float *A, int lda, float *B, int ldb) {
    if (c_trxm(1, o, s, u, t, d, M, N, alpha, A, lda, B, ldb)) { zglos("strsm", "CPU"); CPU(cblas_strsm, trxm_cblas_t)(o, s, u, t, d, M, N, alpha, A, lda, B, ldb); }
}

BLAS_API void cblas_sgemv(int o, int t, int M, int N, float alpha, const float *A, int lda, const float *x, int incx,
                          float beta, float *y, int incy) {
    int w = 1;
    if (!ZLE_O(o) && !ZLE_T(t)) {
        int tr = o == WIERSZ ? PRZEC(t) : !PRZEC(t), Mr = o == WIERSZ ? M : N, Nr = o == WIERSZ ? N : M;
        w = r_gemv(tr, Mr, Nr, alpha, A, lda, x, incx, beta, y, incy);
    }
    if (w) { zglos("sgemv", "CPU"); CPU(cblas_sgemv, void (*)(int, int, int, int, float, const float *, int, const float *, int, float, float *, int))(o, t, M, N, alpha, A, lda, x, incx, beta, y, incy); }
}
BLAS_API void cblas_sger(int o, int M, int N, float alpha, const float *x, int incx, const float *y, int incy, float *A, int lda) {
    int w = 1;
    if (!ZLE_O(o))
        w = o == WIERSZ ? r_ger(M, N, alpha, x, incx, y, incy, A, lda) : r_ger(N, M, alpha, y, incy, x, incx, A, lda);
    if (w) { zglos("sger", "CPU"); CPU(cblas_sger, void (*)(int, int, int, float, const float *, int, const float *, int, float *, int))(o, M, N, alpha, x, incx, y, incy, A, lda); }
}
BLAS_API void cblas_ssymv(int o, int u, int N, float alpha, const float *A, int lda, const float *x, int incx, float beta,
                          float *y, int incy) {
    int w = 1;
    if (!ZLE_O(o) && !ZLE_U(u)) w = r_symv((u == GORNY) == (o == WIERSZ), N, alpha, A, lda, x, incx, beta, y, incy);
    if (w) { zglos("ssymv", "CPU"); CPU(cblas_ssymv, void (*)(int, int, int, float, const float *, int, const float *, int, float, float *, int))(o, u, N, alpha, A, lda, x, incx, beta, y, incy); }
}
static int c_trxv(int rozw, int o, int u, int t, int d, int N, const float *A, int lda, float *x, int incx) {
    if (ZLE_O(o) || ZLE_U(u) || ZLE_T(t) || ZLE_D(d)) return 1;
    int w = o == WIERSZ;
    return r_trxv(rozw, (u == GORNY) == w, w ? PRZEC(t) : !PRZEC(t), d == JEDYNKI, N, A, lda, x, incx);
}
typedef void (*trxv_cblas_t)(int, int, int, int, int, const float *, int, float *, int);
BLAS_API void cblas_strmv(int o, int u, int t, int d, int N, const float *A, int lda, float *x, int incx) {
    if (c_trxv(0, o, u, t, d, N, A, lda, x, incx)) { zglos("strmv", "CPU"); CPU(cblas_strmv, trxv_cblas_t)(o, u, t, d, N, A, lda, x, incx); }
}
BLAS_API void cblas_strsv(int o, int u, int t, int d, int N, const float *A, int lda, float *x, int incx) {
    if (c_trxv(1, o, u, t, d, N, A, lda, x, incx)) { zglos("strsv", "CPU"); CPU(cblas_strsv, trxv_cblas_t)(o, u, t, d, N, A, lda, x, incx); }
}
BLAS_API void cblas_ssyr(int o, int u, int N, float alpha, const float *x, int incx, float *A, int lda) {
    int w = 1;
    if (!ZLE_O(o) && !ZLE_U(u)) w = r_syr2((u == GORNY) == (o == WIERSZ), N, alpha, x, incx, NULL, 0, A, lda);
    if (w) { zglos("ssyr", "CPU"); CPU(cblas_ssyr, void (*)(int, int, int, float, const float *, int, float *, int))(o, u, N, alpha, x, incx, A, lda); }
}
BLAS_API void cblas_ssyr2(int o, int u, int N, float alpha, const float *x, int incx, const float *y, int incy, float *A, int lda) {
    int w = 1;
    if (!ZLE_O(o) && !ZLE_U(u)) w = r_syr2((u == GORNY) == (o == WIERSZ), N, alpha, x, incx, y, incy, A, lda);
    if (w) { zglos("ssyr2", "CPU"); CPU(cblas_ssyr2, void (*)(int, int, int, float, const float *, int, const float *, int, float *, int))(o, u, N, alpha, x, incx, y, incy, A, lda); }
}

// ---------------- CBLAS poziom 1: GPU tylko dla wektorów USM ----------------
static int l1_gpu(int n, int incx, int incy, int dzielona) {   // incy = 1 dla funkcji jednego wektora
    if (n <= 0 || incx <= 0 || incy == 0) return 0;   // krok <= 0: semantyka BLAS różni się między funkcjami
    return na_gpu(duze_l12(n, PROG_L1, dzielona));
}
BLAS_API void cblas_saxpy(int n, float a, const float *x, int incx, float *y, int incy) {
    if (a != 0.0f && l1_gpu(n, incx, incy, 1) && !GPU("saxpy", gen9blas_axpy(GEN9BLAS_S, n, a, x, incx, y, incy))) return;
    zglos("saxpy", "CPU"); CPU(cblas_saxpy, void (*)(int, float, const float *, int, float *, int))(n, a, x, incx, y, incy);
}
BLAS_API void cblas_sscal(int n, float a, float *x, int incx) {
    if (l1_gpu(n, incx, 1, 0) && !GPU("sscal", gen9blas_scal(GEN9BLAS_S, n, a, x, incx))) return;
    zglos("sscal", "CPU"); CPU(cblas_sscal, void (*)(int, float, float *, int))(n, a, x, incx);
}
BLAS_API void cblas_scopy(int n, const float *x, int incx, float *y, int incy) {
    if (l1_gpu(n, incx, incy, 0) && !GPU("scopy", gen9blas_copy(GEN9BLAS_S, n, x, incx, y, incy))) return;
    zglos("scopy", "CPU"); CPU(cblas_scopy, void (*)(int, const float *, int, float *, int))(n, x, incx, y, incy);
}
BLAS_API void cblas_sswap(int n, float *x, int incx, float *y, int incy) {
    if (l1_gpu(n, incx, incy, 0) && !GPU("sswap", gen9blas_swap(GEN9BLAS_S, n, x, incx, y, incy))) return;
    zglos("sswap", "CPU"); CPU(cblas_sswap, void (*)(int, float *, int, float *, int))(n, x, incx, y, incy);
}
BLAS_API void cblas_srot(int n, float *x, int incx, float *y, int incy, float c, float s) {
    if (l1_gpu(n, incx, incy, 0) && !GPU("srot", gen9blas_rot(GEN9BLAS_S, n, x, incx, y, incy, c, s))) return;
    zglos("srot", "CPU"); CPU(cblas_srot, void (*)(int, float *, int, float *, int, float, float))(n, x, incx, y, incy, c, s);
}
BLAS_API void cblas_srotm(int n, float *x, int incx, float *y, int incy, const float *P) {
    const double Pd[5] = { P[0], P[1], P[2], P[3], P[4] };
    if (l1_gpu(n, incx, incy, 0) && !GPU("srotm", gen9blas_rotm(GEN9BLAS_S, n, x, incx, y, incy, Pd))) return;
    zglos("srotm", "CPU"); CPU(cblas_srotm, void (*)(int, float *, int, float *, int, const float *))(n, x, incx, y, incy, P);
}
static double l1_dot(int n, const float *x, int incx, const float *y, int incy, int *zrobione) {
    double w = 0;
    *zrobione = l1_gpu(n, incx, incy, 0) && !GPU("sdot", gen9blas_dot(GEN9BLAS_S, n, x, incx, y, incy, &w));
    return w;
}
BLAS_API float cblas_sdot(int n, const float *x, int incx, const float *y, int incy) {
    int z; double w = l1_dot(n, x, incx, y, incy, &z);
    if (z) return (float)w;
    zglos("sdot", "CPU"); return CPU(cblas_sdot, float (*)(int, const float *, int, const float *, int))(n, x, incx, y, incy);
}
// ---- tylko S: początek
BLAS_API float cblas_sdsdot(int n, float a, const float *x, int incx, const float *y, int incy) {
    int z; double w = l1_dot(n, x, incx, y, incy, &z);
    if (z) return (float)(a + w);
    zglos("sdsdot", "CPU"); return CPU(cblas_sdsdot, float (*)(int, float, const float *, int, const float *, int))(n, a, x, incx, y, incy);
}
BLAS_API double cblas_dsdot(int n, const float *x, int incx, const float *y, int incy) {
    int z; double w = l1_dot(n, x, incx, y, incy, &z);
    if (z) return w;
    zglos("dsdot", "CPU"); return CPU(cblas_dsdot, double (*)(int, const float *, int, const float *, int))(n, x, incx, y, incy);
}
// ---- tylko S: koniec
BLAS_API float cblas_sasum(int n, const float *x, int incx) {
    double w = 0;
    if (l1_gpu(n, incx, 1, 0) && !GPU("sasum", gen9blas_asum(GEN9BLAS_S, n, x, incx, &w))) return (float)w;
    zglos("sasum", "CPU"); return CPU(cblas_sasum, float (*)(int, const float *, int))(n, x, incx);
}
BLAS_API float cblas_snrm2(int n, const float *x, int incx) {
    double w = 0;
    if (l1_gpu(n, incx, 1, 0) && !GPU("snrm2", gen9blas_nrm2(GEN9BLAS_S, n, x, incx, &w))) return (float)w;
    zglos("snrm2", "CPU"); return CPU(cblas_snrm2, float (*)(int, const float *, int))(n, x, incx);
}
BLAS_API size_t cblas_isamax(int n, const float *x, int incx) {
    long w = 0;
    if (l1_gpu(n, incx, 1, 0) && !GPU("isamax", gen9blas_iamax(GEN9BLAS_S, n, x, incx, &w))) return (size_t)w;
    zglos("isamax", "CPU"); return CPU(cblas_isamax, size_t (*)(int, const float *, int))(n, x, incx);
}

// ---------------- Fortran (układ kolumnowy; poziom 1 -- skoki do CPU) ----------------
// Znaki opcji bez rozróżniania wielkości liter; długości napisów (ukryte argumenty gfortran) są ignorowane.
static int zn(const char *c) { return toupper((unsigned char)*c); }
static int f_t(const char *c) { int z = zn(c); return z == 'N' ? 0 : z == 'T' || z == 'C' ? 1 : -1; }
static int f_u(const char *c) { int z = zn(c); return z == 'U' ? 1 : z == 'L' ? 0 : -1; }   // 1: górny
static int f_s(const char *c) { int z = zn(c); return z == 'L' ? 1 : z == 'R' ? 0 : -1; }   // 1: lewa
static int f_d(const char *c) { int z = zn(c); return z == 'U' ? 1 : z == 'N' ? 0 : -1; }   // 1: jedynki

typedef const char cc;
typedef const int ci;
typedef const float cf;
BLAS_API void sgemm_(cc *ta, cc *tb, ci *M, ci *N, ci *K, cf *alpha, cf *A, ci *lda, cf *B, ci *ldb, cf *beta, float *C, ci *ldc) {
    int a = f_t(ta), b = f_t(tb);
    if (a < 0 || b < 0 || r_gemm(b, a, *N, *M, *K, *alpha, B, *ldb, A, *lda, *beta, C, *ldc)) {
        zglos("sgemm_", "CPU"); CPU(sgemm_, void (*)(cc *, cc *, ci *, ci *, ci *, cf *, cf *, ci *, cf *, ci *, cf *, float *, ci *))(ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc); }
}
BLAS_API void ssymm_(cc *sd, cc *up, ci *M, ci *N, cf *alpha, cf *A, ci *lda, cf *B, ci *ldb, cf *beta, float *C, ci *ldc) {
    int s = f_s(sd), u = f_u(up);
    if (s < 0 || u < 0 || r_symm(!s, !u, *N, *M, *alpha, A, *lda, B, *ldb, *beta, C, *ldc)) {
        zglos("ssymm_", "CPU"); CPU(ssymm_, void (*)(cc *, cc *, ci *, ci *, cf *, cf *, ci *, cf *, ci *, cf *, float *, ci *))(sd, up, M, N, alpha, A, lda, B, ldb, beta, C, ldc); }
}
BLAS_API void ssyrk_(cc *up, cc *tr, ci *N, ci *K, cf *alpha, cf *A, ci *lda, cf *beta, float *C, ci *ldc) {
    int u = f_u(up), t = f_t(tr);
    if (u < 0 || t < 0 || r_syrk(!u, !t, *N, *K, *alpha, A, *lda, *beta, C, *ldc)) {
        zglos("ssyrk_", "CPU"); CPU(ssyrk_, void (*)(cc *, cc *, ci *, ci *, cf *, cf *, ci *, cf *, float *, ci *))(up, tr, N, K, alpha, A, lda, beta, C, ldc); }
}
BLAS_API void ssyr2k_(cc *up, cc *tr, ci *N, ci *K, cf *alpha, cf *A, ci *lda, cf *B, ci *ldb, cf *beta, float *C, ci *ldc) {
    int u = f_u(up), t = f_t(tr);
    if (u < 0 || t < 0 || r_syr2k(!u, !t, *N, *K, *alpha, A, *lda, B, *ldb, *beta, C, *ldc)) {
        zglos("ssyr2k_", "CPU"); CPU(ssyr2k_, void (*)(cc *, cc *, ci *, ci *, cf *, cf *, ci *, cf *, ci *, cf *, float *, ci *))(up, tr, N, K, alpha, A, lda, B, ldb, beta, C, ldc); }
}
typedef void (*trxm_f_t)(cc *, cc *, cc *, cc *, ci *, ci *, cf *, cf *, ci *, float *, ci *);
static int f_trxm(int rozw, cc *sd, cc *up, cc *tr, cc *dg, ci *M, ci *N, cf *alpha, cf *A, ci *lda, float *B, ci *ldb) {
    int s = f_s(sd), u = f_u(up), t = f_t(tr), d = f_d(dg);
    return s < 0 || u < 0 || t < 0 || d < 0 || r_trxm(rozw, !s, !u, t, d, *N, *M, *alpha, A, *lda, B, *ldb);
}
BLAS_API void strmm_(cc *sd, cc *up, cc *tr, cc *dg, ci *M, ci *N, cf *alpha, cf *A, ci *lda, float *B, ci *ldb) {
    if (f_trxm(0, sd, up, tr, dg, M, N, alpha, A, lda, B, ldb)) { zglos("strmm_", "CPU"); CPU(strmm_, trxm_f_t)(sd, up, tr, dg, M, N, alpha, A, lda, B, ldb); }
}
BLAS_API void strsm_(cc *sd, cc *up, cc *tr, cc *dg, ci *M, ci *N, cf *alpha, cf *A, ci *lda, float *B, ci *ldb) {
    if (f_trxm(1, sd, up, tr, dg, M, N, alpha, A, lda, B, ldb)) { zglos("strsm_", "CPU"); CPU(strsm_, trxm_f_t)(sd, up, tr, dg, M, N, alpha, A, lda, B, ldb); }
}
// Fortran: przy kroku ujemnym wektor zaczyna się od najniższego adresu -- jak w gen9blas.
BLAS_API void sgemv_(cc *tr, ci *M, ci *N, cf *alpha, cf *A, ci *lda, cf *x, ci *incx, cf *beta, float *y, ci *incy) {
    int t = f_t(tr);
    if (t < 0 || r_gemv(!t, *N, *M, *alpha, A, *lda, x, *incx, *beta, y, *incy)) {
        zglos("sgemv_", "CPU"); CPU(sgemv_, void (*)(cc *, ci *, ci *, cf *, cf *, ci *, cf *, ci *, cf *, float *, ci *))(tr, M, N, alpha, A, lda, x, incx, beta, y, incy); }
}
BLAS_API void sger_(ci *M, ci *N, cf *alpha, cf *x, ci *incx, cf *y, ci *incy, float *A, ci *lda) {
    if (r_ger(*N, *M, *alpha, y, *incy, x, *incx, A, *lda)) {
        zglos("sger_", "CPU"); CPU(sger_, void (*)(ci *, ci *, cf *, cf *, ci *, cf *, ci *, float *, ci *))(M, N, alpha, x, incx, y, incy, A, lda); }
}
BLAS_API void ssymv_(cc *up, ci *N, cf *alpha, cf *A, ci *lda, cf *x, ci *incx, cf *beta, float *y, ci *incy) {
    int u = f_u(up);
    if (u < 0 || r_symv(!u, *N, *alpha, A, *lda, x, *incx, *beta, y, *incy)) {
        zglos("ssymv_", "CPU"); CPU(ssymv_, void (*)(cc *, ci *, cf *, cf *, ci *, cf *, ci *, cf *, float *, ci *))(up, N, alpha, A, lda, x, incx, beta, y, incy); }
}
typedef void (*trxv_f_t)(cc *, cc *, cc *, ci *, cf *, ci *, float *, ci *);
static int f_trxv(int rozw, cc *up, cc *tr, cc *dg, ci *N, cf *A, ci *lda, float *x, ci *incx) {
    int u = f_u(up), t = f_t(tr), d = f_d(dg);
    return u < 0 || t < 0 || d < 0 || r_trxv(rozw, !u, !t, d, *N, A, *lda, x, *incx);
}
BLAS_API void strmv_(cc *up, cc *tr, cc *dg, ci *N, cf *A, ci *lda, float *x, ci *incx) {
    if (f_trxv(0, up, tr, dg, N, A, lda, x, incx)) { zglos("strmv_", "CPU"); CPU(strmv_, trxv_f_t)(up, tr, dg, N, A, lda, x, incx); }
}
BLAS_API void strsv_(cc *up, cc *tr, cc *dg, ci *N, cf *A, ci *lda, float *x, ci *incx) {
    if (f_trxv(1, up, tr, dg, N, A, lda, x, incx)) { zglos("strsv_", "CPU"); CPU(strsv_, trxv_f_t)(up, tr, dg, N, A, lda, x, incx); }
}
BLAS_API void ssyr_(cc *up, ci *N, cf *alpha, cf *x, ci *incx, float *A, ci *lda) {
    int u = f_u(up);
    if (u < 0 || r_syr2(!u, *N, *alpha, x, *incx, NULL, 0, A, *lda)) {
        zglos("ssyr_", "CPU"); CPU(ssyr_, void (*)(cc *, ci *, cf *, cf *, ci *, float *, ci *))(up, N, alpha, x, incx, A, lda); }
}
BLAS_API void ssyr2_(cc *up, ci *N, cf *alpha, cf *x, ci *incx, cf *y, ci *incy, float *A, ci *lda) {
    int u = f_u(up);
    if (u < 0 || r_syr2(!u, *N, *alpha, x, *incx, y, *incy, A, *lda)) {
        zglos("ssyr2_", "CPU"); CPU(ssyr2_, void (*)(cc *, ci *, cf *, cf *, ci *, cf *, ci *, float *, ci *))(up, N, alpha, x, incx, y, incy, A, lda); }
}

// ---------------- poziom 2: pasmowe i spakowane ----------------
// Układ kolumnowy = wierszowy dla Aᵀ: pasmo z zamienionymi KL ↔ KU, trójkąt (uplo) zamieniony, trans zamienione
// (trójkątne). Rozwiązania TBSV/TPSV w trybie auto zawsze na CPU: na GPU sekwencyjne (jedna grupa robocza),
// 6–40× wolniejsze i z większym łącznym czasem CPU (pomiar 2026-11-11, NOTATKI.md).
static int r_gbmv(int tr, int M, int N, int KL, int KU, float alpha, const float *A, int lda, const float *x, int incx,
                  float beta, float *y, int incy) {
    if (M <= 0 || N <= 0 || KL < 0 || KU < 0 || alpha == 0.0f || lda < KL + KU + 1 || !incx || !incy) return 1;
    if (!na_gpu(duze_l2(M, KL + KU + 1, 0))) return 1;
    return GPU("sgbmv", gen9blas_gbmv(GEN9BLAS_S, tr, M, N, KL, KU, alpha, A, lda, x, incx, beta, y, incy));
}
static int r_sbmv(int gorny, int N, int K, float alpha, const float *A, int lda, const float *x, int incx, float beta, float *y, int incy) {
    if (N <= 0 || K < 0 || alpha == 0.0f || lda < K + 1 || !incx || !incy) return 1;
    if (!na_gpu(duze_l2(N, 2 * K + 1, 0))) return 1;
    return GPU("ssbmv", gen9blas_sbmv(GEN9BLAS_S, gorny ? GEN9BLAS_UPPER : GEN9BLAS_LOWER, N, K, alpha, A, lda, x, incx, beta, y, incy));
}
static int r_tbxv(int rozw, int gorny, int tr, int jedn, int N, int K, const float *A, int lda, float *x, int incx) {
    if (N <= 0 || K < 0 || lda < K + 1 || !incx) return 1;
    if (rozw ? pocz() != 2 : !na_gpu(duze_l2(N, K + 1, 0))) return 1;
    int u = gorny ? GEN9BLAS_UPPER : GEN9BLAS_LOWER;
    return rozw ? GPU("stbsv", gen9blas_tbsv(GEN9BLAS_S, u, tr, jedn, N, K, A, lda, x, incx)) : GPU("stbmv", gen9blas_tbmv(GEN9BLAS_S, u, tr, jedn, N, K, A, lda, x, incx));
}
static int r_spmv(int gorny, int N, float alpha, const float *Ap, const float *x, int incx, float beta, float *y, int incy) {
    if (N <= 0 || alpha == 0.0f || !incx || !incy) return 1;
    if (!na_gpu(duze_l2(N, (N + 1) / 2, 0))) return 1;
    return GPU("sspmv", gen9blas_spmv(GEN9BLAS_S, gorny ? GEN9BLAS_UPPER : GEN9BLAS_LOWER, N, alpha, Ap, x, incx, beta, y, incy));
}
static int r_tpxv(int rozw, int gorny, int tr, int jedn, int N, const float *Ap, float *x, int incx) {
    if (N <= 0 || !incx) return 1;
    if (rozw ? pocz() != 2 : !na_gpu(duze_l2(N, (N + 1) / 2, 0))) return 1;
    int u = gorny ? GEN9BLAS_UPPER : GEN9BLAS_LOWER;
    return rozw ? GPU("stpsv", gen9blas_tpsv(GEN9BLAS_S, u, tr, jedn, N, Ap, x, incx)) : GPU("stpmv", gen9blas_tpmv(GEN9BLAS_S, u, tr, jedn, N, Ap, x, incx));
}
static int r_spr2(int gorny, int N, float alpha, const float *x, int incx, const float *y, int incy, float *Ap) {
    if (N <= 0 || alpha == 0.0f || !incx || (y && !incy)) return 1;
    if (!na_gpu(duze_l2(N, (N + 1) / 2, 0))) return 1;
    int u = gorny ? GEN9BLAS_UPPER : GEN9BLAS_LOWER;
    return y ? GPU("sspr2", gen9blas_spr2(GEN9BLAS_S, u, N, alpha, x, incx, y, incy, Ap)) : GPU("sspr", gen9blas_spr(GEN9BLAS_S, u, N, alpha, x, incx, Ap));
}

BLAS_API void cblas_sgbmv(int o, int t, int M, int N, int KL, int KU, float alpha, const float *A, int lda, const float *x, int incx,
                          float beta, float *y, int incy) {
    int w = 1;
    if (!ZLE_O(o) && !ZLE_T(t))
        w = o == WIERSZ ? r_gbmv(PRZEC(t), M, N, KL, KU, alpha, A, lda, x, incx, beta, y, incy)
                        : r_gbmv(!PRZEC(t), N, M, KU, KL, alpha, A, lda, x, incx, beta, y, incy);
    if (w) { zglos("sgbmv", "CPU"); CPU(cblas_sgbmv, void (*)(int, int, int, int, int, int, float, const float *, int, const float *, int, float, float *, int))(o, t, M, N, KL, KU, alpha, A, lda, x, incx, beta, y, incy); }
}
BLAS_API void cblas_ssbmv(int o, int u, int N, int K, float alpha, const float *A, int lda, const float *x, int incx, float beta, float *y, int incy) {
    int w = 1;
    if (!ZLE_O(o) && !ZLE_U(u)) w = r_sbmv((u == GORNY) == (o == WIERSZ), N, K, alpha, A, lda, x, incx, beta, y, incy);
    if (w) { zglos("ssbmv", "CPU"); CPU(cblas_ssbmv, void (*)(int, int, int, int, float, const float *, int, const float *, int, float, float *, int))(o, u, N, K, alpha, A, lda, x, incx, beta, y, incy); }
}
static int c_tbxv(int rozw, int o, int u, int t, int d, int N, int K, const float *A, int lda, float *x, int incx) {
    if (ZLE_O(o) || ZLE_U(u) || ZLE_T(t) || ZLE_D(d)) return 1;
    int w = o == WIERSZ;
    return r_tbxv(rozw, (u == GORNY) == w, w ? PRZEC(t) : !PRZEC(t), d == JEDYNKI, N, K, A, lda, x, incx);
}
typedef void (*tbxv_cblas_t)(int, int, int, int, int, int, const float *, int, float *, int);
BLAS_API void cblas_stbmv(int o, int u, int t, int d, int N, int K, const float *A, int lda, float *x, int incx) {
    if (c_tbxv(0, o, u, t, d, N, K, A, lda, x, incx)) { zglos("stbmv", "CPU"); CPU(cblas_stbmv, tbxv_cblas_t)(o, u, t, d, N, K, A, lda, x, incx); }
}
BLAS_API void cblas_stbsv(int o, int u, int t, int d, int N, int K, const float *A, int lda, float *x, int incx) {
    if (c_tbxv(1, o, u, t, d, N, K, A, lda, x, incx)) { zglos("stbsv", "CPU"); CPU(cblas_stbsv, tbxv_cblas_t)(o, u, t, d, N, K, A, lda, x, incx); }
}
BLAS_API void cblas_sspmv(int o, int u, int N, float alpha, const float *Ap, const float *x, int incx, float beta, float *y, int incy) {
    int w = 1;
    if (!ZLE_O(o) && !ZLE_U(u)) w = r_spmv((u == GORNY) == (o == WIERSZ), N, alpha, Ap, x, incx, beta, y, incy);
    if (w) { zglos("sspmv", "CPU"); CPU(cblas_sspmv, void (*)(int, int, int, float, const float *, const float *, int, float, float *, int))(o, u, N, alpha, Ap, x, incx, beta, y, incy); }
}
static int c_tpxv(int rozw, int o, int u, int t, int d, int N, const float *Ap, float *x, int incx) {
    if (ZLE_O(o) || ZLE_U(u) || ZLE_T(t) || ZLE_D(d)) return 1;
    int w = o == WIERSZ;
    return r_tpxv(rozw, (u == GORNY) == w, w ? PRZEC(t) : !PRZEC(t), d == JEDYNKI, N, Ap, x, incx);
}
typedef void (*tpxv_cblas_t)(int, int, int, int, int, const float *, float *, int);
BLAS_API void cblas_stpmv(int o, int u, int t, int d, int N, const float *Ap, float *x, int incx) {
    if (c_tpxv(0, o, u, t, d, N, Ap, x, incx)) { zglos("stpmv", "CPU"); CPU(cblas_stpmv, tpxv_cblas_t)(o, u, t, d, N, Ap, x, incx); }
}
BLAS_API void cblas_stpsv(int o, int u, int t, int d, int N, const float *Ap, float *x, int incx) {
    if (c_tpxv(1, o, u, t, d, N, Ap, x, incx)) { zglos("stpsv", "CPU"); CPU(cblas_stpsv, tpxv_cblas_t)(o, u, t, d, N, Ap, x, incx); }
}
BLAS_API void cblas_sspr(int o, int u, int N, float alpha, const float *x, int incx, float *Ap) {
    int w = 1;
    if (!ZLE_O(o) && !ZLE_U(u)) w = r_spr2((u == GORNY) == (o == WIERSZ), N, alpha, x, incx, NULL, 0, Ap);
    if (w) { zglos("sspr", "CPU"); CPU(cblas_sspr, void (*)(int, int, int, float, const float *, int, float *))(o, u, N, alpha, x, incx, Ap); }
}
BLAS_API void cblas_sspr2(int o, int u, int N, float alpha, const float *x, int incx, const float *y, int incy, float *Ap) {
    int w = 1;
    if (!ZLE_O(o) && !ZLE_U(u)) w = r_spr2((u == GORNY) == (o == WIERSZ), N, alpha, x, incx, y, incy, Ap);
    if (w) { zglos("sspr2", "CPU"); CPU(cblas_sspr2, void (*)(int, int, int, float, const float *, int, const float *, int, float *))(o, u, N, alpha, x, incx, y, incy, Ap); }
}

BLAS_API void sgbmv_(cc *tr, ci *M, ci *N, ci *KL, ci *KU, cf *alpha, cf *A, ci *lda, cf *x, ci *incx, cf *beta, float *y, ci *incy) {
    int t = f_t(tr);
    if (t < 0 || r_gbmv(!t, *N, *M, *KU, *KL, *alpha, A, *lda, x, *incx, *beta, y, *incy)) {
        zglos("sgbmv_", "CPU"); CPU(sgbmv_, void (*)(cc *, ci *, ci *, ci *, ci *, cf *, cf *, ci *, cf *, ci *, cf *, float *, ci *))(tr, M, N, KL, KU, alpha, A, lda, x, incx, beta, y, incy); }
}
BLAS_API void ssbmv_(cc *up, ci *N, ci *K, cf *alpha, cf *A, ci *lda, cf *x, ci *incx, cf *beta, float *y, ci *incy) {
    int u = f_u(up);
    if (u < 0 || r_sbmv(!u, *N, *K, *alpha, A, *lda, x, *incx, *beta, y, *incy)) {
        zglos("ssbmv_", "CPU"); CPU(ssbmv_, void (*)(cc *, ci *, ci *, cf *, cf *, ci *, cf *, ci *, cf *, float *, ci *))(up, N, K, alpha, A, lda, x, incx, beta, y, incy); }
}
typedef void (*tbxv_f_t)(cc *, cc *, cc *, ci *, ci *, cf *, ci *, float *, ci *);
static int f_tbxv(int rozw, cc *up, cc *tr, cc *dg, ci *N, ci *K, cf *A, ci *lda, float *x, ci *incx) {
    int u = f_u(up), t = f_t(tr), d = f_d(dg);
    return u < 0 || t < 0 || d < 0 || r_tbxv(rozw, !u, !t, d, *N, *K, A, *lda, x, *incx);
}
BLAS_API void stbmv_(cc *up, cc *tr, cc *dg, ci *N, ci *K, cf *A, ci *lda, float *x, ci *incx) {
    if (f_tbxv(0, up, tr, dg, N, K, A, lda, x, incx)) { zglos("stbmv_", "CPU"); CPU(stbmv_, tbxv_f_t)(up, tr, dg, N, K, A, lda, x, incx); }
}
BLAS_API void stbsv_(cc *up, cc *tr, cc *dg, ci *N, ci *K, cf *A, ci *lda, float *x, ci *incx) {
    if (f_tbxv(1, up, tr, dg, N, K, A, lda, x, incx)) { zglos("stbsv_", "CPU"); CPU(stbsv_, tbxv_f_t)(up, tr, dg, N, K, A, lda, x, incx); }
}
BLAS_API void sspmv_(cc *up, ci *N, cf *alpha, cf *Ap, cf *x, ci *incx, cf *beta, float *y, ci *incy) {
    int u = f_u(up);
    if (u < 0 || r_spmv(!u, *N, *alpha, Ap, x, *incx, *beta, y, *incy)) {
        zglos("sspmv_", "CPU"); CPU(sspmv_, void (*)(cc *, ci *, cf *, cf *, cf *, ci *, cf *, float *, ci *))(up, N, alpha, Ap, x, incx, beta, y, incy); }
}
typedef void (*tpxv_f_t)(cc *, cc *, cc *, ci *, cf *, float *, ci *);
static int f_tpxv(int rozw, cc *up, cc *tr, cc *dg, ci *N, cf *Ap, float *x, ci *incx) {
    int u = f_u(up), t = f_t(tr), d = f_d(dg);
    return u < 0 || t < 0 || d < 0 || r_tpxv(rozw, !u, !t, d, *N, Ap, x, *incx);
}
BLAS_API void stpmv_(cc *up, cc *tr, cc *dg, ci *N, cf *Ap, float *x, ci *incx) {
    if (f_tpxv(0, up, tr, dg, N, Ap, x, incx)) { zglos("stpmv_", "CPU"); CPU(stpmv_, tpxv_f_t)(up, tr, dg, N, Ap, x, incx); }
}
BLAS_API void stpsv_(cc *up, cc *tr, cc *dg, ci *N, cf *Ap, float *x, ci *incx) {
    if (f_tpxv(1, up, tr, dg, N, Ap, x, incx)) { zglos("stpsv_", "CPU"); CPU(stpsv_, tpxv_f_t)(up, tr, dg, N, Ap, x, incx); }
}
BLAS_API void sspr_(cc *up, ci *N, cf *alpha, cf *x, ci *incx, float *Ap) {
    int u = f_u(up);
    if (u < 0 || r_spr2(!u, *N, *alpha, x, *incx, NULL, 0, Ap)) {
        zglos("sspr_", "CPU"); CPU(sspr_, void (*)(cc *, ci *, cf *, cf *, ci *, float *))(up, N, alpha, x, incx, Ap); }
}
BLAS_API void sspr2_(cc *up, ci *N, cf *alpha, cf *x, ci *incx, cf *y, ci *incy, float *Ap) {
    int u = f_u(up);
    if (u < 0 || r_spr2(!u, *N, *alpha, x, *incx, y, *incy, Ap)) {
        zglos("sspr2_", "CPU"); CPU(sspr2_, void (*)(cc *, ci *, cf *, cf *, ci *, cf *, ci *, float *))(up, N, alpha, x, incx, y, incy, Ap); }
}

// ==== KONIEC TYPU ====
#include "gen9blas_blas_d.inc"   // wersja D (lib/gen_blas_d.py)
