// Porównanie bibliotek BLAS poziomu 3: libgen9blas (GPU), CLBlast (GPU, OpenCL), OpenBLAS (CPU).
// Użycie: l3 <gen9blas|clblast|openblas> <gemm|symm|syrk|syr2k|trmm|trsm> <h|s|d|x> <n> [sekundy]   (x: gen9blas GEN9BLAS_HS, FP16 z sumą FP32)
//   bez [sekundy]: rozgrzewka + 5 pomiarów (najlepszy czas ściany, średni czas CPU);
//   z [sekundy]: powtarza przez podany czas (do pomiaru mocy: bench/power.sh) i podaje średnie.
// Zmienne: TRYB=gpu (domyślnie: dane już w pamięci biblioteki GPU -- gen9blas: USM, CLBlast: bufory cl_mem)
//          TRYB=host (dane w zwykłej pamięci malloc: gen9blas kopiuje sam, CLBlast: zapis + odczyt buforów w czasie)
//          CZEKAJ=uspienie (CLBlast: czekanie na zdarzenie z nanosleep jak w gen9blas; domyślnie clWaitForEvents)
// Wszystkie macierze row-major, n × n. Wariant: side=L, uplo=L, bez transpozycji, diag nie-jednostkowa, α = 1, β = 0
// (GEMM/SYMM/SYRK/SYR2K). A dla TRMM/TRSM ma przekątną n (dobrze uwarunkowana).
// Błąd: 256 losowych elementów wyniku względem obliczenia w double na tych samych (zaokrąglonych) danych,
// względny do sumy modułów składników (|c - ref| / Σ|a·b|); dla TRSM -- residuum wiersza A·x - b.
// CPU% = czas CPU procesu (wszystkie wątki) / czas ściany: 100% = jeden rdzeń zajęty przez cały czas.
#define CL_TARGET_OPENCL_VERSION 300
#include <clblast_c.h>
#include <clblast_half.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>
#include "../../lib/gen9blas.h"

// CBLAS (OpenBLAS bez nagłówków w systemie: deklaracje z cblas.h, wartości wyliczeń jak w standardzie).
typedef enum { CblasRowMajor = 101 } CBLAS_LAYOUT;
enum { CblasNoTrans = 111, CblasLower = 122, CblasNonUnit = 131, CblasLeft = 141 };
#define DEKL(T, x) \
void cblas_##x##gemm(int, int, int, int, int, int, T, const T *, int, const T *, int, T, T *, int); \
void cblas_##x##symm(int, int, int, int, int, T, const T *, int, const T *, int, T, T *, int); \
void cblas_##x##syrk(int, int, int, int, int, T, const T *, int, T, T *, int); \
void cblas_##x##syr2k(int, int, int, int, int, T, const T *, int, const T *, int, T, T *, int); \
void cblas_##x##trmm(int, int, int, int, int, int, int, T, const T *, int, T *, int); \
void cblas_##x##trsm(int, int, int, int, int, int, int, T, const T *, int, T *, int);
DEKL(float, s)
DEKL(double, d)

#define CHECK(x) do { long e_ = (long)(x); if (e_ != 0) { fprintf(stderr, "%s:%d: %s = %ld\n", __FILE__, __LINE__, #x, e_); exit(1); } } while (0)

static double teraz(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static double cpu(void) { struct rusage r; getrusage(RUSAGE_SELF, &r);
    return r.ru_utime.tv_sec + r.ru_stime.tv_sec + (r.ru_utime.tv_usec + r.ru_stime.tv_usec) * 1e-6; }

static char bib[16], op[8], p; static size_t n, es; static int host;
static double *rA, *rB, *rC;          // dane wejściowe po zaokrągleniu (double), rC: wynik do sprawdzenia
static void *hA, *hB, *hC, *hC0;      // dane w precyzji badanej (hC0: kopia B/C wejściowego dla trmm/trsm)
// CLBlast
static cl_context ctx; static cl_command_queue q; static cl_mem bA, bB, bC; static int usypiaj;

static void zapisz(void *h, size_t i, double v) {
    if (p == 'd') ((double *)h)[i] = v; else if (p == 's') ((float *)h)[i] = (float)v; else ((cl_half *)h)[i] = FloatToHalf((float)v);
}
static double czytaj(const void *h, size_t i) {
    return p == 'd' ? ((const double *)h)[i] : p == 's' ? ((const float *)h)[i] : HalfToFloat(((const cl_half *)h)[i]);
}

static void czekaj_cl(cl_event ev) {
    if (!usypiaj) { CHECK(clWaitForEvents(1, &ev)); clReleaseEvent(ev); return; }
    cl_int st = CL_COMPLETE; long ns = 20000; clFlush(q);
    while (clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof st, &st, NULL) == CL_SUCCESS && st > CL_COMPLETE) {
        struct timespec ts = { 0, ns }; nanosleep(&ts, NULL); if (ns < 500000) ns *= 2;
    }
    clReleaseEvent(ev); CHECK(st < 0);
}

static int rodzaj(void) { return p == 'x' ? GEN9BLAS_HS : p == 'h' ? GEN9BLAS_H : p == 's' ? GEN9BLAS_S : GEN9BLAS_D; }

static void wywolaj_gen9blas(void) {
    int N = (int)n, r = rodzaj();
    if (!strcmp(op, "gemm")) CHECK(gen9blas_gemm(r, 0, 0, N, N, N, 1.0, hA, N, hB, N, 0.0, hC, N));
    else if (!strcmp(op, "symm")) CHECK(gen9blas_symm(r, GEN9BLAS_LEFT, GEN9BLAS_LOWER, N, N, 1.0, hA, N, hB, N, 0.0, hC, N));
    else if (!strcmp(op, "syrk")) CHECK(gen9blas_syrk(r, GEN9BLAS_LOWER, 0, N, N, 1.0, hA, N, 0.0, hC, N));
    else if (!strcmp(op, "syr2k")) CHECK(gen9blas_syr2k(r, GEN9BLAS_LOWER, 0, N, N, 1.0, hA, N, hB, N, 0.0, hC, N));
    else if (!strcmp(op, "trmm")) CHECK(gen9blas_trmm(r, GEN9BLAS_LEFT, GEN9BLAS_LOWER, 0, 0, N, N, 1.0, hA, N, hC, N));
    else if (!strcmp(op, "trsm")) CHECK(gen9blas_trsm(r, GEN9BLAS_LEFT, GEN9BLAS_LOWER, 0, 0, N, N, 1.0, hA, N, hC, N));
}

static void wywolaj_openblas(void) {
    int N = (int)n; CBLAS_LAYOUT L = CblasRowMajor;
    if (p == 's') {
        float *A = hA, *B = hB, *C = hC;
        if (!strcmp(op, "gemm")) cblas_sgemm(L, CblasNoTrans, CblasNoTrans, N, N, N, 1, A, N, B, N, 0, C, N);
        else if (!strcmp(op, "symm")) cblas_ssymm(L, CblasLeft, CblasLower, N, N, 1, A, N, B, N, 0, C, N);
        else if (!strcmp(op, "syrk")) cblas_ssyrk(L, CblasLower, CblasNoTrans, N, N, 1, A, N, 0, C, N);
        else if (!strcmp(op, "syr2k")) cblas_ssyr2k(L, CblasLower, CblasNoTrans, N, N, 1, A, N, B, N, 0, C, N);
        else if (!strcmp(op, "trmm")) cblas_strmm(L, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, C, N);
        else if (!strcmp(op, "trsm")) cblas_strsm(L, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, C, N);
    } else if (p == 'd') {
        double *A = hA, *B = hB, *C = hC;
        if (!strcmp(op, "gemm")) cblas_dgemm(L, CblasNoTrans, CblasNoTrans, N, N, N, 1, A, N, B, N, 0, C, N);
        else if (!strcmp(op, "symm")) cblas_dsymm(L, CblasLeft, CblasLower, N, N, 1, A, N, B, N, 0, C, N);
        else if (!strcmp(op, "syrk")) cblas_dsyrk(L, CblasLower, CblasNoTrans, N, N, 1, A, N, 0, C, N);
        else if (!strcmp(op, "syr2k")) cblas_dsyr2k(L, CblasLower, CblasNoTrans, N, N, 1, A, N, B, N, 0, C, N);
        else if (!strcmp(op, "trmm")) cblas_dtrmm(L, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, C, N);
        else if (!strcmp(op, "trsm")) cblas_dtrsm(L, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, C, N);
    } else { fprintf(stderr, "OpenBLAS nie ma FP16\n"); exit(3); }
}

static void wywolaj_clblast(void) {
    size_t N = n, nb = n * n * es; cl_event ev = NULL;
    const CLBlastLayout L = CLBlastLayoutRowMajor; const CLBlastTranspose T = CLBlastTransposeNo;
    const CLBlastSide S = CLBlastSideLeft; const CLBlastTriangle U = CLBlastTriangleLower; const CLBlastDiagonal D = CLBlastDiagonalNonUnit;
    if (host) {   // dane z pamięci hosta: zapis wejść (bez czekania -- kolejka jest w kolejności)
        CHECK(clEnqueueWriteBuffer(q, bA, CL_FALSE, 0, nb, hA, 0, NULL, NULL));
        if (strcmp(op, "syrk")) CHECK(clEnqueueWriteBuffer(q, !strcmp(op, "trmm") || !strcmp(op, "trsm") ? bC : bB, CL_FALSE, 0, nb,
                                                          !strcmp(op, "trmm") || !strcmp(op, "trsm") ? hC : hB, 0, NULL, NULL));
    }
    int s;
    if (p == 'h') {
        cl_half one = FloatToHalf(1.0f), zero = FloatToHalf(0.0f);
        if (!strcmp(op, "gemm")) s = CLBlastHgemm(L, T, T, N, N, N, one, bA, 0, N, bB, 0, N, zero, bC, 0, N, &q, &ev);
        else if (!strcmp(op, "symm")) s = CLBlastHsymm(L, S, U, N, N, one, bA, 0, N, bB, 0, N, zero, bC, 0, N, &q, &ev);
        else if (!strcmp(op, "syrk")) s = CLBlastHsyrk(L, U, T, N, N, one, bA, 0, N, zero, bC, 0, N, &q, &ev);
        else if (!strcmp(op, "syr2k")) s = CLBlastHsyr2k(L, U, T, N, N, one, bA, 0, N, bB, 0, N, zero, bC, 0, N, &q, &ev);
        else if (!strcmp(op, "trmm")) s = CLBlastHtrmm(L, S, U, T, D, N, N, one, bA, 0, N, bC, 0, N, &q, &ev);
        else { fprintf(stderr, "CLBlast nie ma HTRSM\n"); exit(3); }
    } else if (p == 's') {
        if (!strcmp(op, "gemm")) s = CLBlastSgemm(L, T, T, N, N, N, 1, bA, 0, N, bB, 0, N, 0, bC, 0, N, &q, &ev);
        else if (!strcmp(op, "symm")) s = CLBlastSsymm(L, S, U, N, N, 1, bA, 0, N, bB, 0, N, 0, bC, 0, N, &q, &ev);
        else if (!strcmp(op, "syrk")) s = CLBlastSsyrk(L, U, T, N, N, 1, bA, 0, N, 0, bC, 0, N, &q, &ev);
        else if (!strcmp(op, "syr2k")) s = CLBlastSsyr2k(L, U, T, N, N, 1, bA, 0, N, bB, 0, N, 0, bC, 0, N, &q, &ev);
        else if (!strcmp(op, "trmm")) s = CLBlastStrmm(L, S, U, T, D, N, N, 1, bA, 0, N, bC, 0, N, &q, &ev);
        else s = CLBlastStrsm(L, S, U, T, D, N, N, 1, bA, 0, N, bC, 0, N, &q, &ev);
    } else {
        if (!strcmp(op, "gemm")) s = CLBlastDgemm(L, T, T, N, N, N, 1, bA, 0, N, bB, 0, N, 0, bC, 0, N, &q, &ev);
        else if (!strcmp(op, "symm")) s = CLBlastDsymm(L, S, U, N, N, 1, bA, 0, N, bB, 0, N, 0, bC, 0, N, &q, &ev);
        else if (!strcmp(op, "syrk")) s = CLBlastDsyrk(L, U, T, N, N, 1, bA, 0, N, 0, bC, 0, N, &q, &ev);
        else if (!strcmp(op, "syr2k")) s = CLBlastDsyr2k(L, U, T, N, N, 1, bA, 0, N, bB, 0, N, 0, bC, 0, N, &q, &ev);
        else if (!strcmp(op, "trmm")) s = CLBlastDtrmm(L, S, U, T, D, N, N, 1, bA, 0, N, bC, 0, N, &q, &ev);
        else s = CLBlastDtrsm(L, S, U, T, D, N, N, 1, bA, 0, N, bC, 0, N, &q, &ev);
    }
    if (s != 0) { fprintf(stderr, "CLBlast %c%s: status %d\n", p, op, s); exit(4); }
    if (host) { if (ev) clReleaseEvent(ev); CHECK(clEnqueueReadBuffer(q, bC, CL_FALSE, 0, nb, hC, 0, NULL, &ev)); }
    czekaj_cl(ev);
}

// TRMM/TRSM nadpisują B w miejscu: przed każdym wywołaniem odtwarzamy wejście (poza pomiarem czasu).
static void przywroc(void) {
    if (strcmp(op, "trmm") && strcmp(op, "trsm")) return;
    if (!strcmp(bib, "clblast") && !host) { CHECK(clEnqueueCopyBuffer(q, bB, bC, 0, 0, n * n * es, 0, NULL, NULL)); CHECK(clFinish(q)); }
    else memcpy(hC, hC0, n * n * es);
}

static void wywolaj(void) {
    if (!strcmp(bib, "gen9blas")) wywolaj_gen9blas(); else if (!strcmp(bib, "openblas")) wywolaj_openblas(); else wywolaj_clblast();
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "uzycie: %s gen9blas|clblast|openblas gemm|symm|syrk|syr2k|trmm|trsm h|s|d n [sekundy]\n", argv[0]); return 2; }
    // OpenBLAS przy ładowaniu startuje wątki, które kręcą się w tle (~550% CPU nawet bez wywołań OpenBLAS).
    // Dla bibliotek GPU: ponowne uruchomienie z jednym wątkiem OpenBLAS (używany tylko w referencji -- wcale).
    if (strcmp(argv[1], "openblas") && !getenv("OPENBLAS_NUM_THREADS")) {
        setenv("OPENBLAS_NUM_THREADS", "1", 1); execv("/proc/self/exe", argv); perror("execv"); return 1;
    }
    snprintf(bib, sizeof bib, "%s", argv[1]);
    if (argv[3][0] == 'x' && strcmp(argv[1], "gen9blas")) { fprintf(stderr, "x (FP16 z sumą FP32) tylko dla gen9blas\n"); return 3; } snprintf(op, sizeof op, "%s", argv[2]); p = argv[3][0];
    n = strtoul(argv[4], 0, 0); double sek = argc > 5 ? atof(argv[5]) : 0;
    es = p == 'd' ? 8 : p == 's' ? 4 : 2; size_t nn = n * n, nb = nn * es;
    host = getenv("TRYB") && !strcmp(getenv("TRYB"), "host");
    usypiaj = getenv("CZEKAJ") && !strcmp(getenv("CZEKAJ"), "uspienie");
    int tr = !strcmp(op, "trmm") || !strcmp(op, "trsm");

    int gen9blas = !strcmp(bib, "gen9blas");
    if (gen9blas) { const char *k = getenv("GEN9BLAS_DIR"); CHECK(gen9blas_init(k ? k : "bench") != 0); }
    void *(*alok)(size_t) = gen9blas && !host ? gen9blas_alloc : malloc;
    hA = alok(nb); hB = alok(nb); hC = alok(nb); hC0 = malloc(nb);
    rA = malloc(nn * 8); rB = malloc(nn * 8); rC = malloc(nn * 8);
    srand(1234);
    for (size_t i = 0; i < nn; i++) {
        double a = 2.0 * rand() / RAND_MAX - 1.0, b = 2.0 * rand() / RAND_MAX - 1.0;
        if (tr && i / n == i % n) a = (double)n;
        if (tr && i % n > i / n) a = 0;       // A dolnotrójkątna (górny trójkąt nieużywany, ale zerowy dla referencji)
        zapisz(hA, i, a); rA[i] = czytaj(hA, i); zapisz(hB, i, b); rB[i] = czytaj(hB, i);
        zapisz(hC, i, 0);
    }
    // SYMM/SYRK/SYR2K: biblioteki czytają tylko trójkąt L; referencja liczona z symetrycznego odbicia (rA[i][j], j > i := rA[j][i]).
    if (tr) memcpy(hC0, hB, nb), memcpy(hC, hB, nb);

    if (!strcmp(bib, "clblast")) {
        cl_platform_id plat; cl_device_id dev; cl_int err;
        CHECK(clGetPlatformIDs(1, &plat, NULL)); CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL));
        ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err); CHECK(err);
        q = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err); CHECK(err);
        bA = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nb, hA, &err); CHECK(err);
        bB = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nb, hB, &err); CHECK(err);
        bC = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, nb, hC, &err); CHECK(err);
    }

    przywroc(); double t0 = teraz(); wywolaj(); double t_pierwsze = teraz() - t0;   // kompilacja kerneli, przydziały
    double najl = 1e30, sciana = 0, cpu_s = 0; int k = 0; static double czasy[100000];
    t0 = teraz();
    for (k = 0; sek > 0 ? teraz() - t0 < sek : k < 5; k++) {   // [sekundy]: czas średni, inaczej najlepszy z 5
        przywroc();
        double c0 = cpu(), t = teraz(); wywolaj(); t = teraz() - t; cpu_s += cpu() - c0; sciana += t; if (t < najl) najl = t;
        if (k < 100000) czasy[k] = t;
    }
    int nk = k < 100000 ? k : 100000;   // mediana czasów (w trybie [sekundy] wypisywana obok średniej)
    for (int i = 1; i < nk; i++) { double v = czasy[i]; int j = i; while (j > 0 && czasy[j - 1] > v) { czasy[j] = czasy[j - 1]; j--; } czasy[j] = v; }
    double med = nk ? czasy[nk / 2] : 0;
    if (sek > 0) najl = sciana / k;

    // Wynik do pamięci hosta i sprawdzenie.
    if (!strcmp(bib, "clblast") && !host) CHECK(clEnqueueReadBuffer(q, bC, CL_TRUE, 0, nb, hC, 0, NULL, NULL));
    #define SA(i, j) (j > i && sym ? rA[(j) * n + (i)] : rA[(i) * n + (j)])
    int sym = !strcmp(op, "symm"); double maxe = 0, sume = 0; srand(99); int ile = 0;
    for (int s = 0; s < 256; s++) {
        size_t i = rand() % n, j = rand() % n;
        if ((!strcmp(op, "syrk") || !strcmp(op, "syr2k")) && j > i) { size_t t = i; i = j; j = t; }   // tylko trójkąt L
        double c = czytaj(hC, i * n + j), ref = 0, mag = 0;
        if (!strcmp(op, "gemm") || !strcmp(op, "symm"))
            for (size_t l = 0; l < n; l++) { double t = SA(i, l) * rB[l * n + j]; ref += t; mag += fabs(t); }
        else if (!strcmp(op, "syrk"))
            for (size_t l = 0; l < n; l++) { double t = rA[i * n + l] * rA[j * n + l]; ref += t; mag += fabs(t); }
        else if (!strcmp(op, "syr2k"))
            for (size_t l = 0; l < n; l++) { double t1 = rA[i * n + l] * rB[j * n + l], t2 = rB[i * n + l] * rA[j * n + l];
                ref += t1 + t2; mag += fabs(t1) + fabs(t2); }
        else if (!strcmp(op, "trmm"))
            for (size_t l = 0; l <= i; l++) { double t = rA[i * n + l] * rB[l * n + j]; ref += t; mag += fabs(t); }
        else {   // trsm: residuum (A·X)[i][j] - B[i][j], X z wyniku w badanej precyzji
            for (size_t l = 0; l <= i; l++) { double t = rA[i * n + l] * czytaj(hC, l * n + j); ref += t; mag += fabs(t); }
            ref -= rB[i * n + j]; mag += fabs(rB[i * n + j]); c = 0;
        }
        double e = fabs(c - ref) / (mag > 0 ? mag : 1); sume += e; if (e > maxe) maxe = e; ile++;
    }
    double fl = (!strcmp(op, "gemm") || !strcmp(op, "symm") || !strcmp(op, "syr2k") ? 2.0 : 1.0) * n * n * n;
    printf("%-8s %-5s %c n=%-5zu %s  %9.2f ms  %7.1f GFLOPS  CPU %4.0f%%  blad sr=%.1e max=%.1e  (1. wywolanie %.0f ms, %d wywolan, mediana %.2f ms, min %.2f ms)\n",
           bib, op, p, n, host ? "host" : "gpu ", najl * 1e3, fl / najl / 1e9, 100 * cpu_s / sciana, sume / ile, maxe,
           t_pierwsze * 1e3, k, med * 1e3, czasy[0] * 1e3);
    return 0;
}
