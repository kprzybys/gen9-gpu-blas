// libgen9blas: BLAS poziomu 3 na UHD 620 (Gen9.5) z ręcznymi pętlami GEMM w asemblerze (asm/gemm_gen.py).
// Rdzeń: GEMM wsadowy C_b = alpha * op(A_b)·op(B_b) + beta * C_b (b = 0..count-1, odstępy sA, sB, sC w elementach),
// row-major, dowolne M, N, K. Rodzaje: GEN9BLAS_H (FP16), GEN9BLAS_HS (FP16, suma porcjami do FP32), GEN9BLAS_S (FP32).
// Poziom 3 (SYMM, SYRK, SYR2K, TRMM, TRSM) zbudowany na GEMM, w całości na GPU: rozwijanie macierzy trójkątnej /
// symetrycznej, sklejanie operandów SYR2K i odwrotności bloków diagonalnych TRSM (FP64) to kernele.
// CPU tylko zleca pracę i czeka z uśpieniem (czekaj()). Bez kopii: pamięć USM host (gen9blas_alloc, w Pythonie gen9blas.empty);
// inna pamięć hosta jest kopiowana przez CPU do roboczych buforów USM (sterownik nie rejestruje zwykłej pamięci hosta).
#include "gen9blas.h"
#ifdef GEN9BLAS_NEO   // dawna ścieżka przez NEO (OpenCL) -- tylko do porównań (libgen9blas_neo.so)
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <CL/cl_ext.h>
#else            // domyślnie: kcl -- podzbiór OpenCL bezpośrednio na i915 (kcl.c, BEZNEO.md)
#include "kcl.h"
#endif
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double teraz(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

typedef struct {
    cl_kernel main, pa, pb, pa_t, pb_t, pa_q, pb_w2,   // pa_q, pb_w2: szybsze pakowanie bez transpozycji (NULL: brak)
              epi, epi_tri, tL, tY, tZ, tI[4], pel, skl;   // t*: TRSM; pel: pełna; skl: SYR2K
    int es, tn, km, gotowy;
} rodzaj_t;

static cl_context ctx;
static cl_command_queue q;
static cl_device_id dev;
static rodzaj_t R[4];   // indeks = rodzaj: GEN9BLAS_H, GEN9BLAS_S, GEN9BLAS_HS, GEN9BLAS_D
static cl_mem dAp, dBp, dCp;
static size_t sAp, sBp, sCp;
static double ostatni_ms;
static int profil;   // GEN9BLAS_PROFILE=1: kolejka z profilowaniem, czasy kerneli
static int ostatni_zc;
static char blad[1200];
static clHostMemAllocINTEL_fn usm_alloc;          // cl_intel_unified_shared_memory (zob. obiekt())
static clMemBlockingFreeINTEL_fn usm_free;
static clGetMemAllocInfoINTEL_fn usm_info;
static clSetKernelArgMemPointerINTEL_fn usm_arg;

#define SPRAWDZ(x) do { cl_int e_ = (x); if (e_ != CL_SUCCESS) { snprintf(blad, sizeof blad, "%s:%d: %s = %d", __FILE__, __LINE__, #x, e_); return -1; } } while (0)
#define BLAD(...) do { snprintf(blad, sizeof blad, __VA_ARGS__); return -1; } while (0)

static __attribute__((unused)) unsigned char *wczytaj(const char *p, size_t *len) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); *len = ftell(f); rewind(f);
    unsigned char *s = malloc(*len + 1);
    if (fread(s, 1, *len, f) != *len) { fclose(f); free(s); return NULL; }
    s[*len] = 0; fclose(f); return s;
}

static int program_bin(const char *katalog, const char *plik, const char *kernel, cl_kernel *k) {
#ifndef GEN9BLAS_NEO   // jądra wbudowane (kernels.c, generowane przy budowie z tych samych plików)
    (void)katalog; cl_program pr; cl_int err;
    if (kcl_program(plik, NULL, &pr) != CL_SUCCESS) BLAD("brak wbudowanego programu %s", plik);
    *k = clCreateKernel(pr, kernel, &err); SPRAWDZ(err);
    return 0;
#else
    char p[1024]; size_t len; cl_int err, bst;
    snprintf(p, sizeof p, "%s/%s", katalog, plik);
    unsigned char *bin = wczytaj(p, &len);
    if (!bin) BLAD("brak %.900s (zbuduj: asm/gemm_build.sh h|hd|s)", p);
    cl_program pb = clCreateProgramWithBinary(ctx, 1, &dev, &len, (const unsigned char **)&bin, &bst, &err);
    free(bin); SPRAWDZ(err); SPRAWDZ(bst); SPRAWDZ(clBuildProgram(pb, 1, &dev, "", NULL, NULL));
    *k = clCreateKernel(pb, kernel, &err); SPRAWDZ(err);
    return 0;
#endif
}

static int program_zr(const char *katalog, const char *plik, const char *opcje, cl_program *pr) {
#ifndef GEN9BLAS_NEO
    (void)katalog;
    if (kcl_program(plik, opcje, pr) != CL_SUCCESS) BLAD("brak wbudowanego programu %s [%s]", plik, opcje);
    return 0;
#else
    char p[1024]; size_t len; cl_int err;
    snprintf(p, sizeof p, "%s/%s", katalog, plik);
    char *src = (char *)wczytaj(p, &len);
    if (!src) BLAD("brak %.900s", p);
    *pr = clCreateProgramWithSource(ctx, 1, (const char **)&src, NULL, &err); free(src); SPRAWDZ(err);
    if (clBuildProgram(*pr, 1, &dev, opcje, NULL, NULL) != CL_SUCCESS) {
        clGetProgramBuildInfo(*pr, dev, CL_PROGRAM_BUILD_LOG, sizeof blad, blad, NULL); return -1;
    }
    return 0;
#endif
}

#define KER(pr, nazwa, dst) do { cl_int e2_; dst = clCreateKernel(pr, nazwa, &e2_); SPRAWDZ(e2_); } while (0)

// Kernele poziomów 1–2 (bench/level12.cl): L12[0] FP16 (GEN9BLAS_H, GEN9BLAS_HS), L12[1] FP32.
typedef struct { cl_kernel axpy, scal, copy, copy16, rot, red, iamax, gemv_n, gemv_nw, gemv_t1, gemv_t1b, gemv_t2, ger, syr2, do_f, z_f, trsv_blok, trsv_akt_n, trsv_akt_t, rotm, pmv, psv, spr2, symv1, symv2, chudy_k1, chudy_k4, chudy_k8, chudy_r4, chudy_r8, chudy_n4, chudy_n2; int es; } l12_t;
static l12_t L12[3];   // 0 FP16, 1 FP32, 2 FP64
static int init_l12(const char *katalog) {
    for (int i = 0; i < 3; i++) {
        cl_program pr; const char *suf = "hfd" + i, *opc[3] = { "-cl-std=CL3.0 -DPOL", "-cl-std=CL3.0", "-cl-std=CL3.0 -DDBL" };
        char n[32], sf[2] = { *suf, 0 };
        if (program_zr(katalog, "level12.cl", opc[i], &pr)) return -1;
        l12_t *k = &L12[i]; k->es = i == 0 ? 2 : i == 1 ? 4 : 8; suf = sf;
        struct { const char *n; cl_kernel *k; } t[] = { { "axpy", &k->axpy }, { "scal", &k->scal }, { "copy", &k->copy }, { "copy16", &k->copy16 },
            { "rot", &k->rot }, { "red", &k->red }, { "iamax", &k->iamax }, { "gemv_n", &k->gemv_n }, { "gemv_nw", &k->gemv_nw },
            { "gemv_t1", &k->gemv_t1 }, { "gemv_t1b", &k->gemv_t1b }, { "gemv_t2", &k->gemv_t2 }, { "ger", &k->ger }, { "syr2", &k->syr2 },
            { "do_f", &k->do_f }, { "z_f", &k->z_f }, { "trsv_blok", &k->trsv_blok },
            { "trsv_akt_n", &k->trsv_akt_n }, { "trsv_akt_t", &k->trsv_akt_t }, { "rotm", &k->rotm }, { "pmv", &k->pmv },
            { "psv", &k->psv }, { "spr2", &k->spr2 }, { "symv1", &k->symv1 }, { "symv2", &k->symv2 } };
        for (size_t j = 0; j < sizeof t / sizeof t[0]; j++) { snprintf(n, sizeof n, "%s_%s", t[j].n, suf); KER(pr, n, *t[j].k); }
        if (i < 2) {   // chudy GEMM: tylko FP16 i FP32 (bench/level12.cl, #ifndef DBL)
            struct { const char *n; cl_kernel *k; } c[] = { { "chudy_k1", &k->chudy_k1 }, { "chudy_k4", &k->chudy_k4 }, { "chudy_k8", &k->chudy_k8 },
                { "chudy_r4", &k->chudy_r4 }, { "chudy_r8", &k->chudy_r8 },
                { "chudy_n4", &k->chudy_n4 }, { "chudy_n2", &k->chudy_n2 } };
            for (size_t j = 0; j < sizeof c / sizeof c[0]; j++) { snprintf(n, sizeof n, "%s_%s", c[j].n, suf); KER(pr, n, *c[j].k); }
        }
    }
    return 0;
}

// Jednokrotna: kolejne wywołania (np. warstwa BLAS i moduł Pythona w jednym procesie) nic nie robią -- drugi kontekst
// OpenCL unieważniłby bufory robocze i pamięć USM z pierwszego (błąd -50 przy ustawianiu argumentów).
int gen9blas_init(const char *katalog) {
    static int gotowe;
    if (gotowe) return 0;
    cl_int err; cl_platform_id plat;
    SPRAWDZ(clGetPlatformIDs(1, &plat, NULL));
    SPRAWDZ(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL));
    ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err); SPRAWDZ(err);
    // Profilowanie kolejki tylko przy GEN9BLAS_PROFILE=1: z nim każde zdarzenie (także znacznik w czekaj()) kosztuje
    // ~20 µs CPU więcej (pomiar 2026-11-11: znacznik 21 vs 2,3 µs). Bez profilowania gen9blas_last_time_ms() podaje czas
    // całego wywołania.
    profil = getenv("GEN9BLAS_PROFILE") != NULL;
    cl_queue_properties qp[] = { CL_QUEUE_PROPERTIES, profil ? CL_QUEUE_PROFILING_ENABLE : 0, 0 };
    q = clCreateCommandQueueWithProperties(ctx, dev, qp, &err); SPRAWDZ(err);
    cl_program ph, ps, pd;
    if (program_zr(katalog, "hgemm.cl", "", &ph) || program_zr(katalog, "sgemm.cl", "", &ps) ||
        program_zr(katalog, "dgemm.cl", "", &pd)) return -1;
    usm_alloc = (clHostMemAllocINTEL_fn)clGetExtensionFunctionAddressForPlatform(plat, "clHostMemAllocINTEL");
    usm_free = (clMemBlockingFreeINTEL_fn)clGetExtensionFunctionAddressForPlatform(plat, "clMemBlockingFreeINTEL");
    usm_info = (clGetMemAllocInfoINTEL_fn)clGetExtensionFunctionAddressForPlatform(plat, "clGetMemAllocInfoINTEL");
    usm_arg = (clSetKernelArgMemPointerINTEL_fn)clGetExtensionFunctionAddressForPlatform(plat, "clSetKernelArgMemPointerINTEL");
    if (!usm_alloc || !usm_free || !usm_info || !usm_arg) BLAD("brak rozszerzenia cl_intel_unified_shared_memory");
    if (init_l12(katalog)) return -1;
    // kernele „v”: (Ap, Bp, C, M, N, ldc, K, Kfull) -- także fragmenty danych spakowanych raz (pod-bufory)
    // bin = NULL: główny kernel ze źródła (FP64: dgemm.cl, OpenCL C), nie z binarki asm
    struct { int r; const char *bin, *k, *pa, *pb, *epi, *tri; int es, tn, km; cl_program pr; } t[4] = {
        { GEN9BLAS_H,  "gen_h.bin",  "hgemm_v",    "pk_a2", "pk_bq",  "kpg_epilog_b",  "kpg_epilog_tri",  2, 32, 16, ph },
        { GEN9BLAS_HS, "gen_hd.bin", "hgemm_kc_v", "pk_a3", "pk_b16", "kpg_epilog_b",  "kpg_epilog_tri",  2, 16, GEN9BLAS_KC, ph },
        { GEN9BLAS_S,  "gen_s.bin",  "sgemm_v",    "pkf_a", "pkf_b",  "kpg_epilog_fb", "kpg_epilog_trif", 4, 16, 16, ps },
        { GEN9BLAS_D,  NULL,         "dgemm_v",    "pkd_a", "pkd_b",  "kpg_epilog_db", "kpg_epilog_trid", 8, 16, 16, pd } };
    for (int i = 0; i < 4; i++) {
        rodzaj_t *r = &R[t[i].r]; char n[64];
        if (t[i].bin ? program_bin(katalog, t[i].bin, t[i].k, &r->main) : 0) return -1;
        if (!t[i].bin) KER(t[i].pr, t[i].k, r->main);
        KER(t[i].pr, t[i].pa, r->pa); KER(t[i].pr, t[i].pb, r->pb); KER(t[i].pr, t[i].epi, r->epi);
        KER(t[i].pr, t[i].tri, r->epi_tri);
        const char *suf = t[i].es == 4 ? "f" : t[i].es == 8 ? "d" : "h";
        snprintf(n, sizeof n, "trsm_L_%s", suf); KER(t[i].pr, n, r->tL);
        snprintf(n, sizeof n, "trsm_Y_%s", suf); KER(t[i].pr, n, r->tY);
        snprintf(n, sizeof n, "trsm_zapisz_%s", suf); KER(t[i].pr, n, r->tZ);
        for (int j = 0; j < 4; j++) { snprintf(n, sizeof n, "trsm_inv%d_%s", j, suf); KER(t[i].pr, n, r->tI[j]); }
        snprintf(n, sizeof n, "gen9blas_pelna_%s", suf); KER(t[i].pr, n, r->pel);
        snprintf(n, sizeof n, "gen9blas_sklej_%s", suf); KER(t[i].pr, n, r->skl);
        snprintf(n, sizeof n, "%s_t", t[i].pa); KER(t[i].pr, n, r->pa_t);
        snprintf(n, sizeof n, "%s_t", t[i].pb); KER(t[i].pr, n, r->pb_t);
        cl_int e3;   // opcjonalne szybsze kernele pakowania (GEN9BLAS_PACK=stare: wyłączone)
        int stare = getenv("GEN9BLAS_PACK") && !strcmp(getenv("GEN9BLAS_PACK"), "stare");
        snprintf(n, sizeof n, "%s_q", t[i].pa); r->pa_q = stare ? NULL : clCreateKernel(t[i].pr, n, &e3); if (e3) r->pa_q = NULL;
        snprintf(n, sizeof n, "%s_w2", t[i].pb); r->pb_w2 = stare ? NULL : clCreateKernel(t[i].pr, n, &e3); if (e3) r->pb_w2 = NULL;
        r->es = t[i].es; r->tn = t[i].tn; r->km = t[i].km; r->gotowy = 1;
    }
    blad[0] = 0;
    gotowe = 1;
    return 0;
}

// Czekanie na koniec kolejki z uśpieniem: clFinish / clWaitForEvents / odczyty blokujące w NEO kręcą się aktywnie
// na CPU (~6 W). Marker + clFlush, potem sprawdzanie stanu co 20 µs .. 0,5 ms. GEN9BLAS_WAIT=aktywne: clFinish.
static int czekaj(void) {
#ifndef GEN9BLAS_NEO   // kcl: wysłanie bufora komend i GEM_WAIT (śpi w jądrze Linuksa do przerwania od GPU)
    if (kcl_wait() != CL_SUCCESS) BLAD("błąd wykonania na GPU (kcl)");
    return 0;
#else
    static int aktywnie = -1;
    if (aktywnie < 0) { const char *e = getenv("GEN9BLAS_WAIT"); aktywnie = e && !strcmp(e, "aktywne"); }
    if (aktywnie) { SPRAWDZ(clFinish(q)); return 0; }
    cl_event e; cl_int st = CL_COMPLETE; long ns = 20000;
    SPRAWDZ(clEnqueueMarkerWithWaitList(q, 0, NULL, &e)); SPRAWDZ(clFlush(q));
    while (clGetEventInfo(e, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof st, &st, NULL) == CL_SUCCESS && st > CL_COMPLETE) {
        struct timespec ts = { 0, ns }; nanosleep(&ts, NULL);
        if (ns < 500000) ns *= 2;
    }
    clReleaseEvent(e);
    if (st < 0) BLAD("błąd wykonania na GPU (%d)", st);
    return 0;
#endif
}

// Zdarzenia kerneli GEMM do pomiaru czasu (uruchom(..., mierz=1)); czasy czytane po czekaj().
static cl_event zd[256];
static int nzd;
static double czasy_zdarzen(void) {
    double s = 0;
    if (!profil) { for (int i = 0; i < nzd; i++) clReleaseEvent(zd[i]); nzd = 0; return -1; }   // bez profilowania
    for (int i = 0; i < nzd; i++) { cl_ulong t0, t1;
        clGetEventProfilingInfo(zd[i], CL_PROFILING_COMMAND_START, 8, &t0, NULL); clGetEventProfilingInfo(zd[i], CL_PROFILING_COMMAND_END, 8, &t1, NULL);
        s += (t1 - t0) * 1e-6; clReleaseEvent(zd[i]); }
    nzd = 0; return s;
}

static int bufor(cl_mem *b, size_t *rozm, size_t potrzeba) {
    if (*rozm >= potrzeba) return 0;
    if (*b) clReleaseMemObject(*b);
    cl_int err; *b = clCreateBuffer(ctx, CL_MEM_READ_WRITE, potrzeba, NULL, &err); SPRAWDZ(err);
    *rozm = potrzeba; return 0;
}

// ---------------- pamięć hosta dla kerneli: USM ----------------
// Bez kopii: pamięć USM host (clHostMemAllocINTEL; gen9blas_alloc, w Pythonie gen9blas.empty) -- wskaźnik trafia do kernela
// wprost (clSetKernelArgMemPointerINTEL), także wskaźnik na wnętrze przydziału. Inna pamięć hosta: kopia przez CPU
// (memcpy) do roboczego bufora USM i z powrotem. Sprawdzone 2026-11-11: bufory CL_MEM_USE_HOST_PTR na zwykłej pamięci
// hosta tworzone i zwalniane często (poziomy 1–2) kończyły się w NEO (Gen9, legacy1) awariami: ENOSPC w execbuf,
// host_ptr_manager.cpp:252, reference_tracked_object.h:134, segfault -- obiekty GPU są zamykane asynchronicznie, a nowy
// bufor na tych samych stronach dostaje nakładający się adres GPU. clEnqueueWrite/ReadBuffer z pamięci hosta rejestruje
// ją tak samo. USE_HOST_PTR na pamięci USM z kolei ignoruje przesunięcie wewnątrz przydziału (zły wynik).

void *gen9blas_alloc(size_t bajty) {
    cl_int err;
    if (!usm_alloc) { snprintf(blad, sizeof blad, "gen9blas_alloc: brak gen9blas_init"); return NULL; }
    void *p = usm_alloc(ctx, NULL, bajty ? bajty : 1, 4096, &err);
    if (err != CL_SUCCESS) { snprintf(blad, sizeof blad, "clHostMemAllocINTEL(%zu) = %d", bajty, err); return NULL; }
    return p;
}
void gen9blas_free(void *p) { if (p && usm_free) usm_free(ctx, p); }
static int usm_zawiera(const void *p, size_t bajty);
int gen9blas_is_usm(const void *p, size_t bajty) { return usm_info && p && usm_zawiera(p, bajty); }

// 1, gdy [p, p+bajty) leży w całości w jednym przydziale USM.
static int usm_zawiera(const void *p, size_t bajty) {
    cl_unified_shared_memory_type_intel t = 0; void *baza = NULL; size_t rozm = 0;
    if (usm_info(ctx, p, CL_MEM_ALLOC_TYPE_INTEL, sizeof t, &t, NULL) != CL_SUCCESS || t == CL_MEM_TYPE_UNKNOWN_INTEL) return 0;
    if (usm_info(ctx, p, CL_MEM_ALLOC_BASE_PTR_INTEL, sizeof baza, &baza, NULL) != CL_SUCCESS ||
        usm_info(ctx, p, CL_MEM_ALLOC_SIZE_INTEL, sizeof rozm, &rozm, NULL) != CL_SUCCESS) return 0;
    return (uintptr_t)p + bajty <= (uintptr_t)baza + rozm;
}

// Argument kernela: obiekt cl_mem albo wskaźnik USM.
typedef struct { cl_mem m; void *p; } bf_t;
#define BM(x) ((bf_t){ (x), NULL })
static cl_int ustaw(cl_kernel k, int i, bf_t b) { return b.p ? usm_arg(k, i, b.p) : clSetKernelArg(k, i, sizeof b.m, &b.m); }
#define ARGB(k, i, b) SPRAWDZ(ustaw(k, i, b))

// Roboczy bufor USM (kopia pamięci hosta spoza USM); powiększanie po zakończeniu pracy GPU (blokujące zwolnienie).
// Granica operacji dla kcl_userptr (obszary jednej funkcji BLAS ze wspólnymi stronami -> jeden obiekt).
#ifndef GEN9BLAS_NEO
#define NOWA_OPERACJA() kcl_new_operation()
#else
#define NOWA_OPERACJA() ((void)0)
#endif
typedef struct { void *p; size_t n; } rob_t;
#ifndef GEN9BLAS_NEO
static int bez_userptr = -1;
#endif
static rob_t uA, uB, uC;
static int robocz(rob_t *u, size_t bajty) {
    if (u->n >= bajty) return 0;
    if (u->p) usm_free(ctx, u->p);
    u->p = gen9blas_alloc(bajty); u->n = u->p ? bajty : 0;
    return u->p ? 0 : -1;
}
// Pamięć hosta [ptr, ptr+bajty) dla kernela: USM bez kopii, inaczej kopia do roboczego bufora u (kopiuj: z danymi).
// Przed zapisem do bufora roboczego czekanie na koniec pracy GPU (bufor mógł być użyty wcześniej w tym wywołaniu).
// Bez NEO: zwykła pamięć procesu (np. NumPy) przez userptr z pamięcią podręczną kcl (kcl_userptr) -- też bez kopii.
// zapis: GPU pisze do tego obszaru (userptr tylko do odczytu, gdy nie).
static int obiekt(const void *ptr, size_t bajty, rob_t *u, int kopiuj, int zapis, bf_t *wynik, int *zc) {
    if (usm_zawiera(ptr, bajty)) { *wynik = (bf_t){ NULL, (void *)ptr }; *zc = 1; return 0; }
#ifndef GEN9BLAS_NEO
    if (bez_userptr < 0) bez_userptr = getenv("GEN9BLAS_NO_USERPTR") != NULL;   // porównania: kopia jak dawniej
    cl_mem m;
    if (!bez_userptr && (m = kcl_userptr(ptr, bajty, zapis))) { *wynik = BM(m); *zc = 1; return 0; }
#else
    (void)zapis;
#endif
    if (czekaj() || robocz(u, bajty)) return -1;
    if (kopiuj) memcpy(u->p, ptr, bajty);
    *wynik = (bf_t){ NULL, u->p }; *zc = 0; return 0;
}

#define ARG(k, i, v) SPRAWDZ(clSetKernelArg(k, i, sizeof(v), &(v)))
// Skalar (alpha, beta) dla kerneli poziomu 3: double dla FP64, float dla pozostałych.
static cl_int arg_skalar(const rodzaj_t *r, cl_kernel k, int i, double v) {
    if (r->es == 8) return clSetKernelArg(k, i, sizeof v, &v);
    float f = (float)v; return clSetKernelArg(k, i, sizeof f, &f);
}
#define ARGS(k, i, v) SPRAWDZ(arg_skalar(r, k, i, v))

// Koniec wywołania: czekanie z uśpieniem, potem kopia wyniku do pamięci hosta (gdy szedł przez bufor roboczy).
// UHD 620 dzieli LLC z CPU, a NEO opróżnia pamięć podręczną GPU na końcu zlecenia: wynik w USM widać bez map/unmap.
static int do_hosta(bf_t x, int zc, size_t bajty, void *host) {
    if (czekaj()) return -1;
    if (!zc) memcpy(host, x.p, bajty);
    return 0;
}

// Kernel pakowania i siatka (3D: ostatni wymiar = paczka). A: (Kp, Mp) / A_t: (Mp, Kp) / A_q: (Kp/8·4, Mp/16);
// B: (Np, Kp) / B_t: (Kp, Np) / B_w2: (Np/2, Kp). Wymiar k zawsze w g[0] (A, B_t) albo g[1] (A_t, B, B_w2).
static cl_kernel kernel_pak(rodzaj_t *r, int rolaB, int trans, size_t Rp, size_t Kp, size_t cnt, size_t g[3]) {
    g[2] = cnt;
    if (!rolaB) {
        // pa_q (tylko FP16): work-item = 4 wiersze × 8 k (pk_a2_q) albo 8 wierszy × 4 k (pk_a3_q). W FP32 te same
        // pomysły nie dały zysku (element ma już 4 B: A 0,55 -> 0,62 ms, B 0,41 -> 0,38 ms przy n = 1024).
        if (!trans && r->pa_q) { g[0] = Kp / 2; g[1] = Rp / 16; return r->pa_q; }
        if (trans) { g[0] = Rp; g[1] = Kp; return r->pa_t; }
        g[0] = Kp; g[1] = Rp; return r->pa;
    }
    if (!trans && r->pb_w2) { g[0] = Rp / 2; g[1] = Kp; return r->pb_w2; }
    if (trans) { g[0] = Kp; g[1] = Rp; return r->pb_t; }
    g[0] = Rp; g[1] = Kp; return r->pb;
}

static int gemm_chudy(int rodzaj, int tA, int tB, int M, int N, int K, double alpha, const void *A, int lda,
                      const void *B, int ldb, double beta, void *C, int ldc);
static int prog_chudy(int rodzaj);
int gen9blas_gemm_batched(int rodzaj, int tA, int tB, int M, int N, int K, double alpha,
                     const void *A, int lda, long sA, const void *B, int ldb, long sB,
                     double beta, void *C, int ldc, long sC, int count) {
    NOWA_OPERACJA();
    if (rodzaj < 0 || rodzaj > 3 || !R[rodzaj].gotowy) BLAD("rodzaj %d niedostępny", rodzaj);
    if (M <= 0 || N <= 0 || K <= 0 || count <= 0) return 0;
    if (count == 1 && (M < N ? M : N) <= prog_chudy(rodzaj))   // małe M albo N: chude jądro bez pakowania
        return gemm_chudy(rodzaj, tA, tB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    double t_start = teraz();   // GEN9BLAS_PROFILE=1: czasy etapów na stderr
    rodzaj_t *r = &R[rodzaj];
    int es = r->es, Mp = (M + 15) / 16 * 16, Np = (N + r->tn - 1) / r->tn * r->tn, Kp = (K + r->km - 1) / r->km * r->km;
    size_t c1 = count - 1;
    size_t bA = c1 * sA * es + (tA ? (size_t)(K - 1) * lda + M : (size_t)(M - 1) * lda + K) * es;
    size_t bB = c1 * sB * es + (tB ? (size_t)(N - 1) * ldb + K : (size_t)(K - 1) * ldb + N) * es;
    size_t bC = c1 * sC * es + ((size_t)(M - 1) * ldc + N) * es;
    bf_t xA, xB, xC; int zA, zB, zC;
    if (obiekt(A, bA, &uA, 1, 0, &xA, &zA) || obiekt(B, bB, &uB, 1, 0, &xB, &zB) || obiekt(C, bC, &uC, beta != 0.0f, 1, &xC, &zC))
        return -1;
    ostatni_zc = zA | zB << 1 | zC << 2;
    if (bufor(&dAp, &sAp, (size_t)count * Mp * Kp * es + 4096) || bufor(&dBp, &sBp, (size_t)count * Np * Kp * es + 4096) ||
        bufor(&dCp, &sCp, (size_t)count * Mp * Np * es))
        return -1;
    cl_long lA = sA, lB = sB, lC = sC;
    size_t ga[3], gb[3];
    cl_kernel ka = kernel_pak(r, 0, tA, Mp, Kp, count, ga), kb = kernel_pak(r, 1, tB, Np, Kp, count, gb);
    ARGB(ka, 0, xA); ARG(ka, 1, dAp); ARG(ka, 2, M); ARG(ka, 3, K); ARG(ka, 4, lda); ARG(ka, 5, Kp); ARG(ka, 6, lA); ARG(ka, 7, Mp);
    double t_buf = teraz() - t_start; cl_event evp[3];
    SPRAWDZ(clEnqueueNDRangeKernel(q, ka, 3, NULL, ga, NULL, 0, NULL, &evp[0]));
    ARGB(kb, 0, xB); ARG(kb, 1, dBp); ARG(kb, 2, K); ARG(kb, 3, N); ARG(kb, 4, ldb); ARG(kb, 5, Kp); ARG(kb, 6, lB); ARG(kb, 7, Np);
    SPRAWDZ(clEnqueueNDRangeKernel(q, kb, 3, NULL, gb, NULL, 0, NULL, &evp[1]));
    // Zapis bezpośredni (bez Cp i epilogu): alpha = 1, beta = 0, M i N wielokrotnością kafelka, paczka ciągła w C.
    int wprost = alpha == 1.0 && beta == 0.0 && M == Mp && N == Np && (count == 1 || sC == (long)M * ldc);
    bf_t cel = wprost ? xC : BM(dCp); int ldcel = wprost ? ldc : Np;
    ARG(r->main, 0, dAp); ARG(r->main, 1, dBp); ARGB(r->main, 2, cel); ARG(r->main, 3, Mp); ARG(r->main, 4, Np);
    ARG(r->main, 5, ldcel); ARG(r->main, 6, Kp); ARG(r->main, 7, Kp);
    size_t gm = (size_t)count * (Np / r->tn) * (Mp / 16) * 16, lm = 16;
    cl_event ev;
    SPRAWDZ(clEnqueueNDRangeKernel(q, r->main, 1, NULL, &gm, &lm, 0, NULL, &ev));
    ARG(r->epi, 0, dCp); ARG(r->epi, 1, Np); ARG(r->epi, 2, Mp); ARGB(r->epi, 3, xC); ARG(r->epi, 4, ldc); ARG(r->epi, 5, lC);
    ARGS(r->epi, 6, alpha); ARGS(r->epi, 7, beta);
    size_t ge[3] = { (size_t)N, (size_t)M, (size_t)count };
    if (wprost) SPRAWDZ(clEnqueueMarkerWithWaitList(q, 0, NULL, &evp[2]));
    else SPRAWDZ(clEnqueueNDRangeKernel(q, r->epi, 3, NULL, ge, NULL, 0, NULL, &evp[2]));
    if (do_hosta(xC, zC, bC, C)) return -1;
    if (profil) {
        cl_ulong t0, t1;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof t0, &t0, NULL);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof t1, &t1, NULL);
        ostatni_ms = (t1 - t0) * 1e-6;
    } else ostatni_ms = (teraz() - t_start) * 1e3;
    clReleaseEvent(ev);
    if (getenv("GEN9BLAS_PROFILE")) {
        double tp[3];
        for (int e = 0; e < 3; e++) { cl_ulong a0, a1; clGetEventProfilingInfo(evp[e], CL_PROFILING_COMMAND_START, 8, &a0, NULL);
            clGetEventProfilingInfo(evp[e], CL_PROFILING_COMMAND_END, 8, &a1, NULL); tp[e] = (a1 - a0) * 1e-6; }
        fprintf(stderr, "gen9blas_gemm %dx%dx%d x%d: calosc %.2f ms (bufory %.2f, pak A %.2f, pak B %.2f, kernel %.2f, epilog %.2f), zero-copy %d\n",
                M, N, K, count, (teraz() - t_start) * 1e3, t_buf * 1e3, tp[0], tp[1], ostatni_ms, tp[2], ostatni_zc);
    }
    for (int e = 0; e < 3; e++) clReleaseEvent(evp[e]);
    return 0;
}

// ---------------- ścieżka urządzenia: pakowanie raz, wiele uruchomień kernela na fragmentach ----------------

// Pakuje op(X) do bufora `cel`: rola A (rolaB=0): op(X) R×K, trans: X zapisana K×R; rola B (rolaB=1): op(X) K×R,
// trans: X zapisana R×K. Dopełnienie zerami do Rp × Kp. Pakowanie z bufora urządzenia; dla roli B bez transpozycji można spakować tylko wiersze k w [k0, k0+kn)
// (przesunięcie globalne siatki), np. kolejne bloki rozwiązania w TRSM. kn = 0: całość.
static int pakuj_mem(rodzaj_t *r, int rolaB, int trans, bf_t xX, int ld, int Rr, int K, int Rp, int Kp, cl_mem cel,
                     int k0, int kn) {
    cl_long zero = 0;
    size_t g[3], o[3] = { 0, 0, 0 };
    cl_kernel k = kernel_pak(r, rolaB, trans, Rp, Kp, 1, g);
    if (!rolaB) { ARGB(k, 0, xX); ARG(k, 1, cel); ARG(k, 2, Rr); ARG(k, 3, K); ARG(k, 4, ld); ARG(k, 5, Kp); ARG(k, 6, zero); ARG(k, 7, Rp); }
    else        { ARGB(k, 0, xX); ARG(k, 1, cel); ARG(k, 2, K); ARG(k, 3, Rr); ARG(k, 4, ld); ARG(k, 5, Kp); ARG(k, 6, zero); ARG(k, 7, Rp); }
    if (kn && rolaB && !trans) { g[1] = kn; o[1] = k0; }   // fragment k (B i B_w2: k w g[1])
    SPRAWDZ(clEnqueueNDRangeKernel(q, k, 3, o, g, NULL, 0, NULL, NULL));
    return 0;
}

// Pakuje op(X) z pamięci hosta do bufora `cel` (zob. pakuj_mem).
static int pakuj(rodzaj_t *r, int rolaB, int trans, const void *X, int ld, int Rr, int K, int Rp, int Kp, cl_mem cel) {
    size_t bX = (rolaB ? (trans ? (size_t)(Rr - 1) * ld + K : (size_t)(K - 1) * ld + Rr)
                       : (trans ? (size_t)(K - 1) * ld + Rr : (size_t)(Rr - 1) * ld + K)) * r->es;
    bf_t xX; int z;
    if (obiekt(X, bX, rolaB ? &uB : &uA, 1, 0, &xX, &z)) return -1;
    return pakuj_mem(r, rolaB, trans, xX, ld, Rr, K, Rp, Kp, cel, 0, 0);
}

static cl_mem podbufor(cl_mem b, size_t off, size_t rozm) {
    cl_int err; cl_buffer_region reg = { off, rozm };
    cl_mem s = clCreateSubBuffer(b, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg, &err);
    if (err != CL_SUCCESS) { snprintf(blad, sizeof blad, "clCreateSubBuffer = %d", err); return NULL; }
    return s;
}

// Epilog na buforze urządzenia: Cd = alpha·Cp + beta·Cd (M×N, cały prostokąt).
static int epilog_mem(rodzaj_t *r, cl_mem Cp, int ldp, cl_mem Cd, int ldc, int M, int N, double alpha, double beta) {
    int uplo = 2;
    ARG(r->epi_tri, 0, Cp); ARG(r->epi_tri, 1, ldp); ARG(r->epi_tri, 2, Cd); ARG(r->epi_tri, 3, ldc);
    ARGS(r->epi_tri, 4, alpha); ARGS(r->epi_tri, 5, beta); ARG(r->epi_tri, 6, uplo);
    size_t g[2] = { (size_t)N, (size_t)M };
    SPRAWDZ(clEnqueueNDRangeKernel(q, r->epi_tri, 2, NULL, g, NULL, 0, NULL, NULL));
    return 0;
}

// Kernel GEMM na fragmencie: pod-bufory z przesunięciami (bajty; wyrównane do 128 B). mierz: zdarzenie do zd[].
static int uruchom(rodzaj_t *r, cl_mem Ap, size_t oA, cl_mem Bp, size_t oB, cl_mem Cp, size_t oC,
                   int M, int N, int ldc, int K, int Kfull, int mierz) {
    cl_int err; cl_mem sub[3] = { Ap, Bp, Cp }; size_t off[3] = { oA, oB, oC }; int wl[3] = { 0, 0, 0 };
    for (int i = 0; i < 3; i++) if (off[i]) {
        size_t rozm; clGetMemObjectInfo(sub[i], CL_MEM_SIZE, sizeof rozm, &rozm, NULL);
        cl_buffer_region reg = { off[i], rozm - off[i] };
        sub[i] = clCreateSubBuffer(sub[i], CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg, &err); SPRAWDZ(err); wl[i] = 1;
    }
    ARG(r->main, 0, sub[0]); ARG(r->main, 1, sub[1]); ARG(r->main, 2, sub[2]); ARG(r->main, 3, M); ARG(r->main, 4, N);
    ARG(r->main, 5, ldc); ARG(r->main, 6, K); ARG(r->main, 7, Kfull);
    size_t gm = (size_t)(N / r->tn) * (M / 16) * 16, lm = 16; cl_event ev;
    SPRAWDZ(clEnqueueNDRangeKernel(q, r->main, 1, NULL, &gm, &lm, 0, NULL, &ev));
    if (mierz && nzd < (int)(sizeof zd / sizeof zd[0])) zd[nzd++] = ev; else clReleaseEvent(ev);
    for (int i = 0; i < 3; i++) if (wl[i]) clReleaseMemObject(sub[i]);
    return 0;
}

// Epilog do pamięci hosta: C = alpha·Cp + beta·C na trójkącie uplo (0 dolny, 1 górny, 2 całość).
static int epilog_do(rodzaj_t *r, cl_mem Cp, int ldp, void *C, int M, int N, int ldc, double alpha, double beta, int uplo) {
    size_t bC = ((size_t)(M - 1) * ldc + N) * r->es; bf_t xC; int zC;
    if (obiekt(C, bC, &uC, beta != 0.0f || uplo != 2, 1, &xC, &zC)) return -1;
    ARG(r->epi_tri, 0, Cp); ARG(r->epi_tri, 1, ldp); ARGB(r->epi_tri, 2, xC); ARG(r->epi_tri, 3, ldc);
    ARGS(r->epi_tri, 4, alpha); ARGS(r->epi_tri, 5, beta); ARG(r->epi_tri, 6, uplo);
    size_t g[2] = { (size_t)N, (size_t)M };
    SPRAWDZ(clEnqueueNDRangeKernel(q, r->epi_tri, 2, NULL, g, NULL, 0, NULL, NULL));
    return do_hosta(xC, zC, bC, C);
}

int gen9blas_gemm(int rodzaj, int tA, int tB, int M, int N, int K, double alpha, const void *A, int lda,
             const void *B, int ldb, double beta, void *C, int ldc) {
    return gen9blas_gemm_batched(rodzaj, tA, tB, M, N, K, alpha, A, lda, 0, B, ldb, 0, beta, C, ldc, 0, 1);
}
int gen9blas_hgemm(int M, int N, int K, double alpha, const uint16_t *A, int lda, const uint16_t *B, int ldb,
              double beta, uint16_t *C, int ldc) { return gen9blas_gemm(GEN9BLAS_H, 0, 0, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc); }
int gen9blas_sgemm(int M, int N, int K, double alpha, const float *A, int lda, const float *B, int ldb,
              double beta, float *C, int ldc) { return gen9blas_gemm(GEN9BLAS_S, 0, 0, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc); }

// ---------------- poziom 3 na bazie GEMM ----------------

#define SPR_RODZAJ(rodzaj) do { NOWA_OPERACJA(); if ((rodzaj) < 0 || (rodzaj) > 3 || !R[rodzaj].gotowy) BLAD("rodzaj %d niedostępny", rodzaj); } while (0)
static cl_mem dT, dX, dY;   // robocze: pełna macierz (SYMM, TRMM), X i Y (SYR2K)
static size_t sT, sX, sY;
static int ile_blokow(int n) { const char *e = getenv("GEN9BLAS_BLOCKS"); if (e) return atoi(e); return n >= 512 ? 4 : 1; }

// dT = pełna macierz n×n z trójkąta uplo macierzy A (pamięć hosta), na GPU: sym=1 symetryczna (odbicie),
// sym=0 trójkątna z zerami (diag=1: jedynki); tr=1: wynik transponowany.
static int pelna_dev(rodzaj_t *r, const void *A, int lda, int n, int uplo, int sym, int diag, int tr) {
    size_t bA = ((size_t)(n - 1) * lda + n) * r->es; bf_t xA; int z;
    if (bufor(&dT, &sT, (size_t)n * n * r->es) || obiekt(A, bA, &uA, 1, 0, &xA, &z)) return -1;
    cl_kernel k = r->pel;
    ARGB(k, 0, xA); ARG(k, 1, lda); ARG(k, 2, dT); ARG(k, 3, n); ARG(k, 4, uplo); ARG(k, 5, sym); ARG(k, 6, diag); ARG(k, 7, tr);
    size_t g[2] = { (size_t)(n + 15) / 16 * 16, (size_t)(n + 15) / 16 * 16 }, l[2] = { 16, 16 };
    SPRAWDZ(clEnqueueNDRangeKernel(q, k, 2, NULL, g, l, 0, NULL, NULL));
    return 0;
}

// C = alpha·F·B + beta·C (lewa) albo alpha·B·F + beta·C (prawa), F = dT (n×n, n = M albo N), B i C w pamięci hosta.
// ksztalt F: 0 pełna, 1 dolna, 2 górna -- wtedy p pasów (GEN9BLAS_BLOCKS), każdy tylko po niezerowym zakresie k (~62%
// pracy). C może być tym samym obszarem co B (TRMM, beta = 0): B jest spakowane przed epilogiem (kolejność w kolejce).
static int f_razy_b(rodzaj_t *r, int lewa, int ksztalt, int M, int N, const void *B, int ldb, double alpha, double beta,
                    void *C, int ldc) {
    double t_wej = teraz();
    int es = r->es, n = lewa ? M : N, w;
    int Mp = (M + 63) / 64 * 64, Np = (N + 63) / 64 * 64, Kp = lewa ? Mp : Np;
    if (bufor(&dAp, &sAp, (size_t)Mp * Kp * es + 4096) || bufor(&dBp, &sBp, (size_t)Np * Kp * es + 4096) ||
        bufor(&dCp, &sCp, (size_t)Mp * Np * es))
        return -1;
    if (lewa) w = pakuj_mem(r, 0, 0, BM(dT), n, M, M, Mp, Kp, dAp, 0, 0) || pakuj(r, 1, 0, B, ldb, N, M, Np, Kp, dBp);
    else      w = pakuj(r, 0, 0, B, ldb, M, N, Mp, Kp, dAp) || pakuj_mem(r, 1, 0, BM(dT), n, N, N, Np, Kp, dBp, 0, 0);
    if (w) return -1;
    int p = ksztalt ? ile_blokow(n) : 1, dl = lewa ? Mp : Np, nb = ((dl + p - 1) / p + 63) / 64 * 64;
    nzd = 0;
    for (int i0 = 0; i0 < dl && !w; i0 += nb) {
        int b = dl - i0 < nb ? dl - i0 : nb;
        if (lewa) {   // wiersze i0..i0+b: k w [0, i0+b) (dolna) albo [i0, Kp) (górna)
            int k0 = ksztalt == 2 ? i0 : 0, k1 = ksztalt == 1 ? i0 + b : Kp;
            w = uruchom(r, dAp, ((size_t)i0 * Kp + (size_t)k0 * 16) * es, dBp, (size_t)k0 * r->tn * es, dCp, (size_t)i0 * Np * es,
                        b, Np, Np, k1 - k0, Kp, 1);
        } else {      // kolumny i0..i0+b: k w [i0, Kp) (dolna) albo [0, i0+b) (górna)
            int k0 = ksztalt == 1 ? i0 : 0, k1 = ksztalt == 2 ? i0 + b : Kp;
            w = uruchom(r, dAp, (size_t)k0 * 16 * es, dBp, ((size_t)i0 * Kp + (size_t)k0 * r->tn) * es, dCp, (size_t)i0 * es,
                        Mp, b, Np, k1 - k0, Kp, 1);
        }
    }
    if (w || epilog_do(r, dCp, Np, C, M, N, ldc, alpha, beta, 2)) return -1;
    { double tz = czasy_zdarzen(); ostatni_ms = tz >= 0 ? tz : (teraz() - t_wej) * 1e3; }
    return 0;
}

int gen9blas_symm(int rodzaj, int side, int uplo, int M, int N, double alpha, const void *A, int lda,
             const void *B, int ldb, double beta, void *C, int ldc) {
    SPR_RODZAJ(rodzaj);
    rodzaj_t *r = &R[rodzaj]; int lewa = side == GEN9BLAS_LEFT;
    if (pelna_dev(r, A, lda, lewa ? M : N, uplo, 1, 0, 0)) return -1;
    return f_razy_b(r, lewa, 0, M, N, B, ldb, alpha, beta, C, ldc);
}

int gen9blas_trmm(int rodzaj, int side, int uplo, int transA, int diag, int M, int N, double alpha,
             const void *A, int lda, void *B, int ldb) {
    // B = alpha·op(A)·B (lewa) albo alpha·B·op(A) (prawa). F = op(A) jawnie (dolna, gdy uplo=L xor transA).
    SPR_RODZAJ(rodzaj);
    rodzaj_t *r = &R[rodzaj]; int lewa = side == GEN9BLAS_LEFT, dolna = (uplo == GEN9BLAS_LOWER) != (transA != 0);
    if (pelna_dev(r, A, lda, lewa ? M : N, uplo, 0, diag, transA)) return -1;
    return f_razy_b(r, lewa, dolna ? 1 : 2, M, N, B, ldb, alpha, 0.0f, B, ldb);
}

// Trójkąt (uplo) z alpha·op(X)·op(Y)ᵀ + beta·C, op(X), op(Y): N×K (tX: X, Y zapisane K×N), X i Y na GPU.
// Najpierw pakowanie (trojkat_pakuj), potem p uruchomień kernela
// na pasach wierszy: dolny trójkąt -> kolumny 0..i0+b, górny -> i0..N (~(p+1)/(2p) pracy), epilog na trójkącie.
static int trojkat_pakuj(rodzaj_t *r, int tX, int N, int K, bf_t X, int ldx, bf_t Y, int ldy) {
    int es = r->es, Np = (N + 63) / 64 * 64, Kp = (K + r->km - 1) / r->km * r->km;
    if (bufor(&dAp, &sAp, (size_t)Np * Kp * es + 4096) || bufor(&dBp, &sBp, (size_t)Np * Kp * es + 4096) ||
        bufor(&dCp, &sCp, (size_t)Np * Np * es))
        return -1;
    return pakuj_mem(r, 0, tX, X, ldx, N, K, Np, Kp, dAp, 0, 0) || pakuj_mem(r, 1, !tX, Y, ldy, N, K, Np, Kp, dBp, 0, 0) ? -1 : 0;
}
static int trojkat_licz(rodzaj_t *r, int uplo, int N, int K, double alpha, double beta, void *C, int ldc) {
    double t_wej = teraz();
    int es = r->es, Np = (N + 63) / 64 * 64, Kp = (K + r->km - 1) / r->km * r->km;
    int p = ile_blokow(N), nb = ((Np + p - 1) / p + 63) / 64 * 64;
    nzd = 0;
    for (int i0 = 0; i0 < Np; i0 += nb) {
        int b = Np - i0 < nb ? Np - i0 : nb;
        int c0 = uplo == GEN9BLAS_LOWER ? 0 : i0, c1 = uplo == GEN9BLAS_LOWER ? i0 + b : Np;
        if (uruchom(r, dAp, (size_t)i0 * Kp * es, dBp, (size_t)c0 * Kp * es, dCp, ((size_t)i0 * Np + c0) * es,
                    b, c1 - c0, Np, Kp, Kp, 1)) return -1;
    }
    if (epilog_do(r, dCp, Np, C, N, N, ldc, alpha, beta, uplo)) return -1;
    { double tz = czasy_zdarzen(); ostatni_ms = tz >= 0 ? tz : (teraz() - t_wej) * 1e3; }
    return 0;
}

int gen9blas_syrk(int rodzaj, int uplo, int trans, int N, int K, double alpha, const void *A, int lda,
             double beta, void *C, int ldc) {
    SPR_RODZAJ(rodzaj);
    rodzaj_t *r = &R[rodzaj];
    size_t bA = (trans ? (size_t)(K - 1) * lda + N : (size_t)(N - 1) * lda + K) * r->es; bf_t xA; int z;
    if (obiekt(A, bA, &uA, 1, 0, &xA, &z) || trojkat_pakuj(r, trans, N, K, xA, lda, xA, lda)) return -1;
    return trojkat_licz(r, uplo, N, K, alpha, beta, C, ldc);
}

int gen9blas_syr2k(int rodzaj, int uplo, int trans, int N, int K, double alpha, const void *A, int lda,
              const void *B, int ldb, double beta, void *C, int ldc) {
    // alpha·(op(A)·op(B)ᵀ + op(B)·op(A)ᵀ) = alpha·X·Yᵀ, X = [op(A) | op(B)], Y = [op(B) | op(A)] (N × 2K, sklejane na GPU)
    SPR_RODZAJ(rodzaj);
    rodzaj_t *r = &R[rodzaj]; int es = r->es, K2 = 2 * K;
    size_t bA = (trans ? (size_t)(K - 1) * lda + N : (size_t)(N - 1) * lda + K) * es;
    size_t bB = (trans ? (size_t)(K - 1) * ldb + N : (size_t)(N - 1) * ldb + K) * es;
    int zA, zB; bf_t xA, xB;
    if (bufor(&dX, &sX, (size_t)N * K2 * es) || bufor(&dY, &sY, (size_t)N * K2 * es) ||
        obiekt(A, bA, &uA, 1, 0, &xA, &zA) || obiekt(B, bB, &uB, 1, 0, &xB, &zB))
        return -1;
    cl_kernel k = r->skl;
    ARGB(k, 0, xA); ARG(k, 1, lda); ARGB(k, 2, xB); ARG(k, 3, ldb); ARG(k, 4, dX); ARG(k, 5, dY); ARG(k, 6, K); ARG(k, 7, trans);
    size_t g[2] = { (size_t)K, (size_t)N };
    SPRAWDZ(clEnqueueNDRangeKernel(q, k, 2, NULL, g, NULL, 0, NULL, NULL));
    if (trojkat_pakuj(r, 0, N, K2, BM(dX), K2, BM(dY), K2)) return -1;
    return trojkat_licz(r, uplo, N, K2, alpha, beta, C, ldc);
}

int gen9blas_trsm(int rodzaj, int side, int uplo, int transA, int diag, int M, int N, double alpha,
             const void *A, int lda, void *B, int ldb) {
    // Rozwiązuje op(A)·X = alpha·B (lewa) albo X·op(A) = alpha·B (prawa); X nadpisuje B. FP32 i FP16 (GEN9BLAS_H, GEN9BLAS_HS).
    // Postać L·Y = alpha·R z L dolną (górną sprowadza odwrócenie kolejności indeksów, stronę prawą transpozycja).
    // Wszystko na GPU: budowa L i Y z A, B (trsm_L/trsm_Y), odwrotności bloków diagonalnych L w FP64 (trsm_inv),
    // L pakowane raz, pętla blokowa (nb = 256): Y_i -= L[i, <i]·Y[<i]; Y_i = inv(L_ii)·Y_i; dopakowanie Y_i;
    // na końcu zapis do B (trsm_zapisz).
    SPR_RODZAJ(rodzaj);
    rodzaj_t *r = &R[rodzaj]; int es = r->es;
    int lewa = side == GEN9BLAS_LEFT, n = lewa ? M : N, m = lewa ? N : M;
    int tr = lewa ? transA : !transA, dolna = (uplo == GEN9BLAS_LOWER) != tr, odwr = !dolna;
    const int nb = 256;
    int np = (n + 63) / 64 * 64, mp = (m + 63) / 64 * 64, w = 0, nbl = (n + nb - 1) / nb;
    double tp0 = teraz();
    // A i B dla GPU (B: wejście i wyjście)
    size_t bA = ((size_t)(n - 1) * lda + n) * es, bB = ((size_t)(M - 1) * ldb + N) * es;
    bf_t xA, xB; int zA, zB; cl_event evi = NULL, evi2 = NULL;
    if (obiekt(A, bA, &uA, 1, 0, &xA, &zA) || obiekt(B, bB, &uB, 1, 1, &xB, &zB)) return -1;
    // bufory robocze TRSM (trzymane między wywołaniami): L, Y, Y spakowane, inv(L_ii) spakowane, R_i spakowane,
    // odwrotności bloków (typ elementu), X i Tm (FP64, odwracanie)
    static cl_mem Ldv, Yd, Yp, Ip, Rp, Inv, Xd, Tm;
    static size_t sLdv, sYd, sYp, sIp, sRp, sInv, sXd, sTm;
    size_t blk = (size_t)nbl * nb * nb;
    if (bufor(&Ldv, &sLdv, (size_t)n * n * es) || bufor(&Yd, &sYd, (size_t)np * mp * es) ||
        bufor(&Yp, &sYp, (size_t)mp * np * es + 4096) || bufor(&Ip, &sIp, (size_t)nb * nb * es + 4096) ||
        bufor(&Rp, &sRp, (size_t)mp * nb * es + 4096) || bufor(&Inv, &sInv, blk * es) ||
        bufor(&Xd, &sXd, blk * sizeof(double)) || bufor(&Tm, &sTm, blk * sizeof(double)))
        return -1;
    if (bufor(&dAp, &sAp, (size_t)np * np * es + 4096) || bufor(&dCp, &sCp, (size_t)nb * mp * es)) w = -1;
    if (!w) {   // L = f(A), Y = alpha·R(B), odwrotności bloków L; L pakowane raz
        cl_kernel k = r->tL; int u = uplo;
        ARGB(k, 0, xA); ARG(k, 1, lda); ARG(k, 2, Ldv); ARG(k, 3, n); ARG(k, 4, u); ARG(k, 5, tr); ARG(k, 6, diag); ARG(k, 7, odwr);
        size_t g[2] = { (size_t)n, (size_t)n };
        SPRAWDZ(clEnqueueNDRangeKernel(q, k, 2, NULL, g, NULL, 0, NULL, NULL));
        k = r->tY;
        ARGB(k, 0, xB); ARG(k, 1, ldb); ARG(k, 2, Yd); ARG(k, 3, n); ARG(k, 4, m); ARG(k, 5, mp); ARG(k, 6, lewa); ARG(k, 7, odwr); ARGS(k, 8, alpha);
        size_t gy[2] = { (size_t)mp, (size_t)np };
        SPRAWDZ(clEnqueueNDRangeKernel(q, k, 2, NULL, gy, NULL, 0, NULL, NULL));
        // odwrotności: bloki bazowe 32×32, potem poziomy h = 32..nb/2: X21 = −X22·(L21·X11), na końcu do typu elementu
        k = r->tI[0];
        ARG(k, 0, Ldv); ARG(k, 1, n); ARG(k, 2, nb); ARG(k, 3, Xd);
        size_t g0[2] = { 32, (size_t)nbl * nb / 32 };
        SPRAWDZ(clEnqueueNDRangeKernel(q, k, 2, NULL, g0, NULL, 0, NULL, &evi));
        for (int h = 32; h < nb; h *= 2)
            for (int krok = 1; krok <= 2; krok++) {
                k = r->tI[krok];
                ARG(k, 0, Ldv); ARG(k, 1, n); ARG(k, 2, nb); ARG(k, 3, Xd); ARG(k, 4, Tm); ARG(k, 5, h);
                size_t gh[3] = { (size_t)h, (size_t)h, (size_t)nbl * nb / (2 * h) };
                SPRAWDZ(clEnqueueNDRangeKernel(q, k, 3, NULL, gh, NULL, 0, NULL, NULL));
            }
        k = r->tI[3];
        ARG(k, 0, n); ARG(k, 1, nb); ARG(k, 2, Xd); ARG(k, 3, Inv);
        size_t g3[3] = { (size_t)nb, (size_t)nb, (size_t)nbl };
        SPRAWDZ(clEnqueueNDRangeKernel(q, k, 3, NULL, g3, NULL, 0, NULL, &evi2));
        w = pakuj_mem(r, 0, 0, BM(Ldv), n, n, n, np, np, dAp, 0, 0);
    }
    for (int i0 = 0; i0 < n && !w; i0 += nb) {
        int b = n - i0 < nb ? n - i0 : nb, bp = (b + 63) / 64 * 64;
        cl_mem Yi = podbufor(Yd, (size_t)i0 * mp * es, (size_t)bp * mp * es);
        cl_mem Ii = podbufor(Inv, (size_t)i0 * nb * es, (size_t)b * b * es);
        if (!Yi || !Ii) { w = -1; if (Yi) clReleaseMemObject(Yi); break; }
        if (i0 > 0)   // Y_i -= L[i0:i0+b, 0:i0] · Y[0:i0]
            w = uruchom(r, dAp, (size_t)i0 * np * es, Yp, 0, dCp, 0, bp, mp, mp, i0, np, 0) ||
                epilog_mem(r, dCp, mp, Yi, mp, b, m, -1.0, 1.0);
        if (!w)       // Y_i = inv(L_ii) · Y_i
            w = pakuj_mem(r, 0, 0, BM(Ii), b, b, b, bp, bp, Ip, 0, 0) ||
                pakuj_mem(r, 1, 0, BM(Yi), mp, m, b, mp, bp, Rp, 0, 0) ||
                uruchom(r, Ip, 0, Rp, 0, dCp, 0, bp, mp, mp, bp, bp, 0) ||
                epilog_mem(r, dCp, mp, Yi, mp, b, m, 1.0, 0.0);
        if (!w)       // dopakowanie rozwiązanego bloku (wiersze k = i0..i0+b w Yp)
            w = pakuj_mem(r, 1, 0, BM(Yd), mp, m, n, mp, np, Yp, i0, bp);
        clReleaseMemObject(Yi); clReleaseMemObject(Ii);
    }
    if (!w) {         // zapis do B na GPU, potem do pamięci hosta
        cl_kernel k = r->tZ;
        ARG(k, 0, Yd); ARG(k, 1, mp); ARGB(k, 2, xB); ARG(k, 3, ldb); ARG(k, 4, n); ARG(k, 5, lewa); ARG(k, 6, odwr);
        size_t g[2] = { (size_t)m, (size_t)n };
        SPRAWDZ(clEnqueueNDRangeKernel(q, k, 2, NULL, g, NULL, 0, NULL, NULL));
        w = do_hosta(xB, zB, bB, B);
    }
    ostatni_ms = (teraz() - tp0) * 1e3;   // TRSM: czas całego wywołania
    if (evi && evi2) {
        if (getenv("GEN9BLAS_PROFILE")) { cl_ulong t0, t1; clGetEventProfilingInfo(evi, CL_PROFILING_COMMAND_START, 8, &t0, NULL);
            clGetEventProfilingInfo(evi2, CL_PROFILING_COMMAND_END, 8, &t1, NULL);
            fprintf(stderr, "trsm n=%d m=%d: %.1f ms (odwrotnosci blokow %.1f ms)\n", n, m, ostatni_ms, (t1 - t0) * 1e-6); }
        clReleaseEvent(evi); clReleaseEvent(evi2);
    }
    return w;
}

int gen9blas_available(int rodzaj) { return rodzaj >= 0 && rodzaj <= 3 && R[rodzaj].gotowy; }
double gen9blas_last_time_ms(void) { return ostatni_ms; }
int gen9blas_last_zero_copy(void) { return ostatni_zc; }
const char *gen9blas_error(void) { return blad; }

// ---------------- poziomy 1–2 (pamięć hosta: USM bez kopii albo kopia przez bufor roboczy, zob. obiekt()) ----------------
static cl_mem dP;        // sumy częściowe GEMV T
static size_t sP;
static double *wyn;      // wyniki redukcji (USM host: GPU pisze, CPU czyta po czekaj())
#define GRUPY 96         // grup roboczych (po 256) w redukcjach: 24 EU × 4

static int l12_rodzaj(int rodzaj, l12_t **k, int *es) {
    NOWA_OPERACJA();
    int i = rodzaj == GEN9BLAS_S ? 1 : rodzaj == GEN9BLAS_D ? 2 : rodzaj == GEN9BLAS_H || rodzaj == GEN9BLAS_HS ? 0 : -1;
    if (i < 0 || !L12[i].axpy) BLAD("rodzaj %d niedostępny", rodzaj);
    *k = &L12[i]; *es = L12[i].es; return 0;
}
// Skalar dla kernela w typie obliczeń R (double dla FP64, float dla FP16/FP32).
static cl_int argr(cl_kernel K, int i, double v, int es) {
    if (es == 8) return clSetKernelArg(K, i, sizeof v, &v);
    float f = (float)v; return clSetKernelArg(K, i, sizeof f, &f);
}
#define ARGR(K, i, v) SPRAWDZ(argr(K, i, v, k->es))
#define RS(k) ((k)->es == 8 ? sizeof(double) : sizeof(float))   // rozmiar typu obliczeń
// Operand: obszar [p, p+b) pamięci hosta; g -- wskaźnik dla kernela (p albo bufor roboczy), zc: bez kopii.
typedef struct { const void *p; size_t b; bf_t g; int zc; } op_t;
static rob_t *const robl12[3] = { &uA, &uB, &uC };
static int operandy(op_t *w, int n, const int *kopiuj, const int *pisany) {
    for (int i = 0; i < n; i++)
        if (obiekt(w[i].p, w[i].b, robl12[i], kopiuj[i], pisany ? pisany[i] : 0, &w[i].g, &w[i].zc)) return -1;
    return 0;
}
// Obszar wektora: n elementów co inc. inc < 0 jak w BLAS: wskaźnik na najniższy adres, element 0 na końcu.
static size_t wb(int n, int inc, int es) { return ((size_t)(n - 1) * (size_t)(inc < 0 ? -inc : inc) + 1) * es; }
static cl_long wo(int n, int inc) { return inc < 0 ? (cl_long)(n - 1) * -inc : 0; }
// Koniec: czekanie, kopia zapisanych operandów spoza USM do pamięci hosta.
static int zakoncz(op_t *w, int n, const int *pisany) {
    if (czekaj()) return -1;
    for (int i = 0; i < n; i++) if (pisany[i] && !w[i].zc) memcpy((void *)w[i].p, w[i].g.p, w[i].b);
    return 0;
}
static int start1(cl_kernel k, int n);
// Kernele element po elemencie (axpy, scal, copy, rot, rotm): work-item = 4 elementy.
static int start4(cl_kernel k, int n) { return start1(k, (n + 3) / 4); }
static int start1(cl_kernel k, int n) {
    size_t g = ((size_t)n + 255) / 256 * 256, l = 256;
    SPRAWDZ(clEnqueueNDRangeKernel(q, k, 1, NULL, &g, &l, 0, NULL, NULL)); return 0;
}
#define POCZ(n, ...) l12_t *k; int es; if (l12_rodzaj(rodzaj, &k, &es)) return -1; if ((n) <= 0) return 0; \
                     if (__VA_ARGS__) BLAD("krok wektora 0")

int gen9blas_axpy(int rodzaj, int n, double a, const void *x, int incx, void *y, int incy) {
    POCZ(n, !incx || !incy);
    op_t w[2] = { { x, wb(n, incx, es) }, { y, wb(n, incy, es) } }; const int we[2] = { 1, 1 }, wy[2] = { 0, 1 };
    if (operandy(w, 2, we, wy)) return -1;
    cl_long ox = wo(n, incx), oy = wo(n, incy), ix = incx, iy = incy; cl_kernel K = k->axpy;
    ARG(K, 0, n); ARGR(K, 1, a); ARGB(K, 2, w[0].g); ARG(K, 3, ox); ARG(K, 4, ix); ARGB(K, 5, w[1].g); ARG(K, 6, oy); ARG(K, 7, iy);
    return start4(K, n) || zakoncz(w, 2, wy) ? -1 : 0;
}

int gen9blas_scal(int rodzaj, int n, double a, void *x, int incx) {
    POCZ(n, !incx);
    op_t w[1] = { { x, wb(n, incx, es) } }; const int we[1] = { 1 };
    if (operandy(w, 1, we, we)) return -1;
    cl_long ox = wo(n, incx), ix = incx; cl_kernel K = k->scal;
    ARG(K, 0, n); ARGR(K, 1, a); ARGB(K, 2, w[0].g); ARG(K, 3, ox); ARG(K, 4, ix);
    return start4(K, n) || zakoncz(w, 1, we) ? -1 : 0;
}

int gen9blas_copy(int rodzaj, int n, const void *x, int incx, void *y, int incy) {
    POCZ(n, !incx || !incy);
    // y kopiowane do bufora roboczego także przy kopii: przy kroku > 1 luki między elementami muszą zostać bez zmian
    op_t w[2] = { { x, wb(n, incx, es) }, { y, wb(n, incy, es) } }; const int we[2] = { 1, incy != 1 && incy != -1 }, wy[2] = { 0, 1 };
    if (operandy(w, 2, we, wy)) return -1;
    cl_long ox = wo(n, incx), oy = wo(n, incy), ix = incx, iy = incy;
    if (incx == 1 && incy == 1) {   // 16 B na lane w każdym typie (copy16): FP64 17 -> 24 GB/s
        // FP16: kompilator robi z vload8/vstore8 half SIMD8 i bajtowe odczyty po 4 B (16 GB/s) -- kopiowanie nie zależy
        // od typu, więc parzysta liczba elementów wyrównanych do 4 B idzie jądrem FP32 (n/2 słów 32-bitowych).
        l12_t *kk = k; int nn = n, e = es;
        if (es == 2 && n % 2 == 0 && ((uintptr_t)x | (uintptr_t)y) % 4 == 0) { kk = &L12[1]; nn = n / 2; e = 4; }
        cl_kernel K = kk->copy16; const int gw = e == 2 ? 4 : 16 / e;   // = GW w level12.cl (FP16: 4, FP32: 4, FP64: 2)
        ARG(K, 0, nn); ARGB(K, 1, w[0].g); ARG(K, 2, ox); ARGB(K, 3, w[1].g); ARG(K, 4, oy);
        return start1(K, (nn + gw - 1) / gw) || zakoncz(w, 2, wy) ? -1 : 0;
    }
    cl_kernel K = k->copy;
    ARG(K, 0, n); ARGB(K, 1, w[0].g); ARG(K, 2, ox); ARG(K, 3, ix); ARGB(K, 4, w[1].g); ARG(K, 5, oy); ARG(K, 6, iy);
    return start4(K, n) || zakoncz(w, 2, wy) ? -1 : 0;
}

static int rot(int rodzaj, int n, void *x, int incx, void *y, int incy, double c, double s, int zamiana) {
    POCZ(n, !incx || !incy);
    op_t w[2] = { { x, wb(n, incx, es) }, { y, wb(n, incy, es) } }; const int we[2] = { 1, 1 };
    if (operandy(w, 2, we, we)) return -1;
    cl_long ox = wo(n, incx), oy = wo(n, incy), ix = incx, iy = incy; cl_kernel K = k->rot;
    ARG(K, 0, n); ARGB(K, 1, w[0].g); ARG(K, 2, ox); ARG(K, 3, ix); ARGB(K, 4, w[1].g); ARG(K, 5, oy); ARG(K, 6, iy);
    ARGR(K, 7, c); ARGR(K, 8, s); ARG(K, 9, zamiana);
    return start4(K, n) || zakoncz(w, 2, we) ? -1 : 0;
}
// ROTM: H z param (BLAS): param[0] = flaga: -1 pełna [h11 h12; h21 h22], 0 [1 h12; h21 1], 1 [h11 1; -1 h22],
// -2 jednostkowa (nic do zrobienia); param[1..4] = h11, h21, h12, h22.
int gen9blas_rotm(int rodzaj, int n, void *x, int incx, void *y, int incy, const double *param) {
    double fl = param[0], h11, h12, h21, h22;
    if (fl == -2.0) return 0;
    if (fl < 0) { h11 = param[1]; h21 = param[2]; h12 = param[3]; h22 = param[4]; }
    else if (fl == 0) { h11 = 1; h21 = param[2]; h12 = param[3]; h22 = 1; }
    else { h11 = param[1]; h21 = -1; h12 = 1; h22 = param[4]; }
    POCZ(n, !incx || !incy);
    op_t w[2] = { { x, wb(n, incx, es) }, { y, wb(n, incy, es) } }; const int we[2] = { 1, 1 };
    if (operandy(w, 2, we, we)) return -1;
    cl_long ox = wo(n, incx), oy = wo(n, incy), ix = incx, iy = incy; cl_kernel K = k->rotm;
    ARG(K, 0, n); ARGB(K, 1, w[0].g); ARG(K, 2, ox); ARG(K, 3, ix); ARGB(K, 4, w[1].g); ARG(K, 5, oy); ARG(K, 6, iy);
    ARGR(K, 7, h11); ARGR(K, 8, h12); ARGR(K, 9, h21); ARGR(K, 10, h22);
    return start4(K, n) || zakoncz(w, 2, we) ? -1 : 0;
}

// ROTG i ROTMG: obliczenia na kilku liczbach -- na CPU, algorytmy referencyjne (Netlib BLAS 3.10+).
// ROTG: [c s; -s c]·[a; b] = [r; 0]; *a = r, *b = z (zapis c, s w jednej liczbie), odporne na przepełnienie.
void gen9blas_rotg(float *a, float *b, float *c, float *s) {
    const float safmin = 1.17549435e-38f, safmax = 1.0f / safmin;
    float an = fabsf(*a), bn = fabsf(*b);
    if (bn == 0) { *c = 1; *s = 0; *b = 0; return; }
    if (an == 0) { *c = 0; *s = 1; *a = *b; *b = 1; return; }
    float scl = fminf(safmax, fmaxf(safmin, fmaxf(an, bn)));
    float sigma = an > bn ? copysignf(1, *a) : copysignf(1, *b);
    float r = sigma * scl * sqrtf((*a / scl) * (*a / scl) + (*b / scl) * (*b / scl));
    *c = *a / r; *s = *b / r;
    float z = an > bn ? *s : *c != 0 ? 1 / *c : 1;
    *a = r; *b = z;
}
// ROTMG: zmodyfikowany obrót Givensa (d1, d2 -- kwadraty skal, x1, y1); wynik w param (zob. gen9blas_rotm).
void gen9blas_rotmg(float *d1, float *d2, float *x1, float y1, float *param) {
    const float gam = 4096.0f, gamsq = 16777216.0f, rgamsq = 5.9604645e-8f;
    float fl, h11 = 0, h12 = 0, h21 = 0, h22 = 0;
    if (*d1 < 0) { fl = -1; *d1 = *d2 = *x1 = 0; }
    else {
        float p2 = *d2 * y1;
        if (p2 == 0) { param[0] = -2; return; }
        float p1 = *d1 * *x1, q2 = p2 * y1, q1 = p1 * *x1;
        if (fabsf(q1) > fabsf(q2)) {
            h21 = -y1 / *x1; h12 = p2 / p1;
            float u = 1 - h12 * h21;
            if (u > 0) { fl = 0; *d1 /= u; *d2 /= u; *x1 *= u; }
            else { fl = -1; h11 = h12 = h21 = h22 = 0; *d1 = *d2 = *x1 = 0; }
        } else if (q2 < 0) { fl = -1; h11 = h12 = h21 = h22 = 0; *d1 = *d2 = *x1 = 0; }
        else {
            fl = 1; h11 = p1 / p2; h22 = *x1 / y1;
            float u = 1 + h11 * h22, t = *d2 / u;
            *d2 = *d1 / u; *d1 = t; *x1 = y1 * u;
        }
        if (*d1 != 0)
            while (*d1 <= rgamsq || *d1 >= gamsq) {
                if (fl == 0) { h11 = 1; h22 = 1; } else if (fl > 0) { h21 = -1; h12 = 1; }
                fl = -1;
                if (*d1 <= rgamsq) { *d1 *= gam * gam; *x1 /= gam; h11 /= gam; h12 /= gam; }
                else { *d1 /= gam * gam; *x1 *= gam; h11 *= gam; h12 *= gam; }
            }
        if (*d2 != 0)
            while (fabsf(*d2) <= rgamsq || fabsf(*d2) >= gamsq) {
                if (fl == 0) { h11 = 1; h22 = 1; } else if (fl > 0) { h21 = -1; h12 = 1; }
                fl = -1;
                if (fabsf(*d2) <= rgamsq) { *d2 *= gam * gam; h21 /= gam; h22 /= gam; }
                else { *d2 /= gam * gam; h21 *= gam; h22 *= gam; }
            }
    }
    if (fl < 0) { param[1] = h11; param[2] = h21; param[3] = h12; param[4] = h22; }
    else if (fl == 0) { param[2] = h21; param[3] = h12; }
    else { param[1] = h11; param[4] = h22; }
    param[0] = fl;
}

int gen9blas_swap(int rodzaj, int n, void *x, int incx, void *y, int incy) { return rot(rodzaj, n, x, incx, y, incy, 1, 0, 1); }
int gen9blas_rot(int rodzaj, int n, void *x, int incx, void *y, int incy, double c, double s) { return rot(rodzaj, n, x, incx, y, incy, c, s, 0); }

static int wyniki(void) {
    if (!wyn && !(wyn = gen9blas_alloc(2 * GRUPY * sizeof(double)))) return -1;
    return 0;
}
static int redukcja(int rodzaj, int tryb, int n, const void *x, int incx, const void *y, int incy, double *wynik) {
    *wynik = 0;
    POCZ(n, !incx || (tryb == 0 && !incy));
    op_t w[2] = { { x, wb(n, incx, es) }, { y, tryb == 0 ? wb(n, incy, es) : 0 } }; const int we[2] = { 1, 1 };
    if (operandy(w, tryb == 0 ? 2 : 1, we, NULL) || wyniki()) return -1;
    if (tryb) { w[1] = w[0]; incy = incx; }
    int g = (n + 255) / 256 < GRUPY ? (n + 255) / 256 : GRUPY;
    cl_long ox = wo(n, incx), oy = wo(n, incy), ix = incx, iy = incy; cl_kernel K = k->red;
    ARG(K, 0, n); ARG(K, 1, tryb); ARGB(K, 2, w[0].g); ARG(K, 3, ox); ARG(K, 4, ix); ARGB(K, 5, w[1].g); ARG(K, 6, oy); ARG(K, 7, iy);
    SPRAWDZ(usm_arg(K, 8, wyn));
    size_t G = (size_t)g * 256, Lw = 256;
    SPRAWDZ(clEnqueueNDRangeKernel(q, K, 1, NULL, &G, &Lw, 0, NULL, NULL));
    if (czekaj()) return -1;
    double s = 0; for (int i = 0; i < g; i++) s += wyn[i];
    *wynik = tryb == 2 ? sqrt(s) : s;
    return 0;
}
int gen9blas_dot(int rodzaj, int n, const void *x, int incx, const void *y, int incy, double *wynik) { return redukcja(rodzaj, 0, n, x, incx, y, incy, wynik); }
int gen9blas_asum(int rodzaj, int n, const void *x, int incx, double *wynik) { return redukcja(rodzaj, 1, n, x, incx, NULL, 0, wynik); }
int gen9blas_nrm2(int rodzaj, int n, const void *x, int incx, double *wynik) { return redukcja(rodzaj, 2, n, x, incx, NULL, 0, wynik); }

int gen9blas_iamax(int rodzaj, int n, const void *x, int incx, long *wynik) {
    *wynik = -1;
    POCZ(n, !incx);
    op_t w[1] = { { x, wb(n, incx, es) } }; const int we[1] = { 1 };
    if (operandy(w, 1, we, NULL) || wyniki()) return -1;
    int g = (n + 255) / 256 < GRUPY ? (n + 255) / 256 : GRUPY;
    cl_long ox = wo(n, incx), ix = incx; cl_kernel K = k->iamax;
    ARG(K, 0, n); ARGB(K, 1, w[0].g); ARG(K, 2, ox); ARG(K, 3, ix); SPRAWDZ(usm_arg(K, 4, wyn));
    size_t G = (size_t)g * 256, Lw = 256;
    SPRAWDZ(clEnqueueNDRangeKernel(q, K, 1, NULL, &G, &Lw, 0, NULL, NULL));
    if (czekaj()) return -1;
    // wartości w typie obliczeń (float albo double), indeksy int za 96 wartościami
    const int *ci = (const int *)((const char *)wyn + GRUPY * RS(k));
    double cv[GRUPY];
    for (int i = 0; i < g; i++) cv[i] = k->es == 8 ? ((const double *)wyn)[i] : ((const float *)wyn)[i];
    int b = 0;
    for (int i = 1; i < g; i++) if (cv[i] > cv[b] || (cv[i] == cv[b] && ci[i] < ci[b])) b = i;
    *wynik = ci[b] == 0x7fffffff ? 0 : ci[b];   // same NaN: indeks 0
    return 0;
}

// GEMV na buforach (bez czekania): y = alpha·op(A)·x + beta·y z maską trójkąta tryb (0 całość, 1 j <= i, 2 j >= i,
// 3 j < i, 4 j > i dla elementu A[i][j]); jedn = 1: do sumy dochodzi x z indeksem przekątnej (jedynki na przekątnej).
static int gemv_wew(l12_t *k, int trans, int M, int N, double alpha, bf_t A, int lda, bf_t x, cl_long ox, cl_long ix,
                    double beta, bf_t y, cl_long oy, cl_long iy, int tryb, int jedn) {
    cl_long oa = 0;
    if (!trans) {
        // Wiersze od 1024 kolumn: wariant grupowy (16 subgrup na te same 4 wiersze, mało stron naraz). Na zwykłej pamięci
        // (userptr, strony 4 KB) dawny tracił 2× (n = 4096: 10,5 -> 18,9 GB/s), na USM -5..+20%; przy N <= 512
        // grupowy wolniejszy (za mało kolumn na 16 subgrup). Pomiar: bench/gemv_n.py, results/gemv-n-2026-11-12.txt.
        // GEN9BLAS_GEMV_N=stary|grupowy wymusza wariant (porównania).
        static int wym = -1;
        if (wym < 0) { const char *e = getenv("GEN9BLAS_GEMV_N"); wym = !e ? 0 : !strcmp(e, "stary") ? 1 : !strcmp(e, "grupowy") ? 2 : 0; }
        const int grupowy = wym == 2 || (wym == 0 && N >= 1024);
        cl_kernel K = grupowy ? k->gemv_nw : k->gemv_n;
        ARG(K, 0, M); ARG(K, 1, N); ARGR(K, 2, alpha); ARGB(K, 3, A); ARG(K, 4, oa); ARG(K, 5, lda); ARGB(K, 6, x);
        ARG(K, 7, ox); ARG(K, 8, ix); ARGR(K, 9, beta); ARGB(K, 10, y); ARG(K, 11, oy); ARG(K, 12, iy); ARG(K, 13, tryb); ARG(K, 14, jedn);
        size_t g = ((size_t)(M + 3) / 4 * 16 + 63) / 64 * 64, l = 64;
        if (grupowy) { l = 256; g = (size_t)(M + 3) / 4 * 256; }
        SPRAWDZ(clEnqueueNDRangeKernel(q, K, 1, NULL, &g, &l, 0, NULL, NULL));
        return 0;
    }
    int kol4 = (N + 3) / 4, cz = (16384 + kol4 - 1) / kol4;   // ~16 tys. work-itemów
    if (cz > M) cz = M;
    int wpc = (M + cz - 1) / cz; cz = (M + wpc - 1) / wpc;
    if (bufor(&dP, &sP, (size_t)cz * N * RS(k))) return -1;
    cl_kernel K;
    if (k->es == 2 && !tryb && !jedn) {   // FP16, pełna macierz: odczyty blokowe (gemv_t1b; results/komunikat-16B-...)
        K = k->gemv_t1b;
        ARG(K, 0, M); ARG(K, 1, N); ARG(K, 2, wpc); ARGB(K, 3, A); ARG(K, 4, oa); ARG(K, 5, lda); ARGB(K, 6, x);
        ARG(K, 7, ox); ARG(K, 8, ix); ARG(K, 9, dP);
    } else {
        K = k->gemv_t1;
        ARG(K, 0, M); ARG(K, 1, N); ARG(K, 2, wpc); ARGB(K, 3, A); ARG(K, 4, oa); ARG(K, 5, lda); ARGB(K, 6, x);
        ARG(K, 7, ox); ARG(K, 8, ix); ARG(K, 9, dP); ARG(K, 10, tryb); ARG(K, 11, jedn);
    }
    size_t g[2] = { ((size_t)kol4 + 63) / 64 * 64, (size_t)cz }, l[2] = { 64, 1 };
    SPRAWDZ(clEnqueueNDRangeKernel(q, K, 2, NULL, g, l, 0, NULL, NULL));
    K = k->gemv_t2;
    ARG(K, 0, N); ARG(K, 1, cz); ARG(K, 2, dP); ARGR(K, 3, alpha); ARGR(K, 4, beta); ARGB(K, 5, y); ARG(K, 6, oy); ARG(K, 7, iy);
    return start1(K, N);
}

int gen9blas_gemv(int rodzaj, int trans, int M, int N, double alpha, const void *A, int lda, const void *x, int incx,
             double beta, void *y, int incy) {
    POCZ(M < N ? M : N, !incx || !incy);
    if (lda < N) BLAD("lda < N");
    int nx = trans ? M : N, ny = trans ? N : M;
    op_t w[3] = { { A, ((size_t)(M - 1) * lda + N) * es }, { x, wb(nx, incx, es) }, { y, wb(ny, incy, es) } };
    const int we[3] = { 1, 1, beta != 0.0f || (incy != 1 && incy != -1) }, wy[3] = { 0, 0, 1 };
    if (operandy(w, 3, we, wy)) return -1;
    if (gemv_wew(k, trans, M, N, alpha, w[0].g, lda, w[1].g, wo(nx, incx), incx, beta, w[2].g, wo(ny, incy), incy, 0, 0)) return -1;
    return zakoncz(w, 3, wy);
}

// SYMV: y = alpha·A·x + beta·y, A symetryczna n×n, używany trójkąt uplo. Jedno przejście po zapisanym trójkącie
// (level12.cl: symv1 -- sumy częściowe wierszy do Q i kolumn do P, symv2 -- złożenie).
static cl_mem dQ;
static size_t sQ;
int gen9blas_symv(int rodzaj, int uplo, int n, double alpha, const void *A, int lda, const void *x, int incx,
             double beta, void *y, int incy) {
    POCZ(n, !incx || !incy);
    if (lda < n) BLAD("lda < n");
    op_t w[3] = { { A, ((size_t)(n - 1) * lda + n) * es }, { x, wb(n, incx, es) }, { y, wb(n, incy, es) } };
    const int we[3] = { 1, 1, beta != 0.0f || (incy != 1 && incy != -1) }, wy[3] = { 0, 0, 1 };
    if (operandy(w, 3, we, wy)) return -1;
    int gorny = uplo == GEN9BLAS_UPPER, kol4 = (n + 3) / 4, cz = (16384 + kol4 - 1) / kol4, ng = (n + 63) / 64;
    if (cz > n) cz = n;
    int wpc = (n + cz - 1) / cz; cz = (n + wpc - 1) / wpc;
    if (bufor(&dP, &sP, (size_t)cz * n * RS(k)) || bufor(&dQ, &sQ, (size_t)ng * n * RS(k))) return -1;
    cl_long oa = 0, ox = wo(n, incx), oy = wo(n, incy), ix = incx, iy = incy;
    cl_kernel K = k->symv1;
    ARG(K, 0, n); ARG(K, 1, wpc); ARG(K, 2, gorny); ARGB(K, 3, w[0].g); ARG(K, 4, oa); ARG(K, 5, lda); ARGB(K, 6, w[1].g);
    ARG(K, 7, ox); ARG(K, 8, ix); ARG(K, 9, dP); ARG(K, 10, dQ);
    size_t g[2] = { (size_t)ng * 16, (size_t)cz }, l[2] = { 64, 1 };
    if (g[0] % 64) g[0] = (g[0] + 63) / 64 * 64;
    SPRAWDZ(clEnqueueNDRangeKernel(q, K, 2, NULL, g, l, 0, NULL, NULL));
    K = k->symv2;
    ARG(K, 0, n); ARG(K, 1, cz); ARG(K, 2, gorny); ARG(K, 3, dP); ARG(K, 4, dQ); ARGR(K, 5, alpha); ARGR(K, 6, beta);
    ARGB(K, 7, w[2].g); ARG(K, 8, oy); ARG(K, 9, iy);
    if (start1(K, n)) return -1;
    return zakoncz(w, 3, wy);
}

static cl_mem dTv, dXf;   // TRMV: wynik pośredni (typ elementu); TRSV: wektor roboczy FP32
static size_t sTv, sXf;

// TRMV: x = op(A)·x, A trójkątna n×n (uplo; diag = 1: jedynki na przekątnej). Wynik do bufora, potem kopia do x.
int gen9blas_trmv(int rodzaj, int uplo, int trans, int diag, int n, const void *A, int lda, void *x, int incx) {
    POCZ(n, !incx);
    if (lda < n) BLAD("lda < n");
    op_t w[2] = { { A, ((size_t)(n - 1) * lda + n) * es }, { x, wb(n, incx, es) } };
    const int we[2] = { 1, 1 }, wy[2] = { 0, 1 };
    if (operandy(w, 2, we, wy) || bufor(&dTv, &sTv, (size_t)n * es)) return -1;
    int gorny = uplo == GEN9BLAS_UPPER, tryb = diag ? (gorny ? 4 : 3) : (gorny ? 2 : 1);
    cl_long ox = wo(n, incx), ix = incx, z = 0, j = 1;
    if (gemv_wew(k, trans, n, n, 1.0f, w[0].g, lda, w[1].g, ox, incx, 0.0f, BM(dTv), 0, 1, tryb, diag)) return -1;
    cl_kernel K = k->copy;
    ARG(K, 0, n); ARG(K, 1, dTv); ARG(K, 2, z); ARG(K, 3, j); ARGB(K, 4, w[1].g); ARG(K, 5, ox); ARG(K, 6, ix);
    return start4(K, n) || zakoncz(w, 2, wy) ? -1 : 0;
}

// TRSV: rozwiązanie op(A)·x = b (b w x, wynik w x). Blokowo po 256: blok przekątny w jednej grupie roboczej (podbloki 64
// w pamięci lokalnej), potem aktualizacja reszty wektora; 2 uruchomienia na blok. Wektor roboczy w FP32 (także dla FP16).
// op(A) dolna: w przód. (Wersja jednym uruchomieniem bez globalnej synchronizacji: wolniejsza, NOTATKI.md.)
int gen9blas_trsv(int rodzaj, int uplo, int trans, int diag, int n, const void *A, int lda, void *x, int incx) {
    POCZ(n, !incx);
    if (lda < n) BLAD("lda < n");
    op_t w[2] = { { A, ((size_t)(n - 1) * lda + n) * es }, { x, wb(n, incx, es) } };
    const int we[2] = { 1, 1 }, wy[2] = { 0, 1 };
    if (operandy(w, 2, we, wy) || bufor(&dXf, &sXf, (size_t)n * RS(k))) return -1;
    cl_long ox = wo(n, incx), ix = incx, oa = 0;
    int dolna = (uplo == GEN9BLAS_LOWER) != (trans != 0), tr = trans != 0, dg = diag != 0;
    cl_kernel K = k->do_f;
    ARG(K, 0, n); ARGB(K, 1, w[1].g); ARG(K, 2, ox); ARG(K, 3, ix); ARG(K, 4, dXf);
    if (start1(K, n)) return -1;
    const int nb = 256, nbl = (n + nb - 1) / nb;   // = NBT w level12.cl
    static int prof = -1; if (prof < 0) prof = getenv("GEN9BLAS_PROFILE") != NULL;
    cl_event ev[2 * 1024]; int ne = 0;
    if (prof && 2 * nbl > 2 * 1024) prof = 0;
    for (int s = 0; s < nbl; s++) {
        int p = (dolna ? s : nbl - 1 - s) * nb, b = n - p < nb ? n - p : nb;
        K = k->trsv_blok;
        ARG(K, 0, p); ARG(K, 1, b); ARGB(K, 2, w[0].g); ARG(K, 3, oa); ARG(K, 4, lda); ARG(K, 5, tr); ARG(K, 6, dolna);
        ARG(K, 7, dg); ARG(K, 8, dXf);
        size_t g1 = 256, l1 = 256;
        SPRAWDZ(clEnqueueNDRangeKernel(q, K, 1, NULL, &g1, &l1, 0, NULL, prof ? &ev[ne++] : NULL));
        int r0 = dolna ? p + b : 0, r1 = dolna ? n : p;
        if (r1 <= r0) continue;
        K = tr ? k->trsv_akt_t : k->trsv_akt_n;
        ARG(K, 0, p); ARG(K, 1, b); ARG(K, 2, r0); ARG(K, 3, r1); ARGB(K, 4, w[0].g); ARG(K, 5, oa); ARG(K, 6, lda); ARG(K, 7, dXf);
        size_t g = ((size_t)(r1 - r0) * (tr ? 1 : 16) + 63) / 64 * 64, l = 64;
        SPRAWDZ(clEnqueueNDRangeKernel(q, K, 1, NULL, &g, &l, 0, NULL, prof ? &ev[ne++] : NULL));
    }
    if (prof) {   // GEN9BLAS_PROFILE=1: czasy kerneli (bloki przekątne / aktualizacje) i odstępy między nimi
        if (czekaj()) return -1;
        double tb = 0, ta = 0; cl_ulong pierwszy = 0, ostatni = 0;
        for (int i = 0; i < ne; i++) {
            cl_ulong t0, t1; clGetEventProfilingInfo(ev[i], CL_PROFILING_COMMAND_START, 8, &t0, NULL);
            clGetEventProfilingInfo(ev[i], CL_PROFILING_COMMAND_END, 8, &t1, NULL);
            if (!i) pierwszy = t0;
            ostatni = t1; *(i % 2 ? &ta : &tb) += (t1 - t0) * 1e-6; clReleaseEvent(ev[i]);
        }
        fprintf(stderr, "trsv n=%d: bloki %.2f ms, aktualizacje %.2f ms, od pierwszego do ostatniego %.2f ms (%d kerneli)\n",
                n, tb, ta, (ostatni - pierwszy) * 1e-6, ne);
    }
    K = k->z_f;
    ARG(K, 0, n); ARG(K, 1, dXf); ARGB(K, 2, w[1].g); ARG(K, 3, ox); ARG(K, 4, ix);
    return start1(K, n) || zakoncz(w, 2, wy) ? -1 : 0;
}

// SYR / SYR2 na trójkącie uplo: A = alpha·x·xᵀ + A albo A = alpha·(x·yᵀ + y·xᵀ) + A.
static int syr2(int rodzaj, int uplo, int n, double alpha, const void *x, int incx, const void *y, int incy, void *A, int lda) {
    POCZ(n, !incx || (y && !incy));
    if (lda < n) BLAD("lda < n");
    int dwa = y != NULL;
    op_t w[3] = { { A, ((size_t)(n - 1) * lda + n) * es }, { x, wb(n, incx, es) }, { dwa ? y : x, wb(n, dwa ? incy : incx, es) } };
    const int we[3] = { 1, 1, 1 }, wy[3] = { 1, 0, 0 };
    if (operandy(w, dwa ? 3 : 2, we, wy)) return -1;
    if (!dwa) { w[2] = w[1]; incy = incx; }
    cl_long ox = wo(n, incx), oy = wo(n, incy), oa = 0, ix = incx, iy = incy;
    int tryb = uplo == GEN9BLAS_UPPER ? 2 : 1; cl_kernel K = k->syr2;
    ARG(K, 0, n); ARGR(K, 1, alpha); ARGB(K, 2, w[1].g); ARG(K, 3, ox); ARG(K, 4, ix); ARGB(K, 5, w[2].g); ARG(K, 6, oy);
    ARG(K, 7, iy); ARGB(K, 8, w[0].g); ARG(K, 9, oa); ARG(K, 10, lda); ARG(K, 11, tryb); ARG(K, 12, dwa);
    size_t g[2] = { ((size_t)n + 63) / 64 * 64, (size_t)n }, l[2] = { 64, 1 };
    SPRAWDZ(clEnqueueNDRangeKernel(q, K, 2, NULL, g, l, 0, NULL, NULL));
    return zakoncz(w, 1, wy);
}
int gen9blas_syr(int rodzaj, int uplo, int n, double alpha, const void *x, int incx, void *A, int lda) {
    return syr2(rodzaj, uplo, n, alpha, x, incx, NULL, 0, A, lda);
}
int gen9blas_syr2(int rodzaj, int uplo, int n, double alpha, const void *x, int incx, const void *y, int incy, void *A, int lda) {
    return syr2(rodzaj, uplo, n, alpha, x, incx, y, incy, A, lda);
}

int gen9blas_ger(int rodzaj, int M, int N, double alpha, const void *x, int incx, const void *y, int incy, void *A, int lda) {
    POCZ(M < N ? M : N, !incx || !incy);
    if (lda < N) BLAD("lda < N");
    op_t w[3] = { { x, wb(M, incx, es) }, { y, wb(N, incy, es) }, { A, ((size_t)(M - 1) * lda + N) * es } };
    const int we[3] = { 1, 1, 1 }, wy[3] = { 0, 0, 1 };
    if (operandy(w, 3, we, wy)) return -1;
    cl_long ox = wo(M, incx), oy = wo(N, incy), oa = 0, ix = incx, iy = incy; cl_kernel K = k->ger;
    ARG(K, 0, M); ARG(K, 1, N); ARGR(K, 2, alpha); ARGB(K, 3, w[0].g); ARG(K, 4, ox); ARG(K, 5, ix); ARGB(K, 6, w[1].g);
    ARG(K, 7, oy); ARG(K, 8, iy); ARGB(K, 9, w[2].g); ARG(K, 10, oa); ARG(K, 11, lda);
    size_t g[2] = { ((size_t)N + 63) / 64 * 64, (size_t)M }, l[2] = { 64, 1 };
    SPRAWDZ(clEnqueueNDRangeKernel(q, K, 2, NULL, g, l, 0, NULL, NULL));
    return zakoncz(w, 3, wy);
}

// ---------------- poziom 2: macierze pasmowe i spakowane (układ wierszowy jak CBLAS RowMajor; zob. level12.cl) ----------------
enum { PASMO = 0, SPAK_G = 1, SPAK_D = 2 };
static int rozmiar_el(int rodzaj) { return rodzaj == GEN9BLAS_S ? 4 : rodzaj == GEN9BLAS_D ? 8 : 2; }
static size_t pasmo_b(int wiersze, int lda, int KL, int KU, int es) { return ((size_t)(wiersze - 1) * lda + KL + KU + 1) * es; }
static size_t spak_b(int n, int es) { return (size_t)n * (n + 1) / 2 * es; }

// y = alpha·op(A)·x + beta·y (A w buforze a, opis formatu) -- bez czekania; y może być buforem pośrednim.
static int pmv_wew(l12_t *k, int ny, int nx, int fmt, int KL, int KU, int n, bf_t a, int lda, bf_t x, cl_long ox, cl_long ix,
                   double alpha, double beta, bf_t y, cl_long oy, cl_long iy, int tr, int sym, int gorny, int jedn, int lo, int hi) {
    cl_long oa = 0; cl_kernel K = k->pmv;
    ARG(K, 0, ny); ARG(K, 1, nx); ARG(K, 2, fmt); ARG(K, 3, KL); ARG(K, 4, KU); ARG(K, 5, n); ARGB(K, 6, a); ARG(K, 7, oa);
    ARG(K, 8, lda); ARGB(K, 9, x); ARG(K, 10, ox); ARG(K, 11, ix); ARGR(K, 12, alpha); ARGR(K, 13, beta); ARGB(K, 14, y);
    ARG(K, 15, oy); ARG(K, 16, iy); ARG(K, 17, tr); ARG(K, 18, sym); ARG(K, 19, gorny); ARG(K, 20, jedn); ARG(K, 21, lo); ARG(K, 22, hi);
    return start1(K, ny);
}
// Iloczyn z wynikiem w y (GBMV, SBMV, SPMV).
static int pmv_y(int rodzaj, int M, int N, int fmt, int KL, int KU, int tr, int sym, int gorny, int lo, int hi,
                 double alpha, const void *A, size_t bA, int lda, const void *x, int incx, double beta, void *y, int incy) {
    int nx = tr ? M : N, ny = tr ? N : M;
    POCZ(nx < ny ? nx : ny, !incx || !incy);
    op_t w[3] = { { A, bA }, { x, wb(nx, incx, es) }, { y, wb(ny, incy, es) } };
    const int we[3] = { 1, 1, beta != 0.0f || (incy != 1 && incy != -1) }, wy[3] = { 0, 0, 1 };
    if (operandy(w, 3, we, wy)) return -1;
    if (pmv_wew(k, ny, nx, fmt, KL, KU, N, w[0].g, lda, w[1].g, wo(nx, incx), incx, alpha, beta, w[2].g, wo(ny, incy), incy,
                tr, sym, gorny, 0, lo, hi)) return -1;
    return zakoncz(w, 3, wy);
}
int gen9blas_gbmv(int rodzaj, int trans, int M, int N, int KL, int KU, double alpha, const void *A, int lda, const void *x, int incx,
             double beta, void *y, int incy) {
    if (KL < 0 || KU < 0 || lda < KL + KU + 1) BLAD("gbmv: KL, KU, lda");
    int es = rozmiar_el(rodzaj);
    return pmv_y(rodzaj, M, N, PASMO, KL, KU, trans != 0, 0, 0, trans ? KU : KL, trans ? KL : KU, alpha, A,
                 M > 0 ? pasmo_b(M, lda, KL, KU, es) : 0, lda, x, incx, beta, y, incy);
}
int gen9blas_sbmv(int rodzaj, int uplo, int N, int K, double alpha, const void *A, int lda, const void *x, int incx,
             double beta, void *y, int incy) {
    if (K < 0 || lda < K + 1) BLAD("sbmv: K, lda");
    int es = rozmiar_el(rodzaj), g = uplo == GEN9BLAS_UPPER;
    return pmv_y(rodzaj, N, N, PASMO, g ? 0 : K, g ? K : 0, 0, 1, g, K, K, alpha, A, N > 0 ? pasmo_b(N, lda, 0, K, es) : 0,
                 lda, x, incx, beta, y, incy);
}
int gen9blas_spmv(int rodzaj, int uplo, int N, double alpha, const void *Ap, const void *x, int incx, double beta, void *y, int incy) {
    int es = rozmiar_el(rodzaj), g = uplo == GEN9BLAS_UPPER;
    return pmv_y(rodzaj, N, N, g ? SPAK_G : SPAK_D, 0, 0, 0, 1, g, N, N, alpha, Ap, spak_b(N, es), 0, x, incx, beta, y, incy);
}

// x = op(A)·x (TBMV, TPMV): wynik do bufora pośredniego, potem kopia do x.
static int pmv_x(int rodzaj, int N, int fmt, int KL, int KU, int tr, int jedn, int lo, int hi, const void *A, size_t bA, int lda,
                 void *x, int incx) {
    POCZ(N, !incx);
    op_t w[2] = { { A, bA }, { x, wb(N, incx, es) } };
    const int we[2] = { 1, 1 }, wy[2] = { 0, 1 };
    if (operandy(w, 2, we, wy) || bufor(&dTv, &sTv, (size_t)N * es)) return -1;
    cl_long ox = wo(N, incx), ix = incx, z = 0, j = 1;
    if (pmv_wew(k, N, N, fmt, KL, KU, N, w[0].g, lda, w[1].g, ox, ix, 1.0f, 0.0f, BM(dTv), 0, 1, tr, 0, 0, jedn, lo, hi)) return -1;
    cl_kernel K = k->copy;
    ARG(K, 0, N); ARG(K, 1, dTv); ARG(K, 2, z); ARG(K, 3, j); ARGB(K, 4, w[1].g); ARG(K, 5, ox); ARG(K, 6, ix);
    return start4(K, N) || zakoncz(w, 2, wy) ? -1 : 0;
}
// op(A)·x = b (TBSV, TPSV): jedna grupa robocza, wiersz po wierszu (sekwencyjne z natury), wektor roboczy FP32.
static int psv(int rodzaj, int N, int fmt, int KL, int KU, int tr, int dolna, int jedn, int K_, const void *A, size_t bA, int lda,
               void *x, int incx) {
    POCZ(N, !incx);
    op_t w[2] = { { A, bA }, { x, wb(N, incx, es) } };
    const int we[2] = { 1, 1 }, wy[2] = { 0, 1 };
    if (operandy(w, 2, we, wy) || bufor(&dXf, &sXf, (size_t)N * RS(k))) return -1;
    cl_long ox = wo(N, incx), ix = incx, oa = 0;
    cl_kernel K = k->do_f;
    ARG(K, 0, N); ARGB(K, 1, w[1].g); ARG(K, 2, ox); ARG(K, 3, ix); ARG(K, 4, dXf);
    if (start1(K, N)) return -1;
    K = k->psv;
    ARG(K, 0, N); ARG(K, 1, fmt); ARG(K, 2, KL); ARG(K, 3, KU); ARGB(K, 4, w[0].g); ARG(K, 5, oa); ARG(K, 6, lda); ARG(K, 7, dXf);
    ARG(K, 8, tr); ARG(K, 9, dolna); ARG(K, 10, jedn); ARG(K, 11, K_);
    size_t g = 256, l = 256;
    SPRAWDZ(clEnqueueNDRangeKernel(q, K, 1, NULL, &g, &l, 0, NULL, NULL));
    K = k->z_f;
    ARG(K, 0, N); ARG(K, 1, dXf); ARGB(K, 2, w[1].g); ARG(K, 3, ox); ARG(K, 4, ix);
    return start1(K, N) || zakoncz(w, 2, wy) ? -1 : 0;
}
// op(A) dolna <=> (uplo dolny) xor trans. Zakres niezerowy op(A) w wierszu: dolna [i − K, i], górna [i, i + K].
int gen9blas_tbmv(int rodzaj, int uplo, int trans, int diag, int N, int K, const void *A, int lda, void *x, int incx) {
    if (K < 0 || lda < K + 1) BLAD("tbmv: K, lda");
    int es = rozmiar_el(rodzaj), g = uplo == GEN9BLAS_UPPER, dol = !g != (trans != 0);
    return pmv_x(rodzaj, N, PASMO, g ? 0 : K, g ? K : 0, trans != 0, diag != 0, dol ? K : 0, dol ? 0 : K, A,
                 N > 0 ? pasmo_b(N, lda, 0, K, es) : 0, lda, x, incx);
}
int gen9blas_tpmv(int rodzaj, int uplo, int trans, int diag, int N, const void *Ap, void *x, int incx) {
    int es = rozmiar_el(rodzaj), g = uplo == GEN9BLAS_UPPER, dol = !g != (trans != 0);
    return pmv_x(rodzaj, N, g ? SPAK_G : SPAK_D, 0, 0, trans != 0, diag != 0, dol ? N : 0, dol ? 0 : N, Ap, spak_b(N, es), 0, x, incx);
}
int gen9blas_tbsv(int rodzaj, int uplo, int trans, int diag, int N, int K, const void *A, int lda, void *x, int incx) {
    if (K < 0 || lda < K + 1) BLAD("tbsv: K, lda");
    int es = rozmiar_el(rodzaj), g = uplo == GEN9BLAS_UPPER;
    return psv(rodzaj, N, PASMO, g ? 0 : K, g ? K : 0, trans != 0, !g != (trans != 0), diag != 0, K, A,
               N > 0 ? pasmo_b(N, lda, 0, K, es) : 0, lda, x, incx);
}
int gen9blas_tpsv(int rodzaj, int uplo, int trans, int diag, int N, const void *Ap, void *x, int incx) {
    int es = rozmiar_el(rodzaj), g = uplo == GEN9BLAS_UPPER;
    return psv(rodzaj, N, g ? SPAK_G : SPAK_D, 0, 0, trans != 0, !g != (trans != 0), diag != 0, N, Ap, spak_b(N, es), 0, x, incx);
}
// SPR / SPR2: A = alpha·x·xᵀ + A, A = alpha·(x·yᵀ + y·xᵀ) + A (spakowana, trójkąt uplo).
static int spr2(int rodzaj, int uplo, int n, double alpha, const void *x, int incx, const void *y, int incy, void *Ap) {
    POCZ(n, !incx || (y && !incy));
    int dwa = y != NULL, fmt = uplo == GEN9BLAS_UPPER ? SPAK_G : SPAK_D;
    op_t w[3] = { { Ap, spak_b(n, es) }, { x, wb(n, incx, es) }, { dwa ? y : x, wb(n, dwa ? incy : incx, es) } };
    const int we[3] = { 1, 1, 1 }, wy[3] = { 1, 0, 0 };
    if (operandy(w, dwa ? 3 : 2, we, wy)) return -1;
    if (!dwa) { w[2] = w[1]; incy = incx; }
    cl_long ox = wo(n, incx), oy = wo(n, incy), oa = 0, ix = incx, iy = incy; cl_kernel K = k->spr2;
    ARG(K, 0, n); ARG(K, 1, fmt); ARGR(K, 2, alpha); ARGB(K, 3, w[1].g); ARG(K, 4, ox); ARG(K, 5, ix); ARGB(K, 6, w[2].g);
    ARG(K, 7, oy); ARG(K, 8, iy); ARGB(K, 9, w[0].g); ARG(K, 10, oa); ARG(K, 11, dwa);
    size_t g[2] = { ((size_t)n + 63) / 64 * 64, (size_t)n }, l[2] = { 64, 1 };
    SPRAWDZ(clEnqueueNDRangeKernel(q, K, 2, NULL, g, l, 0, NULL, NULL));
    return zakoncz(w, 1, wy);
}
int gen9blas_spr(int rodzaj, int uplo, int n, double alpha, const void *x, int incx, void *Ap) { return spr2(rodzaj, uplo, n, alpha, x, incx, NULL, 0, Ap); }
int gen9blas_spr2(int rodzaj, int uplo, int n, double alpha, const void *x, int incx, const void *y, int incy, void *Ap) {
    return spr2(rodzaj, uplo, n, alpha, x, incx, y, incy, Ap);
}

// ---------------- chudy GEMM (małe M albo N): PLAN_CHUDY_GEMM.md, results/skinny-gemm-2026-11-12.txt ----------------
// Bez pakowania: wagi (większy operand) czytane wprost raz na kafel wierszy małego operandu, akumulacja w FP32 (także
// dla FP16 -- dokładność co najmniej jak GEN9BLAS_HS). Mały operand „a” (Mc × K) i wagi „w” (K × Nc) w ogólnych krokach;
// przy małym N role zamienione: Cᵀ = op(B)ᵀ · op(A)ᵀ (C zapisywane z zamienionymi krokami).
// GEN9BLAS_SKINNY=0 wyłącza (dawna ścieżka), GEN9BLAS_SKINNY=n ustawia próg min(M, N) ≤ n dla wszystkich typów (pomiary).
#define CHUDY_S 16   // subgrupy na grupę w chudy_r (jak CHUDY_S w level12.cl)
static cl_mem dQ; static size_t sQ;   // sumy częściowe (wagi ciągłe wzdłuż N)
static int prog_chudy(int rodzaj) {
    static int wym = -2;
    if (wym == -2) { const char *e = getenv("GEN9BLAS_SKINNY"); wym = e ? atoi(e) : -1; }
    if (rodzaj == GEN9BLAS_D) return 0;
    if (wym >= 0) return wym;
    // Z pomiaru (results/skinny-gemm-2026-11-12.txt, 8 kształtów, USM i zwykła): przy M = 16 chude szybsze prawie
    // wszędzie (FP16 1,0–2,0×, FP32 0,9–2,2×; słabiej tylko 51866×1280), przy M = 32 FP16 0,3–1,0×, FP32 0,8–1,2×.
    return 16;
}
static int gemm_chudy(int rodzaj, int tA, int tB, int M, int N, int K, double alpha, const void *A, int lda,
                      const void *B, int ldb, double beta, void *C, int ldc) {
    l12_t *k = &L12[rodzaj == GEN9BLAS_S ? 1 : 0];
    const int es = k->es;
    size_t bA = (tA ? (size_t)(K - 1) * lda + M : (size_t)(M - 1) * lda + K) * es;
    size_t bB = (tB ? (size_t)(N - 1) * ldb + K : (size_t)(K - 1) * ldb + N) * es;
    size_t bC = ((size_t)(M - 1) * ldc + N) * es;
    bf_t xA, xB, xC; int zA, zB, zC;
    if (obiekt(A, bA, &uA, 1, 0, &xA, &zA) || obiekt(B, bB, &uB, 1, 0, &xB, &zB) || obiekt(C, bC, &uC, beta != 0.0, 1, &xC, &zC))
        return -1;
    ostatni_zc = zA | zB << 1 | zC << 2;
    // a: Mc × K (kroki sam, sak); w: element (k, n) = w[n·swn + k·swk]; C[m][n] = C[m·scm + n·scn]
    int Mc, Nc; bf_t xa, xw; cl_long sam, sak, swn, swk, scm, scn;
    if (M <= N) {
        Mc = M; Nc = N; xa = xA; xw = xB; scm = ldc; scn = 1;
        sam = tA ? 1 : lda; sak = tA ? lda : 1;          // op(A)[m][k]
        swn = tB ? ldb : 1; swk = tB ? 1 : ldb;           // op(B)[k][n]
    } else {
        Mc = N; Nc = M; xa = xB; xw = xA; scm = 1; scn = ldc;
        sam = tB ? ldb : 1; sak = tB ? 1 : ldb;           // op(B)ᵀ[n][k] = op(B)[k][n]
        swn = tA ? 1 : lda; swk = tA ? lda : 1;           // op(A)ᵀ[k][m] = op(A)[m][k]
    }
    int iK = K, iM = Mc, iN = Nc;
    static int bez_r = -1;   // GEN9BLAS_SKINNY_R=0: dawne chudy_k4/k8 (porównania)
    if (bez_r < 0) { const char *e = getenv("GEN9BLAS_SKINNY_R"); bez_r = e && !atoi(e); }
    const int KL = es == 2 ? 4 : 2;   // elementy k na lane (8 B), jak chudy_r w level12.cl
    const uintptr_t wyr = (uintptr_t)(M <= N ? A : B) | (uintptr_t)(M <= N ? B : A) | (uintptr_t)(sam * es) | (uintptr_t)(swn * es);
    // adres na GPU ma to samo przesunięcie w stronie co na hoście (USM, userptr kcl; bufor roboczy wyrównany)
    // M ≤ 4 przy wierszu wag > 16 KB: dawne jądro (pomiar 2026-11-12: N = 4096, K = 11008 FP32 M = 2–4 0,7×, FP16 0,9–1,0×;
    // luka modelu: 8 długich wierszy naraz na grupę -- strony/TLB?)
    const int r_ok = Mc > 4 || (size_t)K * es <= 16384;
    if (swk == 1 && sak == 1 && Mc >= 2 && K % (8 * KL) == 0 && (wyr & 15) == 0 && r_ok && !bez_r) {
        // przebudowa 2026-11-12 (results/skinny-gemm-rework-2026-11-12.txt): SIMD8, 8 wierszy wag, CHUDY_S subgrup
        const int MT = Mc <= 4 ? 4 : 8;
        cl_kernel Kk = MT == 4 ? k->chudy_r4 : k->chudy_r8;
        ARG(Kk, 0, iM); ARG(Kk, 1, iN); ARG(Kk, 2, iK); SPRAWDZ(argr(Kk, 3, alpha, 4)); SPRAWDZ(argr(Kk, 4, beta, 4));
        ARGB(Kk, 5, xa); ARG(Kk, 6, sam); ARGB(Kk, 7, xw); ARG(Kk, 8, swn); ARGB(Kk, 9, xC); ARG(Kk, 10, scm); ARG(Kk, 11, scn);
        size_t g[2] = { (size_t)(Mc + MT - 1) / MT * 8 * CHUDY_S, (size_t)(Nc + 7) / 8 }, l[2] = { 8 * CHUDY_S, 1 };
        SPRAWDZ(clEnqueueNDRangeKernel(q, Kk, 2, NULL, g, l, 0, NULL, NULL));
    } else if (swk == 1) {   // wagi ciągłe wzdłuż K
        const int MT = Mc == 1 ? 1 : Mc <= 4 ? 4 : 8, RW = MT == 8 ? 2 : 4;   // jak CHUDY_K w level12.cl
        cl_kernel Kk = MT == 1 ? k->chudy_k1 : MT == 4 ? k->chudy_k4 : k->chudy_k8;
        ARG(Kk, 0, iM); ARG(Kk, 1, iN); ARG(Kk, 2, iK); SPRAWDZ(argr(Kk, 3, alpha, es == 8 ? 8 : 4)); SPRAWDZ(argr(Kk, 4, beta, 4));
        ARGB(Kk, 5, xa); ARG(Kk, 6, sam); ARG(Kk, 7, sak); ARGB(Kk, 8, xw); ARG(Kk, 9, swn); ARGB(Kk, 10, xC); ARG(Kk, 11, scm); ARG(Kk, 12, scn);
        size_t g[2] = { (size_t)(Mc + MT - 1) / MT * 256, (size_t)(Nc + RW - 1) / RW }, l[2] = { 256, 1 };
        SPRAWDZ(clEnqueueNDRangeKernel(q, Kk, 2, NULL, g, l, 0, NULL, NULL));
    } else {          // wagi ciągłe wzdłuż N (swn == 1): części K, potem suma i epilog
        const int MT = 4, kol4 = (Nc + 3) / 4, kafle = (Mc + MT - 1) / MT;
        int cz = (8192 + kol4 - 1) / kol4; if (cz > K) cz = K; if (cz < 1) cz = 1;   // ~8 tys. work-itemów na kafel M
        int kpc = (K + cz - 1) / cz; cz = (K + kpc - 1) / kpc;
        if (bufor(&dQ, &sQ, (size_t)cz * Mc * Nc * sizeof(float))) return -1;
        cl_kernel Kn = k->chudy_n4;
        ARG(Kn, 0, iM); ARG(Kn, 1, iN); ARG(Kn, 2, iK); ARG(Kn, 3, kpc); ARGB(Kn, 4, xa); ARG(Kn, 5, sam); ARG(Kn, 6, sak);
        ARGB(Kn, 7, xw); ARG(Kn, 8, swk); ARG(Kn, 9, dQ);
        size_t g[2] = { ((size_t)kol4 + 63) / 64 * 64, (size_t)cz * kafle }, l[2] = { 64, 1 };
        SPRAWDZ(clEnqueueNDRangeKernel(q, Kn, 2, NULL, g, l, 0, NULL, NULL));
        cl_kernel K2 = k->chudy_n2;
        ARG(K2, 0, iM); ARG(K2, 1, iN); ARG(K2, 2, cz); ARG(K2, 3, dQ); SPRAWDZ(argr(K2, 4, alpha, 4)); SPRAWDZ(argr(K2, 5, beta, 4));
        ARGB(K2, 6, xC); ARG(K2, 7, scm); ARG(K2, 8, scn);
        size_t g2[2] = { ((size_t)Nc + 63) / 64 * 64, (size_t)Mc }, l2[2] = { 64, 1 };
        SPRAWDZ(clEnqueueNDRangeKernel(q, K2, 2, NULL, g2, l2, 0, NULL, NULL));
    }
    return do_hosta(xC, zC, bC, C);
}
