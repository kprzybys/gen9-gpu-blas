// kcl: podzbiór OpenCL dla gen9blas.c bezpośrednio na i915 (Gen9, UHD 620), bez NEO. Wiedza i pomiary: ../BEZNEO.md.
//
// Model: kolejka w kolejności = bufor komend (batch) budowany na bieżąco; każde clEnqueueNDRangeKernel dopisuje
// IDD + dane pośrednie + (dla argumentów stateful) binding table i opisy powierzchni + GPGPU_WALKER; między jądrami
// bariera PIPE_CONTROL (CS stall + DC flush) -- bez niej kolejne walkery nachodzą na siebie (BEZNEO.md 6d).
// Bufor jest wysyłany (execbuffer2, softpin) przy clFlush/clFinish/kcl_wait, przy zapełnieniu (64 IDD) albo przy
// pytaniu o stan zdarzenia. Bufory komend rotują (KSLOT); przed ponownym użyciem slotu czekanie na jego koniec.
// Pamięć: cl_mem = obiekt GEM (adres w PPGTT stały: softpin, przydział rosnący, adresy nie są używane ponownie);
// „USM host” (gen9blas_alloc) = obiekt GEM zmapowany WB do CPU (LLC: spójny z GPU), wskaźniki wewnątrz tłumaczone tabelą.
#include "kcl.h"
#include <drm/i915_drm.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "gen9_pack.h"
#include "kcl_desc.h"
#include <time.h>

static uint32_t MOCS_WB = 2 << 1;   // [HIP] indeks 2 tabeli MOCS i915 (SKL) = WB, jak Mesa; KCL_MOCS=n: inny indeks (diagnostyka)
#define L3_SLM 0x60000321u          // L3CNTLREG z SLM (NEO gen9/reg_configs.h)
#define L3_BEZ_SLM 0x80000340u      // L3CNTLREG bez SLM
enum { KSLOT = 8, DYN = 1 << 20, BB = 128 << 10, IDD_MAX = 64,
       O_IND = 0x1000, O_SS = 0xC0000, SS_LEN = 0x10000,   // dyn: IDD | dane pośrednie | binding tables + opisy powierzchni
       MAXBO = 512, TS_MAX = 4096 };

// ---------------- obiekty GEM ----------------
typedef struct kbo { uint32_t h; uint64_t adr, size; void *map; int ref, ro; uint64_t znak; } kbo;   // ro: userptr tylko do odczytu
struct _cl_mem { kbo *bo; uint64_t off, size; struct _cl_mem *rodzic; int ref; };
struct _cl_program { const struct kcl_program_opis *o; int indeks; };
struct _cl_kernel {
    const struct kcl_jadro_opis *o; uint32_t isa_off; int prog;
    struct { kbo *bo; uint64_t adr, rozm; uint8_t v[8]; int ust; } a[24];
};
struct _cl_event { uint64_t seq; int ts, ts_k; int ref; };   // ts, ts_k: znaczniki czasu początku i końca (-1: brak)

static int fd = -1; static uint32_t ctx_id; static int kprofil, kprofil_jadra;
// KCL_PROFILE_KERNELS=1: znaczniki czasu wokół każdego jądra, suma czasów po nazwie wypisywana na końcu procesu
typedef struct { const char *nazwa; int ts0, ts1; } pomiar_t;
static pomiar_t pom[TS_MAX / 2]; static int npom;
static struct { const char *nazwa; double ms; long n; } suma[256]; static int nsuma;
static uint64_t nast_adr = 1ull << 32;
static kbo *isa, *konst[16], *prywatna, *ts_bo;
static uint32_t *isa_off[16];   // [program][jądro] -> offset w isa
static uint32_t ts_n;

static uint64_t kbo_adres(uint64_t size);
static int kbo_nowy(uint64_t size, int mapuj, kbo **wynik) {
    size = (size + 4095) & ~4095ull;
    kbo *b = calloc(1, sizeof *b);
    struct drm_i915_gem_create c = { .size = size };
    if (!b || ioctl(fd, DRM_IOCTL_I915_GEM_CREATE, &c)) { free(b); return -1; }
    b->h = c.handle; b->size = size; b->ref = 1; b->adr = kbo_adres(size);
    if (mapuj) {
        struct drm_i915_gem_mmap_offset m = { .handle = b->h, .flags = I915_MMAP_OFFSET_WB };
        if (ioctl(fd, DRM_IOCTL_I915_GEM_MMAP_OFFSET, &m)) return -1;
        b->map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, m.offset);
        if (b->map == MAP_FAILED) { b->map = NULL; return -1; }
    }
    *wynik = b; return 0;
}
// Adres obiektu w PPGTT (softpin; przydział rosnący, adresy nie są używane ponownie).
static uint64_t kbo_adres(uint64_t size) {
    // Adres w PPGTT: obiekty ≥ 2 MB wyrównane do 2 MB (jak NEO) -- tylko wtedy i915 może użyć stron 2 MB w tablicy
    // stron GPU (mniej chybień TLB przy dużych macierzach); mniejsze do 64 KB.
    static int tryb = -1, licznik;
    if (tryb < 0) tryb = getenv("KCL_ADDR") ? atoi(getenv("KCL_ADDR")) : 0;   // diagnostyka: 0 64K, 1 2M, 2 kolor
    uint64_t wyr = tryb == 1 && size >= (2u << 20) ? (2u << 20) : 0x10000;
    nast_adr = (nast_adr + wyr - 1) & ~(wyr - 1);
    if (tryb == 2) nast_adr += ((uint64_t)(licznik++ * 5) % 16) * 0x1000;   // przesunięcie o 0..15 × 4 KB
    const uint64_t a = nast_adr;
    nast_adr += (size + 0xffff) & ~0xffffull;
    return a;
}
static void kbo_zwolnij(kbo *b) {
    if (!b || --b->ref > 0) return;
    if (b->map) munmap(b->map, b->size);
    struct drm_gem_close c = { .handle = b->h };
    ioctl(fd, DRM_IOCTL_GEM_CLOSE, &c);   // bezpieczne także, gdy GPU jeszcze go używa (i915 trzyma referencję)
    free(b);
}

// ---------------- USM host: obiekty GEM zmapowane do CPU ----------------
typedef struct { uintptr_t p; uint64_t n; kbo *bo; } usm_t;
static usm_t *usm; static int nusm, cusm;
static usm_t *usm_szukaj(const void *ptr) {
    uintptr_t x = (uintptr_t)ptr;
    for (int i = 0; i < nusm; i++) if (x >= usm[i].p && x < usm[i].p + usm[i].n) return &usm[i];
    return NULL;
}

// ---------------- userptr: zwykła pamięć procesu (np. tablice NumPy) jako obiekt GEM, z pamięcią podręczną ----------------
// Fakty i pomiary: BEZNEO.md 6g, experiments/no-neo/userptr.c, test_userptr.py, bench/compare/userptr.py.
// - Przypięcie stron (przy pierwszym execbuf) kosztuje ~0,45 us CPU na stronę 4 KB (mapowanie przez IOMMU), czyli tyle
//   co kopia -- opłaca się tylko obiekt trzymany między wywołaniami (potem GPU czyta go tak szybko jak własny bufor).
// - Gdy pamięć procesu się zmieni (munmap, nowe mapowanie), jądro Linuksa (mmu notifier) unieważnia strony obiektu
//   i przy następnym execbuf przypina je od nowa -- obiekt z pamięci podręcznej pozostaje poprawny.
// - Obszary JEDNEJ operacji, które mają wspólne strony, muszą iść przez jeden obiekt: dwa obiekty na tych samych
//   stronach to dwa adresy GPU tej samej linii pamięci, a zapisy do jednej linii przez dwa adresy w jednym jądrze gubią
//   się (zmierzone: swap/rot, x[0] w linii z końcem y).
// - Obiekt z pamięci podręcznej jest używany tylko przy dokładnie tym samym zakresie stron, co (suma) obszarów bieżącej
//   operacji: to pamięć podana teraz przez wywołującego, więc zwykła i zmapowana. Obiekt szerszy mógłby obejmować
//   strony zajęte od tamtej pory przez co innego (np. mmap obiektu GEM: brak struct page, przypięcie -> EFAULT).
// Dlatego leniwie: kcl_userptr zapisuje obszar i zwraca cl_mem „do rozwiązania”; przy pierwszym clSetKernelArg z takim
// cl_mem wszystkie obszary operacji (gen9blas ustawia argumenty po wszystkich obszarach) dzielą się na grupy nakładających
// się zakresów, a każda grupa dostaje jeden obiekt: z pamięci podręcznej (ten sam zakres sumy) albo nowy.
// Granica operacji: kcl_new_operation() (gen9blas: na początku każdej funkcji BLAS). Obszar zgłoszony po rozwiązaniu: obiekt
// tej operacji, jeśli go zawiera, inaczej NULL (kopia przez CPU).
enum { UP_MAX = 256, UP_MEM = 64, UP_ZGL = 16 };
typedef struct { uintptr_t s, e; kbo *bo; uint64_t uzycie, epoka; } up_t;
static up_t up[UP_MAX]; static int nup; static uint64_t up_licznik, up_bajty, up_limit, up_epoka = 1;
static struct _cl_mem up_mem[UP_MEM]; static int up_mem_i;   // cl_mem dla argumentów (ważne w bieżącej operacji)
static struct { uintptr_t p, s, e; int zapis; struct _cl_mem *m; } zgl[UP_ZGL];   // obszary bieżącej operacji
static int nzgl, rozwiazane; static uint64_t zgl_epoka;
static int cur;
static int wyslij(void), init(void);
// ---- diagnostyka: KCL_PHASES=1 -- czas ściany i CPU wątku w fazach (na stderr przy wyjściu) ----
enum { F_ENQ, F_EXEC, F_WAIT, F_UP, F_N };
static const char *f_nazwa[F_N] = { "EnqueueNDRange", "execbuf (ioctl)", "GEM_WAIT (ioctl)", "userptr" };
static struct { double sc, cpu; long n; } fz[F_N];
static int fazy = -1; static long fz_obiekty;
static double t_sc(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static double t_cpu(void) { struct timespec t; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static void fazy_wypisz(void) {
    for (int i = 0; i < F_N; i++) if (fz[i].n)
        fprintf(stderr, "kcl faza %-18s %8ld x  ściana %8.2f µs  CPU %8.2f µs (na raz)\n", f_nazwa[i], fz[i].n, fz[i].sc / fz[i].n, fz[i].cpu / fz[i].n);
    if (fz[F_EXEC].n) fprintf(stderr, "kcl faza obiektów na execbuf: %.1f\n", (double)fz_obiekty / fz[F_EXEC].n);
}
static int fazy_wl(void) { if (fazy < 0) { fazy = getenv("KCL_PHASES") != NULL; if (fazy) atexit(fazy_wypisz); } return fazy; }
#define FAZA_POCZ double fs_ = fazy_wl() ? t_sc() : 0, fc_ = fazy ? t_cpu() : 0
#define FAZA_KON(i) do { if (fazy) { fz[i].sc += t_sc() - fs_; fz[i].cpu += t_cpu() - fc_; fz[i].n++; } } while (0)
static uint64_t S_cur_seq(void);
void kcl_new_operation(void) { up_epoka++; }
static void up_usun(int i) {
    if (cur >= 0 && up[i].bo->znak == S_cur_seq()) wyslij();   // obiekt w budowanym buforze: najpierw wysłanie
    up_bajty -= up[i].e - up[i].s;
    kbo_zwolnij(up[i].bo);   // GEM_CLOSE także, gdy GPU jeszcze go używa (i915 trzyma referencję)
    up[i] = up[--nup];
}
// Obiekt dla zakresu stron [s, e): z pamięci podręcznej (dokładnie ten zakres) albo nowy. Limit pamięci jest miękki:
// obiektów bieżącej operacji się nie usuwa.
static up_t *up_obiekt(uintptr_t s, uintptr_t e, int zapis) {
    for (int i = 0; i < nup; i++)
        if (up[i].s == s && up[i].e == e && (!zapis || !up[i].bo->ro)) { up[i].uzycie = ++up_licznik; up[i].epoka = up_epoka; return &up[i]; }
    while (nup == UP_MAX || (nup && up_bajty + (e - s) > up_limit)) {   // miejsce: najdawniej używany spoza operacji
        int j = -1;
        for (int i = 0; i < nup; i++) if (up[i].epoka != up_epoka && (j < 0 || up[i].uzycie < up[j].uzycie)) j = i;
        if (j < 0) break;
        up_usun(j);
    }
    if (nup == UP_MAX) return NULL;
    struct drm_i915_gem_userptr u = { .user_ptr = s, .user_size = e - s, .flags = zapis ? 0 : I915_USERPTR_READ_ONLY };
    kbo *b = calloc(1, sizeof *b);
    if (!b || ioctl(fd, DRM_IOCTL_I915_GEM_USERPTR, &u)) { free(b); return NULL; }
    b->h = u.handle; b->size = e - s; b->ref = 1; b->ro = !zapis; b->adr = kbo_adres(e - s);
    up[nup] = (up_t){ s, e, b, ++up_licznik, up_epoka }; up_bajty += e - s;
    return &up[nup++];
}
static void up_przypisz(struct _cl_mem *m, uintptr_t p, up_t *x) {
    // rozmiar do końca obiektu (strony), jak usm_arg: sprzęt sprawdza granice bufora RAW co 4 B
    m->bo = x->bo; m->off = p - x->s; m->size = x->bo->size - (p - x->s);
}
// Rozwiązanie obszarów bieżącej operacji: grupy nakładających się zakresów -> jeden obiekt na grupę.
static int up_rozwiaz(void) {
    if (rozwiazane || zgl_epoka != up_epoka) return 0;
    rozwiazane = 1;
    int grupa[UP_ZGL];
    for (int i = 0; i < nzgl; i++) grupa[i] = i;
    for (int zmiana = 1; zmiana; ) {   // spójne składowe (nakładanie zakresów stron)
        zmiana = 0;
        for (int i = 0; i < nzgl; i++) for (int j = 0; j < nzgl; j++)
            if (grupa[i] != grupa[j] && zgl[i].s < zgl[j].e && zgl[j].s < zgl[i].e) {
                int g = grupa[i] < grupa[j] ? grupa[i] : grupa[j];
                grupa[i] = grupa[j] = g; zmiana = 1;
            }
        for (int i = 0; i < nzgl; i++) while (grupa[grupa[i]] != grupa[i]) grupa[i] = grupa[grupa[i]];
    }
    for (int g = 0; g < nzgl; g++) {
        uintptr_t s = UINTPTR_MAX, e = 0; int zapis = 0, jest = 0;
        for (int i = 0; i < nzgl; i++) if (grupa[i] == g) {
            jest = 1; if (zgl[i].s < s) s = zgl[i].s;
            if (zgl[i].e > e) e = zgl[i].e;
            zapis |= zgl[i].zapis;
        }
        if (!jest) continue;
        up_t *x = up_obiekt(s, e, zapis);
        if (!x) return -1;
        kbo *b = x->bo;   // x może się przesunąć w tablicy przy kolejnych grupach -- zapamiętany obiekt
        for (int i = 0; i < nzgl; i++) if (grupa[i] == g) {
            zgl[i].m->bo = b; zgl[i].m->off = zgl[i].p - s; zgl[i].m->size = b->size - (zgl[i].p - s);
        }
    }
    return 0;
}
static cl_mem kcl_userptr_(const void *ptr, size_t n, int zapis);
cl_mem kcl_userptr(const void *ptr, size_t n, int zapis) { FAZA_POCZ; cl_mem m = kcl_userptr_(ptr, n, zapis); FAZA_KON(F_UP); return m; }
static cl_mem kcl_userptr_(const void *ptr, size_t n, int zapis) {
    if (init() || !ptr || !n) return NULL;
    const uintptr_t p = (uintptr_t)ptr, s = p & ~4095ul, e = (p + n + 4095) & ~4095ul;
    if (e - s > up_limit) return NULL;   // za duże do trzymania: kopia przez CPU
    if (zgl_epoka != up_epoka) { zgl_epoka = up_epoka; nzgl = 0; rozwiazane = 0; }
    struct _cl_mem *m = &up_mem[up_mem_i++ % UP_MEM];
    *m = (struct _cl_mem){ .bo = NULL, .off = 0, .size = n, .ref = 1 };
    if (rozwiazane || nzgl == UP_ZGL) {   // po rozwiązaniu: obiekt tej operacji zawierający obszar albo osobny
        int nakl = 0;
        for (int i = 0; i < nup; i++)
            if (up[i].epoka == up_epoka && up[i].s < e && s < up[i].e) {
                if (up[i].s <= s && e <= up[i].e && (!zapis || !up[i].bo->ro)) { up_przypisz(m, p, &up[i]); return m; }
                nakl = 1;
            }
        if (nakl) return NULL;
        up_t *x = up_obiekt(s, e, zapis);
        if (!x) return NULL;
        up_przypisz(m, p, x); return m;
    }
    zgl[nzgl++] = (typeof(zgl[0])){ p, s, e, zapis, m };
    return m;
}

// ---------------- bufor komend ----------------
typedef struct { kbo *dyn, *bb; uint64_t seq; } slot_t;
static slot_t S[KSLOT];
static int cur = -1;                   // slot budowanego bufora (-1: brak)
static uint64_t S_cur_seq(void) { return S[cur].seq; }
static uint64_t seq_nast = 1, seq_wyslany = 0, seq_gotowy = 0;
static uint32_t *p, *idl, *lri, o_ind, o_ss, nwyw; static int slm_w_buforze;
static kbo *lista[MAXBO]; static int nlista;
static int l3_zawsze_slm = 0;   // KCL_L3=slm: zawsze L3 z SLM (jak NEO na KBL/CFL); pomiar: bez różnicy, domyślnie bez

#define EMIT(T, ...) do { struct GFX9_##T c_ = { GFX9_##T##_header, __VA_ARGS__ }; GFX9_##T##_pack(NULL, p, &c_); p += GFX9_##T##_length; } while (0)

static void dodaj_bo(kbo *b) {
    if (!b || b->znak == S[cur].seq) return;
    b->znak = S[cur].seq; lista[nlista++] = b;
}

static int czekaj_slot(int s) {
    if (!S[s].seq || S[s].seq <= seq_gotowy) return 0;
    struct drm_i915_gem_wait w = { .bo_handle = S[s].bb->h, .timeout_ns = 60ll * 1000000000 };
    FAZA_POCZ;
    while (ioctl(fd, DRM_IOCTL_I915_GEM_WAIT, &w)) { if (errno != EINTR) return -1; }
    FAZA_KON(F_WAIT);
    if (S[s].seq > seq_gotowy) seq_gotowy = S[s].seq;   // kolejka w kolejności: wcześniejsze też gotowe
    return 0;
}

static int otworz_bufor(void) {
    if (cur >= 0) return 0;
    int s = (int)(seq_nast % KSLOT);
    if (czekaj_slot(s)) return -1;
    cur = s; S[s].seq = seq_nast++;
    nlista = 0; nwyw = 0; slm_w_buforze = 0; o_ind = O_IND; o_ss = O_SS;
    p = S[s].bb->map;
    const uint64_t D = S[s].dyn->adr;
    EMIT(PIPE_CONTROL, .CommandStreamerStallEnable = 1, .DCFlushEnable = 1);
    lri = p;   // L3CNTLREG: wartość uzupełniana przy wysłaniu (z SLM, jeśli któreś jądro jej używa)
    EMIT(MI_LOAD_REGISTER_IMM, .RegisterOffset = 0x7034, .DataDWord = L3_BEZ_SLM);
    // Arbitraż wątków EU: round-robin (DebugControlReg2 0xE404 = 0x100), jak NEO na Gen9 (gfx_core_helper_gen9.cpp:
    // getDefaultThreadArbitrationPolicy, command_encoder_gen9.cpp). KCL_ARB=wiek: domyślny sprzętowy (diagnostyka).
    static int arb = -1; if (arb < 0) arb = !(getenv("KCL_ARB") && !strcmp(getenv("KCL_ARB"), "wiek"));
    if (arb) EMIT(MI_LOAD_REGISTER_IMM, .RegisterOffset = 0xE404, .DataDWord = 0x100);
    EMIT(PIPELINE_SELECT, .MaskBits = 3, .PipelineSelection = GPGPU);
    EMIT(STATE_BASE_ADDRESS,
        .GeneralStateBaseAddressModifyEnable = 1, .GeneralStateBaseAddress = 0, .GeneralStateMOCS = MOCS_WB,
        .StatelessDataPortAccessMOCS = MOCS_WB,
        .SurfaceStateBaseAddressModifyEnable = 1, .SurfaceStateBaseAddress = D + O_SS, .SurfaceStateMOCS = MOCS_WB,
        .DynamicStateBaseAddressModifyEnable = 1, .DynamicStateBaseAddress = D, .DynamicStateMOCS = MOCS_WB,
        .IndirectObjectBaseAddressModifyEnable = 1, .IndirectObjectBaseAddress = D, .IndirectObjectMOCS = MOCS_WB,
        .InstructionBaseAddressModifyEnable = 1, .InstructionBaseAddress = isa->adr, .InstructionMOCS = MOCS_WB,
        .GeneralStateBufferSizeModifyEnable = 1, .GeneralStateBufferSize = 0xfffff,
        .DynamicStateBufferSizeModifyEnable = 1, .DynamicStateBufferSize = DYN / 4096,
        .IndirectObjectBufferSizeModifyEnable = 1, .IndirectObjectBufferSize = DYN / 4096,
        .InstructionBuffersizeModifyEnable = 1, .InstructionBufferSize = (uint32_t)(isa->size / 4096),
        .BindlessSurfaceStateBaseAddressModifyEnable = 1, .BindlessSurfaceStateMOCS = MOCS_WB,
        .BindlessSurfaceStateSize = 0xfffff);
    EMIT(PIPE_CONTROL, .CommandStreamerStallEnable = 1, .StateCacheInvalidationEnable = 1,
         .InstructionCacheInvalidateEnable = 1, .ConstantCacheInvalidationEnable = 1, .TextureCacheInvalidationEnable = 1);
    EMIT(MEDIA_VFE_STATE, .MaximumNumberofThreads = 24 * 7 - 1, .NumberofURBEntries = 1,
         .URBEntryAllocationSize = 0x782, .CURBEAllocationSize = 0);
    idl = p;   // długość tablicy IDD uzupełniana przy wysłaniu
    EMIT(MEDIA_INTERFACE_DESCRIPTOR_LOAD, .InterfaceDescriptorTotalLength = 32, .InterfaceDescriptorDataStartAddress = 0);
    // Na liście execbuf tylko obiekty używane: stałe programu i pamięć prywatna dodaje jądro, które ich potrzebuje
    // (kcl_EnqueueNDRangeKernel). i915 płaci za każdy obiekt (przypięcie VMA, synchronizacja): 15 -> 5 obiektów
    // przy małym axpy, pomiar results/overhead-2026-11-12.txt.
    dodaj_bo(isa); if (ts_bo) dodaj_bo(ts_bo);
    return 0;
}

static int wyslij(void) {
    if (cur < 0) return 0;
    slot_t *s = &S[cur];
    EMIT(PIPE_CONTROL, .CommandStreamerStallEnable = 1, .DCFlushEnable = 1);
    EMIT(MI_BATCH_BUFFER_END);
    if ((p - (uint32_t *)s->bb->map) & 1) *p++ = 0;
    { struct GFX9_MI_LOAD_REGISTER_IMM l = { GFX9_MI_LOAD_REGISTER_IMM_header, .RegisterOffset = 0x7034,
          .DataDWord = (slm_w_buforze || l3_zawsze_slm) ? L3_SLM : L3_BEZ_SLM }; GFX9_MI_LOAD_REGISTER_IMM_pack(NULL, lri, &l); }
    { struct GFX9_MEDIA_INTERFACE_DESCRIPTOR_LOAD l = { GFX9_MEDIA_INTERFACE_DESCRIPTOR_LOAD_header,
          .InterfaceDescriptorTotalLength = 32 * (nwyw ? nwyw : 1), .InterfaceDescriptorDataStartAddress = 0 };
      GFX9_MEDIA_INTERFACE_DESCRIPTOR_LOAD_pack(NULL, idl, &l); }
    static struct drm_i915_gem_exec_object2 ob[MAXBO + 2];
    const uint64_t F = EXEC_OBJECT_PINNED | EXEC_OBJECT_SUPPORTS_48B_ADDRESS;
    int n = 0;
    for (int i = 0; i < nlista; i++)
        ob[n++] = (struct drm_i915_gem_exec_object2){ .handle = lista[i]->h, .offset = lista[i]->adr, .flags = F | (lista[i]->ro ? 0 : EXEC_OBJECT_WRITE) };
    ob[n++] = (struct drm_i915_gem_exec_object2){ .handle = s->dyn->h, .offset = s->dyn->adr, .flags = F };
    ob[n++] = (struct drm_i915_gem_exec_object2){ .handle = s->bb->h, .offset = s->bb->adr, .flags = F };
    struct drm_i915_gem_execbuffer2 eb = { .buffers_ptr = (uintptr_t)ob, .buffer_count = n,
        .batch_len = (uint32_t)((p - (uint32_t *)s->bb->map) * 4), .flags = I915_EXEC_RENDER | I915_EXEC_NO_RELOC,
        .rsvd1 = ctx_id };
    int r;
    FAZA_POCZ;
    while ((r = ioctl(fd, DRM_IOCTL_I915_GEM_EXECBUFFER2, &eb)) && errno == EINTR) {}
    FAZA_KON(F_EXEC); if (fazy) fz_obiekty += n;
    seq_wyslany = s->seq; cur = -1;
    if (r) {
        fprintf(stderr, "kcl: execbuffer2: %s\n", strerror(errno));
        return -1;
    }
    return 0;
}

static int czekaj_seq(uint64_t seq) {
    if (cur >= 0 && S[cur].seq <= seq && wyslij()) return -1;
    if (seq <= seq_gotowy) return 0;
    return czekaj_slot((int)(seq % KSLOT));
}

static void zbierz_pomiary(void) {   // po czekaniu na całą pracę: czasy jąder do sum
    for (int i = 0; i < npom; i++) {
        const uint64_t *t = ts_bo->map; double ms = (double)(t[pom[i].ts1] - t[pom[i].ts0]) / 12e3;
        int j = 0; while (j < nsuma && strcmp(suma[j].nazwa, pom[i].nazwa)) j++;
        if (j == nsuma && nsuma < 256) { suma[nsuma].nazwa = pom[i].nazwa; nsuma++; }
        if (j < 256) { suma[j].ms += ms; suma[j].n++; }
    }
    npom = 0;
}
static void wypisz_pomiary(void) {
    zbierz_pomiary();
    for (int j = 0; j < nsuma; j++) fprintf(stderr, "kcl: %-22s %6ld x  %10.2f ms\n", suma[j].nazwa, suma[j].n, suma[j].ms);
}

cl_int kcl_wait(void) {
    uint64_t ost = cur >= 0 ? S[cur].seq : seq_wyslany;
    if (czekaj_seq(ost)) return CL_OUT_OF_RESOURCES;
    if (kprofil_jadra) zbierz_pomiary();
    return CL_SUCCESS;
}

// ---------------- inicjalizacja ----------------
static int init(void) {
    if (fd >= 0) return 0;
    fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
    if (fd < 0) return -1;
    struct drm_i915_gem_context_create cc = { 0 };
    if (ioctl(fd, DRM_IOCTL_I915_GEM_CONTEXT_CREATE, &cc)) return -1;
    ctx_id = cc.ctx_id;
    kprofil_jadra = getenv("KCL_PROFILE_KERNELS") != NULL;
    kprofil = getenv("GEN9BLAS_PROFILE") != NULL || kprofil_jadra;
    const char *um = getenv("KCL_USERPTR_MB");
    up_limit = (uint64_t)(um && *um ? atoi(um) : 1024) << 20;   // przypięta pamięć procesu
    if (getenv("KCL_MOCS")) MOCS_WB = atoi(getenv("KCL_MOCS")) << 1;
    if (getenv("KCL_L3") && !strcmp(getenv("KCL_L3"), "slm")) l3_zawsze_slm = 1;
    uint64_t rozm = 0;
    for (int i = 0; i < kcl_nprogramow; i++)
        for (uint32_t j = 0; j < kcl_programy[i].njader; j++) rozm += (kcl_programy[i].jadra[j].isa_len + 1024 + 63) & ~63u;
    if (kbo_nowy(rozm + 4096, 1, &isa)) return -1;
    uint64_t o = 0;
    for (int i = 0; i < kcl_nprogramow; i++) {
        const struct kcl_program_opis *pr = &kcl_programy[i];
        isa_off[i] = calloc(pr->njader, sizeof(uint32_t));
        for (uint32_t j = 0; j < pr->njader; j++) {
            isa_off[i][j] = (uint32_t)o;
            memcpy((uint8_t *)isa->map + o, pr->jadra[j].isa, pr->jadra[j].isa_len);
            o += (pr->jadra[j].isa_len + 1024 + 63) & ~63u;   // za kodem zera: prefetch instrukcji czyta dalej
        }
        if (pr->nstale) {
            if (kbo_nowy(pr->nstale, 1, &konst[i])) return -1;
            memcpy(konst[i]->map, pr->stale, pr->nstale);
        }
    }
    for (int s = 0; s < KSLOT; s++) if (kbo_nowy(DYN, 1, &S[s].dyn) || kbo_nowy(BB, 1, &S[s].bb)) return -1;
    if (kprofil && kbo_nowy(TS_MAX * 8, 1, &ts_bo)) return -1;
    if (kprofil_jadra) atexit(wypisz_pomiary);
    return 0;
}

static int kcl_wskaznik;   // adres dowolny, unikalny: identyfikatory platformy, urządzenia, kontekstu i kolejki
cl_int kcl_GetPlatformIDs(cl_uint n, cl_platform_id *pl, cl_uint *ile) {
    if (init()) return CL_PLATFORM_NOT_FOUND_KHR;
    if (pl && n) *pl = (cl_platform_id)&kcl_wskaznik;
    if (ile) *ile = 1;
    return CL_SUCCESS;
}
cl_int kcl_GetDeviceIDs(cl_platform_id pl, cl_device_type t, cl_uint n, cl_device_id *d, cl_uint *ile) {
    (void)pl; (void)t;
    if (d && n) *d = (cl_device_id)&kcl_wskaznik;
    if (ile) *ile = 1;
    return CL_SUCCESS;
}
cl_context kcl_CreateContext(const cl_context_properties *pr, cl_uint n, const cl_device_id *d, void *f, void *u, cl_int *e) {
    (void)pr; (void)n; (void)d; (void)f; (void)u;
    *e = init() ? CL_OUT_OF_RESOURCES : CL_SUCCESS; return (cl_context)&kcl_wskaznik;
}
cl_command_queue kcl_CreateCommandQueueWithProperties(cl_context c, cl_device_id d, const cl_queue_properties *pr, cl_int *e) {
    (void)c; (void)d; (void)pr; *e = CL_SUCCESS; return (cl_command_queue)&kcl_wskaznik;
}

cl_int kcl_program(const char *plik, const char *opcje, cl_program *wynik) {
    for (int i = 0; i < kcl_nprogramow; i++) {
        const struct kcl_program_opis *pr = &kcl_programy[i];
        if (strcmp(pr->plik, plik)) continue;
        if ((opcje == NULL) != (pr->opcje == NULL) || (opcje && strcmp(opcje, pr->opcje))) continue;
        struct _cl_program *x = calloc(1, sizeof *x);
        x->o = pr; x->indeks = i; *wynik = x; return CL_SUCCESS;
    }
    return CL_INVALID_PROGRAM;
}

cl_kernel kcl_CreateKernel(cl_program pr, const char *nazwa, cl_int *e) {
    for (uint32_t j = 0; j < pr->o->njader; j++) {
        const struct kcl_jadro_opis *o = &pr->o->jadra[j];
        if (strcmp(o->nazwa, nazwa)) continue;
        if (o->nieobslugiwane[0]) {
            fprintf(stderr, "kcl: jadro %s: nieobslugiwane (%s)\n", nazwa, o->nieobslugiwane);
            *e = CL_INVALID_KERNEL_DEFINITION; return NULL;
        }
        struct _cl_kernel *k = calloc(1, sizeof *k);
        k->o = o; k->isa_off = isa_off[pr->indeks][j]; k->prog = pr->indeks;
        if (o->private_size) {   // pamięć prywatna: private_size × SIMD × (24 EU × 7 wątków), jak NEO
            uint64_t potrzeba = (uint64_t)o->private_size * o->simd * 24 * 7;
            if (!prywatna || prywatna->size < potrzeba) {
                if (prywatna) { kcl_wait(); kbo_zwolnij(prywatna); prywatna = NULL; }
                if (kbo_nowy(potrzeba, 0, &prywatna)) { *e = CL_OUT_OF_RESOURCES; free(k); return NULL; }
            }
        }
        *e = CL_SUCCESS; return k;
    }
    *e = CL_INVALID_KERNEL_NAME; return NULL;
}

// ---------------- pamięć ----------------
cl_mem kcl_CreateBuffer(cl_context c, cl_mem_flags f, size_t n, void *host, cl_int *e) {
    (void)c; (void)f;
    if (host) { *e = CL_INVALID_HOST_PTR; return NULL; }
    struct _cl_mem *m = calloc(1, sizeof *m);
    if (!m || kbo_nowy(n ? n : 1, 0, &m->bo)) { free(m); *e = CL_MEM_OBJECT_ALLOCATION_FAILURE; return NULL; }
    m->size = n; m->ref = 1; *e = CL_SUCCESS; return m;
}
cl_mem kcl_CreateSubBuffer(cl_mem b, cl_mem_flags f, cl_buffer_create_type t, const void *info, cl_int *e) {
    (void)f; (void)t;
    const cl_buffer_region *r = info;
    if (r->origin + r->size > b->size) { *e = CL_INVALID_VALUE; return NULL; }
    struct _cl_mem *m = calloc(1, sizeof *m);
    m->bo = b->bo; m->bo->ref++; m->off = b->off + r->origin; m->size = r->size; m->ref = 1;
    *e = CL_SUCCESS; return m;
}
cl_int kcl_ReleaseMemObject(cl_mem m) {
    if (m && --m->ref == 0) { kbo_zwolnij(m->bo); free(m); }
    return CL_SUCCESS;
}
cl_int kcl_GetMemObjectInfo(cl_mem m, cl_mem_info i, size_t n, void *v, size_t *ile) {
    if (i != CL_MEM_SIZE || n < sizeof(size_t)) return CL_INVALID_VALUE;
    *(size_t *)v = m->size; if (ile) *ile = sizeof(size_t); return CL_SUCCESS;
}

static void *usm_alloc(cl_context c, const cl_mem_properties_intel *pr, size_t n, cl_uint wyr, cl_int *e) {
    (void)c; (void)pr; (void)wyr;
    kbo *b;
    if (init() || kbo_nowy(n ? n : 1, 1, &b)) { *e = CL_OUT_OF_HOST_MEMORY; return NULL; }
    if (nusm == cusm) { cusm = cusm ? 2 * cusm : 64; usm = realloc(usm, cusm * sizeof *usm); }
    usm[nusm++] = (usm_t){ (uintptr_t)b->map, n ? n : 1, b };
    *e = CL_SUCCESS; return b->map;
}
static cl_int usm_free(cl_context c, void *ptr) {
    (void)c;
    for (int i = 0; i < nusm; i++) if (usm[i].p == (uintptr_t)ptr) {
        kcl_wait();   // clMemBlockingFreeINTEL: po zakończeniu pracy GPU
        kbo_zwolnij(usm[i].bo); usm[i] = usm[--nusm]; return CL_SUCCESS;
    }
    return CL_INVALID_VALUE;
}
static cl_int usm_info(cl_context c, const void *ptr, cl_mem_info_intel i, size_t n, void *v, size_t *ile) {
    (void)c; (void)ile;
    usm_t *u = usm_szukaj(ptr);
    if (i == CL_MEM_ALLOC_TYPE_INTEL) { *(cl_unified_shared_memory_type_intel *)v = u ? CL_MEM_TYPE_HOST_INTEL : CL_MEM_TYPE_UNKNOWN_INTEL; return CL_SUCCESS; }
    if (!u) return CL_INVALID_VALUE;
    if (i == CL_MEM_ALLOC_BASE_PTR_INTEL && n >= sizeof(void *)) { *(void **)v = (void *)u->p; return CL_SUCCESS; }
    if (i == CL_MEM_ALLOC_SIZE_INTEL && n >= sizeof(size_t)) { *(size_t *)v = u->n; return CL_SUCCESS; }
    return CL_INVALID_VALUE;
}

// ---------------- argumenty ----------------
cl_int kcl_SetKernelArg(cl_kernel k, cl_uint i, size_t n, const void *v) {
    if (i >= k->o->nargs) return CL_INVALID_ARG_INDEX;
    if (k->o->arg[i].wsk) {
        cl_mem m = v ? *(const cl_mem *)v : NULL;
        if (n != sizeof(cl_mem)) return CL_INVALID_ARG_SIZE;
        if (m && !m->bo && (up_rozwiaz() || !m->bo)) return CL_INVALID_MEM_OBJECT;   // obszar userptr do rozwiązania
        k->a[i].bo = m ? m->bo : NULL; k->a[i].adr = m ? m->bo->adr + m->off : 0; k->a[i].rozm = m ? m->size : 0;
    } else {
        if (n > 8 || (k->o->arg[i].size && n != (size_t)k->o->arg[i].size)) return CL_INVALID_ARG_SIZE;
        memset(k->a[i].v, 0, 8); memcpy(k->a[i].v, v, n);
    }
    k->a[i].ust = 1; return CL_SUCCESS;
}
static cl_int usm_arg(cl_kernel k, cl_uint i, const void *ptr) {
    if (i >= k->o->nargs || !k->o->arg[i].wsk) return CL_INVALID_ARG_INDEX;
    usm_t *u = usm_szukaj(ptr);
    if (!u) return CL_INVALID_ARG_VALUE;
    uint64_t off = (uintptr_t)ptr - u->p;
    // rozmiar powierzchni do końca obiektu GEM (strony), nie przydziału: sprzęt sprawdza granice bufora RAW co 4 B,
    // więc przy rozmiarze niepodzielnym przez 4 ostatni element FP16 czytał się jako 0 (NEO: alignUp(size, 4))
    k->a[i].bo = u->bo; k->a[i].adr = u->bo->adr + off; k->a[i].rozm = u->bo->size - off; k->a[i].ust = 1;
    return CL_SUCCESS;
}

void *kcl_GetExtensionFunctionAddressForPlatform(cl_platform_id pl, const char *n) {
    (void)pl;
    if (!strcmp(n, "clHostMemAllocINTEL")) return (void *)usm_alloc;
    if (!strcmp(n, "clMemBlockingFreeINTEL")) return (void *)usm_free;
    if (!strcmp(n, "clGetMemAllocInfoINTEL")) return (void *)usm_info;
    if (!strcmp(n, "clSetKernelArgMemPointerINTEL")) return (void *)usm_arg;
    return NULL;
}

// ---------------- zdarzenia i znaczniki czasu ----------------
static int znacznik_czasu(void) {   // PIPE_CONTROL z zapisem licznika czasu (12 MHz) po zakończeniu poprzedniej pracy
    if (!kprofil || !ts_bo) return -1;
    int i = (int)(ts_n++ % TS_MAX);
    EMIT(PIPE_CONTROL, .CommandStreamerStallEnable = 1, .PostSyncOperation = 3 /* zapis znacznika czasu */,
         .Address = ts_bo->adr + 8ull * i);
    return i;
}
static cl_event zdarzenie(int ts, int ts_k) {
    struct _cl_event *e = calloc(1, sizeof *e);
    e->seq = S[cur].seq; e->ts = ts; e->ts_k = ts_k; e->ref = 1; return e;
}

cl_int kcl_EnqueueMarkerWithWaitList(cl_command_queue q, cl_uint n, const cl_event *w, cl_event *ev) {
    (void)q; (void)n; (void)w;
    if (otworz_bufor()) return CL_OUT_OF_RESOURCES;
    int ts = znacznik_czasu();
    if (ev) *ev = zdarzenie(ts, ts);   // start = koniec = chwila znacznika
    return CL_SUCCESS;
}
cl_int kcl_Flush(cl_command_queue q) { (void)q; return wyslij() ? CL_OUT_OF_RESOURCES : CL_SUCCESS; }
cl_int kcl_Finish(cl_command_queue q) { (void)q; return kcl_wait(); }
cl_int kcl_WaitForEvents(cl_uint n, const cl_event *e) {
    for (cl_uint i = 0; i < n; i++) if (czekaj_seq(e[i]->seq)) return CL_OUT_OF_RESOURCES;
    return CL_SUCCESS;
}
cl_int kcl_GetEventInfo(cl_event e, cl_event_info i, size_t n, void *v, size_t *ile) {
    (void)ile;
    if (i != CL_EVENT_COMMAND_EXECUTION_STATUS || n < sizeof(cl_int)) return CL_INVALID_VALUE;
    if (cur >= 0 && S[cur].seq <= e->seq && wyslij()) return CL_OUT_OF_RESOURCES;
    cl_int st = CL_COMPLETE;
    if (e->seq > seq_gotowy) {
        struct drm_i915_gem_busy b = { .handle = S[e->seq % KSLOT].bb->h };
        if (S[e->seq % KSLOT].seq == e->seq && !ioctl(fd, DRM_IOCTL_I915_GEM_BUSY, &b) && b.busy) st = CL_RUNNING;
        else if (e->seq > seq_gotowy) seq_gotowy = e->seq;
    }
    *(cl_int *)v = st; return CL_SUCCESS;
}
cl_int kcl_GetEventProfilingInfo(cl_event e, cl_profiling_info i, size_t n, void *v, size_t *ile) {
    (void)ile;
    if (n < sizeof(cl_ulong)) return CL_INVALID_VALUE;
    if (czekaj_seq(e->seq)) return CL_OUT_OF_RESOURCES;
    cl_ulong t = 0;
    const int j = i == CL_PROFILING_COMMAND_END ? e->ts_k : e->ts;
    if (j >= 0) t = ((uint64_t *)ts_bo->map)[j] * 1000ull / 12;   // licznik 12 MHz -> ns
    *(cl_ulong *)v = t; return CL_SUCCESS;
}
cl_int kcl_ReleaseEvent(cl_event e) { if (e && --e->ref == 0) free(e); return CL_SUCCESS; }

// ---------------- wywołanie jądra ----------------
static uint32_t dzielnik(size_t g, uint32_t maks) {   // największy dzielnik g nie większy niż maks
    for (uint32_t d = maks; d > 1; d--) if (g % d == 0) return d;
    return 1;
}

static cl_int enqueue_(cl_command_queue q, cl_kernel k, cl_uint dim, const size_t *off, const size_t *glob,
                       const size_t *lok, cl_uint nw, const cl_event *w, cl_event *ev);
cl_int kcl_EnqueueNDRangeKernel(cl_command_queue q, cl_kernel k, cl_uint dim, const size_t *off, const size_t *glob,
                                const size_t *lok, cl_uint nw, const cl_event *w, cl_event *ev) {
    FAZA_POCZ; cl_int r = enqueue_(q, k, dim, off, glob, lok, nw, w, ev); FAZA_KON(F_ENQ); return r;
}
static cl_int enqueue_(cl_command_queue q, cl_kernel k, cl_uint dim, const size_t *off, const size_t *glob,
                       const size_t *lok, cl_uint nw, const cl_event *w, cl_event *ev) {
    (void)q; (void)nw; (void)w;
    const struct kcl_jadro_opis *o = k->o;
    for (uint32_t i = 0; i < o->nargs; i++) if (!k->a[i].ust) { fprintf(stderr, "kcl: %s: argument %u nieustawiony\n", o->nazwa, i); return CL_INVALID_KERNEL_ARGS; }
    size_t G[3] = { 1, 1, 1 }, O[3] = { 0, 0, 0 }; uint32_t L[3] = { 1, 1, 1 };
    for (cl_uint d = 0; d < dim; d++) { G[d] = glob[d]; if (off) O[d] = off[d]; }
    if (G[0] * G[1] * G[2] == 0) return CL_SUCCESS;
    if (lok) {
        for (cl_uint d = 0; d < dim; d++) { L[d] = (uint32_t)lok[d]; if (!L[d] || G[d] % L[d]) return CL_INVALID_WORK_GROUP_SIZE; }
    } else {   // jak NEO: rozmiar grupy dzielący rozmiar globalny; tu: co najwyżej 1 wątek SIMD na grupę
        L[0] = dzielnik(G[0], o->simd);
        if (dim > 1) L[1] = dzielnik(G[1], o->simd / L[0]);
        if (dim > 2) L[2] = dzielnik(G[2], o->simd / (L[0] * L[1]));
    }
    const uint32_t lt = L[0] * L[1] * L[2], watki = (lt + o->simd - 1) / o->simd;
    if (watki > 64) return CL_INVALID_WORK_GROUP_SIZE;
    const uint32_t pt = o->per_thread ? o->per_thread : 32;   // NEO: co najmniej 1 GRF danych na wątek
    const uint32_t dl = (o->cross + watki * pt + 63) & ~63u;

    if (cur >= 0 && (nwyw >= IDD_MAX || o_ind + dl > O_SS || o_ss + 64 * (o->nbti + 2) > O_SS + SS_LEN ||
                     (uint8_t *)p - (uint8_t *)S[cur].bb->map > BB - 2048 || nlista + o->nargs + 4 > MAXBO))
        if (wyslij()) return CL_OUT_OF_RESOURCES;
    if (otworz_bufor()) return CL_OUT_OF_RESOURCES;
    if (nwyw > 0) EMIT(PIPE_CONTROL, .CommandStreamerStallEnable = 1, .DCFlushEnable = 1);   // kolejka w kolejności
    uint8_t *D = S[cur].dyn->map, *ind = D + o_ind;
    memset(ind, 0, dl);
    for (int i = 0; i < 3; i++) {
        uint32_t g3[3] = { (uint32_t)G[0], (uint32_t)G[1], (uint32_t)G[2] }, o3[3] = { (uint32_t)O[0], (uint32_t)O[1], (uint32_t)O[2] };
        uint32_t n3[3] = { (uint32_t)(G[0] / L[0]), (uint32_t)(G[1] / L[1]), (uint32_t)(G[2] / L[2]) };
        if (o->off_ls[i] >= 0) memcpy(ind + o->off_ls[i], L, 12);
        if (o->off_els[i] >= 0) memcpy(ind + o->off_els[i], L, 12);
        if (o->off_gid[i] >= 0) memcpy(ind + o->off_gid[i], o3, 12);
        if (o->off_gs[i] >= 0) memcpy(ind + o->off_gs[i], g3, 12);
        if (o->off_ng[i] >= 0) memcpy(ind + o->off_ng[i], n3, 12);
    }
    if (o->off_wdim >= 0) memcpy(ind + o->off_wdim, &dim, 4);
    if (o->off_const >= 0) { uint64_t a = konst[k->prog] ? konst[k->prog]->adr : 0; memcpy(ind + o->off_const, &a, 8); dodaj_bo(konst[k->prog]); }
    if (o->off_priv >= 0) { uint64_t a = prywatna ? prywatna->adr : 0; memcpy(ind + o->off_priv, &a, 8); dodaj_bo(prywatna); }
    // argumenty; stateful: opis powierzchni-bufora (jak NEO encodeBuffer) i wpis w binding table
    uint32_t bt = 0; uint32_t *tab = NULL;
    if (o->nbti) {
        o_ss = (o_ss + 63) & ~63u; bt = o_ss - O_SS; tab = (uint32_t *)(D + o_ss);
        memset(tab, 0, 4 * o->nbti); o_ss = (o_ss + 4 * o->nbti + 63) & ~63u;
    }
    for (uint32_t i = 0; i < o->nargs; i++) {
        const struct kcl_arg_opis *a = &o->arg[i];
        if (!a->wsk) { if (a->off >= 0) memcpy(ind + a->off, k->a[i].v, a->size); continue; }
        dodaj_bo(k->a[i].bo);
        const uint64_t adr = k->a[i].adr;
        if (a->off >= 0) memcpy(ind + a->off, &adr, a->size);                 // stateless: adres 64-bit
        if (a->bti >= 0) {
            const uint64_t baza = adr & ~3ull, przes = adr & 3;               // baza powierzchni wyrównana do 4 B
            const uint64_t r = ((k->a[i].rozm + przes) + 3) & ~3ull, len = (r ? r : 4) - 1;   // jak NEO: alignUp(size, 4)
            if (przes && a->off_boff < 0) { fprintf(stderr, "kcl: %s: arg %u niewyrownany\n", o->nazwa, i); return CL_INVALID_ARG_VALUE; }
            struct GFX9_RENDER_SURFACE_STATE ss = { .SurfaceType = SURFTYPE_BUFFER, .SurfaceFormat = 0x1ff /* RAW */,
                .SurfaceVerticalAlignment = VALIGN_4, .TileMode = LINEAR, .MOCS = MOCS_WB,
                .Width = len & 0x7f, .Height = (len >> 7) & 0x3fff, .Depth = (len >> 21) & 0x7ff,
                .SurfaceBaseAddress = baza };
            GFX9_RENDER_SURFACE_STATE_pack(NULL, D + o_ss, &ss);
            tab[a->bti] = o_ss - O_SS; o_ss += 64;
            if (a->off_adr >= 0) memcpy(ind + a->off_adr, &baza, 8);
            if (a->off_boff >= 0) { uint32_t b32 = (uint32_t)przes; memcpy(ind + a->off_boff, &b32, 4); }
        }
    }
    // dane na wątek: lokalne ID (wiersz na wymiar: 32 B dla SIMD8/16, 64 B dla SIMD32)
    if (o->lid) {
        const uint32_t wiersz = o->simd == 32 ? 64 : 32;
        for (uint32_t t = 0; t < watki; t++) {
            uint16_t *x = (uint16_t *)(ind + o->cross + t * pt), *y = x + wiersz / 2, *z = y + wiersz / 2;
            for (uint32_t l = 0; l < o->simd; l++) {
                uint32_t lin = t * o->simd + l; if (lin >= lt) lin = lt - 1;
                x[l] = lin % L[0]; y[l] = (lin / L[0]) % L[1]; z[l] = lin / (L[0] * L[1]);
            }
        }
    }
    uint32_t slm = 0;
    // Kodowanie rozmiaru SLM w IDD (PRM KBL tom 2d, NEO computeSlmValues): log2(potęga2(max(slm, 1K))) − 9, czyli
    // 1K=1, 2K=2, 4K=3 … 64K=7. Uwaga: Mesa genxml (gen80) podaje inne kodowanie (4K=1, 8K=2, …) -- z nim grupy
    // dostawały za mało SLM i nachodziły na siebie (SYMM/TRMM: zły wynik przy więcej niż jednej grupie).
    if (o->slm) { uint32_t v = 1024; slm = 1; while (v < o->slm) { v *= 2; slm++; } slm_w_buforze = 1; }
    struct GFX9_INTERFACE_DESCRIPTOR_DATA idd = {
        .KernelStartPointer = k->isa_off, .NumberofThreadsinGPGPUThreadGroup = watki,
        .CrossThreadConstantDataReadLength = o->cross / 32, .ConstantURBEntryReadLength = pt / 32,
        .BindingTablePointer = bt, .BindingTableEntryCount = o->nbti < 32 ? o->nbti : 31,
        .SharedLocalMemorySize = slm, .BarrierEnable = o->bariera };
    GFX9_INTERFACE_DESCRIPTOR_DATA_pack(NULL, D + 32 * nwyw, &idd);
    const int ts = ev || kprofil_jadra ? znacznik_czasu() : -1;
    const uint32_t reszta = lt % o->simd;
    EMIT(GPGPU_WALKER, .InterfaceDescriptorOffset = nwyw, .IndirectDataLength = dl, .IndirectDataStartAddress = o_ind,
         .SIMDSize = o->simd == 8 ? SIMD8 : o->simd == 16 ? SIMD16 : SIMD32, .ThreadWidthCounterMaximum = watki - 1,
         .ThreadGroupIDXDimension = (uint32_t)(G[0] / L[0]), .ThreadGroupIDYDimension = (uint32_t)(G[1] / L[1]),
         .ThreadGroupIDZDimension = (uint32_t)(G[2] / L[2]),
         .RightExecutionMask = reszta ? (1u << reszta) - 1 : (o->simd == 32 ? 0xffffffffu : (1u << o->simd) - 1),
         .BottomExecutionMask = 0xffffffff);
    EMIT(MEDIA_STATE_FLUSH);
    if (ev || kprofil_jadra) {
        const int ts_k = znacznik_czasu();
        if (ev) *ev = zdarzenie(ts, ts_k);
        if (kprofil_jadra && npom < TS_MAX / 2 - 1) pom[npom++] = (pomiar_t){ o->nazwa, ts, ts_k };
        if (kprofil_jadra && npom >= TS_MAX / 2 - 8 && kcl_wait()) return CL_OUT_OF_RESOURCES;
    }
    o_ind += dl; nwyw++;
    static int kazde = -1; if (kazde < 0) kazde = getenv("KCL_WAIT_EACH") != NULL;   // diagnostyka
    if (kazde && kcl_wait()) return CL_OUT_OF_RESOURCES;
    return CL_SUCCESS;
}
