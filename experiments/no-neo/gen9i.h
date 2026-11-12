// kpgi: uruchamianie jąder na UHD 620 (Gen9) bezpośrednio przez i915, bez NEO. Opis: ../BEZNEO.md.
// Jeden nagłówek: obiekty pamięci (GEM, softpin), bufor komend (preambuła, wywołanie jądra, bariera), wysłanie, czekanie.
#pragma once
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

#define KPGI_SPR(x) do { if ((x) != 0) { fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, strerror(errno)); exit(1); } } while (0)
#define KPGI_MOCS_WB (2 << 1)   // [HIP] indeks 2 tabeli MOCS i915 (SKL) = WB, jak w Mesie

// Opis jądra (generuje kernels.py z .ze_info): dane wspólne (cross-thread) i na wątek (lokalne ID).
// arg[i]: off/size -- miejsce wartości w danych wspólnych (-1: brak), bti -- indeks w binding table (-1: stateless/wartość),
// off_adr / off_boff -- offsety buffer_address (adres 64-bit) i buffer_offset (przesunięcie w powierzchni) dla stateful.
struct kpgi_jadro {
    const char *nazwa; const uint8_t *isa; uint32_t isa_len;
    uint32_t simd, grf, cross, per_thread, nargs;
    struct { int off, size, bti, off_adr, off_boff; } arg[16];
    int off_local_size[3], off_enq_local_size[3], off_gid_offset[3];
    uint32_t nbti;
};

// Obiekt pamięci GPU: uchwyt GEM, adres w PPGTT (softpin), mapowanie dla CPU.
struct kpgi_bo { uint32_t h; uint64_t adr, size; void *map; int pisany; };

// Sterta (jeden obiekt GEM; wszystkie adresy bazowe stanu na nią wskazują):
//   0x00000 deskryptory IDD (64 × 32 B, po jednym na wywołanie jądra w buforze)
//   0x01000 dane pośrednie (cross-thread + per-thread)
//   0x40000 binding tables i opisy powierzchni (64 KB): tu wskazuje Surface State Base Address, bo Binding Table
//           Pointer w IDD to bity 5..15 -- najwyżej 64 KB od tej bazy (przekroczenie: wskaźnik obcięty, zapisy giną)
//   0x80000 kod jąder (wyrównany do 64 B, za każdym 1 KB zer na prefetch instrukcji)
enum { KPGI_STERTA = 2 << 20, KPGI_BB = 64 << 10, KPGI_IDD_MAX = 64,
       KPGI_O_IND = 0x1000, KPGI_O_SS = 0x40000, KPGI_O_KER = 0x80000 };

struct kpgi {
    int fd; uint32_t ctx;
    uint64_t nast_adr;                       // następny wolny adres w PPGTT (prosty przydział rosnący)
    struct kpgi_bo sterta, bb;               // sterta: IDD, dane, powierzchnie, kod jąder; bb: bufor komend
    uint32_t o_ind, o_ss, o_ker, nwyw;       // zajęte miejsce w stercie, liczba wywołań w bieżącym buforze
    uint32_t *p, *idl;                       // kursor w buforze komend; MEDIA_INTERFACE_DESCRIPTOR_LOAD do uzupełnienia
    struct drm_i915_gem_exec_object2 ob[64]; uint32_t nob;
    const struct kpgi_jadro *jadra[32]; uint32_t ker_off[32], njader;
};

static inline void kpgi_bo_nowy(struct kpgi *g, struct kpgi_bo *b, uint64_t size) {
    size = (size + 4095) & ~4095ull;
    struct drm_i915_gem_create c = { .size = size };
    KPGI_SPR(ioctl(g->fd, DRM_IOCTL_I915_GEM_CREATE, &c));
    struct drm_i915_gem_mmap_offset m = { .handle = c.handle, .flags = I915_MMAP_OFFSET_WB };
    KPGI_SPR(ioctl(g->fd, DRM_IOCTL_I915_GEM_MMAP_OFFSET, &m));
    b->map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, g->fd, m.offset);
    if (b->map == MAP_FAILED) { perror("mmap"); exit(1); }
    b->h = c.handle; b->size = size; b->adr = g->nast_adr; b->pisany = 1;
    g->nast_adr += (size + 0xffff) & ~0xffffull;   // odstęp 64 KB między obiektami
}

static inline void kpgi_init(struct kpgi *g) {
    memset(g, 0, sizeof *g);
    g->fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
    if (g->fd < 0) { perror("renderD128"); exit(1); }
    struct drm_i915_gem_context_create cc = { 0 };
    KPGI_SPR(ioctl(g->fd, DRM_IOCTL_I915_GEM_CONTEXT_CREATE, &cc));
    g->ctx = cc.ctx_id; g->nast_adr = 0x10000000;
    kpgi_bo_nowy(g, &g->sterta, KPGI_STERTA); g->sterta.pisany = 0;
    kpgi_bo_nowy(g, &g->bb, KPGI_BB); g->bb.pisany = 0;
    memset(g->sterta.map, 0, KPGI_STERTA);
    g->o_ker = KPGI_O_KER;
}

// Rejestruje jądro: kod do sterty. Zwraca numer jądra.
static inline uint32_t kpgi_jadro_dodaj(struct kpgi *g, const struct kpgi_jadro *j) {
    uint32_t i = g->njader++;
    if (g->o_ker + j->isa_len + 1024 > KPGI_STERTA) { fprintf(stderr, "kpgi: brak miejsca na kod jader\n"); exit(1); }
    g->jadra[i] = j; g->ker_off[i] = g->o_ker;
    memcpy((uint8_t *)g->sterta.map + g->o_ker, j->isa, j->isa_len);
    g->o_ker = (g->o_ker + j->isa_len + 1024 + 63) & ~63u;
    return i;
}

#define KPGI_EMIT(T, ...) do { struct GFX9_##T c_ = { GFX9_##T##_header, __VA_ARGS__ }; GFX9_##T##_pack(NULL, g->p, &c_); g->p += GFX9_##T##_length; } while (0)

// Początek bufora: synchronizacja, potok GPGPU, adresy bazowe (wszystkie = sterta), VFE jak NEO, tablica IDD.
static inline void kpgi_poczatek(struct kpgi *g) {
    g->p = g->bb.map; g->o_ind = KPGI_O_IND; g->o_ss = KPGI_O_SS; g->nob = 0; g->nwyw = 0;
    const uint64_t H = g->sterta.adr; const uint32_t S = KPGI_STERTA / 4096;
    KPGI_EMIT(PIPE_CONTROL, .CommandStreamerStallEnable = 1, .DCFlushEnable = 1);
    KPGI_EMIT(PIPELINE_SELECT, .MaskBits = 3, .PipelineSelection = GPGPU);
    KPGI_EMIT(STATE_BASE_ADDRESS,
        .GeneralStateBaseAddressModifyEnable = 1, .GeneralStateBaseAddress = 0, .GeneralStateMOCS = KPGI_MOCS_WB,
        .StatelessDataPortAccessMOCS = KPGI_MOCS_WB,
        .SurfaceStateBaseAddressModifyEnable = 1, .SurfaceStateBaseAddress = H + KPGI_O_SS, .SurfaceStateMOCS = KPGI_MOCS_WB,
        .DynamicStateBaseAddressModifyEnable = 1, .DynamicStateBaseAddress = H, .DynamicStateMOCS = KPGI_MOCS_WB,
        .IndirectObjectBaseAddressModifyEnable = 1, .IndirectObjectBaseAddress = H, .IndirectObjectMOCS = KPGI_MOCS_WB,
        .InstructionBaseAddressModifyEnable = 1, .InstructionBaseAddress = H, .InstructionMOCS = KPGI_MOCS_WB,
        .GeneralStateBufferSizeModifyEnable = 1, .GeneralStateBufferSize = 0xfffff,
        .DynamicStateBufferSizeModifyEnable = 1, .DynamicStateBufferSize = S,
        .IndirectObjectBufferSizeModifyEnable = 1, .IndirectObjectBufferSize = S,
        .InstructionBuffersizeModifyEnable = 1, .InstructionBufferSize = S,
        .BindlessSurfaceStateBaseAddressModifyEnable = 1, .BindlessSurfaceStateMOCS = KPGI_MOCS_WB,
        .BindlessSurfaceStateSize = 0xfffff);
    KPGI_EMIT(PIPE_CONTROL, .CommandStreamerStallEnable = 1, .StateCacheInvalidationEnable = 1,
              .InstructionCacheInvalidateEnable = 1, .ConstantCacheInvalidationEnable = 1);
    KPGI_EMIT(MEDIA_VFE_STATE, .MaximumNumberofThreads = 24 * 7 - 1, .NumberofURBEntries = 1,
              .URBEntryAllocationSize = 0x782, .CURBEAllocationSize = 0);
    g->idl = g->p;   // długość tablicy IDD uzupełnia kpgi_wyslij (po jednym IDD na wywołanie)
    KPGI_EMIT(MEDIA_INTERFACE_DESCRIPTOR_LOAD, .InterfaceDescriptorTotalLength = 32,
              .InterfaceDescriptorDataStartAddress = 0);
}

// Wywołanie jądra nr j na siatce gx × gy × gz grup; grupa = simd × 1 × 1 work-itemów = 1 wątek. args[i]: wartość
// argumentu i (wskaźnik: adres w PPGTT; przez wartość: młodsze bajty); rozm[i]: rozmiar bufora dla wskaźników przez
// binding table (NULL albo 0: 1 GB).
static inline void kpgi_wywolaj(struct kpgi *g, uint32_t j, uint32_t gx, uint32_t gy, uint32_t gz,
                                const uint64_t *args, const uint64_t *rozm) {
    const struct kpgi_jadro *k = g->jadra[j];
    uint8_t *H = g->sterta.map;
    if (g->nwyw >= KPGI_IDD_MAX) { fprintf(stderr, "kpgi: ponad %d wywolan w jednym buforze\n", KPGI_IDD_MAX); exit(1); }
    // dane pośrednie
    uint8_t *ind = H + g->o_ind;
    memset(ind, 0, k->cross + k->per_thread);
    uint32_t ls[3] = { k->simd, 1, 1 };
    for (int i = 0; i < 3; i++) {
        if (k->off_local_size[i] >= 0) memcpy(ind + k->off_local_size[i], ls, 12);
        if (k->off_enq_local_size[i] >= 0) memcpy(ind + k->off_enq_local_size[i], ls, 12);
    }
    // binding table (wpisy: offsety opisów powierzchni względem surface state base) i opisy powierzchni-buforów
    uint32_t bt = 0;
    if (k->nbti) {
        g->o_ss = (g->o_ss + 63) & ~63u;
        bt = g->o_ss - KPGI_O_SS; uint32_t *tab = (uint32_t *)(H + g->o_ss);   // bt: względem surface state base
        g->o_ss = (g->o_ss + 4 * k->nbti + 63) & ~63u;
        for (uint32_t i = 0; i < k->nbti; i++) tab[i] = 0;
        for (uint32_t i = 0; i < k->nargs; i++) {
            if (k->arg[i].bti < 0) continue;
            const uint64_t r = rozm && rozm[i] ? rozm[i] : (1ull << 30), len = r - 1;
            struct GFX9_RENDER_SURFACE_STATE ss = { .SurfaceType = SURFTYPE_BUFFER, .SurfaceFormat = 0x1ff /* RAW */,
                .SurfaceVerticalAlignment = VALIGN_4, .TileMode = LINEAR, .MOCS = KPGI_MOCS_WB,
                .Width = len & 0x7f, .Height = (len >> 7) & 0x3fff, .Depth = (len >> 21) & 0x7ff,
                .SurfaceBaseAddress = args[i] };
            GFX9_RENDER_SURFACE_STATE_pack(NULL, H + g->o_ss, &ss);
            tab[k->arg[i].bti] = g->o_ss - KPGI_O_SS;
            g->o_ss += 64;
            if (k->arg[i].off_adr >= 0) memcpy(ind + k->arg[i].off_adr, &args[i], 8);
            // buffer_offset = 0: adres bazowy powierzchni = adres bufora (wyrównany)
        }
        if (g->o_ss > KPGI_O_SS + 0x10000) { fprintf(stderr, "kpgi: brak miejsca na opisy powierzchni (64 KB)\n"); exit(1); }
    }
    for (uint32_t i = 0; i < k->nargs; i++)
        if (k->arg[i].off >= 0) memcpy(ind + k->arg[i].off, &args[i], k->arg[i].size);
    // lokalne ID: X = 0..simd-1, Y = Z = 0; wiersz na wymiar: 32 B (SIMD8/16) albo 64 B (SIMD32)
    uint16_t *lid = (uint16_t *)(ind + k->cross);
    for (uint32_t i = 0; i < k->simd; i++) lid[i] = i;
    const uint32_t len = (k->cross + k->per_thread + 63) & ~63u;
    // deskryptor jądra dla tego wywołania
    struct GFX9_INTERFACE_DESCRIPTOR_DATA idd = {
        .KernelStartPointer = g->ker_off[j], .NumberofThreadsinGPGPUThreadGroup = 1,
        .CrossThreadConstantDataReadLength = k->cross / 32, .ConstantURBEntryReadLength = k->per_thread / 32,
        .BindingTablePointer = bt, .BindingTableEntryCount = k->nbti < 32 ? k->nbti : 31 };
    GFX9_INTERFACE_DESCRIPTOR_DATA_pack(NULL, H + 32 * g->nwyw, &idd);
    KPGI_EMIT(GPGPU_WALKER, .InterfaceDescriptorOffset = g->nwyw, .IndirectDataLength = len, .IndirectDataStartAddress = g->o_ind,
              .SIMDSize = k->simd == 8 ? SIMD8 : k->simd == 16 ? SIMD16 : SIMD32,
              .ThreadGroupIDXDimension = gx, .ThreadGroupIDYDimension = gy, .ThreadGroupIDZDimension = gz,
              .RightExecutionMask = k->simd == 32 ? 0xffffffff : (1u << k->simd) - 1, .BottomExecutionMask = 0xffffffff);
    static int bez_msf = -1;
    if (bez_msf < 0) bez_msf = getenv("KPGI_BEZ_MSF") != NULL;   // do testów: bez MEDIA_STATE_FLUSH po walkerze
    if (!bez_msf) KPGI_EMIT(MEDIA_STATE_FLUSH);
    g->o_ind += len; g->nwyw++;
    if (g->o_ind > KPGI_O_SS) { fprintf(stderr, "kpgi: brak miejsca na dane posrednie\n"); exit(1); }
}

// Bariera między zależnymi jądrami: zatrzymanie command streamera do końca poprzednich + zapis danych do L3.
static inline void kpgi_bariera(struct kpgi *g) {
    KPGI_EMIT(PIPE_CONTROL, .CommandStreamerStallEnable = 1, .DCFlushEnable = 1);
}

// Koniec bufora i wysłanie. bo[]: obiekty danych używane przez jądra.
static inline void kpgi_wyslij(struct kpgi *g, struct kpgi_bo **bo, uint32_t nbo) {
    KPGI_EMIT(PIPE_CONTROL, .CommandStreamerStallEnable = 1, .DCFlushEnable = 1);
    KPGI_EMIT(MI_BATCH_BUFFER_END);
    { struct GFX9_MEDIA_INTERFACE_DESCRIPTOR_LOAD l = { GFX9_MEDIA_INTERFACE_DESCRIPTOR_LOAD_header,
          .InterfaceDescriptorTotalLength = 32 * (g->nwyw ? g->nwyw : 1), .InterfaceDescriptorDataStartAddress = 0 };
      GFX9_MEDIA_INTERFACE_DESCRIPTOR_LOAD_pack(NULL, g->idl, &l); }
    if ((g->p - (uint32_t *)g->bb.map) & 1) *g->p++ = 0;
    const uint64_t F = EXEC_OBJECT_PINNED | EXEC_OBJECT_SUPPORTS_48B_ADDRESS;
    g->nob = 0;
    for (uint32_t i = 0; i < nbo; i++)
        g->ob[g->nob++] = (struct drm_i915_gem_exec_object2){ .handle = bo[i]->h, .offset = bo[i]->adr,
            .flags = F | (bo[i]->pisany ? EXEC_OBJECT_WRITE : 0) };
    g->ob[g->nob++] = (struct drm_i915_gem_exec_object2){ .handle = g->sterta.h, .offset = g->sterta.adr, .flags = F };
    g->ob[g->nob++] = (struct drm_i915_gem_exec_object2){ .handle = g->bb.h, .offset = g->bb.adr, .flags = F };
    struct drm_i915_gem_execbuffer2 eb = { .buffers_ptr = (uintptr_t)g->ob, .buffer_count = g->nob,
        .batch_len = (uint32_t)((g->p - (uint32_t *)g->bb.map) * 4), .flags = I915_EXEC_RENDER | I915_EXEC_NO_RELOC,
        .rsvd1 = g->ctx };
    KPGI_SPR(ioctl(g->fd, DRM_IOCTL_I915_GEM_EXECBUFFER2, &eb));
}

// Czekanie na koniec ostatniego wysłania (śpi w jądrze Linuksa). Zwraca 0 albo -1 po przekroczeniu limitu.
static inline int kpgi_czekaj(struct kpgi *g, int64_t limit_ns) {
    struct drm_i915_gem_wait w = { .bo_handle = g->bb.h, .timeout_ns = limit_ns };
    return ioctl(g->fd, DRM_IOCTL_I915_GEM_WAIT, &w) ? -1 : 0;
}
