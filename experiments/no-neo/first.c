// Pierwszy eksperyment bez NEO: jądro store.cl (p[i] = v + i, SIMD16, stateless) wysłane własnym buforem komend
// przez i915 (execbuffer2, softpin), czekanie przez GEM_WAIT z limitem 2 s. Opis: ../BEZNEO.md.
// Użycie: ./pierwszy [liczba_grup]   (grupa = 1 wątek SIMD16 = 16 elementów)
#include <drm/i915_drm.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include "gen9_pack.h"

#define SPR(x) do { if ((x) != 0) { fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, strerror(errno)); exit(1); } } while (0)

// Adresy w PPGTT (softpin) i układ sterty (heap): wszystkie adresy bazowe stanu = HEAP.
enum { HEAP = 0x10000000, OUT = 0x20000000, BATCH = 0x30000000 };
enum { O_IDD = 0x0, O_IND = 0x1000, O_KER = 0x2000, HEAP_SIZE = 0x10000 };
#define MOCS_WB (2 << 1)   // [HIP] indeks 2 tabeli MOCS i915 dla SKL = WB (jak w Mesie); do potwierdzenia

static int fd;
static double teraz(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static uint32_t bo_nowy(uint64_t size, void **map) {
    struct drm_i915_gem_create c = { .size = size };
    SPR(ioctl(fd, DRM_IOCTL_I915_GEM_CREATE, &c));
    struct drm_i915_gem_mmap_offset m = { .handle = c.handle, .flags = I915_MMAP_OFFSET_WB };
    SPR(ioctl(fd, DRM_IOCTL_I915_GEM_MMAP_OFFSET, &m));
    *map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, m.offset);
    if (*map == MAP_FAILED) { perror("mmap"); exit(1); }
    memset(*map, 0, size);
    return c.handle;
}

int main(int argc, char **argv) {
    const uint32_t grupy = argc > 1 ? (uint32_t)atoi(argv[1]) : 16, n = grupy * 16, v = 1000;
    fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
    if (fd < 0) { perror("renderD128"); return 1; }
    struct drm_i915_gem_context_create cc = { 0 };
    SPR(ioctl(fd, DRM_IOCTL_I915_GEM_CONTEXT_CREATE, &cc));

    // Jądro (ISA wycięte z zebin przez extract.py)
    uint8_t isa[4096]; FILE *f = fopen("zapis.isa", "rb"); if (!f) { perror("zapis.isa"); return 1; }
    size_t nisa = fread(isa, 1, sizeof isa, f); fclose(f);

    uint8_t *heap; uint32_t h_heap = bo_nowy(HEAP_SIZE, (void **)&heap);
    uint32_t *out; uint32_t h_out = bo_nowy((n * 4 + 4095) & ~4095u, (void **)&out);
    uint32_t *bb; uint32_t h_bb = bo_nowy(4096, (void **)&bb);
    for (uint32_t i = 0; i < n; i++) out[i] = 0xdeadbeef;
    memcpy(heap + O_KER, isa, nisa);   // za jądrem zera (prefetch instrukcji czyta dalej)

    // Deskryptor jądra (INTERFACE_DESCRIPTOR_DATA, 32 B)
    struct GFX9_INTERFACE_DESCRIPTOR_DATA idd = {
        .KernelStartPointer = O_KER, .NumberofThreadsinGPGPUThreadGroup = 1,
        .CrossThreadConstantDataReadLength = 2,   // 64 B danych wspólnych = 2 GRF (r4, r5)
        .ConstantURBEntryReadLength = 3,          // dane na wątek: lokalne ID X, Y, Z = 3 GRF (r1..r3)
    };
    GFX9_INTERFACE_DESCRIPTOR_DATA_pack(NULL, heap + O_IDD, &idd);

    // Dane pośrednie: najpierw wspólne (offsety z .ze_info), potem na wątek (lokalne ID, u16 × 16)
    uint8_t *ind = heap + O_IND;
    uint32_t lsz[3] = { 16, 1, 1 }; uint64_t adr_p = OUT;
    memcpy(ind + 12, lsz, 12);        // local_size
    memcpy(ind + 32, &adr_p, 8);      // p (stateless, adres 64-bit)
    memcpy(ind + 40, &v, 4);          // v
    memcpy(ind + 44, lsz, 12);        // enqueued_local_size
    uint16_t *lid = (uint16_t *)(ind + 64);
    for (int i = 0; i < 16; i++) lid[i] = i;   // X = 0..15; Y, Z = 0
    const uint32_t ind_len = 192;              // 64 + 96, wyrównane do 64

    // Bufor komend
    uint32_t *p = bb;
    struct GFX9_PIPE_CONTROL pc0 = { GFX9_PIPE_CONTROL_header, .CommandStreamerStallEnable = 1, .DCFlushEnable = 1 };
    GFX9_PIPE_CONTROL_pack(NULL, p, &pc0); p += GFX9_PIPE_CONTROL_length;
    struct GFX9_PIPELINE_SELECT ps = { GFX9_PIPELINE_SELECT_header, .MaskBits = 3, .PipelineSelection = GPGPU };
    GFX9_PIPELINE_SELECT_pack(NULL, p, &ps); p += GFX9_PIPELINE_SELECT_length;
    struct GFX9_STATE_BASE_ADDRESS sba = { GFX9_STATE_BASE_ADDRESS_header,
        .GeneralStateBaseAddressModifyEnable = 1, .GeneralStateBaseAddress = 0, .GeneralStateMOCS = MOCS_WB,
        .StatelessDataPortAccessMOCS = MOCS_WB,
        .SurfaceStateBaseAddressModifyEnable = 1, .SurfaceStateBaseAddress = HEAP, .SurfaceStateMOCS = MOCS_WB,
        .DynamicStateBaseAddressModifyEnable = 1, .DynamicStateBaseAddress = HEAP, .DynamicStateMOCS = MOCS_WB,
        .IndirectObjectBaseAddressModifyEnable = 1, .IndirectObjectBaseAddress = HEAP, .IndirectObjectMOCS = MOCS_WB,
        .InstructionBaseAddressModifyEnable = 1, .InstructionBaseAddress = HEAP, .InstructionMOCS = MOCS_WB,
        .GeneralStateBufferSizeModifyEnable = 1, .GeneralStateBufferSize = 0xfffff,
        .DynamicStateBufferSizeModifyEnable = 1, .DynamicStateBufferSize = HEAP_SIZE / 4096,
        .IndirectObjectBufferSizeModifyEnable = 1, .IndirectObjectBufferSize = HEAP_SIZE / 4096,
        .InstructionBuffersizeModifyEnable = 1, .InstructionBufferSize = HEAP_SIZE / 4096,
        .BindlessSurfaceStateBaseAddressModifyEnable = 1, .BindlessSurfaceStateBaseAddress = 0,
        .BindlessSurfaceStateMOCS = MOCS_WB, .BindlessSurfaceStateSize = 0xfffff };
    GFX9_STATE_BASE_ADDRESS_pack(NULL, p, &sba); p += GFX9_STATE_BASE_ADDRESS_length;
    struct GFX9_PIPE_CONTROL pc1 = { GFX9_PIPE_CONTROL_header, .CommandStreamerStallEnable = 1,
        .StateCacheInvalidationEnable = 1, .InstructionCacheInvalidateEnable = 1, .ConstantCacheInvalidationEnable = 1 };
    GFX9_PIPE_CONTROL_pack(NULL, p, &pc1); p += GFX9_PIPE_CONTROL_length;
    struct GFX9_MEDIA_VFE_STATE vfe = { GFX9_MEDIA_VFE_STATE_header,
        .MaximumNumberofThreads = 24 * 7 - 1, .NumberofURBEntries = 1, .URBEntryAllocationSize = 0x782,
        .CURBEAllocationSize = 0 };   // jak NEO (preamble_bdw_and_later.inl)
    GFX9_MEDIA_VFE_STATE_pack(NULL, p, &vfe); p += GFX9_MEDIA_VFE_STATE_length;
    struct GFX9_MEDIA_INTERFACE_DESCRIPTOR_LOAD midl = { GFX9_MEDIA_INTERFACE_DESCRIPTOR_LOAD_header,
        .InterfaceDescriptorTotalLength = 32, .InterfaceDescriptorDataStartAddress = O_IDD };
    GFX9_MEDIA_INTERFACE_DESCRIPTOR_LOAD_pack(NULL, p, &midl); p += GFX9_MEDIA_INTERFACE_DESCRIPTOR_LOAD_length;
    struct GFX9_GPGPU_WALKER w = { GFX9_GPGPU_WALKER_header,
        .InterfaceDescriptorOffset = 0, .IndirectDataLength = ind_len, .IndirectDataStartAddress = O_IND,
        .SIMDSize = SIMD16, .ThreadWidthCounterMaximum = 0, .ThreadHeightCounterMaximum = 0, .ThreadDepthCounterMaximum = 0,
        .ThreadGroupIDXDimension = grupy, .ThreadGroupIDYDimension = 1, .ThreadGroupIDZDimension = 1,
        .RightExecutionMask = 0xffff, .BottomExecutionMask = 0xffffffff };
    GFX9_GPGPU_WALKER_pack(NULL, p, &w); p += GFX9_GPGPU_WALKER_length;
    struct GFX9_MEDIA_STATE_FLUSH msf = { GFX9_MEDIA_STATE_FLUSH_header };
    GFX9_MEDIA_STATE_FLUSH_pack(NULL, p, &msf); p += GFX9_MEDIA_STATE_FLUSH_length;
    struct GFX9_PIPE_CONTROL pc2 = { GFX9_PIPE_CONTROL_header, .CommandStreamerStallEnable = 1, .DCFlushEnable = 1 };
    GFX9_PIPE_CONTROL_pack(NULL, p, &pc2); p += GFX9_PIPE_CONTROL_length;
    struct GFX9_MI_BATCH_BUFFER_END end = { GFX9_MI_BATCH_BUFFER_END_header };
    GFX9_MI_BATCH_BUFFER_END_pack(NULL, p, &end); p += GFX9_MI_BATCH_BUFFER_END_length;
    if ((p - bb) & 1) *p++ = 0;   // MI_NOOP: długość wielokrotnością 8 B
    const uint32_t bb_len = (uint32_t)((p - bb) * 4);

    struct drm_i915_gem_exec_object2 ob[3] = {
        { .handle = h_heap, .offset = HEAP, .flags = EXEC_OBJECT_PINNED | EXEC_OBJECT_SUPPORTS_48B_ADDRESS },
        { .handle = h_out, .offset = OUT, .flags = EXEC_OBJECT_PINNED | EXEC_OBJECT_SUPPORTS_48B_ADDRESS | EXEC_OBJECT_WRITE },
        { .handle = h_bb, .offset = BATCH, .flags = EXEC_OBJECT_PINNED | EXEC_OBJECT_SUPPORTS_48B_ADDRESS },
    };
    struct drm_i915_gem_execbuffer2 eb = { .buffers_ptr = (uintptr_t)ob, .buffer_count = 3, .batch_len = bb_len,
        .flags = I915_EXEC_RENDER | I915_EXEC_NO_RELOC, .rsvd1 = cc.ctx_id };
    printf("bufor komend: %u B, grup %u, elementow %u\n", bb_len, grupy, n);
    SPR(ioctl(fd, DRM_IOCTL_I915_GEM_EXECBUFFER2, &eb));
    struct drm_i915_gem_wait wt = { .bo_handle = h_bb, .timeout_ns = 2000000000 };
    if (ioctl(fd, DRM_IOCTL_I915_GEM_WAIT, &wt)) {
        fprintf(stderr, "GEM_WAIT: %s (limit 2 s) -- GPU nie skonczylo\n", strerror(errno)); return 2;
    }
    uint32_t zle = 0;
    for (uint32_t i = 0; i < n; i++) if (out[i] != v + i) { if (zle < 5) printf("  out[%u] = 0x%x (oczekiwane %u)\n", i, out[i], v + i); zle++; }
    printf("wynik: %s (%u zlych z %u)\n", zle ? "ZLE" : "OK", zle, n);
    if (zle) return 3;

    // Koszt: wysłanie + czekanie, powtórzone (to samo jądro)
    const int R = 2000; double cz[R], c0 = clock() / (double)CLOCKS_PER_SEC, t0 = teraz();
    for (int r = 0; r < R; r++) {
        double a = teraz();
        SPR(ioctl(fd, DRM_IOCTL_I915_GEM_EXECBUFFER2, &eb));
        wt.timeout_ns = 2000000000; SPR(ioctl(fd, DRM_IOCTL_I915_GEM_WAIT, &wt));
        cz[r] = teraz() - a;
    }
    double sc = teraz() - t0, cpu = clock() / (double)CLOCKS_PER_SEC - c0;
    for (int i = 1; i < R; i++) { double x = cz[i]; int j = i; while (j > 0 && cz[j - 1] > x) { cz[j] = cz[j - 1]; j--; } cz[j] = x; }
    printf("execbuffer2 + GEM_WAIT, %d powtorzen: mediana %.1f us, min %.1f us, CPU %.1f us/wywolanie (%.0f%% rdzenia)\n",
           R, cz[R / 2] * 1e6, cz[0] * 1e6, cpu / R * 1e6, 100 * cpu / sc);
    return 0;
}
