// kcl: the OpenCL subset used by gen9blas.c, implemented directly on i915 (Gen9: UHD 620, HD 530; Gen8 Cherryview: docs/gen8.md), without NEO. Facts and measurements: ../docs/no-neo.md.
//
// Model: an in-order queue = a batch buffer built on the fly; each clEnqueueNDRangeKernel appends
// IDD + indirect data + (for stateful arguments) binding table and surface states + GPGPU_WALKER; between kernels
// a PIPE_CONTROL barrier (CS stall + DC flush) -- without it consecutive walkers overlap (docs/no-neo.md 6d).
// The buffer is submitted (execbuffer2, softpin) on clFlush/clFinish/kcl_wait, when full (64 IDD) or when
// an event status is queried. Batch buffers rotate (KSLOT); before a slot is reused we wait for it to finish.
// Memory: cl_mem = GEM object (PPGTT address fixed: softpin; Gen9: increasing allocation, addresses are never reused;
// Gen8: 4 GB space with reuse, see kbo_addr); "USM host" (gen9blas_alloc) = GEM object mapped WB to the CPU (Gen9 LLC:
// coherent with the GPU; Gen8: made snooped, see kbo_new), pointers inside are translated by a table.
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
#include "kcl_gen8.h"
#include <time.h>

static uint32_t MOCS_WB = 2 << 1;   // [HIP] index 2 of the i915 MOCS table (SKL) = WB, like Mesa; KCL_MOCS=n: a different index (diagnostics)
#define L3_SLM 0x60000321u          // L3CNTLREG with SLM (NEO gen9/reg_configs.h)
#define L3_NO_SLM 0x80000340u       // L3CNTLREG without SLM
// Gen8 Cherryview (docs/gen8.md): MOCS is not a table index but a direct encoding, 0x78 = WB + L3 (Mesa isl.c); L3CNTLREG
// without bit 9 (no GPU hang on L3 parity errors, which would also oblige us to zero SLM/URB), always the SLM layout
// (SLM is inconsistent after every reconfiguration on CHV; Beignet: the same value, "random hang" when switching).
#define MOCS_WB_GEN8 0x78u
#define L3_GEN8 0x60000121u
// Device parameters, read in init(): Gen9.5 (default) or Gen8 Cherryview.
// llc = 0 (CHV): CPU caches are not shared with the GPU; gtt_end != 0: small (32-bit) GPU address space, see kbo_addr.
static struct { int gen8, llc; uint32_t eus, max_group_threads; uint64_t ts_freq, gtt_end;
                const struct kcl_program_desc *progs; int nprogs; } dev;
enum { KSLOT = 8, DYN = 1 << 20, BB = 128 << 10, IDD_MAX = 64,
       O_IND = 0x1000, O_SS = 0xC0000, SS_LEN = 0x10000,   // dyn: IDD | indirect data | binding tables + surface states
       MAXBO = 512, TS_MAX = 4096 };

// ---------------- GEM objects ----------------
typedef struct kbo { uint32_t h; uint64_t addr, size; void *map; int ref, ro; uint64_t mark; } kbo;   // ro: read-only userptr
struct _cl_mem { kbo *bo; uint64_t off, size; struct _cl_mem *parent; int ref; };
struct _cl_program { const struct kcl_program_desc *o; int idx; };
struct _cl_kernel {
    const struct kcl_kernel_desc *o; uint32_t isa_off; int prog;
    struct { kbo *bo; uint64_t addr, sz; uint8_t v[8]; int set; } a[24];
};
struct _cl_event { uint64_t seq; int ts, ts_k; int ref; };   // ts, ts_k: start and end timestamps (-1: none)

static int fd = -1; static uint32_t ctx_id; static int kprofile, kprofile_kernels;
// KCL_PROFILE_KERNELS=1: timestamps around each kernel, the sum of times per name is printed at process exit
typedef struct { const char *name; int ts0, ts1; } meas_t;
static meas_t meas[TS_MAX / 2]; static int nmeas;
static struct { const char *name; double ms; long n; } sums[256]; static int nsum;
static uint64_t next_addr = 1ull << 32;
static kbo *isa, *consts[16], *priv_bo, *ts_bo;
static uint32_t *isa_off[16];   // [program][kernel] -> offset in isa
static uint32_t ts_n;

static uint64_t kbo_addr(uint64_t size);
// CPU access to a GEM object. With LLC (Gen9) both mappings are WB and coherent. Without LLC (CHV, docs/gen8.md section 4)
// a plain object is not coherent with CPU caches: KMAP_RW objects (the CPU reads them) are made snooped (SET_CACHING
// CACHED), KMAP_WRITE objects (the CPU only writes them: batch, indirect data, ISA, constants) are mapped WC.
enum { KMAP_NONE, KMAP_RW, KMAP_WRITE };
static int kbo_new(uint64_t size, int map, kbo **out) {
    size = (size + 4095) & ~4095ull;
    kbo *b = calloc(1, sizeof *b);
    struct drm_i915_gem_create c = { .size = size };
    if (!b || ioctl(fd, DRM_IOCTL_I915_GEM_CREATE, &c)) { free(b); return -1; }
    b->h = c.handle; b->size = size; b->ref = 1; b->addr = kbo_addr(size);
    if (!b->addr) { struct drm_gem_close cl = { .handle = b->h }; ioctl(fd, DRM_IOCTL_GEM_CLOSE, &cl); free(b); return -1; }
    if (!dev.llc && map == KMAP_RW) {
        struct drm_i915_gem_caching ca = { .handle = b->h, .caching = I915_CACHING_CACHED };
        if (ioctl(fd, DRM_IOCTL_I915_GEM_SET_CACHING, &ca)) return -1;
    }
    if (map != KMAP_NONE) {
        struct drm_i915_gem_mmap_offset m = { .handle = b->h,
            .flags = !dev.llc && map == KMAP_WRITE ? I915_MMAP_OFFSET_WC : I915_MMAP_OFFSET_WB };
        if (ioctl(fd, DRM_IOCTL_I915_GEM_MMAP_OFFSET, &m)) return -1;
        b->map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, m.offset);
        if (b->map == MAP_FAILED) { b->map = NULL; return -1; }
    }
    *out = b; return 0;
}
static uint64_t seq_next = 1, seq_sent = 0, seq_done = 0;   // batch sequence numbers: next, last submitted, last finished
// Gen8 (4 GB PPGTT): freed address ranges, reusable once the GPU has finished every batch that could use them (seq).
typedef struct { uint64_t addr, size, seq; } range_t;
static range_t *fr; static int nfr, cfr;
static uint64_t seq_in_use(void);
static int kcl_wait_all(void);
static void range_free(uint64_t addr, uint64_t size) {   // keeps fr sorted by address, merges neighbours
    const uint64_t seq = seq_in_use();
    int i = 0; while (i < nfr && fr[i].addr < addr) i++;
    if (i > 0 && fr[i - 1].addr + fr[i - 1].size == addr) {
        fr[i - 1].size += size; if (seq > fr[i - 1].seq) fr[i - 1].seq = seq; i--;
    } else {
        if (nfr == cfr) { cfr = cfr ? 2 * cfr : 64; fr = realloc(fr, cfr * sizeof *fr); }
        memmove(fr + i + 1, fr + i, (nfr - i) * sizeof *fr); fr[i] = (range_t){ addr, size, seq }; nfr++;
    }
    if (i + 1 < nfr && fr[i].addr + fr[i].size == fr[i + 1].addr) {
        fr[i].size += fr[i + 1].size; if (fr[i + 1].seq > fr[i].seq) fr[i].seq = fr[i + 1].seq;
        memmove(fr + i + 1, fr + i + 2, (nfr - i - 2) * sizeof *fr); nfr--;
    }
}
static uint64_t range_take(uint64_t size) {   // first fit among the ranges the GPU no longer uses; 0: none
    for (int i = 0; i < nfr; i++) {
        if (fr[i].seq > seq_done || fr[i].size < size) continue;
        const uint64_t a = fr[i].addr;
        fr[i].addr += size; fr[i].size -= size;
        if (!fr[i].size) { memmove(fr + i, fr + i + 1, (nfr - i - 1) * sizeof *fr); nfr--; }
        return a;
    }
    return 0;
}
// Object address in the PPGTT (softpin). Gen9: increasing allocation, addresses are never reused. Gen8: below
// dev.gtt_end, freed ranges reused (range_free/range_take); when the space is exhausted, wait for the GPU and retry.
static uint64_t kbo_addr(uint64_t size) {
    if (dev.gtt_end) {
        size = (size + 0xffff) & ~0xffffull;
        uint64_t a = range_take(size);
        if (!a && next_addr + size > dev.gtt_end && nfr && !kcl_wait_all()) a = range_take(size);
        if (a) return a;
        if (next_addr + size > dev.gtt_end) return 0;
        a = next_addr; next_addr += size; return a;
    }
    // PPGTT address: objects >= 2 MB are aligned to 2 MB (like NEO) -- only then can i915 use 2 MB pages in the GPU
    // page table (fewer TLB misses on large matrices); smaller ones to 64 KB.
    static int mode = -1, counter;
    if (mode < 0) mode = getenv("KCL_ADDR") ? atoi(getenv("KCL_ADDR")) : 0;   // diagnostics: 0 64K, 1 2M, 2 colour
    uint64_t align = mode == 1 && size >= (2u << 20) ? (2u << 20) : 0x10000;
    next_addr = (next_addr + align - 1) & ~(align - 1);
    if (mode == 2) next_addr += ((uint64_t)(counter++ * 5) % 16) * 0x1000;   // offset by 0..15 x 4 KB
    const uint64_t a = next_addr;
    next_addr += (size + 0xffff) & ~0xffffull;
    return a;
}
static void kbo_release(kbo *b) {
    if (!b || --b->ref > 0) return;
    if (dev.gtt_end) range_free(b->addr, (b->size + 0xffff) & ~0xffffull);
    if (b->map) munmap(b->map, b->size);
    struct drm_gem_close c = { .handle = b->h };
    ioctl(fd, DRM_IOCTL_GEM_CLOSE, &c);   // safe even if the GPU is still using it (i915 holds a reference)
    free(b);
}

// ---------------- USM host: GEM objects mapped to the CPU ----------------
typedef struct { uintptr_t p; uint64_t n; kbo *bo; } usm_t;
static usm_t *usm; static int nusm, cusm;
static usm_t *usm_find(const void *ptr) {
    uintptr_t x = (uintptr_t)ptr;
    for (int i = 0; i < nusm; i++) if (x >= usm[i].p && x < usm[i].p + usm[i].n) return &usm[i];
    return NULL;
}

// ---------------- userptr: ordinary process memory (e.g. NumPy arrays) as a GEM object, with a cache ----------------
// Facts and measurements: docs/no-neo.md 6g, experiments/no-neo/userptr.c, test_userptr.py, bench/compare/userptr.py.
// - Pinning the pages (on the first execbuf) costs ~0.45 us of CPU per 4 KB page (IOMMU mapping), i.e. as much
//   as a copy -- it pays off only for an object kept between calls (then the GPU reads it as fast as its own buffer).
// - When the process memory changes (munmap, a new mapping), the Linux kernel (mmu notifier) invalidates the object's
//   pages and re-pins them on the next execbuf -- a cached object stays valid.
// - Regions of ONE operation that share pages must go through one object: two objects on the same pages
//   are two GPU addresses of the same memory line, and writes to one line through two addresses in one kernel get
//   lost (measured: swap/rot, x[0] in the same line as the end of y).
// - A cached object is used only for exactly the same page range as the (sum of) regions of the current
//   operation: that is memory just supplied by the caller, hence ordinary and mapped. A wider object could include
//   pages taken since then by something else (e.g. mmap of a GEM object: no struct page, pinning -> EFAULT).
// Hence lazily: kcl_userptr records the region and returns a cl_mem "to be resolved"; on the first clSetKernelArg with such a
// cl_mem all regions of the operation (gen9blas sets the arguments after all regions) are split into groups of overlapping
// ranges, and each group gets one object: from the cache (the same range of the sum) or a new one.
// Operation boundary: kcl_new_operation() (gen9blas: at the start of each BLAS function). A region registered after resolving: the object
// of this operation if it contains it, otherwise NULL (copy through the CPU).
enum { UP_MAX = 256, UP_MEM = 64, UP_REG = 16 };
typedef struct { uintptr_t s, e; kbo *bo; uint64_t used, epoch; } up_t;
static up_t up[UP_MAX]; static int nup; static uint64_t up_counter, up_bytes, up_limit, up_epoch = 1;
static struct _cl_mem up_mem[UP_MEM]; static int up_mem_i;   // cl_mem for arguments (valid within the current operation)
static struct { uintptr_t p, s, e; int wr; struct _cl_mem *m; } reg[UP_REG];   // regions of the current operation
static int nreg, resolved; static uint64_t reg_epoch;
static int cur;
static int submit(void), init(void);
// ---- diagnostics: KCL_PHASES=1 -- wall time and thread CPU time per phase (on stderr at exit) ----
enum { F_ENQ, F_EXEC, F_WAIT, F_UP, F_N };
static const char *f_name[F_N] = { "EnqueueNDRange", "execbuf (ioctl)", "GEM_WAIT (ioctl)", "userptr" };
static struct { double sc, cpu; long n; } fz[F_N];
static int phases = -1; static long fz_objects;
static double t_sc(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static double t_cpu(void) { struct timespec t; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static void phases_print(void) {
    for (int i = 0; i < F_N; i++) if (fz[i].n)
        fprintf(stderr, "kcl phase %-18s %8ld x  wall %8.2f us  CPU %8.2f us (per call)\n", f_name[i], fz[i].n, fz[i].sc / fz[i].n, fz[i].cpu / fz[i].n);
    if (fz[F_EXEC].n) fprintf(stderr, "kcl phase objects per execbuf: %.1f\n", (double)fz_objects / fz[F_EXEC].n);
}
static int phases_on(void) { if (phases < 0) { phases = getenv("KCL_PHASES") != NULL; if (phases) atexit(phases_print); } return phases; }
#define PHASE_BEGIN double fs_ = phases_on() ? t_sc() : 0, fc_ = phases ? t_cpu() : 0
#define PHASE_END(i) do { if (phases) { fz[i].sc += t_sc() - fs_; fz[i].cpu += t_cpu() - fc_; fz[i].n++; } } while (0)
static uint64_t S_cur_seq(void);
void kcl_new_operation(void) { up_epoch++; }
static void up_remove(int i) {
    if (cur >= 0 && up[i].bo->mark == S_cur_seq()) submit();   // object in the buffer being built: submit first
    up_bytes -= up[i].e - up[i].s;
    kbo_release(up[i].bo);   // GEM_CLOSE even if the GPU is still using it (i915 holds a reference)
    up[i] = up[--nup];
}
// Object for the page range [s, e): from the cache (exactly this range) or a new one. The memory limit is soft:
// objects of the current operation are not removed.
static up_t *up_object(uintptr_t s, uintptr_t e, int wr) {
    for (int i = 0; i < nup; i++)
        if (up[i].s == s && up[i].e == e && (!wr || !up[i].bo->ro)) { up[i].used = ++up_counter; up[i].epoch = up_epoch; return &up[i]; }
    while (nup == UP_MAX || (nup && up_bytes + (e - s) > up_limit)) {   // make room: the least recently used one outside the operation
        int j = -1;
        for (int i = 0; i < nup; i++) if (up[i].epoch != up_epoch && (j < 0 || up[i].used < up[j].used)) j = i;
        if (j < 0) break;
        up_remove(j);
    }
    if (nup == UP_MAX) return NULL;
    struct drm_i915_gem_userptr u = { .user_ptr = s, .user_size = e - s, .flags = wr ? 0 : I915_USERPTR_READ_ONLY };
    kbo *b = calloc(1, sizeof *b);
    if (!b || ioctl(fd, DRM_IOCTL_I915_GEM_USERPTR, &u)) { free(b); return NULL; }
    b->h = u.handle; b->size = e - s; b->ref = 1; b->ro = !wr; b->addr = kbo_addr(e - s);
    if (!b->addr) { struct drm_gem_close cl = { .handle = b->h }; ioctl(fd, DRM_IOCTL_GEM_CLOSE, &cl); free(b); return NULL; }
    up[nup] = (up_t){ s, e, b, ++up_counter, up_epoch }; up_bytes += e - s;
    return &up[nup++];
}
static void up_assign(struct _cl_mem *m, uintptr_t p, up_t *x) {
    // size up to the end of the object (pages), like usm_arg: the hardware checks RAW buffer bounds every 4 B
    m->bo = x->bo; m->off = p - x->s; m->size = x->bo->size - (p - x->s);
}
// Resolving the regions of the current operation: groups of overlapping ranges -> one object per group.
static int up_resolve(void) {
    if (resolved || reg_epoch != up_epoch) return 0;
    resolved = 1;
    int group[UP_REG];
    for (int i = 0; i < nreg; i++) group[i] = i;
    for (int changed = 1; changed; ) {   // connected components (overlapping page ranges)
        changed = 0;
        for (int i = 0; i < nreg; i++) for (int j = 0; j < nreg; j++)
            if (group[i] != group[j] && reg[i].s < reg[j].e && reg[j].s < reg[i].e) {
                int g = group[i] < group[j] ? group[i] : group[j];
                group[i] = group[j] = g; changed = 1;
            }
        for (int i = 0; i < nreg; i++) while (group[group[i]] != group[i]) group[i] = group[group[i]];
    }
    for (int g = 0; g < nreg; g++) {
        uintptr_t s = UINTPTR_MAX, e = 0; int wr = 0, found = 0;
        for (int i = 0; i < nreg; i++) if (group[i] == g) {
            found = 1; if (reg[i].s < s) s = reg[i].s;
            if (reg[i].e > e) e = reg[i].e;
            wr |= reg[i].wr;
        }
        if (!found) continue;
        up_t *x = up_object(s, e, wr);
        if (!x) return -1;
        kbo *b = x->bo;   // x may move in the array on later groups -- remember the object
        for (int i = 0; i < nreg; i++) if (group[i] == g) {
            reg[i].m->bo = b; reg[i].m->off = reg[i].p - s; reg[i].m->size = b->size - (reg[i].p - s);
        }
    }
    return 0;
}
static cl_mem kcl_userptr_(const void *ptr, size_t n, int wr);
cl_mem kcl_userptr(const void *ptr, size_t n, int wr) { PHASE_BEGIN; cl_mem m = kcl_userptr_(ptr, n, wr); PHASE_END(F_UP); return m; }
static cl_mem kcl_userptr_(const void *ptr, size_t n, int wr) {
    if (init() || !ptr || !n) return NULL;
    const uintptr_t p = (uintptr_t)ptr, s = p & ~4095ul, e = (p + n + 4095) & ~4095ul;
    if (e - s > up_limit) return NULL;   // too big to hold: copy through the CPU
    if (reg_epoch != up_epoch) { reg_epoch = up_epoch; nreg = 0; resolved = 0; }
    struct _cl_mem *m = &up_mem[up_mem_i++ % UP_MEM];
    *m = (struct _cl_mem){ .bo = NULL, .off = 0, .size = n, .ref = 1 };
    if (resolved || nreg == UP_REG) {   // after resolving: an object of this operation containing the region, or a separate one
        int overlap = 0;
        for (int i = 0; i < nup; i++)
            if (up[i].epoch == up_epoch && up[i].s < e && s < up[i].e) {
                if (up[i].s <= s && e <= up[i].e && (!wr || !up[i].bo->ro)) { up_assign(m, p, &up[i]); return m; }
                overlap = 1;
            }
        if (overlap) return NULL;
        up_t *x = up_object(s, e, wr);
        if (!x) return NULL;
        up_assign(m, p, x); return m;
    }
    reg[nreg++] = (typeof(reg[0])){ p, s, e, wr, m };
    return m;
}

// ---------------- batch buffer ----------------
typedef struct { kbo *dyn, *bb; uint64_t seq; } slot_t;
static slot_t S[KSLOT];
static int cur = -1;                   // slot of the buffer being built (-1: none)
static uint64_t S_cur_seq(void) { return S[cur].seq; }
// GPU resets that hit our context (batch_active + batch_pending from I915_GET_RESET_STATS). After a reset (e.g. a kernel
// running longer than the engine's preempt_timeout_ms while another context waits) GEM_WAIT still succeeds, but the work
// was cut off: the results are wrong. The count is compared after every wait and a change is reported as an error.
static uint32_t resets_seen;
static uint32_t reset_count(void) {
    struct drm_i915_reset_stats rs = { .ctx_id = ctx_id };
    if (ioctl(fd, DRM_IOCTL_I915_GET_RESET_STATS, &rs)) return resets_seen;
    return rs.batch_active + rs.batch_pending;
}
static uint32_t *p, *idl, *lri, o_ind, o_ss, nlaunch; static int slm_in_batch;
static kbo *bo_list[MAXBO]; static int nbo_list;
static int l3_always_slm = 0;   // KCL_L3=slm: always L3 with SLM (like NEO on KBL/CFL); measured: no difference, off by default

#define EMIT(T, ...) do { struct GFX9_##T c_ = { GFX9_##T##_header, __VA_ARGS__ }; GFX9_##T##_pack(NULL, p, &c_); p += GFX9_##T##_length; } while (0)

static void add_bo(kbo *b) {
    if (!b || b->mark == S[cur].seq) return;
    b->mark = S[cur].seq; bo_list[nbo_list++] = b;
}

static int wait_slot(int s) {
    if (!S[s].seq || S[s].seq <= seq_done) return 0;
    struct drm_i915_gem_wait w = { .bo_handle = S[s].bb->h, .timeout_ns = 60ll * 1000000000 };
    PHASE_BEGIN;
    while (ioctl(fd, DRM_IOCTL_I915_GEM_WAIT, &w)) { if (errno != EINTR) return -1; }
    PHASE_END(F_WAIT);
    if (S[s].seq > seq_done) seq_done = S[s].seq;   // in-order queue: earlier ones are done too
    uint32_t rc = reset_count();
    if (rc != resets_seen) {
        resets_seen = rc;
        fprintf(stderr, "kcl: GPU reset during our work (results lost; a kernel too long for preemption?)\n");
        return -1;
    }
    return 0;
}

static int open_batch(void) {
    if (cur >= 0) return 0;
    int s = (int)(seq_next % KSLOT);
    if (wait_slot(s)) return -1;
    cur = s; S[s].seq = seq_next++;
    nbo_list = 0; nlaunch = 0; slm_in_batch = 0; o_ind = O_IND; o_ss = O_SS;
    p = S[s].bb->map;
    const uint64_t D = S[s].dyn->addr;
    EMIT(PIPE_CONTROL, .CommandStreamerStallEnable = 1, .DCFlushEnable = 1);
    lri = p;   // L3CNTLREG: value filled in at submit time (with SLM if any kernel uses it)
    EMIT(MI_LOAD_REGISTER_IMM, .RegisterOffset = 0x7034, .DataDWord = L3_NO_SLM);
    // EU thread arbitration: round-robin (DebugControlReg2 0xE404 = 0x100), like NEO on Gen9 (gfx_core_helper_gen9.cpp:
    // getDefaultThreadArbitrationPolicy, command_encoder_gen9.cpp). KCL_ARB=age: the hardware default (diagnostics).
    // Not on Gen8: the register's bits are not documented for CHV and NEO does not program it on Gen8.
    static int arb = -1; if (arb < 0) arb = !(getenv("KCL_ARB") && !strcmp(getenv("KCL_ARB"), "age"));
    if (arb && !dev.gen8) EMIT(MI_LOAD_REGISTER_IMM, .RegisterOffset = 0xE404, .DataDWord = 0x100);
    if (dev.gen8) {
        p = kcl_gen8_pipeline_select_gpgpu(p);
        // WaSendDummyVFEafterPipelineSelect (PRM CHV vol16): MEDIA_VFE_STATE right after the GPGPU PIPELINE_SELECT
        EMIT(MEDIA_VFE_STATE, .MaximumNumberofThreads = dev.eus * 7 - 1, .NumberofURBEntries = 1,
             .URBEntryAllocationSize = 0x782, .CURBEAllocationSize = 0);
        const struct kcl_sba sba = { .surface = D + O_SS, .dynamic = D, .indirect = D, .instruction = isa->addr,
            .mocs = MOCS_WB, .dynamic_size = DYN / 4096, .indirect_size = DYN / 4096,
            .instruction_size = (uint32_t)(isa->size / 4096) };
        p = kcl_gen8_state_base_address(p, &sba);
    } else {
        EMIT(PIPELINE_SELECT, .MaskBits = 3, .PipelineSelection = GPGPU);
        EMIT(STATE_BASE_ADDRESS,
            .GeneralStateBaseAddressModifyEnable = 1, .GeneralStateBaseAddress = 0, .GeneralStateMOCS = MOCS_WB,
            .StatelessDataPortAccessMOCS = MOCS_WB,
            .SurfaceStateBaseAddressModifyEnable = 1, .SurfaceStateBaseAddress = D + O_SS, .SurfaceStateMOCS = MOCS_WB,
            .DynamicStateBaseAddressModifyEnable = 1, .DynamicStateBaseAddress = D, .DynamicStateMOCS = MOCS_WB,
            .IndirectObjectBaseAddressModifyEnable = 1, .IndirectObjectBaseAddress = D, .IndirectObjectMOCS = MOCS_WB,
            .InstructionBaseAddressModifyEnable = 1, .InstructionBaseAddress = isa->addr, .InstructionMOCS = MOCS_WB,
            .GeneralStateBufferSizeModifyEnable = 1, .GeneralStateBufferSize = 0xfffff,
            .DynamicStateBufferSizeModifyEnable = 1, .DynamicStateBufferSize = DYN / 4096,
            .IndirectObjectBufferSizeModifyEnable = 1, .IndirectObjectBufferSize = DYN / 4096,
            .InstructionBuffersizeModifyEnable = 1, .InstructionBufferSize = (uint32_t)(isa->size / 4096),
            .BindlessSurfaceStateBaseAddressModifyEnable = 1, .BindlessSurfaceStateMOCS = MOCS_WB,
            .BindlessSurfaceStateSize = 0xfffff);
    }
    // Gen8: CS stall needs one of the flush/post-sync bits as well (PRM CHV vol02a p.1098) -> DC flush
    EMIT(PIPE_CONTROL, .CommandStreamerStallEnable = 1, .StateCacheInvalidationEnable = 1,
         .InstructionCacheInvalidateEnable = 1, .ConstantCacheInvalidationEnable = 1, .TextureCacheInvalidationEnable = 1,
         .DCFlushEnable = dev.gen8);
    EMIT(MEDIA_VFE_STATE, .MaximumNumberofThreads = dev.eus * 7 - 1, .NumberofURBEntries = 1,
         .URBEntryAllocationSize = 0x782, .CURBEAllocationSize = 0);
    idl = p;   // IDD table length filled in at submit time
    EMIT(MEDIA_INTERFACE_DESCRIPTOR_LOAD, .InterfaceDescriptorTotalLength = 32, .InterfaceDescriptorDataStartAddress = 0);
    // Only objects in use go on the execbuf list: program constants and private memory are added by the kernel that needs them
    // (kcl_EnqueueNDRangeKernel). i915 pays for every object (VMA pinning, synchronization): 15 -> 5 objects
    // for a small axpy, measurement results/overhead-2026-11-12.txt.
    add_bo(isa); if (ts_bo) add_bo(ts_bo);
    return 0;
}

static int submit(void) {
    if (cur < 0) return 0;
    slot_t *s = &S[cur];
    EMIT(PIPE_CONTROL, .CommandStreamerStallEnable = 1, .DCFlushEnable = 1);
    EMIT(MI_BATCH_BUFFER_END);
    if ((p - (uint32_t *)s->bb->map) & 1) *p++ = 0;
    { struct GFX9_MI_LOAD_REGISTER_IMM l = { GFX9_MI_LOAD_REGISTER_IMM_header, .RegisterOffset = 0x7034,
          .DataDWord = dev.gen8 ? L3_GEN8 : (slm_in_batch || l3_always_slm) ? L3_SLM : L3_NO_SLM };
      GFX9_MI_LOAD_REGISTER_IMM_pack(NULL, lri, &l); }
    { struct GFX9_MEDIA_INTERFACE_DESCRIPTOR_LOAD l = { GFX9_MEDIA_INTERFACE_DESCRIPTOR_LOAD_header,
          .InterfaceDescriptorTotalLength = 32 * (nlaunch ? nlaunch : 1), .InterfaceDescriptorDataStartAddress = 0 };
      GFX9_MEDIA_INTERFACE_DESCRIPTOR_LOAD_pack(NULL, idl, &l); }
    static struct drm_i915_gem_exec_object2 ob[MAXBO + 2];
    const uint64_t F = EXEC_OBJECT_PINNED | EXEC_OBJECT_SUPPORTS_48B_ADDRESS;
    int n = 0;
    for (int i = 0; i < nbo_list; i++)
        ob[n++] = (struct drm_i915_gem_exec_object2){ .handle = bo_list[i]->h, .offset = bo_list[i]->addr, .flags = F | (bo_list[i]->ro ? 0 : EXEC_OBJECT_WRITE) };
    ob[n++] = (struct drm_i915_gem_exec_object2){ .handle = s->dyn->h, .offset = s->dyn->addr, .flags = F };
    ob[n++] = (struct drm_i915_gem_exec_object2){ .handle = s->bb->h, .offset = s->bb->addr, .flags = F };
    struct drm_i915_gem_execbuffer2 eb = { .buffers_ptr = (uintptr_t)ob, .buffer_count = n,
        .batch_len = (uint32_t)((p - (uint32_t *)s->bb->map) * 4), .flags = I915_EXEC_RENDER | I915_EXEC_NO_RELOC,
        .rsvd1 = ctx_id };
    int r;
    if (!dev.llc) __builtin_ia32_sfence();   // WC writes (batch, indirect data) out of the CPU's write-combining buffers
    PHASE_BEGIN;
    while ((r = ioctl(fd, DRM_IOCTL_I915_GEM_EXECBUFFER2, &eb)) && errno == EINTR) {}
    PHASE_END(F_EXEC); if (phases) fz_objects += n;
    seq_sent = s->seq; cur = -1;
    if (r) {
        fprintf(stderr, "kcl: execbuffer2: %s\n", strerror(errno));
        return -1;
    }
    return 0;
}

static int wait_seq(uint64_t seq) {
    if (cur >= 0 && S[cur].seq <= seq && submit()) return -1;
    if (seq <= seq_done) return 0;
    return wait_slot((int)(seq % KSLOT));
}

static void collect_measurements(void) {   // after waiting for all work: kernel times into the sums
    for (int i = 0; i < nmeas; i++) {
        const uint64_t *t = ts_bo->map; double ms = (double)(t[meas[i].ts1] - t[meas[i].ts0]) * 1e3 / dev.ts_freq;
        int j = 0; while (j < nsum && strcmp(sums[j].name, meas[i].name)) j++;
        if (j == nsum && nsum < 256) { sums[nsum].name = meas[i].name; nsum++; }
        if (j < 256) { sums[j].ms += ms; sums[j].n++; }
    }
    nmeas = 0;
}
static void print_measurements(void) {
    collect_measurements();
    for (int j = 0; j < nsum; j++) fprintf(stderr, "kcl: %-22s %6ld x  %10.2f ms\n", sums[j].name, sums[j].n, sums[j].ms);
}

cl_int kcl_wait(void) {
    uint64_t last = cur >= 0 ? S[cur].seq : seq_sent;
    if (wait_seq(last)) return CL_OUT_OF_RESOURCES;
    if (kprofile_kernels) collect_measurements();
    return CL_SUCCESS;
}
int kcl_gen8(void) { return init() ? 0 : dev.gen8; }
static uint64_t seq_in_use(void) { return cur >= 0 ? S[cur].seq : seq_sent; }   // the last batch that may use an object
static int kcl_wait_all(void) { return kcl_wait() != CL_SUCCESS; }

// ---------------- initialization ----------------
static int getparam(int param, int *value) {
    struct drm_i915_getparam g = { .param = param, .value = value };
    return ioctl(fd, DRM_IOCTL_I915_GETPARAM, &g);
}
static int init(void) {
    if (fd >= 0) return 0;
    fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
    if (fd < 0) return -1;
    struct drm_i915_gem_context_create cc = { 0 };
    if (ioctl(fd, DRM_IOCTL_I915_GEM_CONTEXT_CREATE, &cc)) return -1;
    ctx_id = cc.ctx_id;
    int chip = 0, eus = 0, subslices = 0, freq = 0;
    if (getparam(I915_PARAM_CHIPSET_ID, &chip) || getparam(I915_PARAM_EU_TOTAL, &eus) ||
        getparam(I915_PARAM_SUBSLICE_TOTAL, &subslices) || getparam(I915_PARAM_CS_TIMESTAMP_FREQUENCY, &freq) ||
        eus <= 0 || subslices <= 0 || freq <= 0) return -1;
    dev.gen8 = (chip & ~3) == 0x22b0;   // Cherryview/Braswell 0x22b0-0x22b3 (Mesa crocus_pci_ids.h)
    int llc = 0;
    struct drm_i915_gem_context_param gp = { .ctx_id = ctx_id, .param = I915_CONTEXT_PARAM_GTT_SIZE };
    if (getparam(I915_PARAM_HAS_LLC, &llc) || ioctl(fd, DRM_IOCTL_I915_GEM_CONTEXT_GETPARAM, &gp)) return -1;
    dev.llc = llc;
    // A 32-bit PPGTT (CHV: 4 GB, docs/gen8.md): allocate from 1 MB up to 64 KB below the end; softpin above it fails (EINVAL).
    if (gp.value <= (1ull << 32)) { dev.gtt_end = gp.value - 0x10000; next_addr = 1 << 20; }
    dev.eus = (uint32_t)eus; dev.ts_freq = (uint64_t)freq;
    // Threads per work-group: Gen8 at most the threads of one subslice (PRM CHV vol02d p.223: needed for barriers;
    // applied to every kernel to keep it simple); Gen9 unchanged.
    dev.max_group_threads = dev.gen8 ? (uint32_t)(eus / subslices) * 7 : 64;
    dev.progs = dev.gen8 ? kcl_programs_gen8 : kcl_programs;
    dev.nprogs = dev.gen8 ? kcl_nprograms_gen8 : kcl_nprograms;
    if (dev.gen8) MOCS_WB = MOCS_WB_GEN8;
    resets_seen = reset_count();
    kprofile_kernels = getenv("KCL_PROFILE_KERNELS") != NULL;
    kprofile = getenv("GEN9BLAS_PROFILE") != NULL || kprofile_kernels;
    const char *um = getenv("KCL_USERPTR_MB");
    up_limit = (uint64_t)(um && *um ? atoi(um) : 1024) << 20;   // pinned process memory
    if (getenv("KCL_MOCS")) MOCS_WB = (uint32_t)atoi(getenv("KCL_MOCS")) << (dev.gen8 ? 0 : 1);   // Gen8: the raw value
    if (getenv("KCL_L3") && !strcmp(getenv("KCL_L3"), "slm")) l3_always_slm = 1;
    uint64_t sz = 0;
    for (int i = 0; i < dev.nprogs; i++)
        for (uint32_t j = 0; j < dev.progs[i].nkernels; j++) sz += (dev.progs[i].kernels[j].isa_len + 1024 + 63) & ~63u;
    if (kbo_new(sz + 4096, KMAP_WRITE, &isa)) return -1;
    uint64_t o = 0;
    for (int i = 0; i < dev.nprogs; i++) {
        const struct kcl_program_desc *pr = &dev.progs[i];
        isa_off[i] = calloc(pr->nkernels, sizeof(uint32_t));
        for (uint32_t j = 0; j < pr->nkernels; j++) {
            isa_off[i][j] = (uint32_t)o;
            memcpy((uint8_t *)isa->map + o, pr->kernels[j].isa, pr->kernels[j].isa_len);
            o += (pr->kernels[j].isa_len + 1024 + 63) & ~63u;   // zeros after the code: instruction prefetch reads further
        }
        if (pr->nconstants) {
            if (kbo_new(pr->nconstants, KMAP_WRITE, &consts[i])) return -1;
            memcpy(consts[i]->map, pr->constants, pr->nconstants);
        }
    }
    for (int s = 0; s < KSLOT; s++) if (kbo_new(DYN, KMAP_WRITE, &S[s].dyn) || kbo_new(BB, KMAP_WRITE, &S[s].bb)) return -1;
    if (kprofile && kbo_new(TS_MAX * 8, KMAP_RW, &ts_bo)) return -1;
    if (kprofile_kernels) atexit(print_measurements);
    return 0;
}

static int kcl_id_anchor;   // any unique address: identifiers of the platform, device, context and queue
cl_int kcl_GetPlatformIDs(cl_uint n, cl_platform_id *pl, cl_uint *count) {
    if (init()) return CL_PLATFORM_NOT_FOUND_KHR;
    if (pl && n) *pl = (cl_platform_id)&kcl_id_anchor;
    if (count) *count = 1;
    return CL_SUCCESS;
}
cl_int kcl_GetDeviceIDs(cl_platform_id pl, cl_device_type t, cl_uint n, cl_device_id *d, cl_uint *count) {
    (void)pl; (void)t;
    if (d && n) *d = (cl_device_id)&kcl_id_anchor;
    if (count) *count = 1;
    return CL_SUCCESS;
}
cl_context kcl_CreateContext(const cl_context_properties *pr, cl_uint n, const cl_device_id *d, void *f, void *u, cl_int *e) {
    (void)pr; (void)n; (void)d; (void)f; (void)u;
    *e = init() ? CL_OUT_OF_RESOURCES : CL_SUCCESS; return (cl_context)&kcl_id_anchor;
}
cl_command_queue kcl_CreateCommandQueueWithProperties(cl_context c, cl_device_id d, const cl_queue_properties *pr, cl_int *e) {
    (void)c; (void)d; (void)pr; *e = CL_SUCCESS; return (cl_command_queue)&kcl_id_anchor;
}

cl_int kcl_program(const char *file, const char *options, cl_program *out) {
    for (int i = 0; i < dev.nprogs; i++) {
        const struct kcl_program_desc *pr = &dev.progs[i];
        if (strcmp(pr->file, file)) continue;
        if ((options == NULL) != (pr->options == NULL) || (options && strcmp(options, pr->options))) continue;
        struct _cl_program *x = calloc(1, sizeof *x);
        x->o = pr; x->idx = i; *out = x; return CL_SUCCESS;
    }
    return CL_INVALID_PROGRAM;
}

cl_kernel kcl_CreateKernel(cl_program pr, const char *name, cl_int *e) {
    for (uint32_t j = 0; j < pr->o->nkernels; j++) {
        const struct kcl_kernel_desc *o = &pr->o->kernels[j];
        if (strcmp(o->name, name)) continue;
        if (o->unsupported[0]) {
            fprintf(stderr, "kcl: kernel %s: unsupported (%s)\n", name, o->unsupported);
            *e = CL_INVALID_KERNEL_DEFINITION; return NULL;
        }
        struct _cl_kernel *k = calloc(1, sizeof *k);
        k->o = o; k->isa_off = isa_off[pr->idx][j]; k->prog = pr->idx;
        if (o->private_size) {   // private memory: private_size x SIMD x (EUs x 7 threads), like NEO
            uint64_t needed = (uint64_t)o->private_size * o->simd * dev.eus * 7;
            if (!priv_bo || priv_bo->size < needed) {
                if (priv_bo) { kcl_wait(); kbo_release(priv_bo); priv_bo = NULL; }
                if (kbo_new(needed, KMAP_NONE, &priv_bo)) { *e = CL_OUT_OF_RESOURCES; free(k); return NULL; }
            }
        }
        *e = CL_SUCCESS; return k;
    }
    *e = CL_INVALID_KERNEL_NAME; return NULL;
}

// ---------------- memory ----------------
cl_mem kcl_CreateBuffer(cl_context c, cl_mem_flags f, size_t n, void *host, cl_int *e) {
    (void)c; (void)f;
    if (host) { *e = CL_INVALID_HOST_PTR; return NULL; }
    struct _cl_mem *m = calloc(1, sizeof *m);
    if (!m || kbo_new(n ? n : 1, KMAP_NONE, &m->bo)) { free(m); *e = CL_MEM_OBJECT_ALLOCATION_FAILURE; return NULL; }
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
    if (m && --m->ref == 0) { kbo_release(m->bo); free(m); }
    return CL_SUCCESS;
}
cl_int kcl_GetMemObjectInfo(cl_mem m, cl_mem_info i, size_t n, void *v, size_t *count) {
    if (i != CL_MEM_SIZE || n < sizeof(size_t)) return CL_INVALID_VALUE;
    *(size_t *)v = m->size; if (count) *count = sizeof(size_t); return CL_SUCCESS;
}

static void *usm_alloc(cl_context c, const cl_mem_properties_intel *pr, size_t n, cl_uint align, cl_int *e) {
    (void)c; (void)pr; (void)align;
    kbo *b;
    if (init() || kbo_new(n ? n : 1, KMAP_RW, &b)) { *e = CL_OUT_OF_HOST_MEMORY; return NULL; }
    if (nusm == cusm) { cusm = cusm ? 2 * cusm : 64; usm = realloc(usm, cusm * sizeof *usm); }
    usm[nusm++] = (usm_t){ (uintptr_t)b->map, n ? n : 1, b };
    *e = CL_SUCCESS; return b->map;
}
static cl_int usm_free(cl_context c, void *ptr) {
    (void)c;
    for (int i = 0; i < nusm; i++) if (usm[i].p == (uintptr_t)ptr) {
        kcl_wait();   // clMemBlockingFreeINTEL: after the GPU work has finished
        kbo_release(usm[i].bo); usm[i] = usm[--nusm]; return CL_SUCCESS;
    }
    return CL_INVALID_VALUE;
}
static cl_int usm_info(cl_context c, const void *ptr, cl_mem_info_intel i, size_t n, void *v, size_t *count) {
    (void)c; (void)count;
    usm_t *u = usm_find(ptr);
    if (i == CL_MEM_ALLOC_TYPE_INTEL) { *(cl_unified_shared_memory_type_intel *)v = u ? CL_MEM_TYPE_HOST_INTEL : CL_MEM_TYPE_UNKNOWN_INTEL; return CL_SUCCESS; }
    if (!u) return CL_INVALID_VALUE;
    if (i == CL_MEM_ALLOC_BASE_PTR_INTEL && n >= sizeof(void *)) { *(void **)v = (void *)u->p; return CL_SUCCESS; }
    if (i == CL_MEM_ALLOC_SIZE_INTEL && n >= sizeof(size_t)) { *(size_t *)v = u->n; return CL_SUCCESS; }
    return CL_INVALID_VALUE;
}

// ---------------- arguments ----------------
cl_int kcl_SetKernelArg(cl_kernel k, cl_uint i, size_t n, const void *v) {
    if (i >= k->o->nargs) return CL_INVALID_ARG_INDEX;
    if (k->o->arg[i].ptr) {
        cl_mem m = v ? *(const cl_mem *)v : NULL;
        if (n != sizeof(cl_mem)) return CL_INVALID_ARG_SIZE;
        if (m && !m->bo && (up_resolve() || !m->bo)) return CL_INVALID_MEM_OBJECT;   // userptr region to be resolved
        k->a[i].bo = m ? m->bo : NULL; k->a[i].addr = m ? m->bo->addr + m->off : 0; k->a[i].sz = m ? m->size : 0;
    } else {
        if (n > 8 || (k->o->arg[i].size && n != (size_t)k->o->arg[i].size)) return CL_INVALID_ARG_SIZE;
        memset(k->a[i].v, 0, 8); memcpy(k->a[i].v, v, n);
    }
    k->a[i].set = 1; return CL_SUCCESS;
}
static cl_int usm_arg(cl_kernel k, cl_uint i, const void *ptr) {
    if (i >= k->o->nargs || !k->o->arg[i].ptr) return CL_INVALID_ARG_INDEX;
    usm_t *u = usm_find(ptr);
    if (!u) return CL_INVALID_ARG_VALUE;
    uint64_t off = (uintptr_t)ptr - u->p;
    // surface size up to the end of the GEM object (pages), not of the allocation: the hardware checks RAW buffer bounds every 4 B,
    // so with a size not divisible by 4 the last FP16 element was read as 0 (NEO: alignUp(size, 4))
    k->a[i].bo = u->bo; k->a[i].addr = u->bo->addr + off; k->a[i].sz = u->bo->size - off; k->a[i].set = 1;
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

// ---------------- events and timestamps ----------------
static int timestamp_mark(void) {   // PIPE_CONTROL writing the time counter (Gen9 12 MHz, CHV 12.5 MHz) after the previous work has finished
    if (!kprofile || !ts_bo) return -1;
    int i = (int)(ts_n++ % TS_MAX);
    EMIT(PIPE_CONTROL, .CommandStreamerStallEnable = 1, .PostSyncOperation = 3 /* write timestamp */,
         .Address = ts_bo->addr + 8ull * i);
    return i;
}
static cl_event make_event(int ts, int ts_k) {
    struct _cl_event *e = calloc(1, sizeof *e);
    e->seq = S[cur].seq; e->ts = ts; e->ts_k = ts_k; e->ref = 1; return e;
}

cl_int kcl_EnqueueMarkerWithWaitList(cl_command_queue q, cl_uint n, const cl_event *w, cl_event *ev) {
    (void)q; (void)n; (void)w;
    if (open_batch()) return CL_OUT_OF_RESOURCES;
    int ts = timestamp_mark();
    if (ev) *ev = make_event(ts, ts);   // start = end = the moment of the marker
    return CL_SUCCESS;
}
cl_int kcl_Flush(cl_command_queue q) { (void)q; return submit() ? CL_OUT_OF_RESOURCES : CL_SUCCESS; }
cl_int kcl_Finish(cl_command_queue q) { (void)q; return kcl_wait(); }
cl_int kcl_WaitForEvents(cl_uint n, const cl_event *e) {
    for (cl_uint i = 0; i < n; i++) if (wait_seq(e[i]->seq)) return CL_OUT_OF_RESOURCES;
    return CL_SUCCESS;
}
cl_int kcl_GetEventInfo(cl_event e, cl_event_info i, size_t n, void *v, size_t *count) {
    (void)count;
    if (i != CL_EVENT_COMMAND_EXECUTION_STATUS || n < sizeof(cl_int)) return CL_INVALID_VALUE;
    if (cur >= 0 && S[cur].seq <= e->seq && submit()) return CL_OUT_OF_RESOURCES;
    cl_int st = CL_COMPLETE;
    if (e->seq > seq_done) {
        struct drm_i915_gem_busy b = { .handle = S[e->seq % KSLOT].bb->h };
        if (S[e->seq % KSLOT].seq == e->seq && !ioctl(fd, DRM_IOCTL_I915_GEM_BUSY, &b) && b.busy) st = CL_RUNNING;
        else if (e->seq > seq_done) seq_done = e->seq;
    }
    *(cl_int *)v = st; return CL_SUCCESS;
}
cl_int kcl_GetEventProfilingInfo(cl_event e, cl_profiling_info i, size_t n, void *v, size_t *count) {
    (void)count;
    if (n < sizeof(cl_ulong)) return CL_INVALID_VALUE;
    if (wait_seq(e->seq)) return CL_OUT_OF_RESOURCES;
    cl_ulong t = 0;
    const int j = i == CL_PROFILING_COMMAND_END ? e->ts_k : e->ts;
    if (j >= 0) {   // counter -> ns (split so that ticks * 1e9 cannot overflow)
        const uint64_t c = ((uint64_t *)ts_bo->map)[j];
        t = c / dev.ts_freq * 1000000000ull + c % dev.ts_freq * 1000000000ull / dev.ts_freq;
    }
    *(cl_ulong *)v = t; return CL_SUCCESS;
}
cl_int kcl_ReleaseEvent(cl_event e) { if (e && --e->ref == 0) free(e); return CL_SUCCESS; }

// ---------------- kernel launch ----------------
static uint32_t divisor(size_t g, uint32_t maxv) {   // the largest divisor of g not greater than maxv
    for (uint32_t d = maxv; d > 1; d--) if (g % d == 0) return d;
    return 1;
}

static cl_int enqueue_(cl_command_queue q, cl_kernel k, cl_uint dim, const size_t *off, const size_t *glob,
                       const size_t *lsz, cl_uint nw, const cl_event *w, cl_event *ev);
cl_int kcl_EnqueueNDRangeKernel(cl_command_queue q, cl_kernel k, cl_uint dim, const size_t *off, const size_t *glob,
                                const size_t *lsz, cl_uint nw, const cl_event *w, cl_event *ev) {
    PHASE_BEGIN; cl_int r = enqueue_(q, k, dim, off, glob, lsz, nw, w, ev); PHASE_END(F_ENQ); return r;
}
static cl_int enqueue_(cl_command_queue q, cl_kernel k, cl_uint dim, const size_t *off, const size_t *glob,
                       const size_t *lsz, cl_uint nw, const cl_event *w, cl_event *ev) {
    (void)q; (void)nw; (void)w;
    const struct kcl_kernel_desc *o = k->o;
    for (uint32_t i = 0; i < o->nargs; i++) if (!k->a[i].set) { fprintf(stderr, "kcl: %s: argument %u not set\n", o->name, i); return CL_INVALID_KERNEL_ARGS; }
    size_t G[3] = { 1, 1, 1 }, O[3] = { 0, 0, 0 }; uint32_t L[3] = { 1, 1, 1 };
    for (cl_uint d = 0; d < dim; d++) { G[d] = glob[d]; if (off) O[d] = off[d]; }
    if (G[0] * G[1] * G[2] == 0) return CL_SUCCESS;
    if (lsz) {
        for (cl_uint d = 0; d < dim; d++) { L[d] = (uint32_t)lsz[d]; if (!L[d] || G[d] % L[d]) return CL_INVALID_WORK_GROUP_SIZE; }
    } else {   // like NEO: a work-group size dividing the global size; here: at most 1 SIMD thread per work-group
        L[0] = divisor(G[0], o->simd);
        if (dim > 1) L[1] = divisor(G[1], o->simd / L[0]);
        if (dim > 2) L[2] = divisor(G[2], o->simd / (L[0] * L[1]));
    }
    const uint32_t lt = L[0] * L[1] * L[2], threads = (lt + o->simd - 1) / o->simd;
    if (threads > dev.max_group_threads) return CL_INVALID_WORK_GROUP_SIZE;
    const uint32_t pt = o->per_thread ? o->per_thread : 32;   // NEO: at least 1 GRF of data per thread
    const uint32_t dlen = (o->cross + threads * pt + 63) & ~63u;

    if (cur >= 0 && (nlaunch >= IDD_MAX || o_ind + dlen > O_SS || o_ss + 64 * (o->nbti + 2) > O_SS + SS_LEN ||
                     (uint8_t *)p - (uint8_t *)S[cur].bb->map > BB - 2048 || nbo_list + o->nargs + 4 > MAXBO))
        if (submit()) return CL_OUT_OF_RESOURCES;
    if (open_batch()) return CL_OUT_OF_RESOURCES;
    if (nlaunch > 0) EMIT(PIPE_CONTROL, .CommandStreamerStallEnable = 1, .DCFlushEnable = 1);   // in-order queue
    uint8_t *D = S[cur].dyn->map, *ind = D + o_ind;
    memset(ind, 0, dlen);
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
    if (o->off_const >= 0) { uint64_t a = consts[k->prog] ? consts[k->prog]->addr : 0; memcpy(ind + o->off_const, &a, 8); add_bo(consts[k->prog]); }
    if (o->off_priv >= 0) { uint64_t a = priv_bo ? priv_bo->addr : 0; memcpy(ind + o->off_priv, &a, 8); add_bo(priv_bo); }
    // arguments; stateful: buffer surface state (like NEO encodeBuffer) and an entry in the binding table
    uint32_t bt = 0; uint32_t *tab = NULL;
    if (o->nbti) {
        o_ss = (o_ss + 63) & ~63u; bt = o_ss - O_SS; tab = (uint32_t *)(D + o_ss);
        memset(tab, 0, 4 * o->nbti); o_ss = (o_ss + 4 * o->nbti + 63) & ~63u;
    }
    for (uint32_t i = 0; i < o->nargs; i++) {
        const struct kcl_arg_desc *a = &o->arg[i];
        if (!a->ptr) { if (a->off >= 0) memcpy(ind + a->off, k->a[i].v, a->size); continue; }
        add_bo(k->a[i].bo);
        const uint64_t addr = k->a[i].addr;
        if (a->off >= 0) memcpy(ind + a->off, &addr, a->size);                 // stateless: 64-bit address
        if (a->bti >= 0) {
            const uint64_t base = addr & ~3ull, shift = addr & 3;               // surface base aligned to 4 B
            const uint64_t r = ((k->a[i].sz + shift) + 3) & ~3ull, len = (r ? r : 4) - 1;   // jak NEO: alignUp(size, 4)
            if (shift && a->off_boff < 0) { fprintf(stderr, "kcl: %s: arg %u misaligned\n", o->name, i); return CL_INVALID_ARG_VALUE; }
            struct GFX9_RENDER_SURFACE_STATE ss = { .SurfaceType = SURFTYPE_BUFFER, .SurfaceFormat = 0x1ff /* RAW */,
                .SurfaceVerticalAlignment = VALIGN_4, .TileMode = LINEAR, .MOCS = MOCS_WB,
                .Width = len & 0x7f, .Height = (len >> 7) & 0x3fff, .Depth = (len >> 21) & 0x7ff,
                .SurfaceBaseAddress = base };
            GFX9_RENDER_SURFACE_STATE_pack(NULL, D + o_ss, &ss);
            tab[a->bti] = o_ss - O_SS; o_ss += 64;
            if (a->off_addr >= 0) memcpy(ind + a->off_addr, &base, 8);
            if (a->off_boff >= 0) { uint32_t b32 = (uint32_t)shift; memcpy(ind + a->off_boff, &b32, 4); }
        }
    }
    // per-thread data: local ID (a row per dimension: 32 B for SIMD8/16, 64 B for SIMD32)
    if (o->lid) {
        const uint32_t row = o->simd == 32 ? 64 : 32;
        for (uint32_t t = 0; t < threads; t++) {
            uint16_t *x = (uint16_t *)(ind + o->cross + t * pt), *y = x + row / 2, *z = y + row / 2;
            for (uint32_t l = 0; l < o->simd; l++) {
                uint32_t lin = t * o->simd + l; if (lin >= lt) lin = lt - 1;
                x[l] = lin % L[0]; y[l] = (lin / L[0]) % L[1]; z[l] = lin / (L[0] * L[1]);
            }
        }
    }
    uint32_t slm = 0;
    // SLM size encoding in the IDD (PRM KBL vol. 2d, NEO computeSlmValues): log2(pow2(max(slm, 1K))) - 9, i.e.
    // 1K=1, 2K=2, 4K=3 ... 64K=7. Note: the Gen8 encoding (Mesa genxml gen80.xml: 4K=1, 8K=2, ...) differs; Mesa's gen90.xml has the right one -- with Gen8 values work-groups
    // got too little SLM and overlapped (SYMM/TRMM: wrong result with more than one work-group).
    if (o->slm && dev.gen8) {   // Gen8: 4 KB steps, powers of two: 4K=1, 8K=2, 16K=4 ... 64K=16 (PRM CHV vol02d p.223)
        uint32_t v = 4096; slm = 1; while (v < o->slm) { v *= 2; slm *= 2; } slm_in_batch = 1;
    } else if (o->slm) { uint32_t v = 1024; slm = 1; while (v < o->slm) { v *= 2; slm++; } slm_in_batch = 1; }
    struct GFX9_INTERFACE_DESCRIPTOR_DATA idd = {
        .KernelStartPointer = k->isa_off, .NumberofThreadsinGPGPUThreadGroup = threads,
        .CrossThreadConstantDataReadLength = o->cross / 32, .ConstantURBEntryReadLength = pt / 32,
        .BindingTablePointer = bt, .BindingTableEntryCount = o->nbti < 32 ? o->nbti : 31,
        .SharedLocalMemorySize = slm, .BarrierEnable = o->barrier || (dev.gen8 && o->slm) };   // CHV: SLM needs a barrier
    GFX9_INTERFACE_DESCRIPTOR_DATA_pack(NULL, D + 32 * nlaunch, &idd);
    const int ts = ev || kprofile_kernels ? timestamp_mark() : -1;
    const uint32_t rem = lt % o->simd;
    EMIT(GPGPU_WALKER, .InterfaceDescriptorOffset = nlaunch, .IndirectDataLength = dlen, .IndirectDataStartAddress = o_ind,
         .SIMDSize = o->simd == 8 ? SIMD8 : o->simd == 16 ? SIMD16 : SIMD32, .ThreadWidthCounterMaximum = threads - 1,
         .ThreadGroupIDXDimension = (uint32_t)(G[0] / L[0]), .ThreadGroupIDYDimension = (uint32_t)(G[1] / L[1]),
         .ThreadGroupIDZDimension = (uint32_t)(G[2] / L[2]),
         .RightExecutionMask = rem ? (1u << rem) - 1 : (o->simd == 32 ? 0xffffffffu : (1u << o->simd) - 1),
         .BottomExecutionMask = 0xffffffff);
    EMIT(MEDIA_STATE_FLUSH);
    if (ev || kprofile_kernels) {
        const int ts_k = timestamp_mark();
        if (ev) *ev = make_event(ts, ts_k);
        if (kprofile_kernels && nmeas < TS_MAX / 2 - 1) meas[nmeas++] = (meas_t){ o->name, ts, ts_k };
        if (kprofile_kernels && nmeas >= TS_MAX / 2 - 8 && kcl_wait()) return CL_OUT_OF_RESOURCES;
    }
    o_ind += dlen; nlaunch++;
    // Gen8 has no preemption: a batch runs to its end, and i915's heartbeat resets the engine after ~10 s
    // (docs/gen8.md section 4). One request per kernel keeps each request to one (band-split) launch.
    if (dev.gen8 && submit()) return CL_OUT_OF_RESOURCES;
    static int each = -1; if (each < 0) each = getenv("KCL_WAIT_EACH") != NULL;   // diagnostics
    if (each && kcl_wait()) return CL_OUT_OF_RESOURCES;
    return CL_SUCCESS;
}
