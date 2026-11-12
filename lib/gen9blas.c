// libgen9blas: level 3 BLAS on UHD 620 (Gen9.5) with hand-written assembly GEMM loops (asm/gemm_gen.py).
// Core: batched GEMM C_b = alpha * op(A_b)·op(B_b) + beta * C_b (b = 0..count-1, strides sA, sB, sC in elements),
// row-major, arbitrary M, N, K. Types: GEN9BLAS_H (FP16), GEN9BLAS_HS (FP16, accumulated in chunks into FP32), GEN9BLAS_S (FP32).
// Level 3 (SYMM, SYRK, SYR2K, TRMM, TRSM) is built on GEMM, entirely on the GPU: expanding the triangular /
// symmetric matrix, joining the SYR2K operands and the inverses of the TRSM diagonal blocks (FP64) are kernels.
// The CPU only submits work and sleeps while waiting (wait_gpu()). Zero-copy: USM host memory (gen9blas_alloc, gen9blas.empty in Python);
// other host memory is copied by the CPU into USM work buffers (the driver does not register ordinary host memory).
#include "gen9blas.h"
#ifdef GEN9BLAS_NEO   // old NEO (OpenCL) path -- for comparisons only (libgen9blas_neo.so)
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <CL/cl_ext.h>
#else            // default: kcl -- an OpenCL subset directly on i915 (kcl.c, docs/no-neo.md)
#include "kcl.h"
#endif
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

typedef struct {
    cl_kernel main, pa, pb, pa_t, pb_t, pa_q, pb_w2,   // pa_q, pb_w2: faster packing without transpose (NULL: absent)
              epi, epi_tri, tL, tY, tZ, tI[4], expand, syr2k_merge;   // t*: TRSM; expand: full matrix; syr2k_merge: SYR2K join
    int es, tn, km, ready;
} type_t;

static cl_context ctx;
static cl_command_queue q;
static cl_device_id dev;
static type_t types[4];   // index = type: GEN9BLAS_H, GEN9BLAS_S, GEN9BLAS_HS, GEN9BLAS_D
static cl_mem dAp, dBp, dCp;
static size_t sAp, sBp, sCp;
static double last_ms;
static int profile;   // GEN9BLAS_PROFILE=1: profiling queue, kernel times
static int last_zc;
static char errbuf[1200];
static clHostMemAllocINTEL_fn usm_alloc;          // cl_intel_unified_shared_memory (see get_obj())
static clMemBlockingFreeINTEL_fn usm_free;
static clGetMemAllocInfoINTEL_fn usm_info;
static clSetKernelArgMemPointerINTEL_fn usm_arg;

#define CHECK(x) do { cl_int e_ = (x); if (e_ != CL_SUCCESS) { snprintf(errbuf, sizeof errbuf, "%s:%d: %s = %d", __FILE__, __LINE__, #x, e_); return -1; } } while (0)
#define FAIL(...) do { snprintf(errbuf, sizeof errbuf, __VA_ARGS__); return -1; } while (0)

static __attribute__((unused)) unsigned char *load_file(const char *p, size_t *len) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); *len = ftell(f); rewind(f);
    unsigned char *s = malloc(*len + 1);
    if (fread(s, 1, *len, f) != *len) { fclose(f); free(s); return NULL; }
    s[*len] = 0; fclose(f); return s;
}

static int program_bin(const char *dir, const char *file, const char *kernel, cl_kernel *k) {
#ifndef GEN9BLAS_NEO   // built-in kernels (kernels.c, generated at build time from the same files)
    (void)dir; cl_program pr; cl_int err;
    if (kcl_program(file, NULL, &pr) != CL_SUCCESS) FAIL("no built-in program %s", file);
    *k = clCreateKernel(pr, kernel, &err); CHECK(err);
    return 0;
#else
    char p[1024]; size_t len; cl_int err, bst;
    snprintf(p, sizeof p, "%s/%s", dir, file);
    unsigned char *bin = load_file(p, &len);
    if (!bin) FAIL("missing %.900s (build: make -C asm)", p);
    cl_program pb = clCreateProgramWithBinary(ctx, 1, &dev, &len, (const unsigned char **)&bin, &bst, &err);
    free(bin); CHECK(err); CHECK(bst); CHECK(clBuildProgram(pb, 1, &dev, "", NULL, NULL));
    *k = clCreateKernel(pb, kernel, &err); CHECK(err);
    return 0;
#endif
}

static int program_src(const char *dir, const char *file, const char *options, cl_program *pr) {
#ifndef GEN9BLAS_NEO
    (void)dir;
    if (kcl_program(file, options, pr) != CL_SUCCESS) FAIL("no built-in program %s [%s]", file, options);
    return 0;
#else
    char p[1024]; size_t len; cl_int err;
    snprintf(p, sizeof p, "%s/%s", dir, file);
    char *src = (char *)load_file(p, &len);
    if (!src) FAIL("missing %.900s", p);
    *pr = clCreateProgramWithSource(ctx, 1, (const char **)&src, NULL, &err); free(src); CHECK(err);
    if (clBuildProgram(*pr, 1, &dev, options, NULL, NULL) != CL_SUCCESS) {
        clGetProgramBuildInfo(*pr, dev, CL_PROGRAM_BUILD_LOG, sizeof errbuf, errbuf, NULL); return -1;
    }
    return 0;
#endif
}

#define KER(pr, name, dst) do { cl_int e2_; dst = clCreateKernel(pr, name, &e2_); CHECK(e2_); } while (0)

// Level 1–2 kernels (bench/level12.cl): L12[0] FP16 (GEN9BLAS_H, GEN9BLAS_HS), L12[1] FP32.
typedef struct { cl_kernel axpy, scal, copy, copy16, rot, red, iamax, gemv_n, gemv_nw, gemv_t1, gemv_t1b, gemv_t2, ger, syr2, to_f, from_f, trsv_block, trsv_upd_n, trsv_upd_t, rotm, pmv, psv, spr2, symv1, symv2, skinny_k1, skinny_k4, skinny_k8, skinny_r4, skinny_r8, skinny_n4, skinny_n2; int es; } l12_t;
static l12_t L12[3];   // 0 FP16, 1 FP32, 2 FP64
static int init_l12(const char *dir) {
    for (int i = 0; i < 3; i++) {
        cl_program pr; const char *suf = "hfd" + i, *opc[3] = { "-cl-std=CL3.0 -DHALF", "-cl-std=CL3.0", "-cl-std=CL3.0 -DDBL" };
        char n[32], sf[2] = { *suf, 0 };
        if (program_src(dir, "level12.cl", opc[i], &pr)) return -1;
        l12_t *k = &L12[i]; k->es = i == 0 ? 2 : i == 1 ? 4 : 8; suf = sf;
        struct { const char *n; cl_kernel *k; } t[] = { { "axpy", &k->axpy }, { "scal", &k->scal }, { "copy", &k->copy }, { "copy16", &k->copy16 },
            { "rot", &k->rot }, { "red", &k->red }, { "iamax", &k->iamax }, { "gemv_n", &k->gemv_n }, { "gemv_nw", &k->gemv_nw },
            { "gemv_t1", &k->gemv_t1 }, { "gemv_t1b", &k->gemv_t1b }, { "gemv_t2", &k->gemv_t2 }, { "ger", &k->ger }, { "syr2", &k->syr2 },
            { "to_f", &k->to_f }, { "from_f", &k->from_f }, { "trsv_block", &k->trsv_block },
            { "trsv_upd_n", &k->trsv_upd_n }, { "trsv_upd_t", &k->trsv_upd_t }, { "rotm", &k->rotm }, { "pmv", &k->pmv },
            { "psv", &k->psv }, { "spr2", &k->spr2 }, { "symv1", &k->symv1 }, { "symv2", &k->symv2 } };
        for (size_t j = 0; j < sizeof t / sizeof t[0]; j++) { snprintf(n, sizeof n, "%s_%s", t[j].n, suf); KER(pr, n, *t[j].k); }
        if (i < 2) {   // skinny GEMM: FP16 and FP32 only (bench/level12.cl, #ifndef DBL)
            struct { const char *n; cl_kernel *k; } c[] = { { "skinny_k1", &k->skinny_k1 }, { "skinny_k4", &k->skinny_k4 }, { "skinny_k8", &k->skinny_k8 },
                { "skinny_r4", &k->skinny_r4 }, { "skinny_r8", &k->skinny_r8 },
                { "skinny_n4", &k->skinny_n4 }, { "skinny_n2", &k->skinny_n2 } };
            for (size_t j = 0; j < sizeof c / sizeof c[0]; j++) { snprintf(n, sizeof n, "%s_%s", c[j].n, suf); KER(pr, n, *c[j].k); }
        }
    }
    return 0;
}

// One-shot: subsequent calls (e.g. the BLAS layer and the Python module in one process) do nothing -- a second OpenCL
// context would invalidate the work buffers and USM memory of the first (error -50 when setting arguments).
int gen9blas_init(const char *dir) {
    static int inited;
    if (inited) return 0;
    cl_int err; cl_platform_id plat;
    CHECK(clGetPlatformIDs(1, &plat, NULL));
    CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL));
    ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err); CHECK(err);
    // Queue profiling only with GEN9BLAS_PROFILE=1: with it every event (including the marker in wait_gpu()) costs
    // ~20 µs more CPU (measured 2026-11-11: marker 21 vs 2.3 µs). Without profiling gen9blas_last_time_ms() returns the time
    // of the whole call.
    profile = getenv("GEN9BLAS_PROFILE") != NULL;
    cl_queue_properties qp[] = { CL_QUEUE_PROPERTIES, profile ? CL_QUEUE_PROFILING_ENABLE : 0, 0 };
    q = clCreateCommandQueueWithProperties(ctx, dev, qp, &err); CHECK(err);
    cl_program ph, ps, pd;
    if (program_src(dir, "hgemm.cl", "", &ph) || program_src(dir, "sgemm.cl", "", &ps) ||
        program_src(dir, "dgemm.cl", "", &pd)) return -1;
    usm_alloc = (clHostMemAllocINTEL_fn)clGetExtensionFunctionAddressForPlatform(plat, "clHostMemAllocINTEL");
    usm_free = (clMemBlockingFreeINTEL_fn)clGetExtensionFunctionAddressForPlatform(plat, "clMemBlockingFreeINTEL");
    usm_info = (clGetMemAllocInfoINTEL_fn)clGetExtensionFunctionAddressForPlatform(plat, "clGetMemAllocInfoINTEL");
    usm_arg = (clSetKernelArgMemPointerINTEL_fn)clGetExtensionFunctionAddressForPlatform(plat, "clSetKernelArgMemPointerINTEL");
    if (!usm_alloc || !usm_free || !usm_info || !usm_arg) FAIL("missing cl_intel_unified_shared_memory extension");
    if (init_l12(dir)) return -1;
    // "v" kernels: (Ap, Bp, C, M, N, ldc, K, Kfull) -- also for chunks of data packed once (sub-buffers)
    // bin = NULL: main kernel from source (FP64: dgemm.cl, OpenCL C), not from an asm binary
    struct { int r; const char *bin, *k, *pa, *pb, *epi, *tri; int es, tn, km; cl_program pr; } t[4] = {
        { GEN9BLAS_H,  "gen_h.bin",  "hgemm_v",    "pk_a2", "pk_bq",  "gen9blas_epilog_b",  "gen9blas_epilog_tri",  2, 32, 16, ph },
        { GEN9BLAS_HS, "gen_hd.bin", "hgemm_kc_v", "pk_a3", "pk_b16", "gen9blas_epilog_b",  "gen9blas_epilog_tri",  2, 16, GEN9BLAS_KC, ph },
        { GEN9BLAS_S,  "gen_s.bin",  "sgemm_v",    "pkf_a", "pkf_b",  "gen9blas_epilog_fb", "gen9blas_epilog_trif", 4, 16, 16, ps },
        { GEN9BLAS_D,  NULL,         "dgemm_v",    "pkd_a", "pkd_b",  "gen9blas_epilog_db", "gen9blas_epilog_trid", 8, 16, 16, pd } };
    for (int i = 0; i < 4; i++) {
        type_t *r = &types[t[i].r]; char n[64];
        if (t[i].bin ? program_bin(dir, t[i].bin, t[i].k, &r->main) : 0) return -1;
        if (!t[i].bin) KER(t[i].pr, t[i].k, r->main);
        KER(t[i].pr, t[i].pa, r->pa); KER(t[i].pr, t[i].pb, r->pb); KER(t[i].pr, t[i].epi, r->epi);
        KER(t[i].pr, t[i].tri, r->epi_tri);
        const char *suf = t[i].es == 4 ? "f" : t[i].es == 8 ? "d" : "h";
        snprintf(n, sizeof n, "trsm_L_%s", suf); KER(t[i].pr, n, r->tL);
        snprintf(n, sizeof n, "trsm_Y_%s", suf); KER(t[i].pr, n, r->tY);
        snprintf(n, sizeof n, "trsm_store_%s", suf); KER(t[i].pr, n, r->tZ);
        for (int j = 0; j < 4; j++) { snprintf(n, sizeof n, "trsm_inv%d_%s", j, suf); KER(t[i].pr, n, r->tI[j]); }
        snprintf(n, sizeof n, "gen9blas_expand_%s", suf); KER(t[i].pr, n, r->expand);
        snprintf(n, sizeof n, "gen9blas_syr2k_merge_%s", suf); KER(t[i].pr, n, r->syr2k_merge);
        snprintf(n, sizeof n, "%s_t", t[i].pa); KER(t[i].pr, n, r->pa_t);
        snprintf(n, sizeof n, "%s_t", t[i].pb); KER(t[i].pr, n, r->pb_t);
        cl_int e3;   // optional faster packing kernels (GEN9BLAS_PACK=old: disabled)
        int old_pack = getenv("GEN9BLAS_PACK") && !strcmp(getenv("GEN9BLAS_PACK"), "old");
        snprintf(n, sizeof n, "%s_q", t[i].pa); r->pa_q = old_pack ? NULL : clCreateKernel(t[i].pr, n, &e3); if (e3) r->pa_q = NULL;
        snprintf(n, sizeof n, "%s_w2", t[i].pb); r->pb_w2 = old_pack ? NULL : clCreateKernel(t[i].pr, n, &e3); if (e3) r->pb_w2 = NULL;
        r->es = t[i].es; r->tn = t[i].tn; r->km = t[i].km; r->ready = 1;
    }
    errbuf[0] = 0;
    inited = 1;
    return 0;
}

// Sleeping wait for the queue to finish: clFinish / clWaitForEvents / blocking reads in NEO spin actively
// on the CPU (~6 W). Marker + clFlush, then polling the state every 20 µs .. 0.5 ms. GEN9BLAS_WAIT=busy: clFinish.
static int wait_gpu(void) {
#ifndef GEN9BLAS_NEO   // kcl: submit the batch buffer and GEM_WAIT (sleeps in the Linux kernel until the GPU interrupt)
    if (kcl_wait() != CL_SUCCESS) FAIL("GPU execution error (kcl)");
    return 0;
#else
    static int busy = -1;
    if (busy < 0) { const char *e = getenv("GEN9BLAS_WAIT"); busy = e && !strcmp(e, "busy"); }
    if (busy) { CHECK(clFinish(q)); return 0; }
    cl_event e; cl_int st = CL_COMPLETE; long ns = 20000;
    CHECK(clEnqueueMarkerWithWaitList(q, 0, NULL, &e)); CHECK(clFlush(q));
    while (clGetEventInfo(e, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof st, &st, NULL) == CL_SUCCESS && st > CL_COMPLETE) {
        struct timespec ts = { 0, ns }; nanosleep(&ts, NULL);
        if (ns < 500000) ns *= 2;
    }
    clReleaseEvent(e);
    if (st < 0) FAIL("GPU execution error (%d)", st);
    return 0;
#endif
}

// GEMM kernel events for time measurement (launch(..., measure=1)); times are read after wait_gpu().
static cl_event evs[256];
static int nev;
static double event_times(void) {
    double s = 0;
    if (!profile) { for (int i = 0; i < nev; i++) clReleaseEvent(evs[i]); nev = 0; return -1; }   // without profiling
    for (int i = 0; i < nev; i++) { cl_ulong t0, t1;
        clGetEventProfilingInfo(evs[i], CL_PROFILING_COMMAND_START, 8, &t0, NULL); clGetEventProfilingInfo(evs[i], CL_PROFILING_COMMAND_END, 8, &t1, NULL);
        s += (t1 - t0) * 1e-6; clReleaseEvent(evs[i]); }
    nev = 0; return s;
}

static int ensure_buf(cl_mem *b, size_t *size, size_t needed) {
    if (*size >= needed) return 0;
    if (*b) clReleaseMemObject(*b);
    cl_int err; *b = clCreateBuffer(ctx, CL_MEM_READ_WRITE, needed, NULL, &err); CHECK(err);
    *size = needed; return 0;
}

// ---------------- host memory for kernels: USM ----------------
// Zero-copy: USM host memory (clHostMemAllocINTEL; gen9blas_alloc, gen9blas.empty in Python) -- the pointer goes to the kernel
// directly (clSetKernelArgMemPointerINTEL), including a pointer into the interior of an allocation. Other host memory: copy by the CPU
// (memcpy) into a USM work buffer and back. Verified 2026-11-11: CL_MEM_USE_HOST_PTR buffers on ordinary host
// memory, created and freed often (levels 1–2), ended in NEO (Gen9, legacy1) failures: ENOSPC in execbuf,
// host_ptr_manager.cpp:252, reference_tracked_object.h:134, segfault -- GPU objects are closed asynchronously, and a new
// buffer on the same pages gets an overlapping GPU address. clEnqueueWrite/ReadBuffer from host memory registers
// it the same way. USE_HOST_PTR on USM memory, in turn, ignores the offset inside an allocation (wrong result).

void *gen9blas_alloc(size_t bytes) {
    cl_int err;
    if (!usm_alloc) { snprintf(errbuf, sizeof errbuf, "gen9blas_alloc: gen9blas_init not called"); return NULL; }
    void *p = usm_alloc(ctx, NULL, bytes ? bytes : 1, 4096, &err);
    if (err != CL_SUCCESS) { snprintf(errbuf, sizeof errbuf, "clHostMemAllocINTEL(%zu) = %d", bytes, err); return NULL; }
    return p;
}
void gen9blas_free(void *p) { if (p && usm_free) usm_free(ctx, p); }
static int usm_contains(const void *p, size_t bytes);
int gen9blas_is_usm(const void *p, size_t bytes) { return usm_info && p && usm_contains(p, bytes); }

// 1 if [p, p+bytes) lies entirely within a single USM allocation.
static int usm_contains(const void *p, size_t bytes) {
    cl_unified_shared_memory_type_intel t = 0; void *base = NULL; size_t size = 0;
    if (usm_info(ctx, p, CL_MEM_ALLOC_TYPE_INTEL, sizeof t, &t, NULL) != CL_SUCCESS || t == CL_MEM_TYPE_UNKNOWN_INTEL) return 0;
    if (usm_info(ctx, p, CL_MEM_ALLOC_BASE_PTR_INTEL, sizeof base, &base, NULL) != CL_SUCCESS ||
        usm_info(ctx, p, CL_MEM_ALLOC_SIZE_INTEL, sizeof size, &size, NULL) != CL_SUCCESS) return 0;
    return (uintptr_t)p + bytes <= (uintptr_t)base + size;
}

// Kernel argument: a cl_mem object or a USM pointer.
typedef struct { cl_mem m; void *p; } bf_t;
#define BM(x) ((bf_t){ (x), NULL })
static cl_int set_arg(cl_kernel k, int i, bf_t b) { return b.p ? usm_arg(k, i, b.p) : clSetKernelArg(k, i, sizeof b.m, &b.m); }
#define ARGB(k, i, b) CHECK(set_arg(k, i, b))

// USM work buffer (copy of non-USM host memory); growing it waits for the GPU to finish (blocking free).
// Operation boundary for kcl_userptr (regions of one BLAS function sharing pages -> one object).
#ifndef GEN9BLAS_NEO
#define NEW_OPERATION() kcl_new_operation()
#else
#define NEW_OPERATION() ((void)0)
#endif
typedef struct { void *p; size_t n; } work_t;
#ifndef GEN9BLAS_NEO
static int no_userptr = -1;
#endif
static work_t uA, uB, uC;
static int ensure_work(work_t *u, size_t bytes) {
    if (u->n >= bytes) return 0;
    if (u->p) usm_free(ctx, u->p);
    u->p = gen9blas_alloc(bytes); u->n = u->p ? bytes : 0;
    return u->p ? 0 : -1;
}
// Host memory [ptr, ptr+bytes) for a kernel: USM zero-copy, otherwise a copy into the work buffer u (copy_in: with data).
// Before writing to the work buffer, wait for the GPU to finish (the buffer may have been used earlier in this call).
// Without NEO: ordinary process memory (e.g. NumPy) via userptr with the kcl cache (kcl_userptr) -- also zero-copy.
// writes: the GPU writes to this region (userptr read-only otherwise).
static int get_obj(const void *ptr, size_t bytes, work_t *u, int copy_in, int writes, bf_t *result, int *zc) {
    if (usm_contains(ptr, bytes)) { *result = (bf_t){ NULL, (void *)ptr }; *zc = 1; return 0; }
#ifndef GEN9BLAS_NEO
    if (no_userptr < 0) no_userptr = getenv("GEN9BLAS_NO_USERPTR") != NULL;   // comparisons: copy as before
    cl_mem m;
    if (!no_userptr && (m = kcl_userptr(ptr, bytes, writes))) { *result = BM(m); *zc = 1; return 0; }
#else
    (void)writes;
#endif
    if (wait_gpu() || ensure_work(u, bytes)) return -1;
    if (copy_in) memcpy(u->p, ptr, bytes);
    *result = (bf_t){ NULL, u->p }; *zc = 0; return 0;
}

#define ARG(k, i, v) CHECK(clSetKernelArg(k, i, sizeof(v), &(v)))
// Scalar (alpha, beta) for level 3 kernels: double for FP64, float otherwise.
static cl_int arg_scalar(const type_t *r, cl_kernel k, int i, double v) {
    if (r->es == 8) return clSetKernelArg(k, i, sizeof v, &v);
    float f = (float)v; return clSetKernelArg(k, i, sizeof f, &f);
}
#define ARGS(k, i, v) CHECK(arg_scalar(r, k, i, v))

// End of a call: sleeping wait, then copy of the result to host memory (when it went through a work buffer).
// UHD 620 shares the LLC with the CPU, and NEO flushes the GPU cache at the end of a job: a result in USM is visible without map/unmap.
static int to_host(bf_t x, int zc, size_t bytes, void *host) {
    if (wait_gpu()) return -1;
    if (!zc) memcpy(host, x.p, bytes);
    return 0;
}

// Packing kernel and grid (3D: last dimension = batch). A: (Kp, Mp) / A_t: (Mp, Kp) / A_q: (Kp/8·4, Mp/16);
// B: (Np, Kp) / B_t: (Kp, Np) / B_w2: (Np/2, Kp). The k dimension is always in g[0] (A, B_t) or g[1] (A_t, B, B_w2).
static cl_kernel pack_kernel(type_t *r, int roleB, int trans, size_t Rp, size_t Kp, size_t cnt, size_t g[3]) {
    g[2] = cnt;
    if (!roleB) {
        // pa_q (FP16 only): work-item = 4 rows × 8 k (pk_a2_q) or 8 rows × 4 k (pk_a3_q). In FP32 the same
        // ideas gave no gain (the element is already 4 B: A 0.55 -> 0.62 ms, B 0.41 -> 0.38 ms at n = 1024).
        if (!trans && r->pa_q) { g[0] = Kp / 2; g[1] = Rp / 16; return r->pa_q; }
        if (trans) { g[0] = Rp; g[1] = Kp; return r->pa_t; }
        g[0] = Kp; g[1] = Rp; return r->pa;
    }
    if (!trans && r->pb_w2) { g[0] = Rp / 2; g[1] = Kp; return r->pb_w2; }
    if (trans) { g[0] = Kp; g[1] = Rp; return r->pb_t; }
    g[0] = Rp; g[1] = Kp; return r->pb;
}

static int gemm_skinny(int type, int tA, int tB, int M, int N, int K, double alpha, const void *A, int lda,
                      const void *B, int ldb, double beta, void *C, int ldc);
static int skinny_threshold(int type);
int gen9blas_gemm_batched(int type, int tA, int tB, int M, int N, int K, double alpha,
                     const void *A, int lda, long sA, const void *B, int ldb, long sB,
                     double beta, void *C, int ldc, long sC, int count) {
    NEW_OPERATION();
    if (type < 0 || type > 3 || !types[type].ready) FAIL("type %d unavailable", type);
    if (M <= 0 || N <= 0 || K <= 0 || count <= 0) return 0;
    if (count == 1 && (M < N ? M : N) <= skinny_threshold(type))   // small M or N: skinny kernel without packing
        return gemm_skinny(type, tA, tB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    double t_start = now_s();   // GEN9BLAS_PROFILE=1: stage times on stderr
    type_t *r = &types[type];
    int es = r->es, Mp = (M + 15) / 16 * 16, Np = (N + r->tn - 1) / r->tn * r->tn, Kp = (K + r->km - 1) / r->km * r->km;
    size_t c1 = count - 1;
    size_t bA = c1 * sA * es + (tA ? (size_t)(K - 1) * lda + M : (size_t)(M - 1) * lda + K) * es;
    size_t bB = c1 * sB * es + (tB ? (size_t)(N - 1) * ldb + K : (size_t)(K - 1) * ldb + N) * es;
    size_t bC = c1 * sC * es + ((size_t)(M - 1) * ldc + N) * es;
    bf_t xA, xB, xC; int zA, zB, zC;
    if (get_obj(A, bA, &uA, 1, 0, &xA, &zA) || get_obj(B, bB, &uB, 1, 0, &xB, &zB) || get_obj(C, bC, &uC, beta != 0.0f, 1, &xC, &zC))
        return -1;
    last_zc = zA | zB << 1 | zC << 2;
    if (ensure_buf(&dAp, &sAp, (size_t)count * Mp * Kp * es + 4096) || ensure_buf(&dBp, &sBp, (size_t)count * Np * Kp * es + 4096) ||
        ensure_buf(&dCp, &sCp, (size_t)count * Mp * Np * es))
        return -1;
    cl_long lA = sA, lB = sB, lC = sC;
    size_t ga[3], gb[3];
    cl_kernel ka = pack_kernel(r, 0, tA, Mp, Kp, count, ga), kb = pack_kernel(r, 1, tB, Np, Kp, count, gb);
    ARGB(ka, 0, xA); ARG(ka, 1, dAp); ARG(ka, 2, M); ARG(ka, 3, K); ARG(ka, 4, lda); ARG(ka, 5, Kp); ARG(ka, 6, lA); ARG(ka, 7, Mp);
    double t_buf = now_s() - t_start; cl_event evp[3];
    CHECK(clEnqueueNDRangeKernel(q, ka, 3, NULL, ga, NULL, 0, NULL, &evp[0]));
    ARGB(kb, 0, xB); ARG(kb, 1, dBp); ARG(kb, 2, K); ARG(kb, 3, N); ARG(kb, 4, ldb); ARG(kb, 5, Kp); ARG(kb, 6, lB); ARG(kb, 7, Np);
    CHECK(clEnqueueNDRangeKernel(q, kb, 3, NULL, gb, NULL, 0, NULL, &evp[1]));
    // Direct write (no Cp and no epilogue): alpha = 1, beta = 0, M and N multiples of the tile, batch contiguous in C.
    int direct = alpha == 1.0 && beta == 0.0 && M == Mp && N == Np && (count == 1 || sC == (long)M * ldc);
    bf_t target = direct ? xC : BM(dCp); int ldtarget = direct ? ldc : Np;
    ARG(r->main, 0, dAp); ARG(r->main, 1, dBp); ARGB(r->main, 2, target); ARG(r->main, 3, Mp); ARG(r->main, 4, Np);
    ARG(r->main, 5, ldtarget); ARG(r->main, 6, Kp); ARG(r->main, 7, Kp);
    size_t gm = (size_t)count * (Np / r->tn) * (Mp / 16) * 16, lm = 16;
    cl_event ev;
    CHECK(clEnqueueNDRangeKernel(q, r->main, 1, NULL, &gm, &lm, 0, NULL, &ev));
    ARG(r->epi, 0, dCp); ARG(r->epi, 1, Np); ARG(r->epi, 2, Mp); ARGB(r->epi, 3, xC); ARG(r->epi, 4, ldc); ARG(r->epi, 5, lC);
    ARGS(r->epi, 6, alpha); ARGS(r->epi, 7, beta);
    size_t ge[3] = { (size_t)N, (size_t)M, (size_t)count };
    if (direct) CHECK(clEnqueueMarkerWithWaitList(q, 0, NULL, &evp[2]));
    else CHECK(clEnqueueNDRangeKernel(q, r->epi, 3, NULL, ge, NULL, 0, NULL, &evp[2]));
    if (to_host(xC, zC, bC, C)) return -1;
    if (profile) {
        cl_ulong t0, t1;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof t0, &t0, NULL);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof t1, &t1, NULL);
        last_ms = (t1 - t0) * 1e-6;
    } else last_ms = (now_s() - t_start) * 1e3;
    clReleaseEvent(ev);
    if (getenv("GEN9BLAS_PROFILE")) {
        double tp[3];
        for (int e = 0; e < 3; e++) { cl_ulong a0, a1; clGetEventProfilingInfo(evp[e], CL_PROFILING_COMMAND_START, 8, &a0, NULL);
            clGetEventProfilingInfo(evp[e], CL_PROFILING_COMMAND_END, 8, &a1, NULL); tp[e] = (a1 - a0) * 1e-6; }
        fprintf(stderr, "gen9blas_gemm %dx%dx%d x%d: total %.2f ms (buffers %.2f, pack A %.2f, pack B %.2f, kernel %.2f, epilogue %.2f), zero-copy %d\n",
                M, N, K, count, (now_s() - t_start) * 1e3, t_buf * 1e3, tp[0], tp[1], last_ms, tp[2], last_zc);
    }
    for (int e = 0; e < 3; e++) clReleaseEvent(evp[e]);
    return 0;
}

// ---------------- device path: pack once, many kernel launches on chunks ----------------

// Packs op(X) into the buffer `target`: role A (roleB=0): op(X) R×K, trans: X stored as K×R; role B (roleB=1): op(X) K×R,
// trans: X stored as R×K. Zero padding up to Rp × Kp. Packing from a device buffer; for role B without transpose only the rows k in [k0, k0+kn)
// can be packed (global grid offset), e.g. successive solution blocks in TRSM. kn = 0: everything.
static int pack_dev(type_t *r, int roleB, int trans, bf_t xX, int ld, int Rr, int K, int Rp, int Kp, cl_mem target,
                     int k0, int kn) {
    cl_long zero = 0;
    size_t g[3], o[3] = { 0, 0, 0 };
    cl_kernel k = pack_kernel(r, roleB, trans, Rp, Kp, 1, g);
    if (!roleB) { ARGB(k, 0, xX); ARG(k, 1, target); ARG(k, 2, Rr); ARG(k, 3, K); ARG(k, 4, ld); ARG(k, 5, Kp); ARG(k, 6, zero); ARG(k, 7, Rp); }
    else        { ARGB(k, 0, xX); ARG(k, 1, target); ARG(k, 2, K); ARG(k, 3, Rr); ARG(k, 4, ld); ARG(k, 5, Kp); ARG(k, 6, zero); ARG(k, 7, Rp); }
    if (kn && roleB && !trans) { g[1] = kn; o[1] = k0; }   // k chunk (B and B_w2: k in g[1])
    CHECK(clEnqueueNDRangeKernel(q, k, 3, o, g, NULL, 0, NULL, NULL));
    return 0;
}

// Packs op(X) from host memory into the buffer `target` (see pack_dev).
static int pack_host(type_t *r, int roleB, int trans, const void *X, int ld, int Rr, int K, int Rp, int Kp, cl_mem target) {
    size_t bX = (roleB ? (trans ? (size_t)(Rr - 1) * ld + K : (size_t)(K - 1) * ld + Rr)
                       : (trans ? (size_t)(K - 1) * ld + Rr : (size_t)(Rr - 1) * ld + K)) * r->es;
    bf_t xX; int z;
    if (get_obj(X, bX, roleB ? &uB : &uA, 1, 0, &xX, &z)) return -1;
    return pack_dev(r, roleB, trans, xX, ld, Rr, K, Rp, Kp, target, 0, 0);
}

static cl_mem subbuffer(cl_mem b, size_t off, size_t size) {
    cl_int err; cl_buffer_region reg = { off, size };
    cl_mem s = clCreateSubBuffer(b, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg, &err);
    if (err != CL_SUCCESS) { snprintf(errbuf, sizeof errbuf, "clCreateSubBuffer = %d", err); return NULL; }
    return s;
}

// Epilogue on a device buffer: Cd = alpha·Cp + beta·Cd (M×N, the whole rectangle).
static int epilogue_dev(type_t *r, cl_mem Cp, int ldp, cl_mem Cd, int ldc, int M, int N, double alpha, double beta) {
    int uplo = 2;
    ARG(r->epi_tri, 0, Cp); ARG(r->epi_tri, 1, ldp); ARG(r->epi_tri, 2, Cd); ARG(r->epi_tri, 3, ldc);
    ARGS(r->epi_tri, 4, alpha); ARGS(r->epi_tri, 5, beta); ARG(r->epi_tri, 6, uplo);
    size_t g[2] = { (size_t)N, (size_t)M };
    CHECK(clEnqueueNDRangeKernel(q, r->epi_tri, 2, NULL, g, NULL, 0, NULL, NULL));
    return 0;
}

// GEMM kernel on a chunk: sub-buffers with offsets (bytes; aligned to 128 B). measure: event goes to evs[].
static int launch(type_t *r, cl_mem Ap, size_t oA, cl_mem Bp, size_t oB, cl_mem Cp, size_t oC,
                   int M, int N, int ldc, int K, int Kfull, int measure) {
    cl_int err; cl_mem sub[3] = { Ap, Bp, Cp }; size_t off[3] = { oA, oB, oC }; int sub_made[3] = { 0, 0, 0 };
    for (int i = 0; i < 3; i++) if (off[i]) {
        size_t size; clGetMemObjectInfo(sub[i], CL_MEM_SIZE, sizeof size, &size, NULL);
        cl_buffer_region reg = { off[i], size - off[i] };
        sub[i] = clCreateSubBuffer(sub[i], CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &reg, &err); CHECK(err); sub_made[i] = 1;
    }
    ARG(r->main, 0, sub[0]); ARG(r->main, 1, sub[1]); ARG(r->main, 2, sub[2]); ARG(r->main, 3, M); ARG(r->main, 4, N);
    ARG(r->main, 5, ldc); ARG(r->main, 6, K); ARG(r->main, 7, Kfull);
    size_t gm = (size_t)(N / r->tn) * (M / 16) * 16, lm = 16; cl_event ev;
    CHECK(clEnqueueNDRangeKernel(q, r->main, 1, NULL, &gm, &lm, 0, NULL, &ev));
    if (measure && nev < (int)(sizeof evs / sizeof evs[0])) evs[nev++] = ev; else clReleaseEvent(ev);
    for (int i = 0; i < 3; i++) if (sub_made[i]) clReleaseMemObject(sub[i]);
    return 0;
}

// Epilogue to host memory: C = alpha·Cp + beta·C on the uplo triangle (0 lower, 1 upper, 2 whole).
static int epilogue_host(type_t *r, cl_mem Cp, int ldp, void *C, int M, int N, int ldc, double alpha, double beta, int uplo) {
    size_t bC = ((size_t)(M - 1) * ldc + N) * r->es; bf_t xC; int zC;
    if (get_obj(C, bC, &uC, beta != 0.0f || uplo != 2, 1, &xC, &zC)) return -1;
    ARG(r->epi_tri, 0, Cp); ARG(r->epi_tri, 1, ldp); ARGB(r->epi_tri, 2, xC); ARG(r->epi_tri, 3, ldc);
    ARGS(r->epi_tri, 4, alpha); ARGS(r->epi_tri, 5, beta); ARG(r->epi_tri, 6, uplo);
    size_t g[2] = { (size_t)N, (size_t)M };
    CHECK(clEnqueueNDRangeKernel(q, r->epi_tri, 2, NULL, g, NULL, 0, NULL, NULL));
    return to_host(xC, zC, bC, C);
}

int gen9blas_gemm(int type, int tA, int tB, int M, int N, int K, double alpha, const void *A, int lda,
             const void *B, int ldb, double beta, void *C, int ldc) {
    return gen9blas_gemm_batched(type, tA, tB, M, N, K, alpha, A, lda, 0, B, ldb, 0, beta, C, ldc, 0, 1);
}
int gen9blas_hgemm(int M, int N, int K, double alpha, const uint16_t *A, int lda, const uint16_t *B, int ldb,
              double beta, uint16_t *C, int ldc) { return gen9blas_gemm(GEN9BLAS_H, 0, 0, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc); }
int gen9blas_sgemm(int M, int N, int K, double alpha, const float *A, int lda, const float *B, int ldb,
              double beta, float *C, int ldc) { return gen9blas_gemm(GEN9BLAS_S, 0, 0, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc); }

// ---------------- level 3 built on GEMM ----------------

#define CHECK_TYPE(type) do { NEW_OPERATION(); if ((type) < 0 || (type) > 3 || !types[type].ready) FAIL("type %d unavailable", type); } while (0)
static cl_mem dT, dX, dY;   // work buffers: full matrix (SYMM, TRMM), X and Y (SYR2K)
static size_t sT, sX, sY;
static int num_blocks(int n) { const char *e = getenv("GEN9BLAS_BLOCKS"); if (e) return atoi(e); return n >= 512 ? 4 : 1; }

// dT = full n×n matrix from the uplo triangle of A (host memory), on the GPU: sym=1 symmetric (mirrored),
// sym=0 triangular with zeros (diag=1: unit diagonal); tr=1: transposed result.
static int full_dev(type_t *r, const void *A, int lda, int n, int uplo, int sym, int diag, int tr) {
    size_t bA = ((size_t)(n - 1) * lda + n) * r->es; bf_t xA; int z;
    if (ensure_buf(&dT, &sT, (size_t)n * n * r->es) || get_obj(A, bA, &uA, 1, 0, &xA, &z)) return -1;
    cl_kernel k = r->expand;
    ARGB(k, 0, xA); ARG(k, 1, lda); ARG(k, 2, dT); ARG(k, 3, n); ARG(k, 4, uplo); ARG(k, 5, sym); ARG(k, 6, diag); ARG(k, 7, tr);
    size_t g[2] = { (size_t)(n + 15) / 16 * 16, (size_t)(n + 15) / 16 * 16 }, l[2] = { 16, 16 };
    CHECK(clEnqueueNDRangeKernel(q, k, 2, NULL, g, l, 0, NULL, NULL));
    return 0;
}

// C = alpha·F·B + beta·C (left) or alpha·B·F + beta·C (right), F = dT (n×n, n = M or N), B and C in host memory.
// shape F: 0 full, 1 lower, 2 upper -- then p bands (GEN9BLAS_BLOCKS), each only over the nonzero range of k (~62%
// of the work). C may be the same region as B (TRMM, beta = 0): B is packed before the epilogue (queue order).
static int f_times_b(type_t *r, int left, int shape, int M, int N, const void *B, int ldb, double alpha, double beta,
                    void *C, int ldc) {
    double t_in = now_s();
    int es = r->es, n = left ? M : N, w;
    int Mp = (M + 63) / 64 * 64, Np = (N + 63) / 64 * 64, Kp = left ? Mp : Np;
    if (ensure_buf(&dAp, &sAp, (size_t)Mp * Kp * es + 4096) || ensure_buf(&dBp, &sBp, (size_t)Np * Kp * es + 4096) ||
        ensure_buf(&dCp, &sCp, (size_t)Mp * Np * es))
        return -1;
    if (left) w = pack_dev(r, 0, 0, BM(dT), n, M, M, Mp, Kp, dAp, 0, 0) || pack_host(r, 1, 0, B, ldb, N, M, Np, Kp, dBp);
    else      w = pack_host(r, 0, 0, B, ldb, M, N, Mp, Kp, dAp) || pack_dev(r, 1, 0, BM(dT), n, N, N, Np, Kp, dBp, 0, 0);
    if (w) return -1;
    int p = shape ? num_blocks(n) : 1, dim = left ? Mp : Np, nb = ((dim + p - 1) / p + 63) / 64 * 64;
    nev = 0;
    for (int i0 = 0; i0 < dim && !w; i0 += nb) {
        int b = dim - i0 < nb ? dim - i0 : nb;
        if (left) {   // rows i0..i0+b: k in [0, i0+b) (lower) or [i0, Kp) (upper)
            int k0 = shape == 2 ? i0 : 0, k1 = shape == 1 ? i0 + b : Kp;
            w = launch(r, dAp, ((size_t)i0 * Kp + (size_t)k0 * 16) * es, dBp, (size_t)k0 * r->tn * es, dCp, (size_t)i0 * Np * es,
                        b, Np, Np, k1 - k0, Kp, 1);
        } else {      // columns i0..i0+b: k in [i0, Kp) (lower) or [0, i0+b) (upper)
            int k0 = shape == 1 ? i0 : 0, k1 = shape == 2 ? i0 + b : Kp;
            w = launch(r, dAp, (size_t)k0 * 16 * es, dBp, ((size_t)i0 * Kp + (size_t)k0 * r->tn) * es, dCp, (size_t)i0 * es,
                        Mp, b, Np, k1 - k0, Kp, 1);
        }
    }
    if (w || epilogue_host(r, dCp, Np, C, M, N, ldc, alpha, beta, 2)) return -1;
    { double tz = event_times(); last_ms = tz >= 0 ? tz : (now_s() - t_in) * 1e3; }
    return 0;
}

int gen9blas_symm(int type, int side, int uplo, int M, int N, double alpha, const void *A, int lda,
             const void *B, int ldb, double beta, void *C, int ldc) {
    CHECK_TYPE(type);
    type_t *r = &types[type]; int left = side == GEN9BLAS_LEFT;
    if (full_dev(r, A, lda, left ? M : N, uplo, 1, 0, 0)) return -1;
    return f_times_b(r, left, 0, M, N, B, ldb, alpha, beta, C, ldc);
}

int gen9blas_trmm(int type, int side, int uplo, int transA, int diag, int M, int N, double alpha,
             const void *A, int lda, void *B, int ldb) {
    // B = alpha·op(A)·B (left) or alpha·B·op(A) (right). F = op(A) explicitly (lower when uplo=L xor transA).
    CHECK_TYPE(type);
    type_t *r = &types[type]; int left = side == GEN9BLAS_LEFT, lower = (uplo == GEN9BLAS_LOWER) != (transA != 0);
    if (full_dev(r, A, lda, left ? M : N, uplo, 0, diag, transA)) return -1;
    return f_times_b(r, left, lower ? 1 : 2, M, N, B, ldb, alpha, 0.0f, B, ldb);
}

// Triangle (uplo) of alpha·op(X)·op(Y)ᵀ + beta·C, op(X), op(Y): N×K (tX: X, Y stored as K×N), X and Y on the GPU.
// First packing (tri_pack), then p kernel launches
// on row bands: lower triangle -> columns 0..i0+b, upper -> i0..N (~(p+1)/(2p) of the work), epilogue on the triangle.
static int tri_pack(type_t *r, int tX, int N, int K, bf_t X, int ldx, bf_t Y, int ldy) {
    int es = r->es, Np = (N + 63) / 64 * 64, Kp = (K + r->km - 1) / r->km * r->km;
    if (ensure_buf(&dAp, &sAp, (size_t)Np * Kp * es + 4096) || ensure_buf(&dBp, &sBp, (size_t)Np * Kp * es + 4096) ||
        ensure_buf(&dCp, &sCp, (size_t)Np * Np * es))
        return -1;
    return pack_dev(r, 0, tX, X, ldx, N, K, Np, Kp, dAp, 0, 0) || pack_dev(r, 1, !tX, Y, ldy, N, K, Np, Kp, dBp, 0, 0) ? -1 : 0;
}
static int tri_compute(type_t *r, int uplo, int N, int K, double alpha, double beta, void *C, int ldc) {
    double t_in = now_s();
    int es = r->es, Np = (N + 63) / 64 * 64, Kp = (K + r->km - 1) / r->km * r->km;
    int p = num_blocks(N), nb = ((Np + p - 1) / p + 63) / 64 * 64;
    nev = 0;
    for (int i0 = 0; i0 < Np; i0 += nb) {
        int b = Np - i0 < nb ? Np - i0 : nb;
        int c0 = uplo == GEN9BLAS_LOWER ? 0 : i0, c1 = uplo == GEN9BLAS_LOWER ? i0 + b : Np;
        if (launch(r, dAp, (size_t)i0 * Kp * es, dBp, (size_t)c0 * Kp * es, dCp, ((size_t)i0 * Np + c0) * es,
                    b, c1 - c0, Np, Kp, Kp, 1)) return -1;
    }
    if (epilogue_host(r, dCp, Np, C, N, N, ldc, alpha, beta, uplo)) return -1;
    { double tz = event_times(); last_ms = tz >= 0 ? tz : (now_s() - t_in) * 1e3; }
    return 0;
}

int gen9blas_syrk(int type, int uplo, int trans, int N, int K, double alpha, const void *A, int lda,
             double beta, void *C, int ldc) {
    CHECK_TYPE(type);
    type_t *r = &types[type];
    size_t bA = (trans ? (size_t)(K - 1) * lda + N : (size_t)(N - 1) * lda + K) * r->es; bf_t xA; int z;
    if (get_obj(A, bA, &uA, 1, 0, &xA, &z) || tri_pack(r, trans, N, K, xA, lda, xA, lda)) return -1;
    return tri_compute(r, uplo, N, K, alpha, beta, C, ldc);
}

int gen9blas_syr2k(int type, int uplo, int trans, int N, int K, double alpha, const void *A, int lda,
              const void *B, int ldb, double beta, void *C, int ldc) {
    // alpha·(op(A)·op(B)ᵀ + op(B)·op(A)ᵀ) = alpha·X·Yᵀ, X = [op(A) | op(B)], Y = [op(B) | op(A)] (N × 2K, joined on the GPU)
    CHECK_TYPE(type);
    type_t *r = &types[type]; int es = r->es, K2 = 2 * K;
    size_t bA = (trans ? (size_t)(K - 1) * lda + N : (size_t)(N - 1) * lda + K) * es;
    size_t bB = (trans ? (size_t)(K - 1) * ldb + N : (size_t)(N - 1) * ldb + K) * es;
    int zA, zB; bf_t xA, xB;
    if (ensure_buf(&dX, &sX, (size_t)N * K2 * es) || ensure_buf(&dY, &sY, (size_t)N * K2 * es) ||
        get_obj(A, bA, &uA, 1, 0, &xA, &zA) || get_obj(B, bB, &uB, 1, 0, &xB, &zB))
        return -1;
    cl_kernel k = r->syr2k_merge;
    ARGB(k, 0, xA); ARG(k, 1, lda); ARGB(k, 2, xB); ARG(k, 3, ldb); ARG(k, 4, dX); ARG(k, 5, dY); ARG(k, 6, K); ARG(k, 7, trans);
    size_t g[2] = { (size_t)K, (size_t)N };
    CHECK(clEnqueueNDRangeKernel(q, k, 2, NULL, g, NULL, 0, NULL, NULL));
    if (tri_pack(r, 0, N, K2, BM(dX), K2, BM(dY), K2)) return -1;
    return tri_compute(r, uplo, N, K2, alpha, beta, C, ldc);
}

int gen9blas_trsm(int type, int side, int uplo, int transA, int diag, int M, int N, double alpha,
             const void *A, int lda, void *B, int ldb) {
    // Solves op(A)·X = alpha·B (left) or X·op(A) = alpha·B (right); X overwrites B. FP32 and FP16 (GEN9BLAS_H, GEN9BLAS_HS).
    // Form L·Y = alpha·R with L lower (an upper one is reduced by reversing the index order, the right side by transposition).
    // Everything on the GPU: building L and Y from A, B (trsm_L/trsm_Y), inverses of the diagonal blocks of L in FP64 (trsm_inv),
    // L packed once, block loop (nb = 256): Y_i -= L[i, <i]·Y[<i]; Y_i = inv(L_ii)·Y_i; packing Y_i afterwards;
    // finally the write to B (trsm_zapisz).
    CHECK_TYPE(type);
    type_t *r = &types[type]; int es = r->es;
    int left = side == GEN9BLAS_LEFT, n = left ? M : N, m = left ? N : M;
    int tr = left ? transA : !transA, lower = (uplo == GEN9BLAS_LOWER) != tr, rev = !lower;
    const int nb = 256;
    int np = (n + 63) / 64 * 64, mp = (m + 63) / 64 * 64, w = 0, nbl = (n + nb - 1) / nb;
    double tp0 = now_s();
    // A and B for the GPU (B: input and output)
    size_t bA = ((size_t)(n - 1) * lda + n) * es, bB = ((size_t)(M - 1) * ldb + N) * es;
    bf_t xA, xB; int zA, zB; cl_event evi = NULL, evi2 = NULL;
    if (get_obj(A, bA, &uA, 1, 0, &xA, &zA) || get_obj(B, bB, &uB, 1, 1, &xB, &zB)) return -1;
    // TRSM work buffers (kept between calls): L, Y, Y packed, inv(L_ii) packed, R_i packed,
    // block inverses (element type), X and Tm (FP64, inversion)
    static cl_mem Ldv, Yd, Yp, Ip, Rp, Inv, Xd, Tm;
    static size_t sLdv, sYd, sYp, sIp, sRp, sInv, sXd, sTm;
    size_t blk = (size_t)nbl * nb * nb;
    if (ensure_buf(&Ldv, &sLdv, (size_t)n * n * es) || ensure_buf(&Yd, &sYd, (size_t)np * mp * es) ||
        ensure_buf(&Yp, &sYp, (size_t)mp * np * es + 4096) || ensure_buf(&Ip, &sIp, (size_t)nb * nb * es + 4096) ||
        ensure_buf(&Rp, &sRp, (size_t)mp * nb * es + 4096) || ensure_buf(&Inv, &sInv, blk * es) ||
        ensure_buf(&Xd, &sXd, blk * sizeof(double)) || ensure_buf(&Tm, &sTm, blk * sizeof(double)))
        return -1;
    if (ensure_buf(&dAp, &sAp, (size_t)np * np * es + 4096) || ensure_buf(&dCp, &sCp, (size_t)nb * mp * es)) w = -1;
    if (!w) {   // L = f(A), Y = alpha·R(B), inverses of the L blocks; L packed once
        cl_kernel k = r->tL; int u = uplo;
        ARGB(k, 0, xA); ARG(k, 1, lda); ARG(k, 2, Ldv); ARG(k, 3, n); ARG(k, 4, u); ARG(k, 5, tr); ARG(k, 6, diag); ARG(k, 7, rev);
        size_t g[2] = { (size_t)n, (size_t)n };
        CHECK(clEnqueueNDRangeKernel(q, k, 2, NULL, g, NULL, 0, NULL, NULL));
        k = r->tY;
        ARGB(k, 0, xB); ARG(k, 1, ldb); ARG(k, 2, Yd); ARG(k, 3, n); ARG(k, 4, m); ARG(k, 5, mp); ARG(k, 6, left); ARG(k, 7, rev); ARGS(k, 8, alpha);
        size_t gy[2] = { (size_t)mp, (size_t)np };
        CHECK(clEnqueueNDRangeKernel(q, k, 2, NULL, gy, NULL, 0, NULL, NULL));
        // inverses: 32×32 base blocks, then levels h = 32..nb/2: X21 = −X22·(L21·X11), finally converted to the element type
        k = r->tI[0];
        ARG(k, 0, Ldv); ARG(k, 1, n); ARG(k, 2, nb); ARG(k, 3, Xd);
        size_t g0[2] = { 32, (size_t)nbl * nb / 32 };
        CHECK(clEnqueueNDRangeKernel(q, k, 2, NULL, g0, NULL, 0, NULL, &evi));
        for (int h = 32; h < nb; h *= 2)
            for (int step = 1; step <= 2; step++) {
                k = r->tI[step];
                ARG(k, 0, Ldv); ARG(k, 1, n); ARG(k, 2, nb); ARG(k, 3, Xd); ARG(k, 4, Tm); ARG(k, 5, h);
                size_t gh[3] = { (size_t)h, (size_t)h, (size_t)nbl * nb / (2 * h) };
                CHECK(clEnqueueNDRangeKernel(q, k, 3, NULL, gh, NULL, 0, NULL, NULL));
            }
        k = r->tI[3];
        ARG(k, 0, n); ARG(k, 1, nb); ARG(k, 2, Xd); ARG(k, 3, Inv);
        size_t g3[3] = { (size_t)nb, (size_t)nb, (size_t)nbl };
        CHECK(clEnqueueNDRangeKernel(q, k, 3, NULL, g3, NULL, 0, NULL, &evi2));
        w = pack_dev(r, 0, 0, BM(Ldv), n, n, n, np, np, dAp, 0, 0);
    }
    for (int i0 = 0; i0 < n && !w; i0 += nb) {
        int b = n - i0 < nb ? n - i0 : nb, bp = (b + 63) / 64 * 64;
        cl_mem Yi = subbuffer(Yd, (size_t)i0 * mp * es, (size_t)bp * mp * es);
        cl_mem Ii = subbuffer(Inv, (size_t)i0 * nb * es, (size_t)b * b * es);
        if (!Yi || !Ii) { w = -1; if (Yi) clReleaseMemObject(Yi); break; }
        if (i0 > 0)   // Y_i -= L[i0:i0+b, 0:i0] · Y[0:i0]
            w = launch(r, dAp, (size_t)i0 * np * es, Yp, 0, dCp, 0, bp, mp, mp, i0, np, 0) ||
                epilogue_dev(r, dCp, mp, Yi, mp, b, m, -1.0, 1.0);
        if (!w)       // Y_i = inv(L_ii) · Y_i
            w = pack_dev(r, 0, 0, BM(Ii), b, b, b, bp, bp, Ip, 0, 0) ||
                pack_dev(r, 1, 0, BM(Yi), mp, m, b, mp, bp, Rp, 0, 0) ||
                launch(r, Ip, 0, Rp, 0, dCp, 0, bp, mp, mp, bp, bp, 0) ||
                epilogue_dev(r, dCp, mp, Yi, mp, b, m, 1.0, 0.0);
        if (!w)       // packing the solved block (rows k = i0..i0+b in Yp)
            w = pack_dev(r, 1, 0, BM(Yd), mp, m, n, mp, np, Yp, i0, bp);
        clReleaseMemObject(Yi); clReleaseMemObject(Ii);
    }
    if (!w) {         // write to B on the GPU, then to host memory
        cl_kernel k = r->tZ;
        ARG(k, 0, Yd); ARG(k, 1, mp); ARGB(k, 2, xB); ARG(k, 3, ldb); ARG(k, 4, n); ARG(k, 5, left); ARG(k, 6, rev);
        size_t g[2] = { (size_t)m, (size_t)n };
        CHECK(clEnqueueNDRangeKernel(q, k, 2, NULL, g, NULL, 0, NULL, NULL));
        w = to_host(xB, zB, bB, B);
    }
    last_ms = (now_s() - tp0) * 1e3;   // TRSM: time of the whole call
    if (evi && evi2) {
        if (getenv("GEN9BLAS_PROFILE")) { cl_ulong t0, t1; clGetEventProfilingInfo(evi, CL_PROFILING_COMMAND_START, 8, &t0, NULL);
            clGetEventProfilingInfo(evi2, CL_PROFILING_COMMAND_END, 8, &t1, NULL);
            fprintf(stderr, "trsm n=%d m=%d: %.1f ms (block inverses %.1f ms)\n", n, m, last_ms, (t1 - t0) * 1e-6); }
        clReleaseEvent(evi); clReleaseEvent(evi2);
    }
    return w;
}

int gen9blas_available(int type) { return type >= 0 && type <= 3 && types[type].ready; }
double gen9blas_last_time_ms(void) { return last_ms; }
int gen9blas_last_zero_copy(void) { return last_zc; }
const char *gen9blas_error(void) { return errbuf; }

// ---------------- levels 1–2 (host memory: USM zero-copy or a copy through a work buffer, see get_obj()) ----------------
static cl_mem dP;        // GEMV T partial sums
static size_t sP;
static double *red_out;      // reduction results (USM host: the GPU writes, the CPU reads after wait_gpu())
#define GROUPS 96         // work-groups (of 256) in reductions: 24 EUs × 4

static int l12_type(int type, l12_t **k, int *es) {
    NEW_OPERATION();
    int i = type == GEN9BLAS_S ? 1 : type == GEN9BLAS_D ? 2 : type == GEN9BLAS_H || type == GEN9BLAS_HS ? 0 : -1;
    if (i < 0 || !L12[i].axpy) FAIL("type %d unavailable", type);
    *k = &L12[i]; *es = L12[i].es; return 0;
}
// Scalar for a kernel in the compute type R (double for FP64, float for FP16/FP32).
static cl_int argr(cl_kernel K, int i, double v, int es) {
    if (es == 8) return clSetKernelArg(K, i, sizeof v, &v);
    float f = (float)v; return clSetKernelArg(K, i, sizeof f, &f);
}
#define ARGR(K, i, v) CHECK(argr(K, i, v, k->es))
#define RS(k) ((k)->es == 8 ? sizeof(double) : sizeof(float))   // size of the compute type
// Operand: region [p, p+b) of host memory; g -- pointer for the kernel (p or a work buffer), zc: zero-copy.
typedef struct { const void *p; size_t b; bf_t g; int zc; } op_t;
static work_t *const work_l12[3] = { &uA, &uB, &uC };
static int load_operands(op_t *w, int n, const int *copy_in, const int *written) {
    for (int i = 0; i < n; i++)
        if (get_obj(w[i].p, w[i].b, work_l12[i], copy_in[i], written ? written[i] : 0, &w[i].g, &w[i].zc)) return -1;
    return 0;
}
// Vector region: n elements spaced inc apart. inc < 0 as in BLAS: pointer to the lowest address, element 0 at the end.
static size_t vec_bytes(int n, int inc, int es) { return ((size_t)(n - 1) * (size_t)(inc < 0 ? -inc : inc) + 1) * es; }
static cl_long vec_off(int n, int inc) { return inc < 0 ? (cl_long)(n - 1) * -inc : 0; }
// End: wait, copy the written non-USM operands back to host memory.
static int finish(op_t *w, int n, const int *written) {
    if (wait_gpu()) return -1;
    for (int i = 0; i < n; i++) if (written[i] && !w[i].zc) memcpy((void *)w[i].p, w[i].g.p, w[i].b);
    return 0;
}
static int start1(cl_kernel k, int n);
// Element-wise kernels (axpy, scal, copy, rot, rotm): work-item = 4 elements.
static int start4(cl_kernel k, int n) { return start1(k, (n + 3) / 4); }
static int start1(cl_kernel k, int n) {
    size_t g = ((size_t)n + 255) / 256 * 256, l = 256;
    CHECK(clEnqueueNDRangeKernel(q, k, 1, NULL, &g, &l, 0, NULL, NULL)); return 0;
}
#define BEGIN(n, ...) l12_t *k; int es; if (l12_type(type, &k, &es)) return -1; if ((n) <= 0) return 0; \
                     if (__VA_ARGS__) FAIL("vector increment 0")

int gen9blas_axpy(int type, int n, double a, const void *x, int incx, void *y, int incy) {
    BEGIN(n, !incx || !incy);
    op_t w[2] = { { x, vec_bytes(n, incx, es) }, { y, vec_bytes(n, incy, es) } }; const int cin[2] = { 1, 1 }, cout[2] = { 0, 1 };
    if (load_operands(w, 2, cin, cout)) return -1;
    cl_long ox = vec_off(n, incx), oy = vec_off(n, incy), ix = incx, iy = incy; cl_kernel K = k->axpy;
    ARG(K, 0, n); ARGR(K, 1, a); ARGB(K, 2, w[0].g); ARG(K, 3, ox); ARG(K, 4, ix); ARGB(K, 5, w[1].g); ARG(K, 6, oy); ARG(K, 7, iy);
    return start4(K, n) || finish(w, 2, cout) ? -1 : 0;
}

int gen9blas_scal(int type, int n, double a, void *x, int incx) {
    BEGIN(n, !incx);
    op_t w[1] = { { x, vec_bytes(n, incx, es) } }; const int cin[1] = { 1 };
    if (load_operands(w, 1, cin, cin)) return -1;
    cl_long ox = vec_off(n, incx), ix = incx; cl_kernel K = k->scal;
    ARG(K, 0, n); ARGR(K, 1, a); ARGB(K, 2, w[0].g); ARG(K, 3, ox); ARG(K, 4, ix);
    return start4(K, n) || finish(w, 1, cin) ? -1 : 0;
}

int gen9blas_copy(int type, int n, const void *x, int incx, void *y, int incy) {
    BEGIN(n, !incx || !incy);
    // y is also copied into the work buffer for COPY: with increment > 1 the gaps between elements must stay unchanged
    op_t w[2] = { { x, vec_bytes(n, incx, es) }, { y, vec_bytes(n, incy, es) } }; const int cin[2] = { 1, incy != 1 && incy != -1 }, cout[2] = { 0, 1 };
    if (load_operands(w, 2, cin, cout)) return -1;
    cl_long ox = vec_off(n, incx), oy = vec_off(n, incy), ix = incx, iy = incy;
    if (incx == 1 && incy == 1) {   // 16 B per lane in every type (copy16): FP64 17 -> 24 GB/s
        // FP16: the compiler turns vload8/vstore8 of half into SIMD8 with 4 B byte reads (16 GB/s) -- copying does not depend
        // on the type, so an even number of elements aligned to 4 B goes through the FP32 kernel (n/2 32-bit words).
        l12_t *kk = k; int nn = n, e = es;
        if (es == 2 && n % 2 == 0 && ((uintptr_t)x | (uintptr_t)y) % 4 == 0) { kk = &L12[1]; nn = n / 2; e = 4; }
        cl_kernel K = kk->copy16; const int gw = e == 2 ? 4 : 16 / e;   // = GW in level12.cl (FP16: 4, FP32: 4, FP64: 2)
        ARG(K, 0, nn); ARGB(K, 1, w[0].g); ARG(K, 2, ox); ARGB(K, 3, w[1].g); ARG(K, 4, oy);
        return start1(K, (nn + gw - 1) / gw) || finish(w, 2, cout) ? -1 : 0;
    }
    cl_kernel K = k->copy;
    ARG(K, 0, n); ARGB(K, 1, w[0].g); ARG(K, 2, ox); ARG(K, 3, ix); ARGB(K, 4, w[1].g); ARG(K, 5, oy); ARG(K, 6, iy);
    return start4(K, n) || finish(w, 2, cout) ? -1 : 0;
}

static int rot(int type, int n, void *x, int incx, void *y, int incy, double c, double s, int swap_flag) {
    BEGIN(n, !incx || !incy);
    op_t w[2] = { { x, vec_bytes(n, incx, es) }, { y, vec_bytes(n, incy, es) } }; const int cin[2] = { 1, 1 };
    if (load_operands(w, 2, cin, cin)) return -1;
    cl_long ox = vec_off(n, incx), oy = vec_off(n, incy), ix = incx, iy = incy; cl_kernel K = k->rot;
    ARG(K, 0, n); ARGB(K, 1, w[0].g); ARG(K, 2, ox); ARG(K, 3, ix); ARGB(K, 4, w[1].g); ARG(K, 5, oy); ARG(K, 6, iy);
    ARGR(K, 7, c); ARGR(K, 8, s); ARG(K, 9, swap_flag);
    return start4(K, n) || finish(w, 2, cin) ? -1 : 0;
}
// ROTM: H from param (BLAS): param[0] = flag: -1 full [h11 h12; h21 h22], 0 [1 h12; h21 1], 1 [h11 1; -1 h22],
// -2 identity (nothing to do); param[1..4] = h11, h21, h12, h22.
int gen9blas_rotm(int type, int n, void *x, int incx, void *y, int incy, const double *param) {
    double fl = param[0], h11, h12, h21, h22;
    if (fl == -2.0) return 0;
    if (fl < 0) { h11 = param[1]; h21 = param[2]; h12 = param[3]; h22 = param[4]; }
    else if (fl == 0) { h11 = 1; h21 = param[2]; h12 = param[3]; h22 = 1; }
    else { h11 = param[1]; h21 = -1; h12 = 1; h22 = param[4]; }
    BEGIN(n, !incx || !incy);
    op_t w[2] = { { x, vec_bytes(n, incx, es) }, { y, vec_bytes(n, incy, es) } }; const int cin[2] = { 1, 1 };
    if (load_operands(w, 2, cin, cin)) return -1;
    cl_long ox = vec_off(n, incx), oy = vec_off(n, incy), ix = incx, iy = incy; cl_kernel K = k->rotm;
    ARG(K, 0, n); ARGB(K, 1, w[0].g); ARG(K, 2, ox); ARG(K, 3, ix); ARGB(K, 4, w[1].g); ARG(K, 5, oy); ARG(K, 6, iy);
    ARGR(K, 7, h11); ARGR(K, 8, h12); ARGR(K, 9, h21); ARGR(K, 10, h22);
    return start4(K, n) || finish(w, 2, cin) ? -1 : 0;
}

// ROTG and ROTMG: computations on a few numbers -- on the CPU, reference algorithms (Netlib BLAS 3.10+).
// ROTG: [c s; -s c]·[a; b] = [r; 0]; *a = r, *b = z (c, s encoded in a single number), overflow-safe.
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
// ROTMG: modified Givens rotation (d1, d2 -- squared scales, x1, y1); result in param (see gen9blas_rotm).
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

int gen9blas_swap(int type, int n, void *x, int incx, void *y, int incy) { return rot(type, n, x, incx, y, incy, 1, 0, 1); }
int gen9blas_rot(int type, int n, void *x, int incx, void *y, int incy, double c, double s) { return rot(type, n, x, incx, y, incy, c, s, 0); }

static int alloc_red_out(void) {
    if (!red_out && !(red_out = gen9blas_alloc(2 * GROUPS * sizeof(double)))) return -1;
    return 0;
}
static int reduction(int type, int mode, int n, const void *x, int incx, const void *y, int incy, double *result) {
    *result = 0;
    BEGIN(n, !incx || (mode == 0 && !incy));
    op_t w[2] = { { x, vec_bytes(n, incx, es) }, { y, mode == 0 ? vec_bytes(n, incy, es) : 0 } }; const int cin[2] = { 1, 1 };
    if (load_operands(w, mode == 0 ? 2 : 1, cin, NULL) || alloc_red_out()) return -1;
    if (mode) { w[1] = w[0]; incy = incx; }
    int g = (n + 255) / 256 < GROUPS ? (n + 255) / 256 : GROUPS;
    cl_long ox = vec_off(n, incx), oy = vec_off(n, incy), ix = incx, iy = incy; cl_kernel K = k->red;
    ARG(K, 0, n); ARG(K, 1, mode); ARGB(K, 2, w[0].g); ARG(K, 3, ox); ARG(K, 4, ix); ARGB(K, 5, w[1].g); ARG(K, 6, oy); ARG(K, 7, iy);
    CHECK(usm_arg(K, 8, red_out));
    size_t G = (size_t)g * 256, Lw = 256;
    CHECK(clEnqueueNDRangeKernel(q, K, 1, NULL, &G, &Lw, 0, NULL, NULL));
    if (wait_gpu()) return -1;
    double s = 0; for (int i = 0; i < g; i++) s += red_out[i];
    *result = mode == 2 ? sqrt(s) : s;
    return 0;
}
int gen9blas_dot(int type, int n, const void *x, int incx, const void *y, int incy, double *result) { return reduction(type, 0, n, x, incx, y, incy, result); }
int gen9blas_asum(int type, int n, const void *x, int incx, double *result) { return reduction(type, 1, n, x, incx, NULL, 0, result); }
int gen9blas_nrm2(int type, int n, const void *x, int incx, double *result) { return reduction(type, 2, n, x, incx, NULL, 0, result); }

int gen9blas_iamax(int type, int n, const void *x, int incx, long *result) {
    *result = -1;
    BEGIN(n, !incx);
    op_t w[1] = { { x, vec_bytes(n, incx, es) } }; const int cin[1] = { 1 };
    if (load_operands(w, 1, cin, NULL) || alloc_red_out()) return -1;
    int g = (n + 255) / 256 < GROUPS ? (n + 255) / 256 : GROUPS;
    cl_long ox = vec_off(n, incx), ix = incx; cl_kernel K = k->iamax;
    ARG(K, 0, n); ARGB(K, 1, w[0].g); ARG(K, 2, ox); ARG(K, 3, ix); CHECK(usm_arg(K, 4, red_out));
    size_t G = (size_t)g * 256, Lw = 256;
    CHECK(clEnqueueNDRangeKernel(q, K, 1, NULL, &G, &Lw, 0, NULL, NULL));
    if (wait_gpu()) return -1;
    // values in the compute type (float or double), int indices after the 96 values
    const int *ci = (const int *)((const char *)red_out + GROUPS * RS(k));
    double cv[GROUPS];
    for (int i = 0; i < g; i++) cv[i] = k->es == 8 ? ((const double *)red_out)[i] : ((const float *)red_out)[i];
    int b = 0;
    for (int i = 1; i < g; i++) if (cv[i] > cv[b] || (cv[i] == cv[b] && ci[i] < ci[b])) b = i;
    *result = ci[b] == 0x7fffffff ? 0 : ci[b];   // all NaN: index 0
    return 0;
}

// GEMV on buffers (no waiting): y = alpha·op(A)·x + beta·y with the triangle mask mode (0 whole, 1 j <= i, 2 j >= i,
// 3 j < i, 4 j > i for element A[i][j]); unit = 1: x at the diagonal index is added to the sum (unit diagonal).
static int gemv_inner(l12_t *k, int trans, int M, int N, double alpha, bf_t A, int lda, bf_t x, cl_long ox, cl_long ix,
                    double beta, bf_t y, cl_long oy, cl_long iy, int mode, int unit) {
    cl_long oa = 0;
    if (!trans) {
        // Rows of 1024 columns and more: the grouped variant (16 subgroups on the same 4 rows, few pages at once). On ordinary memory
        // (userptr, 4 KB pages) the old one lost 2× (n = 4096: 10.5 -> 18.9 GB/s), on USM -5..+20%; with N <= 512
        // grouped is slower (too few columns for 16 subgroups). Measurement: bench/gemv_n.py, results/gemv-n-2026-11-12.txt.
        // GEN9BLAS_GEMV_N=old|grouped forces a variant (comparisons).
        static int forced = -1;
        if (forced < 0) { const char *e = getenv("GEN9BLAS_GEMV_N"); forced = !e ? 0 : !strcmp(e, "old") ? 1 : !strcmp(e, "grouped") ? 2 : 0; }
        const int grouped = forced == 2 || (forced == 0 && N >= 1024);
        cl_kernel K = grouped ? k->gemv_nw : k->gemv_n;
        ARG(K, 0, M); ARG(K, 1, N); ARGR(K, 2, alpha); ARGB(K, 3, A); ARG(K, 4, oa); ARG(K, 5, lda); ARGB(K, 6, x);
        ARG(K, 7, ox); ARG(K, 8, ix); ARGR(K, 9, beta); ARGB(K, 10, y); ARG(K, 11, oy); ARG(K, 12, iy); ARG(K, 13, mode); ARG(K, 14, unit);
        size_t g = ((size_t)(M + 3) / 4 * 16 + 63) / 64 * 64, l = 64;
        if (grouped) { l = 256; g = (size_t)(M + 3) / 4 * 256; }
        CHECK(clEnqueueNDRangeKernel(q, K, 1, NULL, &g, &l, 0, NULL, NULL));
        return 0;
    }
    int cols4 = (N + 3) / 4, parts = (16384 + cols4 - 1) / cols4;   // ~16 thousand work-items
    if (parts > M) parts = M;
    int rpp = (M + parts - 1) / parts; parts = (M + rpp - 1) / rpp;
    if (ensure_buf(&dP, &sP, (size_t)parts * N * RS(k))) return -1;
    cl_kernel K;
    if (k->es == 2 && !mode && !unit) {   // FP16, full matrix: block reads (gemv_t1b; results/message-16B-...)
        K = k->gemv_t1b;
        ARG(K, 0, M); ARG(K, 1, N); ARG(K, 2, rpp); ARGB(K, 3, A); ARG(K, 4, oa); ARG(K, 5, lda); ARGB(K, 6, x);
        ARG(K, 7, ox); ARG(K, 8, ix); ARG(K, 9, dP);
    } else {
        K = k->gemv_t1;
        ARG(K, 0, M); ARG(K, 1, N); ARG(K, 2, rpp); ARGB(K, 3, A); ARG(K, 4, oa); ARG(K, 5, lda); ARGB(K, 6, x);
        ARG(K, 7, ox); ARG(K, 8, ix); ARG(K, 9, dP); ARG(K, 10, mode); ARG(K, 11, unit);
    }
    size_t g[2] = { ((size_t)cols4 + 63) / 64 * 64, (size_t)parts }, l[2] = { 64, 1 };
    CHECK(clEnqueueNDRangeKernel(q, K, 2, NULL, g, l, 0, NULL, NULL));
    K = k->gemv_t2;
    ARG(K, 0, N); ARG(K, 1, parts); ARG(K, 2, dP); ARGR(K, 3, alpha); ARGR(K, 4, beta); ARGB(K, 5, y); ARG(K, 6, oy); ARG(K, 7, iy);
    return start1(K, N);
}

int gen9blas_gemv(int type, int trans, int M, int N, double alpha, const void *A, int lda, const void *x, int incx,
             double beta, void *y, int incy) {
    BEGIN(M < N ? M : N, !incx || !incy);
    if (lda < N) FAIL("lda < N");
    int nx = trans ? M : N, ny = trans ? N : M;
    op_t w[3] = { { A, ((size_t)(M - 1) * lda + N) * es }, { x, vec_bytes(nx, incx, es) }, { y, vec_bytes(ny, incy, es) } };
    const int cin[3] = { 1, 1, beta != 0.0f || (incy != 1 && incy != -1) }, cout[3] = { 0, 0, 1 };
    if (load_operands(w, 3, cin, cout)) return -1;
    if (gemv_inner(k, trans, M, N, alpha, w[0].g, lda, w[1].g, vec_off(nx, incx), incx, beta, w[2].g, vec_off(ny, incy), incy, 0, 0)) return -1;
    return finish(w, 3, cout);
}

// SYMV: y = alpha·A·x + beta·y, A symmetric n×n, the uplo triangle is used. One pass over the stored triangle
// (level12.cl: symv1 -- partial sums of rows into Q and of columns into P, symv2 -- combination).
static cl_mem dQ;
static size_t sQ;
int gen9blas_symv(int type, int uplo, int n, double alpha, const void *A, int lda, const void *x, int incx,
             double beta, void *y, int incy) {
    BEGIN(n, !incx || !incy);
    if (lda < n) FAIL("lda < n");
    op_t w[3] = { { A, ((size_t)(n - 1) * lda + n) * es }, { x, vec_bytes(n, incx, es) }, { y, vec_bytes(n, incy, es) } };
    const int cin[3] = { 1, 1, beta != 0.0f || (incy != 1 && incy != -1) }, cout[3] = { 0, 0, 1 };
    if (load_operands(w, 3, cin, cout)) return -1;
    int upper = uplo == GEN9BLAS_UPPER, cols4 = (n + 3) / 4, parts = (16384 + cols4 - 1) / cols4, ng = (n + 63) / 64;
    if (parts > n) parts = n;
    int rpp = (n + parts - 1) / parts; parts = (n + rpp - 1) / rpp;
    if (ensure_buf(&dP, &sP, (size_t)parts * n * RS(k)) || ensure_buf(&dQ, &sQ, (size_t)ng * n * RS(k))) return -1;
    cl_long oa = 0, ox = vec_off(n, incx), oy = vec_off(n, incy), ix = incx, iy = incy;
    cl_kernel K = k->symv1;
    ARG(K, 0, n); ARG(K, 1, rpp); ARG(K, 2, upper); ARGB(K, 3, w[0].g); ARG(K, 4, oa); ARG(K, 5, lda); ARGB(K, 6, w[1].g);
    ARG(K, 7, ox); ARG(K, 8, ix); ARG(K, 9, dP); ARG(K, 10, dQ);
    size_t g[2] = { (size_t)ng * 16, (size_t)parts }, l[2] = { 64, 1 };
    if (g[0] % 64) g[0] = (g[0] + 63) / 64 * 64;
    CHECK(clEnqueueNDRangeKernel(q, K, 2, NULL, g, l, 0, NULL, NULL));
    K = k->symv2;
    ARG(K, 0, n); ARG(K, 1, parts); ARG(K, 2, upper); ARG(K, 3, dP); ARG(K, 4, dQ); ARGR(K, 5, alpha); ARGR(K, 6, beta);
    ARGB(K, 7, w[2].g); ARG(K, 8, oy); ARG(K, 9, iy);
    if (start1(K, n)) return -1;
    return finish(w, 3, cout);
}

static cl_mem dTv, dXf;   // TRMV: intermediate result (element type); TRSV: FP32 work vector
static size_t sTv, sXf;

// TRMV: x = op(A)·x, A triangular n×n (uplo; diag = 1: unit diagonal). The result goes to a buffer, then is copied to x.
int gen9blas_trmv(int type, int uplo, int trans, int diag, int n, const void *A, int lda, void *x, int incx) {
    BEGIN(n, !incx);
    if (lda < n) FAIL("lda < n");
    op_t w[2] = { { A, ((size_t)(n - 1) * lda + n) * es }, { x, vec_bytes(n, incx, es) } };
    const int cin[2] = { 1, 1 }, cout[2] = { 0, 1 };
    if (load_operands(w, 2, cin, cout) || ensure_buf(&dTv, &sTv, (size_t)n * es)) return -1;
    int upper = uplo == GEN9BLAS_UPPER, mode = diag ? (upper ? 4 : 3) : (upper ? 2 : 1);
    cl_long ox = vec_off(n, incx), ix = incx, z = 0, j = 1;
    if (gemv_inner(k, trans, n, n, 1.0f, w[0].g, lda, w[1].g, ox, incx, 0.0f, BM(dTv), 0, 1, mode, diag)) return -1;
    cl_kernel K = k->copy;
    ARG(K, 0, n); ARG(K, 1, dTv); ARG(K, 2, z); ARG(K, 3, j); ARGB(K, 4, w[1].g); ARG(K, 5, ox); ARG(K, 6, ix);
    return start4(K, n) || finish(w, 2, cout) ? -1 : 0;
}

// TRSV: solve op(A)·x = b (b in x, result in x). In blocks of 256: the diagonal block in a single work-group (sub-blocks of 64
// in local memory), then an update of the rest of the vector; 2 launches per block. FP32 work vector (also for FP16).
// op(A) lower: forward. (A single-launch version without global synchronization: slower, docs/pitfalls.md.)
int gen9blas_trsv(int type, int uplo, int trans, int diag, int n, const void *A, int lda, void *x, int incx) {
    BEGIN(n, !incx);
    if (lda < n) FAIL("lda < n");
    op_t w[2] = { { A, ((size_t)(n - 1) * lda + n) * es }, { x, vec_bytes(n, incx, es) } };
    const int cin[2] = { 1, 1 }, cout[2] = { 0, 1 };
    if (load_operands(w, 2, cin, cout) || ensure_buf(&dXf, &sXf, (size_t)n * RS(k))) return -1;
    cl_long ox = vec_off(n, incx), ix = incx, oa = 0;
    int lower = (uplo == GEN9BLAS_LOWER) != (trans != 0), tr = trans != 0, dg = diag != 0;
    cl_kernel K = k->to_f;
    ARG(K, 0, n); ARGB(K, 1, w[1].g); ARG(K, 2, ox); ARG(K, 3, ix); ARG(K, 4, dXf);
    if (start1(K, n)) return -1;
    const int nb = 256, nbl = (n + nb - 1) / nb;   // = NBT in level12.cl
    static int prof = -1; if (prof < 0) prof = getenv("GEN9BLAS_PROFILE") != NULL;
    cl_event ev[2 * 1024]; int ne = 0;
    if (prof && 2 * nbl > 2 * 1024) prof = 0;
    for (int s = 0; s < nbl; s++) {
        int p = (lower ? s : nbl - 1 - s) * nb, b = n - p < nb ? n - p : nb;
        K = k->trsv_block;
        ARG(K, 0, p); ARG(K, 1, b); ARGB(K, 2, w[0].g); ARG(K, 3, oa); ARG(K, 4, lda); ARG(K, 5, tr); ARG(K, 6, lower);
        ARG(K, 7, dg); ARG(K, 8, dXf);
        size_t g1 = 256, l1 = 256;
        CHECK(clEnqueueNDRangeKernel(q, K, 1, NULL, &g1, &l1, 0, NULL, prof ? &ev[ne++] : NULL));
        int r0 = lower ? p + b : 0, r1 = lower ? n : p;
        if (r1 <= r0) continue;
        K = tr ? k->trsv_upd_t : k->trsv_upd_n;
        ARG(K, 0, p); ARG(K, 1, b); ARG(K, 2, r0); ARG(K, 3, r1); ARGB(K, 4, w[0].g); ARG(K, 5, oa); ARG(K, 6, lda); ARG(K, 7, dXf);
        size_t g = ((size_t)(r1 - r0) * (tr ? 1 : 16) + 63) / 64 * 64, l = 64;
        CHECK(clEnqueueNDRangeKernel(q, K, 1, NULL, &g, &l, 0, NULL, prof ? &ev[ne++] : NULL));
    }
    if (prof) {   // GEN9BLAS_PROFILE=1: kernel times (diagonal blocks / updates) and the gaps between them
        if (wait_gpu()) return -1;
        double tb = 0, ta = 0; cl_ulong first_t = 0, last_t = 0;
        for (int i = 0; i < ne; i++) {
            cl_ulong t0, t1; clGetEventProfilingInfo(ev[i], CL_PROFILING_COMMAND_START, 8, &t0, NULL);
            clGetEventProfilingInfo(ev[i], CL_PROFILING_COMMAND_END, 8, &t1, NULL);
            if (!i) first_t = t0;
            last_t = t1; *(i % 2 ? &ta : &tb) += (t1 - t0) * 1e-6; clReleaseEvent(ev[i]);
        }
        fprintf(stderr, "trsv n=%d: blocks %.2f ms, updates %.2f ms, first to last %.2f ms (%d kernels)\n",
                n, tb, ta, (last_t - first_t) * 1e-6, ne);
    }
    K = k->from_f;
    ARG(K, 0, n); ARG(K, 1, dXf); ARGB(K, 2, w[1].g); ARG(K, 3, ox); ARG(K, 4, ix);
    return start1(K, n) || finish(w, 2, cout) ? -1 : 0;
}

// SYR / SYR2 on the uplo triangle: A = alpha·x·xᵀ + A or A = alpha·(x·yᵀ + y·xᵀ) + A.
static int syr2(int type, int uplo, int n, double alpha, const void *x, int incx, const void *y, int incy, void *A, int lda) {
    BEGIN(n, !incx || (y && !incy));
    if (lda < n) FAIL("lda < n");
    int two = y != NULL;
    op_t w[3] = { { A, ((size_t)(n - 1) * lda + n) * es }, { x, vec_bytes(n, incx, es) }, { two ? y : x, vec_bytes(n, two ? incy : incx, es) } };
    const int cin[3] = { 1, 1, 1 }, cout[3] = { 1, 0, 0 };
    if (load_operands(w, two ? 3 : 2, cin, cout)) return -1;
    if (!two) { w[2] = w[1]; incy = incx; }
    cl_long ox = vec_off(n, incx), oy = vec_off(n, incy), oa = 0, ix = incx, iy = incy;
    int mode = uplo == GEN9BLAS_UPPER ? 2 : 1; cl_kernel K = k->syr2;
    ARG(K, 0, n); ARGR(K, 1, alpha); ARGB(K, 2, w[1].g); ARG(K, 3, ox); ARG(K, 4, ix); ARGB(K, 5, w[2].g); ARG(K, 6, oy);
    ARG(K, 7, iy); ARGB(K, 8, w[0].g); ARG(K, 9, oa); ARG(K, 10, lda); ARG(K, 11, mode); ARG(K, 12, two);
    size_t g[2] = { ((size_t)n + 63) / 64 * 64, (size_t)n }, l[2] = { 64, 1 };
    CHECK(clEnqueueNDRangeKernel(q, K, 2, NULL, g, l, 0, NULL, NULL));
    return finish(w, 1, cout);
}
int gen9blas_syr(int type, int uplo, int n, double alpha, const void *x, int incx, void *A, int lda) {
    return syr2(type, uplo, n, alpha, x, incx, NULL, 0, A, lda);
}
int gen9blas_syr2(int type, int uplo, int n, double alpha, const void *x, int incx, const void *y, int incy, void *A, int lda) {
    return syr2(type, uplo, n, alpha, x, incx, y, incy, A, lda);
}

int gen9blas_ger(int type, int M, int N, double alpha, const void *x, int incx, const void *y, int incy, void *A, int lda) {
    BEGIN(M < N ? M : N, !incx || !incy);
    if (lda < N) FAIL("lda < N");
    op_t w[3] = { { x, vec_bytes(M, incx, es) }, { y, vec_bytes(N, incy, es) }, { A, ((size_t)(M - 1) * lda + N) * es } };
    const int cin[3] = { 1, 1, 1 }, cout[3] = { 0, 0, 1 };
    if (load_operands(w, 3, cin, cout)) return -1;
    cl_long ox = vec_off(M, incx), oy = vec_off(N, incy), oa = 0, ix = incx, iy = incy; cl_kernel K = k->ger;
    ARG(K, 0, M); ARG(K, 1, N); ARGR(K, 2, alpha); ARGB(K, 3, w[0].g); ARG(K, 4, ox); ARG(K, 5, ix); ARGB(K, 6, w[1].g);
    ARG(K, 7, oy); ARG(K, 8, iy); ARGB(K, 9, w[2].g); ARG(K, 10, oa); ARG(K, 11, lda);
    size_t g[2] = { ((size_t)N + 63) / 64 * 64, (size_t)M }, l[2] = { 64, 1 };
    CHECK(clEnqueueNDRangeKernel(q, K, 2, NULL, g, l, 0, NULL, NULL));
    return finish(w, 3, cout);
}

// ---------------- level 2: banded and packed matrices (row-major layout as in CBLAS RowMajor; see level12.cl) ----------------
enum { BAND = 0, PACK_U = 1, PACK_L = 2 };
static int elem_size(int type) { return type == GEN9BLAS_S ? 4 : type == GEN9BLAS_D ? 8 : 2; }
static size_t band_bytes(int rows, int lda, int KL, int KU, int es) { return ((size_t)(rows - 1) * lda + KL + KU + 1) * es; }
static size_t packed_bytes(int n, int es) { return (size_t)n * (n + 1) / 2 * es; }

// y = alpha·op(A)·x + beta·y (A in buffer a, format description) -- no waiting; y may be an intermediate buffer.
static int pmv_inner(l12_t *k, int ny, int nx, int fmt, int KL, int KU, int n, bf_t a, int lda, bf_t x, cl_long ox, cl_long ix,
                   double alpha, double beta, bf_t y, cl_long oy, cl_long iy, int tr, int sym, int upper, int unit, int lo, int hi) {
    cl_long oa = 0; cl_kernel K = k->pmv;
    ARG(K, 0, ny); ARG(K, 1, nx); ARG(K, 2, fmt); ARG(K, 3, KL); ARG(K, 4, KU); ARG(K, 5, n); ARGB(K, 6, a); ARG(K, 7, oa);
    ARG(K, 8, lda); ARGB(K, 9, x); ARG(K, 10, ox); ARG(K, 11, ix); ARGR(K, 12, alpha); ARGR(K, 13, beta); ARGB(K, 14, y);
    ARG(K, 15, oy); ARG(K, 16, iy); ARG(K, 17, tr); ARG(K, 18, sym); ARG(K, 19, upper); ARG(K, 20, unit); ARG(K, 21, lo); ARG(K, 22, hi);
    return start1(K, ny);
}
// Product with the result in y (GBMV, SBMV, SPMV).
static int pmv_y(int type, int M, int N, int fmt, int KL, int KU, int tr, int sym, int upper, int lo, int hi,
                 double alpha, const void *A, size_t bA, int lda, const void *x, int incx, double beta, void *y, int incy) {
    int nx = tr ? M : N, ny = tr ? N : M;
    BEGIN(nx < ny ? nx : ny, !incx || !incy);
    op_t w[3] = { { A, bA }, { x, vec_bytes(nx, incx, es) }, { y, vec_bytes(ny, incy, es) } };
    const int cin[3] = { 1, 1, beta != 0.0f || (incy != 1 && incy != -1) }, cout[3] = { 0, 0, 1 };
    if (load_operands(w, 3, cin, cout)) return -1;
    if (pmv_inner(k, ny, nx, fmt, KL, KU, N, w[0].g, lda, w[1].g, vec_off(nx, incx), incx, alpha, beta, w[2].g, vec_off(ny, incy), incy,
                tr, sym, upper, 0, lo, hi)) return -1;
    return finish(w, 3, cout);
}
int gen9blas_gbmv(int type, int trans, int M, int N, int KL, int KU, double alpha, const void *A, int lda, const void *x, int incx,
             double beta, void *y, int incy) {
    if (KL < 0 || KU < 0 || lda < KL + KU + 1) FAIL("gbmv: KL, KU, lda");
    int es = elem_size(type);
    return pmv_y(type, M, N, BAND, KL, KU, trans != 0, 0, 0, trans ? KU : KL, trans ? KL : KU, alpha, A,
                 M > 0 ? band_bytes(M, lda, KL, KU, es) : 0, lda, x, incx, beta, y, incy);
}
int gen9blas_sbmv(int type, int uplo, int N, int K, double alpha, const void *A, int lda, const void *x, int incx,
             double beta, void *y, int incy) {
    if (K < 0 || lda < K + 1) FAIL("sbmv: K, lda");
    int es = elem_size(type), g = uplo == GEN9BLAS_UPPER;
    return pmv_y(type, N, N, BAND, g ? 0 : K, g ? K : 0, 0, 1, g, K, K, alpha, A, N > 0 ? band_bytes(N, lda, 0, K, es) : 0,
                 lda, x, incx, beta, y, incy);
}
int gen9blas_spmv(int type, int uplo, int N, double alpha, const void *Ap, const void *x, int incx, double beta, void *y, int incy) {
    int es = elem_size(type), g = uplo == GEN9BLAS_UPPER;
    return pmv_y(type, N, N, g ? PACK_U : PACK_L, 0, 0, 0, 1, g, N, N, alpha, Ap, packed_bytes(N, es), 0, x, incx, beta, y, incy);
}

// x = op(A)·x (TBMV, TPMV): the result goes to an intermediate buffer, then is copied to x.
static int pmv_x(int type, int N, int fmt, int KL, int KU, int tr, int unit, int lo, int hi, const void *A, size_t bA, int lda,
                 void *x, int incx) {
    BEGIN(N, !incx);
    op_t w[2] = { { A, bA }, { x, vec_bytes(N, incx, es) } };
    const int cin[2] = { 1, 1 }, cout[2] = { 0, 1 };
    if (load_operands(w, 2, cin, cout) || ensure_buf(&dTv, &sTv, (size_t)N * es)) return -1;
    cl_long ox = vec_off(N, incx), ix = incx, z = 0, j = 1;
    if (pmv_inner(k, N, N, fmt, KL, KU, N, w[0].g, lda, w[1].g, ox, ix, 1.0f, 0.0f, BM(dTv), 0, 1, tr, 0, 0, unit, lo, hi)) return -1;
    cl_kernel K = k->copy;
    ARG(K, 0, N); ARG(K, 1, dTv); ARG(K, 2, z); ARG(K, 3, j); ARGB(K, 4, w[1].g); ARG(K, 5, ox); ARG(K, 6, ix);
    return start4(K, N) || finish(w, 2, cout) ? -1 : 0;
}
// op(A)·x = b (TBSV, TPSV): a single work-group, row by row (inherently sequential), FP32 work vector.
static int psv(int type, int N, int fmt, int KL, int KU, int tr, int lower, int unit, int K_, const void *A, size_t bA, int lda,
               void *x, int incx) {
    BEGIN(N, !incx);
    op_t w[2] = { { A, bA }, { x, vec_bytes(N, incx, es) } };
    const int cin[2] = { 1, 1 }, cout[2] = { 0, 1 };
    if (load_operands(w, 2, cin, cout) || ensure_buf(&dXf, &sXf, (size_t)N * RS(k))) return -1;
    cl_long ox = vec_off(N, incx), ix = incx, oa = 0;
    cl_kernel K = k->to_f;
    ARG(K, 0, N); ARGB(K, 1, w[1].g); ARG(K, 2, ox); ARG(K, 3, ix); ARG(K, 4, dXf);
    if (start1(K, N)) return -1;
    K = k->psv;
    ARG(K, 0, N); ARG(K, 1, fmt); ARG(K, 2, KL); ARG(K, 3, KU); ARGB(K, 4, w[0].g); ARG(K, 5, oa); ARG(K, 6, lda); ARG(K, 7, dXf);
    ARG(K, 8, tr); ARG(K, 9, lower); ARG(K, 10, unit); ARG(K, 11, K_);
    size_t g = 256, l = 256;
    CHECK(clEnqueueNDRangeKernel(q, K, 1, NULL, &g, &l, 0, NULL, NULL));
    K = k->from_f;
    ARG(K, 0, N); ARG(K, 1, dXf); ARGB(K, 2, w[1].g); ARG(K, 3, ox); ARG(K, 4, ix);
    return start1(K, N) || finish(w, 2, cout) ? -1 : 0;
}
// op(A) lower <=> (uplo lower) xor trans. Nonzero range of op(A) in a row: lower [i − K, i], upper [i, i + K].
int gen9blas_tbmv(int type, int uplo, int trans, int diag, int N, int K, const void *A, int lda, void *x, int incx) {
    if (K < 0 || lda < K + 1) FAIL("tbmv: K, lda");
    int es = elem_size(type), g = uplo == GEN9BLAS_UPPER, low = !g != (trans != 0);
    return pmv_x(type, N, BAND, g ? 0 : K, g ? K : 0, trans != 0, diag != 0, low ? K : 0, low ? 0 : K, A,
                 N > 0 ? band_bytes(N, lda, 0, K, es) : 0, lda, x, incx);
}
int gen9blas_tpmv(int type, int uplo, int trans, int diag, int N, const void *Ap, void *x, int incx) {
    int es = elem_size(type), g = uplo == GEN9BLAS_UPPER, low = !g != (trans != 0);
    return pmv_x(type, N, g ? PACK_U : PACK_L, 0, 0, trans != 0, diag != 0, low ? N : 0, low ? 0 : N, Ap, packed_bytes(N, es), 0, x, incx);
}
int gen9blas_tbsv(int type, int uplo, int trans, int diag, int N, int K, const void *A, int lda, void *x, int incx) {
    if (K < 0 || lda < K + 1) FAIL("tbsv: K, lda");
    int es = elem_size(type), g = uplo == GEN9BLAS_UPPER;
    return psv(type, N, BAND, g ? 0 : K, g ? K : 0, trans != 0, !g != (trans != 0), diag != 0, K, A,
               N > 0 ? band_bytes(N, lda, 0, K, es) : 0, lda, x, incx);
}
int gen9blas_tpsv(int type, int uplo, int trans, int diag, int N, const void *Ap, void *x, int incx) {
    int es = elem_size(type), g = uplo == GEN9BLAS_UPPER;
    return psv(type, N, g ? PACK_U : PACK_L, 0, 0, trans != 0, !g != (trans != 0), diag != 0, N, Ap, packed_bytes(N, es), 0, x, incx);
}
// SPR / SPR2: A = alpha·x·xᵀ + A, A = alpha·(x·yᵀ + y·xᵀ) + A (packed, uplo triangle).
static int spr2(int type, int uplo, int n, double alpha, const void *x, int incx, const void *y, int incy, void *Ap) {
    BEGIN(n, !incx || (y && !incy));
    int two = y != NULL, fmt = uplo == GEN9BLAS_UPPER ? PACK_U : PACK_L;
    op_t w[3] = { { Ap, packed_bytes(n, es) }, { x, vec_bytes(n, incx, es) }, { two ? y : x, vec_bytes(n, two ? incy : incx, es) } };
    const int cin[3] = { 1, 1, 1 }, cout[3] = { 1, 0, 0 };
    if (load_operands(w, two ? 3 : 2, cin, cout)) return -1;
    if (!two) { w[2] = w[1]; incy = incx; }
    cl_long ox = vec_off(n, incx), oy = vec_off(n, incy), oa = 0, ix = incx, iy = incy; cl_kernel K = k->spr2;
    ARG(K, 0, n); ARG(K, 1, fmt); ARGR(K, 2, alpha); ARGB(K, 3, w[1].g); ARG(K, 4, ox); ARG(K, 5, ix); ARGB(K, 6, w[2].g);
    ARG(K, 7, oy); ARG(K, 8, iy); ARGB(K, 9, w[0].g); ARG(K, 10, oa); ARG(K, 11, two);
    size_t g[2] = { ((size_t)n + 63) / 64 * 64, (size_t)n }, l[2] = { 64, 1 };
    CHECK(clEnqueueNDRangeKernel(q, K, 2, NULL, g, l, 0, NULL, NULL));
    return finish(w, 1, cout);
}
int gen9blas_spr(int type, int uplo, int n, double alpha, const void *x, int incx, void *Ap) { return spr2(type, uplo, n, alpha, x, incx, NULL, 0, Ap); }
int gen9blas_spr2(int type, int uplo, int n, double alpha, const void *x, int incx, const void *y, int incy, void *Ap) {
    return spr2(type, uplo, n, alpha, x, incx, y, incy, Ap);
}

// ---------------- skinny GEMM (small M or N): docs/skinny-gemm.md, results/skinny-gemm-2026-11-12.txt ----------------
// No packing: the weights (the larger operand) are read directly once per tile of rows of the small operand, FP32 accumulation (also
// for FP16 -- accuracy at least as good as GEN9BLAS_HS). The small operand "a" (Mc × K) and the weights "w" (K × Nc) with general strides;
// for small N the roles are swapped: Cᵀ = op(B)ᵀ · op(A)ᵀ (C is written with swapped strides).
// GEN9BLAS_SKINNY=0 disables it (old path), GEN9BLAS_SKINNY=n sets the threshold min(M, N) ≤ n for all types (measurements).
#define SKINNY_S 16   // subgroups per group in skinny_r (like SKINNY_S in level12.cl)
static cl_mem dQ; static size_t sQ;   // partial sums (weights contiguous along N)
static int skinny_threshold(int type) {
    static int forced = -2;
    if (forced == -2) { const char *e = getenv("GEN9BLAS_SKINNY"); forced = e ? atoi(e) : -1; }
    if (type == GEN9BLAS_D) return 0;
    if (forced >= 0) return forced;
    // From measurement (results/skinny-gemm-2026-11-12.txt, 8 shapes, USM and ordinary memory): at M = 16 skinny is faster almost
    // everywhere (FP16 1.0–2.0×, FP32 0.9–2.2×; weaker only for 51866×1280), at M = 32 FP16 0.3–1.0×, FP32 0.8–1.2×.
    return 16;
}
static int gemm_skinny(int type, int tA, int tB, int M, int N, int K, double alpha, const void *A, int lda,
                      const void *B, int ldb, double beta, void *C, int ldc) {
    l12_t *k = &L12[type == GEN9BLAS_S ? 1 : 0];
    const int es = k->es;
    size_t bA = (tA ? (size_t)(K - 1) * lda + M : (size_t)(M - 1) * lda + K) * es;
    size_t bB = (tB ? (size_t)(N - 1) * ldb + K : (size_t)(K - 1) * ldb + N) * es;
    size_t bC = ((size_t)(M - 1) * ldc + N) * es;
    bf_t xA, xB, xC; int zA, zB, zC;
    if (get_obj(A, bA, &uA, 1, 0, &xA, &zA) || get_obj(B, bB, &uB, 1, 0, &xB, &zB) || get_obj(C, bC, &uC, beta != 0.0, 1, &xC, &zC))
        return -1;
    last_zc = zA | zB << 1 | zC << 2;
    // a: Mc × K (strides sam, sak); w: element (k, n) = w[n·swn + k·swk]; C[m][n] = C[m·scm + n·scn]
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
    static int no_r = -1;   // GEN9BLAS_SKINNY_R=0: old skinny_k4/k8 (comparisons)
    if (no_r < 0) { const char *e = getenv("GEN9BLAS_SKINNY_R"); no_r = e && !atoi(e); }
    const int KL = es == 2 ? 4 : 2;   // k elements per lane (8 B), as in skinny_r in level12.cl
    const uintptr_t align_or = (uintptr_t)(M <= N ? A : B) | (uintptr_t)(M <= N ? B : A) | (uintptr_t)(sam * es) | (uintptr_t)(swn * es);
    // the GPU address has the same offset within a page as on the host (USM, kcl userptr; work buffer aligned)
    // M ≤ 4 with a weight row > 16 KB: old kernel (measured 2026-11-12: N = 4096, K = 11008 FP32 M = 2–4 0.7×, FP16 0.9–1.0×;
    // model gap: 8 long rows at once per group -- pages/TLB?)
    const int r_ok = Mc > 4 || (size_t)K * es <= 16384;
    if (swk == 1 && sak == 1 && Mc >= 2 && K % (8 * KL) == 0 && (align_or & 15) == 0 && r_ok && !no_r) {
        // rework of 2026-11-12 (results/skinny-gemm-rework-2026-11-12.txt): SIMD8, 8 weight rows, SKINNY_S subgroups
        const int MT = Mc <= 4 ? 4 : 8;
        cl_kernel Kk = MT == 4 ? k->skinny_r4 : k->skinny_r8;
        ARG(Kk, 0, iM); ARG(Kk, 1, iN); ARG(Kk, 2, iK); CHECK(argr(Kk, 3, alpha, 4)); CHECK(argr(Kk, 4, beta, 4));
        ARGB(Kk, 5, xa); ARG(Kk, 6, sam); ARGB(Kk, 7, xw); ARG(Kk, 8, swn); ARGB(Kk, 9, xC); ARG(Kk, 10, scm); ARG(Kk, 11, scn);
        size_t g[2] = { (size_t)(Mc + MT - 1) / MT * 8 * SKINNY_S, (size_t)(Nc + 7) / 8 }, l[2] = { 8 * SKINNY_S, 1 };
        CHECK(clEnqueueNDRangeKernel(q, Kk, 2, NULL, g, l, 0, NULL, NULL));
    } else if (swk == 1) {   // weights contiguous along K
        const int MT = Mc == 1 ? 1 : Mc <= 4 ? 4 : 8, RW = MT == 8 ? 2 : 4;   // as SKINNY_K in level12.cl
        cl_kernel Kk = MT == 1 ? k->skinny_k1 : MT == 4 ? k->skinny_k4 : k->skinny_k8;
        ARG(Kk, 0, iM); ARG(Kk, 1, iN); ARG(Kk, 2, iK); CHECK(argr(Kk, 3, alpha, es == 8 ? 8 : 4)); CHECK(argr(Kk, 4, beta, 4));
        ARGB(Kk, 5, xa); ARG(Kk, 6, sam); ARG(Kk, 7, sak); ARGB(Kk, 8, xw); ARG(Kk, 9, swn); ARGB(Kk, 10, xC); ARG(Kk, 11, scm); ARG(Kk, 12, scn);
        size_t g[2] = { (size_t)(Mc + MT - 1) / MT * 256, (size_t)(Nc + RW - 1) / RW }, l[2] = { 256, 1 };
        CHECK(clEnqueueNDRangeKernel(q, Kk, 2, NULL, g, l, 0, NULL, NULL));
    } else {          // weights contiguous along N (swn == 1): parts of K, then the sum and epilogue
        const int MT = 4, cols4 = (Nc + 3) / 4, tiles = (Mc + MT - 1) / MT;
        int parts = (8192 + cols4 - 1) / cols4; if (parts > K) parts = K; if (parts < 1) parts = 1;   // ~8 thousand work-items per M tile
        int kpp = (K + parts - 1) / parts; parts = (K + kpp - 1) / kpp;
        if (ensure_buf(&dQ, &sQ, (size_t)parts * Mc * Nc * sizeof(float))) return -1;
        cl_kernel Kn = k->skinny_n4;
        ARG(Kn, 0, iM); ARG(Kn, 1, iN); ARG(Kn, 2, iK); ARG(Kn, 3, kpp); ARGB(Kn, 4, xa); ARG(Kn, 5, sam); ARG(Kn, 6, sak);
        ARGB(Kn, 7, xw); ARG(Kn, 8, swk); ARG(Kn, 9, dQ);
        size_t g[2] = { ((size_t)cols4 + 63) / 64 * 64, (size_t)parts * tiles }, l[2] = { 64, 1 };
        CHECK(clEnqueueNDRangeKernel(q, Kn, 2, NULL, g, l, 0, NULL, NULL));
        cl_kernel K2 = k->skinny_n2;
        ARG(K2, 0, iM); ARG(K2, 1, iN); ARG(K2, 2, parts); ARG(K2, 3, dQ); CHECK(argr(K2, 4, alpha, 4)); CHECK(argr(K2, 5, beta, 4));
        ARGB(K2, 6, xC); ARG(K2, 7, scm); ARG(K2, 8, scn);
        size_t g2[2] = { ((size_t)Nc + 63) / 64 * 64, (size_t)Mc }, l2[2] = { 64, 1 };
        CHECK(clEnqueueNDRangeKernel(q, K2, 2, NULL, g2, l2, 0, NULL, NULL));
    }
    return to_host(xC, zC, bC, C);
}
