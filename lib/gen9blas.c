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
#include <limits.h>
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
static cl_kernel k_rope_m, k_setrows, k_cpy_any, k_upscale, k_dec_part, k_dec_comb;   // gen9blas_rope_multi, _set_rows_f16, _cpy_any, _upscale_f32 (NULL if the binary lacks them)
static cl_kernel k_skp[3];   // gen9blas_skinny_pb16_4 / _8 / _16
static cl_kernel k_pa3_f32, k_epi_hf, k_swiglu, k_dequant, k_dequant2, k_pa3_v, k_epi_8, k_epi_8x;
static cl_kernel k_rms, k_binop, k_rope, k_lnorm, k_gelu, k_lnorm_pack, k_gelu_erf, k_cvt_heads, k_im2col, k_relu;   // ggml per-layer elementwise ops (NULL if the binary lacks them)   // k_dequant: NULL if the binary lacks it   // ggml backend: F16 weights x F32 activations (gemm_pb_f16w, ffn_f16w)
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
    KER(ph, "pk_a3_f32", k_pa3_f32); KER(ph, "gen9blas_epilog_hf", k_epi_hf); KER(ph, "gen9blas_swiglu_pack_a3", k_swiglu);   // F16 weights x F32 activations
    { cl_int e4; k_dequant = clCreateKernel(ph, "gen9blas_dequant_b16", &e4); if (e4) k_dequant = NULL; }
    { cl_int e5; k_dequant2 = clCreateKernel(ph, "gen9blas_dequant_b16s", &e5); if (e5) k_dequant2 = NULL; }
    { cl_int e10; k_rope_m = clCreateKernel(ph, "gen9blas_rope_multi", &e10); if (e10) k_rope_m = NULL; }
    { cl_int e11; k_setrows = clCreateKernel(ph, "gen9blas_set_rows_f16", &e11); if (e11) k_setrows = NULL; }
    { cl_int e12; k_cpy_any = clCreateKernel(ph, "gen9blas_cpy_any", &e12); if (e12) k_cpy_any = NULL; }
    { cl_int e13; k_upscale = clCreateKernel(ph, "gen9blas_upscale_f32", &e13); if (e13) k_upscale = NULL; }
    { cl_int e14; k_dec_part = clCreateKernel(ph, "gen9blas_attn_dec_part", &e14); if (e14) k_dec_part = NULL; }
    { cl_int e15; k_dec_comb = clCreateKernel(ph, "gen9blas_attn_dec_comb", &e15); if (e15) k_dec_comb = NULL; }
    for (int i = 0; i < 3; i++) {   // skinny products on HS-packed weights (gen9blas_gemm_pb_skinny); NULL if the binary lacks them
        char n[40]; cl_int e9; snprintf(n, sizeof n, "gen9blas_skinny_pb16_%d", 4 << i);
        k_skp[i] = clCreateKernel(ph, n, &e9); if (e9) k_skp[i] = NULL;
    }
    { cl_int e6; k_pa3_v = clCreateKernel(ph, "pk_a3_f32v", &e6); if (e6) k_pa3_v = NULL; }
    { cl_int e7; k_epi_8 = clCreateKernel(ph, "gen9blas_epilog_hf8", &e7); if (e7) k_epi_8 = NULL; }
    { cl_int e8; k_epi_8x = clCreateKernel(ph, "gen9blas_epilog_hf8x", &e8); if (e8) k_epi_8x = NULL; }   // fused bias/GELU/residual
    { cl_int e8; k_rms = clCreateKernel(ph, "gen9blas_rms_norm", &e8); if (e8) k_rms = NULL;
      k_binop = clCreateKernel(ph, "gen9blas_binop_rows", &e8); if (e8) k_binop = NULL;
      k_rope = clCreateKernel(ph, "gen9blas_rope", &e8); if (e8) k_rope = NULL;
      k_lnorm = clCreateKernel(ph, "gen9blas_layer_norm", &e8); if (e8) k_lnorm = NULL;
      { cl_int e9; k_lnorm_pack = clCreateKernel(ph, "gen9blas_layer_norm_pack", &e9); if (e9) k_lnorm_pack = NULL; }
      k_gelu = clCreateKernel(ph, "gen9blas_gelu", &e8); if (e8) k_gelu = NULL;
      k_gelu_erf = clCreateKernel(ph, "gen9blas_gelu_erf", &e8); if (e8) k_gelu_erf = NULL;
      k_cvt_heads = clCreateKernel(ph, "gen9blas_cvt_f16_heads", &e8); if (e8) k_cvt_heads = NULL;
      k_im2col = clCreateKernel(ph, "gen9blas_im2col", &e8); if (e8) k_im2col = NULL;
      k_relu = clCreateKernel(ph, "gen9blas_relu", &e8); if (e8) k_relu = NULL; }
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
static int force_copy;   // per call: copy through the CPU instead of userptr (gen9blas_pack_b retry after a userptr failure)
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
    if (!no_userptr && !force_copy && (m = kcl_userptr(ptr, bytes, writes))) { *result = BM(m); *zc = 1; return 0; }
#else
    (void)writes;
#endif
    if (wait_gpu() || ensure_work(u, bytes)) return -1;
    if (copy_in) memcpy(u->p, ptr, bytes);
    *result = (bf_t){ NULL, u->p }; *zc = 0; return 0;
}

// Zero-copy only (USM or userptr), no wait and no copy: 0 = *result set, 1 = would need a copy.
static int get_obj_zc(const void *ptr, size_t bytes, int writes, bf_t *result) {
    if (usm_contains(ptr, bytes)) { *result = (bf_t){ NULL, (void *)ptr }; return 0; }
#ifndef GEN9BLAS_NEO
    if (no_userptr < 0) no_userptr = getenv("GEN9BLAS_NO_USERPTR") != NULL;
    cl_mem m;
    if (!no_userptr && !force_copy && (m = kcl_userptr(ptr, bytes, writes))) { *result = BM(m); return 0; }
#else
    (void)writes;
#endif
    return 1;
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

static double max_launch_ms(void);
static double est_ms(type_t *r, double M, double N, double K);
static int launch(type_t *r, cl_mem Ap, size_t oA, cl_mem Bp, size_t oB, cl_mem Cp, size_t oC,
                   int M, int N, int ldc, int K, int Kfull, int measure);
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
    double t_one = est_ms(r, Mp, Np, Kp);
    int split = t_one * count > max_launch_ms();   // too long for one launch: bands (launch()) per matrix, epilogue
    int direct = !split && alpha == 1.0 && beta == 0.0 && M == Mp && N == Np && (count == 1 || sC == (long)M * ldc);
    bf_t target = direct ? xC : BM(dCp); int ldtarget = direct ? ldc : Np;
    ARG(r->main, 0, dAp); ARG(r->main, 1, dBp); ARGB(r->main, 2, target); ARG(r->main, 3, Mp); ARG(r->main, 4, Np);
    ARG(r->main, 5, ldtarget); ARG(r->main, 6, Kp); ARG(r->main, 7, Kp);
    size_t gm = (size_t)count * (Np / r->tn) * (Mp / 16) * 16, lm = 16;
    cl_event ev = NULL;
    if (!split) CHECK(clEnqueueNDRangeKernel(q, r->main, 1, NULL, &gm, &lm, 0, NULL, &ev));
    else {
        nev = 0;
        for (int b = 0; b < count; b++)
            if (launch(r, dAp, (size_t)b * Mp * Kp * es, dBp, (size_t)b * Np * Kp * es, dCp, (size_t)b * Mp * Np * es,
                       Mp, Np, Np, Kp, Kp, 1)) return -1;
    }
    ARG(r->epi, 0, dCp); ARG(r->epi, 1, Np); ARG(r->epi, 2, Mp); ARGB(r->epi, 3, xC); ARG(r->epi, 4, ldc); ARG(r->epi, 5, lC);
    ARGS(r->epi, 6, alpha); ARGS(r->epi, 7, beta);
    size_t ge[3] = { (size_t)N, (size_t)M, (size_t)count };
    if (direct) CHECK(clEnqueueMarkerWithWaitList(q, 0, NULL, &evp[2]));
    else CHECK(clEnqueueNDRangeKernel(q, r->epi, 3, NULL, ge, NULL, 0, NULL, &evp[2]));
    if (to_host(xC, zC, bC, C)) return -1;
    if (split) { double tz = event_times(); last_ms = tz >= 0 ? tz : (now_s() - t_start) * 1e3; }
    else if (profile) {
        cl_ulong t0, t1;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof t0, &t0, NULL);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof t1, &t1, NULL);
        last_ms = (t1 - t0) * 1e-6;
    } else last_ms = (now_s() - t_start) * 1e3;
    if (ev) clReleaseEvent(ev);
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

// Longest planned GEMM kernel launch. kcl does not set up mid-thread preemption, so a launch that runs longer than the
// engine's preempt_timeout_ms (640 ms by default) while another context (e.g. the desktop) waits for the GPU gets the
// engine reset and its results lost. Long launches are split into row bands, each estimated below this time from a
// conservative throughput per type. GEN9BLAS_MAX_LAUNCH_MS overrides it (default 100).
static double max_launch_ms(void) {
    static double v = 0;
    if (v <= 0) { const char *e = getenv("GEN9BLAS_MAX_LAUNCH_MS"); v = e && atof(e) > 0 ? atof(e) : 100; }
    return v;
}
static double est_ms(type_t *r, double M, double N, double K) {   // conservative GFLOPS of the main kernels
    double gf = r == &types[GEN9BLAS_D] ? 45 : r == &types[GEN9BLAS_S] ? 300 : r == &types[GEN9BLAS_HS] ? 500 : 600;
    return 2 * M * N * K / (gf * 1e6);
}

// GEMM kernel on a chunk: sub-buffers with offsets (bytes; aligned to 128 B). measure: event goes to evs[].
static int launch1(type_t *r, cl_mem Ap, size_t oA, cl_mem Bp, size_t oB, cl_mem Cp, size_t oC,
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
// As launch1, split into bands of rows (multiples of 16; packed A: row i0 at i0·Kfull, C: row i0 at i0·ldc) so that each
// launch stays below max_launch_ms().
static int launch(type_t *r, cl_mem Ap, size_t oA, cl_mem Bp, size_t oB, cl_mem Cp, size_t oC,
                   int M, int N, int ldc, int K, int Kfull, int measure) {
    int parts = (int)(est_ms(r, M, N, K) / max_launch_ms()) + 1;
    int mb = ((M + parts - 1) / parts + 15) / 16 * 16;
    if (mb >= M) return launch1(r, Ap, oA, Bp, oB, Cp, oC, M, N, ldc, K, Kfull, measure);
    for (int i0 = 0; i0 < M; i0 += mb) {
        int b = M - i0 < mb ? M - i0 : mb;
        if (launch1(r, Ap, oA + (size_t)i0 * Kfull * r->es, Bp, oB, Cp, oC + (size_t)i0 * ldc * r->es, b, N, ldc, K, Kfull, measure))
            return -1;
    }
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

// ---------------- packed B kept between calls (constant weights; ggml backend, docs/ggml-backend-design.md) ----------------

struct gen9blas_packed { int type, K, N, Np, Kp; cl_mem buf; size_t bufsize; };

// Spare packed buffers: gen9blas_packed_free keeps up to SPARES freed buffers (GEN9BLAS_SPARE_MB in all, default 512)
// and the next packing takes the smallest one that fits. A new buffer costs about as much as the dequantization
// itself (i915 zeroes its pages: ~3.7 GB/s with userptr, 2026-11-13); this matters for weights packed on every use.
#define SPARES 4
static struct { cl_mem m; size_t sz; } spare[SPARES];
static size_t spare_cap(void) {
    static size_t v = (size_t)-1;
    if (v == (size_t)-1) { const char *e = getenv("GEN9BLAS_SPARE_MB"); v = (size_t)(e ? atof(e) : 512) << 20; }
    return v;
}
static cl_mem packed_buf(size_t need, size_t *got, cl_int *err) {
    int best = -1;
    for (int i = 0; i < SPARES; i++)
        if (spare[i].m && spare[i].sz >= need && (best < 0 || spare[i].sz < spare[best].sz)) best = i;
    if (best >= 0) { cl_mem m = spare[best].m; *got = spare[best].sz; spare[best].m = NULL; *err = CL_SUCCESS; return m; }
    *got = need;
    return clCreateBuffer(ctx, CL_MEM_READ_WRITE, need, NULL, err);
}
static void packed_buf_release(cl_mem m, size_t sz) {
    size_t total = sz; int freei = -1, smallest = -1;
    for (int i = 0; i < SPARES; i++) {
        if (!spare[i].m) { if (freei < 0) freei = i; continue; }
        total += spare[i].sz;
        if (smallest < 0 || spare[i].sz < spare[smallest].sz) smallest = i;
    }
    if (freei < 0 && smallest >= 0 && spare[smallest].sz < sz) {   // full: replace the smallest by a larger one
        total -= spare[smallest].sz; clReleaseMemObject(spare[smallest].m); spare[smallest].m = NULL; freei = smallest;
    }
    if (freei < 0 || total > spare_cap()) { clReleaseMemObject(m); return; }
    spare[freei].m = m; spare[freei].sz = sz;
}

static size_t pack_band_bytes(void) {
    static size_t v = 0;
    if (!v) { const char *e = getenv("GEN9BLAS_PACK_BAND_MB"); v = (size_t)(e && atof(e) > 0 ? atof(e) : 128) << 20; }
    return v;
}

// Packs op(B) into P->buf in bands of whole 32-column groups (n), each launch at most pack_band_bytes() of output.
static int pack_b_bands(type_t *r, int tB, const void *B, int ldb, gen9blas_packed *P) {
    size_t bB = (tB ? (size_t)(P->N - 1) * ldb + P->K : (size_t)(P->K - 1) * ldb + P->N) * r->es;
    bf_t xB; int z;
    if (get_obj(B, bB, &uB, 1, 0, &xB, &z)) return -1;
    size_t g[3], o[3] = { 0, 0, 0 };
    cl_kernel k = pack_kernel(r, 1, tB, P->Np, P->Kp, 1, g);
    int d = tB ? 1 : 0, per = (!tB && k == r->pb_w2) ? 2 : 1;   // grid dimension of n; work-items per 2 columns (w2)
    cl_long zero = 0;
    ARGB(k, 0, xB); ARG(k, 1, P->buf); ARG(k, 2, P->K); ARG(k, 3, P->N); ARG(k, 4, ldb); ARG(k, 5, P->Kp); ARG(k, 6, zero); ARG(k, 7, P->Np);
    size_t nb = pack_band_bytes() / ((size_t)P->Kp * r->es) / 32 * 32;
    if (nb < 32) nb = 32;
    for (size_t n0 = 0; n0 < (size_t)P->Np; n0 += nb) {
        size_t n = (size_t)P->Np - n0 < nb ? (size_t)P->Np - n0 : nb;
        o[d] = n0 / per; g[d] = n / per;
        CHECK(clEnqueueNDRangeKernel(q, k, 3, o, g, NULL, 0, NULL, NULL));
        CHECK(clFlush(q));   // one band per submission (kcl: each flush is one batch buffer)
    }
    if (wait_gpu()) return -1;
#ifndef GEN9BLAS_NEO
    if (z) kcl_userptr_release(B, bB);   // the original is not needed on the GPU any more: unpin it
#endif
    return 0;
}

gen9blas_packed *gen9blas_pack_b(int type, int tB, int K, int N, const void *B, int ldb) {
    NEW_OPERATION();
    if (type < 0 || type > 3 || !types[type].ready) { snprintf(errbuf, sizeof errbuf, "type %d unavailable", type); return NULL; }
    if (K <= 0 || N <= 0) { snprintf(errbuf, sizeof errbuf, "pack_b: K = %d, N = %d", K, N); return NULL; }
    type_t *r = &types[type];
    gen9blas_packed *P = calloc(1, sizeof *P);
    if (!P) return NULL;
    P->type = type; P->K = K; P->N = N;
    P->Np = (N + r->tn - 1) / r->tn * r->tn; P->Kp = (K + r->km - 1) / r->km * r->km;
    cl_int err;
    P->buf = packed_buf((size_t)P->Np * P->Kp * r->es + 4096, &P->bufsize, &err);
    if (err != CL_SUCCESS) { snprintf(errbuf, sizeof errbuf, "pack_b: clCreateBuffer = %d", err); free(P); return NULL; }
    if (pack_b_bands(r, tB, B, ldb, P)) {   // userptr failures show up late (argument / submission): retry with a copy
        force_copy = 1; NEW_OPERATION();
        int rc = pack_b_bands(r, tB, B, ldb, P);
        force_copy = 0;
        if (rc) { packed_buf_release(P->buf, P->bufsize); free(P); return NULL; }
    }
    return P;
}

// Quantized weights (ggml block layouts) dequantized on the GPU straight into a GEN9BLAS_HS packed B (see gen9blas.h).
static int pack_b_q_bands(int qt, size_t rb, const void *W, gen9blas_packed *P, int *nf) {
    size_t bW = (size_t)P->N * rb;
    bf_t xW; int z;
    if (get_obj(W, bW, &uB, 1, 0, &xW, &z)) return -1;
    bf_t xnf = { NULL, nf };
    cl_ulong rbl = rb;
    // v2 (SLM staging, contiguous 32 B writes; needs K % 256 == 0); GEN9BLAS_DEQ=v1 keeps the first kernel for comparisons
    static int v1 = -1;
    if (v1 < 0) { const char *e = getenv("GEN9BLAS_DEQ"); v1 = e && !strcmp(e, "v1"); }
    // v2 loads 16 B for Q8_0 / Q4_K: every row start W + n * rb must be 16 B aligned (always so for ggml tensors: 32 B
    // aligned data, rows of 272 / 144 B multiples); anything else goes to v1. Q6_K loads 2 B.
    const int aligned = qt == 2 ? !(((uintptr_t)W | rb) & 1) : !(((uintptr_t)W | rb) & 15);
    if (k_dequant2 && !v1 && P->K % 256 == 0 && aligned) {
        cl_kernel k2 = k_dequant2;
        ARGB(k2, 0, xW); ARG(k2, 1, rbl); ARG(k2, 2, qt); ARG(k2, 3, P->N); ARG(k2, 4, P->Kp); ARGB(k2, 7, xnf);
        ARG(k2, 6, P->buf);
        size_t nb = pack_band_bytes() / ((size_t)P->Kp * 2) / 16 * 16;
        if (nb < 16) nb = 16;
        for (size_t n0 = 0; n0 < (size_t)P->Np; n0 += nb) {
            const size_t rows = (size_t)P->Np - n0 < nb ? (size_t)P->Np - n0 : nb;
            int n0i = (int)n0;
            ARG(k2, 5, n0i);
            size_t g[2] = { rows / 16 * 128, (size_t)P->K / 256 }, l[2] = { 128, 1 };
            CHECK(clEnqueueNDRangeKernel(q, k2, 2, NULL, g, l, 0, NULL, NULL));
            CHECK(clFlush(q));
        }
        goto done;
    }
    ARGB(k_dequant, 0, xW); ARG(k_dequant, 1, rbl); ARG(k_dequant, 2, qt); ARG(k_dequant, 3, P->N); ARG(k_dequant, 4, P->Kp);
    ARG(k_dequant, 5, P->buf); ARGB(k_dequant, 6, xnf);
    size_t nb = pack_band_bytes() / ((size_t)P->Kp * 2) / 16 * 16;
    if (nb < 16) nb = 16;
    for (size_t n0 = 0; n0 < (size_t)P->Np; n0 += nb) {
        size_t o[2] = { n0, 0 }, g[2] = { (size_t)P->Np - n0 < nb ? (size_t)P->Np - n0 : nb, (size_t)P->K / (qt == 0 ? 32 : 256) };
        CHECK(clEnqueueNDRangeKernel(q, k_dequant, 2, o, g, NULL, 0, NULL, NULL));
        CHECK(clFlush(q));
    }
done:
    if (wait_gpu()) return -1;
#ifndef GEN9BLAS_NEO
    if (z) kcl_userptr_release(W, bW);
#endif
    return 0;
}

gen9blas_packed *gen9blas_pack_b_q(int qtype, int K, int N, const void *W, size_t row_bytes, int *out_of_range) {
    static int *nf;
    NEW_OPERATION();
    *out_of_range = 0;
    if (!k_dequant || !types[GEN9BLAS_HS].ready) { snprintf(errbuf, sizeof errbuf, "pack_b_q: dequant kernel unavailable"); return NULL; }
    const int qb = qtype == GEN9BLAS_Q8_0 ? 32 : 256;
    const size_t bb = qtype == GEN9BLAS_Q8_0 ? 34 : qtype == GEN9BLAS_Q4_K ? 144 : qtype == GEN9BLAS_Q6_K ? 210 : 0;
    if (!bb || K <= 0 || N <= 0 || K % qb || row_bytes < (size_t)K / qb * bb) {
        snprintf(errbuf, sizeof errbuf, "pack_b_q: qtype %d, K = %d, N = %d, row_bytes = %zu", qtype, K, N, row_bytes); return NULL;
    }
    if (!nf && !(nf = gen9blas_alloc(64))) return NULL;
    type_t *r = &types[GEN9BLAS_HS];
    gen9blas_packed *P = calloc(1, sizeof *P);
    if (!P) return NULL;
    P->type = GEN9BLAS_HS; P->K = K; P->N = N;
    P->Np = (N + r->tn - 1) / r->tn * r->tn; P->Kp = (K + r->km - 1) / r->km * r->km;
    if (P->Kp != K) { snprintf(errbuf, sizeof errbuf, "pack_b_q: K = %d not a multiple of %d", K, r->km); free(P); return NULL; }
    cl_int err;
    P->buf = packed_buf((size_t)P->Np * P->Kp * 2 + 4096, &P->bufsize, &err);
    if (err != CL_SUCCESS) { snprintf(errbuf, sizeof errbuf, "pack_b_q: clCreateBuffer = %d", err); free(P); return NULL; }
    *nf = 0;
    if (pack_b_q_bands(qtype, row_bytes, W, P, nf)) {   // userptr failures show up late: retry with a copy
        force_copy = 1; NEW_OPERATION(); *nf = 0;
        int rc = pack_b_q_bands(qtype, row_bytes, W, P, nf);
        force_copy = 0;
        if (rc) { packed_buf_release(P->buf, P->bufsize); free(P); return NULL; }
    }
    *out_of_range = *nf != 0;
    return P;
}

void gen9blas_packed_free(gen9blas_packed *P) {
    if (!P) return;
    wait_gpu();
    packed_buf_release(P->buf, P->bufsize); free(P);
}

int gen9blas_gemm_pb(int type, int tA, int M, double alpha, const void *A, int lda, const gen9blas_packed *P,
                     double beta, void *C, int ldc) {
    NEW_OPERATION();
    if (!P || P->type != type || !types[type].ready) FAIL("gemm_pb: packed B of another type or type %d unavailable", type);
    if (M <= 0) return 0;
    type_t *r = &types[type];
    int es = r->es, N = P->N, K = P->K, Np = P->Np, Kp = P->Kp, Mp = (M + 15) / 16 * 16;
    size_t bA = (tA ? (size_t)(K - 1) * lda + M : (size_t)(M - 1) * lda + K) * es, bC = ((size_t)(M - 1) * ldc + N) * es;
    bf_t xA, xC; int zA, zC;
    if (get_obj(A, bA, &uA, 1, 0, &xA, &zA) || get_obj(C, bC, &uC, beta != 0.0, 1, &xC, &zC)) return -1;
    last_zc = zA | 2 | zC << 2;
    if (ensure_buf(&dAp, &sAp, (size_t)Mp * Kp * es + 4096) || ensure_buf(&dCp, &sCp, (size_t)Mp * Np * es)) return -1;
    size_t ga[3]; cl_long zero = 0;
    cl_kernel ka = pack_kernel(r, 0, tA, Mp, Kp, 1, ga);
    ARGB(ka, 0, xA); ARG(ka, 1, dAp); ARG(ka, 2, M); ARG(ka, 3, K); ARG(ka, 4, lda); ARG(ka, 5, Kp); ARG(ka, 6, zero); ARG(ka, 7, Mp);
    CHECK(clEnqueueNDRangeKernel(q, ka, 3, NULL, ga, NULL, 0, NULL, NULL));
    int split = est_ms(r, Mp, Np, Kp) > max_launch_ms();
    int direct = !split && alpha == 1.0 && beta == 0.0 && M == Mp && N == Np;
    if (!split) {
        bf_t target = direct ? xC : BM(dCp); int ldt = direct ? ldc : Np;
        ARG(r->main, 0, dAp); ARG(r->main, 1, P->buf); ARGB(r->main, 2, target); ARG(r->main, 3, Mp); ARG(r->main, 4, Np);
        ARG(r->main, 5, ldt); ARG(r->main, 6, Kp); ARG(r->main, 7, Kp);
        size_t gm = (size_t)(Np / r->tn) * (Mp / 16) * 16, lm = 16;
        CHECK(clEnqueueNDRangeKernel(q, r->main, 1, NULL, &gm, &lm, 0, NULL, NULL));
    } else if (launch(r, dAp, 0, P->buf, 0, dCp, 0, Mp, Np, Np, Kp, Kp, 0)) return -1;
    if (!direct) {   // epilogue in row bands of at most pack_band_bytes() of C each
        cl_long lC = 0;
        ARG(r->epi, 0, dCp); ARG(r->epi, 1, Np); ARG(r->epi, 2, Mp); ARGB(r->epi, 3, xC); ARG(r->epi, 4, ldc); ARG(r->epi, 5, lC);
        ARGS(r->epi, 6, alpha); ARGS(r->epi, 7, beta);
        size_t mb = pack_band_bytes() / ((size_t)N * es * 2);
        if (mb < 1) mb = 1;
        for (size_t i0 = 0; i0 < (size_t)M; i0 += mb) {
            size_t o[3] = { 0, i0, 0 }, ge[3] = { (size_t)N, (size_t)M - i0 < mb ? (size_t)M - i0 : mb, 1 };
            CHECK(clEnqueueNDRangeKernel(q, r->epi, 3, o, ge, NULL, 0, NULL, NULL));
        }
    }
    return to_host(xC, zC, bC, C);
}

// C (F32, M x N) = A (F32, M x K) . Bp, Bp packed with type GEN9BLAS_HS (FP16 weights). A is rounded to FP16 while packing,
// the accurate FP16 GEMM writes FP16 into a work buffer, the epilogue converts to F32 and sets *nonfinite when any result
// is inf / NaN (then the caller recomputes in F32). Epilogue in row bands as in gemm_pb.
// F32 activations -> A3 packed FP16 (16 x 16 tiles) and FP16 GEMM result -> F32 C: the 16 B-per-lane kernels
// (pk_a3_f32v, gen9blas_epilog_hf8, same values bit for bit) unless GEN9BLAS_PKV=0 or the binary lacks them.
static int pkv(void) { static int v = -1; if (v < 0) { const char *e = getenv("GEN9BLAS_PKV"); v = !(e && !strcmp(e, "0")); } return v; }
static int pack_a3_f32(bf_t xA, cl_mem dAp, int M, int K, int lda, int Kp, int Mp) {
    cl_long zero = 0;
    cl_kernel k = pkv() && k_pa3_v ? k_pa3_v : k_pa3_f32;
    ARGB(k, 0, xA); ARG(k, 1, dAp); ARG(k, 2, M); ARG(k, 3, K); ARG(k, 4, lda); ARG(k, 5, Kp); ARG(k, 6, zero); ARG(k, 7, Mp);
    if (k == k_pa3_v) {
        size_t g[3] = { (size_t)Kp, (size_t)Mp / 16, 1 }, l[3] = { 16, 1, 1 };
        CHECK(clEnqueueNDRangeKernel(q, k, 3, NULL, g, l, 0, NULL, NULL));
    } else {
        size_t g[3] = { (size_t)Kp, (size_t)Mp, 1 };
        CHECK(clEnqueueNDRangeKernel(q, k, 3, NULL, g, NULL, 0, NULL, NULL));
    }
    return 0;
}
static int epilog_hf(cl_mem dCp, int ldp, int Mp, bf_t xC, int ldc, int N, int M, bf_t xnf) {
    float one = 1.0f, zf = 0.0f; cl_long lC = 0;
    const int v8 = pkv() && k_epi_8;
    cl_kernel k = v8 ? k_epi_8 : k_epi_hf;
    ARG(k, 0, dCp); ARG(k, 1, ldp); ARG(k, 2, Mp); ARGB(k, 3, xC); ARG(k, 4, ldc); ARG(k, 5, lC); ARG(k, 6, one); ARG(k, 7, zf);
    if (v8) { ARG(k, 8, N); ARGB(k, 9, xnf); } else { ARGB(k, 8, xnf); }
    size_t mb = pack_band_bytes() / ((size_t)N * 6);
    if (mb < 1) mb = 1;
    for (size_t i0 = 0; i0 < (size_t)M; i0 += mb) {
        size_t o[3] = { 0, i0, 0 }, ge[3] = { v8 ? (size_t)(N + 3) / 4 : (size_t)N, (size_t)M - i0 < mb ? (size_t)M - i0 : mb, 1 };
        CHECK(clEnqueueNDRangeKernel(q, k, 3, o, ge, NULL, 0, NULL, NULL));
    }
    return 0;
}

// Epilogue with bias / GELU / residual fused in (gen9blas_epilog_hf8x), banded like epilog_hf.
static int epilog_hfx(cl_mem dCp, int ldp, int Mp, bf_t xC, int ldc, int N, int M, bf_t xnf, bf_t xb, bf_t xR, int ldr, int flags) {
    cl_kernel k = k_epi_8x;
    ARG(k, 0, dCp); ARG(k, 1, ldp); ARG(k, 2, Mp); ARGB(k, 3, xC); ARG(k, 4, ldc); ARG(k, 5, N); ARGB(k, 6, xnf);
    ARGB(k, 7, xb); ARGB(k, 8, xR); ARG(k, 9, ldr); ARG(k, 10, flags);
    size_t mb = pack_band_bytes() / ((size_t)N * 6);
    if (mb < 1) mb = 1;
    for (size_t i0 = 0; i0 < (size_t)M; i0 += mb) {
        size_t o[3] = { 0, i0, 0 }, ge[3] = { (size_t)(N + 3) / 4, (size_t)M - i0 < mb ? (size_t)M - i0 : mb, 1 };
        CHECK(clEnqueueNDRangeKernel(q, k, 3, o, ge, NULL, 0, NULL, NULL));
    }
    return 0;
}
typedef struct { const float *bias; int gelu; const float *R; int ldr; } epi_ext_t;
// No-wait mode (gen9blas_nowait, gen9blas.h): products that would wait for their FP16-overflow flag instead take a flag
// slot from one USM array and return without waiting; gen9blas_nowait_check reads and clears the slots (GPU idle).
static int nw_mode, *nw_flags, nw_cap, nw_used;
static int *nw_take(void) {
    if (!nw_mode) return NULL;
    if (!nw_flags) { nw_cap = 4096; if (!(nw_flags = gen9blas_alloc((size_t)nw_cap * 4))) { nw_cap = 0; return NULL; } memset(nw_flags, 0, (size_t)nw_cap * 4); }
    return nw_used < nw_cap ? nw_flags + nw_used++ : NULL;
}
void gen9blas_nowait(int on) { nw_mode = on; }
int gen9blas_nowait_check(void) {
    int n = 0;
    for (int i = 0; i < nw_used; i++) n += nw_flags[i] != 0;
    if (nw_used) memset(nw_flags, 0, (size_t)nw_used * 4);
    nw_used = 0;
    return n;
}
static int gemm_pb_f16w_any(int ahalf, int M, const void *A, int lda, const gen9blas_packed *P, float *C, int ldc, int *nonfinite,
                            const epi_ext_t *ext, cl_mem preA, int *nw);
// Packed GEMM input made once and reused (gen9blas_layer_norm_pack -> gen9blas_gemm_pa_f16w_ex).
struct gen9blas_packed_a { cl_mem buf; size_t cap; int M, K, Kp; };
gen9blas_packed_a *gen9blas_packed_a_new(void) { return calloc(1, sizeof(gen9blas_packed_a)); }
void gen9blas_packed_a_free(gen9blas_packed_a *pa) { if (!pa) return; if (pa->buf) clReleaseMemObject(pa->buf); free(pa); }
int gen9blas_norm_pack_supported(void) { return k_lnorm_pack != NULL && k_epi_8x != NULL; }
int gen9blas_gemm_pa_f16w_ex(const gen9blas_packed_a *pa, const gen9blas_packed *P, float *C, int ldc,
                             const float *bias, int gelu, const float *R, int ldr, int *nonfinite) {
    if (!pa || !pa->buf || !P || pa->K != P->K || pa->Kp != P->Kp) FAIL("gemm_pa_f16w_ex: packed A does not match B (K %d/%d, Kp %d/%d)",
                                                                      pa ? pa->K : -1, P ? P->K : -1, pa ? pa->Kp : -1, P ? P->Kp : -1);
    const epi_ext_t e = { bias, gelu, R, ldr };
    const int any = bias || gelu || R;
    if (any && !k_epi_8x) FAIL("gemm_pa_f16w_ex: the binary lacks gen9blas_epilog_hf8x");
    return gemm_pb_f16w_any(0, pa->M, NULL, 0, P, C, ldc, nonfinite, any ? &e : NULL, pa->buf, NULL);
}

int gen9blas_epilog_ext_supported(void) { return k_epi_8x != NULL; }
int gen9blas_gemm_pb_f16w_ex(int M, const float *A, int lda, const gen9blas_packed *P, float *C, int ldc,
                             const float *bias, int gelu, const float *R, int ldr, int *nonfinite) {
    if (!k_epi_8x) FAIL("gemm_pb_f16w_ex: the binary lacks gen9blas_epilog_hf8x");
    const epi_ext_t e = { bias, gelu, R, ldr };
    return gemm_pb_f16w_any(0, M, A, lda, P, C, ldc, nonfinite, &e, NULL, NULL);
}
int gen9blas_gemm_pb_f16w(int M, const float *A, int lda, const gen9blas_packed *P, float *C, int ldc, int *nonfinite) {
    return gemm_pb_f16w_any(0, M, A, lda, P, C, ldc, nonfinite, NULL, NULL, NULL);
}
// The same with FP16 activations (ggml conv_1d: im2col result as src0, F16 kernel as src1): packed by the HS pa kernel.
int gen9blas_gemm_pb_f16w_h(int M, const uint16_t *A, int lda, const gen9blas_packed *P, float *C, int ldc, int *nonfinite) {
    return gemm_pb_f16w_any(1, M, A, lda, P, C, ldc, nonfinite, NULL, NULL, NULL);
}

// Small batches (M <= 16) on HS-packed weights without packing A: gen9blas_skinny_pb16_<MT> (bench/hgemm.cl), F32 products
// and sums, F32 result straight into C. Returns 1 (nothing done) when the kernel is missing or M is out of range.
int gen9blas_skinny_pb_supported(void) { return k_skp[0] && k_skp[1] && k_skp[2]; }
int gen9blas_gemm_pb_skinny(int M, const float *A, int lda, const gen9blas_packed *P, float *C, int ldc, int *nonfinite) {
    static int *nf;
    NEW_OPERATION();
    if (!P || P->type != GEN9BLAS_HS) FAIL("gemm_pb_skinny: needs B packed as GEN9BLAS_HS");
    if (M < 1 || M > 16 || !gen9blas_skinny_pb_supported()) return 1;
    if (!nf && !(nf = gen9blas_alloc(64))) return -1;
    *nonfinite = 0; *nf = 0;
    const int N = P->N, K = P->K, Kp = P->Kp, Np = P->Np;
    const size_t bA = ((size_t)(M - 1) * lda + K) * 4, bC = ((size_t)(M - 1) * ldc + N) * 4;
    bf_t xA, xC; int zA, zC;
    if (get_obj(A, bA, &uA, 1, 0, &xA, &zA) || get_obj(C, bC, &uC, 0, 1, &xC, &zC)) return -1;
    cl_kernel k = k_skp[M <= 4 ? 0 : M <= 8 ? 1 : 2];
    bf_t xnf = { NULL, nf };
    ARGB(k, 0, xA); ARG(k, 1, lda); ARG(k, 2, M); ARG(k, 3, K); ARG(k, 4, P->buf); ARG(k, 5, Kp);
    ARGB(k, 6, xC); ARG(k, 7, ldc); ARG(k, 8, N); ARGB(k, 9, xnf);
    size_t g = (size_t)(Np / 16) * 256, l = 256;
    CHECK(clEnqueueNDRangeKernel(q, k, 1, NULL, &g, &l, 0, NULL, NULL));
    if (to_host(xC, zC, bC, C)) return -1;
    *nonfinite = *nf != 0;
    return 0;
}
static int gemm_pb_f16w_any(int ahalf, int M, const void *A, int lda, const gen9blas_packed *P, float *C, int ldc, int *nonfinite,
                            const epi_ext_t *ext, cl_mem preA, int *nw) {
    static int *nf;   // USM flag word (coherent with the GPU through the LLC)
    NEW_OPERATION();
    if (!P || P->type != GEN9BLAS_HS || !types[GEN9BLAS_HS].ready || !k_pa3_f32) FAIL("gemm_pb_f16w: needs B packed as GEN9BLAS_HS");
    if (!nf && !(nf = gen9blas_alloc(64))) return -1;
    *nonfinite = 0;
    if (M <= 0) return 0;
    type_t *r = &types[GEN9BLAS_HS];
    int N = P->N, K = P->K, Np = P->Np, Kp = P->Kp, Mp = (M + 15) / 16 * 16;
    size_t bA = preA ? 0 : ((size_t)(M - 1) * lda + K) * (ahalf ? 2 : 4), bC = ((size_t)(M - 1) * ldc + N) * 4;
    bf_t xA = BM(NULL), xC; int zA = 1, zC;
    if ((!preA && get_obj(A, bA, &uA, 1, 0, &xA, &zA)) || get_obj(C, bC, &uC, 0, 1, &xC, &zC)) return -1;
    if (!nw && zC) nw = nw_take();   // no-wait mode: only when the result lands in place (no copy-out after a wait)
    if (ensure_buf(&dAp, &sAp, (size_t)Mp * Kp * 2 + 4096) || ensure_buf(&dCp, &sCp, (size_t)Mp * Np * 2)) return -1;
    if (!nw) *nf = 0;
    cl_mem Ap = preA ? preA : dAp;   // preA: already packed (gen9blas_layer_norm_pack)
    if (preA) {
    } else if (ahalf) {   // FP16 rows -> A3 tiles: the HS type's own packing kernel (as gemm_pb)
        size_t ga[3]; cl_long zero = 0;
        cl_kernel ka = pack_kernel(r, 0, 0, Mp, Kp, 1, ga);
        ARGB(ka, 0, xA); ARG(ka, 1, dAp); ARG(ka, 2, M); ARG(ka, 3, K); ARG(ka, 4, lda); ARG(ka, 5, Kp); ARG(ka, 6, zero); ARG(ka, 7, Mp);
        CHECK(clEnqueueNDRangeKernel(q, ka, 3, NULL, ga, NULL, 0, NULL, NULL));
    } else if (pack_a3_f32(xA, dAp, M, K, lda, Kp, Mp)) return -1;
    if (est_ms(r, Mp, Np, Kp) <= max_launch_ms()) {
        ARG(r->main, 0, Ap); ARG(r->main, 1, P->buf); ARG(r->main, 2, dCp); ARG(r->main, 3, Mp); ARG(r->main, 4, Np);
        ARG(r->main, 5, Np); ARG(r->main, 6, Kp); ARG(r->main, 7, Kp);
        size_t gm = (size_t)(Np / r->tn) * (Mp / 16) * 16, lm = 16;
        CHECK(clEnqueueNDRangeKernel(q, r->main, 1, NULL, &gm, &lm, 0, NULL, NULL));
    } else if (launch(r, Ap, 0, P->buf, 0, dCp, 0, Mp, Np, Np, Kp, Kp, 0)) return -1;
    bf_t xnf = { NULL, nw ? nw : nf };
    if (ext) {   // bias (row vector, N floats) / GELU / residual R (M x N, row stride ldr; may be C itself)
        static work_t uBias, uR;
        static float dummy[4];
        bf_t xb, xR; int zb, zR;
        const int flags = (ext->bias ? 1 : 0) | (ext->gelu ? 2 : 0) | (ext->R ? 4 : 0);
        if (get_obj(ext->bias ? ext->bias : dummy, ext->bias ? (size_t)N * 4 : sizeof dummy, &uBias, 1, 0, &xb, &zb)) return -1;
        if (ext->R && ext->R == C && ext->ldr == ldc) {   // in place: the same object as C
            xR = xC;
            if (!zC) memcpy(xC.p, C, bC);                  // work-buffer path: C's current values are the residual
        }
        else if (get_obj(ext->R ? ext->R : dummy, ext->R ? ((size_t)(M - 1) * ext->ldr + N) * 4 : sizeof dummy, &uR, 1, 0, &xR, &zR)) return -1;
        if (epilog_hfx(dCp, Np, Mp, xC, ldc, N, M, xnf, xb, xR, ext->R ? ext->ldr : 0, flags)) return -1;
    } else if (epilog_hf(dCp, Np, Mp, xC, ldc, N, M, xnf)) return -1;
    if (nw) return 0;   // queued: the caller waits later (gen9blas_flush) and reads its flag slot
    if (to_host(xC, zC, bC, C)) return -1;
    *nonfinite = *nf != 0;
    return 0;
}



// GEMM of a packed A (device) with a packed B into a device buffer C (row stride Np), banded when long.
static int gemm_dev(type_t *r, cl_mem Ap, cl_mem Bp, cl_mem C, int Mp, int Np, int Kp) {
    if (est_ms(r, Mp, Np, Kp) > max_launch_ms()) return launch(r, Ap, 0, Bp, 0, C, 0, Mp, Np, Np, Kp, Kp, 0);
    ARG(r->main, 0, Ap); ARG(r->main, 1, Bp); ARG(r->main, 2, C); ARG(r->main, 3, Mp); ARG(r->main, 4, Np);
    ARG(r->main, 5, Np); ARG(r->main, 6, Kp); ARG(r->main, 7, Kp);
    size_t gm = (size_t)(Np / r->tn) * (Mp / 16) * 16, lm = 16;
    CHECK(clEnqueueNDRangeKernel(q, r->main, 1, NULL, &gm, &lm, 0, NULL, NULL));
    return 0;
}

// n (1..8) products of one F32 input with HS-packed weights (ggml: Q/K/V MUL_MATs on the same normed input): A packed to
// FP16 once, then GEMM + epilogue per weight, one wait. nonfinite[i]: output i had inf / NaN (caller recomputes it in F32).
// Returns 1 (nothing done) when an output is not zero-copy (the copy path has one work buffer for C).
int gen9blas_gemm_multi_f16w(int M, const float *A, int lda, int n, const gen9blas_packed *const *P, float *const *C,
                             const int *ldc, int *nonfinite) {
    static int *nf;   // one USM flag word per output
    NEW_OPERATION();
    if (n < 1 || n > 8 || !types[GEN9BLAS_HS].ready || !k_pa3_f32) FAIL("gemm_multi_f16w: bad arguments");
    if (!nf && !(nf = gen9blas_alloc(64))) return -1;
    const int K = P[0]->K, Kp = P[0]->Kp, Mp = (M + 15) / 16 * 16;
    int Npmax = 0;
    for (int i = 0; i < n; i++) {
        if (!P[i] || P[i]->type != GEN9BLAS_HS || P[i]->K != K || P[i]->Kp != Kp) FAIL("gemm_multi_f16w: B %d not HS-packed with K = %d", i, K);
        if (P[i]->Np > Npmax) Npmax = P[i]->Np;
        nonfinite[i] = 0;
    }
    if (M <= 0) return 0;
    type_t *r = &types[GEN9BLAS_HS];
    bf_t xA, xC[8]; int zA, zC[8];
    if (get_obj(A, ((size_t)(M - 1) * lda + K) * 4, &uA, 1, 0, &xA, &zA)) return -1;
    for (int i = 0; i < n; i++) {
        if (get_obj(C[i], ((size_t)(M - 1) * ldc[i] + P[i]->N) * 4, &uC, 0, 1, &xC[i], &zC[i])) return -1;
        if (!zC[i]) return 1;
    }
    if (ensure_buf(&dAp, &sAp, (size_t)Mp * Kp * 2 + 4096) || ensure_buf(&dCp, &sCp, (size_t)Mp * Npmax * 2)) return -1;
    int *nws[8] = { 0 };
    for (int i = 0; i < n; i++) { nws[i] = nw_take(); if (!nws[i]) nf[i] = 0; }   // no-wait mode: own flag slots (outputs are zero-copy here)
    if (pack_a3_f32(xA, dAp, M, K, lda, Kp, Mp)) return -1;
    for (int i = 0; i < n; i++) {
        if (gemm_dev(r, dAp, P[i]->buf, dCp, Mp, P[i]->Np, Kp)) return -1;
        bf_t xnf = { NULL, nws[i] ? nws[i] : nf + i };
        if (epilog_hf(dCp, P[i]->Np, Mp, xC[i], ldc[i], P[i]->N, M, xnf)) return -1;
    }
    int all = 1;
    for (int i = 0; i < n; i++) all &= nws[i] != NULL;
    if (all) return 0;          // no-wait mode
    if (wait_gpu()) return -1;
    for (int i = 0; i < n; i++) nonfinite[i] = !nws[i] && nf[i] != 0;   // calls with a slot report through gen9blas_nowait_check
    return 0;
}


// Gated FFN with FP16 weights (ggml: down(swiglu(gate(x), up(x)))), all intermediates on the GPU:
// Y (F32, M x Nd) = (silu(X Wg) * (X Wu)) Wd, with Pg, Pu (K -> N) and Pd (N -> Nd) packed as GEN9BLAS_HS.
// X is packed to FP16 once; gate and up stay FP16; h is computed in F32 and rounded to FP16 into down's packed A;
// down's result goes through the FP16 -> F32 epilogue. *nonfinite = 1 on any inf / NaN / FP16 overflow on the way.
static cl_mem dG, dU, dH; static size_t sG, sU, sH;
int gen9blas_ffn_f16w(int M, const float *X, int ldx, const gen9blas_packed *Pg, const gen9blas_packed *Pu,
                      const gen9blas_packed *Pd, float *Y, int ldy, int *nonfinite) {
    static int *nf;
    NEW_OPERATION();
    if (!Pg || !Pu || !Pd || Pg->type != GEN9BLAS_HS || Pu->type != GEN9BLAS_HS || Pd->type != GEN9BLAS_HS || !k_swiglu)
        FAIL("ffn_f16w: needs weights packed as GEN9BLAS_HS");
    if (Pg->K != Pu->K || Pg->N != Pu->N || Pd->K != Pg->N || Pg->Np != Pu->Np)
        FAIL("ffn_f16w: shapes do not chain (gate %dx%d, up %dx%d, down %dx%d)", Pg->K, Pg->N, Pu->K, Pu->N, Pd->K, Pd->N);
    if (!nf && !(nf = gen9blas_alloc(64))) return -1;
    *nonfinite = 0;
    if (M <= 0) return 0;
    type_t *r = &types[GEN9BLAS_HS];
    int K = Pg->K, Kp = Pg->Kp, N = Pg->N, Np = Pg->Np, Nd = Pd->N, Npd = Pd->Np, Mp = (M + 15) / 16 * 16;
    size_t bX = ((size_t)(M - 1) * ldx + K) * 4, bY = ((size_t)(M - 1) * ldy + Nd) * 4;
    bf_t xX, xY; int zX, zY;
    if (get_obj(X, bX, &uA, 1, 0, &xX, &zX) || get_obj(Y, bY, &uC, 0, 1, &xY, &zY)) return -1;
    int *nw = zY ? nw_take() : NULL;   // no-wait mode
    if (ensure_buf(&dAp, &sAp, (size_t)Mp * Kp * 2 + 4096) || ensure_buf(&dG, &sG, (size_t)Mp * Np * 2) ||
        ensure_buf(&dU, &sU, (size_t)Mp * Np * 2) || ensure_buf(&dH, &sH, (size_t)Mp * Pd->Kp * 2 + 4096) ||
        ensure_buf(&dCp, &sCp, (size_t)Mp * Npd * 2)) return -1;
    if (!nw) *nf = 0;
    bf_t xnf = { NULL, nw ? nw : nf };
    if (pack_a3_f32(xX, dAp, M, K, ldx, Kp, Mp)) return -1;
    if (gemm_dev(r, dAp, Pg->buf, dG, Mp, Np, Kp) || gemm_dev(r, dAp, Pu->buf, dU, Mp, Np, Kp)) return -1;
    int Kpd = Pd->Kp;
    ARG(k_swiglu, 0, dG); ARG(k_swiglu, 1, dU); ARG(k_swiglu, 2, Np); ARG(k_swiglu, 3, M); ARG(k_swiglu, 4, N);
    ARG(k_swiglu, 5, dH); ARG(k_swiglu, 6, Kpd); ARGB(k_swiglu, 7, xnf);
    size_t gs[2] = { (size_t)Kpd, (size_t)Mp };
    CHECK(clEnqueueNDRangeKernel(q, k_swiglu, 2, NULL, gs, NULL, 0, NULL, NULL));
    if (gemm_dev(r, dH, Pd->buf, dCp, Mp, Npd, Kpd)) return -1;
    if (epilog_hf(dCp, Npd, Mp, xY, ldy, Nd, M, xnf)) return -1;
    if (nw) return 0;
    if (to_host(xY, zY, bY, Y)) return -1;
    *nonfinite = *nf != 0;
    return 0;
}

// ---------------- ggml backend: per-layer elementwise ops (queued without a wait) ----------------
// Each call is its own operation (userptr regions); without a wait between operations the host buffers must be registered
// as spans (gen9blas_span), so that a tensor written by one operation and read by the next has one GPU address
// (docs/no-neo.md 6g.5). An output that is not zero-copy (copied through a work buffer) forces the wait.
int gen9blas_span(const void *p, size_t bytes, int writes) {
#ifndef GEN9BLAS_NEO
    return kcl_span_add(p, bytes, writes);
#else
    (void)p; (void)bytes; (void)writes; return -1;
#endif
}
void gen9blas_span_clear(void) {
#ifndef GEN9BLAS_NEO
    kcl_span_clear();
#endif
}
int gen9blas_flush(void) { return wait_gpu(); }
static work_t uD;
static int ew_done(int sync, bf_t xY, int zY, size_t bY, void *Y) { return sync || !zY ? to_host(xY, zY, bY, Y) : 0; }

int gen9blas_rms_norm(int M, int K, const float *X, int ldx, const float *w, float eps, float *Y, int ldy, int sync) {
    NEW_OPERATION();
    if (!k_rms) FAIL("rms_norm: kernel not in this build");
    if (M <= 0 || K <= 0) return 0;
    size_t bX = ((size_t)(M - 1) * ldx + K) * 4, bY = ((size_t)(M - 1) * ldy + K) * 4;
    bf_t xX, xW = BM(NULL), xY; int zX, zW, zY, hw = w != NULL;
    if (get_obj(X, bX, &uA, 1, 0, &xX, &zX) || (hw && get_obj(w, (size_t)K * 4, &uB, 1, 0, &xW, &zW)) ||
        get_obj(Y, bY, &uC, 0, 1, &xY, &zY)) return -1;
    if (!hw) xW = xX;   // any valid buffer; not read
    ARGB(k_rms, 0, xX); ARG(k_rms, 1, ldx); ARGB(k_rms, 2, xW); ARG(k_rms, 3, hw); ARG(k_rms, 4, eps); ARGB(k_rms, 5, xY);
    ARG(k_rms, 6, ldy); ARG(k_rms, 7, K);
    size_t g[2] = { 128, (size_t)M }, l[2] = { 128, 1 };
    CHECK(clEnqueueNDRangeKernel(q, k_rms, 2, NULL, g, l, 0, NULL, NULL));
    return ew_done(sync, xY, zY, bY, Y);
}

int gen9blas_binop_rows(int op, int M, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc, int sync) {
    return gen9blas_binop_rows2(op, M, K, A, lda, B, ldb, 1, C, ldc, sync);
}
int gen9blas_binop_rows2(int op, int M, int K, const float *A, int lda, const float *B, int ldb, int bs0, float *C, int ldc, int sync) {
    NEW_OPERATION();
    if (!k_binop) FAIL("binop_rows: kernel not in this build");
    if (M <= 0 || K <= 0) return 0;
    size_t bA = ((size_t)(M - 1) * lda + K) * 4, bB = ((size_t)(ldb ? M - 1 : 0) * ldb + (bs0 ? K : 1)) * 4, bC = ((size_t)(M - 1) * ldc + K) * 4;
    bf_t xA, xB, xC; int zA, zB, zC;
    if (get_obj(A, bA, &uA, 1, 0, &xA, &zA) || get_obj(B, bB, &uB, 1, 0, &xB, &zB) || get_obj(C, bC, &uC, 0, 1, &xC, &zC)) return -1;
    ARGB(k_binop, 0, xA); ARG(k_binop, 1, lda); ARGB(k_binop, 2, xB); ARG(k_binop, 3, ldb); ARGB(k_binop, 4, xC);
    ARG(k_binop, 5, ldc); ARG(k_binop, 6, op); ARG(k_binop, 7, bs0);
    size_t g[2] = { (size_t)K, (size_t)M };
    CHECK(clEnqueueNDRangeKernel(q, k_binop, 2, NULL, g, NULL, 0, NULL, NULL));
    return ew_done(sync, xC, zC, bC, C);
}

int gen9blas_layer_norm(int M, int K, const float *X, int ldx, const float *w, const float *b, float eps, float *Y, int ldy, int sync) {
    NEW_OPERATION();
    if (!k_lnorm) FAIL("layer_norm: kernel not in this build");
    if (M <= 0 || K <= 0) return 0;
    size_t bX = ((size_t)(M - 1) * ldx + K) * 4, bY = ((size_t)(M - 1) * ldy + K) * 4;
    bf_t xX, xW = BM(NULL), xB = BM(NULL), xY; int zX, zW, zB, zY, hw = w != NULL, hb = b != NULL;
    if (get_obj(X, bX, &uA, 1, 0, &xX, &zX) || (hw && get_obj(w, (size_t)K * 4, &uB, 1, 0, &xW, &zW)) ||
        (hb && get_obj(b, (size_t)K * 4, &uD, 1, 0, &xB, &zB)) || get_obj(Y, bY, &uC, 0, 1, &xY, &zY)) return -1;
    if (!hw) xW = xX;
    if (!hb) xB = xX;
    ARGB(k_lnorm, 0, xX); ARG(k_lnorm, 1, ldx); ARGB(k_lnorm, 2, xW); ARG(k_lnorm, 3, hw); ARGB(k_lnorm, 4, xB); ARG(k_lnorm, 5, hb);
    ARG(k_lnorm, 6, eps); ARGB(k_lnorm, 7, xY); ARG(k_lnorm, 8, ldy); ARG(k_lnorm, 9, K);
    size_t g[2] = { 128, (size_t)M }, l[2] = { 128, 1 };
    CHECK(clEnqueueNDRangeKernel(q, k_lnorm, 2, NULL, g, l, 0, NULL, NULL));
    return ew_done(sync, xY, zY, bY, Y);
}

// gen9blas_layer_norm + the result packed as the FP16 A3 input of the next GEMMs (pa, reused by gen9blas_gemm_pa_f16w_ex).
int gen9blas_layer_norm_pack(int M, int K, const float *X, int ldx, const float *w, const float *b, float eps, float *Y, int ldy,
                             gen9blas_packed_a *pa, int sync) {
    NEW_OPERATION();
    if (!k_lnorm_pack) FAIL("layer_norm_pack: kernel not in this build");
    if (M <= 0 || K <= 0 || !pa) FAIL("layer_norm_pack: bad arguments");
    const type_t *r = &types[GEN9BLAS_HS];
    const int Kp = (K + r->km - 1) / r->km * r->km, Mp = (M + 15) / 16 * 16;
    if (ensure_buf(&pa->buf, &pa->cap, (size_t)Mp * Kp * 2 + 4096)) return -1;
    pa->M = M; pa->K = K; pa->Kp = Kp;
    size_t bX = ((size_t)(M - 1) * ldx + K) * 4, bY = ((size_t)(M - 1) * ldy + K) * 4;
    bf_t xX, xW = BM(NULL), xB = BM(NULL), xY; int zX, zW, zB, zY, hw = w != NULL, hb = b != NULL;
    if (get_obj(X, bX, &uA, 1, 0, &xX, &zX) || (hw && get_obj(w, (size_t)K * 4, &uB, 1, 0, &xW, &zW)) ||
        (hb && get_obj(b, (size_t)K * 4, &uD, 1, 0, &xB, &zB)) || get_obj(Y, bY, &uC, 0, 1, &xY, &zY)) return -1;
    if (!hw) xW = xX;
    if (!hb) xB = xX;
    cl_kernel k = k_lnorm_pack;
    ARGB(k, 0, xX); ARG(k, 1, ldx); ARGB(k, 2, xW); ARG(k, 3, hw); ARGB(k, 4, xB); ARG(k, 5, hb); ARG(k, 6, eps);
    ARGB(k, 7, xY); ARG(k, 8, ldy); ARG(k, 9, K); ARG(k, 10, pa->buf); ARG(k, 11, Kp); ARG(k, 12, M);
    size_t g[2] = { 128, (size_t)Mp }, l[2] = { 128, 1 };
    CHECK(clEnqueueNDRangeKernel(q, k, 2, NULL, g, l, 0, NULL, NULL));
    return ew_done(sync, xY, zY, bY, Y);
}

int gen9blas_gelu(size_t n, const float *X, float *Y, int sync) {
    NEW_OPERATION();
    if (!k_gelu) FAIL("gelu: kernel not in this build");
    if (!n) return 0;
    bf_t xX, xY; int zX, zY;
    if (get_obj(X, n * 4, &uA, 1, 0, &xX, &zX) || get_obj(Y, n * 4, &uC, 0, 1, &xY, &zY)) return -1;
    ARGB(k_gelu, 0, xX); ARGB(k_gelu, 1, xY);
    CHECK(clEnqueueNDRangeKernel(q, k_gelu, 1, NULL, &n, NULL, 0, NULL, NULL));
    return ew_done(sync, xY, zY, n * 4, Y);
}

int gen9blas_cvt_f16_heads(int rows, int H, int D, const float *X, long s_row, long s_head, uint16_t *Y, int sync) {
    NEW_OPERATION();
    if (!k_cvt_heads) FAIL("cvt_f16_heads: kernel not in this build");
    if (rows <= 0 || H <= 0 || D <= 0) return 0;
    size_t bX = ((size_t)(rows - 1) * s_row + (size_t)(H - 1) * s_head + D) * 4, bY = (size_t)rows * H * D * 2;
    bf_t xX, xY; int zX, zY;
    if (get_obj(X, bX, &uA, 1, 0, &xX, &zX) || get_obj(Y, bY, &uC, 0, 1, &xY, &zY)) return -1;
    ARGB(k_cvt_heads, 0, xX); ARG(k_cvt_heads, 1, s_row); ARG(k_cvt_heads, 2, s_head); ARGB(k_cvt_heads, 3, xY);
    size_t g[3] = { (size_t)D, (size_t)H, (size_t)rows };
    CHECK(clEnqueueNDRangeKernel(q, k_cvt_heads, 3, NULL, g, NULL, 0, NULL, NULL));
    return ew_done(sync, xY, zY, bY, Y);
}

int gen9blas_im2col(int N, int IC, int IH, int IW, const float *X, long ofs0, long ofs1, int KH, int KW, int OH, int OW,
                    int s0, int s1, int p0, int p1, int d0, int d1, void *Y, int f16, int sync) {
    NEW_OPERATION();
    if (!k_im2col) FAIL("im2col: kernel not in this build");
    if (N <= 0 || IC <= 0 || OH <= 0 || OW <= 0 || KH <= 0 || KW <= 0) return 0;
    size_t bX = ((size_t)(N - 1) * ofs0 + (size_t)(IC - 1) * ofs1 + (size_t)IH * IW) * 4;
    size_t bY = (size_t)N * OH * OW * IC * KH * KW * (f16 ? 2 : 4);
    bf_t xX, xY; int zX, zY;
    if (get_obj(X, bX, &uA, 1, 0, &xX, &zX) || get_obj(Y, bY, &uC, 0, 1, &xY, &zY)) return -1;
    ARGB(k_im2col, 0, xX); ARG(k_im2col, 1, ofs0); ARG(k_im2col, 2, ofs1); ARG(k_im2col, 3, IH); ARG(k_im2col, 4, IW);
    ARG(k_im2col, 5, KH); ARG(k_im2col, 6, KW); ARG(k_im2col, 7, OH); ARG(k_im2col, 8, s0); ARG(k_im2col, 9, s1);
    ARG(k_im2col, 10, p0); ARG(k_im2col, 11, p1); ARG(k_im2col, 12, d0); ARG(k_im2col, 13, d1); ARG(k_im2col, 14, f16);
    ARGB(k_im2col, 15, xY);
    size_t g[3] = { (size_t)IC * KH * KW, (size_t)OW, (size_t)N * OH };
    CHECK(clEnqueueNDRangeKernel(q, k_im2col, 3, NULL, g, NULL, 0, NULL, NULL));
    return ew_done(sync, xY, zY, bY, Y);
}

int gen9blas_relu(size_t n, const float *X, float *Y, int sync) {
    NEW_OPERATION();
    if (!k_relu) FAIL("relu: kernel not in this build");
    if (!n) return 0;
    bf_t xX, xY; int zX, zY;
    if (X == Y) {
        if (get_obj(Y, n * 4, &uC, 1, 1, &xY, &zY)) return -1;
        xX = xY;
    } else if (get_obj(X, n * 4, &uA, 1, 0, &xX, &zX) || get_obj(Y, n * 4, &uC, 0, 1, &xY, &zY)) return -1;
    ARGB(k_relu, 0, xX); ARGB(k_relu, 1, xY);
    CHECK(clEnqueueNDRangeKernel(q, k_relu, 1, NULL, &n, NULL, 0, NULL, NULL));
    return ew_done(sync, xY, zY, n * 4, Y);
}

int gen9blas_gelu_erf(size_t n, const float *X, float *Y, int sync) {
    NEW_OPERATION();
    if (!k_gelu_erf) FAIL("gelu_erf: kernel not in this build");
    if (!n) return 0;
    bf_t xX, xY; int zX, zY;
    if (X == Y) {   // in place: one object
        if (get_obj(Y, n * 4, &uC, 1, 1, &xY, &zY)) return -1;
        xX = xY;
    } else if (get_obj(X, n * 4, &uA, 1, 0, &xX, &zX) || get_obj(Y, n * 4, &uC, 0, 1, &xY, &zY)) return -1;
    ARGB(k_gelu_erf, 0, xX); ARGB(k_gelu_erf, 1, xY);
    CHECK(clEnqueueNDRangeKernel(q, k_gelu_erf, 1, NULL, &n, NULL, 0, NULL, NULL));
    return ew_done(sync, xY, zY, n * 4, Y);
}

int gen9blas_rope(int n_dims, int nh, int n_tok, const float *X, long xs1, long xs2, float *Y, long ys1, long ys2,
                  const int *pos, const float *ff, int neox, float freq_base, float freq_scale, float ext_factor,
                  float attn_factor, const float corr[2], int sync) {
    NEW_OPERATION();
    if (!k_rope) FAIL("rope: kernel not in this build");
    if (n_dims <= 0 || nh <= 0 || n_tok <= 0) return 0;
    if (n_dims % 2 || xs1 < n_dims || ys1 < n_dims || xs2 < (long)nh * xs1 - (xs1 - n_dims) || ys2 < (long)nh * ys1 - (ys1 - n_dims))
        FAIL("rope: unsupported layout");
    size_t bX = ((size_t)(n_tok - 1) * xs2 + (size_t)(nh - 1) * xs1 + n_dims) * 4;
    size_t bY = ((size_t)(n_tok - 1) * ys2 + (size_t)(nh - 1) * ys1 + n_dims) * 4;
    bf_t xX, xY, xP, xF = BM(NULL); int zX, zY, zP, zF, hf = ff != NULL;
    if (get_obj(X, bX, &uA, 1, 0, &xX, &zX) || get_obj(pos, (size_t)n_tok * 4, &uB, 1, 0, &xP, &zP) ||
        (hf && get_obj(ff, (size_t)n_dims / 2 * 4, &uD, 1, 0, &xF, &zF)) || get_obj(Y, bY, &uC, 0, 1, &xY, &zY)) return -1;
    if (!hf) xF = xP;
    const float theta_scale = powf(freq_base, -2.0f / n_dims);   // as ggml-cpu
    cl_long l1 = xs1, l2 = xs2, m1 = ys1, m2 = ys2;
    ARGB(k_rope, 0, xX); ARG(k_rope, 1, l1); ARG(k_rope, 2, l2); ARGB(k_rope, 3, xY); ARG(k_rope, 4, m1); ARG(k_rope, 5, m2);
    ARGB(k_rope, 6, xP); ARGB(k_rope, 7, xF); ARG(k_rope, 8, hf); ARG(k_rope, 9, nh); ARG(k_rope, 10, n_dims); ARG(k_rope, 11, neox);
    ARG(k_rope, 12, theta_scale); ARG(k_rope, 13, freq_scale); ARG(k_rope, 14, ext_factor); ARG(k_rope, 15, attn_factor);
    ARG(k_rope, 16, corr[0]); ARG(k_rope, 17, corr[1]);
    size_t g[2] = { (size_t)n_dims / 2, (size_t)n_tok };
    CHECK(clEnqueueNDRangeKernel(q, k_rope, 2, NULL, g, NULL, 0, NULL, NULL));
    return ew_done(sync, xY, zY, bY, Y);
}

// ggml SET_ROWS F32 -> F16 (gen9blas.h). src / dst strides in elements, index strides in bytes; dst_bytes = extent of dst.
int gen9blas_set_rows_supported(void) { return k_setrows != NULL; }
int gen9blas_set_rows_f16(int nc, int nr, int ne02, int ne03, const float *src, long s1, long s2, long s3,
                          const void *idx, size_t idx_bytes, long b10, long b11, long b12, int idx64, int ne11, int ne12,
                          uint16_t *dst, size_t dst_bytes, long d1, long d2, long d3, long ne1, int sync, int *bad) {
    static int *flag;
    NEW_OPERATION();
    if (!k_setrows) FAIL("set_rows: kernel not in this build");
    if (nc <= 0 || nr <= 0 || ne02 <= 0 || ne03 <= 0) return 0;
    if (!flag && !(flag = gen9blas_alloc(64))) return -1;
    size_t bS = ((size_t)(nr - 1) * s1 + (size_t)(ne02 - 1) * s2 + (size_t)(ne03 - 1) * s3 + nc) * 4;
    bf_t xS, xI, xD; int zS, zI, zD;
    if (get_obj(src, bS, &uA, 1, 0, &xS, &zS) || get_obj(idx, idx_bytes, &uB, 1, 0, &xI, &zI) ||
        get_obj(dst, dst_bytes, &uC, 1, 1, &xD, &zD)) return -1;
    if (!zD) FAIL("set_rows: destination is not zero-copy");   // a copy-in/copy-out of the whole cache would cost more than the op
    *flag = 0;
    bf_t xF = { NULL, flag };
    cl_long a1 = s1, a2 = s2, a3 = s3, c10 = b10, c11 = b11, c12 = b12, e1 = d1, e2 = d2, e3 = d3, n1 = ne1;
    cl_kernel k = k_setrows;
    ARGB(k, 0, xS); ARG(k, 1, a1); ARG(k, 2, a2); ARG(k, 3, a3); ARGB(k, 4, xI); ARG(k, 5, c10); ARG(k, 6, c11); ARG(k, 7, c12);
    ARG(k, 8, idx64); ARG(k, 9, ne11); ARG(k, 10, ne12); ARG(k, 11, ne02); ARGB(k, 12, xD); ARG(k, 13, e1); ARG(k, 14, e2);
    ARG(k, 15, e3); ARG(k, 16, n1); ARGB(k, 17, xF);
    size_t g[3] = { (size_t)nc, (size_t)nr, (size_t)ne02 * ne03 };
    CHECK(clEnqueueNDRangeKernel(q, k, 3, NULL, g, NULL, 0, NULL, NULL));
    if (sync) { if (wait_gpu()) return -1; if (bad) *bad = *flag; }
    return 0;
}

// ggml_rope_multi, mode MROPE (sections s[4]): pos holds 4 * n_tok positions (t, h, w, e). Layout rules as gen9blas_rope.
int gen9blas_rope_multi_supported(void) { return k_rope_m != NULL; }
int gen9blas_rope_multi(int n_dims, int nh, int n_tok, const float *X, long xs1, long xs2, float *Y, long ys1, long ys2,
                        const int *pos, const float *ff, const int sections[4], float freq_base, float freq_scale,
                        float ext_factor, float attn_factor, const float corr[2], int vision, int sync) {
    NEW_OPERATION();
    if (!k_rope_m) FAIL("rope_multi: kernel not in this build");
    if (n_dims <= 0 || nh <= 0 || n_tok <= 0) return 0;
    const int ne0 = vision ? 2 * n_dims : n_dims;   // elements per head row that are rotated (vision: all, ne0 = 2 n_dims)
    if (n_dims % 2 || xs1 < ne0 || ys1 < ne0 || xs2 < (long)nh * xs1 - (xs1 - ne0) || ys2 < (long)nh * ys1 - (ys1 - ne0))
        FAIL("rope_multi: unsupported layout");
    if (sections[0] + sections[1] + sections[2] + sections[3] <= 0) FAIL("rope_multi: empty sections");
    size_t bX = ((size_t)(n_tok - 1) * xs2 + (size_t)(nh - 1) * xs1 + ne0) * 4;
    size_t bY = ((size_t)(n_tok - 1) * ys2 + (size_t)(nh - 1) * ys1 + ne0) * 4;
    bf_t xX, xY, xP, xF = BM(NULL); int zX, zY, zP, zF, hf = ff != NULL;
    if (get_obj(X, bX, &uA, 1, 0, &xX, &zX) || get_obj(pos, (size_t)n_tok * 16, &uB, 1, 0, &xP, &zP) ||
        (hf && get_obj(ff, (size_t)(vision ? n_dims : n_dims / 2) * 4, &uD, 1, 0, &xF, &zF)) || get_obj(Y, bY, &uC, 0, 1, &xY, &zY)) return -1;
    if (!hf) xF = xP;
    const float theta_scale = powf(freq_base, -2.0f / n_dims);   // as ggml-cpu
    cl_long l1 = xs1, l2 = xs2, m1 = ys1, m2 = ys2;
    cl_kernel k = k_rope_m;
    ARGB(k, 0, xX); ARG(k, 1, l1); ARG(k, 2, l2); ARGB(k, 3, xY); ARG(k, 4, m1); ARG(k, 5, m2);
    ARGB(k, 6, xP); ARGB(k, 7, xF); ARG(k, 8, hf); ARG(k, 9, nh); ARG(k, 10, n_dims); ARG(k, 11, n_tok);
    ARG(k, 12, sections[0]); ARG(k, 13, sections[1]); ARG(k, 14, sections[2]); ARG(k, 15, sections[3]);
    ARG(k, 16, theta_scale); ARG(k, 17, freq_scale); ARG(k, 18, ext_factor); ARG(k, 19, attn_factor);
    ARG(k, 20, corr[0]); ARG(k, 21, corr[1]);
    const int half = vision ? n_dims : n_dims / 2;
    ARG(k, 22, half); ARG(k, 23, vision);
    size_t g[2] = { (size_t)half, (size_t)n_tok };
    CHECK(clEnqueueNDRangeKernel(q, k, 2, NULL, g, NULL, 0, NULL, NULL));
    return ew_done(sync, xY, zY, bY, Y);
}

// ggml CPY / CONT / DUP, F32 / F16 in and out, byte strides, dst shape may differ (same element count): gen9blas.h.
static size_t extent4(const long ne[4], const long nb[4], size_t es) {
    size_t e = es;
    for (int i = 0; i < 4; i++) e += (size_t)(ne[i] - 1) * (size_t)nb[i];
    return e;
}
int gen9blas_cpy_supported(void) { return k_cpy_any != NULL && k_upscale != NULL; }
int gen9blas_cpy_any(const void *src, int tin, const long sne[4], const long snb[4], void *dst, int tout, const long dne[4],
                     const long dnb[4], int sync) {
    NEW_OPERATION();
    if (!k_cpy_any) FAIL("cpy_any: kernel not in this build");
    long n = 1, m = 1;
    for (int i = 0; i < 4; i++) {
        if (sne[i] <= 0 || dne[i] <= 0) return 0;
        if (snb[i] < 0 || dnb[i] < 0 || sne[i] > INT_MAX || dne[i] > INT_MAX) FAIL("cpy_any: unsupported layout");
        n *= sne[i]; m *= dne[i];
    }
    if (n != m || (tin | tout) & ~1 || dne[0] * dne[1] > INT_MAX || dne[2] * dne[3] > INT_MAX) FAIL("cpy_any: unsupported shape");
    const int same = sne[0] == dne[0] && sne[1] == dne[1] && sne[2] == dne[2] && sne[3] == dne[3];
    const size_t bS = extent4(sne, snb, tin ? 2 : 4), bD = extent4(dne, dnb, tout ? 2 : 4);
    bf_t xS, xD; int zS, zD;
    if (get_obj(src, bS, &uA, 1, 0, &xS, &zS) || get_obj(dst, bD, &uC, 1, 1, &xD, &zD)) return -1;   // copy-in: dst may be a strided view
    cl_long a0 = snb[0], a1 = snb[1], a2 = snb[2], a3 = snb[3], c0 = dnb[0], c1 = dnb[1], c2 = dnb[2], c3 = dnb[3];
    int s0 = (int)sne[0], s1 = (int)sne[1], s2 = (int)sne[2], d0 = (int)dne[0], d1 = (int)dne[1], d2 = (int)dne[2];
    cl_kernel k = k_cpy_any;
    ARGB(k, 0, xS); ARG(k, 1, a0); ARG(k, 2, a1); ARG(k, 3, a2); ARG(k, 4, a3); ARG(k, 5, s0); ARG(k, 6, s1); ARG(k, 7, s2);
    ARGB(k, 8, xD); ARG(k, 9, c0); ARG(k, 10, c1); ARG(k, 11, c2); ARG(k, 12, c3); ARG(k, 13, d0); ARG(k, 14, d1); ARG(k, 15, d2);
    ARG(k, 16, tin); ARG(k, 17, tout); ARG(k, 18, same);
    size_t g[3] = { (size_t)dne[0], (size_t)dne[1], (size_t)(dne[2] * dne[3]) };
    CHECK(clEnqueueNDRangeKernel(q, k, 3, NULL, g, NULL, 0, NULL, NULL));
    return ew_done(sync, xD, zD, bD, dst);
}

// ggml UPSCALE (ggml_interpolate) F32: mode 0 nearest / 1 bilinear, flags as ggml (bit 8 align corners, bit 9 antialias).
int gen9blas_upscale_f32(const float *src, const long sne[4], const long snb[4], float *dst, const long dne[4], const long dnb[4],
                         int mode_flags, int sync) {
    NEW_OPERATION();
    if (!k_upscale) FAIL("upscale: kernel not in this build");
    for (int i = 0; i < 4; i++) {
        if (sne[i] <= 0 || dne[i] <= 0) return 0;
        if (snb[i] < 0 || dnb[i] < 0 || dne[i] > INT_MAX) FAIL("upscale: unsupported layout");
    }
    const int mode = mode_flags & 0xFF, ac = (mode_flags >> 8) & 1, aa = (mode_flags >> 9) & 1;
    if ((mode != 0 && mode != 1) || (mode_flags & ~0x3FF) || dne[0] * dne[1] > INT_MAX || dne[2] * dne[3] > INT_MAX) FAIL("upscale: unsupported mode");
    float sf0 = (float)dne[0] / sne[0], sf1 = (float)dne[1] / sne[1], sf2 = (float)dne[2] / sne[2], sf3 = (float)dne[3] / sne[3];
    float po = 0.5f;
    if (ac) {   // as ggml-cpu
        po = 0.0f;
        sf0 = dne[0] > 1 && sne[0] > 1 ? (float)(dne[0] - 1) / (sne[0] - 1) : sf0;
        sf1 = dne[1] > 1 && sne[1] > 1 ? (float)(dne[1] - 1) / (sne[1] - 1) : sf1;
    }
    const size_t bS = extent4(sne, snb, 4), bD = extent4(dne, dnb, 4);
    bf_t xS, xD; int zS, zD;
    if (get_obj(src, bS, &uA, 1, 0, &xS, &zS) || get_obj(dst, bD, &uC, 1, 1, &xD, &zD)) return -1;
    cl_long a0 = snb[0], a1 = snb[1], a2 = snb[2], a3 = snb[3], c0 = dnb[0], c1 = dnb[1], c2 = dnb[2], c3 = dnb[3];
    int s0 = (int)sne[0], s1 = (int)sne[1], n2 = (int)dne[2];
    cl_kernel k = k_upscale;
    ARGB(k, 0, xS); ARG(k, 1, a0); ARG(k, 2, a1); ARG(k, 3, a2); ARG(k, 4, a3); ARG(k, 5, s0); ARG(k, 6, s1);
    ARGB(k, 7, xD); ARG(k, 8, c0); ARG(k, 9, c1); ARG(k, 10, c2); ARG(k, 11, c3); ARG(k, 12, n2);
    ARG(k, 13, sf0); ARG(k, 14, sf1); ARG(k, 15, sf2); ARG(k, 16, sf3); ARG(k, 17, mode); ARG(k, 18, aa); ARG(k, 19, po);
    size_t g[3] = { (size_t)dne[0], (size_t)dne[1], (size_t)(dne[2] * dne[3]) };
    CHECK(clEnqueueNDRangeKernel(q, k, 3, NULL, g, NULL, 0, NULL, NULL));
    return ew_done(sync, xD, zD, bD, dst);
}

// Decoding attention over KV streams (gen9blas.h): split-K partial kernel + merge, queued (sync: wait at the end).
int gen9blas_attn_dec_supported(int D, int G) { return k_dec_part && k_dec_comb && D == 128 && G >= 1 && G <= 8; }
int gen9blas_attn_dec(int D, int H, int HKV, int ns, int nq, int nk, const float *Q, long qt, long qh, long qs,
                      const uint16_t *K, long kk, long kh, long ks, size_t k_bytes, const uint16_t *V, long vk, long vh, long vs,
                      size_t v_bytes, float *O, float scale, const int *vis, int dvis, int chunk, int sync) {
    static float *part; static size_t part_n;
    static int *visring; static int vslot;   // 256 slots of 64 ints: launches stay queued; wait only when the ring wraps
    NEW_OPERATION();
    if (!k_dec_part || !k_dec_comb) FAIL("attn_dec: kernels not in this build");
    const int G = HKV > 0 ? H / HKV : 0;
    if (D != 128 || G < 1 || G > 8 || H != G * HKV || ns < 1 || nq < 1 || nk < 1 || chunk < 16 || chunk % 16) FAIL("attn_dec: unsupported shape");
    if (((uintptr_t)K | (uintptr_t)V | (uintptr_t)(kk * 2) | (uintptr_t)(kh * 2) | (uintptr_t)(vk * 2) | (uintptr_t)(vh * 2)) & 15)
        FAIL("attn_dec: K/V rows not 16-byte aligned");
    const int nchunk = (nk + chunk - 1) / chunk;
    const size_t pn = (size_t)ns * nq * H * nchunk * (2 + 128);
    if (pn > part_n) {
        if (part) { if (wait_gpu()) return -1; usm_free(ctx, part); }
        part = gen9blas_alloc(pn * 4); part_n = part ? pn : 0;
        if (!part) FAIL("attn_dec: scratch of %zu B", pn * 4);
    }
    if (ns > 64) FAIL("attn_dec: more than 64 streams");
    if (!visring && !(visring = gen9blas_alloc((size_t)256 * 64 * 4))) FAIL("attn_dec: vis buffer");
    if (vslot == 256) { if (wait_gpu()) return -1; vslot = 0; }
    int *visb = visring + (size_t)64 * vslot++;
    memcpy(visb, vis, (size_t)ns * 4);
    const size_t bQ = ((size_t)(ns - 1) * qs + (size_t)(nq - 1) * qt + (size_t)(H - 1) * qh + 128) * 4;
    const size_t bO = (size_t)ns * nq * H * 128 * 4;
    bf_t xQ, xK, xV, xO, xP = { NULL, part }, xS = { NULL, visb }; int zQ, zO;
    // K/V zero-copy or nothing, checked before any wait or copy: a copy of the whole cache per call would cost more than the op
    if (get_obj_zc(K, k_bytes, 0, &xK) || get_obj_zc(V, v_bytes, 0, &xV)) { vslot--; FAIL("attn_dec: K/V not zero-copy"); }
    if (get_obj(Q, bQ, &uA, 1, 0, &xQ, &zQ) || get_obj(O, bO, &uC, 0, 1, &xO, &zO)) return -1;
    const float qscale = scale * 1.4426950408889634f;
    cl_long a1 = qt, a2 = qh, a3 = qs, b1 = kk, b2 = kh, b3 = ks, c1 = vk, c2 = vh, c3 = vs;
    cl_kernel k = k_dec_part;
    ARGB(k, 0, xQ); ARG(k, 1, a1); ARG(k, 2, a2); ARG(k, 3, a3); ARGB(k, 4, xK); ARG(k, 5, b1); ARG(k, 6, b2); ARG(k, 7, b3);
    ARGB(k, 8, xV); ARG(k, 9, c1); ARG(k, 10, c2); ARG(k, 11, c3); ARGB(k, 12, xS); ARG(k, 13, dvis); ARG(k, 14, nq);
    ARG(k, 15, HKV); ARG(k, 16, G); ARG(k, 17, nk); ARG(k, 18, chunk); ARG(k, 19, nchunk); ARG(k, 20, qscale); ARGB(k, 21, xP);
    size_t g[3] = { 16, (size_t)nchunk * ((G + 1) / 2), (size_t)ns * nq * HKV }, l[3] = { 16, 1, 1 };   // 2 heads per work-group (DEC_HG)
    CHECK(clEnqueueNDRangeKernel(q, k, 3, NULL, g, l, 0, NULL, NULL));
    k = k_dec_comb;
    ARGB(k, 0, xP); ARGB(k, 1, xO); ARG(k, 2, H); ARG(k, 3, nchunk);
    size_t g2[3] = { 128, (size_t)H, (size_t)ns * nq };
    CHECK(clEnqueueNDRangeKernel(q, k, 3, NULL, g2, NULL, 0, NULL, NULL));
    return ew_done(sync, xO, zO, bO, O);
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
