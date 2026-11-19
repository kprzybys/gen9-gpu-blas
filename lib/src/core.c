// libgen9blas: level 3 BLAS on UHD 620 (Gen9.5) with hand-written assembly GEMM loops (asm/gemm_gen.py).
// Core: batched GEMM C_b = alpha * op(A_b)·op(B_b) + beta * C_b (b = 0..count-1, strides sA, sB, sC in elements),
// row-major, arbitrary M, N, K. Types: GEN9BLAS_H (FP16), GEN9BLAS_HS (FP16, accumulated in chunks into FP32), GEN9BLAS_S (FP32).
// Level 3 (SYMM, SYRK, SYR2K, TRMM, TRSM) is built on GEMM, entirely on the GPU: expanding the triangular /
// symmetric matrix, joining the SYR2K operands and the inverses of the TRSM diagonal blocks (FP64) are kernels.
// The CPU only submits work and sleeps while waiting (wait_gpu()). Zero-copy: USM host memory (gen9blas_alloc, gen9blas.empty in Python);
// other host memory is copied by the CPU into USM work buffers (the driver does not register ordinary host memory).
// This file: GPU context, programs and kernels, waiting, work buffers, host memory (USM / userptr), launches.
#include "internal.h"
double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }


cl_context ctx;
cl_command_queue q;
static cl_device_id dev;
type_t types[4];   // index = type: GEN9BLAS_H, GEN9BLAS_S, GEN9BLAS_HS, GEN9BLAS_D
cl_mem dAp, dBp, dCp;
size_t sAp, sBp, sCp;
double last_ms;
int profile;   // GEN9BLAS_PROFILE=1: profiling queue, kernel times
int last_zc;
char errbuf[1200];
int gst;
static clHostMemAllocINTEL_fn usm_alloc;          // cl_intel_unified_shared_memory (see get_obj())
clMemBlockingFreeINTEL_fn usm_free;
static clGetMemAllocInfoINTEL_fn usm_info;
clSetKernelArgMemPointerINTEL_fn usm_arg;


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
    if (kcl_program(file, NULL, &pr) != CL_SUCCESS) FAILC(GEN9BLAS_ERR_UNSUPPORTED, "no built-in program %s", file);
    *k = clCreateKernel(pr, kernel, &err); CHECK(err);
    return 0;
#else
    char p[1024]; size_t len; cl_int err, bst;
    snprintf(p, sizeof p, "%s/%s", dir, file);
    unsigned char *bin = load_file(p, &len);
    if (!bin) FAILC(GEN9BLAS_ERR_UNSUPPORTED, "missing %.900s (build: make -C asm)", p);
    cl_program pb = clCreateProgramWithBinary(ctx, 1, &dev, &len, (const unsigned char **)&bin, &bst, &err);
    free(bin); CHECK(err); CHECK(bst); CHECK(clBuildProgram(pb, 1, &dev, "", NULL, NULL));
    *k = clCreateKernel(pb, kernel, &err); CHECK(err);
    return 0;
#endif
}

static int program_src(const char *dir, const char *file, const char *options, cl_program *pr) {
#ifndef GEN9BLAS_NEO
    (void)dir;
    if (kcl_program(file, options, pr) != CL_SUCCESS) FAILC(GEN9BLAS_ERR_UNSUPPORTED, "no built-in program %s [%s]", file, options);
    return 0;
#else
    char p[1024]; size_t len; cl_int err;
    snprintf(p, sizeof p, "%s/%s", dir, file);
    char *src = (char *)load_file(p, &len);
    if (!src) FAILC(GEN9BLAS_ERR_UNSUPPORTED, "missing %.900s", p);
    *pr = clCreateProgramWithSource(ctx, 1, (const char **)&src, NULL, &err); free(src); CHECK(err);
    if (clBuildProgram(*pr, 1, &dev, options, NULL, NULL) != CL_SUCCESS) {
        clGetProgramBuildInfo(*pr, dev, CL_PROGRAM_BUILD_LOG, sizeof errbuf, errbuf, NULL); return ERR_RET;
    }
    return 0;
#endif
}


// Level 1–2 kernels (lib/kernels/level12.cl): L12[0] FP16 (GEN9BLAS_H, GEN9BLAS_HS), L12[1] FP32.
l12_t L12[3];   // 0 FP16, 1 FP32, 2 FP64
// Gen8 Cherryview (kcl, docs/gen8.md): no FP16 for now (its kernels need scratch there), GEMM main kernels from OpenCL C
// instead of the Gen9 assembler binaries, gemv_n_f unavailable (scratch) -> the grouped GEMV for every N.
// Gen12LP Tiger Lake (kcl, docs/gen12lp.md): no FP64 (no dgemm.cl, no level12 -DDBL), no assembler binaries (Gen9
// encoding): hgemm_v / hgemm_kc_v / sgemm_v from OpenCL C; skinny_n4 spills there and is optional.
int gen8, gen12;
static int init_l12(const char *dir) {
    for (int i = 0; i < 3; i++) {
        if ((gen8 && i == 0) || (gen12 && i == 2)) continue;
        cl_program pr; const char *suf = "hfd" + i, *opc[3] = { "-cl-std=CL3.0 -DHALF", "-cl-std=CL3.0", "-cl-std=CL3.0 -DDBL" };
        char n[32], sf[2] = { *suf, 0 };
        if (program_src(dir, "level12.cl", opc[i], &pr)) return ERR_RET;
        l12_t *k = &L12[i]; k->es = i == 0 ? 2 : i == 1 ? 4 : 8; suf = sf;
        struct { const char *n; cl_kernel *k; } t[] = { { "axpy", &k->axpy }, { "scal", &k->scal }, { "copy", &k->copy }, { "copy16", &k->copy16 },
            { "rot", &k->rot }, { "red", &k->red }, { "iamax", &k->iamax }, { "gemv_n", &k->gemv_n }, { "gemv_nw", &k->gemv_nw },
            { "gemv_t1", &k->gemv_t1 }, { "gemv_t1b", &k->gemv_t1b }, { "gemv_t2", &k->gemv_t2 }, { "ger", &k->ger }, { "syr2", &k->syr2 },
            { "to_f", &k->to_f }, { "from_f", &k->from_f }, { "trsv_block", &k->trsv_block },
            { "trsv_upd_n", &k->trsv_upd_n }, { "trsv_upd_t", &k->trsv_upd_t }, { "rotm", &k->rotm }, { "pmv", &k->pmv },
            { "psv", &k->psv }, { "spr2", &k->spr2 }, { "symv1", &k->symv1 }, { "symv2", &k->symv2 } };
        for (size_t j = 0; j < sizeof t / sizeof t[0]; j++) {
            snprintf(n, sizeof n, "%s_%s", t[j].n, suf);
            if (gen8 && t[j].k == &k->gemv_n) { cl_int e; k->gemv_n = clCreateKernel(pr, n, &e); if (e) k->gemv_n = NULL; continue; }
            KER(pr, n, *t[j].k);
        }
        if (i < 2) {   // skinny GEMM: FP16 and FP32 only (lib/kernels/level12.cl, #ifndef DBL)
            struct { const char *n; cl_kernel *k; } c[] = { { "skinny_k1", &k->skinny_k1 }, { "skinny_k4", &k->skinny_k4 }, { "skinny_k8", &k->skinny_k8 },
                { "skinny_r4", &k->skinny_r4 }, { "skinny_r8", &k->skinny_r8 },
                { "skinny_n4", &k->skinny_n4 }, { "skinny_n2", &k->skinny_n2 } };
            for (size_t j = 0; j < sizeof c / sizeof c[0]; j++) {
                snprintf(n, sizeof n, "%s_%s", c[j].n, suf);
                if (gen12 && c[j].k == &k->skinny_n4) { cl_int e; k->skinny_n4 = clCreateKernel(pr, n, &e); if (e) k->skinny_n4 = NULL; continue; }
                KER(pr, n, *c[j].k);
            }
        }
    }
    return 0;
}

// One-shot (gen9blas_init below): later calls (e.g. the BLAS layer and the Python module in one process) do nothing -- a second OpenCL
// context would invalidate the work buffers and USM memory of the first (error -50 when setting arguments).
cl_kernel k_rope_m, k_setrows, k_cpy_any, k_upscale, k_dec_part, k_dec_comb;   // gen9blas_rope_multi, _set_rows_f16, _cpy_any, _upscale_f32 (NULL if the binary lacks them)
cl_kernel k_skp[3];   // gen9blas_skinny_pb16_4 / _8 / _16
cl_kernel k_pa3_f32, k_epi_hf, k_swiglu, k_dequant, k_dequant2, k_pa3_v, k_epi_8, k_epi_8x;
cl_kernel k_rms, k_binop, k_rope, k_lnorm, k_gelu, k_lnorm_pack, k_gelu_erf, k_cvt_heads, k_im2col, k_relu;   // ggml per-layer elementwise ops (NULL if the binary lacks them)   // k_dequant: NULL if the binary lacks it   // ggml backend: F16 weights x F32 activations (gemm_pb_f16w, ffn_f16w)
static int init_impl(const char *dir) {
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
#ifndef GEN9BLAS_NEO
    gen8 = kcl_gen8(); gen12 = kcl_gen12();
#endif
    cl_program ph, ps, pd = NULL;
    if (program_src(dir, "hgemm.cl", "", &ph) || program_src(dir, "sgemm.cl", "", &ps) ||
        (!gen12 && program_src(dir, "dgemm.cl", "", &pd))) return ERR_RET;
    usm_alloc = (clHostMemAllocINTEL_fn)clGetExtensionFunctionAddressForPlatform(plat, "clHostMemAllocINTEL");
    usm_free = (clMemBlockingFreeINTEL_fn)clGetExtensionFunctionAddressForPlatform(plat, "clMemBlockingFreeINTEL");
    usm_info = (clGetMemAllocInfoINTEL_fn)clGetExtensionFunctionAddressForPlatform(plat, "clGetMemAllocInfoINTEL");
    usm_arg = (clSetKernelArgMemPointerINTEL_fn)clGetExtensionFunctionAddressForPlatform(plat, "clSetKernelArgMemPointerINTEL");
    if (!usm_alloc || !usm_free || !usm_info || !usm_arg) FAILC(GEN9BLAS_ERR_UNSUPPORTED, "missing cl_intel_unified_shared_memory extension");
    if (init_l12(dir)) return ERR_RET;
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
    // Gen8 SGEMM: sgemm8_v32 (16 x 32 tile, SIMD8, its own packed layouts; lib/kernels/sgemm.cl); GEN9BLAS_SGEMM8=v: the
    // OpenCL frame sgemm_v of the Gen9 kernel (comparisons, docs/gen8.md 6d)
    const char *s8 = getenv("GEN9BLAS_SGEMM8");
    if (gen8 && !(s8 && !strcmp(s8, "v"))) { t[2].k = "sgemm8_v32"; t[2].pa = "pkf_a8"; t[2].pb = "pkf_b32"; t[2].tn = 32; }
    // Gen8 DGEMM: dgemm8_v832 (8 x 32 tiles, SIMD8, B in 32-column panels; lib/kernels/dgemm_core.cl); GEN9BLAS_DGEMM8=v: dgemm_v
    const char *d8 = getenv("GEN9BLAS_DGEMM8");
    if (gen8 && !(d8 && !strcmp(d8, "v"))) { t[3].k = "dgemm8_v832"; t[3].pb = "pkd_b32"; t[3].tn = 32; }
    for (int i = 0; i < 4; i++) {
        if (gen8 && t[i].es == 2) continue;   // FP16: not on Gen8 yet
        if (gen12 && t[i].es == 8) continue;  // no FP64 on Gen12LP
        type_t *r = &types[t[i].r]; char n[64];
        // Gen8, Gen12: main kernels from OpenCL C (hgemm.cl / sgemm.cl, same arguments as the assembler binaries)
        const char *bin = gen8 || gen12 ? NULL : t[i].bin;
        if (bin ? program_bin(dir, bin, t[i].k, &r->main) : 0) return ERR_RET;
        if (!bin) KER(t[i].pr, t[i].k, r->main);
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
        r->sg = (gen8 && t[i].r == GEN9BLAS_S && t[i].tn == 32) ? 8 : 16;
    }
    errbuf[0] = 0;
    return 0;
}

// Once per process (pthread_once): a failed initialization is final -- retrying would create a second context next to
// the half-initialized first one. Its error text is kept and restored on every later call.
static pthread_once_t init_once = PTHREAD_ONCE_INIT;
static int init_status = -1;
static char init_err[sizeof errbuf];
static void init_run(void) {
    const char *d = getenv("GEN9BLAS_DIR");   // NEO build only: kernel sources (kcl has them built in)
    init_status = init_impl(d ? d : "lib/kernels");
    if (init_status) snprintf(init_err, sizeof init_err, "%s", errbuf[0] ? errbuf : "GPU initialization failed");
}
int gen9blas_init(void) {
    pthread_once(&init_once, init_run);
    if (init_status) { snprintf(errbuf, sizeof errbuf, "%s", init_err); gst = GEN9BLAS_ERR_NO_DEVICE; return gst; }
    return 0;
}
int gen9blas_version(void) { return GEN9BLAS_VERSION; }
int gen9blas_device_gen(void) { return gen9blas_init() ? 0 : gen8 ? 8 : gen12 ? 12 : 9; }
int gen9blas_has(int f) {
    if (gen9blas_init()) return 0;
    switch (f) {
    case GEN9BLAS_FEAT_FP16: return types[GEN9BLAS_H].ready && types[GEN9BLAS_HS].ready;
    case GEN9BLAS_FEAT_FP32: return types[GEN9BLAS_S].ready;
    case GEN9BLAS_FEAT_FP64: return types[GEN9BLAS_D].ready;
    case GEN9BLAS_FEAT_SKINNY_PB: return k_skp[0] && k_skp[1] && k_skp[2];
    case GEN9BLAS_FEAT_EPILOG_EXT: return k_epi_8x != NULL;
    case GEN9BLAS_FEAT_NORM_PACK: return k_lnorm_pack != NULL && k_epi_8x != NULL;
    case GEN9BLAS_FEAT_ROPE_MULTI: return k_rope_m != NULL;
    case GEN9BLAS_FEAT_CPY: return k_cpy_any != NULL && k_upscale != NULL;
    case GEN9BLAS_FEAT_SET_ROWS: return k_setrows != NULL;
    default: return 0;
    }
}

// Sleeping wait for the queue to finish: clFinish / clWaitForEvents / blocking reads in NEO spin actively
// on the CPU (~6 W). Marker + clFlush, then polling the state every 20 µs .. 0.5 ms. GEN9BLAS_WAIT=busy: clFinish.
int wait_gpu(void) {
#ifndef GEN9BLAS_NEO   // kcl: submit the batch buffer and GEM_WAIT (sleeps in the Linux kernel until the GPU interrupt)
    if (kcl_wait() != CL_SUCCESS) FAILC(GEN9BLAS_ERR, "GPU execution error (kcl)");
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
    if (st < 0) FAILC(GEN9BLAS_ERR, "GPU execution error (%d)", st);
    return 0;
#endif
}

// GEMM kernel events for time measurement (launch(..., measure=1)); times are read after wait_gpu().
cl_event evs[256];
int nev;
double event_times(void) {
    double s = 0;
    if (!profile) { for (int i = 0; i < nev; i++) clReleaseEvent(evs[i]); nev = 0; return -1; }   // without profiling
    for (int i = 0; i < nev; i++) { cl_ulong t0, t1;
        clGetEventProfilingInfo(evs[i], CL_PROFILING_COMMAND_START, 8, &t0, NULL); clGetEventProfilingInfo(evs[i], CL_PROFILING_COMMAND_END, 8, &t1, NULL);
        s += (t1 - t0) * 1e-6; clReleaseEvent(evs[i]); }
    nev = 0; return s;
}

int ensure_buf(cl_mem *b, size_t *size, size_t needed) {
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
    if (!usm_alloc && gen9blas_init()) return NULL;   // initializes on first use
    void *p = usm_alloc(ctx, NULL, bytes ? bytes : 1, 4096, &err);
    if (err != CL_SUCCESS) { snprintf(errbuf, sizeof errbuf, "clHostMemAllocINTEL(%zu) = %d", bytes, err); gst = GEN9BLAS_ERR_OUT_OF_MEMORY; return NULL; }
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
cl_int set_arg(cl_kernel k, int i, bf_t b) { return b.p ? usm_arg(k, i, b.p) : clSetKernelArg(k, i, sizeof b.m, &b.m); }

// USM work buffer (copy of non-USM host memory); growing it waits for the GPU to finish (blocking free).
// Operation boundary for kcl_userptr (regions of one BLAS function sharing pages -> one object).
#ifndef GEN9BLAS_NEO
static int no_userptr = -1;
#endif
work_t uA, uB, uC;
int force_copy;   // per call: copy through the CPU instead of userptr (gen9blas_pack_b retry after a userptr failure)
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
int get_obj(const void *ptr, size_t bytes, work_t *u, int copy_in, int writes, bf_t *result, int *zc) {
    if (usm_contains(ptr, bytes)) { *result = (bf_t){ NULL, (void *)ptr }; *zc = 1; return 0; }
#ifndef GEN9BLAS_NEO
    if (no_userptr < 0) no_userptr = getenv("GEN9BLAS_NO_USERPTR") != NULL;   // comparisons: copy as before
    cl_mem m;
    if (!no_userptr && !force_copy && (m = kcl_userptr(ptr, bytes, writes))) { *result = BM(m); *zc = 1; return 0; }
#else
    (void)writes;
#endif
    if (wait_gpu() || ensure_work(u, bytes)) return ERR_RET;
    if (copy_in) memcpy(u->p, ptr, bytes);
    *result = (bf_t){ NULL, u->p }; *zc = 0; return 0;
}

// Zero-copy only (USM or userptr), no wait and no copy: 0 = *result set, 1 = would need a copy.
int get_obj_zc(const void *ptr, size_t bytes, int writes, bf_t *result) {
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

// Scalar (alpha, beta) for level 3 kernels: double for FP64, float otherwise.
cl_int arg_scalar(const type_t *r, cl_kernel k, int i, double v) {
    if (r->es == 8) return clSetKernelArg(k, i, sizeof v, &v);
    float f = (float)v; return clSetKernelArg(k, i, sizeof f, &f);
}

// End of a call: sleeping wait, then copy of the result to host memory (when it went through a work buffer).
// UHD 620 shares the LLC with the CPU, and NEO flushes the GPU cache at the end of a job: a result in USM is visible without map/unmap.
int to_host(bf_t x, int zc, size_t bytes, void *host) {
    if (wait_gpu()) return ERR_RET;
    if (!zc) memcpy(host, x.p, bytes);
    return 0;
}

// Packing kernel and grid (3D: last dimension = batch). A: (Kp, Mp) / A_t: (Mp, Kp) / A_q: (Kp/8·4, Mp/16);
// B: (Np, Kp) / B_t: (Kp, Np) / B_w2: (Np/2, Kp). The k dimension is always in g[0] (A, B_t) or g[1] (A_t, B, B_w2).
cl_kernel pack_kernel(type_t *r, int roleB, int trans, size_t Rp, size_t Kp, size_t cnt, size_t g[3]) {
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

double gen9blas_last_time_ms(void) { return last_ms; }
int gen9blas_last_zero_copy(void) { return last_zc; }
const char *gen9blas_error(void) { return errbuf; }
int gen9blas_last_status(void) { return gst; }
int lib_fail(int code, const char *fmt, ...) {   // error text + status for units without the macros (attn.c)
    va_list ap; va_start(ap, fmt); vsnprintf(errbuf, sizeof errbuf, fmt, ap); va_end(ap);
    gst = code; return code;
}
