// attn_kcl: the hand-scheduled attention kernel (asm/attn_gen.py, *_tok binaries: Q in llama.cpp's layout) run the
// way the ggml backend would run it: through kcl (lib/kcl.c, i915 without NEO), with Q, K, V and O in ordinary
// process memory (i915 userptr, kcl_userptr cache) laid out like llama.cpp's KV cache (docs/attention-gpu.md section 10).
//
// Usage: attn_kcl <H> <HKV> <D> <n_q> <n_k> <RQ> <NK>
// Layout (llama.cpp 2cbe0ec, src/llama-kv-cache.cpp, src/llama-graph.cpp build_attn_mha):
//   - one host buffer for all layers, allocated like ggml's CPU buffer (posix_memalign 64 B, cleared), tensors in
//     creation order k_l0, v_l0, k_l1, v_l1, ...; each tensor KV_SIZE rows of HKV*D F16 values (V not transposed
//     with flash attention); the attention reads the view of the first n_k rows (n_kv, padded to 256 by llama.cpp);
//   - Q F32 [D, n_q, H] after ggml_permute(0,2,1,3) of Qcur [D, H, n_q]: memory Q[t][h][d]; O F32 [D, H, n_q]:
//     memory O[t][h][d]; both in one compute-buffer-like allocation reused by every layer.
// Per layer, like the graph (one op at a time): kcl_new_operation, userptr regions K, V, Q, O, k_kt (K -> Kt, an own
// GPU buffer with NK*D*16 B padding) and k_attn, one submission, wait.
// Environment:
//   BIN       program in the kernel table (default attn_qwen_tok.bin for D 128, attn_llama_tok.bin for D 64)
//   KV_SIZE   rows per layer tensor (default n_k); LAYERS (default 28); PASSES timed passes after pass 0 (default 5)
//   REG       userptr region of K/V: "tensor" (whole layer tensor, KV_SIZE rows; default) or "view" (n_k rows)
//   LOCAL     work-group size of k_attn (default 64: +4 % Qwen / +1.5 % Llama over 128); KT_LOCAL of k_kt (default 256)
//   STEP      reference sampling as in bench/attn (default 7; 1 = whole output; 0 = no check); CHECK_LAYERS
//             comma-separated layers to check (default first and last)
//   QSIGMA    score scale as in bench/attn (default 5)
//   VIS0      causal binaries (*_tokc, docs/attention-gpu.md section 11): token t sees min(n_k, VIS0 + DVIS*t) keys
//             (DVIS default 1); default binary attn_*_tokc.bin; FLOP and the FP64 reference count visible keys only
// Output: pass 0 (first use: userptr pinning of every layer's pages) separately from the steady state; kernel time
// from GPU timestamps (PIPE_CONTROL, 12 MHz), wall time per layer incl. submission and wait, CPU time, GPU clock
// sampled every 2 ms by a helper thread, error vs FP64.
#define _GNU_SOURCE
#include "kcl.h"
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CHECK(x) do { cl_int e_ = (x); if (e_ != CL_SUCCESS) { fprintf(stderr, "%s:%d: %s = %d\n", __FILE__, __LINE__, #x, e_); exit(1); } } while (0)

static double now_ms(clockid_t c) { struct timespec t; clock_gettime(c, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

// ---- GPU clock sampler ----
static volatile int sampling, sampler_stop; static long fsum, fcnt;
static int read_freq(void) {
    FILE *f = fopen("/sys/class/drm/card0/gt_act_freq_mhz", "r"); int v = -1;
    if (!f) f = fopen("/sys/class/drm/card1/gt_act_freq_mhz", "r");
    if (f) { if (fscanf(f, "%d", &v) != 1) v = -1; fclose(f); }
    return v;
}
static void *sampler(void *a) {
    (void)a;
    while (!sampler_stop) {
        if (sampling) { int f = read_freq(); if (f > 0) { fsum += f; fcnt++; } }
        struct timespec ts = {0, 2000000}; nanosleep(&ts, NULL);
    }
    return NULL;
}

// ---- data as in bench/attn.c ----
static uint16_t f2h(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000; int e = (int)((x >> 23) & 0xff) - 127 + 15; uint32_t m = x & 0x7fffff;
    if (e <= 0) {
        if (e < -10) return sign;
        m |= 0x800000; int sh = 14 - e; uint32_t hm = m >> sh, rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (hm & 1))) hm++;
        return sign | hm;
    }
    uint32_t hm = m >> 13, rem = m & 0x1fff;
    uint32_t h = sign | (e << 10) | hm;
    if (rem > 0x1000 || (rem == 0x1000 && (hm & 1))) h++;
    return h;
}
static double h2d(uint16_t h) {
    int e = (h >> 10) & 0x1f, m = h & 0x3ff; double v;
    if (e == 0) v = ldexp(m, -24); else v = ldexp(m | 0x400, e - 25);
    return (h & 0x8000) ? -v : v;
}
static uint64_t rng = 0x9E3779B97F4A7C15ull;
static double urand(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return ((rng >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
static double nrand(void) { return sqrt(-2 * log(urand())) * cos(6.283185307179586 * urand()); }

static void *ggml_like_alloc(size_t n) {   // ggml_aligned_malloc (ggml.c): posix_memalign with 64 B alignment
    void *p = NULL;
    if (posix_memalign(&p, 64, n)) { perror("posix_memalign"); exit(1); }
    return p;
}

static int cmp_d(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static double median(double *v, int n) { qsort(v, n, sizeof *v, cmp_d); return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]); }

int main(int argc, char **argv) {
    if (argc < 8) { fprintf(stderr, "usage: %s H HKV D n_q n_k RQ NK\n", argv[0]); return 2; }
    int H = atoi(argv[1]), HKV = atoi(argv[2]), D = atoi(argv[3]), nq = atoi(argv[4]), nk = atoi(argv[5]);
    int RQ = atoi(argv[6]), NK = atoi(argv[7]), G = H / HKV;
    int vis0 = getenv("VIS0") ? atoi(getenv("VIS0")) : 0, dvis = getenv("DVIS") ? atoi(getenv("DVIS")) : 1;
    const char *bin = getenv("BIN") ? getenv("BIN") : D == 128 ? (vis0 ? "attn_qwen_tokc.bin" : "attn_qwen_tok.bin")
                                                              : (vis0 ? "attn_llama_tokc.bin" : "attn_llama_tok.bin");
    int kv_size = getenv("KV_SIZE") ? atoi(getenv("KV_SIZE")) : nk;
    int L = getenv("LAYERS") ? atoi(getenv("LAYERS")) : 28, passes = getenv("PASSES") ? atoi(getenv("PASSES")) : 5;
    int reg_view = getenv("REG") && !strcmp(getenv("REG"), "view");
    int step = getenv("STEP") ? atoi(getenv("STEP")) : 7;
    double qsigma = getenv("QSIGMA") ? atof(getenv("QSIGMA")) : 5.0;
    if (nk <= 0 || nq <= 0 || (nq * G) % RQ || nk % (8 * NK) || kv_size < nk || L < 1 || passes < 1) {
        fprintf(stderr, "need n_q > 0, n_k > 0, n_q*G divisible by RQ (%d), n_k divisible by 8*NK (%d), KV_SIZE >= n_k\n", RQ, 8 * NK);
        return 2;
    }
    size_t global = (size_t)HKV * (nq * G / RQ) * 8;
    size_t local = getenv("LOCAL") ? strtoull(getenv("LOCAL"), 0, 0) : 64;   // section 10: 64 > 128 by 4 % (Qwen)
    if (!getenv("LOCAL")) while (local > 8 && global % local) local /= 2;     // as lib/gen9blas_attn.c
    if (local == 0 || local % 8 || global % local) {
        fprintf(stderr, "LOCAL=%zu: must be a multiple of 8 (full subgroups) dividing the global size %zu\n", local, global);
        return 2;
    }
    size_t gkt = (size_t)nk * HKV * D, lkt = getenv("KT_LOCAL") ? strtoull(getenv("KT_LOCAL"), 0, 0) : 256;
    if (lkt == 0 || gkt % lkt) { fprintf(stderr, "KT_LOCAL=%zu must divide %zu\n", lkt, gkt); return 2; }

    // Host memory like llama.cpp: one KV buffer for all layers, cleared (llama_kv_cache clears its buffers), the
    // first n_k rows of every layer written (the CPU's set_rows), Q and O in a compute-buffer-like allocation.
    const size_t row = (size_t)HKV * D * 2, S = (size_t)kv_size * row;   // bytes per row / per layer tensor
    uint8_t *kv = ggml_like_alloc(2 * L * S);
    memset(kv, 0, 2 * L * S);
    for (int l = 0; l < L; l++)
        for (int w = 0; w < 2; w++) {
            uint16_t *t = (uint16_t *)(kv + (2 * (size_t)l + w) * S);
            for (size_t i = 0; i < (size_t)nk * HKV * D; i++) t[i] = f2h((float)nrand());
        }
    const size_t qn = (size_t)nq * H * D;
    float *QO = ggml_like_alloc(2 * qn * 4), *Q = QO, *O = QO + qn;   // Q and O adjacent (one compute buffer)
    for (size_t i = 0; i < qn; i++) Q[i] = (float)(qsigma * nrand());
    memset(O, 0, qn * 4);

    setenv("GEN9BLAS_PROFILE", "1", 0);   // kcl: timestamp buffer for event profiling
    cl_platform_id plat; cl_device_id dev; cl_int err;
    CHECK(clGetPlatformIDs(1, &plat, NULL)); CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL));
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err); CHECK(err);
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err); CHECK(err);
    cl_program prog;
    if (kcl_program(bin, NULL, &prog) != CL_SUCCESS) { fprintf(stderr, "no program %s in the kernel table\n", bin); return 1; }
    cl_kernel kkt = clCreateKernel(prog, "k_kt", &err); CHECK(err);
    cl_kernel ka = clCreateKernel(prog, "k_attn", &err); CHECK(err);
    const size_t ktpad = (size_t)NK * D * 16;
    cl_mem bKt = clCreateBuffer(ctx, CL_MEM_READ_WRITE, S + ktpad, NULL, &err); CHECK(err);
    float qscale = (float)(1.0 / sqrt((double)D) * 1.4426950408889634);

    printf("# %s H=%d HKV=%d D=%d nq=%d nk=%d RQ=%d NK=%d | KV_SIZE=%d LAYERS=%d REG=%s local=%zu kt_local=%zu | KV buffer %.1f MB"
           " at page offset %lu, Q/O page offset %lu | VIS0=%d DVIS=%d\n", bin, H, HKV, D, nq, nk, RQ, NK, kv_size, L,
           reg_view ? "view" : "tensor", local, lkt, 2.0 * L * S / 1e6, (unsigned long)((uintptr_t)kv & 4095),
           (unsigned long)((uintptr_t)QO & 4095), vis0, dvis);

    pthread_t th; pthread_create(&th, NULL, sampler, NULL);
    double flop = 4.0 * nq * nk * H * D;
    if (vis0) { flop = 0; for (int t = 0; t < nq; t++) { long v = vis0 + (long)dvis * t; flop += 4.0 * H * D * (v < nk ? v : nk); } }
    const int P = passes + 1;
    double *t_kt = calloc((size_t)P * L, sizeof(double)), *t_a = calloc((size_t)P * L, sizeof(double));
    double *t_w = calloc((size_t)P * L, sizeof(double)), *p_wall = calloc(P, sizeof(double)), *p_cpu = calloc(P, sizeof(double));
    long fs = 0, fc = 0;

    // one layer: userptr regions, arguments, k_kt + k_attn, one submission, wait
    #define RUN_LAYER(l, ev1, ev2) do { \
        const uint8_t *Kl = kv + 2 * (size_t)(l) * S, *Vl = Kl + S; const size_t kb = reg_view ? (size_t)nk * row : S; \
        kcl_new_operation(); \
        cl_mem mK = kcl_userptr(Kl, kb, 0), mV = kcl_userptr(Vl, kb, 0), mQ = kcl_userptr(Q, qn * 4, 0), mO = kcl_userptr(O, qn * 4, 1); \
        if (!mK || !mV || !mQ || !mO) { fprintf(stderr, "kcl_userptr failed\n"); exit(1); } \
        CHECK(clSetKernelArg(kkt, 0, sizeof mK, &mK)); CHECK(clSetKernelArg(kkt, 1, sizeof bKt, &bKt)); \
        CHECK(clSetKernelArg(kkt, 2, sizeof nk, &nk)); \
        CHECK(clSetKernelArg(ka, 0, sizeof mQ, &mQ)); CHECK(clSetKernelArg(ka, 1, sizeof bKt, &bKt)); \
        CHECK(clSetKernelArg(ka, 2, sizeof mV, &mV)); CHECK(clSetKernelArg(ka, 3, sizeof mO, &mO)); \
        CHECK(clSetKernelArg(ka, 4, sizeof nq, &nq)); CHECK(clSetKernelArg(ka, 5, sizeof nk, &nk)); \
        CHECK(clSetKernelArg(ka, 6, sizeof qscale, &qscale)); \
        if (vis0) { CHECK(clSetKernelArg(ka, 7, sizeof vis0, &vis0)); CHECK(clSetKernelArg(ka, 8, sizeof dvis, &dvis)); } \
        CHECK(clEnqueueNDRangeKernel(q, kkt, 1, NULL, &gkt, &lkt, 0, NULL, ev1)); \
        CHECK(clEnqueueNDRangeKernel(q, ka, 1, NULL, &global, &local, 0, NULL, ev2)); \
        CHECK(kcl_wait()); \
    } while (0)

    for (int p = 0; p < P; p++) {
        double w0 = now_ms(CLOCK_MONOTONIC), c0 = now_ms(CLOCK_THREAD_CPUTIME_ID);
        if (p > 0) { fsum = fcnt = 0; sampling = 1; }
        for (int l = 0; l < L; l++) {
            cl_event e1, e2;
            double a = now_ms(CLOCK_MONOTONIC);
            RUN_LAYER(l, &e1, &e2);
            t_w[p * L + l] = now_ms(CLOCK_MONOTONIC) - a;
            cl_ulong s1, f1, s2, f2;
            CHECK(clGetEventProfilingInfo(e1, CL_PROFILING_COMMAND_START, sizeof s1, &s1, NULL));
            CHECK(clGetEventProfilingInfo(e1, CL_PROFILING_COMMAND_END, sizeof f1, &f1, NULL));
            CHECK(clGetEventProfilingInfo(e2, CL_PROFILING_COMMAND_START, sizeof s2, &s2, NULL));
            CHECK(clGetEventProfilingInfo(e2, CL_PROFILING_COMMAND_END, sizeof f2, &f2, NULL));
            t_kt[p * L + l] = (f1 - s1) * 1e-6; t_a[p * L + l] = (f2 - s2) * 1e-6;
            clReleaseEvent(e1); clReleaseEvent(e2);
        }
        p_wall[p] = now_ms(CLOCK_MONOTONIC) - w0; p_cpu[p] = now_ms(CLOCK_THREAD_CPUTIME_ID) - c0;
        if (p > 0) { sampling = 0; fs += fsum; fc += fcnt; }
    }
    sampler_stop = 1; pthread_join(th, NULL);

    // per pass sums
    double sa[64], skt[64], sw[64];
    for (int p = 0; p < P && p < 64; p++) {
        sa[p] = skt[p] = sw[p] = 0;
        for (int l = 0; l < L; l++) { sa[p] += t_a[p * L + l]; skt[p] += t_kt[p * L + l]; sw[p] += t_w[p * L + l]; }
    }
    double best_a = 1e30, best_kt = 1e30;   // best single layer (comparable with bench/attn's best of 5)
    for (int i = L; i < P * L; i++) { if (t_a[i] < best_a) best_a = t_a[i]; if (t_kt[i] < best_kt) best_kt = t_kt[i]; }
    int np = P - 1 < 63 ? P - 1 : 63;
    double ma[64], mkt[64], mw[64], mpw[64], mpc[64];
    for (int i = 0; i < np; i++) { ma[i] = sa[i + 1]; mkt[i] = skt[i + 1]; mw[i] = sw[i + 1]; mpw[i] = p_wall[i + 1]; mpc[i] = p_cpu[i + 1]; }
    double A = median(ma, np), KT = median(mkt, np), W = median(mw, np), PW = median(mpw, np), PC = median(mpc, np);
    printf("pass 0 (first use): wall %.1f ms, CPU (main thread) %.1f ms, kernels attn %.1f + kt %.2f ms -> first-use extra %.1f ms wall"
           " over the steady pass (%.2f ms per layer)\n", p_wall[0], p_cpu[0], sa[0], skt[0], p_wall[0] - PW, (p_wall[0] - PW) / L);
    // host part of a layer = wall - kernel times (userptr pinning happens in the first execbuf of each object)
    double h0 = 0, hs[64];
    for (int l = 0; l < L; l++) h0 += t_w[l] - t_kt[l] - t_a[l];
    for (int i = 0; i < np; i++) { hs[i] = 0; for (int l = 0; l < L; l++) hs[i] += t_w[(i + 1) * L + l] - t_kt[(i + 1) * L + l] - t_a[(i + 1) * L + l]; }
    double HS = median(hs, np);
    printf("host part (wall - kernels) per layer: pass 0 %.3f ms, steady %.3f ms -> first use %.3f ms per layer;"
           " pass 0 layers 0-3: wall/attn", h0 / L, HS / L, (h0 - HS) / L);
    for (int l = 0; l < L && l < 4; l++) printf(" %.2f/%.2f", t_w[l], t_a[l]);
    printf(" ms; steady median layer wall %.2f ms\n", W / L);
    printf("steady (median of %d passes x %d layers): attn %.2f ms/layer, kt %.3f ms/layer, wall %.2f ms/layer, pass CPU (main thread) %.1f ms"
           " | attn %.1f GFLOPS, incl. kt %.1f, incl. kt + submit/wait %.1f | best layer attn %.3f ms = %.1f GFLOPS"
           " (kt %.3f) | GPU~%ld MHz\n", np, L, A / L, KT / L, W / L, PC, flop * L / (A * 1e-3) / 1e9,
           flop * L / ((A + KT) * 1e-3) / 1e9, flop * L / (W * 1e-3) / 1e9, best_a, flop / (best_a * 1e-3) / 1e9,
           best_kt, fc ? fs / fc : -1);
    printf("steady passes (attn sum, ms):");
    for (int i = 1; i < P; i++) printf(" %.1f", sa[i]);
    printf("\n");

    // error vs FP64 on the layers in CHECK_LAYERS
    if (step > 0) {
        char lst[256]; snprintf(lst, sizeof lst, "%s", getenv("CHECK_LAYERS") ? getenv("CHECK_LAYERS") : "");
        int lay[64], nl = 0;
        if (!lst[0]) { lay[nl++] = 0; if (L > 1) lay[nl++] = L - 1; }
        else for (char *t = strtok(lst, ","); t && nl < 64; t = strtok(NULL, ",")) lay[nl++] = atoi(t);
        double *s = malloc(nk * sizeof(double)), *o = malloc(D * sizeof(double));
        double *Kd = malloc((size_t)nk * HKV * D * sizeof(double)), *Vd = malloc((size_t)nk * HKV * D * sizeof(double));
        for (int li = 0; li < nl; li++) {
            int l = lay[li];
            memset(O, 0xff, qn * 4);   // NaN: rows the kernel does not write show up
            RUN_LAYER(l, NULL, NULL);
            const uint16_t *K = (const uint16_t *)(kv + 2 * (size_t)l * S), *V = (const uint16_t *)(kv + (2 * (size_t)l + 1) * S);
            for (size_t i = 0; i < (size_t)nk * HKV * D; i++) { Kd[i] = h2d(K[i]); Vd[i] = h2d(V[i]); }
            double maxerr = 0, maxref = 0; int nan = 0, nrows = 0;
            for (int t = 0; t < nq; t = (t + step < nq || t == nq - 1) ? t + step : nq - 1) {
                for (int h = 0; h < H; h++) {
                    const float *qr = Q + ((size_t)t * H + h) * D; int hkv = h / G; double mx = -1e300, lsum = 0;
                    long nvl = vis0 ? vis0 + (long)dvis * t : nk; const int nv = nvl < nk ? (int)nvl : nk;
                    for (int j = 0; j < nv; j++) {
                        const double *kr = Kd + ((size_t)j * HKV + hkv) * D; double a = 0;
                        for (int d = 0; d < D; d++) a += (double)qr[d] * kr[d];
                        s[j] = a / sqrt((double)D); if (s[j] > mx) mx = s[j];
                    }
                    for (int d = 0; d < D; d++) o[d] = 0;
                    for (int j = 0; j < nv; j++) {
                        double pj = exp(s[j] - mx); lsum += pj; const double *vr = Vd + ((size_t)j * HKV + hkv) * D;
                        for (int d = 0; d < D; d++) o[d] += pj * vr[d];
                    }
                    const float *og = O + ((size_t)t * H + h) * D;
                    for (int d = 0; d < D; d++) {
                        double ref = o[d] / lsum, e = fabs(og[d] - ref);
                        if (!(e == e)) nan = 1;
                        if (e > maxerr) maxerr = e;
                        if (fabs(ref) > maxref) maxref = fabs(ref);
                    }
                }
                nrows++;
                if (t == nq - 1) break;
            }
            printf("layer %d: err %.2e%s (tokens %d/%d, STEP %d)\n", l, maxerr / maxref, nan ? " NaN" : "", nrows, nq, step);
        }
    }
    return 0;
}
