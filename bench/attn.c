// attn: standalone test of the fused attention kernel in attn.cl (docs/attention-gpu.md, stage 2).
// Usage: attn <H> <HKV> <D> <n_q> <n_k> <RQ> <NK> [extra build options]   (env CL=<source.cl or binary.bin>)
// QTOK=1: Q in llama.cpp's layout Q[t][h][d] (binaries asm/*_tok.bin, built with -DQTOK); default Q[h][t][d].
// Random Q (float), K, V (half) as llama.cpp flash_attn_ext gives them; one layer, non-causal.
// Prints: build parameters, SIMD width, spill / private bytes, best time of the K transpose and of the attention
// kernel (REPS runs after a warm-up), effective rate 4 * n_q * n_k * H * D / time, GPU clock during the best run,
// and the error max |O - O_ref| / max |O_ref| against an FP64 CPU reference on query tokens 0, STEP, 2*STEP, ...
// and the last one, all heads (default STEP 7: coprime with the 16/32/64 query rows of a work-group, so every
// subgroup position is visited; STEP=1 checks the whole output, use it for acceptance).
// Requirements of the kernels (also of the hand-assembled binaries, asm/attn_gen.py), checked here:
//   n_k > 0 and divisible by 8*NK; n_q*G divisible by RQ; work-group size (LOCAL, default 64 for .bin, 128 for
//   OpenCL C) divisible by 8
//   (full SIMD8 subgroups) and dividing the global size; the K-transposed buffer Kt allocated with KT_PAD extra
//   bytes, default NK*D*16 = one block of 8*NK keys: the sched binaries prefetch the next key block also after the
//   last one (attn_gen.py header, docs/attention-gpu.md section 9).
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(x) do { cl_int e_ = (x); if (e_ != CL_SUCCESS) { fprintf(stderr, "%s:%d: %s = %d\n", __FILE__, __LINE__, #x, e_); exit(1); } } while (0)
#define CL_KERNEL_SPILL_MEM_SIZE_INTEL 0x4109

static char *read_file(const char *p, size_t *len) {
    FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f);
    char *s = malloc(n + 1); if (fread(s, 1, n, f) != (size_t)n) exit(1); s[n] = 0; fclose(f);
    if (len) *len = n;
    return s;
}

static int read_freq(void) {
    FILE *f = fopen("/sys/class/drm/card0/gt_act_freq_mhz", "r"); int v = -1;
    if (!f) f = fopen("/sys/class/drm/card1/gt_act_freq_mhz", "r");
    if (f) { if (fscanf(f, "%d", &v) != 1) v = -1; fclose(f); }
    return v;
}

// float <-> IEEE half, round to nearest even (inputs here are normal, no overflow).
static uint16_t f2h(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000; int e = (int)((x >> 23) & 0xff) - 127 + 15; uint32_t m = x & 0x7fffff;
    if (e <= 0) {   // subnormal half
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

int main(int argc, char **argv) {
    if (argc < 8) { fprintf(stderr, "usage: %s H HKV D n_q n_k RQ NK [build options]\n", argv[0]); return 2; }
    int H = atoi(argv[1]), HKV = atoi(argv[2]), D = atoi(argv[3]), nq = atoi(argv[4]), nk = atoi(argv[5]);
    int RQ = atoi(argv[6]), NK = atoi(argv[7]);
    const char *extra = argc > 8 ? argv[8] : "";
    int G = H / HKV, reps = getenv("REPS") ? atoi(getenv("REPS")) : 5;
    double qsigma = getenv("QSIGMA") ? atof(getenv("QSIGMA")) : 5.0;   // score std = qsigma (K ~ N(0,1))
    const char *src_file = getenv("CL") ? getenv("CL") : "attn.cl";
    if (nk <= 0 || nq <= 0 || (nq * G) % RQ || nk % (8 * NK)) {
        fprintf(stderr, "need n_q > 0, n_k > 0, n_q * G divisible by RQ (%d), n_k divisible by 8 * NK (%d)\n", RQ, 8 * NK);
        return 2;
    }

    // Data: Q[h][t][d] (QTOK=1: Q[t][h][d]) float, K/V[t][hkv][d] half, O[t][h][d] float.
    int qtok = getenv("QTOK") && atoi(getenv("QTOK"));
    size_t nQ = (size_t)H * nq * D, nKV = (size_t)nk * HKV * D, nO = (size_t)nq * H * D;
    float *Q = malloc(nQ * 4), *O = malloc(nO * 4);
    uint16_t *K = malloc(nKV * 2), *V = malloc(nKV * 2);
    for (size_t i = 0; i < nQ; i++) Q[i] = (float)(qsigma * nrand());
    for (size_t i = 0; i < nKV; i++) K[i] = f2h((float)nrand());
    for (size_t i = 0; i < nKV; i++) V[i] = f2h((float)nrand());

    cl_platform_id plat; cl_device_id dev; cl_int err;
    CHECK(clGetPlatformIDs(1, &plat, NULL));
    CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL));
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err); CHECK(err);
    cl_queue_properties qp[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, qp, &err); CHECK(err);

    char opts[1024];
    snprintf(opts, sizeof opts, "-cl-fast-relaxed-math -cl-mad-enable -DD=%d -DG=%d -DHKV=%d -DRQ=%d -DNK=%d %s",
             D, G, HKV, RQ, NK, extra);
    // CL=<file>.bin: a binary from ocloc (hand-patched assembly, asm/attn_gen.py; built for one shape, the
    // -D options are then ignored and must match the arguments).
    size_t src_len; char *src = read_file(src_file, &src_len);
    int is_bin = strlen(src_file) > 4 && !strcmp(src_file + strlen(src_file) - 4, ".bin");
    cl_program prog;
    if (is_bin) {
        cl_int bst;
        prog = clCreateProgramWithBinary(ctx, 1, &dev, &src_len, (const unsigned char **)&src, &bst, &err);
        CHECK(err); CHECK(bst);
    } else {
        prog = clCreateProgramWithSource(ctx, 1, (const char **)&src, NULL, &err); CHECK(err);
    }
    if (clBuildProgram(prog, 1, &dev, opts, NULL, NULL) != CL_SUCCESS) {
        static char log[1 << 20]; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof log, log, NULL);
        fprintf(stderr, "build error:\n%s\n", log); return 1;
    }
    cl_kernel kkt = clCreateKernel(prog, "k_kt", &err); CHECK(err);
    cl_kernel ka = clCreateKernel(prog, "k_attn", &err); CHECK(err);

    // default work-group size: 64 for the asm binaries (+4 % Qwen, +1.5 % Llama over 128, any size 8..96 is as
    // good; docs/attention-gpu.md section 10), 128 for OpenCL C sources (stage 2: 128 > 64 by 10 % for variant A)
    size_t dflt = is_bin ? 64 : 128;
    size_t local = getenv("LOCAL") ? strtoull(getenv("LOCAL"), 0, 0) : dflt;
    size_t global = (size_t)HKV * (nq * G / RQ) * 8;
    if (local == 0 || local % 8 || global % local) {
        if (getenv("LOCAL")) {
            fprintf(stderr, "LOCAL=%zu: must be a multiple of 8 (full subgroups) dividing the global size %zu\n", local, global);
            return 2;
        }
        size_t l = dflt; while (global % l) l -= 8;          // global is a multiple of 8, so this ends at >= 8
        fprintf(stderr, "warning: global size %zu not divisible by 128, work-group size %zu\n", global, l);
        local = l;
    }
    size_t ktpad = getenv("KT_PAD") ? strtoull(getenv("KT_PAD"), 0, 0) : (size_t)NK * D * 16;
    size_t sg = 0; cl_ulong spill = 0, priv = 0;
    clGetKernelSubGroupInfo(ka, dev, CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE, sizeof local, &local, sizeof sg, &sg, NULL);
    clGetKernelWorkGroupInfo(ka, dev, CL_KERNEL_SPILL_MEM_SIZE_INTEL, sizeof spill, &spill, NULL);
    clGetKernelWorkGroupInfo(ka, dev, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof priv, &priv, NULL);

    cl_mem bQ = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nQ * 4, Q, &err); CHECK(err);
    cl_mem bK = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nKV * 2, K, &err); CHECK(err);
    cl_mem bV = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nKV * 2, V, &err); CHECK(err);
    cl_mem bKt = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nKV * 2 + ktpad, NULL, &err); CHECK(err);
    cl_mem bO = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, nO * 4, NULL, &err); CHECK(err);
    float qscale = (float)(1.0 / sqrt((double)D) * 1.4426950408889634);
    CHECK(clSetKernelArg(kkt, 0, sizeof bK, &bK)); CHECK(clSetKernelArg(kkt, 1, sizeof bKt, &bKt));
    CHECK(clSetKernelArg(kkt, 2, sizeof nk, &nk));
    CHECK(clSetKernelArg(ka, 0, sizeof bQ, &bQ)); CHECK(clSetKernelArg(ka, 1, sizeof bKt, &bKt));
    CHECK(clSetKernelArg(ka, 2, sizeof bV, &bV)); CHECK(clSetKernelArg(ka, 3, sizeof bO, &bO));
    CHECK(clSetKernelArg(ka, 4, sizeof nq, &nq)); CHECK(clSetKernelArg(ka, 5, sizeof nk, &nk));
    CHECK(clSetKernelArg(ka, 6, sizeof qscale, &qscale));

    size_t gkt = nKV;
    double best_kt = 1e30, best_a = 1e30; int best_freq = -1;
    for (int r = 0; r < reps + 1; r++) {   // run 0 is the warm-up
        cl_event e1, e2;
        CHECK(clEnqueueNDRangeKernel(q, kkt, 1, NULL, &gkt, NULL, 0, NULL, &e1));
        CHECK(clEnqueueNDRangeKernel(q, ka, 1, NULL, &global, &local, 0, NULL, &e2));
        CHECK(clFlush(q));
        long fsum = 0; int fcnt = 0; cl_int st;
        do {   // sample the GPU clock while the attention kernel runs
            CHECK(clGetEventInfo(e2, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof st, &st, NULL));
            if (st != CL_COMPLETE) { int fr = read_freq(); if (fr > 0) { fsum += fr; fcnt++; } }
            struct timespec ts = {0, 200000}; nanosleep(&ts, NULL);
        } while (st > CL_COMPLETE);
        cl_ulong t0, t1, t2, t3;
        CHECK(clGetEventProfilingInfo(e1, CL_PROFILING_COMMAND_START, sizeof t0, &t0, NULL));
        CHECK(clGetEventProfilingInfo(e1, CL_PROFILING_COMMAND_END, sizeof t1, &t1, NULL));
        CHECK(clGetEventProfilingInfo(e2, CL_PROFILING_COMMAND_START, sizeof t2, &t2, NULL));
        CHECK(clGetEventProfilingInfo(e2, CL_PROFILING_COMMAND_END, sizeof t3, &t3, NULL));
        double ms_kt = (t1 - t0) * 1e-6, ms_a = (t3 - t2) * 1e-6;
        if (r > 0 && ms_kt + ms_a < best_kt + best_a) { best_kt = ms_kt; best_a = ms_a; best_freq = fcnt ? (int)(fsum / fcnt) : -1; }
        clReleaseEvent(e1); clReleaseEvent(e2);
    }
    CHECK(clEnqueueReadBuffer(q, bO, CL_TRUE, 0, nO * 4, O, 0, NULL, NULL));

    // FP64 reference on a sample of query tokens (all heads): every STEP-th token plus the last one.
    int step = getenv("STEP") ? atoi(getenv("STEP")) : 7, nrows = 0;
    if (step < 1) step = 1;
    double maxerr = 0, maxref = 0, *s = malloc(nk * sizeof(double)), *o = malloc(D * sizeof(double));
    double *Kd = malloc(nKV * sizeof(double)), *Vd = malloc(nKV * sizeof(double));
    for (size_t i = 0; i < nKV; i++) { Kd[i] = h2d(K[i]); Vd[i] = h2d(V[i]); }
    int nan = 0;
    for (int t = 0; t < nq; t = (t + step < nq || t == nq - 1) ? t + step : nq - 1) {
        for (int h = 0; h < H; h++) {
            const float *qr = Q + (qtok ? (size_t)t * H + h : (size_t)h * nq + t) * D; int hkv = h / G; double mx = -1e300, l = 0;
            for (int j = 0; j < nk; j++) {
                const double *kr = Kd + ((size_t)j * HKV + hkv) * D; double a = 0;
                for (int d = 0; d < D; d++) a += (double)qr[d] * kr[d];
                s[j] = a / sqrt((double)D); if (s[j] > mx) mx = s[j];
            }
            for (int d = 0; d < D; d++) o[d] = 0;
            for (int j = 0; j < nk; j++) {
                double p = exp(s[j] - mx); l += p; const double *vr = Vd + ((size_t)j * HKV + hkv) * D;
                for (int d = 0; d < D; d++) o[d] += p * vr[d];
            }
            const float *og = O + ((size_t)t * H + h) * D;
            for (int d = 0; d < D; d++) {
                double ref = o[d] / l, e = fabs(og[d] - ref);
                if (!(e == e)) nan = 1;
                if (e > maxerr) maxerr = e;
                if (fabs(ref) > maxref) maxref = fabs(ref);
            }
        }
        nrows++;
        if (t == nq - 1) break;
    }

    double flop = 4.0 * nq * nk * H * D;
    printf("H=%d HKV=%d D=%d nq=%d nk=%d RQ=%d NK=%d %s%s | sg=%zu local=%zu spill=%llu private=%llu ktpad=%zu | kt %.3f ms"
           " attn %.3f ms | attn %.1f GFLOPS, incl. kt %.1f | err %.2e%s (tokens %d/%d, STEP %d) | GPU~%d MHz\n",
           H, HKV, D, nq, nk, RQ, NK, qtok ? "QTOK " : "", extra, sg, local, (unsigned long long)spill, (unsigned long long)priv, ktpad, best_kt,
           best_a, flop / (best_a * 1e-3) / 1e9, flop / ((best_a + best_kt) * 1e-3) / 1e9, maxerr / maxref,
           nan ? " NaN" : "", nrows, nq, step, best_freq);
    return 0;
}
