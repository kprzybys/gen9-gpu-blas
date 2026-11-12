// Measurement program for custom HGEMM kernels: hgemm <file.cl|file.bin> <kernel> <N> [compiler options]
// Square N×N matrices, row-major, FP16, values in [-1, 1]. A and B as 2D images (CL_R, CL_HALF_FLOAT) over buffers.
// Grid: global = (N/32 * 16, N/16), local = (16, LY), LY from env LY (default 4).
// Prints: best of 5 time, GFLOPS, relative error on 256 elements against double (as in gemm_clblast).
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <clblast_half.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Waiting for kernel completion. WAIT=sleep: polling every 0.5 ms with the thread sleeping
// (instead of clWaitForEvents, which in the NEO driver spins actively on the CPU).
static void wait_event(cl_event ev) {
    if (getenv("WAIT") && !strcmp(getenv("WAIT"), "sleep")) {
        cl_int st;
        do {
            struct timespec ts = {0, 500000}; nanosleep(&ts, NULL);
            clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof st, &st, NULL);
        } while (st > CL_COMPLETE);
    } else {
        clWaitForEvents(1, &ev);
    }
}

#define CHECK(x) do { cl_int e_ = (x); if (e_ != CL_SUCCESS) { fprintf(stderr, "%s:%d: %s = %d\n", __FILE__, __LINE__, #x, e_); exit(1); } } while (0)

static unsigned char *read_file(const char *p, size_t *len) {
    FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
    fseek(f, 0, SEEK_END); *len = ftell(f); rewind(f);
    unsigned char *s = malloc(*len + 1); if (fread(s, 1, *len, f) != *len) exit(1); s[*len] = 0; fclose(f); return s;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s file.cl|.bin kernel N [options]\n", argv[0]); return 2; }
    const char *file = argv[1], *kname = argv[2]; int N = atoi(argv[3]);
    const char *opts = argc > 4 ? argv[4] : "";
    size_t LY = getenv("LY") ? strtoul(getenv("LY"), 0, 0) : 4;
    int PAD = getenv("PAD") ? atoi(getenv("PAD")) : 0;   // row padding (elements): ld = N + PAD
    int LD = N + PAD;
    size_t nn = (size_t)N * N, nl = (size_t)N * LD;

    cl_platform_id plat; cl_device_id dev; cl_int err;
    CHECK(clGetPlatformIDs(1, &plat, NULL));
    CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL));
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err); CHECK(err);
    cl_queue_properties qp[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, qp, &err); CHECK(err);

    size_t flen; unsigned char *src = read_file(file, &flen);
    cl_program prog;
    if (flen > 4 && !strcmp(file + strlen(file) - 4, ".bin")) {
        cl_int bst; prog = clCreateProgramWithBinary(ctx, 1, &dev, &flen, (const unsigned char **)&src, &bst, &err); CHECK(err); CHECK(bst);
    } else {
        prog = clCreateProgramWithSource(ctx, 1, (const char **)&src, NULL, &err); CHECK(err);
    }
    if (clBuildProgram(prog, 1, &dev, opts, NULL, NULL) != CL_SUCCESS) {
        char log[65536]; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof log, log, NULL);
        fprintf(stderr, "build error:\n%s\n", log); return 1;
    }
    cl_kernel k = clCreateKernel(prog, kname, &err); CHECK(err);

    cl_half *hA = calloc(nl, 2), *hB = calloc(nl, 2), *hC = calloc(nl, 2);
    double *rA = malloc(nn * 8), *rB = malloc(nn * 8);
    srand(1234);
    for (size_t i = 0; i < 2 * nn; i++) {
        float v = 2.0f * rand() / RAND_MAX - 1.0f; cl_half x = FloatToHalf(v);
        size_t j = i % nn, at = j / N * LD + j % N;
        if (i < nn) { hA[at] = x; rA[j] = HalfToFloat(x); } else { hB[at] = x; rB[j] = HalfToFloat(x); }
    }
    cl_mem bA = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nl * 2, hA, &err); CHECK(err);
    cl_mem bB = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nl * 2, hB, &err); CHECK(err);
    cl_mem bC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nl * 2, NULL, &err); CHECK(err);
    cl_image_format fmt = { CL_R, CL_HALF_FLOAT };
    cl_image_desc d = { CL_MEM_OBJECT_IMAGE2D, (size_t)N, (size_t)N, 1, 1, (size_t)LD * 2, 0, 0, 0, { NULL } };
    int pack = getenv("PACK") ? atoi(getenv("PACK")) : 0;
    double t_pack = 0;
    cl_mem srcA = bA, srcB = bB;
    if (pack) {
        // packing kernels always from the hgemm.cl source (also when the main kernel is from .bin)
        size_t pl; unsigned char *ps = read_file(getenv("PACK_CL") ? getenv("PACK_CL") : "hgemm.cl", &pl);
        cl_program pp = clCreateProgramWithSource(ctx, 1, (const char **)&ps, NULL, &err); CHECK(err);
        CHECK(clBuildProgram(pp, 1, &dev, "", NULL, NULL));
        // PACK=2: A2 layout (pack_a2) for the hand-written asm loop
        // PACK=3: A2 layout as in PACK=2, but fast kernels pack_a2_16 / pack_b16 (16 elements per work-item)
        // PACK=6: pk_a2 + pk_bq (hgemm_16x32q), PACK=7: pk_a3 + pk_b16 (hgemm_16x16kc_b); batched signature
        cl_kernel ka = clCreateKernel(pp, pack == 7 ? "pk_a3" : pack == 6 ? "pk_a2" : pack == 5 ? "pack_a3_pad" : pack == 4 ? "pack_a2_g" : pack == 3 ? "pack_a2_16" : pack == 2 ? "pack_a2" : "pack_a", &err); CHECK(err);
        // PACK=4: pack_a2_g (full 32 B rows) + pack_b16
        cl_kernel kb = clCreateKernel(pp, pack == 7 ? "pk_b16" : pack == 6 ? "pk_bq" : pack == 5 ? "pack_b_pad" : pack >= 3 ? "pack_b16" : "pack_b", &err); CHECK(err);
        srcA = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nn * 2 + 4096, NULL, &err); CHECK(err);
        srcB = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nn * 2 + 4096, NULL, &err); CHECK(err);
        CHECK(clSetKernelArg(ka, 0, sizeof bA, &bA)); CHECK(clSetKernelArg(ka, 1, sizeof srcA, &srcA));
        CHECK(clSetKernelArg(ka, 2, sizeof N, &N)); CHECK(clSetKernelArg(ka, 3, sizeof LD, &LD));
        CHECK(clSetKernelArg(kb, 0, sizeof bB, &bB)); CHECK(clSetKernelArg(kb, 1, sizeof srcB, &srcB));
        CHECK(clSetKernelArg(kb, 2, sizeof N, &N)); CHECK(clSetKernelArg(kb, 3, sizeof LD, &LD));
        size_t g2[2] = { (size_t)N, (size_t)N }, g16[2] = { (size_t)N / 16, (size_t)N };
        size_t ga4[1] = { nn / 16 };
        if (pack >= 6) { cl_long z = 0;
            CHECK(clSetKernelArg(ka, 6, sizeof z, &z)); CHECK(clSetKernelArg(ka, 7, sizeof N, &N));
            CHECK(clSetKernelArg(kb, 6, sizeof z, &z)); CHECK(clSetKernelArg(kb, 7, sizeof N, &N)); }
        if (pack >= 3) {   // signatures: (X, Xp, rows, cols, ld, Kp)
            CHECK(clSetKernelArg(ka, 2, sizeof N, &N)); CHECK(clSetKernelArg(ka, 3, sizeof N, &N));
            CHECK(clSetKernelArg(ka, 4, sizeof LD, &LD)); CHECK(clSetKernelArg(ka, 5, sizeof N, &N));
            CHECK(clSetKernelArg(kb, 2, sizeof N, &N)); CHECK(clSetKernelArg(kb, 3, sizeof N, &N));
            CHECK(clSetKernelArg(kb, 4, sizeof LD, &LD)); CHECK(clSetKernelArg(kb, 5, sizeof N, &N));
        }
        for (int r = 0; r < 3; r++) {   // last pass is the measured one
            cl_event e1, e2; cl_ulong a0, a1, b0, b1;
            if (pack == 4) CHECK(clEnqueueNDRangeKernel(q, ka, 1, NULL, ga4, NULL, 0, NULL, &e1));
            else CHECK(clEnqueueNDRangeKernel(q, ka, 2, NULL, pack == 3 ? g16 : g2, NULL, 0, NULL, &e1));
            CHECK(clEnqueueNDRangeKernel(q, kb, 2, NULL, (pack == 3 || pack == 4) ? g16 : g2, NULL, 0, NULL, &e2)); CHECK(clFinish(q));
            clGetEventProfilingInfo(e1, CL_PROFILING_COMMAND_START, 8, &a0, NULL); clGetEventProfilingInfo(e1, CL_PROFILING_COMMAND_END, 8, &a1, NULL);
            clGetEventProfilingInfo(e2, CL_PROFILING_COMMAND_START, 8, &b0, NULL); clGetEventProfilingInfo(e2, CL_PROFILING_COMMAND_END, 8, &b1, NULL);
            t_pack = ((a1 - a0) + (b1 - b0)) * 1e-9;
            if (getenv("PACK_T")) fprintf(stderr, "packing A %.3f ms, B %.3f ms\n", (a1 - a0) * 1e-6, (b1 - b0) * 1e-6);
        }
    }
    // PACK=1: kernel gets packed buffers; otherwise 2D images over row-major buffers
    cl_mem iA = srcA, iB = srcB;
    if (!pack) {
        d.buffer = srcA; iA = clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &d, NULL, &err); CHECK(err);
        d.buffer = srcB; iB = clCreateImage(ctx, CL_MEM_READ_ONLY, &fmt, &d, NULL, &err); CHECK(err);
    }

    CHECK(clSetKernelArg(k, 0, sizeof iA, &iA)); CHECK(clSetKernelArg(k, 1, sizeof iB, &iB));
    CHECK(clSetKernelArg(k, 2, sizeof bC, &bC));
    CHECK(clSetKernelArg(k, 3, sizeof N, &N)); CHECK(clSetKernelArg(k, 4, sizeof LD, &LD)); CHECK(clSetKernelArg(k, 5, sizeof N, &N));
    if (getenv("NEW")) {   // *_v kernels: (Ap, Bp, C, M, N, ldc, K, Kfull)
        CHECK(clSetKernelArg(k, 3, sizeof N, &N)); CHECK(clSetKernelArg(k, 4, sizeof N, &N)); CHECK(clSetKernelArg(k, 5, sizeof LD, &LD));
        CHECK(clSetKernelArg(k, 6, sizeof N, &N)); CHECK(clSetKernelArg(k, 7, sizeof N, &N)); }
    // ND1=1: 1D grid, one 16x32 tile per subgroup, LY subgroups per work-group
    int nd1 = getenv("ND1") && atoi(getenv("ND1"));
    size_t gws[2] = { (size_t)N / 32 * 16, (size_t)N / 16 }, lws[2] = { 16, LY };
    int TN = getenv("TN") ? atoi(getenv("TN")) : 32;   // tile width (32: hgemm_16x32*, 16: hgemm_16x16kc)
    if (nd1) { gws[0] = (size_t)(N / TN) * (N / 16) * 16; lws[0] = 16 * LY; gws[1] = lws[1] = 1; }
    if (!nd1 && gws[1] % LY) { fprintf(stderr, "N/16 is not divisible by LY=%zu\n", LY); return 2; }

    double best = 1e30;
    int reps = getenv("REPS") ? atoi(getenv("REPS")) : 6;   // REPS: more passes (e.g. for power measurement)
    for (int r = 0; r < reps; r++) {
        cl_event ev; cl_ulong t0, t1;
        CHECK(clEnqueueNDRangeKernel(q, k, nd1 ? 1 : 2, NULL, gws, lws, 0, NULL, &ev)); CHECK(clFlush(q)); wait_event(ev);
        CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof t0, &t0, NULL));
        CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof t1, &t1, NULL));
        if (r > 0 && (t1 - t0) * 1e-9 < best) best = (t1 - t0) * 1e-9;
        // TRACE=k: every k passes print the average performance of the last k (for long-running tests)
        static double acc_t = 0; acc_t += (t1 - t0) * 1e-9;
        if (getenv("TRACE") && (r + 1) % atoi(getenv("TRACE")) == 0) {
            int tk = atoi(getenv("TRACE"));
            fprintf(stderr, "trace %5d  %7.1f GFLOPS\n", r + 1, 2.0 * N * N * (double)N * tk / acc_t / 1e9); acc_t = 0;
        }
        clReleaseEvent(ev);
    }
    CHECK(clEnqueueReadBuffer(q, bC, CL_TRUE, 0, nl * 2, hC, 0, NULL, NULL));
    if (getenv("DUMP")) {
        for (int r = 0; r < 16; r++) { for (int c = 0; c < 32; c++) printf("%g ", HalfToFloat(hC[(size_t)r * LD + c])); printf("\n"); }
        return 0;
    }
    double maxerr = 0, sumerr = 0; srand(99);
    for (int s = 0; s < 256; s++) {
        size_t i = rand() % N, j = rand() % N; double ref = 0, mag = 0;
        for (int kk = 0; kk < N; kk++) { double t = rA[i * N + kk] * rB[(size_t)kk * N + j]; ref += t; mag += fabs(t); }
        double e = fabs(HalfToFloat(hC[i * LD + j]) - ref) / mag; sumerr += e; if (e > maxerr) maxerr = e;
    }
    printf("%-14s N=%-5d ld=%-5d LY=%-2zu time=%9.3f ms  %7.1f GFLOPS  rel_err: avg=%.2e max=%.2e",
           kname, N, LD, LY, best * 1e3, 2.0 * N * N * (double)N / best / 1e9, sumerr / 256, maxerr);
    if (pack) printf("  | with packing: %.3f ms, %.1f GFLOPS", (best + t_pack) * 1e3, 2.0 * N * N * (double)N / (best + t_pack) / 1e9);
    printf("\n");
    return 0;
}
