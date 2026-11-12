// SGEMM measurement program (FP32): sgemm <file.cl|.bin> <kernel> <N> [options]. N×N matrices, row-major, packed on the GPU
// by the pack_af/pack_bf kernels from sgemm.cl (env PACK_CL, default sgemm.cl). 1D grid: one 16×16 tile per subgroup.
// Prints the best of 5 time, GFLOPS (without and with packing) and the relative error on 256 elements (double reference).
// env: WAIT=sleep (wait with sleeping), DUMP=1 (print tile 0 as numbers: for the asm register map).
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(x) do { cl_int e_ = (x); if (e_ != CL_SUCCESS) { fprintf(stderr, "%s:%d: %s = %d\n", __FILE__, __LINE__, #x, e_); exit(1); } } while (0)

static unsigned char *rd(const char *p, size_t *len) {
    FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
    fseek(f, 0, SEEK_END); *len = ftell(f); rewind(f);
    unsigned char *s = malloc(*len + 1); if (fread(s, 1, *len, f) != *len) exit(1); s[*len] = 0; fclose(f); return s;
}
static double run(cl_command_queue q, cl_kernel k, cl_uint dim, size_t *g, size_t *l) {
    cl_event ev; cl_ulong t0, t1;
    CHECK(clEnqueueNDRangeKernel(q, k, dim, NULL, g, l, 0, NULL, &ev)); CHECK(clFlush(q));
    if (getenv("WAIT") && !strcmp(getenv("WAIT"), "sleep")) {
        cl_int st; do { struct timespec ts = {0, 500000}; nanosleep(&ts, NULL);
            clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof st, &st, NULL); } while (st > CL_COMPLETE);
    } else CHECK(clWaitForEvents(1, &ev));
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, 8, &t0, NULL);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, 8, &t1, NULL);
    clReleaseEvent(ev); return (t1 - t0) * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s file kernel N [options]\n", argv[0]); return 2; }
    const char *file = argv[1], *kname = argv[2]; int N = atoi(argv[3]); const char *opts = argc > 4 ? argv[4] : "";
    size_t nn = (size_t)N * N; cl_int err;
    cl_platform_id plat; cl_device_id dev;
    CHECK(clGetPlatformIDs(1, &plat, NULL)); CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL));
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err); CHECK(err);
    cl_queue_properties qp[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, qp, &err); CHECK(err);
    size_t len; unsigned char *src = rd(file, &len); cl_program prog;
    if (len > 4 && !strcmp(file + strlen(file) - 4, ".bin")) {
        cl_int bst; prog = clCreateProgramWithBinary(ctx, 1, &dev, &len, (const unsigned char **)&src, &bst, &err); CHECK(err); CHECK(bst);
    } else { prog = clCreateProgramWithSource(ctx, 1, (const char **)&src, NULL, &err); CHECK(err); }
    if (clBuildProgram(prog, 1, &dev, opts, NULL, NULL) != CL_SUCCESS) {
        char log[65536]; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof log, log, NULL); fprintf(stderr, "%s\n", log); return 1; }
    cl_kernel k = clCreateKernel(prog, kname, &err); CHECK(err);
    unsigned char *ps = rd(getenv("PACK_CL") ? getenv("PACK_CL") : "sgemm.cl", &len);
    cl_program pp = clCreateProgramWithSource(ctx, 1, (const char **)&ps, NULL, &err); CHECK(err); CHECK(clBuildProgram(pp, 1, &dev, "", NULL, NULL));
    int p2 = getenv("PACK2") ? atoi(getenv("PACK2")) : 0;   // PACK2=1: layouts for the asm loop
    // PACK2=2: pkf_a / pkf_b (batched signature, layouts as in pack_af2/bf2)
    cl_kernel ka = clCreateKernel(pp, p2 == 2 ? "pkf_a" : p2 ? "pack_af2" : "pack_af", &err); CHECK(err);
    cl_kernel kb = clCreateKernel(pp, p2 == 2 ? "pkf_b" : p2 ? "pack_bf2" : "pack_bf", &err); CHECK(err);

    float *hA = malloc(nn * 4), *hB = malloc(nn * 4), *hC = malloc(nn * 4);
    srand(1234); for (size_t i = 0; i < nn; i++) hA[i] = 2.0f * rand() / RAND_MAX - 1.0f;
    for (size_t i = 0; i < nn; i++) hB[i] = 2.0f * rand() / RAND_MAX - 1.0f;
    cl_mem bA = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nn * 4, hA, &err); CHECK(err);
    cl_mem bB = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nn * 4, hB, &err); CHECK(err);
    cl_mem pA = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nn * 4 + 4096, NULL, &err); CHECK(err);
    cl_mem pB = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nn * 4 + 4096, NULL, &err); CHECK(err);
    cl_mem bC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nn * 4, NULL, &err); CHECK(err);
    CHECK(clSetKernelArg(ka, 0, 8, &bA)); CHECK(clSetKernelArg(ka, 1, 8, &pA)); CHECK(clSetKernelArg(ka, 2, 4, &N)); CHECK(clSetKernelArg(ka, 3, 4, &N));
    CHECK(clSetKernelArg(kb, 0, 8, &bB)); CHECK(clSetKernelArg(kb, 1, 8, &pB)); CHECK(clSetKernelArg(kb, 2, 4, &N)); CHECK(clSetKernelArg(kb, 3, 4, &N));
    if (p2 == 2) { cl_long z = 0;   // (X, Xp, rows, cols, ld, Kp, sX, Mp)
        CHECK(clSetKernelArg(ka, 2, 4, &N)); CHECK(clSetKernelArg(ka, 3, 4, &N)); CHECK(clSetKernelArg(ka, 4, 4, &N));
        CHECK(clSetKernelArg(ka, 5, 4, &N)); CHECK(clSetKernelArg(ka, 6, 8, &z)); CHECK(clSetKernelArg(ka, 7, 4, &N));
        CHECK(clSetKernelArg(kb, 2, 4, &N)); CHECK(clSetKernelArg(kb, 3, 4, &N)); CHECK(clSetKernelArg(kb, 4, 4, &N));
        CHECK(clSetKernelArg(kb, 5, 4, &N)); CHECK(clSetKernelArg(kb, 6, 8, &z)); CHECK(clSetKernelArg(kb, 7, 4, &N)); }
    size_t g2[2] = { (size_t)N, (size_t)N }; double tp = 0;
    for (int r = 0; r < 3; r++) tp = run(q, ka, 2, g2, NULL) + run(q, kb, 2, g2, NULL);
    CHECK(clSetKernelArg(k, 0, 8, &pA)); CHECK(clSetKernelArg(k, 1, 8, &pB)); CHECK(clSetKernelArg(k, 2, 8, &bC));
    CHECK(clSetKernelArg(k, 3, 4, &N)); CHECK(clSetKernelArg(k, 4, 4, &N)); CHECK(clSetKernelArg(k, 5, 4, &N));
    if (getenv("NEW")) { CHECK(clSetKernelArg(k, 6, 4, &N)); CHECK(clSetKernelArg(k, 7, 4, &N)); }
    size_t gm = (size_t)(N / 16) * (N / 16) * 16, lm = 16; double best = 1e30;
    int reps = getenv("REPS") ? atoi(getenv("REPS")) : 6;
    for (int r = 0; r < reps; r++) { double t = run(q, k, 1, &gm, &lm); if (r > 0 && t < best) best = t; }
    CHECK(clEnqueueReadBuffer(q, bC, CL_TRUE, 0, nn * 4, hC, 0, NULL, NULL));
    if (getenv("DUMP")) { for (int r = 0; r < 16; r++) { for (int c = 0; c < 16; c++) printf("%g ", hC[(size_t)r * N + c]); printf("\n"); } return 0; }
    double mx = 0, avg = 0; srand(99);
    for (int s = 0; s < 256; s++) {
        size_t i = rand() % N, j = rand() % N; double ref = 0, mag = 0;
        for (int kk = 0; kk < N; kk++) { double t = (double)hA[i * N + kk] * hB[(size_t)kk * N + j]; ref += t; mag += fabs(t); }
        double e = fabs(hC[i * N + j] - ref) / mag; avg += e; if (e > mx) mx = e;
    }
    double fl = 2.0 * N * N * (double)N;
    printf("%-14s N=%-5d time=%9.3f ms  %7.1f GFLOPS  rel_err: avg=%.2e max=%.2e  | with packing: %.1f GFLOPS\n",
           kname, N, best * 1e3, fl / best / 1e9, avg / 256, mx, fl / (best + tp) / 1e9);
    return 0;
}
