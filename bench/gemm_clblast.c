// Step 0: GEMM reference point on the GPU through CLBlast.
// Usage: gemm_clblast <s|h|d> <N> [seconds]
//   without [seconds]: best of 5 runs (after warm-up and kernel compilation);
//   with [seconds]: repeats GEMM for the given time (for power measurement) and reports the mean.
// Accuracy: 256 random elements of C compared with a double computation
// on the same input data (already rounded to the precision).
#define CL_TARGET_OPENCL_VERSION 300
#include <clblast_c.h>
#include <clblast_half.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(x) do { cl_int e_ = (cl_int)(x); if (e_ != 0) { fprintf(stderr, "%s:%d: %s = %d\n", __FILE__, __LINE__, #x, e_); exit(1); } } while (0)

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s s|h|d N [seconds]\n", argv[0]); return 2; }
    char p = argv[1][0]; size_t N = strtoul(argv[2], 0, 0); double secs = argc > 3 ? atof(argv[3]) : 0;
    size_t es = p == 'd' ? 8 : p == 's' ? 4 : 2, nn = N * N;
    size_t LD = N + (getenv("PAD") ? strtoul(getenv("PAD"), 0, 0) : 0);   // PAD: row padding (hypothesis testing)

    cl_platform_id plat; cl_device_id dev; cl_int err;
    CHECK(clGetPlatformIDs(1, &plat, NULL));
    CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL));
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err); CHECK(err);
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err); CHECK(err);

    // Data: random from [-1, 1]; ref* are the values after rounding to the tested precision.
    double *refA = malloc(nn * 8), *refB = malloc(nn * 8);
    void *hA = calloc(N * LD, es), *hB = calloc(N * LD, es), *hC = calloc(N * LD, es);
    srand(1234);
    for (size_t i = 0; i < 2 * nn; i++) {
        double v = 2.0 * rand() / RAND_MAX - 1.0;
        double *ref = i < nn ? refA : refB; void *h = i < nn ? hA : hB; size_t j = i % nn, jl = j / N * LD + j % N;
        if (p == 'd') { ((double *)h)[jl] = v; ref[j] = v; }
        else if (p == 's') { ((float *)h)[jl] = (float)v; ref[j] = (float)v; }
        else { cl_half x = FloatToHalf((float)v); ((cl_half *)h)[jl] = x; ref[j] = HalfToFloat(x); }
    }
    cl_mem A = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * LD * es, hA, &err); CHECK(err);
    cl_mem B = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * LD * es, hB, &err); CHECK(err);
    cl_mem C = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * LD * es, NULL, &err); CHECK(err);

    #define GEMM() do { cl_event ev; \
        if (p == 's') CHECK(CLBlastSgemm(CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeNo, N, N, N, 1.0f, A, 0, LD, B, 0, LD, 0.0f, C, 0, LD, &q, &ev)); \
        else if (p == 'd') CHECK(CLBlastDgemm(CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeNo, N, N, N, 1.0, A, 0, LD, B, 0, LD, 0.0, C, 0, LD, &q, &ev)); \
        else CHECK(CLBlastHgemm(CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeNo, N, N, N, FloatToHalf(1.0f), A, 0, LD, B, 0, LD, FloatToHalf(0.0f), C, 0, LD, &q, &ev)); \
        CHECK(clWaitForEvents(1, &ev)); clReleaseEvent(ev); } while (0)

    double t0 = now(); GEMM(); double t_first = now() - t0;   // first call: kernel compilation
    double flop = 2.0 * N * N * N, best = 1e30; int runs = 0;
    if (secs > 0) {
        double start = now();
        do { GEMM(); runs++; } while (now() - start < secs);
        best = (now() - start) / runs;
    } else {
        for (int r = 0; r < 5; r++) { t0 = now(); GEMM(); double t = now() - t0; if (t < best) best = t; }
    }

    // Accuracy: relative error |c - ref| / sum|a*b| (independent of cancellation of terms).
    CHECK(clEnqueueReadBuffer(q, C, CL_TRUE, 0, N * LD * es, hC, 0, NULL, NULL));
    double maxerr = 0, sumerr = 0; srand(99);
    for (int s = 0; s < 256; s++) {
        size_t i = rand() % N, j = rand() % N; double ref = 0, mag = 0;
        for (size_t k = 0; k < N; k++) { double t = refA[i * N + k] * refB[k * N + j]; ref += t; mag += fabs(t); }
        double c = p == 'd' ? ((double *)hC)[i * LD + j] : p == 's' ? ((float *)hC)[i * LD + j] : HalfToFloat(((cl_half *)hC)[i * LD + j]);
        double e = fabs(c - ref) / mag; sumerr += e; if (e > maxerr) maxerr = e;
    }
    printf("clblast %cgemm N=%-5zu time=%9.3f ms  %7.1f GFLOPS  rel_err: avg=%.2e max=%.2e  (first call %.0f ms%s)\n",
           p, N, best * 1e3, flop / best / 1e9, sumerr / 256, maxerr, t_first * 1e3, secs > 0 ? ", continuous mode" : "");
    return 0;
}
