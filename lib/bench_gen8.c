// Gen8 performance-model ceilings (docs/gen8.md): FMA peak (FP32, FP64) and memory bandwidth (read, write, copy),
// kernels from bench/gen8_probe.cl through kcl. Best of 5 timed runs after a warm-up, wall time around clFinish.
// Build: make -C lib bench_gen8 (Gen8 only). Run: ./bench_gen8 [MB per buffer, default 128]
#include "kcl.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHECK(x) do { cl_int e_ = (x); if (e_ != CL_SUCCESS) { fprintf(stderr, "%s: %d\n", #x, e_); exit(2); } } while (0)
static clHostMemAllocINTEL_fn alloc; static clSetKernelArgMemPointerINTEL_fn arg_ptr;
static cl_context ctx; static cl_command_queue q;

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static double best_time(cl_kernel k, size_t g, size_t l) {
    double best = 1e9;
    for (int r = 0; r < 6; r++) {
        const double t = now();
        CHECK(clEnqueueNDRangeKernel(q, k, 1, NULL, &g, l ? &l : NULL, 0, NULL, NULL)); CHECK(clFinish(q));
        if (r) { const double d = now() - t; if (d < best) best = d; }   // r = 0: warm-up
    }
    return best;
}
int main(int argc, char **argv) {
    const long mb = argc > 1 ? atol(argv[1]) : 128;
    cl_platform_id plat; cl_device_id dev; cl_int err; cl_program pr;
    CHECK(clGetPlatformIDs(1, &plat, NULL)); CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL));
    ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err); CHECK(err);
    q = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err); CHECK(err);
    if (!kcl_gen8()) { fprintf(stderr, "Gen8 only\n"); return 2; }
    alloc = (clHostMemAllocINTEL_fn)clGetExtensionFunctionAddressForPlatform(plat, "clHostMemAllocINTEL");
    arg_ptr = (clSetKernelArgMemPointerINTEL_fn)clGetExtensionFunctionAddressForPlatform(plat, "clSetKernelArgMemPointerINTEL");
    CHECK(kcl_program("gen8_probe.cl", "-cl-intel-greater-than-4GB-buffer-required", &pr));
    const size_t G = 12 * 7 * 16 * 8;   // 8 thread groups' worth per hardware thread slot (12 EU x 7 threads, SIMD16)
    float *out = alloc(ctx, NULL, G * 8, 64, &err); CHECK(err);
    { cl_kernel k = clCreateKernel(pr, "peak_f32", &err); CHECK(err);
      const int it = 32768; const float a = 0.999f, b = 0.001f;
      CHECK(arg_ptr(k, 0, out)); CHECK(clSetKernelArg(k, 1, 4, &it)); CHECK(clSetKernelArg(k, 2, 4, &a)); CHECK(clSetKernelArg(k, 3, 4, &b));
      const double t = best_time(k, G, 0);
      printf("peak FP32 FMA: %7.1f GFLOPS (%.2f ms)\n", (double)G * it * 16 * 2 / t / 1e9, t * 1e3); }
    { cl_kernel k = clCreateKernel(pr, "peak_f64", &err); CHECK(err);
      const int it = 8192; const double a = 0.999, b = 0.001; const size_t g = G / 2;
      CHECK(arg_ptr(k, 0, out)); CHECK(clSetKernelArg(k, 1, 4, &it)); CHECK(clSetKernelArg(k, 2, 8, &a)); CHECK(clSetKernelArg(k, 3, 8, &b));
      const double t = best_time(k, g, 0);
      printf("peak FP64 FMA: %7.1f GFLOPS (%.2f ms)\n", (double)g * it * 8 * 2 / t / 1e9, t * 1e3); }
    const long n4 = mb * (1 << 20) / 16;
    float *p = alloc(ctx, NULL, n4 * 16, 4096, &err); CHECK(err);
    float *p2 = alloc(ctx, NULL, n4 * 16, 4096, &err); CHECK(err);
    for (long i = 0; i < n4 * 4; i++) { p[i] = 1.0f; p2[i] = 0.0f; }
    const size_t gm = 12 * 7 * 16 * 4;
    { cl_kernel k = clCreateKernel(pr, "mem_read", &err); CHECK(err);
      CHECK(arg_ptr(k, 0, p)); CHECK(clSetKernelArg(k, 1, 8, &n4)); CHECK(arg_ptr(k, 2, out));
      const double t = best_time(k, gm, 0); printf("memory read : %6.2f GB/s (%ld MB)\n", n4 * 16.0 / t / 1e9, mb); }
    { cl_kernel k = clCreateKernel(pr, "mem_write", &err); CHECK(err);
      CHECK(arg_ptr(k, 0, p2)); CHECK(clSetKernelArg(k, 1, 8, &n4));
      const double t = best_time(k, gm, 0); printf("memory write: %6.2f GB/s\n", n4 * 16.0 / t / 1e9); }
    { cl_kernel k = clCreateKernel(pr, "mem_copy", &err); CHECK(err);
      CHECK(arg_ptr(k, 0, p)); CHECK(arg_ptr(k, 1, p2)); CHECK(clSetKernelArg(k, 2, 8, &n4));
      const double t = best_time(k, gm, 0); printf("memory copy : %6.2f GB/s (read + write)\n", 2 * n4 * 16.0 / t / 1e9); }
    return 0;
}
