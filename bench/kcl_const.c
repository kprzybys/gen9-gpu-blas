// kcl_const.c -- run kcl_const.cl through kcl (prebuilt whl binary, table from lib/gen_kernels.py) and compare with
// the CPU. PASS = max relative error < 1e-5. Build/run: make kcl_const && ./kcl_const  (no NEO involved).
#include "kcl.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const int n = 1 << 16;
    static const float TAB[8] = { 1.5f, -2.25f, 3.0f, 0.125f, 7.0f, -0.5f, 2.0f, 9.75f };
    float *x = aligned_alloc(4096, n * 4), *y = aligned_alloc(4096, n * 4);
    for (int i = 0; i < n; i++) { x[i] = 0.01f + 0.37f * (i % 997); y[i] = 12345.0f; }
    cl_platform_id pl; cl_device_id dev; cl_int e; cl_program prog;
    if (clGetPlatformIDs(1, &pl, NULL) || clGetDeviceIDs(pl, CL_DEVICE_TYPE_GPU, 1, &dev, NULL)) { puts("FAIL: no device"); return 1; }
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &e);
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, NULL, &e);
    if (kcl_program("kcl_const.bin", NULL, &prog) != CL_SUCCESS) { puts("FAIL: kcl_const.bin not in the table"); return 1; }
    cl_kernel k = clCreateKernel(prog, "k_const", &e);
    if (e != CL_SUCCESS) { printf("FAIL (refused): clCreateKernel = %d (kernel marked unsupported)\n", e); return 1; }
    kcl_new_operation();
    cl_mem mx = kcl_userptr(x, n * 4, 0); clSetKernelArg(k, 0, sizeof mx, &mx);
    cl_mem my = kcl_userptr(y, n * 4, 1); clSetKernelArg(k, 1, sizeof my, &my);
    clSetKernelArg(k, 2, sizeof n, &n);
    size_t gs = n, ls = 64;
    if (clEnqueueNDRangeKernel(q, k, 1, NULL, &gs, &ls, 0, NULL, NULL) || kcl_wait()) { puts("FAIL: launch"); return 1; }
    double worst = 0; int at = 0;
    for (int i = 0; i < n; i++) {
        double ref = TAB[i & 7] * log10((double)x[i]) + exp(-(double)x[i]);
        double rel = fabs(y[i] - ref) / (fabs(ref) > 1e-3 ? fabs(ref) : 1e-3);
        if (rel > worst) { worst = rel; at = i; }
    }
    printf("%s: max relative error %.3e at %d (gpu %.7g, cpu %.7g)\n", worst < 1e-5 ? "PASS" : "FAIL", worst, at, y[at],
           TAB[at & 7] * log10((double)x[at]) + exp(-(double)x[at]));
    return worst < 1e-5 ? 0 : 1;
}
