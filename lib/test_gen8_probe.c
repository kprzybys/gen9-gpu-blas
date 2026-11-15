// Gen8 Cherryview probe (docs/gen8.md 6a): runs one kernel of bench/gen8_probe.cl through kcl and checks it on the CPU.
//   ./test_gen8_probe df      probe_df_dword: DF arithmetic, dword messages only
//   ./test_gen8_probe qword   probe_qword: A64 QWord scattered read (the message of the hanging kernels), no DF type
// Build: make -C lib test_gen8_probe (kcl only, no gen9blas). Exit code 0 = correct.
#include "kcl.h"
#include <stdio.h>
#include <string.h>

#define CHECK(x) do { cl_int e_ = (x); if (e_ != CL_SUCCESS) { fprintf(stderr, "%s: %d\n", #x, e_); return 2; } } while (0)

int main(int argc, char **argv) {
    const int df = argc > 1 && !strcmp(argv[1], "df"), qw = argc > 1 && !strcmp(argv[1], "qword");
    if (!df && !qw) { fprintf(stderr, "usage: %s df|qword\n", argv[0]); return 2; }
    cl_platform_id plat; cl_device_id dev; cl_int err;
    CHECK(clGetPlatformIDs(1, &plat, NULL));
    CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL));
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err); CHECK(err);
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err); CHECK(err);
    if (!kcl_gen8()) { fprintf(stderr, "not a Gen8 GPU: the probes exist only in kernels_gen8.c\n"); return 2; }
    clHostMemAllocINTEL_fn alloc = (clHostMemAllocINTEL_fn)clGetExtensionFunctionAddressForPlatform(plat, "clHostMemAllocINTEL");
    clSetKernelArgMemPointerINTEL_fn arg_ptr =
        (clSetKernelArgMemPointerINTEL_fn)clGetExtensionFunctionAddressForPlatform(plat, "clSetKernelArgMemPointerINTEL");
    cl_program pr;
    CHECK(kcl_program("gen8_probe.cl", "-cl-intel-greater-than-4GB-buffer-required", &pr));
    cl_kernel k = clCreateKernel(pr, df ? "probe_df_dword" : "probe_qword", &err); CHECK(err);
    enum { N = 64 };
    double *x = alloc(ctx, NULL, N * 8, 64, &err); CHECK(err);
    double *y = alloc(ctx, NULL, N * 8, 64, &err); CHECK(err);
    for (int i = 0; i < N; i++) { x[i] = 0.5 * i - 7.25; y[i] = 0; }
    CHECK(arg_ptr(k, 0, x)); CHECK(arg_ptr(k, 1, y));
    const double s = 3.0;
    if (df) CHECK(clSetKernelArg(k, 2, sizeof s, &s));
    const size_t g = N;
    CHECK(clEnqueueNDRangeKernel(q, k, 1, NULL, &g, NULL, 0, NULL, NULL));
    CHECK(clFinish(q));
    int bad = 0;
    for (int i = 0; i < N; i++) {
        uint64_t got, want, xi; memcpy(&got, &y[i], 8); memcpy(&xi, &x[i], 8);
        if (df) { const double w = x[i] * s; memcpy(&want, &w, 8); } else want = xi + 1;
        if (got != want) { if (bad < 4) fprintf(stderr, "i=%d got %016llx want %016llx\n", i, (unsigned long long)got, (unsigned long long)want); bad++; }
    }
    printf("%s: %s (%d of %d wrong)\n", df ? "probe_df_dword" : "probe_qword", bad ? "WRONG" : "OK", bad, N);
    return bad ? 1 : 0;
}
