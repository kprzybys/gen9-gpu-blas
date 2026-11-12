// Measurement of B packing variants (bench/packing.cl): kernel time (profiling, best of 20), GB/s (read + write),
// byte-for-byte match with the reference pack_b_pad. Data: B n × n FP16 in a device buffer, Kp = Np = n rounded up to 64.
// Usage: packing <kernel> <n> <gx> <gy> [local_x]   (run from the repo root)
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { cl_int e_ = (x); if (e_ != CL_SUCCESS) { fprintf(stderr, "%s:%d: %s = %d\n", __FILE__, __LINE__, #x, e_); exit(1); } } while (0)

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s kernel n gx gy [lx]\n", argv[0]); return 2; }
    int n = atoi(argv[2]), np = (n + 63) / 64 * 64; size_t gx = strtoul(argv[3], 0, 0), gy = strtoul(argv[4], 0, 0);
    size_t lx = argc > 5 ? strtoul(argv[5], 0, 0) : 0;
    FILE *f = fopen("bench/packing.cl", "rb"); fseek(f, 0, SEEK_END); long len = ftell(f); rewind(f);
    char *src = malloc(len + 1); if (fread(src, 1, len, f) != (size_t)len) return 1; src[len] = 0; fclose(f);
    cl_platform_id pl; cl_device_id dev; cl_int err;
    CHECK(clGetPlatformIDs(1, &pl, NULL)); CHECK(clGetDeviceIDs(pl, CL_DEVICE_TYPE_GPU, 1, &dev, NULL));
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err); CHECK(err);
    cl_queue_properties pr[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, pr, &err); CHECK(err);
    cl_program p = clCreateProgramWithSource(ctx, 1, (const char **)&src, NULL, &err); CHECK(err);
    if (clBuildProgram(p, 1, &dev, "-cl-std=CL2.0", NULL, NULL) != CL_SUCCESS) {
        char log[16384]; clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, sizeof log, log, NULL); fprintf(stderr, "%s\n", log); return 1;
    }
    size_t nb = (size_t)n * n * 2, pb = (size_t)np * np * 2;
    unsigned short *h = malloc(nb), *w1 = malloc(pb), *w2 = malloc(pb);
    srand(1); for (size_t i = 0; i < (size_t)n * n; i++) h[i] = rand() & 0x7bff;
    // SOURCE=usm: B in host USM memory (clHostMemAllocINTEL, as gen9blas.empty), argument via SetKernelArgMemPointer.
    int usm = getenv("SOURCE") && !strcmp(getenv("SOURCE"), "usm");
    typedef void *(*halloc_t)(cl_context, const void *, size_t, cl_uint, cl_int *);
    typedef cl_int (*setp_t)(cl_kernel, cl_uint, const void *);
    halloc_t halloc = (halloc_t)clGetExtensionFunctionAddressForPlatform(pl, "clHostMemAllocINTEL");
    setp_t setp = (setp_t)clGetExtensionFunctionAddressForPlatform(pl, "clSetKernelArgMemPointerINTEL");
    void *hB = NULL;
    cl_mem B = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nb, h, &err); CHECK(err);
    if (usm) { hB = halloc(ctx, NULL, nb, 4096, &err); CHECK(err); memcpy(hB, h, nb); }
    cl_mem P1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, pb, NULL, &err); CHECK(err);
    cl_mem P2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, pb, NULL, &err); CHECK(err);
    unsigned char ff = 0xff; CHECK(clEnqueueFillBuffer(q, P2, &ff, 1, 0, pb, 0, NULL, NULL));
    const char *ref_name = getenv("REF") ? getenv("REF") : "pack_b_pad";   // REF: reference kernel (same output layout)
    cl_kernel kw = clCreateKernel(p, ref_name, &err); CHECK(err);
    cl_kernel kt = clCreateKernel(p, argv[1], &err); CHECK(err);
    cl_kernel ks[2] = { kw, kt }; cl_mem out[2] = { P1, P2 };
    for (int i = 0; i < 2; i++) {
        if (usm) CHECK(setp(ks[i], 0, hB)); else CHECK(clSetKernelArg(ks[i], 0, sizeof B, &B)); CHECK(clSetKernelArg(ks[i], 1, sizeof out[i], &out[i]));
        CHECK(clSetKernelArg(ks[i], 2, 4, &n)); CHECK(clSetKernelArg(ks[i], 3, 4, &n));
        CHECK(clSetKernelArg(ks[i], 4, 4, &n)); CHECK(clSetKernelArg(ks[i], 5, 4, &np));
        cl_uint na; clGetKernelInfo(ks[i], CL_KERNEL_NUM_ARGS, sizeof na, &na, NULL);
        if (na == 8) { cl_long z = 0; CHECK(clSetKernelArg(ks[i], 6, 8, &z)); CHECK(clSetKernelArg(ks[i], 7, 4, &np)); }
    }
    size_t gw[3] = { np, np, 1 }, g[3] = { gx, gy, 1 }, l[3] = { lx, 1, 1 };
    CHECK(clEnqueueNDRangeKernel(q, kw, 3, NULL, gw, NULL, 0, NULL, NULL));
    double best = 1e30;
    for (int r = 0; r < 20; r++) {
        cl_event ev; CHECK(clEnqueueNDRangeKernel(q, kt, 3, NULL, g, lx ? l : NULL, 0, NULL, &ev)); CHECK(clWaitForEvents(1, &ev));
        cl_ulong t0, t1; clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, 8, &t0, NULL);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, 8, &t1, NULL); clReleaseEvent(ev);
        if ((t1 - t0) * 1e-9 < best) best = (t1 - t0) * 1e-9;
    }
    CHECK(clEnqueueReadBuffer(q, P1, CL_TRUE, 0, pb, w1, 0, NULL, NULL)); CHECK(clEnqueueReadBuffer(q, P2, CL_TRUE, 0, pb, w2, 0, NULL, NULL));
    size_t bad = 0; for (size_t i = 0; i < pb / 2; i++) bad += w1[i] != w2[i];
    printf("%s %-12s n=%-5d %8.3f ms %6.1f GB/s  %s\n", usm ? "usm" : "buf", argv[1], n, best * 1e3, (nb + pb) / best / 1e9, bad ? "BAD" : "OK");
    if (bad) printf("  differing elements: %zu of %zu\n", bad, pb / 2);
    return 0;
}
