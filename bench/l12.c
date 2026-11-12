// Energy measurement of levels 1-2 on the GPU: l12 saxpy|sgemv <n> <seconds>. Waiting with sleep (CPU free).
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define CHECK(x) do { cl_int e_ = (x); if (e_) { fprintf(stderr, "%d: %s = %d\n", __LINE__, #x, e_); exit(1); } } while (0)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    const char *op = argv[1]; size_t n = strtoul(argv[2], 0, 0); double secs = atof(argv[3]);
    cl_platform_id p; cl_device_id d; cl_int e;
    CHECK(clGetPlatformIDs(1, &p, NULL)); CHECK(clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &d, NULL));
    cl_context c = clCreateContext(NULL, 1, &d, NULL, NULL, &e); cl_command_queue q = clCreateCommandQueueWithProperties(c, d, NULL, &e);
    FILE *f = fopen("l12.cl", "rb"); char src[8192]; size_t l = fread(src, 1, sizeof src - 1, f); src[l] = 0; fclose(f);
    const char *s = src; cl_program pr = clCreateProgramWithSource(c, 1, &s, NULL, &e); CHECK(clBuildProgram(pr, 1, &d, "", NULL, NULL));
    cl_kernel k = clCreateKernel(pr, strcmp(op, "saxpy") ? "sgemv_n" : "saxpy", &e); CHECK(e);
    size_t elem = strcmp(op, "saxpy") ? n * n : n, bytes;
    cl_mem a = clCreateBuffer(c, CL_MEM_READ_WRITE, elem * 4, NULL, &e), x = clCreateBuffer(c, CL_MEM_READ_WRITE, n * 4, NULL, &e),
           y = clCreateBuffer(c, CL_MEM_READ_WRITE, n * 4, NULL, &e);
    float z = 0.5f; CHECK(clEnqueueFillBuffer(q, a, &z, 4, 0, elem * 4, 0, NULL, NULL));
    CHECK(clEnqueueFillBuffer(q, x, &z, 4, 0, n * 4, 0, NULL, NULL)); CHECK(clEnqueueFillBuffer(q, y, &z, 4, 0, n * 4, 0, NULL, NULL));
    size_t g, lw = 16; int N = (int)n;
    if (!strcmp(op, "saxpy")) { CHECK(clSetKernelArg(k, 0, 8, &a)); CHECK(clSetKernelArg(k, 1, 8, &y)); CHECK(clSetKernelArg(k, 2, 4, &z)); g = n; bytes = 12 * n; }
    else { CHECK(clSetKernelArg(k, 0, 8, &a)); CHECK(clSetKernelArg(k, 1, 8, &x)); CHECK(clSetKernelArg(k, 2, 8, &y)); CHECK(clSetKernelArg(k, 3, 4, &N)); g = n * 16; bytes = 4 * n * n; }
    CHECK(clFinish(q));
    double t0 = now(); long it = 0;
    while (now() - t0 < secs) {
        cl_event ev; for (int r = 0; r < 20; r++) CHECK(clEnqueueNDRangeKernel(q, k, 1, NULL, &g, strcmp(op, "saxpy") ? &lw : NULL, 0, NULL, r == 19 ? &ev : NULL));
        CHECK(clFlush(q)); cl_int st;
        do { struct timespec ts = {0, 500000}; nanosleep(&ts, NULL); clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, 4, &st, NULL); } while (st > CL_COMPLETE);
        clReleaseEvent(ev); it += 20;
    }
    double t = now() - t0;
    printf("GPU %s n=%zu: %.1f GB/s\n", op, n, bytes * (double)it / t / 1e9);
    return 0;
}
