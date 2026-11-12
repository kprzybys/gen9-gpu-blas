// Porównanie: to samo jądro (store.cl) przez OpenCL/NEO -- enqueue + czekanie (clFinish albo odpytywanie z uśpieniem).
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double teraz(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    int uspij = argc > 1 && !strcmp(argv[1], "uspienie");
    cl_platform_id pl; cl_device_id dev; cl_int e;
    clGetPlatformIDs(1, &pl, NULL); clGetDeviceIDs(pl, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &e);
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, NULL, &e);
    FILE *f = fopen("store.cl", "rb"); char src[4096]; size_t n = fread(src, 1, sizeof src - 1, f); src[n] = 0; fclose(f);
    const char *s = src; cl_program p = clCreateProgramWithSource(ctx, 1, &s, NULL, &e);
    clBuildProgram(p, 1, &dev, "-cl-intel-greater-than-4GB-buffer-required", NULL, NULL);
    cl_kernel k = clCreateKernel(p, "zapis", &e);
    cl_mem b = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 256 * 4, NULL, &e);
    cl_uint v = 1000; clSetKernelArg(k, 0, sizeof b, &b); clSetKernelArg(k, 1, 4, &v);
    size_t g = 256, l = 16;
    for (int r = 0; r < 100; r++) { clEnqueueNDRangeKernel(q, k, 1, NULL, &g, &l, 0, NULL, NULL); clFinish(q); }
    const int R = 2000; static double cz[2000]; double c0 = clock() / (double)CLOCKS_PER_SEC, t0 = teraz();
    for (int r = 0; r < R; r++) {
        double a = teraz();
        if (!uspij) { clEnqueueNDRangeKernel(q, k, 1, NULL, &g, &l, 0, NULL, NULL); clFinish(q); }
        else {  // jak czekaj() w gen9blas: znacznik + flush + odpytywanie z nanosleep (20 us, podwajane do 500 us)
            cl_event ev; clEnqueueNDRangeKernel(q, k, 1, NULL, &g, &l, 0, NULL, &ev); clFlush(q);
            cl_int st; long ns = 20000;
            while (clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof st, &st, NULL) == CL_SUCCESS && st > CL_COMPLETE) {
                struct timespec ts = { 0, ns }; nanosleep(&ts, NULL); if (ns < 500000) ns *= 2; }
            clReleaseEvent(ev);
        }
        cz[r] = teraz() - a;
    }
    double sc = teraz() - t0, cpu = clock() / (double)CLOCKS_PER_SEC - c0;
    for (int i = 1; i < R; i++) { double x = cz[i]; int j = i; while (j > 0 && cz[j - 1] > x) { cz[j] = cz[j - 1]; j--; } cz[j] = x; }
    printf("NEO %-9s: mediana %.1f us, min %.1f us, CPU %.1f us/wywolanie (%.0f%% rdzenia)\n", uspij ? "uspienie" : "clFinish",
           cz[R / 2] * 1e6, cz[0] * 1e6, cpu / R * 1e6, 100 * cpu / sc);
    return 0;
}
