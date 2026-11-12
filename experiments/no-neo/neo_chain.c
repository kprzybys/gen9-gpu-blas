// NEO: ten sam łańcuch co chain.c (zapis + K-1 × kopiuj, kolejka w kolejności), jedno czekanie na końcu
// (clFinish albo odpytywanie z uśpieniem jak czekaj() w gen9blas). Użycie: ./neo_lancuch K N [uspienie]
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double teraz(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int por(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
int main(int argc, char **argv) {
    int K = atoi(argv[1]), N = atoi(argv[2]), usp = argc > 3 && !strcmp(argv[3], "uspienie");
    cl_platform_id pl; cl_device_id dev; cl_int e;
    clGetPlatformIDs(1, &pl, NULL); clGetDeviceIDs(pl, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &e);
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, NULL, &e);
    FILE *f = fopen("store.cl", "rb"); static char src[16384]; size_t n = fread(src, 1, sizeof src - 1, f); src[n] = 0; fclose(f);
    const char *s = src; cl_program p = clCreateProgramWithSource(ctx, 1, &s, NULL, &e);
    clBuildProgram(p, 1, &dev, "-cl-intel-greater-than-4GB-buffer-required", NULL, NULL);
    cl_kernel kz = clCreateKernel(p, "zapis", &e), k1 = clCreateKernel(p, "kopiuj", &e), k2 = clCreateKernel(p, "kopiuj", &e);
    cl_mem X = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * 4, NULL, &e), Y = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * 4, NULL, &e);
    cl_uint v = 1000; clSetKernelArg(kz, 0, sizeof X, &X); clSetKernelArg(kz, 1, 4, &v);
    clSetKernelArg(k1, 0, sizeof X, &X); clSetKernelArg(k1, 1, sizeof Y, &Y);
    clSetKernelArg(k2, 0, sizeof Y, &Y); clSetKernelArg(k2, 1, sizeof X, &X);
    size_t g = N, l = 16;
    enum { R = 200 }; double t[R], c0 = 0;
    for (int r = -20; r < R; r++) {
        if (r == 0) c0 = clock() / (double)CLOCKS_PER_SEC;
        double a = teraz(); cl_event ev = NULL;
        clEnqueueNDRangeKernel(q, kz, 1, NULL, &g, &l, 0, NULL, NULL);
        for (int k = 1; k < K; k++) clEnqueueNDRangeKernel(q, k % 2 ? k1 : k2, 1, NULL, &g, &l, 0, NULL, k == K - 1 ? &ev : NULL);
        if (!usp) clFinish(q);
        else { clFlush(q); cl_int st; long ns = 20000;
            while (clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof st, &st, NULL) == CL_SUCCESS && st > CL_COMPLETE) {
                struct timespec ts = { 0, ns }; nanosleep(&ts, NULL); if (ns < 500000) ns *= 2; } }
        if (ev) clReleaseEvent(ev);
        if (r >= 0) t[r] = teraz() - a;
    }
    double cpu = (clock() / (double)CLOCKS_PER_SEC - c0) / R;
    qsort(t, R, sizeof *t, por);
    printf("NEO %-8s K = %d, N = %d: mediana %.1f us, CPU %.1f us\n", usp ? "uspienie" : "clFinish", K, N, t[R / 2] * 1e6, cpu * 1e6);
    return 0;
}
