// clbench: minimalny program do mikrobenchmarków OpenCL na UHD 620.
// Każdy kernel ma sygnaturę: kernel(__global void *out, int n).
// Użycie: clbench <plik.cl> <kernel> <global> <local> <n> <ops_na_item_na_iter> [opcje kompilatora]
// Wypisuje: najlepszy czas, szybkość (Gop/s) i średnie taktowanie GPU w czasie najlepszego przebiegu.
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(x) do { cl_int e_ = (x); if (e_ != CL_SUCCESS) { fprintf(stderr, "%s:%d: %s = %d\n", __FILE__, __LINE__, #x, e_); exit(1); } } while (0)
#define FREQ_FILE "/sys/class/drm/card1/gt_act_freq_mhz"

static char *read_file(const char *p) {
    FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f);
    char *s = malloc(n + 1); if (fread(s, 1, n, f) != (size_t)n) exit(1); s[n] = 0; fclose(f); return s;
}

static int read_freq(void) {
    FILE *f = fopen(FREQ_FILE, "r"); int v = -1;
    if (f) { if (fscanf(f, "%d", &v) != 1) v = -1; fclose(f); }
    return v;
}

int main(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "uzycie: %s plik.cl kernel global local n ops [opcje]\n", argv[0]); return 2; }
    const char *file = argv[1], *kname = argv[2];
    size_t global = strtoull(argv[3], 0, 0), local = strtoull(argv[4], 0, 0);
    int n = atoi(argv[5]); double ops = atof(argv[6]);
    const char *opts = argc > 7 ? argv[7] : "";
    int reps = getenv("REPS") ? atoi(getenv("REPS")) : 5;

    cl_platform_id plat; cl_device_id dev;
    CHECK(clGetPlatformIDs(1, &plat, NULL));
    CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL));
    cl_int err;
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err); CHECK(err);
    cl_queue_properties qp[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, qp, &err); CHECK(err);

    // Plik .bin = gotowa binarka GPU (np. z ocloc asm), inaczej źródło OpenCL C.
    cl_program prog;
    size_t flen = strlen(file);
    if (flen > 4 && !strcmp(file + flen - 4, ".bin")) {
        FILE *bf = fopen(file, "rb"); if (!bf) { perror(file); return 1; }
        fseek(bf, 0, SEEK_END); size_t blen = ftell(bf); rewind(bf);
        unsigned char *bin = malloc(blen); if (fread(bin, 1, blen, bf) != blen) return 1; fclose(bf);
        cl_int bst;
        prog = clCreateProgramWithBinary(ctx, 1, &dev, &blen, (const unsigned char **)&bin, &bst, &err); CHECK(err); CHECK(bst);
    } else {
        char *src = read_file(file);
        prog = clCreateProgramWithSource(ctx, 1, (const char **)&src, NULL, &err); CHECK(err);
    }
    if (clBuildProgram(prog, 1, &dev, opts, NULL, NULL) != CL_SUCCESS) {
        char log[65536]; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sizeof log, log, NULL);
        fprintf(stderr, "blad kompilacji:\n%s\n", log); return 1;
    }
    cl_kernel k = clCreateKernel(prog, kname, &err); CHECK(err);

    size_t sg = 0;
    clGetKernelSubGroupInfo(k, dev, CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE, sizeof local, &local, sizeof sg, &sg, NULL);
    size_t outbytes = global * 16 + 4096;
    cl_mem out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, outbytes, NULL, &err); CHECK(err);
    { cl_uchar z = 0; CHECK(clEnqueueFillBuffer(q, out, &z, 1, 0, outbytes, 0, NULL, NULL)); }
    CHECK(clSetKernelArg(k, 0, sizeof out, &out));
    CHECK(clSetKernelArg(k, 1, sizeof n, &n));
    // Opcjonalne wejścia: BUF=<bajty> -> argument 2 (bufor wyzerowany), IMG=<szer>x<wys> -> kolejny argument
    // (obraz 2D, CL_R / CL_UNSIGNED_INT32, szerokość w pikselach po 4 B).
    int argi = 2;
    if (getenv("BUF")) {
        size_t bb = strtoull(getenv("BUF"), 0, 0);
        cl_mem in = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bb, NULL, &err); CHECK(err);
        cl_uint z = 0; CHECK(clEnqueueFillBuffer(q, in, &z, 4, 0, bb, 0, NULL, NULL));
        CHECK(clSetKernelArg(k, argi++, sizeof in, &in));
    }
    if (getenv("IMG")) {
        size_t w = 0, h = 0; sscanf(getenv("IMG"), "%zux%zu", &w, &h);
        cl_image_format fmt = { CL_R, CL_UNSIGNED_INT32 };
        cl_image_desc desc = { CL_MEM_OBJECT_IMAGE2D, w, h, 1, 1, 0, 0, 0, 0, { NULL } };
        cl_mem im = clCreateImage(ctx, CL_MEM_READ_WRITE, &fmt, &desc, NULL, &err); CHECK(err);
        cl_uint z[4] = {0}; size_t o3[3] = {0, 0, 0}, r3[3] = {w, h, 1};
        CHECK(clEnqueueFillImage(q, im, z, o3, r3, 0, NULL, NULL));
        CHECK(clSetKernelArg(k, argi++, sizeof im, &im));
    }
    CHECK(clFinish(q));

    double best = 1e30; int best_freq = -1;
    for (int r = 0; r < reps + 1; r++) {   // przebieg 0 to rozgrzewka
        cl_event ev;
        CHECK(clEnqueueNDRangeKernel(q, k, 1, NULL, &global, local ? &local : NULL, 0, NULL, &ev));
        CHECK(clFlush(q));
        long fsum = 0; int fcnt = 0; cl_int st;
        do {   // próbkowanie taktowania w trakcie pracy kernela
            CHECK(clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof st, &st, NULL));
            if (st <= CL_SUBMITTED) { int fr = read_freq(); if (fr > 0) { fsum += fr; fcnt++; } }
            struct timespec ts = {0, 200000}; nanosleep(&ts, NULL);
        } while (st > CL_COMPLETE);
        cl_ulong t0, t1;
        CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof t0, &t0, NULL));
        CHECK(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof t1, &t1, NULL));
        double ms = (t1 - t0) * 1e-6;
        if (r > 0 && ms < best) { best = ms; best_freq = fcnt ? (int)(fsum / fcnt) : -1; }
        clReleaseEvent(ev);
    }
    double gops = ops * (double)global * n / (best * 1e-3) / 1e9;
    printf("%-14s global=%-8zu local=%-4zu sg=%-2zu n=%-6d czas=%9.3f ms  %8.2f Gop/s  GPU~%d MHz\n",
           kname, global, local, sg, n, best, gops, best_freq);
    return 0;
}
