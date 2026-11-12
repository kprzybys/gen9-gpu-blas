// cblas_sgemm on the GPU through CLBlast, for comparison with libgen9blas_blas.so on the same path
// (llama.cpp GGML_BLAS=ON, LD_PRELOAD). Every call: copy A and B into new OpenCL buffers (CL_MEM_COPY_HOST_PTR),
// CLBlastSgemm, read C back. All other CBLAS/BLAS symbols stay with the program's own BLAS.
// Build: gcc -O2 -Wall -shared -fPIC -o libclblast_cblas.so clblast_cblas.c -lclblast -lOpenCL -lpthread
// Use:   LD_PRELOAD=$PWD/libclblast_cblas.so <program>     CLBLAST_CBLAS_INFO=1 prints the call count at exit.
#define CL_TARGET_OPENCL_VERSION 300
#include <clblast_c.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static cl_context ctx;
static cl_command_queue q;
static pthread_once_t once = PTHREAD_ONCE_INIT;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static long calls;
static double secs;

#define CHECK(x) do { cl_int e_ = (cl_int)(x); if (e_ != 0) { \
    fprintf(stderr, "clblast_cblas: %s:%d: %s = %d\n", __FILE__, __LINE__, #x, e_); abort(); } } while (0)

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void init(void) {
    cl_platform_id plat; cl_device_id dev; cl_int err;
    CHECK(clGetPlatformIDs(1, &plat, NULL));
    CHECK(clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL));
    ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err); CHECK(err);
    q = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err); CHECK(err);
}

__attribute__((destructor)) static void report(void) {
    if (getenv("CLBLAST_CBLAS_INFO"))
        fprintf(stderr, "clblast_cblas: %ld sgemm calls on the GPU, %.3f s inside cblas_sgemm\n", calls, secs);
}

// CBLAS enum values are the same as CLBlast's (RowMajor 101, ColMajor 102, NoTrans 111, Trans 112, ConjTrans 113).
void cblas_sgemm(int order, int ta, int tb, int M, int N, int K, float alpha, const float *A, int lda,
                 const float *B, int ldb, float beta, float *C, int ldc) {
    pthread_once(&once, init);
    pthread_mutex_lock(&mu);
    double t0 = now();
    int row = order == 101, tA = ta != 111, tB = tb != 111;
    size_t nA = (size_t)((row != tA) ? M : K) * lda;   // rows (row-major) or columns (column-major) times leading dim
    size_t nB = (size_t)((row != tB) ? K : N) * ldb;
    size_t nC = (size_t)(row ? M : N) * ldc;
    cl_int err;
    cl_mem a = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nA * 4, (void *)A, &err); CHECK(err);
    cl_mem b = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nB * 4, (void *)B, &err); CHECK(err);
    cl_mem c = clCreateBuffer(ctx, CL_MEM_READ_WRITE | (beta != 0.0f ? CL_MEM_COPY_HOST_PTR : 0), nC * 4,
                              beta != 0.0f ? (void *)C : NULL, &err); CHECK(err);
    cl_event ev;
    CHECK(CLBlastSgemm(row ? CLBlastLayoutRowMajor : CLBlastLayoutColMajor, (CLBlastTranspose)ta, (CLBlastTranspose)tb,
                       M, N, K, alpha, a, 0, lda, b, 0, ldb, beta, c, 0, ldc, &q, &ev));
    CHECK(clWaitForEvents(1, &ev)); clReleaseEvent(ev);
    CHECK(clEnqueueReadBuffer(q, c, CL_TRUE, 0, nC * 4, C, 0, NULL, NULL));
    clReleaseMemObject(a); clReleaseMemObject(b); clReleaseMemObject(c);
    calls++; secs += now() - t0;
    pthread_mutex_unlock(&mu);
}
