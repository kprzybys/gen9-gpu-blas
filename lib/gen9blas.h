// libgen9blas: BLAS (levels 1-3) on Intel Gen9 iGPUs (UHD 620 / HD 530; Gen8 Cherryview partly) with hand-written
// assembly GEMM loops. See docs/api.md. Neural-network operations (ggml backend): gen9blas_ml.h.
//
// Conventions
// - All matrices are row-major. Elements: uint16_t (FP16), float (FP32) or double (FP64), passed as void*.
// - Result: a gen9blas_status: 0 = success, < 0 = error (description: gen9blas_error()), GEN9BLAS_DECLINED (1) =
//   "not handled, nothing was written" (only functions documented to decline; the caller computes another way).
//   Functions returning a pointer return NULL on error; gen9blas_last_status() gives the status of the last call.
// - Threads: the library has process-wide state (work buffers, the last error, no-wait mode). Calls must be serialized
//   by the caller (one thread at a time, e.g. under one mutex); gen9blas_error() is valid until the next call.
// - Lifetime: handles (gen9blas_packed_b, gen9blas_packed_a) are owned by the caller and freed with their _free function;
//   free them only after the work that uses them finished (gen9blas_flush() when no-wait / sync = 0 calls are queued).
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define GEN9BLAS_VERSION_MAJOR 0
#define GEN9BLAS_VERSION_MINOR 2
#define GEN9BLAS_VERSION_PATCH 0
#define GEN9BLAS_VERSION (GEN9BLAS_VERSION_MAJOR * 10000 + GEN9BLAS_VERSION_MINOR * 100 + GEN9BLAS_VERSION_PATCH)
#ifndef GEN9BLAS_API
#define GEN9BLAS_API __attribute__((visibility("default")))
#endif

enum { GEN9BLAS_H = 0,    // FP16, FP16 accumulation (fastest)
       GEN9BLAS_S = 1,    // FP32
       GEN9BLAS_HS = 2,   // FP16, accumulated in chunks of GEN9BLAS_KC products into FP32 (more accurate)
       GEN9BLAS_D = 3 };  // FP64
#define GEN9BLAS_KC 32
enum { GEN9BLAS_NO_TRANS = 0, GEN9BLAS_TRANS = 1 };      // trans / transA / transB
enum { GEN9BLAS_LEFT = 0, GEN9BLAS_RIGHT = 1 };          // side
enum { GEN9BLAS_LOWER = 0, GEN9BLAS_UPPER = 1 };         // uplo
enum { GEN9BLAS_NON_UNIT = 0, GEN9BLAS_UNIT = 1 };       // diag

typedef enum {
    GEN9BLAS_OK = 0,
    GEN9BLAS_DECLINED = 1,                // not handled (shape / memory the fast path does not take); nothing was written
    GEN9BLAS_ERR = -1,                    // GPU execution or driver error, or an error without a finer class
    GEN9BLAS_ERR_UNSUPPORTED = -2,        // type / kernel / layout not available in this build or on this device
    GEN9BLAS_ERR_INVALID_ARG = -3,        // invalid arguments (leading dimensions, mismatched handles, ...)
    GEN9BLAS_ERR_OUT_OF_MEMORY = -4,      // GPU or host allocation failed
    GEN9BLAS_ERR_NO_DEVICE = -5,          // initialization failed: no usable GPU (permanent for the process)
} gen9blas_status;

// ---- Library ----
// Version of the loaded library (GEN9BLAS_VERSION of its build): compare with GEN9BLAS_VERSION to detect a mismatch.
GEN9BLAS_API int gen9blas_version(void);
// Opens the GPU and loads the kernels; once per process (later calls return the first result, a failure is final).
// gen9blas_has() and gen9blas_alloc() call it themselves. The NEO comparison build reads kernel sources from GEN9BLAS_DIR.
GEN9BLAS_API int gen9blas_init(void);
// Device generation: 9 (Gen9 / Gen9.5), 8 (Gen8 Cherryview), 0 (no device; initializes like gen9blas_has).
GEN9BLAS_API int gen9blas_device_gen(void);
enum {
    GEN9BLAS_FEAT_FP16 = 0,    // GEN9BLAS_H / GEN9BLAS_HS kernels (not on Gen8 yet)
    GEN9BLAS_FEAT_FP32,        // GEN9BLAS_S
    GEN9BLAS_FEAT_FP64,        // GEN9BLAS_D
    GEN9BLAS_FEAT_SKINNY_PB,   // gen9blas_gemm_pb_skinny
    GEN9BLAS_FEAT_EPILOG_EXT,  // gen9blas_gemm_pb_f16w_ex (fused bias / GELU / residual)
    GEN9BLAS_FEAT_NORM_PACK,   // gen9blas_layer_norm_pack + gen9blas_gemm_pa_f16w_ex
    GEN9BLAS_FEAT_ROPE_MULTI,  // gen9blas_rope_multi
    GEN9BLAS_FEAT_CPY,         // gen9blas_cpy_any and gen9blas_upscale_f32
    GEN9BLAS_FEAT_SET_ROWS,    // gen9blas_set_rows_f16
};
// 1 if the feature is available on this device and build, 0 otherwise (also for features unknown to this library).
GEN9BLAS_API int gen9blas_has(int feature);
GEN9BLAS_API const char *gen9blas_error(void);
GEN9BLAS_API int gen9blas_last_status(void);   // gen9blas_status of the last call (for functions returning pointers)
GEN9BLAS_API double gen9blas_last_time_ms(void);  // GEN9BLAS_PROFILE=1: GEMM kernel time of the last call; else the whole call
GEN9BLAS_API int gen9blas_last_zero_copy(void);   // GEMM: bits 1 = A, 2 = B, 4 = C zero-copy

// Zero-copy memory (USM host, aligned to 4096 B): the GPU reads and writes pointers from gen9blas_alloc (including pointers
// into the interior of an allocation) directly. Other host memory goes through i915 userptr (kcl) or a CPU copy.
GEN9BLAS_API void *gen9blas_alloc(size_t bytes);
GEN9BLAS_API void gen9blas_free(void *p);
GEN9BLAS_API int gen9blas_is_usm(const void *p, size_t bytes);   // 1: [p, p+bytes) inside a single gen9blas_alloc allocation

// ---- Level 3 ----
// GEMM: C = alpha·op(A)·op(B) + beta·C; op(A): M×K (transA: A stored as K×M), op(B): K×N (transB: B stored as N×K).
GEN9BLAS_API int gen9blas_gemm(int type, int transA, int transB, int M, int N, int K, double alpha, const void *A, int lda,
                               const void *B, int ldb, double beta, void *C, int ldc);
// Strided batched GEMM: for b = 0..count-1 the matrices are A + b·sA, B + b·sB, C + b·sC (strides in elements).
// One kernel launch for the whole batch.
GEN9BLAS_API int gen9blas_gemm_batched(int type, int transA, int transB, int M, int N, int K, double alpha,
                                       const void *A, int lda, int64_t sA, const void *B, int ldb, int64_t sB,
                                       double beta, void *C, int ldc, int64_t sC, int count);
// Packed B kept on the GPU (e.g. constant weights): pack op(B) (K×N; transB: B stored as N×K) once, multiply many times.
// Packing runs in launches of at most GEN9BLAS_PACK_BAND_MB of packed data (default 128; 2.3-4 GB/s, so 30-55 ms each).
// gemm_pb: C = alpha·op(A)·Bp + beta·C with op(A) M×K, A of the type Bp was packed with (GEN9BLAS_S, _H, _HS, _D).
// pack_b: NULL on error (gen9blas_error()). packed_b_free waits for queued work first.
typedef struct gen9blas_packed_b gen9blas_packed_b;
GEN9BLAS_API gen9blas_packed_b *gen9blas_pack_b(int type, int transB, int K, int N, const void *B, int ldb);
GEN9BLAS_API int gen9blas_gemm_pb(int transA, int M, double alpha, const void *A, int lda, const gen9blas_packed_b *Bp,
                                  double beta, void *C, int ldc);
GEN9BLAS_API void gen9blas_packed_b_free(gen9blas_packed_b *Bp);
// SYMM: C = alpha·A·B + beta·C (left, A: M×M symmetric) or alpha·B·A + beta·C (right, A: N×N). The uplo triangle is used.
GEN9BLAS_API int gen9blas_symm(int type, int side, int uplo, int M, int N, double alpha, const void *A, int lda,
                               const void *B, int ldb, double beta, void *C, int ldc);
// SYRK: C = alpha·op(A)·op(A)ᵀ + beta·C (C: N×N; op(A) = A: N×K or Aᵀ, A: K×N). Only the uplo triangle is modified.
GEN9BLAS_API int gen9blas_syrk(int type, int uplo, int trans, int N, int K, double alpha, const void *A, int lda,
                               double beta, void *C, int ldc);
// SYR2K: C = alpha·(op(A)·op(B)ᵀ + op(B)·op(A)ᵀ) + beta·C. Only the uplo triangle is modified.
GEN9BLAS_API int gen9blas_syr2k(int type, int uplo, int trans, int N, int K, double alpha, const void *A, int lda,
                                const void *B, int ldb, double beta, void *C, int ldc);
// TRMM: B = alpha·op(A)·B (left, A: M×M) or alpha·B·op(A) (right, A: N×N); A triangular per uplo, diag = 1: unit.
GEN9BLAS_API int gen9blas_trmm(int type, int side, int uplo, int transA, int diag, int M, int N, double alpha,
                               const void *A, int lda, void *B, int ldb);
// TRSM: solve op(A)·X = alpha·B (left) or X·op(A) = alpha·B (right); X overwrites B. FP32, FP64 and FP16 (H, HS).
GEN9BLAS_API int gen9blas_trsm(int type, int side, int uplo, int transA, int diag, int M, int N, double alpha,
                               const void *A, int lda, void *B, int ldb);
// Shortcuts without transpose.
GEN9BLAS_API int gen9blas_hgemm(int M, int N, int K, double alpha, const uint16_t *A, int lda, const uint16_t *B, int ldb,
                                double beta, uint16_t *C, int ldc);
GEN9BLAS_API int gen9blas_sgemm(int M, int N, int K, double alpha, const float *A, int lda, const float *B, int ldb,
                                double beta, float *C, int ldc);

// ---- Levels 1-2 (type: GEN9BLAS_S, GEN9BLAS_D or GEN9BLAS_H; GEN9BLAS_HS = GEN9BLAS_H). FP16: computed in FP32, reductions in FP64.
// Vectors: n elements spaced inc apart (inc != 0; inc < 0 as in BLAS: pointer to the lowest address).
GEN9BLAS_API int gen9blas_axpy(int type, int n, double a, const void *x, int incx, void *y, int incy);    // y = a·x + y
GEN9BLAS_API int gen9blas_scal(int type, int n, double a, void *x, int incx);                             // x = a·x
GEN9BLAS_API int gen9blas_copy(int type, int n, const void *x, int incx, void *y, int incy);             // y = x
GEN9BLAS_API int gen9blas_swap(int type, int n, void *x, int incx, void *y, int incy);                   // x <-> y
GEN9BLAS_API int gen9blas_rot(int type, int n, void *x, int incx, void *y, int incy, double c, double s);  // Givens rotation
GEN9BLAS_API int gen9blas_rotm(int type, int n, void *x, int incx, void *y, int incy, const double *param);  // modified rotation
GEN9BLAS_API void gen9blas_rotg(float *a, float *b, float *c, float *s);                  // CPU: Givens parameters (a = r, b = z)
GEN9BLAS_API void gen9blas_rotmg(float *d1, float *d2, float *x1, float y1, float *param); // CPU: modified rotation parameters
GEN9BLAS_API int gen9blas_dot(int type, int n, const void *x, int incx, const void *y, int incy, double *result);
GEN9BLAS_API int gen9blas_asum(int type, int n, const void *x, int incx, double *result);                // Σ|x|
GEN9BLAS_API int gen9blas_nrm2(int type, int n, const void *x, int incx, double *result);                // ‖x‖₂
GEN9BLAS_API int gen9blas_iamax(int type, int n, const void *x, int incx, int64_t *result);              // 0-based (first max |x|)
// GEMV: y = alpha·op(A)·x + beta·y, A: M×N (trans: y has length N, x has length M). beta = 0: y is not read.
GEN9BLAS_API int gen9blas_gemv(int type, int trans, int M, int N, double alpha, const void *A, int lda, const void *x, int incx,
                               double beta, void *y, int incy);
// SYMV: y = alpha·A·x + beta·y, A symmetric n×n (the uplo triangle is used).
GEN9BLAS_API int gen9blas_symv(int type, int uplo, int n, double alpha, const void *A, int lda, const void *x, int incx,
                               double beta, void *y, int incy);
// TRMV: x = op(A)·x; TRSV: solve op(A)·x = b (b given in x). A triangular n×n (uplo; diag = 1: unit).
GEN9BLAS_API int gen9blas_trmv(int type, int uplo, int trans, int diag, int n, const void *A, int lda, void *x, int incx);
GEN9BLAS_API int gen9blas_trsv(int type, int uplo, int trans, int diag, int n, const void *A, int lda, void *x, int incx);
// SYR: A = alpha·x·xᵀ + A; SYR2: A = alpha·(x·yᵀ + y·xᵀ) + A. Only the uplo triangle is modified.
GEN9BLAS_API int gen9blas_syr(int type, int uplo, int n, double alpha, const void *x, int incx, void *A, int lda);
GEN9BLAS_API int gen9blas_syr2(int type, int uplo, int n, double alpha, const void *x, int incx, const void *y, int incy,
                               void *A, int lda);
// GER: A = alpha·x·yᵀ + A, A: M×N.
GEN9BLAS_API int gen9blas_ger(int type, int M, int N, double alpha, const void *x, int incx, const void *y, int incy,
                              void *A, int lda);

// ---- Level 2: banded and packed matrices, row-major layout as in CBLAS RowMajor ----
// Band: A(i,j) = A[i·lda + KL + j − i] for −KL ≤ j − i ≤ KU (lda ≥ KL + KU + 1); symmetric/triangular banded:
// upper KL = 0, KU = K, lower KL = K, KU = 0. Packed upper: row i (j = i..n−1) starts at i·n − i(i−1)/2; lower:
// row i (j = 0..i) starts at i(i+1)/2. TBSV/TPSV is computed by a single work-group row by row (inherently sequential).
GEN9BLAS_API int gen9blas_gbmv(int type, int trans, int M, int N, int KL, int KU, double alpha, const void *A, int lda,
                               const void *x, int incx, double beta, void *y, int incy);
GEN9BLAS_API int gen9blas_sbmv(int type, int uplo, int N, int K, double alpha, const void *A, int lda, const void *x, int incx,
                               double beta, void *y, int incy);
GEN9BLAS_API int gen9blas_tbmv(int type, int uplo, int trans, int diag, int N, int K, const void *A, int lda, void *x, int incx);
GEN9BLAS_API int gen9blas_tbsv(int type, int uplo, int trans, int diag, int N, int K, const void *A, int lda, void *x, int incx);
GEN9BLAS_API int gen9blas_spmv(int type, int uplo, int N, double alpha, const void *Ap, const void *x, int incx, double beta,
                               void *y, int incy);
GEN9BLAS_API int gen9blas_tpmv(int type, int uplo, int trans, int diag, int N, const void *Ap, void *x, int incx);
GEN9BLAS_API int gen9blas_tpsv(int type, int uplo, int trans, int diag, int N, const void *Ap, void *x, int incx);
GEN9BLAS_API int gen9blas_spr(int type, int uplo, int N, double alpha, const void *x, int incx, void *Ap);
GEN9BLAS_API int gen9blas_spr2(int type, int uplo, int N, double alpha, const void *x, int incx, const void *y, int incy,
                               void *Ap);

#ifdef __cplusplus
}
#endif
