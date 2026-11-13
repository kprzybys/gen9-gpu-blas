// libgen9blas: BLAS (levels 1–3) on Intel UHD 620 (Gen9.5) with hand-written assembly GEMM loops. See docs/api.md.
// All matrices are row-major. Elements: uint16_t (FP16) or float (FP32), pointers passed as void*.
// Result 0 = success, -1 = error (description: gen9blas_error()).
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

enum { GEN9BLAS_H = 0,    // FP16, FP16 accumulation (fastest)
       GEN9BLAS_S = 1,    // FP32
       GEN9BLAS_HS = 2,   // FP16, accumulated in chunks of GEN9BLAS_KC products into FP32 (more accurate)
       GEN9BLAS_D = 3 };  // FP64
#define GEN9BLAS_KC 32
enum { GEN9BLAS_LEFT = 0, GEN9BLAS_RIGHT = 1 };          // side
enum { GEN9BLAS_LOWER = 0, GEN9BLAS_UPPER = 1 };         // uplo
                                               // trans / transA: 0 = none, 1 = transpose; diag: 0 = regular, 1 = unit

int gen9blas_init(const char *dir);             // directory with gen_h.bin, gen_hd.bin, gen_s.bin, hgemm.cl, sgemm.cl, dgemm.cl, level12.cl

// Zero-copy memory (USM host, aligned to 4096 B): the GPU reads and writes pointers from gen9blas_alloc (including pointers
// into the interior of an allocation) directly. Other host memory is copied by the CPU on every call. After gen9blas_init.
void *gen9blas_alloc(size_t bytes);
void gen9blas_free(void *p);
int gen9blas_is_usm(const void *p, size_t bytes);   // 1: [p, p+bytes) inside a single gen9blas_alloc allocation

// GEMM: C = alpha·op(A)·op(B) + beta·C; op(A): M×K (transA: A stored as K×M), op(B): K×N (transB: B stored as N×K).
int gen9blas_gemm(int type, int transA, int transB, int M, int N, int K, double alpha, const void *A, int lda,
             const void *B, int ldb, double beta, void *C, int ldc);
// Strided batched GEMM: for b = 0..count-1 the matrices are A + b·sA, B + b·sB, C + b·sC (strides in elements).
// One kernel launch for the whole batch.
int gen9blas_gemm_batched(int type, int transA, int transB, int M, int N, int K, double alpha,
                     const void *A, int lda, long sA, const void *B, int ldb, long sB,
                     double beta, void *C, int ldc, long sC, int count);
// Packed B kept on the GPU (e.g. constant weights): pack op(B) (K×N; transB: B stored as N×K) once, multiply many times.
// Packing runs in launches of at most GEN9BLAS_PACK_BAND_MB of packed data (default 128; 2.3-4 GB/s, so 30-55 ms each).
// gemm_pb: C = alpha·op(A)·Bp + beta·C with op(A) M×K. Same types as gen9blas_gemm whose A and B element types are
// equal (GEN9BLAS_S, GEN9BLAS_H, GEN9BLAS_HS, GEN9BLAS_D). NULL / -1 on error (gen9blas_error()).
typedef struct gen9blas_packed gen9blas_packed;
gen9blas_packed *gen9blas_pack_b(int type, int transB, int K, int N, const void *B, int ldb);
int gen9blas_gemm_pb(int type, int transA, int M, double alpha, const void *A, int lda, const gen9blas_packed *Bp,
                     double beta, void *C, int ldc);
void gen9blas_packed_free(gen9blas_packed *Bp);
// ggml backend: C (F32, M x N) = A (F32, M x K) . Bp with Bp = gen9blas_pack_b(GEN9BLAS_HS, ...) of FP16 weights.
// A and the result are rounded to FP16 once each (accurate FP16 GEMM: FP32 sums of 32-product FP16 chunks).
// *nonfinite = 1 if any result is inf / NaN (FP16 overflow): the caller should recompute in F32.
int gen9blas_gemm_pb_f16w(int M, const float *A, int lda, const gen9blas_packed *Bp, float *C, int ldc, int *nonfinite);
// ggml backend, gated FFN with FP16 weights kept on the GPU: Y = (silu(X Wg) * (X Wu)) Wd (F32 X: M x K, Y: M x Nd);
// Pg, Pu: K -> N, Pd: N -> Nd, all packed as GEN9BLAS_HS. Intermediates stay FP16 on the GPU; *nonfinite as above.
// ggml backend, per-layer elementwise ops on host F32 memory. Queued without a wait (sync = 0) when every operand is
// zero-copy: they run in the batch of the next call that waits (or gen9blas_flush); the caller must register the host
// buffers involved with gen9blas_span first (else regions of different operations may alias). sync = 1 or a copied
// operand: waits before returning. 0 or -1 (gen9blas_error()).
int gen9blas_span(const void *p, size_t bytes, int writes);   // 0 registered, -1 not (then use sync = 1)
void gen9blas_span_clear(void);
int gen9blas_flush(void);
// Y = X / sqrt(mean(X^2) + eps) [* w], M rows of K (w = NULL: no weight).
int gen9blas_rms_norm(int M, int K, const float *X, int ldx, const float *w, float eps, float *Y, int ldy, int sync);
// C = A + B (op 0) or A * B (op 1), M rows of K; ldb = 0: B is one row broadcast to all rows.
int gen9blas_binop_rows(int op, int M, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc, int sync);
// ggml ROPE forward, modes NORMAL (neox 0) / NEOX (neox 1), rotating all n_dims = head size elements: X, Y element
// (i, h, t) at t * s2 + h * s1 + i floats; pos[n_tok]; ff: freq factors (n_dims/2) or NULL; corr: ggml_rope_yarn_corr_dims.
int gen9blas_rope(int n_dims, int nh, int n_tok, const float *X, long xs1, long xs2, float *Y, long ys1, long ys2,
                  const int *pos, const float *ff, int neox, float freq_base, float freq_scale, float ext_factor,
                  float attn_factor, const float corr[2], int sync);
int gen9blas_ffn_f16w(int M, const float *X, int ldx, const gen9blas_packed *Pg, const gen9blas_packed *Pu,
                      const gen9blas_packed *Pd, float *Y, int ldy, int *nonfinite);
// ggml backend, quantized weights: N rows of K values in ggml block layout (row n at W + n * row_bytes; qtype
// GEN9BLAS_Q8_0: K % 32 == 0, GEN9BLAS_Q4_K / GEN9BLAS_Q6_K: K % 256 == 0) dequantized on the GPU into a packed B for
// gen9blas_gemm_pb_f16w (GEN9BLAS_HS, as gen9blas_pack_b(GEN9BLAS_HS, 1, K, N, ...) of the FP16 values; bit-identical to
// ggml's dequantize_row_* + round-to-nearest-even FP16). *out_of_range = 1 if any value does not fit FP16: the result
// is then unusable (free it; use FP32). NULL on error or if the kernel is unavailable.
enum { GEN9BLAS_Q8_0 = 0, GEN9BLAS_Q4_K = 1, GEN9BLAS_Q6_K = 2 };
gen9blas_packed *gen9blas_pack_b_q(int qtype, int K, int N, const void *W, size_t row_bytes, int *out_of_range);
// SYMM: C = alpha·A·B + beta·C (left, A: M×M symmetric) or alpha·B·A + beta·C (right, A: N×N). The uplo triangle is used.
int gen9blas_symm(int type, int side, int uplo, int M, int N, double alpha, const void *A, int lda,
             const void *B, int ldb, double beta, void *C, int ldc);
// SYRK: C = alpha·op(A)·op(A)ᵀ + beta·C (C: N×N; op(A) = A: N×K or Aᵀ, A: K×N). Only the uplo triangle is modified.
int gen9blas_syrk(int type, int uplo, int trans, int N, int K, double alpha, const void *A, int lda, double beta, void *C, int ldc);
// SYR2K: C = alpha·(op(A)·op(B)ᵀ + op(B)·op(A)ᵀ) + beta·C. Only the uplo triangle is modified.
int gen9blas_syr2k(int type, int uplo, int trans, int N, int K, double alpha, const void *A, int lda,
              const void *B, int ldb, double beta, void *C, int ldc);
// TRMM: B = alpha·op(A)·B (left, A: M×M) or alpha·B·op(A) (right, A: N×N); A triangular per uplo, diag = 1: unit.
int gen9blas_trmm(int type, int side, int uplo, int transA, int diag, int M, int N, double alpha,
             const void *A, int lda, void *B, int ldb);
// TRSM: solve op(A)·X = alpha·B (left) or X·op(A) = alpha·B (right); X overwrites B. FP32, FP64 and FP16 (GEN9BLAS_H, GEN9BLAS_HS).
int gen9blas_trsm(int type, int side, int uplo, int transA, int diag, int M, int N, double alpha,
             const void *A, int lda, void *B, int ldb);

// Shortcuts without transpose.
int gen9blas_hgemm(int M, int N, int K, double alpha, const uint16_t *A, int lda, const uint16_t *B, int ldb,
              double beta, uint16_t *C, int ldc);
int gen9blas_sgemm(int M, int N, int K, double alpha, const float *A, int lda, const float *B, int ldb,
              double beta, float *C, int ldc);

// ---- Levels 1–2 (type: GEN9BLAS_S, GEN9BLAS_D or GEN9BLAS_H; GEN9BLAS_HS = GEN9BLAS_H). FP16: computed in FP32, reductions in FP64. ----
// Vectors: n elements spaced inc apart (inc != 0; inc < 0 as in BLAS: pointer to the lowest address).
int gen9blas_axpy(int type, int n, double a, const void *x, int incx, void *y, int incy);    // y = a·x + y
int gen9blas_scal(int type, int n, double a, void *x, int incx);                             // x = a·x
int gen9blas_copy(int type, int n, const void *x, int incx, void *y, int incy);             // y = x
int gen9blas_swap(int type, int n, void *x, int incx, void *y, int incy);                   // x <-> y
int gen9blas_rot(int type, int n, void *x, int incx, void *y, int incy, double c, double s);  // Givens rotation
int gen9blas_rotm(int type, int n, void *x, int incx, void *y, int incy, const double *param);  // modified rotation
void gen9blas_rotg(float *a, float *b, float *c, float *s);                  // CPU: Givens rotation parameters (a = r, b = z)
void gen9blas_rotmg(float *d1, float *d2, float *x1, float y1, float *param); // CPU: modified rotation parameters
int gen9blas_dot(int type, int n, const void *x, int incx, const void *y, int incy, double *result);
int gen9blas_asum(int type, int n, const void *x, int incx, double *result);                // Σ|x|
int gen9blas_nrm2(int type, int n, const void *x, int incx, double *result);                // ‖x‖₂
int gen9blas_iamax(int type, int n, const void *x, int incx, long *result);                 // 0-based index (first max |x|)
// GEMV: y = alpha·op(A)·x + beta·y, A: M×N (trans: y has length N, x has length M). beta = 0: y is not read.
int gen9blas_gemv(int type, int trans, int M, int N, double alpha, const void *A, int lda, const void *x, int incx,
             double beta, void *y, int incy);
// SYMV: y = alpha·A·x + beta·y, A symmetric n×n (the uplo triangle is used).
int gen9blas_symv(int type, int uplo, int n, double alpha, const void *A, int lda, const void *x, int incx,
             double beta, void *y, int incy);
// TRMV: x = op(A)·x; TRSV: solve op(A)·x = b (b given in x). A triangular n×n (uplo; diag = 1: unit).
int gen9blas_trmv(int type, int uplo, int trans, int diag, int n, const void *A, int lda, void *x, int incx);
int gen9blas_trsv(int type, int uplo, int trans, int diag, int n, const void *A, int lda, void *x, int incx);
// SYR: A = alpha·x·xᵀ + A; SYR2: A = alpha·(x·yᵀ + y·xᵀ) + A. Only the uplo triangle is modified.
int gen9blas_syr(int type, int uplo, int n, double alpha, const void *x, int incx, void *A, int lda);
int gen9blas_syr2(int type, int uplo, int n, double alpha, const void *x, int incx, const void *y, int incy, void *A, int lda);
// GER: A = alpha·x·yᵀ + A, A: M×N.
int gen9blas_ger(int type, int M, int N, double alpha, const void *x, int incx, const void *y, int incy, void *A, int lda);

// ---- Level 2: banded and packed matrices, row-major layout as in CBLAS RowMajor ----
// Band: A(i,j) = A[i·lda + KL + j − i] for −KL ≤ j − i ≤ KU (lda ≥ KL + KU + 1); symmetric/triangular banded:
// upper KL = 0, KU = K, lower KL = K, KU = 0. Packed upper: row i (j = i..n−1) starts at i·n − i(i−1)/2; lower:
// row i (j = 0..i) starts at i(i+1)/2. TBSV/TPSV is computed by a single work-group row by row (inherently sequential).
int gen9blas_gbmv(int type, int trans, int M, int N, int KL, int KU, double alpha, const void *A, int lda, const void *x, int incx,
             double beta, void *y, int incy);
int gen9blas_sbmv(int type, int uplo, int N, int K, double alpha, const void *A, int lda, const void *x, int incx, double beta, void *y, int incy);
int gen9blas_tbmv(int type, int uplo, int trans, int diag, int N, int K, const void *A, int lda, void *x, int incx);
int gen9blas_tbsv(int type, int uplo, int trans, int diag, int N, int K, const void *A, int lda, void *x, int incx);
int gen9blas_spmv(int type, int uplo, int N, double alpha, const void *Ap, const void *x, int incx, double beta, void *y, int incy);
int gen9blas_tpmv(int type, int uplo, int trans, int diag, int N, const void *Ap, void *x, int incx);
int gen9blas_tpsv(int type, int uplo, int trans, int diag, int N, const void *Ap, void *x, int incx);
int gen9blas_spr(int type, int uplo, int N, double alpha, const void *x, int incx, void *Ap);
int gen9blas_spr2(int type, int uplo, int N, double alpha, const void *x, int incx, const void *y, int incy, void *Ap);

int gen9blas_available(int type);         // 1 if the kernel of the given type has been loaded
double gen9blas_last_time_ms(void);     // GEN9BLAS_PROFILE=1: GEMM kernel time of the last call; without it -- time of the whole call
int gen9blas_last_zero_copy(void);      // GEMM: bits 1 = A, 2 = B, 4 = C zero-copy (memory from gen9blas_alloc)
const char *gen9blas_error(void);

// ---- Flash attention for the ggml backend (lib/gen9blas_attn.c, docs/attention-gpu.md section 11; kcl build only) ----
// llama.cpp layouts: Q and O F32 [token][H][D]; K and V F16 rows of HKV*D values (row j, head h at h*D), the first nk
// rows used (nk a multiple of 16); k_bytes / v_bytes: bytes from K / V to the end of the whole KV tensor (pinned once).
// Token t attends to keys 0 .. min(nk, vis0 + dvis*t) - 1 (dvis 1: causal, 0: the same count for all), scores scaled
// by scale, FP32 accumulation. Shapes with a built-in binary: (D, H/HKV, HKV) = (128, 6, 2), (64, 4, 8) and (64, 1, 20).
// Returns 0 (done), 1 (not handled: unsupported shape or memory not mappable; nothing was written), -1 (GPU error:
// O is undefined, recompute it). *launches (may be NULL): number of k_attn launches (split at ~2e10 FLOP).
int gen9blas_attn_supported(int D, int H, int HKV);
int gen9blas_attn_f16kv(int D, int H, int HKV, int nq, int nk, const float *Q, const uint16_t *K, size_t k_bytes,
                        const uint16_t *V, size_t v_bytes, float *O, float scale, int vis0, int dvis, int *launches);
const char *gen9blas_attn_error(void);

#ifdef __cplusplus
}
#endif
