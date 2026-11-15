// libgen9blas, neural-network operations for the ggml backend and applications: GEMM with FP16 / quantized weights
// kept on the GPU, fused epilogues, per-layer elementwise ops with ggml semantics, attention.
// UNSTABLE: this part follows ggml and may change in any 0.x release; check gen9blas_version() at run time.
// Conventions (results, threads, handle lifetime) as in gen9blas.h.
#pragma once
#include "gen9blas.h"
#ifdef __cplusplus
extern "C" {
#endif

// ---- GEMM with FP16 weights packed once (Bp = gen9blas_pack_b(GEN9BLAS_HS, ...) or gen9blas_pack_b_q) ----
// C (F32, M x N) = A (F32, M x K) . Bp. A and the result are rounded to FP16 once each (accurate FP16 GEMM: FP32 sums of
// 32-product FP16 chunks). *nonfinite = 1 if any result is inf / NaN (FP16 overflow): the caller should recompute in F32.
GEN9BLAS_API int gen9blas_gemm_pb_f16w(int M, const float *A, int lda, const gen9blas_packed_b *Bp, float *C, int ldc,
                                       int *nonfinite);
// the same with FP16 activations (lda in elements)
GEN9BLAS_API int gen9blas_gemm_pb_f16w_h(int M, const uint16_t *A, int lda, const gen9blas_packed_b *Bp, float *C, int ldc,
                                         int *nonfinite);
// gemm_pb_f16w with ops fused into the epilogue (whisper encoder): C = [gelu](A.Bp + bias) [+ R]. bias: N floats or NULL;
// gelu: 0/1 (ggml GELU, tanh form with FP16 rounding as ggml-cpu); R: M x N (row stride ldr) or NULL, may be C itself with
// ldr == ldc. Bit-identical to gemm_pb_f16w followed by the separate ADD / GELU kernels. *nonfinite as gemm_pb_f16w (raw
// GEMM value): then recompute unfused. Needs GEN9BLAS_FEAT_EPILOG_EXT.
GEN9BLAS_API int gen9blas_gemm_pb_f16w_ex(int M, const float *A, int lda, const gen9blas_packed_b *Bp, float *C, int ldc,
                                          const float *bias, int gelu, const float *R, int ldr, int *nonfinite);
// Small batches, 1 <= M <= 16 (several sequences decoded together): C = A . Bp with F32 activations read directly (no
// packing), FP32 products and sums, F32 result. Returns 1 (nothing done) if the kernels are missing or M is out of range.
GEN9BLAS_API int gen9blas_gemm_pb_skinny(int M, const float *A, int lda, const gen9blas_packed_b *Bp, float *C, int ldc,
                                         int *nonfinite);
// n (1..8) products of one F32 input with HS-packed weights (Q/K/V), A packed once, one wait; nonfinite[i] per output.
// Returns 1 if not applicable (an output not zero-copy): compute them one by one.
GEN9BLAS_API int gen9blas_gemm_multi_f16w(int M, const float *A, int lda, int n, const gen9blas_packed_b *const *P,
                                          float *const *C, const int *ldc, int *nonfinite);
// Gated FFN: Y = (silu(X Wg) * (X Wu)) Wd (F32 X: M x K, Y: M x Nd); Pg, Pu: K -> N, Pd: N -> Nd, all packed as
// GEN9BLAS_HS. Intermediates stay FP16 on the GPU; *nonfinite as above.
GEN9BLAS_API int gen9blas_ffn_f16w(int M, const float *X, int ldx, const gen9blas_packed_b *Pg, const gen9blas_packed_b *Pu,
                                   const gen9blas_packed_b *Pd, float *Y, int ldy, int *nonfinite);
// Quantized weights: N rows of K values in ggml block layout (row n at W + n * row_bytes; qtype GEN9BLAS_Q8_0: K % 32 == 0,
// GEN9BLAS_Q4_K / GEN9BLAS_Q6_K: K % 256 == 0) dequantized on the GPU into a packed B for gemm_pb_f16w (as
// gen9blas_pack_b(GEN9BLAS_HS, 1, K, N, ...) of the FP16 values; bit-identical to ggml's dequantize_row_* + RTE to FP16).
// *out_of_range = 1 if any value does not fit FP16: the result is unusable (free it; use FP32). NULL on error / no kernel.
enum { GEN9BLAS_Q8_0 = 0, GEN9BLAS_Q4_K = 1, GEN9BLAS_Q6_K = 2 };
GEN9BLAS_API gen9blas_packed_b *gen9blas_pack_b_q(int qtype, int K, int N, const void *W, size_t row_bytes, int *out_of_range);

// A GEMM input packed once (FP16 A3 tiles) and used by several gen9blas_gemm_pa_f16w_ex calls: whisper's LayerNorm writes
// it while normalising (gen9blas_layer_norm_pack: same Y as gen9blas_layer_norm plus the packed copy), Q / K / V / fc1 then
// skip their own packing. Bit-identical GEMM inputs. Needs GEN9BLAS_FEAT_NORM_PACK. packed_a_free does not wait: flush first.
typedef struct gen9blas_packed_a gen9blas_packed_a;
GEN9BLAS_API gen9blas_packed_a *gen9blas_packed_a_create(void);
GEN9BLAS_API void gen9blas_packed_a_free(gen9blas_packed_a *pa);
GEN9BLAS_API int gen9blas_layer_norm_pack(int M, int K, const float *X, int ldx, const float *w, const float *b, float eps,
                                          float *Y, int ldy, gen9blas_packed_a *pa, int sync);
GEN9BLAS_API int gen9blas_gemm_pa_f16w_ex(const gen9blas_packed_a *pa, const gen9blas_packed_b *Bp, float *C, int ldc,
                                          const float *bias, int gelu, const float *R, int ldr, int *nonfinite);

// ---- Queued work ----
// No-wait mode for chains of small products (decoding several sequences): while on, gemm_pb_f16w(_ex), ffn_f16w and
// gemm_multi_f16w queue their work and return without waiting whenever their output is zero-copy (USM, userptr or a span),
// with *nonfinite left 0; the GPU sets one flag slot per call on FP16 overflow. After gen9blas_flush() (GPU idle),
// gen9blas_nowait_pending() returns how many calls overflowed and clears the slots; the caller must then recompute (or
// fail) everything that used those results.
GEN9BLAS_API void gen9blas_set_nowait(int on);
GEN9BLAS_API int gen9blas_nowait_pending(void);
// Elementwise ops below run on host F32 memory and are queued without a wait (sync = 0) when every operand is zero-copy:
// they run in the batch of the next call that waits (or gen9blas_flush). The caller must register the host buffers
// involved with gen9blas_span first (else regions of different operations may alias). sync = 1 or a copied operand:
// waits before returning.
GEN9BLAS_API int gen9blas_span(const void *p, size_t bytes, int writes);   // 0 registered, -1 not (then use sync = 1)
GEN9BLAS_API void gen9blas_span_clear(void);
GEN9BLAS_API int gen9blas_flush(void);

// ---- Elementwise ops (ggml semantics, bit-identical to ggml-cpu unless stated) ----
// Y = X / sqrt(mean(X^2) + eps) [* w], M rows of K (w = NULL: no weight).
GEN9BLAS_API int gen9blas_rms_norm(int M, int K, const float *X, int ldx, const float *w, float eps, float *Y, int ldy, int sync);
// ggml NORM (mean / variance over K) [* w] [+ b] (w, b: one row of K or NULL).
GEN9BLAS_API int gen9blas_layer_norm(int M, int K, const float *X, int ldx, const float *w, const float *b, float eps,
                                     float *Y, int ldy, int sync);
// C = A + B (op 0) or A * B (op 1), M rows of K. ldb = 0: B is one row broadcast to all rows; bs0 = 0: B has one value
// per row (broadcast along the row: ggml conv bias [1, rows]); bs0 = 1: B has K values per row.
GEN9BLAS_API int gen9blas_binop_rows(int op, int M, int K, const float *A, int lda, const float *B, int ldb, int bs0,
                                     float *C, int ldc, int sync);
// ggml GELU (tanh form, FP16 table semantics of ggml-cpu) / GELU_ERF (exact erf form, F32) / RELU over n contiguous
// floats; X == Y allowed.
GEN9BLAS_API int gen9blas_gelu(size_t n, const float *X, float *Y, int sync);
GEN9BLAS_API int gen9blas_gelu_erf(size_t n, const float *X, float *Y, int sync);
GEN9BLAS_API int gen9blas_relu(size_t n, const float *X, float *Y, int sync);
// ggml IM2COL: X [N, IC, IH, IW] (image / channel offsets ofs0 / ofs1 in floats, rows of IW contiguous) -> Y [N, OH, OW,
// IC*KH*KW], F32 or FP16 (f16, RTE); zero padding. 1-D convolution: IH = KH = OH = 1.
GEN9BLAS_API int gen9blas_im2col(int N, int IC, int IH, int IW, const float *X, int64_t ofs0, int64_t ofs1, int KH, int KW,
                                 int OH, int OW, int s0, int s1, int p0, int p1, int d0, int d1, void *Y, int f16, int sync);
// F32 rows x H heads x D (element (r, h, d) at X[r * s_row + h * s_head + d]) -> FP16 [r][h][d] contiguous, RTE, saturated.
GEN9BLAS_API int gen9blas_cvt_f16_heads(int rows, int H, int D, const float *X, int64_t s_row, int64_t s_head, uint16_t *Y,
                                        int sync);
// ggml ROPE forward, modes NORMAL (neox 0) / NEOX (neox 1), rotating all n_dims = head size elements: X, Y element
// (i, h, t) at t * s2 + h * s1 + i floats; pos[n_tok]; ff: freq factors (n_dims/2) or NULL; corr: ggml_rope_yarn_corr_dims.
GEN9BLAS_API int gen9blas_rope(int n_dims, int nh, int n_tok, const float *X, int64_t xs1, int64_t xs2, float *Y, int64_t ys1,
                               int64_t ys2, const int *pos, const float *ff, int neox, float freq_base, float freq_scale,
                               float ext_factor, float attn_factor, const float corr[2], int sync);
// ggml_rope_multi, mode GGML_ROPE_TYPE_MROPE (vision = 0: sections[4], pos = 4 * n_tok positions, NEOX pair order, pair j
// takes the position of its sector j % sum(sections)) or GGML_ROPE_TYPE_VISION (vision = 1: n_dims = ne0 / 2, pairs
// (i, i + n_dims), power of theta reset per section). Needs GEN9BLAS_FEAT_ROPE_MULTI.
GEN9BLAS_API int gen9blas_rope_multi(int n_dims, int nh, int n_tok, const float *X, int64_t xs1, int64_t xs2, float *Y,
                                     int64_t ys1, int64_t ys2, const int *pos, const float *ff, const int sections[4],
                                     float freq_base, float freq_scale, float ext_factor, float attn_factor,
                                     const float corr[2], int vision, int sync);
// ggml CPY / CONT / DUP: src (type tin) -> dst (type tout), types GEN9BLAS_S (F32) or GEN9BLAS_H (F16; F32 -> F16 RTE);
// ne / nb as ggml (nb in bytes, any non-negative strides); dst shape may differ from src with the same element count
// (row-major order). Needs GEN9BLAS_FEAT_CPY.
GEN9BLAS_API int gen9blas_cpy_any(const void *src, int tin, const int64_t sne[4], const int64_t snb[4], void *dst, int tout,
                                  const int64_t dne[4], const int64_t dnb[4], int sync);
// ggml UPSCALE (ggml_interpolate) F32, mode_flags as ggml: 0 nearest, 1 bilinear, | 1<<8 align corners, | 1<<9 antialias.
GEN9BLAS_API int gen9blas_upscale_f32(const float *src, const int64_t sne[4], const int64_t snb[4], float *dst,
                                      const int64_t dne[4], const int64_t dnb[4], int mode_flags, int sync);
// ggml SET_ROWS with F32 source rows and an F16 destination (llama.cpp's KV-cache writes): dst row idx[i] of plane
// (i02, i03) = src row i, RTE rounding. idx int64 (idx64 = 1) or int32, byte strides b10/b11/b12, broadcast over planes
// as ggml-cpu. dst must be zero-copy (USM or userptr). sync: wait and report *bad (index out of range).
// Needs GEN9BLAS_FEAT_SET_ROWS.
GEN9BLAS_API int gen9blas_set_rows_f16(int nc, int nr, int ne02, int ne03, const float *src, int64_t s1, int64_t s2,
                                       int64_t s3, const void *idx, size_t idx_bytes, int64_t b10, int64_t b11, int64_t b12,
                                       int idx64, int ne11, int ne12, uint16_t *dst, size_t dst_bytes, int64_t d1,
                                       int64_t d2, int64_t d3, int64_t ne1, int sync, int *bad);

// ---- Attention ----
// Decoding over KV streams (llama-server -np N), D 128, G = H / HKV <= 8: Q F32 element (d, token t, head h, stream s)
// at Q[s*qs + t*qt + h*qh + d]; K / V F16 in place, element (d, key k, KV head g, stream s) at [s*ks + k*kk + g*kh + d]
// (k_bytes / v_bytes: readable extent); O F32 [s][t][h][d] contiguous. Token t of stream s sees keys [0, min(vis[s] +
// dvis*t, nk)). chunk: keys per work-group (multiple of 16). Queued; sync = 1 waits. Split-K + merge (two kernels).
GEN9BLAS_API int gen9blas_attn_dec_supported(int D, int H, int HKV);
GEN9BLAS_API int gen9blas_attn_dec(int D, int H, int HKV, int ns, int nq, int nk, const float *Q, int64_t qt, int64_t qh,
                                   int64_t qs, const uint16_t *K, int64_t kk, int64_t kh, int64_t ks, size_t k_bytes,
                                   const uint16_t *V, int64_t vk, int64_t vh, int64_t vs, size_t v_bytes, float *O,
                                   float scale, const int *vis, int dvis, int chunk, int sync);
// Flash attention (prefill; docs/attention-gpu.md section 11; kcl build only). llama.cpp layouts: Q and O F32 [token][H][D];
// K and V F16 rows of HKV*D values (row j, head h at h*D), the first nk rows used (nk a multiple of 16); k_bytes / v_bytes:
// bytes from K / V to the end of the whole KV tensor (pinned once). Token t attends to keys 0 .. min(nk, vis0 + dvis*t) - 1
// (dvis 1: causal, 0: the same count for all), scores scaled by scale, FP32 accumulation. Shapes: those with a built-in
// binary (gen9blas_attn_supported). Returns 0 (done), GEN9BLAS_DECLINED (not handled: unsupported shape or memory not
// mappable; nothing was written), < 0 (error: O is undefined, recompute it). *launches (may be NULL): k_attn launches (split at ~2e10 FLOP).
GEN9BLAS_API int gen9blas_attn_supported(int D, int H, int HKV);
GEN9BLAS_API int gen9blas_attn_f16kv(int D, int H, int HKV, int nq, int nk, const float *Q, const uint16_t *K, size_t k_bytes,
                                     const uint16_t *V, size_t v_bytes, float *O, float scale, int vis0, int dvis,
                                     int *launches);
// Several KV streams (ggml ne[3]) in one operation: stream s has Q/O at Q + s*sq (floats, rows [t][h][d]), K/V at
// K + s*sk / V + s*sv (halves, rows of HKV*D), visible keys min(nk, vis0[s] + dvis*t). One k_kt, one k_attn launch over all
// streams, one wait. Returns 1 (not handled) if no stream binary exists for (D, H/HKV, HKV), the layout does not fit or the
// work would need more than one launch; -1 on a GPU error.
GEN9BLAS_API int gen9blas_attn_streams_supported(int D, int H, int HKV);
GEN9BLAS_API int gen9blas_attn_f16kv_streams(int D, int H, int HKV, int ns, int nq, int nk, const float *Q, int64_t sq,
                                             const uint16_t *K, int64_t sk, size_t k_bytes, const uint16_t *V, int64_t sv,
                                             size_t v_bytes, float *O, float scale, const int *vis0, int dvis, int *launches);

#ifdef __cplusplus
}
#endif
