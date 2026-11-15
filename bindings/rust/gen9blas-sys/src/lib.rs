//! Raw FFI bindings to libgen9blas 0.2 (`lib/gen9blas.h`, `lib/gen9blas_ml.h`). Hand-written; kept in sync with the
//! headers by `tools/check_rust_ffi.py`. Everything here is `unsafe`; the safe API is the `gen9blas` crate.
//!
//! Conventions of the C library: row-major matrices; results are `gen9blas_status` values (0 ok, 1 declined, < 0
//! error); calls must be serialized by the caller (process-wide state).
#![allow(non_camel_case_types)]
#![no_std]

use core::ffi::{c_char, c_double, c_float, c_int, c_void};

pub const GEN9BLAS_VERSION_MAJOR: c_int = 0;
pub const GEN9BLAS_VERSION_MINOR: c_int = 2;
pub const GEN9BLAS_VERSION_PATCH: c_int = 0;
pub const GEN9BLAS_VERSION: c_int = GEN9BLAS_VERSION_MAJOR * 10000 + GEN9BLAS_VERSION_MINOR * 100 + GEN9BLAS_VERSION_PATCH;

// gen9blas_status
pub const GEN9BLAS_OK: c_int = 0;
pub const GEN9BLAS_DECLINED: c_int = 1;
pub const GEN9BLAS_ERR: c_int = -1;
pub const GEN9BLAS_ERR_UNSUPPORTED: c_int = -2;
pub const GEN9BLAS_ERR_INVALID_ARG: c_int = -3;
pub const GEN9BLAS_ERR_OUT_OF_MEMORY: c_int = -4;
pub const GEN9BLAS_ERR_NO_DEVICE: c_int = -5;

// element types
pub const GEN9BLAS_H: c_int = 0;
pub const GEN9BLAS_S: c_int = 1;
pub const GEN9BLAS_HS: c_int = 2;
pub const GEN9BLAS_D: c_int = 3;
pub const GEN9BLAS_KC: c_int = 32;
pub const GEN9BLAS_NO_TRANS: c_int = 0;
pub const GEN9BLAS_TRANS: c_int = 1;
pub const GEN9BLAS_LEFT: c_int = 0;
pub const GEN9BLAS_RIGHT: c_int = 1;
pub const GEN9BLAS_LOWER: c_int = 0;
pub const GEN9BLAS_UPPER: c_int = 1;
pub const GEN9BLAS_NON_UNIT: c_int = 0;
pub const GEN9BLAS_UNIT: c_int = 1;

// gen9blas_has features
pub const GEN9BLAS_FEAT_FP16: c_int = 0;
pub const GEN9BLAS_FEAT_FP32: c_int = 1;
pub const GEN9BLAS_FEAT_FP64: c_int = 2;
pub const GEN9BLAS_FEAT_SKINNY_PB: c_int = 3;
pub const GEN9BLAS_FEAT_EPILOG_EXT: c_int = 4;
pub const GEN9BLAS_FEAT_NORM_PACK: c_int = 5;
pub const GEN9BLAS_FEAT_ROPE_MULTI: c_int = 6;
pub const GEN9BLAS_FEAT_CPY: c_int = 7;
pub const GEN9BLAS_FEAT_SET_ROWS: c_int = 8;

// quantized weight types (gen9blas_pack_b_q)
pub const GEN9BLAS_Q8_0: c_int = 0;
pub const GEN9BLAS_Q4_K: c_int = 1;
pub const GEN9BLAS_Q6_K: c_int = 2;

#[repr(C)]
pub struct gen9blas_packed_b {
    _private: [u8; 0],
}
#[repr(C)]
pub struct gen9blas_packed_a {
    _private: [u8; 0],
}

extern "C" {
    // ---- library ----
    pub fn gen9blas_version() -> c_int;
    pub fn gen9blas_init() -> c_int;
    pub fn gen9blas_device_gen() -> c_int;
    pub fn gen9blas_has(feature: c_int) -> c_int;
    pub fn gen9blas_error() -> *const c_char;
    pub fn gen9blas_last_status() -> c_int;
    pub fn gen9blas_last_time_ms() -> c_double;
    pub fn gen9blas_last_zero_copy() -> c_int;
    pub fn gen9blas_alloc(bytes: usize) -> *mut c_void;
    pub fn gen9blas_free(p: *mut c_void);
    pub fn gen9blas_is_usm(p: *const c_void, bytes: usize) -> c_int;

    // ---- level 3 ----
    pub fn gen9blas_gemm(type_: c_int, trans_a: c_int, trans_b: c_int, m: c_int, n: c_int, k: c_int, alpha: c_double,
                         a: *const c_void, lda: c_int, b: *const c_void, ldb: c_int, beta: c_double, c: *mut c_void,
                         ldc: c_int) -> c_int;
    pub fn gen9blas_gemm_batched(type_: c_int, trans_a: c_int, trans_b: c_int, m: c_int, n: c_int, k: c_int,
                                 alpha: c_double, a: *const c_void, lda: c_int, s_a: i64, b: *const c_void, ldb: c_int,
                                 s_b: i64, beta: c_double, c: *mut c_void, ldc: c_int, s_c: i64, count: c_int) -> c_int;
    pub fn gen9blas_pack_b(type_: c_int, trans_b: c_int, k: c_int, n: c_int, b: *const c_void, ldb: c_int)
                           -> *mut gen9blas_packed_b;
    pub fn gen9blas_gemm_pb(trans_a: c_int, m: c_int, alpha: c_double, a: *const c_void, lda: c_int,
                            bp: *const gen9blas_packed_b, beta: c_double, c: *mut c_void, ldc: c_int) -> c_int;
    pub fn gen9blas_packed_b_free(bp: *mut gen9blas_packed_b);
    pub fn gen9blas_symm(type_: c_int, side: c_int, uplo: c_int, m: c_int, n: c_int, alpha: c_double, a: *const c_void,
                         lda: c_int, b: *const c_void, ldb: c_int, beta: c_double, c: *mut c_void, ldc: c_int) -> c_int;
    pub fn gen9blas_syrk(type_: c_int, uplo: c_int, trans: c_int, n: c_int, k: c_int, alpha: c_double, a: *const c_void,
                         lda: c_int, beta: c_double, c: *mut c_void, ldc: c_int) -> c_int;
    pub fn gen9blas_syr2k(type_: c_int, uplo: c_int, trans: c_int, n: c_int, k: c_int, alpha: c_double, a: *const c_void,
                          lda: c_int, b: *const c_void, ldb: c_int, beta: c_double, c: *mut c_void, ldc: c_int) -> c_int;
    pub fn gen9blas_trmm(type_: c_int, side: c_int, uplo: c_int, trans_a: c_int, diag: c_int, m: c_int, n: c_int,
                         alpha: c_double, a: *const c_void, lda: c_int, b: *mut c_void, ldb: c_int) -> c_int;
    pub fn gen9blas_trsm(type_: c_int, side: c_int, uplo: c_int, trans_a: c_int, diag: c_int, m: c_int, n: c_int,
                         alpha: c_double, a: *const c_void, lda: c_int, b: *mut c_void, ldb: c_int) -> c_int;
    pub fn gen9blas_hgemm(m: c_int, n: c_int, k: c_int, alpha: c_double, a: *const u16, lda: c_int, b: *const u16,
                          ldb: c_int, beta: c_double, c: *mut u16, ldc: c_int) -> c_int;
    pub fn gen9blas_sgemm(m: c_int, n: c_int, k: c_int, alpha: c_double, a: *const c_float, lda: c_int,
                          b: *const c_float, ldb: c_int, beta: c_double, c: *mut c_float, ldc: c_int) -> c_int;

    // ---- levels 1-2 ----
    pub fn gen9blas_axpy(type_: c_int, n: c_int, a: c_double, x: *const c_void, incx: c_int, y: *mut c_void,
                         incy: c_int) -> c_int;
    pub fn gen9blas_scal(type_: c_int, n: c_int, a: c_double, x: *mut c_void, incx: c_int) -> c_int;
    pub fn gen9blas_copy(type_: c_int, n: c_int, x: *const c_void, incx: c_int, y: *mut c_void, incy: c_int) -> c_int;
    pub fn gen9blas_swap(type_: c_int, n: c_int, x: *mut c_void, incx: c_int, y: *mut c_void, incy: c_int) -> c_int;
    pub fn gen9blas_rot(type_: c_int, n: c_int, x: *mut c_void, incx: c_int, y: *mut c_void, incy: c_int, c: c_double,
                        s: c_double) -> c_int;
    pub fn gen9blas_rotm(type_: c_int, n: c_int, x: *mut c_void, incx: c_int, y: *mut c_void, incy: c_int,
                         param: *const c_double) -> c_int;
    pub fn gen9blas_rotg(a: *mut c_float, b: *mut c_float, c: *mut c_float, s: *mut c_float);
    pub fn gen9blas_rotmg(d1: *mut c_float, d2: *mut c_float, x1: *mut c_float, y1: c_float, param: *mut c_float);
    pub fn gen9blas_dot(type_: c_int, n: c_int, x: *const c_void, incx: c_int, y: *const c_void, incy: c_int,
                        result: *mut c_double) -> c_int;
    pub fn gen9blas_asum(type_: c_int, n: c_int, x: *const c_void, incx: c_int, result: *mut c_double) -> c_int;
    pub fn gen9blas_nrm2(type_: c_int, n: c_int, x: *const c_void, incx: c_int, result: *mut c_double) -> c_int;
    pub fn gen9blas_iamax(type_: c_int, n: c_int, x: *const c_void, incx: c_int, result: *mut i64) -> c_int;
    pub fn gen9blas_gemv(type_: c_int, trans: c_int, m: c_int, n: c_int, alpha: c_double, a: *const c_void, lda: c_int,
                         x: *const c_void, incx: c_int, beta: c_double, y: *mut c_void, incy: c_int) -> c_int;
    pub fn gen9blas_symv(type_: c_int, uplo: c_int, n: c_int, alpha: c_double, a: *const c_void, lda: c_int,
                         x: *const c_void, incx: c_int, beta: c_double, y: *mut c_void, incy: c_int) -> c_int;
    pub fn gen9blas_trmv(type_: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, a: *const c_void, lda: c_int,
                         x: *mut c_void, incx: c_int) -> c_int;
    pub fn gen9blas_trsv(type_: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, a: *const c_void, lda: c_int,
                         x: *mut c_void, incx: c_int) -> c_int;
    pub fn gen9blas_syr(type_: c_int, uplo: c_int, n: c_int, alpha: c_double, x: *const c_void, incx: c_int,
                        a: *mut c_void, lda: c_int) -> c_int;
    pub fn gen9blas_syr2(type_: c_int, uplo: c_int, n: c_int, alpha: c_double, x: *const c_void, incx: c_int,
                         y: *const c_void, incy: c_int, a: *mut c_void, lda: c_int) -> c_int;
    pub fn gen9blas_ger(type_: c_int, m: c_int, n: c_int, alpha: c_double, x: *const c_void, incx: c_int,
                        y: *const c_void, incy: c_int, a: *mut c_void, lda: c_int) -> c_int;
    pub fn gen9blas_gbmv(type_: c_int, trans: c_int, m: c_int, n: c_int, kl: c_int, ku: c_int, alpha: c_double,
                         a: *const c_void, lda: c_int, x: *const c_void, incx: c_int, beta: c_double, y: *mut c_void,
                         incy: c_int) -> c_int;
    pub fn gen9blas_sbmv(type_: c_int, uplo: c_int, n: c_int, k: c_int, alpha: c_double, a: *const c_void, lda: c_int,
                         x: *const c_void, incx: c_int, beta: c_double, y: *mut c_void, incy: c_int) -> c_int;
    pub fn gen9blas_tbmv(type_: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, k: c_int, a: *const c_void,
                         lda: c_int, x: *mut c_void, incx: c_int) -> c_int;
    pub fn gen9blas_tbsv(type_: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, k: c_int, a: *const c_void,
                         lda: c_int, x: *mut c_void, incx: c_int) -> c_int;
    pub fn gen9blas_spmv(type_: c_int, uplo: c_int, n: c_int, alpha: c_double, ap: *const c_void, x: *const c_void,
                         incx: c_int, beta: c_double, y: *mut c_void, incy: c_int) -> c_int;
    pub fn gen9blas_tpmv(type_: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, ap: *const c_void,
                         x: *mut c_void, incx: c_int) -> c_int;
    pub fn gen9blas_tpsv(type_: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, ap: *const c_void,
                         x: *mut c_void, incx: c_int) -> c_int;
    pub fn gen9blas_spr(type_: c_int, uplo: c_int, n: c_int, alpha: c_double, x: *const c_void, incx: c_int,
                        ap: *mut c_void) -> c_int;
    pub fn gen9blas_spr2(type_: c_int, uplo: c_int, n: c_int, alpha: c_double, x: *const c_void, incx: c_int,
                         y: *const c_void, incy: c_int, ap: *mut c_void) -> c_int;

    // ---- gen9blas_ml.h (unstable: follows ggml) ----
    pub fn gen9blas_gemm_pb_f16w(m: c_int, a: *const c_float, lda: c_int, bp: *const gen9blas_packed_b, c: *mut c_float,
                                 ldc: c_int, nonfinite: *mut c_int) -> c_int;
    pub fn gen9blas_gemm_pb_f16w_h(m: c_int, a: *const u16, lda: c_int, bp: *const gen9blas_packed_b, c: *mut c_float,
                                   ldc: c_int, nonfinite: *mut c_int) -> c_int;
    pub fn gen9blas_gemm_pb_f16w_ex(m: c_int, a: *const c_float, lda: c_int, bp: *const gen9blas_packed_b,
                                    c: *mut c_float, ldc: c_int, bias: *const c_float, gelu: c_int, r: *const c_float,
                                    ldr: c_int, nonfinite: *mut c_int) -> c_int;
    pub fn gen9blas_gemm_pb_skinny(m: c_int, a: *const c_float, lda: c_int, bp: *const gen9blas_packed_b,
                                   c: *mut c_float, ldc: c_int, nonfinite: *mut c_int) -> c_int;
    pub fn gen9blas_gemm_multi_f16w(m: c_int, a: *const c_float, lda: c_int, n: c_int,
                                    p: *const *const gen9blas_packed_b, c: *const *mut c_float, ldc: *const c_int,
                                    nonfinite: *mut c_int) -> c_int;
    pub fn gen9blas_ffn_f16w(m: c_int, x: *const c_float, ldx: c_int, pg: *const gen9blas_packed_b,
                             pu: *const gen9blas_packed_b, pd: *const gen9blas_packed_b, y: *mut c_float, ldy: c_int,
                             nonfinite: *mut c_int) -> c_int;
    pub fn gen9blas_pack_b_q(qtype: c_int, k: c_int, n: c_int, w: *const c_void, row_bytes: usize,
                             out_of_range: *mut c_int) -> *mut gen9blas_packed_b;
    pub fn gen9blas_packed_a_create() -> *mut gen9blas_packed_a;
    pub fn gen9blas_packed_a_free(pa: *mut gen9blas_packed_a);
    pub fn gen9blas_layer_norm_pack(m: c_int, k: c_int, x: *const c_float, ldx: c_int, w: *const c_float,
                                    b: *const c_float, eps: c_float, y: *mut c_float, ldy: c_int,
                                    pa: *mut gen9blas_packed_a, sync: c_int) -> c_int;
    pub fn gen9blas_gemm_pa_f16w_ex(pa: *const gen9blas_packed_a, bp: *const gen9blas_packed_b, c: *mut c_float,
                                    ldc: c_int, bias: *const c_float, gelu: c_int, r: *const c_float, ldr: c_int,
                                    nonfinite: *mut c_int) -> c_int;
    pub fn gen9blas_set_nowait(on: c_int);
    pub fn gen9blas_nowait_pending() -> c_int;
    pub fn gen9blas_span(p: *const c_void, bytes: usize, writes: c_int) -> c_int;
    pub fn gen9blas_span_clear();
    pub fn gen9blas_flush() -> c_int;
    pub fn gen9blas_rms_norm(m: c_int, k: c_int, x: *const c_float, ldx: c_int, w: *const c_float, eps: c_float,
                             y: *mut c_float, ldy: c_int, sync: c_int) -> c_int;
    pub fn gen9blas_layer_norm(m: c_int, k: c_int, x: *const c_float, ldx: c_int, w: *const c_float, b: *const c_float,
                               eps: c_float, y: *mut c_float, ldy: c_int, sync: c_int) -> c_int;
    pub fn gen9blas_binop_rows(op: c_int, m: c_int, k: c_int, a: *const c_float, lda: c_int, b: *const c_float,
                               ldb: c_int, bs0: c_int, c: *mut c_float, ldc: c_int, sync: c_int) -> c_int;
    pub fn gen9blas_gelu(n: usize, x: *const c_float, y: *mut c_float, sync: c_int) -> c_int;
    pub fn gen9blas_gelu_erf(n: usize, x: *const c_float, y: *mut c_float, sync: c_int) -> c_int;
    pub fn gen9blas_relu(n: usize, x: *const c_float, y: *mut c_float, sync: c_int) -> c_int;
    pub fn gen9blas_im2col(n: c_int, ic: c_int, ih: c_int, iw: c_int, x: *const c_float, ofs0: i64, ofs1: i64,
                           kh: c_int, kw: c_int, oh: c_int, ow: c_int, s0: c_int, s1: c_int, p0: c_int, p1: c_int,
                           d0: c_int, d1: c_int, y: *mut c_void, f16: c_int, sync: c_int) -> c_int;
    pub fn gen9blas_cvt_f16_heads(rows: c_int, h: c_int, d: c_int, x: *const c_float, s_row: i64, s_head: i64,
                                  y: *mut u16, sync: c_int) -> c_int;
    pub fn gen9blas_rope(n_dims: c_int, nh: c_int, n_tok: c_int, x: *const c_float, xs1: i64, xs2: i64, y: *mut c_float,
                         ys1: i64, ys2: i64, pos: *const c_int, ff: *const c_float, neox: c_int, freq_base: c_float,
                         freq_scale: c_float, ext_factor: c_float, attn_factor: c_float, corr: *const c_float,
                         sync: c_int) -> c_int;
    pub fn gen9blas_rope_multi(n_dims: c_int, nh: c_int, n_tok: c_int, x: *const c_float, xs1: i64, xs2: i64,
                               y: *mut c_float, ys1: i64, ys2: i64, pos: *const c_int, ff: *const c_float,
                               sections: *const c_int, freq_base: c_float, freq_scale: c_float, ext_factor: c_float,
                               attn_factor: c_float, corr: *const c_float, vision: c_int, sync: c_int) -> c_int;
    pub fn gen9blas_cpy_any(src: *const c_void, tin: c_int, sne: *const i64, snb: *const i64, dst: *mut c_void,
                            tout: c_int, dne: *const i64, dnb: *const i64, sync: c_int) -> c_int;
    pub fn gen9blas_upscale_f32(src: *const c_float, sne: *const i64, snb: *const i64, dst: *mut c_float,
                                dne: *const i64, dnb: *const i64, mode_flags: c_int, sync: c_int) -> c_int;
    pub fn gen9blas_set_rows_f16(nc: c_int, nr: c_int, ne02: c_int, ne03: c_int, src: *const c_float, s1: i64, s2: i64,
                                 s3: i64, idx: *const c_void, idx_bytes: usize, b10: i64, b11: i64, b12: i64,
                                 idx64: c_int, ne11: c_int, ne12: c_int, dst: *mut u16, dst_bytes: usize, d1: i64,
                                 d2: i64, d3: i64, ne1: i64, sync: c_int, bad: *mut c_int) -> c_int;
    pub fn gen9blas_attn_dec_supported(d: c_int, h: c_int, hkv: c_int) -> c_int;
    pub fn gen9blas_attn_dec(d: c_int, h: c_int, hkv: c_int, ns: c_int, nq: c_int, nk: c_int, q: *const c_float,
                             qt: i64, qh: i64, qs: i64, k: *const u16, kk: i64, kh: i64, ks: i64, k_bytes: usize,
                             v: *const u16, vk: i64, vh: i64, vs: i64, v_bytes: usize, o: *mut c_float,
                             scale: c_float, vis: *const c_int, dvis: c_int, chunk: c_int, sync: c_int) -> c_int;
    pub fn gen9blas_attn_supported(d: c_int, h: c_int, hkv: c_int) -> c_int;
    pub fn gen9blas_attn_f16kv(d: c_int, h: c_int, hkv: c_int, nq: c_int, nk: c_int, q: *const c_float, k: *const u16,
                               k_bytes: usize, v: *const u16, v_bytes: usize, o: *mut c_float, scale: c_float,
                               vis0: c_int, dvis: c_int, launches: *mut c_int) -> c_int;
    pub fn gen9blas_attn_streams_supported(d: c_int, h: c_int, hkv: c_int) -> c_int;
    pub fn gen9blas_attn_f16kv_streams(d: c_int, h: c_int, hkv: c_int, ns: c_int, nq: c_int, nk: c_int,
                                       q: *const c_float, sq: i64, k: *const u16, sk: i64, k_bytes: usize,
                                       v: *const u16, sv: i64, v_bytes: usize, o: *mut c_float, scale: c_float,
                                       vis0: *const c_int, dvis: c_int, launches: *mut c_int) -> c_int;
}
