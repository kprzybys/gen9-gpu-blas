# Changelog

## 0.2.0 (unreleased)

Breaking release: the API is reorganized; GPU arithmetic is unchanged (same kernels, bit-identical results).

### Layout
- Library kernels moved from `bench/` to `lib/kernels/`; `lib/gen9blas.c` split into `lib/src/` (`core`, `gemm`,
  `packed`, `ops`, `level3`, `level12`, `attn`, `cblas`).
- Public headers: `gen9blas.h` (BLAS) and the new `gen9blas_ml.h` (neural-network operations for the ggml backend;
  unstable, follows ggml).
- `libkcl.a`: kcl for programs with their own kernel tables (link it instead of compiling `lib/kcl.c`).
- `libgen9blas.so` exports only the `gen9blas_*` API (built with `-fvisibility=hidden`); `kcl_*` is no longer exported.

### API: old -> new
| 0.1 | 0.2 |
|---|---|
| `gen9blas_init(const char *dir)` | `gen9blas_init(void)`: once per process, thread-safe, a failure is final; NEO builds read `GEN9BLAS_DIR` |
| — | `gen9blas_version()`, `GEN9BLAS_VERSION*` macros |
| — | `gen9blas_device_gen()` (9, 8 or 0) |
| `gen9blas_available(type)` | `gen9blas_has(GEN9BLAS_FEAT_FP16 / _FP32 / _FP64)` |
| `gen9blas_skinny_pb_supported()` | `gen9blas_has(GEN9BLAS_FEAT_SKINNY_PB)` |
| `gen9blas_epilog_ext_supported()` | `gen9blas_has(GEN9BLAS_FEAT_EPILOG_EXT)` |
| `gen9blas_norm_pack_supported()` | `gen9blas_has(GEN9BLAS_FEAT_NORM_PACK)` |
| `gen9blas_rope_multi_supported()` | `gen9blas_has(GEN9BLAS_FEAT_ROPE_MULTI)` |
| `gen9blas_cpy_supported()` | `gen9blas_has(GEN9BLAS_FEAT_CPY)` |
| `gen9blas_set_rows_supported()` | `gen9blas_has(GEN9BLAS_FEAT_SET_ROWS)` |
| `gen9blas_attn_dec_supported(D, G)` | `gen9blas_attn_dec_supported(D, H, HKV)` (like the other attention queries) |
| `gen9blas_packed` (type) | `gen9blas_packed_b` |
| `gen9blas_packed_free(p)` | `gen9blas_packed_b_free(p)` |
| `gen9blas_packed_a_new()` | `gen9blas_packed_a_create()` |
| `gen9blas_gemm_pb(type, transA, ...)` | `gen9blas_gemm_pb(transA, ...)` (the type comes from the packed handle) |
| `gen9blas_nowait(on)` | `gen9blas_set_nowait(on)` |
| `gen9blas_nowait_check()` | `gen9blas_nowait_pending()` |
| `gen9blas_binop_rows(op, M, K, A, lda, B, ldb, C, ldc, sync)` and `gen9blas_binop_rows2(..., bs0, ...)` | one `gen9blas_binop_rows(op, M, K, A, lda, B, ldb, bs0, C, ldc, sync)` (bs0 = 1: the old binop_rows) |
| `gen9blas_cpy_any`: `tin` / `tout` 0 = F32, 1 = F16 | `GEN9BLAS_S` / `GEN9BLAS_H` (note: the numeric values swap) |
| `long` strides, offsets and `iamax` result | `int64_t` (same ABI on 64-bit Linux) |
| — | named constants `GEN9BLAS_NO_TRANS/TRANS`, `GEN9BLAS_NON_UNIT/UNIT` (parameters stay `int`) |

### Fixes
- `libgen9blas_blas`: the error text is read under the GPU lock (was read after releasing it).
- A failed initialization no longer creates a second GPU context when retried.

## 0.1.0

First public release (tag `v0.1.0`, DOI in `CITATION.cff`).
