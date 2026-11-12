/* Minimalny zamiennik util/bitpack_helpers.h z Mesy dla nagłówka wygenerowanego przez gen_pack_header.py. */
#pragma once
#include <stdbool.h>
#include <stdint.h>
static inline uint64_t util_bitpack_uint(uint64_t v, uint32_t start, uint32_t end) { (void)end; return v << start; }
static inline uint64_t util_bitpack_uint_nonzero(uint64_t v, uint32_t start, uint32_t end) { (void)end; return v << start; }
static inline uint64_t util_bitpack_sint(int64_t v, uint32_t start, uint32_t end) {
    const int bits = end - start + 1; const uint64_t mask = bits == 64 ? ~0ull : (1ull << bits) - 1;
    return ((uint64_t)v & mask) << start;
}
static inline uint64_t util_bitpack_sint_nonzero(int64_t v, uint32_t start, uint32_t end) { return util_bitpack_sint(v, start, end); }
static inline uint64_t util_bitpack_ufixed(float v, uint32_t start, uint32_t end, uint32_t frac) {
    (void)end; return (uint64_t)(uint32_t)(v * (float)(1u << frac)) << start;
}
static inline uint32_t util_bitpack_float(float v) { union { float f; uint32_t u; } x = { v }; return x.u; }
