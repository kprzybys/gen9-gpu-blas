/* Minimalny zamiennik genX_helpers.h z Mesy: adresy to zwykłe uint64_t (softpin, bez relokacji). */
#pragma once
#include <stdint.h>
typedef void __gen_user_data;
typedef uint64_t __gen_address_type;
static inline uint64_t __gen_offset(uint64_t v, uint32_t start, uint32_t end) { (void)start; (void)end; return v; }
static inline uint64_t __gen_address(__gen_user_data *data, void *location, __gen_address_type address, uint32_t delta,
                                     uint32_t start, uint32_t end) {
    (void)data; (void)location; (void)start;
    uint64_t a = address + delta;
    if (end < 63) { const uint32_t s = 63 - end; a = (a << s) >> s; }
    return a;
}
