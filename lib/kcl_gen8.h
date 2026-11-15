// Gen8 (Cherryview) commands whose layout differs from Gen9 (docs/gen8.md, section 5). They live in their own file
// because Mesa's gen8_pack.h and gen9_pack.h define the same enumerators and cannot be included together.
#pragma once
#include <stdint.h>

struct kcl_sba {   // STATE_BASE_ADDRESS as kcl programs it (general state = 0, size 4 GB)
    uint64_t surface, dynamic, indirect, instruction;
    uint32_t mocs, dynamic_size, indirect_size, instruction_size;   // sizes in 4 KB pages
};

uint32_t *kcl_gen8_pipeline_select_gpgpu(uint32_t *p);   // no Mask Bits on Gen8 (PRM CHV vol02a p.1104)
uint32_t *kcl_gen8_state_base_address(uint32_t *p, const struct kcl_sba *s);   // 16 DW, no bindless fields
