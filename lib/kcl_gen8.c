// Gen8 (Cherryview) variants of the two commands whose layout differs from Gen9 (see kcl_gen8.h).
#include "kcl_gen8.h"
#include <stdbool.h>
#include "gen8_pack.h"

uint32_t *kcl_gen8_pipeline_select_gpgpu(uint32_t *p) {
    struct GFX8_PIPELINE_SELECT c = { GFX8_PIPELINE_SELECT_header, .PipelineSelection = GPGPU };
    GFX8_PIPELINE_SELECT_pack(NULL, p, &c);
    return p + GFX8_PIPELINE_SELECT_length;
}

uint32_t *kcl_gen8_state_base_address(uint32_t *p, const struct kcl_sba *s) {
    struct GFX8_STATE_BASE_ADDRESS c = { GFX8_STATE_BASE_ADDRESS_header,
        .GeneralStateBaseAddressModifyEnable = 1, .GeneralStateBaseAddress = 0, .GeneralStateMOCS = s->mocs,
        .StatelessDataPortAccessMOCS = s->mocs,
        .SurfaceStateBaseAddressModifyEnable = 1, .SurfaceStateBaseAddress = s->surface, .SurfaceStateMOCS = s->mocs,
        .DynamicStateBaseAddressModifyEnable = 1, .DynamicStateBaseAddress = s->dynamic, .DynamicStateMOCS = s->mocs,
        .IndirectObjectBaseAddressModifyEnable = 1, .IndirectObjectBaseAddress = s->indirect, .IndirectObjectMOCS = s->mocs,
        .InstructionBaseAddressModifyEnable = 1, .InstructionBaseAddress = s->instruction, .InstructionMOCS = s->mocs,
        .GeneralStateBufferSizeModifyEnable = 1, .GeneralStateBufferSize = 0xfffff,
        .DynamicStateBufferSizeModifyEnable = 1, .DynamicStateBufferSize = s->dynamic_size,
        .IndirectObjectBufferSizeModifyEnable = 1, .IndirectObjectBufferSize = s->indirect_size,
        .InstructionBuffersizeModifyEnable = 1, .InstructionBufferSize = s->instruction_size };
    GFX8_STATE_BASE_ADDRESS_pack(NULL, p, &c);
    return p + GFX8_STATE_BASE_ADDRESS_length;
}
