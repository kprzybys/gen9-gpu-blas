/*
 * Copyright (C) 2016 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


/* Instructions, enums and structures for SKL.
 *
 * This file has been generated, do not hand edit.
 */

#ifndef GFX9_SKL_PACK_H
#define GFX9_SKL_PACK_H

#ifndef __OPENCL_VERSION__
#include <stdio.h>
#include "util/bitpack_helpers.h"
#include "genX_helpers.h"
#else
#include "genX_cl_helpers.h"
#endif


enum GFX9_3D_Color_Buffer_Blend_Factor {
   BLENDFACTOR_ONE                      =      1,
   BLENDFACTOR_SRC_COLOR                =      2,
   BLENDFACTOR_SRC_ALPHA                =      3,
   BLENDFACTOR_DST_ALPHA                =      4,
   BLENDFACTOR_DST_COLOR                =      5,
   BLENDFACTOR_SRC_ALPHA_SATURATE       =      6,
   BLENDFACTOR_CONST_COLOR              =      7,
   BLENDFACTOR_CONST_ALPHA              =      8,
   BLENDFACTOR_SRC1_COLOR               =      9,
   BLENDFACTOR_SRC1_ALPHA               =     10,
   BLENDFACTOR_ZERO                     =     17,
   BLENDFACTOR_INV_SRC_COLOR            =     18,
   BLENDFACTOR_INV_SRC_ALPHA            =     19,
   BLENDFACTOR_INV_DST_ALPHA            =     20,
   BLENDFACTOR_INV_DST_COLOR            =     21,
   BLENDFACTOR_INV_CONST_COLOR          =     23,
   BLENDFACTOR_INV_CONST_ALPHA          =     24,
   BLENDFACTOR_INV_SRC1_COLOR           =     25,
   BLENDFACTOR_INV_SRC1_ALPHA           =     26,
};

enum GFX9_3D_Color_Buffer_Blend_Function {
   BLENDFUNCTION_ADD                    =      0,
   BLENDFUNCTION_SUBTRACT               =      1,
   BLENDFUNCTION_REVERSE_SUBTRACT       =      2,
   BLENDFUNCTION_MIN                    =      3,
   BLENDFUNCTION_MAX                    =      4,
};

enum GFX9_3D_Compare_Function {
   COMPAREFUNCTION_ALWAYS               =      0,
   COMPAREFUNCTION_NEVER                =      1,
   COMPAREFUNCTION_LESS                 =      2,
   COMPAREFUNCTION_EQUAL                =      3,
   COMPAREFUNCTION_LEQUAL               =      4,
   COMPAREFUNCTION_GREATER              =      5,
   COMPAREFUNCTION_NOTEQUAL             =      6,
   COMPAREFUNCTION_GEQUAL               =      7,
};

enum GFX9_3D_Logic_Op_Function {
   LOGICOP_CLEAR                        =      0,
   LOGICOP_NOR                          =      1,
   LOGICOP_AND_INVERTED                 =      2,
   LOGICOP_COPY_INVERTED                =      3,
   LOGICOP_AND_REVERSE                  =      4,
   LOGICOP_INVERT                       =      5,
   LOGICOP_XOR                          =      6,
   LOGICOP_NAND                         =      7,
   LOGICOP_AND                          =      8,
   LOGICOP_EQUIV                        =      9,
   LOGICOP_NOOP                         =     10,
   LOGICOP_OR_INVERTED                  =     11,
   LOGICOP_COPY                         =     12,
   LOGICOP_OR_REVERSE                   =     13,
   LOGICOP_OR                           =     14,
   LOGICOP_SET                          =     15,
};

enum GFX9_3D_Prim_Topo_Type {
   _3DPRIM_POINTLIST                    =      1,
   _3DPRIM_LINELIST                     =      2,
   _3DPRIM_LINESTRIP                    =      3,
   _3DPRIM_TRILIST                      =      4,
   _3DPRIM_TRISTRIP                     =      5,
   _3DPRIM_TRIFAN                       =      6,
   _3DPRIM_QUADLIST                     =      7,
   _3DPRIM_QUADSTRIP                    =      8,
   _3DPRIM_LINELIST_ADJ                 =      9,
   _3DPRIM_LINESTRIP_ADJ                =     10,
   _3DPRIM_TRILIST_ADJ                  =     11,
   _3DPRIM_TRISTRIP_ADJ                 =     12,
   _3DPRIM_TRISTRIP_REVERSE             =     13,
   _3DPRIM_POLYGON                      =     14,
   _3DPRIM_RECTLIST                     =     15,
   _3DPRIM_LINELOOP                     =     16,
   _3DPRIM_POINTLIST_BF                 =     17,
   _3DPRIM_LINESTRIP_CONT               =     18,
   _3DPRIM_LINESTRIP_BF                 =     19,
   _3DPRIM_LINESTRIP_CONT_BF            =     20,
   _3DPRIM_TRIFAN_NOSTIPPLE             =     22,
   _3DPRIM_PATCHLIST_1                  =     32,
   _3DPRIM_PATCHLIST_2                  =     33,
   _3DPRIM_PATCHLIST_3                  =     34,
   _3DPRIM_PATCHLIST_4                  =     35,
   _3DPRIM_PATCHLIST_5                  =     36,
   _3DPRIM_PATCHLIST_6                  =     37,
   _3DPRIM_PATCHLIST_7                  =     38,
   _3DPRIM_PATCHLIST_8                  =     39,
   _3DPRIM_PATCHLIST_9                  =     40,
   _3DPRIM_PATCHLIST_10                 =     41,
   _3DPRIM_PATCHLIST_11                 =     42,
   _3DPRIM_PATCHLIST_12                 =     43,
   _3DPRIM_PATCHLIST_13                 =     44,
   _3DPRIM_PATCHLIST_14                 =     45,
   _3DPRIM_PATCHLIST_15                 =     46,
   _3DPRIM_PATCHLIST_16                 =     47,
   _3DPRIM_PATCHLIST_17                 =     48,
   _3DPRIM_PATCHLIST_18                 =     49,
   _3DPRIM_PATCHLIST_19                 =     50,
   _3DPRIM_PATCHLIST_20                 =     51,
   _3DPRIM_PATCHLIST_21                 =     52,
   _3DPRIM_PATCHLIST_22                 =     53,
   _3DPRIM_PATCHLIST_23                 =     54,
   _3DPRIM_PATCHLIST_24                 =     55,
   _3DPRIM_PATCHLIST_25                 =     56,
   _3DPRIM_PATCHLIST_26                 =     57,
   _3DPRIM_PATCHLIST_27                 =     58,
   _3DPRIM_PATCHLIST_28                 =     59,
   _3DPRIM_PATCHLIST_29                 =     60,
   _3DPRIM_PATCHLIST_30                 =     61,
   _3DPRIM_PATCHLIST_31                 =     62,
   _3DPRIM_PATCHLIST_32                 =     63,
};

enum GFX9_3D_Stencil_Operation {
   STENCILOP_KEEP                       =      0,
   STENCILOP_ZERO                       =      1,
   STENCILOP_REPLACE                    =      2,
   STENCILOP_INCRSAT                    =      3,
   STENCILOP_DECRSAT                    =      4,
   STENCILOP_INCR                       =      5,
   STENCILOP_DECR                       =      6,
   STENCILOP_INVERT                     =      7,
};

enum GFX9_3D_Vertex_Component_Control {
   VFCOMP_NOSTORE                       =      0,
   VFCOMP_STORE_SRC                     =      1,
   VFCOMP_STORE_0                       =      2,
   VFCOMP_STORE_1_FP                    =      3,
   VFCOMP_STORE_1_INT                   =      4,
   VFCOMP_STORE_PID                     =      7,
};

enum GFX9_Atomic_OPCODE {
   MI_ATOMIC_OP_AND                     =      1,
   MI_ATOMIC_OP_OR                      =      2,
   MI_ATOMIC_OP_XOR                     =      3,
   MI_ATOMIC_OP_MOVE                    =      4,
   MI_ATOMIC_OP_INC                     =      5,
   MI_ATOMIC_OP_DEC                     =      6,
   MI_ATOMIC_OP_ADD                     =      7,
   MI_ATOMIC_OP_SUB                     =      8,
   MI_ATOMIC_OP_RSUB                    =      9,
   MI_ATOMIC_OP_IMAX                    =     10,
   MI_ATOMIC_OP_IMIN                    =     11,
   MI_ATOMIC_OP_UMAX                    =     12,
   MI_ATOMIC_OP_UMIN                    =     13,
   MI_ATOMIC_OP_CMP_WR                  =     14,
   MI_ATOMIC_OP_PREDEC                  =     15,
   MI_ATOMIC_OP_AND8B                   =     33,
   MI_ATOMIC_OP_OR8B                    =     34,
   MI_ATOMIC_OP_XOR8B                   =     35,
   MI_ATOMIC_OP_MOVE8B                  =     36,
   MI_ATOMIC_OP_INC8B                   =     37,
   MI_ATOMIC_OP_DEC8B                   =     38,
   MI_ATOMIC_OP_ADD8B                   =     39,
   MI_ATOMIC_OP_SUB8B                   =     40,
   MI_ATOMIC_OP_RSUB8B                  =     41,
   MI_ATOMIC_OP_IMAX8B                  =     42,
   MI_ATOMIC_OP_IMIN8B                  =     43,
   MI_ATOMIC_OP_UMAX8B                  =     44,
   MI_ATOMIC_OP_UMIN8B                  =     45,
   MI_ATOMIC_OP_CMP_WR8B                =     46,
   MI_ATOMIC_OP_PREDEC8B                =     47,
   MI_ATOMIC_OP_CMP_WR16B               =     78,
};

enum GFX9_Attribute_Component_Format {
   ACF_DISABLED                         =      0,
   ACF_XY                               =      1,
   ACF_XYZ                              =      2,
   ACF_XYZW                             =      3,
};

enum GFX9_ShaderChannelSelect {
   SCS_ZERO                             =      0,
   SCS_ONE                              =      1,
   SCS_RED                              =      4,
   SCS_GREEN                            =      5,
   SCS_BLUE                             =      6,
   SCS_ALPHA                            =      7,
};

enum GFX9_TextureCoordinateMode {
   TCM_WRAP                             =      0,
   TCM_MIRROR                           =      1,
   TCM_CLAMP                            =      2,
   TCM_CUBE                             =      3,
   TCM_CLAMP_BORDER                     =      4,
   TCM_MIRROR_ONCE                      =      5,
   TCM_HALF_BORDER                      =      6,
};

enum GFX9_WRAP_SHORTEST_ENABLE {
   WSE_X                                =      1,
   WSE_Y                                =      2,
   WSE_XY                               =      3,
   WSE_Z                                =      4,
   WSE_XZ                               =      5,
   WSE_YZ                               =      6,
   WSE_XYZ                              =      7,
   WSE_W                                =      8,
   WSE_XW                               =      9,
   WSE_YW                               =     10,
   WSE_XYW                              =     11,
   WSE_ZW                               =     12,
   WSE_XZW                              =     13,
   WSE_YZW                              =     14,
   WSE_XYZW                             =     15,
};

#define GFX9_BINDING_TABLE_STATE_length        1
struct GFX9_BINDING_TABLE_STATE {
   uint64_t                             SurfaceStatePointer;
};

static inline __attribute__((always_inline)) void
GFX9_BINDING_TABLE_STATE_pack(__attribute__((unused)) __gen_user_data *data,
                              __attribute__((unused)) void * restrict dst,
                              __attribute__((unused)) const struct GFX9_BINDING_TABLE_STATE * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      __gen_offset(values->SurfaceStatePointer, 6, 31);
}

#define GFX9_INTERFACE_DESCRIPTOR_DATA_length      8
struct GFX9_INTERFACE_DESCRIPTOR_DATA {
   uint64_t                             KernelStartPointer;
   bool                                 SoftwareExceptionEnable;
   bool                                 MaskStackExceptionEnable;
   bool                                 IllegalOpcodeExceptionEnable;
   uint32_t                             FloatingPointMode;
#define IEEE754                                  0
#define Alternate                                1
   uint32_t                             ThreadPriority;
#define NormalPriority                           0
#define HighPriority                             1
   bool                                 SingleProgramFlow;
   uint32_t                             DenormMode;
#define Ftz                                      0
#define SetByKernel                              1
   uint32_t                             SamplerCount;
#define Nosamplersused                           0
#define Between1and4samplersused                 1
#define Between5and8samplersused                 2
#define Between9and12samplersused                3
#define Between13and16samplersused               4
   uint64_t                             SamplerStatePointer;
   uint32_t                             BindingTableEntryCount;
   uint64_t                             BindingTablePointer;
   uint32_t                             ConstantURBEntryReadOffset;
   uint32_t                             ConstantURBEntryReadLength;
   uint32_t                             NumberofThreadsinGPGPUThreadGroup;
   bool                                 GlobalBarrierEnable;
   uint32_t                             SharedLocalMemorySize;
#define Encodes0K                                0
#define Encodes1K                                1
#define Encodes2K                                2
#define Encodes4K                                3
#define Encodes8K                                4
#define Encodes16K                               5
#define Encodes32K                               6
#define Encodes64K                               7
   bool                                 BarrierEnable;
   uint32_t                             RoundingMode;
#define RTNE                                     0
#define RU                                       1
#define RD                                       2
#define RTZ                                      3
   uint32_t                             CrossThreadConstantDataReadLength;
};

static inline __attribute__((always_inline)) void
GFX9_INTERFACE_DESCRIPTOR_DATA_pack(__attribute__((unused)) __gen_user_data *data,
                                    __attribute__((unused)) void * restrict dst,
                                    __attribute__((unused)) const struct GFX9_INTERFACE_DESCRIPTOR_DATA * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   const uint64_t v0 =
      __gen_offset(values->KernelStartPointer, 6, 47);
   dw[0] = v0;
   dw[1] = v0 >> 32;

   dw[2] =
      util_bitpack_uint(values->SoftwareExceptionEnable, 7, 7) |
      util_bitpack_uint(values->MaskStackExceptionEnable, 11, 11) |
      util_bitpack_uint(values->IllegalOpcodeExceptionEnable, 13, 13) |
      util_bitpack_uint(values->FloatingPointMode, 16, 16) |
      util_bitpack_uint(values->ThreadPriority, 17, 17) |
      util_bitpack_uint(values->SingleProgramFlow, 18, 18) |
      util_bitpack_uint(values->DenormMode, 19, 19);

   dw[3] =
      util_bitpack_uint(values->SamplerCount, 2, 4) |
      __gen_offset(values->SamplerStatePointer, 5, 31);

   dw[4] =
      util_bitpack_uint(values->BindingTableEntryCount, 0, 4) |
      __gen_offset(values->BindingTablePointer, 5, 15);

   dw[5] =
      util_bitpack_uint(values->ConstantURBEntryReadOffset, 0, 15) |
      util_bitpack_uint(values->ConstantURBEntryReadLength, 16, 31);

   dw[6] =
      util_bitpack_uint(values->NumberofThreadsinGPGPUThreadGroup, 0, 9) |
      util_bitpack_uint(values->GlobalBarrierEnable, 15, 15) |
      util_bitpack_uint(values->SharedLocalMemorySize, 16, 20) |
      util_bitpack_uint(values->BarrierEnable, 21, 21) |
      util_bitpack_uint(values->RoundingMode, 22, 23);

   dw[7] =
      util_bitpack_uint(values->CrossThreadConstantDataReadLength, 0, 7);
}

#define GFX9_RENDER_SURFACE_STATE_length      16
struct GFX9_RENDER_SURFACE_STATE {
   bool                                 CubeFaceEnablePositiveZ;
   bool                                 CubeFaceEnableNegativeZ;
   bool                                 CubeFaceEnablePositiveY;
   bool                                 CubeFaceEnableNegativeY;
   bool                                 CubeFaceEnablePositiveX;
   bool                                 CubeFaceEnableNegativeX;
   uint32_t                             MediaBoundaryPixelMode;
#define NORMAL_MODE                              0
#define PROGRESSIVE_FRAME                        2
#define INTERLACED_FRAME                         3
   uint32_t                             RenderCacheReadWriteMode;
#define WriteOnlyCache                           0
#define ReadWriteCache                           1
   bool                                 SamplerL2BypassModeDisable;
   uint32_t                             VerticalLineStrideOffset;
   uint32_t                             VerticalLineStride;
   uint32_t                             TileMode;
#define LINEAR                                   0
#define WMAJOR                                   1
#define XMAJOR                                   2
#define YMAJOR                                   3
   uint32_t                             SurfaceHorizontalAlignment;
#define HALIGN_4                                 1
#define HALIGN_8                                 2
#define HALIGN_16                                3
   uint32_t                             SurfaceVerticalAlignment;
#define VALIGN_4                                 1
#define VALIGN_8                                 2
#define VALIGN_16                                3
   uint32_t                             SurfaceFormat;
   bool                                 SurfaceArray;
   uint32_t                             SurfaceType;
#define SURFTYPE_1D                              0
#define SURFTYPE_2D                              1
#define SURFTYPE_3D                              2
#define SURFTYPE_CUBE                            3
#define SURFTYPE_BUFFER                          4
#define SURFTYPE_STRBUF                          5
#define SURFTYPE_NULL                            7
   uint32_t                             SurfaceQPitch;
   float                                BaseMipLevel;
   uint32_t                             MOCS;
   uint32_t                             Width;
   uint32_t                             Height;
   uint32_t                             SurfacePitch;
   uint32_t                             Depth;
   uint32_t                             MultisamplePositionPaletteIndex;
   uint32_t                             NumberofMultisamples;
#define MULTISAMPLECOUNT_1                       0
#define MULTISAMPLECOUNT_2                       1
#define MULTISAMPLECOUNT_4                       2
#define MULTISAMPLECOUNT_8                       3
#define MULTISAMPLECOUNT_16                      4
   uint32_t                             MultisampledSurfaceStorageFormat;
#define MSFMT_MSS                                0
#define MSFMT_DEPTH_STENCIL                      1
   uint32_t                             RenderTargetViewExtent;
   uint32_t                             MinimumArrayElement;
   uint32_t                             RenderTargetAndSampleUnormRotation;
#define _0DEG                                    0
#define _90DEG                                   1
#define _180DEG                                  2
#define _270DEG                                  3
   uint32_t                             MIPCountLOD;
   uint32_t                             SurfaceMinLOD;
   uint32_t                             MipTailStartLOD;
   uint32_t                             CoherencyType;
#define GPUcoherent                              0
#define IAcoherent                               1
   uint32_t                             TiledResourceMode;
#define NONE                                     0
#define _4KB                                     1
#define _64KB                                    2
#define TILEYF                                   1
#define TILEYS                                   2
   bool                                 EWADisableForCube;
   uint32_t                             YOffset;
   uint32_t                             XOffset;
   uint32_t                             AuxiliarySurfaceMode;
#define AUX_NONE                                 0
#define AUX_CCS_D                                1
#define AUX_APPEND                               2
#define AUX_HIZ                                  3
#define AUX_CCS_E                                5
   uint32_t                             YOffsetforUorUVPlane;
   uint32_t                             AuxiliarySurfacePitch;
   uint32_t                             AuxiliarySurfaceQPitch;
   uint32_t                             XOffsetforUorUVPlane;
   bool                                 SeparateUVPlaneEnable;
   float                                ResourceMinLOD;
   enum GFX9_ShaderChannelSelect        ShaderChannelSelectAlpha;
   enum GFX9_ShaderChannelSelect        ShaderChannelSelectBlue;
   enum GFX9_ShaderChannelSelect        ShaderChannelSelectGreen;
   enum GFX9_ShaderChannelSelect        ShaderChannelSelectRed;
   bool                                 MemoryCompressionEnable;
   uint32_t                             MemoryCompressionMode;
#define Horizontal                               0
#define Vertical                                 1
   __gen_address_type                   SurfaceBaseAddress;
   uint32_t                             QuiltWidth;
   uint32_t                             QuiltHeight;
   __gen_address_type                   AuxiliarySurfaceBaseAddress;
   uint32_t                             AuxiliaryTableIndexforMediaCompressedSurface;
   uint32_t                             YOffsetforVPlane;
   uint32_t                             XOffsetforVPlane;
   float                                HierarchicalDepthClearValue;
   int32_t                              RedClearColor;
   int32_t                              GreenClearColor;
   int32_t                              BlueClearColor;
   int32_t                              AlphaClearColor;
};

static inline __attribute__((always_inline)) void
GFX9_RENDER_SURFACE_STATE_pack(__attribute__((unused)) __gen_user_data *data,
                               __attribute__((unused)) void * restrict dst,
                               __attribute__((unused)) const struct GFX9_RENDER_SURFACE_STATE * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      util_bitpack_uint(values->CubeFaceEnablePositiveZ, 0, 0) |
      util_bitpack_uint(values->CubeFaceEnableNegativeZ, 1, 1) |
      util_bitpack_uint(values->CubeFaceEnablePositiveY, 2, 2) |
      util_bitpack_uint(values->CubeFaceEnableNegativeY, 3, 3) |
      util_bitpack_uint(values->CubeFaceEnablePositiveX, 4, 4) |
      util_bitpack_uint(values->CubeFaceEnableNegativeX, 5, 5) |
      util_bitpack_uint(values->MediaBoundaryPixelMode, 6, 7) |
      util_bitpack_uint(values->RenderCacheReadWriteMode, 8, 8) |
      util_bitpack_uint(values->SamplerL2BypassModeDisable, 9, 9) |
      util_bitpack_uint(values->VerticalLineStrideOffset, 10, 10) |
      util_bitpack_uint(values->VerticalLineStride, 11, 11) |
      util_bitpack_uint(values->TileMode, 12, 13) |
      util_bitpack_uint(values->SurfaceHorizontalAlignment, 14, 15) |
      util_bitpack_uint(values->SurfaceVerticalAlignment, 16, 17) |
      util_bitpack_uint(values->SurfaceFormat, 18, 27) |
      util_bitpack_uint(values->SurfaceArray, 28, 28) |
      util_bitpack_uint(values->SurfaceType, 29, 31);

   dw[1] =
      util_bitpack_uint(values->SurfaceQPitch, 0, 14) |
      util_bitpack_ufixed(values->BaseMipLevel, 19, 23, 1) |
      util_bitpack_uint_nonzero(values->MOCS, 24, 30);

   dw[2] =
      util_bitpack_uint(values->Width, 0, 13) |
      util_bitpack_uint(values->Height, 16, 29);

   dw[3] =
      util_bitpack_uint(values->SurfacePitch, 0, 17) |
      util_bitpack_uint(values->Depth, 21, 31);

   dw[4] =
      util_bitpack_uint(values->MultisamplePositionPaletteIndex, 0, 2) |
      util_bitpack_uint(values->NumberofMultisamples, 3, 5) |
      util_bitpack_uint(values->MultisampledSurfaceStorageFormat, 6, 6) |
      util_bitpack_uint(values->RenderTargetViewExtent, 7, 17) |
      util_bitpack_uint(values->MinimumArrayElement, 18, 28) |
      util_bitpack_uint(values->RenderTargetAndSampleUnormRotation, 29, 30);

   dw[5] =
      util_bitpack_uint(values->MIPCountLOD, 0, 3) |
      util_bitpack_uint(values->SurfaceMinLOD, 4, 7) |
      util_bitpack_uint(values->MipTailStartLOD, 8, 11) |
      util_bitpack_uint(values->CoherencyType, 14, 14) |
      util_bitpack_uint(values->TiledResourceMode, 18, 19) |
      util_bitpack_uint(values->EWADisableForCube, 20, 20) |
      util_bitpack_uint(values->YOffset, 21, 23) |
      util_bitpack_uint(values->XOffset, 25, 31);

   dw[6] =
      util_bitpack_uint(values->AuxiliarySurfaceMode, 0, 2) |
      util_bitpack_uint(values->YOffsetforUorUVPlane, 0, 13) |
      util_bitpack_uint(values->AuxiliarySurfacePitch, 3, 11) |
      util_bitpack_uint(values->AuxiliarySurfaceQPitch, 16, 30) |
      util_bitpack_uint(values->XOffsetforUorUVPlane, 16, 29) |
      util_bitpack_uint(values->SeparateUVPlaneEnable, 31, 31);

   dw[7] =
      util_bitpack_ufixed(values->ResourceMinLOD, 0, 11, 8) |
      util_bitpack_uint(values->ShaderChannelSelectAlpha, 16, 18) |
      util_bitpack_uint(values->ShaderChannelSelectBlue, 19, 21) |
      util_bitpack_uint(values->ShaderChannelSelectGreen, 22, 24) |
      util_bitpack_uint(values->ShaderChannelSelectRed, 25, 27) |
      util_bitpack_uint(values->MemoryCompressionEnable, 30, 30) |
      util_bitpack_uint(values->MemoryCompressionMode, 31, 31);

   const uint64_t v8_address =
      __gen_address(data, &dw[8], values->SurfaceBaseAddress, 0, 0, 63);
   dw[8] = v8_address;
   dw[9] = v8_address >> 32;

   const uint64_t v10 =
      util_bitpack_uint(values->QuiltWidth, 0, 4) |
      util_bitpack_uint(values->QuiltHeight, 5, 9) |
      util_bitpack_uint(values->AuxiliaryTableIndexforMediaCompressedSurface, 21, 31) |
      util_bitpack_uint(values->YOffsetforVPlane, 32, 45) |
      util_bitpack_uint(values->XOffsetforVPlane, 48, 61);
   const uint64_t v10_address =
      __gen_address(data, &dw[10], values->AuxiliarySurfaceBaseAddress, v10, 12, 63);
   dw[10] = v10_address;
   dw[11] = (v10_address >> 32) | (v10 >> 32);

   dw[12] =
      util_bitpack_float(values->HierarchicalDepthClearValue) |
      util_bitpack_sint(values->RedClearColor, 0, 31);

   dw[13] =
      util_bitpack_sint(values->GreenClearColor, 0, 31);

   dw[14] =
      util_bitpack_sint(values->BlueClearColor, 0, 31);

   dw[15] =
      util_bitpack_sint(values->AlphaClearColor, 0, 31);
}

#define GFX9_GPGPU_WALKER_length              15
#define GFX9_GPGPU_WALKER_length_bias          2
#define GFX9_GPGPU_WALKER_header                \
   .DWordLength                         =     13,  \
   .SubOpcode                           =      5,  \
   .MediaCommandOpcode                  =      1,  \
   .Pipeline                            =      2,  \
   .CommandType                         =      3

struct GFX9_GPGPU_WALKER {
   uint32_t                             DWordLength;
   bool                                 PredicateEnable;
   bool                                 IndirectParameterEnable;
   uint32_t                             SubOpcode;
   uint32_t                             MediaCommandOpcode;
   uint32_t                             Pipeline;
   uint32_t                             CommandType;
   uint32_t                             InterfaceDescriptorOffset;
   uint32_t                             IndirectDataLength;
   uint64_t                             IndirectDataStartAddress;
   uint32_t                             ThreadWidthCounterMaximum;
   uint32_t                             ThreadHeightCounterMaximum;
   uint32_t                             ThreadDepthCounterMaximum;
   uint32_t                             SIMDSize;
#define SIMD8                                    0
#define SIMD16                                   1
#define SIMD32                                   2
   uint32_t                             ThreadGroupIDStartingX;
   uint32_t                             ThreadGroupIDXDimension;
   uint32_t                             ThreadGroupIDStartingY;
   uint32_t                             ThreadGroupIDYDimension;
   uint32_t                             ThreadGroupIDStartingResumeZ;
   uint32_t                             ThreadGroupIDZDimension;
   uint32_t                             RightExecutionMask;
   uint32_t                             BottomExecutionMask;
};

static inline __attribute__((always_inline)) void
GFX9_GPGPU_WALKER_pack(__attribute__((unused)) __gen_user_data *data,
                       __attribute__((unused)) void * restrict dst,
                       __attribute__((unused)) const struct GFX9_GPGPU_WALKER * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      util_bitpack_uint(values->DWordLength, 0, 7) |
      util_bitpack_uint(values->PredicateEnable, 8, 8) |
      util_bitpack_uint(values->IndirectParameterEnable, 10, 10) |
      util_bitpack_uint(values->SubOpcode, 16, 23) |
      util_bitpack_uint(values->MediaCommandOpcode, 24, 26) |
      util_bitpack_uint(values->Pipeline, 27, 28) |
      util_bitpack_uint(values->CommandType, 29, 31);

   dw[1] =
      util_bitpack_uint(values->InterfaceDescriptorOffset, 0, 5);

   dw[2] =
      util_bitpack_uint(values->IndirectDataLength, 0, 16);

   dw[3] =
      __gen_offset(values->IndirectDataStartAddress, 6, 31);

   dw[4] =
      util_bitpack_uint(values->ThreadWidthCounterMaximum, 0, 5) |
      util_bitpack_uint(values->ThreadHeightCounterMaximum, 8, 13) |
      util_bitpack_uint(values->ThreadDepthCounterMaximum, 16, 21) |
      util_bitpack_uint(values->SIMDSize, 30, 31);

   dw[5] =
      util_bitpack_uint(values->ThreadGroupIDStartingX, 0, 31);

   dw[6] = 0;

   dw[7] =
      util_bitpack_uint(values->ThreadGroupIDXDimension, 0, 31);

   dw[8] =
      util_bitpack_uint(values->ThreadGroupIDStartingY, 0, 31);

   dw[9] = 0;

   dw[10] =
      util_bitpack_uint(values->ThreadGroupIDYDimension, 0, 31);

   dw[11] =
      util_bitpack_uint(values->ThreadGroupIDStartingResumeZ, 0, 31);

   dw[12] =
      util_bitpack_uint(values->ThreadGroupIDZDimension, 0, 31);

   dw[13] =
      util_bitpack_uint(values->RightExecutionMask, 0, 31);

   dw[14] =
      util_bitpack_uint(values->BottomExecutionMask, 0, 31);
}

#define GFX9_MEDIA_INTERFACE_DESCRIPTOR_LOAD_length      4
#define GFX9_MEDIA_INTERFACE_DESCRIPTOR_LOAD_length_bias      2
#define GFX9_MEDIA_INTERFACE_DESCRIPTOR_LOAD_header\
   .DWordLength                         =      2,  \
   .SubOpcode                           =      2,  \
   .MediaCommandOpcode                  =      0,  \
   .Pipeline                            =      2,  \
   .CommandType                         =      3

struct GFX9_MEDIA_INTERFACE_DESCRIPTOR_LOAD {
   uint32_t                             DWordLength;
   uint32_t                             SubOpcode;
   uint32_t                             MediaCommandOpcode;
   uint32_t                             Pipeline;
   uint32_t                             CommandType;
   uint32_t                             InterfaceDescriptorTotalLength;
   uint64_t                             InterfaceDescriptorDataStartAddress;
};

static inline __attribute__((always_inline)) void
GFX9_MEDIA_INTERFACE_DESCRIPTOR_LOAD_pack(__attribute__((unused)) __gen_user_data *data,
                                          __attribute__((unused)) void * restrict dst,
                                          __attribute__((unused)) const struct GFX9_MEDIA_INTERFACE_DESCRIPTOR_LOAD * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      util_bitpack_uint(values->DWordLength, 0, 15) |
      util_bitpack_uint(values->SubOpcode, 16, 23) |
      util_bitpack_uint(values->MediaCommandOpcode, 24, 26) |
      util_bitpack_uint(values->Pipeline, 27, 28) |
      util_bitpack_uint(values->CommandType, 29, 31);

   dw[1] = 0;

   dw[2] =
      util_bitpack_uint(values->InterfaceDescriptorTotalLength, 0, 16);

   dw[3] =
      __gen_offset(values->InterfaceDescriptorDataStartAddress, 0, 31);
}

#define GFX9_MEDIA_STATE_FLUSH_length          2
#define GFX9_MEDIA_STATE_FLUSH_length_bias      2
#define GFX9_MEDIA_STATE_FLUSH_header           \
   .DWordLength                         =      0,  \
   .SubOpcode                           =      4,  \
   .MediaCommandOpcode                  =      0,  \
   .Pipeline                            =      2,  \
   .CommandType                         =      3

struct GFX9_MEDIA_STATE_FLUSH {
   uint32_t                             DWordLength;
   uint32_t                             SubOpcode;
   uint32_t                             MediaCommandOpcode;
   uint32_t                             Pipeline;
   uint32_t                             CommandType;
   uint32_t                             InterfaceDescriptorOffset;
   uint32_t                             WatermarkRequired;
   bool                                 FlushtoGO;
};

static inline __attribute__((always_inline)) void
GFX9_MEDIA_STATE_FLUSH_pack(__attribute__((unused)) __gen_user_data *data,
                            __attribute__((unused)) void * restrict dst,
                            __attribute__((unused)) const struct GFX9_MEDIA_STATE_FLUSH * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      util_bitpack_uint(values->DWordLength, 0, 15) |
      util_bitpack_uint(values->SubOpcode, 16, 23) |
      util_bitpack_uint(values->MediaCommandOpcode, 24, 26) |
      util_bitpack_uint(values->Pipeline, 27, 28) |
      util_bitpack_uint(values->CommandType, 29, 31);

   dw[1] =
      util_bitpack_uint(values->InterfaceDescriptorOffset, 0, 5) |
      util_bitpack_uint(values->WatermarkRequired, 6, 6) |
      util_bitpack_uint(values->FlushtoGO, 7, 7);
}

#define GFX9_MEDIA_VFE_STATE_length            9
#define GFX9_MEDIA_VFE_STATE_length_bias       2
#define GFX9_MEDIA_VFE_STATE_header             \
   .DWordLength                         =      7,  \
   .SubOpcode                           =      0,  \
   .MediaCommandOpcode                  =      0,  \
   .Pipeline                            =      2,  \
   .CommandType                         =      3

struct GFX9_MEDIA_VFE_STATE {
   uint32_t                             DWordLength;
   uint32_t                             SubOpcode;
   uint32_t                             MediaCommandOpcode;
   uint32_t                             Pipeline;
   uint32_t                             CommandType;
   uint32_t                             PerThreadScratchSpace;
   uint32_t                             StackSize;
   __gen_address_type                   ScratchSpaceBasePointer;
   uint32_t                             ResetGatewayTimer;
#define Maintainingtheexistingtimestampstate     0
#define Resettingrelativetimerandlatchingtheglobaltimestamp 1
   uint32_t                             NumberofURBEntries;
   uint32_t                             MaximumNumberofThreads;
   uint32_t                             SliceDisable;
#define AllSubslicesEnabled                      0
#define OnlySlice0Enabled                        1
#define OnlySlice0Subslice0Enabled               3
   uint32_t                             CURBEAllocationSize;
   uint32_t                             URBEntryAllocationSize;
   uint32_t                             ScoreboardMask;
   uint32_t                             ScoreboardType;
#define StallingScoreboard                       0
#define NonStallingScoreboard                    1
   bool                                 ScoreboardEnable;
   int32_t                              Scoreboard0DeltaX;
   int32_t                              Scoreboard0DeltaY;
   int32_t                              Scoreboard1DeltaX;
   int32_t                              Scoreboard1DeltaY;
   int32_t                              Scoreboard2DeltaX;
   int32_t                              Scoreboard2DeltaY;
   int32_t                              Scoreboard3DeltaX;
   int32_t                              Scoreboard3DeltaY;
   int32_t                              Scoreboard4DeltaX;
   int32_t                              Scoreboard4DeltaY;
   int32_t                              Scoreboard5DeltaX;
   int32_t                              Scoreboard5DeltaY;
   int32_t                              Scoreboard6DeltaX;
   int32_t                              Scoreboard6DeltaY;
   int32_t                              Scoreboard7DeltaX;
   int32_t                              Scoreboard7DeltaY;
};

static inline __attribute__((always_inline)) void
GFX9_MEDIA_VFE_STATE_pack(__attribute__((unused)) __gen_user_data *data,
                          __attribute__((unused)) void * restrict dst,
                          __attribute__((unused)) const struct GFX9_MEDIA_VFE_STATE * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      util_bitpack_uint(values->DWordLength, 0, 15) |
      util_bitpack_uint(values->SubOpcode, 16, 23) |
      util_bitpack_uint(values->MediaCommandOpcode, 24, 26) |
      util_bitpack_uint(values->Pipeline, 27, 28) |
      util_bitpack_uint(values->CommandType, 29, 31);

   const uint64_t v1 =
      util_bitpack_uint(values->PerThreadScratchSpace, 0, 3) |
      util_bitpack_uint(values->StackSize, 4, 7);
   const uint64_t v1_address =
      __gen_address(data, &dw[1], values->ScratchSpaceBasePointer, v1, 10, 47);
   dw[1] = v1_address;
   dw[2] = (v1_address >> 32) | (v1 >> 32);

   dw[3] =
      util_bitpack_uint(values->ResetGatewayTimer, 7, 7) |
      util_bitpack_uint(values->NumberofURBEntries, 8, 15) |
      util_bitpack_uint(values->MaximumNumberofThreads, 16, 31);

   dw[4] =
      util_bitpack_uint(values->SliceDisable, 0, 1);

   dw[5] =
      util_bitpack_uint(values->CURBEAllocationSize, 0, 15) |
      util_bitpack_uint(values->URBEntryAllocationSize, 16, 31);

   dw[6] =
      util_bitpack_uint(values->ScoreboardMask, 0, 7) |
      util_bitpack_uint(values->ScoreboardType, 30, 30) |
      util_bitpack_uint(values->ScoreboardEnable, 31, 31);

   dw[7] =
      util_bitpack_sint(values->Scoreboard0DeltaX, 0, 3) |
      util_bitpack_sint(values->Scoreboard0DeltaY, 4, 7) |
      util_bitpack_sint(values->Scoreboard1DeltaX, 8, 11) |
      util_bitpack_sint(values->Scoreboard1DeltaY, 12, 15) |
      util_bitpack_sint(values->Scoreboard2DeltaX, 16, 19) |
      util_bitpack_sint(values->Scoreboard2DeltaY, 20, 23) |
      util_bitpack_sint(values->Scoreboard3DeltaX, 24, 27) |
      util_bitpack_sint(values->Scoreboard3DeltaY, 28, 31);

   dw[8] =
      util_bitpack_sint(values->Scoreboard4DeltaX, 0, 3) |
      util_bitpack_sint(values->Scoreboard4DeltaY, 4, 7) |
      util_bitpack_sint(values->Scoreboard5DeltaX, 8, 11) |
      util_bitpack_sint(values->Scoreboard5DeltaY, 12, 15) |
      util_bitpack_sint(values->Scoreboard6DeltaX, 16, 19) |
      util_bitpack_sint(values->Scoreboard6DeltaY, 20, 23) |
      util_bitpack_sint(values->Scoreboard7DeltaX, 24, 27) |
      util_bitpack_sint(values->Scoreboard7DeltaY, 28, 31);
}

#define GFX9_MI_BATCH_BUFFER_END_length        1
#define GFX9_MI_BATCH_BUFFER_END_length_bias      1
#define GFX9_MI_BATCH_BUFFER_END_header         \
   .MICommandOpcode                     =     10,  \
   .CommandType                         =      0

struct GFX9_MI_BATCH_BUFFER_END {
   uint32_t                             MICommandOpcode;
   uint32_t                             CommandType;
};

static inline __attribute__((always_inline)) void
GFX9_MI_BATCH_BUFFER_END_pack(__attribute__((unused)) __gen_user_data *data,
                              __attribute__((unused)) void * restrict dst,
                              __attribute__((unused)) const struct GFX9_MI_BATCH_BUFFER_END * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      util_bitpack_uint(values->MICommandOpcode, 23, 28) |
      util_bitpack_uint(values->CommandType, 29, 31);
}

#define GFX9_MI_LOAD_REGISTER_IMM_length       3
#define GFX9_MI_LOAD_REGISTER_IMM_length_bias      2
#define GFX9_MI_LOAD_REGISTER_IMM_header        \
   .DWordLength                         =      1,  \
   .MICommandOpcode                     =     34,  \
   .CommandType                         =      0

struct GFX9_MI_LOAD_REGISTER_IMM {
   uint32_t                             DWordLength;
   uint32_t                             ByteWriteDisables;
   uint32_t                             MICommandOpcode;
   uint32_t                             CommandType;
   uint64_t                             RegisterOffset;
   uint32_t                             DataDWord;
   /* variable length fields follow */
};

static inline __attribute__((always_inline)) void
GFX9_MI_LOAD_REGISTER_IMM_pack(__attribute__((unused)) __gen_user_data *data,
                               __attribute__((unused)) void * restrict dst,
                               __attribute__((unused)) const struct GFX9_MI_LOAD_REGISTER_IMM * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      util_bitpack_uint(values->DWordLength, 0, 7) |
      util_bitpack_uint(values->ByteWriteDisables, 8, 11) |
      util_bitpack_uint(values->MICommandOpcode, 23, 28) |
      util_bitpack_uint(values->CommandType, 29, 31);

   dw[1] =
      __gen_offset(values->RegisterOffset, 2, 22);

   dw[2] =
      util_bitpack_uint(values->DataDWord, 0, 31);
}

#define GFX9_MI_NOOP_length                    1
#define GFX9_MI_NOOP_length_bias               1
#define GFX9_MI_NOOP_header                     \
   .MICommandOpcode                     =      0,  \
   .CommandType                         =      0

struct GFX9_MI_NOOP {
   uint32_t                             IdentificationNumber;
   bool                                 IdentificationNumberRegisterWriteEnable;
   uint32_t                             MICommandOpcode;
   uint32_t                             CommandType;
};

static inline __attribute__((always_inline)) void
GFX9_MI_NOOP_pack(__attribute__((unused)) __gen_user_data *data,
                  __attribute__((unused)) void * restrict dst,
                  __attribute__((unused)) const struct GFX9_MI_NOOP * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      util_bitpack_uint(values->IdentificationNumber, 0, 21) |
      util_bitpack_uint(values->IdentificationNumberRegisterWriteEnable, 22, 22) |
      util_bitpack_uint(values->MICommandOpcode, 23, 28) |
      util_bitpack_uint(values->CommandType, 29, 31);
}

#define GFX9_MI_STORE_REGISTER_MEM_length      4
#define GFX9_MI_STORE_REGISTER_MEM_length_bias      2
#define GFX9_MI_STORE_REGISTER_MEM_header       \
   .DWordLength                         =      2,  \
   .MICommandOpcode                     =     36,  \
   .CommandType                         =      0

struct GFX9_MI_STORE_REGISTER_MEM {
   uint32_t                             DWordLength;
   bool                                 PredicateEnable;
   bool                                 UseGlobalGTT;
   uint32_t                             MICommandOpcode;
   uint32_t                             CommandType;
   uint64_t                             RegisterAddress;
   __gen_address_type                   MemoryAddress;
};

static inline __attribute__((always_inline)) void
GFX9_MI_STORE_REGISTER_MEM_pack(__attribute__((unused)) __gen_user_data *data,
                                __attribute__((unused)) void * restrict dst,
                                __attribute__((unused)) const struct GFX9_MI_STORE_REGISTER_MEM * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      util_bitpack_uint(values->DWordLength, 0, 7) |
      util_bitpack_uint(values->PredicateEnable, 21, 21) |
      util_bitpack_uint(values->UseGlobalGTT, 22, 22) |
      util_bitpack_uint(values->MICommandOpcode, 23, 28) |
      util_bitpack_uint(values->CommandType, 29, 31);

   dw[1] =
      __gen_offset(values->RegisterAddress, 2, 22);

   const uint64_t v2_address =
      __gen_address(data, &dw[2], values->MemoryAddress, 0, 2, 63);
   dw[2] = v2_address;
   dw[3] = v2_address >> 32;
}

#define GFX9_PIPELINE_SELECT_length            1
#define GFX9_PIPELINE_SELECT_length_bias       1
#define GFX9_PIPELINE_SELECT_header             \
   ._3DCommandSubOpcode                 =      4,  \
   ._3DCommandOpcode                    =      1,  \
   .CommandSubType                      =      1,  \
   .CommandType                         =      3

struct GFX9_PIPELINE_SELECT {
   uint32_t                             PipelineSelection;
#define _3D                                      0
#define Media                                    1
#define GPGPU                                    2
   bool                                 MediaSamplerDOPClockGateEnable;
   bool                                 ForceMediaAwake;
   uint32_t                             MaskBits;
   uint32_t                             _3DCommandSubOpcode;
   uint32_t                             _3DCommandOpcode;
   uint32_t                             CommandSubType;
   uint32_t                             CommandType;
};

static inline __attribute__((always_inline)) void
GFX9_PIPELINE_SELECT_pack(__attribute__((unused)) __gen_user_data *data,
                          __attribute__((unused)) void * restrict dst,
                          __attribute__((unused)) const struct GFX9_PIPELINE_SELECT * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      util_bitpack_uint(values->PipelineSelection, 0, 1) |
      util_bitpack_uint(values->MediaSamplerDOPClockGateEnable, 4, 4) |
      util_bitpack_uint(values->ForceMediaAwake, 5, 5) |
      util_bitpack_uint(values->MaskBits, 8, 15) |
      util_bitpack_uint(values->_3DCommandSubOpcode, 16, 23) |
      util_bitpack_uint(values->_3DCommandOpcode, 24, 26) |
      util_bitpack_uint(values->CommandSubType, 27, 28) |
      util_bitpack_uint(values->CommandType, 29, 31);
}

#define GFX9_PIPE_CONTROL_length               6
#define GFX9_PIPE_CONTROL_length_bias          2
#define GFX9_PIPE_CONTROL_header                \
   .DWordLength                         =      4,  \
   ._3DCommandSubOpcode                 =      0,  \
   ._3DCommandOpcode                    =      2,  \
   .CommandSubType                      =      3,  \
   .CommandType                         =      3

struct GFX9_PIPE_CONTROL {
   uint32_t                             DWordLength;
   uint32_t                             _3DCommandSubOpcode;
   uint32_t                             _3DCommandOpcode;
   uint32_t                             CommandSubType;
   uint32_t                             CommandType;
   bool                                 DepthCacheFlushEnable;
   bool                                 StallAtPixelScoreboard;
   bool                                 StateCacheInvalidationEnable;
   bool                                 ConstantCacheInvalidationEnable;
   bool                                 VFCacheInvalidationEnable;
   bool                                 DCFlushEnable;
   bool                                 PipeControlFlushEnable;
   bool                                 NotifyEnable;
   bool                                 IndirectStatePointersDisable;
   bool                                 TextureCacheInvalidationEnable;
   bool                                 InstructionCacheInvalidateEnable;
   bool                                 RenderTargetCacheFlushEnable;
   bool                                 DepthStallEnable;
   uint32_t                             PostSyncOperation;
#define NoWrite                                  0
#define WriteImmediateData                       1
#define WritePSDepthCount                        2
#define WriteTimestamp                           3
   bool                                 GenericMediaStateClear;
   bool                                 TLBInvalidate;
   bool                                 GlobalSnapshotCountReset;
   bool                                 CommandStreamerStallEnable;
   uint32_t                             StoreDataIndex;
   uint32_t                             LRIPostSyncOperation;
#define NoLRIOperation                           0
#define MMIOWriteImmediateData                   1
   uint32_t                             DestinationAddressType;
#define DAT_PPGTT                                0
#define DAT_GGTT                                 1
   bool                                 FlushLLC;
   __gen_address_type                   Address;
   uint64_t                             ImmediateData;
};

static inline __attribute__((always_inline)) void
GFX9_PIPE_CONTROL_pack(__attribute__((unused)) __gen_user_data *data,
                       __attribute__((unused)) void * restrict dst,
                       __attribute__((unused)) const struct GFX9_PIPE_CONTROL * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      util_bitpack_uint(values->DWordLength, 0, 7) |
      util_bitpack_uint(values->_3DCommandSubOpcode, 16, 23) |
      util_bitpack_uint(values->_3DCommandOpcode, 24, 26) |
      util_bitpack_uint(values->CommandSubType, 27, 28) |
      util_bitpack_uint(values->CommandType, 29, 31);

   dw[1] =
      util_bitpack_uint(values->DepthCacheFlushEnable, 0, 0) |
      util_bitpack_uint(values->StallAtPixelScoreboard, 1, 1) |
      util_bitpack_uint(values->StateCacheInvalidationEnable, 2, 2) |
      util_bitpack_uint(values->ConstantCacheInvalidationEnable, 3, 3) |
      util_bitpack_uint(values->VFCacheInvalidationEnable, 4, 4) |
      util_bitpack_uint(values->DCFlushEnable, 5, 5) |
      util_bitpack_uint(values->PipeControlFlushEnable, 7, 7) |
      util_bitpack_uint(values->NotifyEnable, 8, 8) |
      util_bitpack_uint(values->IndirectStatePointersDisable, 9, 9) |
      util_bitpack_uint(values->TextureCacheInvalidationEnable, 10, 10) |
      util_bitpack_uint(values->InstructionCacheInvalidateEnable, 11, 11) |
      util_bitpack_uint(values->RenderTargetCacheFlushEnable, 12, 12) |
      util_bitpack_uint(values->DepthStallEnable, 13, 13) |
      util_bitpack_uint(values->PostSyncOperation, 14, 15) |
      util_bitpack_uint(values->GenericMediaStateClear, 16, 16) |
      util_bitpack_uint(values->TLBInvalidate, 18, 18) |
      util_bitpack_uint(values->GlobalSnapshotCountReset, 19, 19) |
      util_bitpack_uint(values->CommandStreamerStallEnable, 20, 20) |
      util_bitpack_uint(values->StoreDataIndex, 21, 21) |
      util_bitpack_uint(values->LRIPostSyncOperation, 23, 23) |
      util_bitpack_uint(values->DestinationAddressType, 24, 24) |
      util_bitpack_uint(values->FlushLLC, 26, 26);

   const uint64_t v2_address =
      __gen_address(data, &dw[2], values->Address, 0, 2, 47);
   dw[2] = v2_address;
   dw[3] = v2_address >> 32;

   const uint64_t v4 =
      util_bitpack_uint(values->ImmediateData, 0, 63);
   dw[4] = v4;
   dw[5] = v4 >> 32;
}

#define GFX9_STATE_BASE_ADDRESS_length        19
#define GFX9_STATE_BASE_ADDRESS_length_bias      2
#define GFX9_STATE_BASE_ADDRESS_header          \
   .DWordLength                         =     17,  \
   ._3DCommandSubOpcode                 =      1,  \
   ._3DCommandOpcode                    =      1,  \
   .CommandSubType                      =      0,  \
   .CommandType                         =      3

struct GFX9_STATE_BASE_ADDRESS {
   uint32_t                             DWordLength;
   uint32_t                             _3DCommandSubOpcode;
   uint32_t                             _3DCommandOpcode;
   uint32_t                             CommandSubType;
   uint32_t                             CommandType;
   bool                                 GeneralStateBaseAddressModifyEnable;
   uint32_t                             GeneralStateMOCS;
   __gen_address_type                   GeneralStateBaseAddress;
   uint32_t                             StatelessDataPortAccessMOCS;
   bool                                 SurfaceStateBaseAddressModifyEnable;
   uint32_t                             SurfaceStateMOCS;
   __gen_address_type                   SurfaceStateBaseAddress;
   bool                                 DynamicStateBaseAddressModifyEnable;
   uint32_t                             DynamicStateMOCS;
   __gen_address_type                   DynamicStateBaseAddress;
   bool                                 IndirectObjectBaseAddressModifyEnable;
   uint32_t                             IndirectObjectMOCS;
   __gen_address_type                   IndirectObjectBaseAddress;
   bool                                 InstructionBaseAddressModifyEnable;
   uint32_t                             InstructionMOCS;
   __gen_address_type                   InstructionBaseAddress;
   bool                                 GeneralStateBufferSizeModifyEnable;
   uint32_t                             GeneralStateBufferSize;
   bool                                 DynamicStateBufferSizeModifyEnable;
   uint32_t                             DynamicStateBufferSize;
   bool                                 IndirectObjectBufferSizeModifyEnable;
   uint32_t                             IndirectObjectBufferSize;
   bool                                 InstructionBuffersizeModifyEnable;
   uint32_t                             InstructionBufferSize;
   bool                                 BindlessSurfaceStateBaseAddressModifyEnable;
   uint32_t                             BindlessSurfaceStateMOCS;
   __gen_address_type                   BindlessSurfaceStateBaseAddress;
   uint32_t                             BindlessSurfaceStateSize;
};

static inline __attribute__((always_inline)) void
GFX9_STATE_BASE_ADDRESS_pack(__attribute__((unused)) __gen_user_data *data,
                             __attribute__((unused)) void * restrict dst,
                             __attribute__((unused)) const struct GFX9_STATE_BASE_ADDRESS * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      util_bitpack_uint(values->DWordLength, 0, 7) |
      util_bitpack_uint(values->_3DCommandSubOpcode, 16, 23) |
      util_bitpack_uint(values->_3DCommandOpcode, 24, 26) |
      util_bitpack_uint(values->CommandSubType, 27, 28) |
      util_bitpack_uint(values->CommandType, 29, 31);

   const uint64_t v1 =
      util_bitpack_uint(values->GeneralStateBaseAddressModifyEnable, 0, 0) |
      util_bitpack_uint_nonzero(values->GeneralStateMOCS, 4, 10);
   const uint64_t v1_address =
      __gen_address(data, &dw[1], values->GeneralStateBaseAddress, v1, 12, 63);
   dw[1] = v1_address;
   dw[2] = (v1_address >> 32) | (v1 >> 32);

   dw[3] =
      util_bitpack_uint_nonzero(values->StatelessDataPortAccessMOCS, 16, 22);

   const uint64_t v4 =
      util_bitpack_uint(values->SurfaceStateBaseAddressModifyEnable, 0, 0) |
      util_bitpack_uint_nonzero(values->SurfaceStateMOCS, 4, 10);
   const uint64_t v4_address =
      __gen_address(data, &dw[4], values->SurfaceStateBaseAddress, v4, 12, 63);
   dw[4] = v4_address;
   dw[5] = (v4_address >> 32) | (v4 >> 32);

   const uint64_t v6 =
      util_bitpack_uint(values->DynamicStateBaseAddressModifyEnable, 0, 0) |
      util_bitpack_uint_nonzero(values->DynamicStateMOCS, 4, 10);
   const uint64_t v6_address =
      __gen_address(data, &dw[6], values->DynamicStateBaseAddress, v6, 12, 63);
   dw[6] = v6_address;
   dw[7] = (v6_address >> 32) | (v6 >> 32);

   const uint64_t v8 =
      util_bitpack_uint(values->IndirectObjectBaseAddressModifyEnable, 0, 0) |
      util_bitpack_uint_nonzero(values->IndirectObjectMOCS, 4, 10);
   const uint64_t v8_address =
      __gen_address(data, &dw[8], values->IndirectObjectBaseAddress, v8, 12, 63);
   dw[8] = v8_address;
   dw[9] = (v8_address >> 32) | (v8 >> 32);

   const uint64_t v10 =
      util_bitpack_uint(values->InstructionBaseAddressModifyEnable, 0, 0) |
      util_bitpack_uint_nonzero(values->InstructionMOCS, 4, 10);
   const uint64_t v10_address =
      __gen_address(data, &dw[10], values->InstructionBaseAddress, v10, 12, 63);
   dw[10] = v10_address;
   dw[11] = (v10_address >> 32) | (v10 >> 32);

   dw[12] =
      util_bitpack_uint(values->GeneralStateBufferSizeModifyEnable, 0, 0) |
      util_bitpack_uint(values->GeneralStateBufferSize, 12, 31);

   dw[13] =
      util_bitpack_uint(values->DynamicStateBufferSizeModifyEnable, 0, 0) |
      util_bitpack_uint(values->DynamicStateBufferSize, 12, 31);

   dw[14] =
      util_bitpack_uint(values->IndirectObjectBufferSizeModifyEnable, 0, 0) |
      util_bitpack_uint(values->IndirectObjectBufferSize, 12, 31);

   dw[15] =
      util_bitpack_uint(values->InstructionBuffersizeModifyEnable, 0, 0) |
      util_bitpack_uint(values->InstructionBufferSize, 12, 31);

   const uint64_t v16 =
      util_bitpack_uint(values->BindlessSurfaceStateBaseAddressModifyEnable, 0, 0) |
      util_bitpack_uint_nonzero(values->BindlessSurfaceStateMOCS, 4, 10);
   const uint64_t v16_address =
      __gen_address(data, &dw[16], values->BindlessSurfaceStateBaseAddress, v16, 12, 63);
   dw[16] = v16_address;
   dw[17] = (v16_address >> 32) | (v16 >> 32);

   dw[18] =
      util_bitpack_uint(values->BindlessSurfaceStateSize, 12, 31);
}

#define GFX9_L3CNTLREG_num                0x7034
#define GFX9_L3CNTLREG_length                  1
struct GFX9_L3CNTLREG {
   bool                                 SLMEnable;
   uint32_t                             URBAllocation;
   uint32_t                             ROAllocation;
   uint32_t                             DCAllocation;
   uint32_t                             AllAllocation;
};

static inline __attribute__((always_inline)) void
GFX9_L3CNTLREG_pack(__attribute__((unused)) __gen_user_data *data,
                    __attribute__((unused)) void * restrict dst,
                    __attribute__((unused)) const struct GFX9_L3CNTLREG * restrict values)
{
   uint32_t * restrict dw = (uint32_t * restrict) dst;

   dw[0] =
      util_bitpack_uint(values->SLMEnable, 0, 0) |
      util_bitpack_uint(values->URBAllocation, 1, 7) |
      util_bitpack_uint(values->ROAllocation, 11, 17) |
      util_bitpack_uint(values->DCAllocation, 18, 24) |
      util_bitpack_uint(values->AllAllocation, 25, 31);
}

#endif /* GFX9_SKL_PACK_H */
