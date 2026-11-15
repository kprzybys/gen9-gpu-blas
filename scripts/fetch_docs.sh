#!/bin/sh
# Fetches the reference documentation into third_party/ (not in git; state as of 2026-11-12). Files already present are kept.
# Intel's index page blocks automated clients (403), but files from cdrdv2.intel.com download directly.
set -e
cd -P "$(dirname "$0")/.." && mkdir -p third_party && cd third_party
UA="Mozilla/5.0 (X11; Linux x86_64)"
mkdir -p pdf txt khronos
while read id name; do
  [ -f "pdf/$name" ] || curl -sfL -A "$UA" -o "pdf/$name" "https://cdrdv2.intel.com/v1/dl/getContent/$id"
done <<'L'
723910 intel-gfx-prm-osrc-kbl-vol01-preface.pdf
723916 intel-gfx-prm-osrc-kbl-vol02a-commandreference-instructions.pdf
723917 intel-gfx-prm-osrc-kbl-vol02b-commandreference-enumerations.pdf
723936 intel-gfx-prm-osrc-kbl-vol02d-commandreference-structures.pdf
723943 intel-gfx-prm-osrc-kbl-vol03-gpu_overview.pdf
723945 intel-gfx-prm-osrc-kbl-vol04-configurations.pdf
723948 intel-gfx-prm-osrc-kbl-vol05-memory_views.pdf
723952 intel-gfx-prm-osrc-kbl-vol07-3d_media_gpgpu.pdf
723951 intel-gfx-prm-osrc-kbl-vol06-command_stream_programming.pdf
723921 intel-gfx-prm-osrc-kbl-vol02c-commandreference-registers-part1.pdf
723924 intel-gfx-prm-osrc-kbl-vol02c-commandreference-registers-part2_0.pdf
L
f=pdf/the-compute-architecture-of-intel-processor-graphics-gen9-v1d0.pdf
[ -f "$f" ] || curl -sfL -A "$UA" -o "$f" https://www.intel.com/content/dam/develop/external/us/en/documents/the-compute-architecture-of-intel-processor-graphics-gen9-v1d0.pdf
sha256sum_check() {
  # Warn (do not fail silently) when a downloaded PDF differs from the one the documentation was written against.
  sha256sum -c --quiet - 2>&1 <<'S' || echo "WARNING: some PDFs differ from the reference checksums (Intel may have re-issued them); page numbers in docs/ may not match" >&2
1833ab7fcd7f132f88504e2b243fe357f7f86ac1350838aacf407e291c03dbaf  pdf/intel-gfx-prm-osrc-kbl-vol01-preface.pdf
0f2bd4d5fdcdb502606bda01d91247a4101a076c8f1ffeb620fc42beb3f141a4  pdf/intel-gfx-prm-osrc-kbl-vol02a-commandreference-instructions.pdf
bcae959ab2f87fb7cf00412d396a97fb7e62b3824bc131e411677d97833d8e0c  pdf/intel-gfx-prm-osrc-kbl-vol02b-commandreference-enumerations.pdf
c6fcd0256fe9dcc51695d68e7a12e19a87e17b40004ff23277f1a452caa72faf  pdf/intel-gfx-prm-osrc-kbl-vol02c-commandreference-registers-part1.pdf
542454f3c013ab291296b9fb26cbc9fe16c465783f7436fe43e728dcde01de8f  pdf/intel-gfx-prm-osrc-kbl-vol02c-commandreference-registers-part2_0.pdf
3aca1f3230584dc598799871ba901f5b40fcbd01f19edd41dd167a9e6768a805  pdf/intel-gfx-prm-osrc-kbl-vol02d-commandreference-structures.pdf
9aec33ce96790de7b0402744591788041221a9e2ec92fc2ae280003f505591ae  pdf/intel-gfx-prm-osrc-kbl-vol03-gpu_overview.pdf
aa1d9bc58ad40bf5ef914de8edcfbda825bfdd279944b719f4c406fda897a481  pdf/intel-gfx-prm-osrc-kbl-vol04-configurations.pdf
8b15ca979b4f356888dd0b4b46482cd05457949974c1dde1abd6914f911fd8e0  pdf/intel-gfx-prm-osrc-kbl-vol05-memory_views.pdf
432bda751f1ef8925ee65eec3f2dedc7359c88441a03cbdef601e5f62686cc9b  pdf/intel-gfx-prm-osrc-kbl-vol06-command_stream_programming.pdf
22aa547d0b634af217917d71642791f522ace6a9567aed74451c040fb99a7bee  pdf/intel-gfx-prm-osrc-kbl-vol07-3d_media_gpgpu.pdf
5a944ad0bf9b078816480017bae4c31bfb927678c525e0f35b25154464a59f14  pdf/the-compute-architecture-of-intel-processor-graphics-gen9-v1d0.pdf
b7856dc37eaec700f536dd46d420566df988609af0ec8637e63c70f46eb79388  pdf/intel-gfx-prm-osrc-chv-bsw-vol01-preface.pdf
8160b3ccec3e8ff4c8880e68c75860d424e21f1285c4a36b345bcd11b7bed23d  pdf/intel-gfx-prm-osrc-chv-bsw-vol02a-commandreference-instructions.pdf
bb1aa4c3ae761da97bd062aed12da0f70d54486ebc62c8b2295c5600ee4070cf  pdf/intel-gfx-prm-osrc-chv-bsw-vol02b-commandreference-enumerations.pdf
bd244693e90ec38f13a343c7d294dac639b5d4732ffa721b7d493311daa46825  pdf/intel-gfx-prm-osrc-chv-bsw-vol02c-commandreference-registers.pdf
36c834f1de8031a8d1e816ff955b8a21947af452bb531fed6a4320eff8170a64  pdf/intel-gfx-prm-osrc-chv-bsw-vol02d-commandreference-structures.pdf
9a366599bca17d80c89d5636c572c7f8af12f8844de8787b9acf63eee8147e6d  pdf/intel-gfx-prm-osrc-chv-bsw-vol03-gpu-overview.pdf
70f66858160430030f76141ea5370a0f8517786821db4ee51ec773aa674583cb  pdf/intel-gfx-prm-osrc-chv-bsw-vol04-configurations.pdf
ee33cf83538efdf4bb1edb89c5d00d16d7780b0fea3e7a04cdec5340690bd713  pdf/intel-gfx-prm-osrc-chv-bsw-vol05-memory-views.pdf
69db0f8445fff568751dcc51b1623449ecfb8558e2a2f5fb0a6e1c3811b72f06  pdf/intel-gfx-prm-osrc-chv-bsw-vol06-command-stream-programming.pdf
665b536f44bb73c64248bdca313faf8a9423b1f3b8359cb3e56c4345be6becc4  pdf/intel-gfx-prm-osrc-chv-bsw-vol07-3d-media-gpgpu.pdf
6ef668518f386f85596cc6cbc817e827c376ebee2ea7af197e6aec85e8d894b8  pdf/intel-gfx-prm-osrc-chv-bsw-vol16-workarounds.pdf
S
}
# Gen8 (Cherryview/Braswell, docs/gen8.md): the PRM from the x.org mirror (Intel's cdrdv2 IDs for CHV are not known);
# no browser User-Agent here -- x.org answers it with 418
for name in vol01-preface vol02a-commandreference-instructions vol02b-commandreference-enumerations \
            vol02c-commandreference-registers vol02d-commandreference-structures vol03-gpu-overview vol04-configurations \
            vol05-memory-views vol06-command-stream-programming vol07-3d-media-gpgpu vol16-workarounds; do
  f=pdf/intel-gfx-prm-osrc-chv-bsw-$name.pdf
  [ -f "$f" ] || curl -sfL -o "$f" "https://www.x.org/docs/intel/CHV/intel-gfx-prm-osrc-chv-bsw-$name.pdf"
done
sha256sum_check
for p in pdf/*.pdf; do b=$(basename "$p" .pdf); [ -f "txt/$b.txt" ] || pdftotext -layout "$p" "txt/$b.txt"; done
for e in cl_intel_subgroups cl_intel_subgroups_short cl_intel_subgroups_char cl_intel_required_subgroup_size cl_intel_spirv_subgroups; do
  [ -f "khronos/$e.md" ] || curl -sfL "https://registry.khronos.org/OpenCL/extensions/intel/$e.html" | pandoc -f html -t gfm-raw_html --wrap=none -o "khronos/$e.md"
done
[ -f khronos/cl_intel_media_block_io.txt ] || curl -sfL -o khronos/cl_intel_media_block_io.txt https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_media_block_io.txt
# IGA + vISA documentation from the compiler, matching the installed intel-igc-core (1.0.17537.24)
D=igc-1.0.17537.24
if [ ! -d "$D" ]; then
  t=$(mktemp -d)
  git clone -q --filter=blob:none --sparse --depth 1 --branch igc-1.0.17537.24 https://github.com/intel/intel-graphics-compiler.git "$t"
  (cd "$t" && git sparse-checkout set visa/iga documentation)
  mkdir -p "$D"; cp -r "$t/visa/iga/IGALibrary" "$D/"; cp -r "$t/documentation/visa" "$D/visa-docs"; cp "$t/LICENSE.md" "$D/"
  (cd "$t" && git log -1 --format='%H %cd') > "$D/VERSION.txt"; rm -rf "$t"
fi
# Mesa (brw compiler): validator, bank conflicts, latencies, assembler
M=mesa-25.2.8
if [ ! -d "$M" ]; then
  t=$(mktemp -d)
  git clone -q --filter=blob:none --sparse --depth 1 --branch mesa-25.2.8 https://gitlab.freedesktop.org/mesa/mesa.git "$t"
  (cd "$t" && git sparse-checkout set src/intel/compiler)
  mkdir -p "$M"
  for f in brw_asm.c brw_asm.h brw_asm_internal.h brw_asm_tool.c brw_disasm.c brw_disasm.h brw_eu_compact.c brw_eu_defines.h \
           brw_eu_emit.c brw_eu_inst.h brw_eu_validate.c brw_gram.y brw_lex.l brw_opt_bank_conflicts.cpp brw_reg.h \
           brw_schedule_instructions.cpp brw_validate.cpp test_eu_validate.cpp brw_eu.h brw_eu.c; do cp "$t/src/intel/compiler/$f" "$M/"; done
  (cd "$t" && git log -1 --format='mesa-25.2.8 %H %cd') > "$M/VERSION.txt"; rm -rf "$t"
fi
# Mesa genxml: bit-level definitions of Gen9 commands and structures (gen90.xml imports gen80 -> gen75 -> ... -> gen40)
if [ ! -f "$M/genxml/gen90.xml" ]; then
  t=$(mktemp -d)
  git clone -q --filter=blob:none --sparse --depth 1 --branch mesa-25.2.8 https://gitlab.freedesktop.org/mesa/mesa.git "$t"
  (cd "$t" && git sparse-checkout set src/intel/genxml)
  mkdir -p "$M/genxml"
  for f in gen40.xml gen45.xml gen50.xml gen60.xml gen70.xml gen75.xml gen80.xml gen90.xml gen_pack_header.py intel_genxml.py util.py README; do
    cp "$t/src/intel/genxml/$f" "$M/genxml/"; done
  rm -rf "$t"
fi
# Intel Compute Runtime (NEO): how it programs the Gen9 compute pipeline (docs/no-neo.md), the files that were consulted
N=neo-24.35.30872.36
if [ ! -d "$N" ]; then
  t=$(mktemp -d)
  git clone -q --filter=blob:none --sparse --depth 1 --branch 24.35.30872.36 https://github.com/intel/compute-runtime.git "$t"
  (cd "$t" && git sparse-checkout set shared/source opencl/source)
  for f in shared/source/gen9/preamble_gen9.cpp shared/source/gen9/hw_cmds_base.h shared/source/gen9/command_encoder_gen9.cpp \
           shared/source/gen9/reg_configs.h shared/source/gen9/gfx_core_helper_gen9.cpp \
           shared/source/command_container/command_encoder.inl shared/source/command_container/command_encoder_bdw_and_later.inl \
           shared/source/generated/gen9/hw_cmds_generated_gen9.inl \
           shared/source/helpers/gfx_core_helper_bdw_and_later.inl shared/source/helpers/per_thread_data.h shared/source/helpers/l3_range.h \
           shared/source/helpers/gfx_core_helper_base.inl shared/source/helpers/state_base_address_bdw.inl \
           shared/source/helpers/local_id_gen_special.inl shared/source/helpers/preamble_bdw_and_later.inl shared/source/helpers/local_id_gen.h \
           shared/source/helpers/local_id_gen.inl shared/source/helpers/state_base_address_bdw_and_later.inl \
           opencl/source/gen9/gpgpu_walker_gen9.cpp opencl/source/command_queue/gpgpu_walker_bdw_and_later.inl \
           opencl/source/command_queue/gpgpu_walker_base.inl opencl/source/command_queue/gpgpu_walker.h \
           opencl/source/helpers/hardware_commands_helper_base.inl opencl/source/helpers/hardware_commands_helper_bdw_and_later.inl \
           opencl/source/helpers/hardware_commands_helper.h; do
    mkdir -p "$N/$(dirname "$f")"; cp "$t/$f" "$N/$f"
  done
  cp "$t/LICENSE.md" "$N/"; (cd "$t" && git log -1 --format='NEO 24.35.30872.36 %H %cd') > "$N/VERSION.txt"; rm -rf "$t"
fi
# NEO Gen8 (Broadwell) files: how NEO programmed the Gen8 compute pipeline (docs/gen8.md); NEO never supported Cherryview
if [ ! -f "$N/shared/source/gen8/preamble_gen8.cpp" ]; then
  t=$(mktemp -d)
  git clone -q --filter=blob:none --sparse --depth 1 --branch 24.35.30872.36 https://github.com/intel/compute-runtime.git "$t"
  (cd "$t" && git sparse-checkout set shared/source/gen8 shared/source/generated/gen8 opencl/source/gen8)
  for d in shared/source/gen8 shared/source/generated/gen8 opencl/source/gen8; do mkdir -p "$N/$d"; cp -r "$t/$d/." "$N/$d/"; done
  rm -rf "$t"
fi
# Mesa ISL (MOCS per generation), the Gen8 compiler "elk" (Cherryview EU rules) and device info (docs/gen8.md)
for f in isl/isl.c compiler/elk/elk_eu_validate.c compiler/elk/elk_fs.cpp dev/intel_device_info.c dev/intel_device_info.h; do
  d="$M/$(dirname "$f" | sed 's|^compiler/||')"; mkdir -p "$d"
  [ -f "$d/$(basename "$f")" ] || curl -sfL -o "$d/$(basename "$f")" "https://gitlab.freedesktop.org/mesa/mesa/-/raw/mesa-25.2.8/src/intel/$f"
done
# Patch-token binary format (what ocloc emits for -device bdw): token definitions from IGC, decoder from NEO (docs/gen8.md)
P=patchtokens
mkdir -p "$P"
for f in patch_list.h patch_shared.h patch_g7.h patch_g8.h patch_g9.h; do
  [ -f "$P/$f" ] || curl -sfL --max-time 120 -o "$P/$f" "https://raw.githubusercontent.com/intel/intel-graphics-compiler/igc-1.0.17537.24/IGC/AdaptorOCL/ocl_igc_shared/executable_format/$f"
done
for f in device_binary_format/patchtokens_decoder.cpp device_binary_format/patchtokens_decoder.h kernel/kernel_descriptor_from_patchtokens.cpp; do
  [ -f "$P/$(basename $f)" ] || curl -sfL --max-time 120 -o "$P/$(basename $f)" "https://raw.githubusercontent.com/intel/compute-runtime/24.35.30872.36/shared/source/$f"
done
# Beignet 1.3.2: Intel's old OpenCL stack, the only one that supported Cherryview (L3 values, Gen8 GPGPU state)
B=beignet-1.3.2
mkdir -p "$B"
[ -f "$B/intel_gpgpu.c" ] || curl -sfL --max-time 120 -o "$B/intel_gpgpu.c" "https://raw.githubusercontent.com/intel/beignet/Release_v1.3.2/src/intel/intel_gpgpu.c"
# intel-gpu-tools: gpgpu_fill (the smallest example of a Gen9 compute kernel through i915, without NEO)
I=igt-2.6
if [ ! -d "$I" ]; then
  t=$(mktemp -d)
  git clone -q --filter=blob:none --sparse --depth 1 --branch v2.6 https://gitlab.freedesktop.org/drm/igt-gpu-tools.git "$t"
  (cd "$t" && git sparse-checkout set lib)
  mkdir -p "$I"
  for f in gpgpu_fill.c gpgpu_fill.h gpu_cmds.c gpu_cmds.h gen9_render.h gen8_media.h gen7_media.h intel_batchbuffer.h intel_reg.h; do cp "$t/lib/$f" "$I/"; done
  cp "$t/COPYING" "$I/"; (cd "$t" && git log -1 --format='igt v2.6 %H %cd') > "$I/VERSION.txt"; rm -rf "$t"
fi
# i915 from the Linux kernel matching the running one (7.0): userptr, page pinning, execbuffer (docs/no-neo.md 6g)
L=linux-i915
mkdir -p "$L"
# Gen8 (docs/gen8.md): MOCS, heartbeat/preemption, CPU cache domains without LLC, 32-bit PPGTT, CHV device info, workarounds
for f in gem/i915_gem_userptr.c gem/i915_gem_pages.c gem/i915_gem_object.c gem/i915_gem_execbuffer.c \
         gem/i915_gem_domain.c gem/i915_gem_mman.c gt/intel_mocs.c gt/intel_engine_heartbeat.c gt/intel_execlists_submission.c \
         gt/gen8_ppgtt.c gt/gen8_engine_cs.c gt/intel_workarounds.c i915_pci.c \
         gem/i915_gem_shmem.c gem/i915_gem_context.c gem/i915_gem_context_types.h i915_vma.c i915_cmd_parser.c i915_request.c \
         gt/intel_gtt.c gt/intel_gtt.h gt/intel_reset.c gt/intel_gt_clock_utils.c Kconfig.profile; do
  [ -f "$L/$(basename $f)" ] || curl -sfL --max-time 120 -o "$L/$(basename $f)" "https://raw.githubusercontent.com/torvalds/linux/v7.0/drivers/gpu/drm/i915/$f"
done
echo done
