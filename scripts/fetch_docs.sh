#!/bin/sh
# Fetches the reference documentation into third_party/ (not in git; state as of 2026-11-11). Files already present are kept.
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
S
}
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
for f in gem/i915_gem_userptr.c gem/i915_gem_pages.c gem/i915_gem_object.c gem/i915_gem_execbuffer.c; do
  [ -f "$L/$(basename $f)" ] || curl -sfL -o "$L/$(basename $f)" "https://raw.githubusercontent.com/torvalds/linux/v7.0/drivers/gpu/drm/i915/$f"
done
echo done
