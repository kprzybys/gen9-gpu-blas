#!/bin/sh
# Odtwarza pobranie dokumentacji (stan 2026-11-11). PDF-y są w git LFS, ten skrypt służy do weryfikacji i aktualizacji.
# Strona indeksu Intela blokuje automaty (403), ale pliki z cdrdv2.intel.com pobierają się bezpośrednio.
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
for p in pdf/*.pdf; do b=$(basename "$p" .pdf); [ -f "txt/$b.txt" ] || pdftotext -layout "$p" "txt/$b.txt"; done
for e in cl_intel_subgroups cl_intel_subgroups_short cl_intel_subgroups_char cl_intel_required_subgroup_size cl_intel_spirv_subgroups; do
  [ -f "khronos/$e.md" ] || curl -sfL "https://registry.khronos.org/OpenCL/extensions/intel/$e.html" | pandoc -f html -t gfm-raw_html --wrap=none -o "khronos/$e.md"
done
[ -f khronos/cl_intel_media_block_io.txt ] || curl -sfL -o khronos/cl_intel_media_block_io.txt https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_media_block_io.txt
# IGA + dokumentacja vISA z kompilatora w wersji zgodnej z zainstalowanym intel-igc-core (1.0.17537.24)
D=igc-1.0.17537.24
if [ ! -d "$D" ]; then
  t=$(mktemp -d)
  git clone -q --filter=blob:none --sparse --depth 1 --branch igc-1.0.17537.24 https://github.com/intel/intel-graphics-compiler.git "$t"
  (cd "$t" && git sparse-checkout set visa/iga documentation)
  mkdir -p "$D"; cp -r "$t/visa/iga/IGALibrary" "$D/"; cp -r "$t/documentation/visa" "$D/visa-docs"; cp "$t/LICENSE.md" "$D/"
  (cd "$t" && git log -1 --format='%H %cd') > "$D/WERSJA.txt"; rm -rf "$t"
fi
# Mesa (kompilator brw): walidator, konflikty banków, opóźnienia, asembler
M=mesa-25.2.8
if [ ! -d "$M" ]; then
  t=$(mktemp -d)
  git clone -q --filter=blob:none --sparse --depth 1 --branch mesa-25.2.8 https://gitlab.freedesktop.org/mesa/mesa.git "$t"
  (cd "$t" && git sparse-checkout set src/intel/compiler)
  mkdir -p "$M"
  for f in brw_asm.c brw_asm.h brw_asm_internal.h brw_asm_tool.c brw_disasm.c brw_disasm.h brw_eu_compact.c brw_eu_defines.h \
           brw_eu_emit.c brw_eu_inst.h brw_eu_validate.c brw_gram.y brw_lex.l brw_opt_bank_conflicts.cpp brw_reg.h \
           brw_schedule_instructions.cpp brw_validate.cpp test_eu_validate.cpp brw_eu.h brw_eu.c; do cp "$t/src/intel/compiler/$f" "$M/"; done
  (cd "$t" && git log -1 --format='mesa-25.2.8 %H %cd') > "$M/WERSJA.txt"; rm -rf "$t"
fi
# Mesa genxml: definicje komend i struktur Gen9 bit po bicie (gen90.xml importuje gen80 -> gen75 -> ... -> gen40)
if [ ! -f "$M/genxml/gen90.xml" ]; then
  t=$(mktemp -d)
  git clone -q --filter=blob:none --sparse --depth 1 --branch mesa-25.2.8 https://gitlab.freedesktop.org/mesa/mesa.git "$t"
  (cd "$t" && git sparse-checkout set src/intel/genxml)
  mkdir -p "$M/genxml"
  for f in gen40.xml gen45.xml gen50.xml gen60.xml gen70.xml gen75.xml gen80.xml gen90.xml gen_pack_header.py intel_genxml.py util.py README; do
    cp "$t/src/intel/genxml/$f" "$M/genxml/"; done
  rm -rf "$t"
fi
# intel-gpu-tools: gpgpu_fill (najmniejszy przykład jądra obliczeniowego Gen9 przez i915, bez NEO)
I=igt-2.6
if [ ! -d "$I" ]; then
  t=$(mktemp -d)
  git clone -q --filter=blob:none --sparse --depth 1 --branch v2.6 https://gitlab.freedesktop.org/drm/igt-gpu-tools.git "$t"
  (cd "$t" && git sparse-checkout set lib)
  mkdir -p "$I"
  for f in gpgpu_fill.c gpgpu_fill.h gpu_cmds.c gpu_cmds.h gen9_render.h gen8_media.h gen7_media.h intel_batchbuffer.h intel_reg.h; do cp "$t/lib/$f" "$I/"; done
  cp "$t/COPYING" "$I/"; (cd "$t" && git log -1 --format='igt v2.6 %H %cd') > "$I/WERSJA.txt"; rm -rf "$t"
fi
echo gotowe
# i915 z jądra Linuksa zgodnego z uruchomionym (7.0): userptr, przypinanie stron, execbuffer (BEZNEO.md 6g)
L=linux-i915
mkdir -p "$L"
for f in gem/i915_gem_userptr.c gem/i915_gem_pages.c gem/i915_gem_object.c gem/i915_gem_execbuffer.c; do
  [ -f "$L/$(basename $f)" ] || curl -sfL -o "$L/$(basename $f)" "https://raw.githubusercontent.com/torvalds/linux/v7.0/drivers/gpu/drm/i915/$f"
done
