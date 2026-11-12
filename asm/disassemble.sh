#!/bin/sh
# Step 1 of the workflow: OpenCL C -> binary for UHD 620 (whl) -> directory with editable machine code.
# Usage: asm/disassemble.sh <kernel.cl> <dir>   ->  <dir>/.text.<kernel>.asm to edit
# Note: must be -device whl (not cfl). A binary built for cfl is rejected (CL_INVALID_BINARY),
# and with SPIR-V attached the driver silently recompiles the kernel and ignores our machine code.
set -e
src=$1; dir=$2; tmp=$(mktemp -d)
ocloc compile -q -file "$src" -device whl -output "$tmp/k" >/dev/null
rm -rf "$dir"; ocloc disasm -file "$tmp"/k_*.bin -dump "$dir" -device whl >/dev/null 2>&1
rm -f "$dir/.spv"; sed -i '/^\.spv,/d' "$dir/sections.txt"   # no SPIR-V: the driver must use our ISA
rm -rf "$tmp"; ls "$dir"/.text.*.asm
