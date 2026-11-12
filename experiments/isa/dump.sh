#!/bin/sh
# Compiles a .cl file for the UHD 620 (cfl) and writes the machine code of each kernel to experiments/isa/out/<file>/<kernel>.asm
set -e
cd -P "$(dirname "$0")"
src=${1:-map.cl}; opts=${2:-}
name=$(basename "$src" .cl); out=out/$name; tmp=$(mktemp -d)
rm -rf "$out"; mkdir -p "$out"
ocloc compile -q -file "$src" -device cfl -output "$tmp/k" -options "$opts" >/dev/null
ocloc disasm -file "$tmp/k_cfl.bin" -dump "$tmp/dis" -device cfl >/dev/null 2>&1 || true
for f in "$tmp"/dis/.text.*.asm; do k=${f##*/.text.}; sed '/^\s*illegal/d' "$f" > "$out/$k"; done
rm -rf "$tmp"; ls "$out" | wc -l
