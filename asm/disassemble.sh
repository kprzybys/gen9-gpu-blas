#!/bin/sh
# Krok 1 obiegu: OpenCL C -> binarka dla UHD 620 (whl) -> katalog z edytowalnym kodem maszynowym.
# Użycie: asm/disassemble.sh <kernel.cl> <katalog>   ->  <katalog>/.text.<kernel>.asm do edycji
# Uwaga: musi być -device whl (nie cfl). Binarka dla cfl jest odrzucana (CL_INVALID_BINARY),
# a z dołączonym SPIR-V sterownik po cichu kompiluje kernel od nowa i ignoruje nasz kod maszynowy.
set -e
src=$1; dir=$2; tmp=$(mktemp -d)
ocloc compile -q -file "$src" -device whl -output "$tmp/k" >/dev/null
rm -rf "$dir"; ocloc disasm -file "$tmp"/k_*.bin -dump "$dir" -device whl >/dev/null 2>&1
rm -f "$dir/.spv"; sed -i '/^\.spv,/d' "$dir/sections.txt"   # bez SPIR-V: sterownik musi użyć naszego ISA
rm -rf "$tmp"; ls "$dir"/.text.*.asm
