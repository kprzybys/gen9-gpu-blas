#!/bin/sh
# Krok 2 obiegu: katalog z (edytowanym) kodem maszynowym -> binarka do clCreateProgramWithBinary.
# Użycie: asm/assemble.sh <katalog> <wynik.bin>     (ocloc asm zapisuje wynik w katalogu dump, stąd mv)
set -e
dir=$1; out=$2; name=$(basename "$out")
ocloc asm -file "$name" -dump "$dir" -device whl >/dev/null
mv "$dir/$name" "$out"; ls -l "$out"
