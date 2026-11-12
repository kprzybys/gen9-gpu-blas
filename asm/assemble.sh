#!/bin/sh
# Step 2 of the workflow: directory with (edited) machine code -> binary for clCreateProgramWithBinary.
# Usage: asm/assemble.sh <dir> <out.bin>     (ocloc asm writes the result into the dump dir, hence the mv)
set -e
dir=$1; out=$2; name=$(basename "$out")
ocloc asm -file "$name" -dump "$dir" -device whl >/dev/null
mv "$dir/$name" "$out"; ls -l "$out"
