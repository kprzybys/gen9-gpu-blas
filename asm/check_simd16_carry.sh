#!/bin/sh
# Is the accurate HGEMM with the SIMD16 mixed-mode carry (bench/gen_hd.bin, GEMM_CARRY=simd16) bit-identical to the
# PRM-safe form with two SIMD8 adds (GEMM_CARRY=simd8) on THIS GPU? Run on every new GPU before trusting gen_hd.bin.
# Builds both variants and a library from each in a temporary copy of the tree (the repo is not touched), then runs the
# same GEMM inputs (edge shapes, FP16 subnormals, large values, inf/NaN) through both and compares the output bytes.
# Usage: asm/check_simd16_carry.sh            (needs ocloc, gcc; uses the GPU for ~1 min)
set -e
repo=$(cd -P "$(dirname "$0")/.." && pwd); tmp=$(mktemp -d)
trap 'rm -r "$tmp"' EXIT
for v in simd16 simd8; do
  mkdir -p "$tmp/$v"; cp -r "$repo/lib" "$repo/bench" "$tmp/$v/"
  (cd "$repo/asm" && GEMM_CARRY=$v python3 -I gemm_gen.py hd loop "$tmp/$v/bench/gen_hd.bin" >/dev/null)
  (cd "$tmp/$v/lib" && python3 gen_kernels.py ../bench .kernels > kernels.c && make libgen9blas.so >/dev/null 2>&1)
  (cd "$tmp/$v" && python3 "$repo/asm/carry_inputs.py" "$tmp/$v.out")
done
if cmp -s "$tmp/simd16.out" "$tmp/simd8.out"; then echo "OK: SIMD16 carry bit-identical to two SIMD8 adds on this GPU"
else echo "DIFFERENT: use GEMM_CARRY=simd8 for gen_hd.bin on this GPU"; exit 1; fi
