#!/bin/sh
# Energy per call (level 3): each case runs 6 s in continuous mode under bench/power.sh.
# Energy = Package power × call time. Run from the repo root.
L3=bench/compare/l3
while read b op p n; do
  sh bench/power.sh $L3 $b $op $p $n 6
done <<'L'
gen9blas gemm s 4096
clblast gemm s 4096
openblas gemm s 4096
gen9blas gemm s 1024
clblast gemm s 1024
openblas gemm s 1024
gen9blas gemm h 4096
clblast gemm h 4096
gen9blas trsm s 4096
openblas trsm s 4096
gen9blas gemm d 2048
clblast gemm d 2048
openblas gemm d 2048
L
