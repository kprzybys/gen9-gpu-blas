#!/bin/sh
# Throughput and latency of arithmetic instructions. Result: results/arith-throughput-latency-<date>.txt
cd -P "$(dirname "$0")"
[ -x clbench ] || gcc -O2 -Wall -o clbench clbench.c -lOpenCL
B="./clbench peak.cl"; G=131072
run() { t=$1; shift; printf '%-7s ' "$t"; $B "$@"; }
{
echo "# throughput (Gop/s; FMA = 2 op)"
for t in float half double; do n=1000; [ $t = double ] && n=250
  run $t k_fma  $G 256 $n 256 "-DT=$t"; run $t k_fmav $G 256 $n 256 "-DT=$t"
  run $t k_mad  $G 256 $n 256 "-DT=$t"
  run $t k_mul  $G 256 $n 128 "-DT=$t"; run $t k_mulv $G 256 $n 128 "-DT=$t"
  run $t k_add  $G 256 $n 128 "-DT=$t"; run $t k_addv $G 256 $n 128 "-DT=$t"; done
run half k_mad $G 256 1000 256 "-DT=half -cl-fast-relaxed-math"
for t in int short long; do n=1000; [ $t = long ] && n=100
  run $t k_mul   $G 256 $n 128 "-DT=$t -DINT"; run $t k_mulv  $G 256 $n 128 "-DT=$t -DINT"
  run $t k_addx  $G 256 $n 256 "-DT=$t -DINT"; run $t k_addxv $G 256 $n 256 "-DT=$t -DINT"; done
echo "# latency: 1 SIMD32 thread, 1 chain; clocks per step = time*f/(n*16)"
for s in "float k_fma_lat" "half k_fma_lat" "double k_fma_lat" "float k_add_lat" "float k_mul_lat" "int k_mul_lat -DINT" "short k_mul_lat -DINT" "int k_addx_lat -DINT"; do
  set -- $s; run $1 $2 32 32 20000 16 "-DT=$1 $3"; done
} | tee "../results/arith-throughput-latency-$(date +%F).txt"
