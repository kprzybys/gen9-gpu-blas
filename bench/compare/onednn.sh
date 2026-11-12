#!/bin/sh
# GEMM: oneDNN v3.6 (GPU, OpenCL, jit:gemm z nGEN -- ostatnia wersja z katalogiem jąder dla Gen9) wobec libgen9blas.
# oneDNN zbudowany poza repo: ~/src/onednn-3.6 (cmake -DDNNL_CPU_RUNTIME=NONE -DDNNL_GPU_RUNTIME=OCL, cel benchdnn).
# Pomiar: benchdnn --mode=P (dane już na GPU, czas min z wielu powtórzeń) -- porównywać z gen9blas w trybie gpu (USM).
# Uruchom z katalogu repo: sh bench/compare/onednn.sh > results/...
B=${BENCHDNN:-$HOME/src/onednn-3.6/build/tests/benchdnn/benchdnn}
L3=bench/compare/l3
for n in 1024 2048 3000 4096; do
  for dt in f32 f16; do
    $B --matmul --engine=gpu --mode=P --dt=$dt --perf-template="onednn %prb% min %-time% ms %-Gflops% GFLOPS" ${n}x${n}:${n}x${n} 2>&1 | grep "^onednn"
  done
  $L3 gen9blas gemm s $n; $L3 gen9blas gemm h $n; $L3 gen9blas gemm x $n
done
echo "== poprawność oneDNN (benchdnn --mode=C)"
for dt in f32 f16; do $B --matmul --engine=gpu --mode=C --dt=$dt 1024x1024:1024x1024 2>&1 | grep -E "tests:|FAIL"; done
