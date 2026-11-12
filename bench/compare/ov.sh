#!/bin/sh
# GEMM: OpenVINO (GPU) wobec libgen9blas. Uruchom z katalogu repo: sh bench/compare/ov.sh > results/...
# openvino+W: B jako stała w grafie (wagi sieci, ścieżka FullyConnected) -- najszybsza ścieżka OpenVINO.
# OpenVINO dostaje tablice NumPy (pamięć hosta), więc gen9blas mierzony w obu trybach: gpu (USM) i host (zwykła pamięć).
L3=bench/compare/l3
for n in 1024 2048 3000 4096; do
  python3 bench/compare/ov_gemm.py f32 $n; OV_STALA=1 python3 bench/compare/ov_gemm.py f32 $n; $L3 gen9blas gemm s $n; TRYB=host $L3 gen9blas gemm s $n
  python3 bench/compare/ov_gemm.py f16 $n; OV_STALA=1 python3 bench/compare/ov_gemm.py f16 $n; $L3 gen9blas gemm h $n; $L3 gen9blas gemm x $n; TRYB=host $L3 gen9blas gemm x $n
done
echo "== energia (bench/power.sh, 6 s ciągle)"
for c in "python3 bench/compare/ov_gemm.py f16 4096 6" "env OV_STALA=1 python3 bench/compare/ov_gemm.py f16 4096 6" \
         "$L3 gen9blas gemm x 4096 6" "$L3 gen9blas gemm h 4096 6" "env OV_STALA=1 python3 bench/compare/ov_gemm.py f32 4096 6" "$L3 gen9blas gemm s 4096 6"; do
  sh bench/power.sh $c
done
