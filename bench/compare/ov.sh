#!/bin/sh
# GEMM: OpenVINO (GPU) vs libgen9blas. Run from the repo root: sh bench/compare/ov.sh > results/...
# openvino+W: B as a constant in the graph (network weights, FullyConnected path) -- the fastest OpenVINO path.
# OpenVINO receives NumPy arrays (host memory), so gen9blas is measured in both modes: gpu (USM) and host (plain memory).
L3=bench/compare/l3
for n in 1024 2048 3000 4096; do
  python3 bench/compare/ov_gemm.py f32 $n; OV_CONST=1 python3 bench/compare/ov_gemm.py f32 $n; $L3 gen9blas gemm s $n; MODE=host $L3 gen9blas gemm s $n
  python3 bench/compare/ov_gemm.py f16 $n; OV_CONST=1 python3 bench/compare/ov_gemm.py f16 $n; $L3 gen9blas gemm h $n; $L3 gen9blas gemm x $n; MODE=host $L3 gen9blas gemm x $n
done
echo "== energy (bench/power.sh, 6 s continuous)"
for c in "python3 bench/compare/ov_gemm.py f16 4096 6" "env OV_CONST=1 python3 bench/compare/ov_gemm.py f16 4096 6" \
         "$L3 gen9blas gemm x 4096 6" "$L3 gen9blas gemm h 4096 6" "env OV_CONST=1 python3 bench/compare/ov_gemm.py f32 4096 6" "$L3 gen9blas gemm s 4096 6"; do
  sh bench/power.sh $c
done
