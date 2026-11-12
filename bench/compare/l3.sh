#!/bin/sh
# Full level-3 comparison series (gen9blas, CLBlast, OpenBLAS). Run from the repo root: sh bench/compare/l3.sh > results/...
# Combinations without an implementation (OpenBLAS FP16, CLBlast HTRSM) are skipped after the program's message.
L3=bench/compare/l3
for p in s h d; do
  [ $p = d ] && ns="1024 2048 4096" || ns="1024 2048 3000 4096"
  for op in gemm symm syrk syr2k trmm trsm; do
    for n in $ns; do
      for b in gen9blas clblast openblas; do $L3 $b $op $p $n 2>&1 | grep -v "has no"; done
    done
  done
done
echo "== CLBlast waiting by sleeping (WAIT=sleep) and host mode (MODE=host: data in plain memory)"
for p in s h; do for op in gemm trsm; do for n in 2048 4096; do
  WAIT=sleep $L3 clblast $op $p $n 2>&1 | grep -v "has no"
  MODE=host $L3 gen9blas $op $p $n; MODE=host $L3 clblast $op $p $n 2>&1 | grep -v "has no"
done; done; done
