#!/bin/sh
# Pełna seria porównania poziomu 3 (gen9blas, CLBlast, OpenBLAS). Uruchom z katalogu repo: sh bench/compare/l3.sh > results/...
# Kombinacje bez implementacji (OpenBLAS FP16, CLBlast HTRSM) są pomijane po komunikacie programu.
L3=bench/compare/l3
for p in s h d; do
  [ $p = d ] && ns="1024 2048 4096" || ns="1024 2048 3000 4096"
  for op in gemm symm syrk syr2k trmm trsm; do
    for n in $ns; do
      for b in gen9blas clblast openblas; do $L3 $b $op $p $n 2>&1 | grep -v "nie ma"; done
    done
  done
done
echo "== CLBlast z czekaniem przez uśpienie (CZEKAJ=uspienie) i tryb host (TRYB=host: dane w zwykłej pamięci)"
for p in s h; do for op in gemm trsm; do for n in 2048 4096; do
  CZEKAJ=uspienie $L3 clblast $op $p $n 2>&1 | grep -v "nie ma"
  TRYB=host $L3 gen9blas $op $p $n; TRYB=host $L3 clblast $op $p $n 2>&1 | grep -v "nie ma"
done; done; done
