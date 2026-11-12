#!/bin/sh
# userptr vs copy: 3 rounds alternating (separate processes), result in results/userptr-<date>.txt
cd "$(dirname "$0")"
for r in 1 2 3; do
    echo "== round $r: userptr"; OPENBLAS_NUM_THREADS=1 python3 userptr.py
    echo "== round $r: copy";   GEN9BLAS_NO_USERPTR=1 OPENBLAS_NUM_THREADS=1 python3 userptr.py
done
