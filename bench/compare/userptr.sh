#!/bin/sh
# userptr wobec kopii: 3 rundy na przemian (osobne procesy), wynik w results/userptr-<data>.txt
cd "$(dirname "$0")"
for r in 1 2 3; do
    echo "== runda $r: userptr"; OPENBLAS_NUM_THREADS=1 python3 userptr.py
    echo "== runda $r: kopia";   GEN9BLAS_NO_USERPTR=1 OPENBLAS_NUM_THREADS=1 python3 userptr.py
done
