#!/bin/bash
# L3 / data-port ceiling sweep (bench/l3port.cl): Gop/s = GB/s of block reads. Each launch <= ~25 ms; nice 19, ionice idle.
# Results: results/l3port-2026-11-13.txt
set -u; cd "$(dirname "$0")"; [ -f ~/neo/env.sh ] && . ~/neo/env.sh   # tc: NEO legacy outside the system
[ -x clbench ] || gcc -O2 -Wall -o clbench clbench.c -lOpenCL
N=${N:-4096}; G=16384; export BUF=$((1 << 20))
run() { nice -n 19 ionice -c3 ./clbench l3port.cl k_port $G "$@"; }
echo "# $(date -Is) tc HD 530, k_port, global $G (2048 subgroups), n $N blocks/subgroup, Gop/s = GB/s of block reads"
for L in 64 128; do for PF in 2 4; do for M in 0 8 12 16 24 32; do
  printf 'local %-3s PF %s MADS %-2s ' $L $PF $M
  run $L $N 16 "-DMADS=$M -DPF=$PF" | sed 's/^k_port *//'
done; done; done
echo "# region 1 MB (beyond L3, from LLC) and SHARE 8, local 64, PF 2"
for M in 0 16; do printf 'MASK 1M  MADS %-2s ' $M; run 64 $N 16 "-DMADS=$M -DPF=2 -DMASK=0xFFFFF" | sed 's/^k_port *//'; done
for M in 0 16; do printf 'SHARE 8  MADS %-2s ' $M; run 64 $N 16 "-DMADS=$M -DPF=2 -DSHARE=8" | sed 's/^k_port *//'; done
