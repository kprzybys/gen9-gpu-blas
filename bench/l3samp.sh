#!/bin/bash
# Texture sampler vs L3 data port (bench/l3samp.cl, bench/l3port.cl). Results: results/l3samp-2026-11-13.txt
set -u; cd "$(dirname "$0")"
[ -x clbench ] || gcc -O2 -Wall -o clbench clbench.c -lOpenCL
N=4096; G=16384
r() { nice -n 19 ./clbench "$@" | sed 's/global=16384 *//; s/ sg=8 *//; s/n=4096 *//'; }
echo "# $(date -Is) $(hostname) GPU max $(cat /sys/class/drm/card*/gt_max_freq_mhz | head -1) MHz"
for L in 64 128; do
  printf 'port   L%-3s PF4      ' $L; BUF=1048576 r l3port.cl k_port $G $L $N 16 "-DMADS=0 -DPF=4"
  for PF in 2 4; do
    printf 'samp   L%-3s PF%s 64K  ' $L $PF; IMG4=1 IMG=64x64 r l3samp.cl k_samp $G $L $N 16 "-DPF=$PF -DW=64 -DH=64"
    printf 'both   L%-3s PF%s 64K  ' $L $PF; BUF=1048576 IMG4=1 IMG=64x64 r l3samp.cl k_both $G $L $N 32 "-DPF=$PF -DW=64 -DH=64"
  done
done
printf 'samp   L64  PF4 1M   '; IMG4=1 IMG=256x256 r l3samp.cl k_samp $G 64 $N 16 "-DPF=4 -DW=256 -DH=256"
printf 'both   L64  PF4 1M   '; BUF=1048576 IMG4=1 IMG=256x256 r l3samp.cl k_both $G 64 $N 32 "-DPF=4 -DW=256 -DH=256 -DMASK=0xFFFFF"

# unbalanced mixes (RP port blocks per sampler read) and image sizes
for RP in 1 2 3 4 6; do
  printf 'both RP%s L64 PF2 ' $RP; BUF=1048576 IMG4=1 IMG=64x64 r l3samp.cl k_both $G 64 $N $((16 * (RP + 1))) "-DPF=2 -DRP=$RP -DW=64 -DH=64"
done
printf 'port alone        '; BUF=1048576 r l3port.cl k_port $G 64 $N 16 "-DMADS=0 -DPF=4"
printf 'samp alone        '; IMG4=1 IMG=64x64 r l3samp.cl k_samp $G 64 $N 16 "-DPF=4 -DW=64 -DH=64"
for S in "128 128 0x3FFFF 256K" "256 256 0xFFFFF 1M"; do set -- $S
  printf 'samp alone %-4s     ' $4; IMG4=1 IMG=${1}x${2} r l3samp.cl k_samp $G 64 $N 16 "-DPF=4 -DW=$1 -DH=$2"
  printf 'port alone %-4s     ' $4; BUF=1048576 r l3port.cl k_port $G 64 $N 16 "-DMADS=0 -DPF=4 -DMASK=$3"
  for RP in 1 2 3; do
    printf 'both RP%s %-4s      ' $RP $4; BUF=1048576 IMG4=1 IMG=${1}x${2} r l3samp.cl k_both $G 64 $N $((16 * (RP + 1))) "-DPF=2 -DRP=$RP -DW=$1 -DH=$2 -DMASK=$3"
  done
done
