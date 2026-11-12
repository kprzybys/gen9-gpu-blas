#!/bin/sh
# Step 0 (GEMM): reference points of CLBlast (GPU) and OpenBLAS (CPU) for various N. Result: results/gemm0-<date>.txt
cd -P "$(dirname "$0")"
[ -x gemm_clblast ] || gcc -O2 -Wall -o gemm_clblast gemm_clblast.c -lclblast -lOpenCL -lm
{
echo "# $(date -Is)  GPU: $(cat /sys/class/drm/card1/gt_max_freq_mhz) MHz max, power AC=$(cat /sys/class/power_supply/AC*/online)"
for N in 128 256 512 1024 2048 4096; do
  ./gemm_clblast s $N; ./gemm_clblast h $N; ./gemm_clblast d $N
  python3 gemm_cpu.py f32 $N; python3 gemm_cpu.py f64 $N
done
} | tee "../results/gemm0-$(date +%F).txt"
