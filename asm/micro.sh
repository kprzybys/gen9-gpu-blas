#!/bin/sh
# Runs all microbenchmarks from asm/micro.py. Result: results/micro-<date>.txt
cd -P "$(dirname "$0")"
B=$(mktemp -d)
{
echo "# $(date -Is) asm microbenchmarks (Gop/s = GFLOPS for FP32 mad; theoretical peak 422.4)"
python3 -I micro.py --list | while read t desc; do
  n=$(python3 -I micro.py "$t" "$B/$t.bin") || { echo "$t ERROR"; continue; }
  printf '%-22s %s  | %s\n' "$t" "$(../bench/clbench "$B/$t.bin" k_fma 131072 256 2000 "$n" | grep -oE '[0-9.]+ Gop/s')" "$desc"
done
} | tee "../results/micro-$(date +%F).txt"
rm -rf "$B"
