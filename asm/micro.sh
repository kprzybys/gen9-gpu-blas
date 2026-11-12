#!/bin/sh
# Uruchamia wszystkie mikrotesty z asm/micro.py. Wynik: results/mikro-<data>.txt
cd -P "$(dirname "$0")"
B=$(mktemp -d)
{
echo "# $(date -Is) mikrotesty asm (Gop/s = GFLOPS dla mad FP32; szczyt teoretyczny 422,4)"
python3 -I micro.py --lista | while read t opis; do
  n=$(python3 -I micro.py "$t" "$B/$t.bin") || { echo "$t BLAD"; continue; }
  printf '%-22s %s  | %s\n' "$t" "$(../bench/clbench "$B/$t.bin" k_fma 131072 256 2000 "$n" | grep -oE '[0-9.]+ Gop/s')" "$opis"
done
} | tee "../results/mikro-$(date +%F).txt"
rm -rf "$B"
