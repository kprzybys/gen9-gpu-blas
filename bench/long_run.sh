#!/bin/sh
# Test długotrwały (~90 s): HGEMM asm na GPU, potem OpenBLAS f32 na CPU. Co 5 s: taktowanie i moc (intel_gpu_top).
# Wynik: results/dlugi-<data>.txt
cd -P "$(dirname "$0")"
OUT=../results/dlugi-$(date +%F).txt
probki() {  # $1 = plik JSON intel_gpu_top
  python3 -I - "$1" <<'P'
import json, re, sys
t = open(sys.argv[1]).read().strip()
t = "[" + re.sub(r"\}\s*,?\s*\{", "},{", t.lstrip("[").rstrip("],")) + "]"
for i, x in enumerate(json.loads(t)):
    print(f"  t={5*i:4d} s  GPU {x['frequency']['actual']:5.0f} MHz  Render {x['engines']['Render/3D']['busy']:5.1f}%  "
          f"GPU {x['power']['GPU']:5.1f} W  Package {x['power']['Package']:5.1f} W")
P
}
{
echo "# $(date -Is) test dlugotrwaly, zasilacz AC=$(cat /sys/class/power_supply/AC*/online)"
for zad in gpu cpu; do
  j=$(mktemp); intel_gpu_top -J -s 5000 -o "$j" >/dev/null 2>&1 & p=$!
  if [ $zad = gpu ]; then
    echo "## HGEMM asm N=2048, 4000 przebiegow (WAIT=sleep), slad co 400"
    WAIT=sleep TRACE=400 REPS=4000 PACK=2 ND1=1 LY=1 ./hgemm hgemm_asm.bin hgemm_16x32p 2048 2>&1
  else
    echo "## OpenBLAS f32 N=2048, 90 s"
    python3 gemm_cpu.py f32 2048 90 2>&1
  fi
  kill -INT $p; wait $p 2>/dev/null; probki "$j"; rm -f "$j"
  sleep 20 2>/dev/null || python3 -c 'import time; time.sleep(20)'   # ostygnięcie
done
} > "$OUT" 2>&1
echo "$OUT"
