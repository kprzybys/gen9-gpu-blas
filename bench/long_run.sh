#!/bin/sh
# Long run (~90 s): HGEMM asm on the GPU, then OpenBLAS f32 on the CPU. Every 5 s: clock and power (intel_gpu_top).
# Result: results/long-run-<date>.txt
cd -P "$(dirname "$0")"
OUT=../results/long-run-$(date +%F).txt
samples() {  # $1 = intel_gpu_top JSON file
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
echo "# $(date -Is) long run, power supply AC=$(cat /sys/class/power_supply/AC*/online)"
for job in gpu cpu; do
  j=$(mktemp); intel_gpu_top -J -s 5000 -o "$j" >/dev/null 2>&1 & p=$!
  if [ $job = gpu ]; then
    echo "## HGEMM asm N=2048, 4000 runs (WAIT=sleep), trace every 400"
    WAIT=sleep TRACE=400 REPS=4000 PACK=2 ND1=1 LY=1 ./hgemm hgemm_asm.bin hgemm_16x32p 2048 2>&1
  else
    echo "## OpenBLAS f32 N=2048, 90 s"
    python3 gemm_cpu.py f32 2048 90 2>&1
  fi
  kill -INT $p; wait $p 2>/dev/null; samples "$j"; rm -f "$j"
  sleep 20 2>/dev/null || python3 -c 'import time; time.sleep(20)'   # cool-down
done
} > "$OUT" 2>&1
echo "$OUT"
