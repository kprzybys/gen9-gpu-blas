#!/bin/sh
# Power measurement: runs a command in continuous mode with intel_gpu_top alongside (samples every 500 ms).
# Prints the median: GPU clock, Render engine busy, GPU power and whole-processor power (Package).
# Usage: power.sh <command...>   (the command should run >= 4 s)
tmp=$(mktemp)
intel_gpu_top -J -s 500 -o "$tmp" >/dev/null 2>&1 &
pid=$!
"$@"
kill -INT $pid; wait $pid 2>/dev/null
python3 -I - "$tmp" <<'P'
import json, re, statistics, sys
t = open(sys.argv[1]).read().strip()
t = "[" + re.sub(r"\}\s*,?\s*\{", "},{", t.lstrip("[").rstrip("],")) + "]"
d = json.loads(t)[2:-1]          # without the ramp-up and the tail
med = lambda xs: statistics.median(xs) if xs else float("nan")
f = med([x["frequency"]["actual"] for x in d])
r = med([x["engines"]["Render/3D"]["busy"] for x in d if "Render/3D" in x["engines"]])
g = med([x["power"]["GPU"] for x in d]); p = med([x["power"]["Package"] for x in d])
print(f"   power: GPU {f:.0f} MHz, Render {r:.0f}%, GPU {g:.1f} W, Package {p:.1f} W  ({len(d)} samples)")
P
rm -f "$tmp"
