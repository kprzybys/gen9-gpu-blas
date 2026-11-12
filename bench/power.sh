#!/bin/sh
# Pomiar mocy: uruchamia polecenie w trybie ciągłym i równolegle intel_gpu_top (próbki co 500 ms).
# Wypisuje medianę: taktowanie GPU, zajętość silnika Render, moc GPU i całego procesora (Package).
# Użycie: power.sh <polecenie...>   (polecenie powinno działać >= 4 s)
tmp=$(mktemp)
intel_gpu_top -J -s 500 -o "$tmp" >/dev/null 2>&1 &
pid=$!
"$@"
kill -INT $pid; wait $pid 2>/dev/null
python3 -I - "$tmp" <<'P'
import json, re, statistics, sys
t = open(sys.argv[1]).read().strip()
t = "[" + re.sub(r"\}\s*,?\s*\{", "},{", t.lstrip("[").rstrip("],")) + "]"
d = json.loads(t)[2:-1]          # bez rozbiegu i końcówki
med = lambda xs: statistics.median(xs) if xs else float("nan")
f = med([x["frequency"]["actual"] for x in d])
r = med([x["engines"]["Render/3D"]["busy"] for x in d if "Render/3D" in x["engines"]])
g = med([x["power"]["GPU"] for x in d]); p = med([x["power"]["Package"] for x in d])
print(f"   moc: GPU {f:.0f} MHz, Render {r:.0f}%, GPU {g:.1f} W, Package {p:.1f} W  ({len(d)} probek)")
P
rm -f "$tmp"
