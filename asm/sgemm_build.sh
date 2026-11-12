#!/bin/sh
# Buduje SGEMM z ręczną pętlą asm: bench/sgemm.cl -> szablon -> mapa akumulatorów (uruchomienie na GPU) -> bench/sgemm_asm.bin
set -e
cd -P "$(dirname "$0")"
cp ../bench/sgemm.cl sgemm.cl
./disassemble.sh sgemm.cl sgemm.d >/dev/null
tmp=$(mktemp -d)
python3 -I sgemm_asm.py mapa "$tmp/mapa.bin"
(cd ../bench && DUMP=1 ./sgemm "$tmp/mapa.bin" sgemm_16x16p 64) > "$tmp/mapa.txt"
python3 -I - "$tmp/mapa.txt" sgemm_map.txt <<'P'
import sys
rows = [l.split() for l in open(sys.argv[1]) if l.strip()][:16]
out = []
for r in range(16):
    v = set(rows[r]); assert len(v) == 1, (r, v); out.append((int(float(v.pop())), r))
assert len({o[0] for o in out}) == 16
open(sys.argv[2], "w").write("".join(f"{i} {r}\n" for i, r in sorted(out)))
P
python3 -I sgemm_asm.py petla ../bench/sgemm_asm.bin
rm -rf "$tmp"
