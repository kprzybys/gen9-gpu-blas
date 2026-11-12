#!/bin/sh
# Buduje HGEMM dokładny z ręczną pętlą asm: bench/hgemm.cl -> szablon hgemm_16x16kc -> mapy akumulatorów f i h
# (dwa uruchomienia na GPU) -> bench/hgemm_kc_asm.bin
set -e
cd -P "$(dirname "$0")"
cp ../bench/hgemm.cl hgemm_kc.cl
./disassemble.sh hgemm_kc.cl hgemm_kc.d >/dev/null
tmp=$(mktemp -d)
: > "$tmp/mapa"
for rodz in f h; do
  python3 -I hgemm_kc_asm.py mapa_$rodz "$tmp/m.bin"
  (cd ../bench && DUMP=1 PACK=5 TN=16 ND1=1 LY=1 ./hgemm "$tmp/m.bin" hgemm_16x16kc 64) > "$tmp/m.txt"
  python3 -I - "$tmp/m.txt" "$rodz" >> "$tmp/mapa" <<'P'
import sys
rows = [l.split()[:16] for l in open(sys.argv[1]) if l.strip()][:16]
seen = set()
for r in range(16):
    v = set(rows[r]); assert len(v) == 1, (r, v)
    i = int(float(v.pop())); seen.add(i); print(sys.argv[2], i, r)
assert len(seen) == 16
P
done
mv "$tmp/mapa" hgemm_kc_map.txt
python3 -I hgemm_kc_asm.py petla ../bench/hgemm_kc_asm.bin
rm -rf "$tmp"
