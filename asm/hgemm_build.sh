#!/bin/sh
# Buduje kernel HGEMM z ręczną pętlą w asemblerze: bench/hgemm.cl -> szablon -> mapa akumulatorów -> bench/hgemm_asm.bin
# Krok „mapa” uruchamia kernel na GPU, żeby odczytać, w którym rejestrze epilog kompilatora trzyma który wiersz C.
set -e
cd -P "$(dirname "$0")"
cp ../bench/hgemm.cl hgemm.cl
./disassemble.sh hgemm.cl hgemm.d >/dev/null
tmp=$(mktemp -d)
python3 -I hgemm_asm.py mapa "$tmp/mapa.bin"
(cd ../bench && DUMP=1 PACK=2 ND1=1 LY=1 ./hgemm "$tmp/mapa.bin" hgemm_16x32p 64) > "$tmp/mapa.txt"
python3 -I - "$tmp/mapa.txt" hgemm_map.txt <<'P'
import sys
rows = [l.split() for l in open(sys.argv[1]) if l.strip()][:16]
out = []
for r in range(16):
    for c in range(2):
        vals = {rows[r][c * 16 + j] for j in range(16)}
        assert len(vals) == 1, (r, c, vals)
        out.append((int(float(vals.pop())), r, c))
assert len({o[0] for o in out}) == 32
open(sys.argv[2], "w").write("".join(f"{i} {r} {c}\n" for i, r, c in sorted(out)))
P
python3 -I hgemm_asm.py petla ../bench/hgemm_asm.bin
rm -rf "$tmp"
