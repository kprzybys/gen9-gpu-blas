#!/bin/sh
# Buduje kernel GEMM z ręczną pętlą asm: gemm_build.sh h|hd|s  ->  bench/gen_<rodzaj>.bin
# Kroki: szablon (OpenCL C -> kod maszynowy) -> mapa akumulatorów (uruchomienie na GPU) -> pętla z gemm_gen.py.
set -e
cd -P "$(dirname "$0")"
R=$1
case $R in
  h)  CL=hgemm; K=hgemm_v;    RUN="env NOWE=1 PACK=6 ND1=1 LY=1 ../bench/hgemm"; MAPY="mapa" ;;
  hd) CL=hgemm; K=hgemm_kc_v; RUN="env NOWE=1 PACK=7 TN=16 ND1=1 LY=1 ../bench/hgemm"; MAPY="mapa_f mapa_h" ;;
  s)  CL=sgemm; K=sgemm_v;    RUN="env NOWE=1 PACK2=2 ../bench/sgemm"; MAPY="mapa" ;;
  *) echo "uzycie: $0 h|hd|s"; exit 2 ;;
esac
cp ../bench/$CL.cl gen_$R.cl
./disassemble.sh gen_$R.cl gen_$R.d >/dev/null
tmp=$(mktemp -d); : > "$tmp/mapa"
for m in $MAPY; do
  python3 -I gemm_gen.py $R $m "$tmp/m.bin"
  (cd ../bench && DUMP=1 $(echo $RUN | sed 's|\.\./bench/|./|') "$tmp/m.bin" $K 64) > "$tmp/m.txt"
  python3 -I - "$tmp/m.txt" $R $m >> "$tmp/mapa" <<'P'
import sys
plik, rodz, tryb = sys.argv[1:]
rows = [l.split() for l in open(plik) if l.strip()][:16]
grupy = 2 if rodz == "h" else 1
seen = set()
for r in range(16):
    for c in range(grupy):
        v = set(rows[r][c * 16:(c + 1) * 16]); assert len(v) == 1, (r, c, v)
        i = int(float(v.pop())); seen.add(i)
        print(f"{i} {r} {c}" if rodz == "h" else f"{'f' if tryb == 'mapa_f' else 'h'} {i} {r}")
assert len(seen) == 16 * grupy
P
done
mv "$tmp/mapa" gen_${R}_mapa.txt
python3 -I gemm_gen.py $R petla ../bench/gen_$R.bin
rm -rf "$tmp"
