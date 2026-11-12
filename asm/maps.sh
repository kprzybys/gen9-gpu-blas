#!/bin/sh
# Regenerates one accumulator map (register -> row/column of C) by running the kernel on the GPU.
# Usage: asm/maps.sh h|hd|s|hgemm|hgemm_kc|sgemm      (called for all kinds by `make -C asm maps`)
# Needs: bench/hgemm and bench/sgemm built (make -C bench), the disassembly dirs (make builds them), and an Intel GPU.
# The compiler's epilog keeps each row of C in some register; the generator in `map` mode writes a kernel in which
# accumulator number i is the constant i, the kernel is run with DUMP=1, and the printed values give the mapping.
set -e
cd -P "$(dirname "$0")"
K=$1
# GEN: generator command; MODES: map modes; RUN: bench program with its environment; KERNEL: kernel to run;
# GROUPS: column groups of 16 per row; LABEL: map line prefix; SORT: sort lines by register number
case $K in
  h)         GEN="gemm_gen.py h";  MODES="map";        RUN="env NEW=1 PACK=6 ND1=1 LY=1 ./hgemm";           KERNEL=hgemm_v;      GROUPS=2; LABEL="";    SORT=0; MAP=gen_h_map.txt ;;
  hd)        GEN="gemm_gen.py hd"; MODES="map_f map_h"; RUN="env NEW=1 PACK=7 TN=16 ND1=1 LY=1 ./hgemm";   KERNEL=hgemm_kc_v;   GROUPS=1; LABEL="fh";  SORT=0; MAP=gen_hd_map.txt ;;
  s)         GEN="gemm_gen.py s";  MODES="map";        RUN="env NEW=1 PACK2=2 ./sgemm";                    KERNEL=sgemm_v;      GROUPS=1; LABEL="h";   SORT=0; MAP=gen_s_map.txt ;;
  hgemm)     GEN="hgemm_asm.py";   MODES="map";        RUN="env PACK=2 ND1=1 LY=1 ./hgemm";                KERNEL=hgemm_16x32p; GROUPS=2; LABEL="";    SORT=1; MAP=hgemm_map.txt ;;
  hgemm_kc)  GEN="hgemm_kc_asm.py"; MODES="map_f map_h"; RUN="env PACK=5 TN=16 ND1=1 LY=1 ./hgemm";       KERNEL=hgemm_16x16kc; GROUPS=1; LABEL="fh"; SORT=0; MAP=hgemm_kc_map.txt ;;
  sgemm)     GEN="sgemm_asm.py";   MODES="map";        RUN="./sgemm";                                      KERNEL=sgemm_16x16p; GROUPS=1; LABEL="";    SORT=1; MAP=sgemm_map.txt ;;
  *) echo "usage: $0 h|hd|s|hgemm|hgemm_kc|sgemm"; exit 2 ;;
esac
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
: > "$tmp/map"
for m in $MODES; do
  python3 -I $GEN $m "$tmp/m.bin"
  (cd ../bench && env DUMP=1 $RUN "$tmp/m.bin" $KERNEL 64) > "$tmp/m.txt"
  python3 -I - "$tmp/m.txt" "$GROUPS" "$LABEL" "$SORT" "$m" >> "$tmp/map" <<'P'
import sys
path, groups, label, sort, mode = sys.argv[1:]
groups, sort = int(groups), int(sort)
if len(label) == 2:   # "fh": f for map_f, h for map_h
    label = label[0] if mode.endswith("_f") else label[1]
rows = [l.split() for l in open(path) if l.strip()][:16]
out = []
for r in range(16):
    for c in range(groups):
        v = set(rows[r][c * 16:(c + 1) * 16]); assert len(v) == 1, (r, c, v)
        out.append((int(float(v.pop())), r, c))
assert len({o[0] for o in out}) == 16 * groups
if sort:
    out.sort()
for i, r, c in out:
    cols = f"{r} {c}" if groups == 2 else f"{r}"
    print(f"{label} {i} {cols}".lstrip())
P
done
mv "$tmp/map" "$MAP"
