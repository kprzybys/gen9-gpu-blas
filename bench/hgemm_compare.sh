#!/bin/sh
# GEMM comparison: our HGEMM (asm and OpenCL C) vs CLBlast (H/SGEMM) vs OpenBLAS (f32). Result: results/hgemm-compare-<date>.txt
cd -P "$(dirname "$0")"
[ -x hgemm_asm.bin ] || make -C ../asm
g() { grep -oE '[0-9.]+ GFLOPS' | head -1 | awk '{print $1}'; }
gp() { grep -oE 'with packing: [0-9.]+ ms, [0-9.]+' | awk '{print $NF}'; }
{
echo "# $(date -Is)  GFLOPS (best of 5); '+pack' = including packing of A and B on the GPU"
printf '%6s | %9s %9s | %9s %9s | %9s | %8s %8s | %8s\n' N asm "asm+pack" "OCL-C p" "+pack" "OCL-C img" "CLB hg" "CLB sg" "OBLAS32"
for N in 256 512 1024 1536 2048 3072 4096; do
  a=$(PACK=2 ND1=1 LY=1 ./hgemm hgemm_asm.bin hgemm_16x32p $N); 
  p=$(PACK=1 ND1=1 LY=1 ./hgemm hgemm.cl hgemm_16x32p $N "-DGM=16")
  i=$(./hgemm hgemm.cl hgemm_16x32 $N | g)
  h=$(./gemm_clblast h $N | g); s=$(./gemm_clblast s $N | g); o=$(python3 gemm_cpu.py f32 $N | g)
  printf '%6s | %9s %9s | %9s %9s | %9s | %8s %8s | %8s\n' $N "$(echo "$a" | g)" "$(echo "$a" | gp)" "$(echo "$p" | g)" "$(echo "$p" | gp)" "$i" "$h" "$s" "$o"
done
echo "# relative error (N=2048) asm: $(PACK=2 ND1=1 LY=1 ./hgemm hgemm_asm.bin hgemm_16x32p 2048 | grep -oE 'avg=[0-9.e-]+ max=[0-9.e-]+')"
echo "# relative error (N=2048) CLBlast hgemm: $(./gemm_clblast h 2048 | grep -oE 'avg=[0-9.e-]+ max=[0-9.e-]+')"
echo "# power, continuous mode ~4 s, N=2048:"
printf '#   asm:            '; ./power.sh sh -c 'PACK=2 ND1=1 LY=1 REPS=200 ./hgemm hgemm_asm.bin hgemm_16x32p 2048 >/dev/null' 
printf '#   CLBlast hgemm:  '; ./power.sh ./gemm_clblast h 2048 4 | tail -1
printf '#   OpenBLAS f32:   '; ./power.sh python3 gemm_cpu.py f32 2048 4 | tail -1
} | tee "../results/hgemm-compare-$(date +%F).txt"
