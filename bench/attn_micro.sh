#!/bin/sh
# Attention microbenchmarks (docs/attention-gpu.md, section 7): exp throughput and SLM read bandwidth.
# Usage: attn_micro.sh <gpu tag, e.g. uhd620 or hd530>. Result: results/attn-micro-<tag>-<date>.txt
cd -P "$(dirname "$0")"
[ -x clbench ] || gcc -O2 -Wall -o clbench clbench.c -lOpenCL
T=${1:-uhd620}
{
echo "# exp, FP32, 8 independent chains (Gop/s; one op = exp + the multiply feeding it)"
for e in exp native_exp; do printf '%-16s ' "$e"; ./clbench attn_micro.cl k_exp 131072 256 200 64 "-DEXP=$e"; done
printf '%-16s ' "exp fast-relaxed"; ./clbench attn_micro.cl k_exp 131072 256 200 64 "-DEXP=exp -cl-fast-relaxed-math"
echo "# SLM reads, float4 per lane, work-group 256, 32 KB (GB/s); PAT 0 consecutive, 1 broadcast, 2 stride 2"
for p in 0 1 2; do printf 'PAT=%-12s ' "$p"; ./clbench attn_micro.cl k_slm 24576 256 2000 128 "-DPAT=$p"; done
} | tee "../results/attn-micro-$T-$(date +%F).txt"
