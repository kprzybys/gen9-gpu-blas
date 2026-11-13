#!/bin/sh
# Fused attention kernel (bench/attn.cl) on one layer: Qwen2.5-1.5B and Llama-3.2-1B shapes, n_q 512, n_k 4096.
# Usage: attn.sh "<RQ> <NK> [options]" ...  (each argument is one variant for both shapes; Llama runs with
# RQ = LLAMA_RQ if set). Run with nice -n 19 ionice -c3 on a shared machine. Example: LLAMA_RQ=4 ./attn.sh "2 2"
cd -P "$(dirname "$0")"
[ attn -nt attn.c ] || gcc -O2 -Wall -o attn attn.c -lOpenCL -lm || exit 1
for v in "$@"; do
    set -- $v; rq=$1; nk=$2; shift 2
    ./attn 12 2 128 512 4096 $rq $nk "$*"
    ./attn 32 8 64 512 4096 ${LLAMA_RQ:-$rq} $nk "$*"
done
