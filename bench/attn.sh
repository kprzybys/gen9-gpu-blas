#!/bin/sh
# Fused attention kernel (bench/attn.cl) on one layer: Qwen2.5-1.5B and Llama-3.2-1B shapes, n_q 512, n_k 4096.
# Usage: attn.sh "<RQ> <NK> [options]" ...  (each argument is one variant for both shapes; Llama runs with
# RQ = LLAMA_RQ if set). Run with nice -n 19 ionice -c3 on a shared machine. Example: LLAMA_RQ=4 ./attn.sh "2 2"
# Hand-assembled binaries (one per shape): CL_QWEN=<qwen.bin> CL_LLAMA=<llama.bin> (default: CL, else attn.cl;
# relative paths are relative to bench/).
# RAW=<file>: append the raw output of every run to <file> (a "# ..." header with date, host and the exact command,
# then the program's line unchanged), so results files can quote what bench/attn printed, not only tables.
case $RAW in ''|/*) ;; *) RAW=$PWD/$RAW ;; esac
cd -P "$(dirname "$0")"
[ attn -nt attn.c ] || gcc -O2 -Wall -o attn attn.c -lOpenCL -lm || exit 1
run() {   # run <CL file> <args...>
    cl=$1; shift
    if [ -n "$RAW" ]; then
        echo "# $(date '+%F %T') $(hostname) STEP=${STEP:-7} REPS=${REPS:-5} CL=$cl ./attn $*" >> "$RAW"
        CL=$cl ./attn "$@" 2>&1 | tee -a "$RAW"
    else
        CL=$cl ./attn "$@"
    fi
}
for v in "$@"; do
    set -- $v; rq=$1; nk=$2; shift 2
    run "${CL_QWEN:-${CL:-attn.cl}}" 12 2 128 512 4096 $rq $nk "$*"
    run "${CL_LLAMA:-${CL:-attn.cl}}" 32 8 64 512 4096 ${LLAMA_RQ:-$rq} $nk "$*"
done
