#!/bin/bash
# OA calibration for bench/l3port.cl: L3ShaderThroughput vs known block-read bytes (2048 subgroups x n x 128 B per launch).
set -u; cd "$(dirname "$0")"; [ -n "${NEO_ENV:-}" ] && . "$NEO_ENV"   # NEO_ENV: env script of a NEO install outside the system (legacy NEO on HD 530)
N=${N:-4096}; export BUF=$((1 << 20)) REPS=2
for cfg in "64 -DMADS=0 -DPF=4" "64 -DMADS=16 -DPF=2" "64 -DMADS=16 -DPF=4" "128 -DMADS=16 -DPF=4"; do
  set -- $cfg; L=$1; shift
  echo "## local $L $* (bytes per launch $((2048 * N * 128)))"
  nice -n 19 bash oa.sh -n 3 -b 90 nice -n 19 ./clbench l3port.cl k_port 16384 $L $N 16 "$*"
done
