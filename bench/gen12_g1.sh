#!/bin/sh
# Gen12LP step G1 (docs/gen12lp.md section 6): peak FP32/FP16/DP4a and read bandwidth through NEO, no kcl.
# Run on the target from a directory holding clbench, peak.cl, peak_dp4a.cl, memory.cl.
cd -P "$(dirname "$0")"
G=$((48 * 7 * 32 * 4))   # 4 waves of SIMD32 threads over 48 EU x 7 threads
run() { printf '%-28s ' "$1"; shift; "$@" 2>&1 | tail -1; }
echo "# $(date -Is) $(cat /sys/class/drm/card0/gt_max_freq_mhz) MHz max"
run "fp32 fmav"   ./clbench peak.cl k_fmav $G 256 4000 256 "-DT=float"
run "fp16 fmav"   ./clbench peak.cl k_fmav $G 256 4000 256 "-DT=half"
run "int8 dp4a 16 chains" ./clbench peak_dp4a.cl k_dp4a $G 256 4000 2048 "-cl-std=CL3.0 -DCH=16"
for mb in 1 8 64 256; do
  m=$((mb * 1024 * 1024))
  BUF=$m run "owb2 read ${mb} MB (GB/s)" ./clbench memory.cl k_owb2 $((48 * 7 * 8 * 8)) 64 20000 32 "-DMASK=$((m - 1))"
done
