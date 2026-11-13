#!/bin/sh
# Run a GPU workload under the i915 OA counters (metric set ComputeBasic) and print a summary (bench/oa_sum.py):
# EU active / stall / idle, thread occupancy, both FPU pipes, send pipe, L3 / GTI / SLM throughput.
# Needs igt-gpu-tools (i915-perf-recorder, i915-perf-reader) and dev.i915.perf_stream_paranoid = 0.
# Usage: bench/oa.sh [-n runs (default 3)] [-b min GpuBusy %, default 95] <command> [args...]
# The workload should keep the GPU busy for tens of ms per run; reports are averaged over whole launches.
# Raw reader output stays in $OA_KEEP if set (for results files). See docs/isa-notes.md, "OA counters".
n=3; b=95
while [ $# -gt 0 ]; do case $1 in -n) n=$2; shift 2;; -b) b=$2; shift 2;; *) break;; esac; done
[ $# -gt 0 ] || { echo "usage: $0 [-n runs] [-b busy%] <command> [args...]" >&2; exit 2; }
[ "$(cat /proc/sys/dev/i915/perf_stream_paranoid)" = 0 ] || { echo "oa.sh: dev.i915.perf_stream_paranoid != 0" >&2; exit 2; }
d=$(mktemp -d); here=$(cd -P "$(dirname "$0")" && pwd)
i915-perf-recorder -m ComputeBasic -p 0.005 -o "$d/rec" >"$d/rec.log" 2>&1 &
p=$!
sleep 1
i=0; while [ $i -lt "$n" ]; do "$@" || echo "oa.sh: workload exited with $?" >&2; i=$((i + 1)); done
sleep 0.5; kill -INT $p; wait $p
C=GpuBusy,GpuTime,EuActive,EuStall,EuThreadOccupancy,EuFpuBothActive,Fpu0Active,Fpu1Active,EuSendActive,EuAvgIpcRate
C=$C,AvgGpuCoreFrequency,L3ShaderThroughput,GtiReadThroughput,GtiWriteThroughput,SlmBytesRead,SlmBytesWritten
i915-perf-reader -r -c "$C" "$d/rec" >"$d/rep.txt" 2>&1 || { cat "$d/rec.log" "$d/rep.txt" >&2; rm -rf "$d"; exit 1; }
python3 -I "$here/oa_sum.py" "$d/rep.txt" "$b"
[ -n "$OA_KEEP" ] && cp "$d/rep.txt" "$OA_KEEP"
rm -rf "$d"
