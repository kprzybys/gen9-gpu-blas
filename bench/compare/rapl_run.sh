#!/bin/bash
# Run a command and print RAPL energy (J) per domain + wall time. Needs read access to
# /sys/class/powercap/intel-rapl:0/**/energy_uj (root-only by default: CVE-2020-8694).
# Usage: rapl_run.sh <command...>   -> last line: "RAPL wall_s pkg_J core_J uncore_J dram_J"
R=/sys/class/powercap/intel-rapl:0
DOM=("$R" "$R/intel-rapl:0:0" "$R/intel-rapl:0:1" "$R/intel-rapl:0:2")
rd() { for d in "${DOM[@]}"; do cat "$d/energy_uj"; done; }
MAX=(); for d in "${DOM[@]}"; do MAX+=("$(cat "$d/max_energy_range_uj")"); done
A=($(rd)); t0=$(date +%s.%N)
"$@"
rc=$?
t1=$(date +%s.%N); B=($(rd))
out="RAPL $(echo "$t1 - $t0" | bc)"
for i in 0 1 2 3; do d=$(( B[i] - A[i] )); (( d < 0 )) && d=$(( d + MAX[i] )); out+=" $(echo "scale=3; $d/1000000" | bc)"; done
echo "$out"
exit $rc
