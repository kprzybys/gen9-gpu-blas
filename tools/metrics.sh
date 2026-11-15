#!/bin/bash
# Refactor progress metrics (plan 2026-11-15): size, cross-directory dependencies, kernels, duplication, portability.
# Re-run after each stage and diff against results/refactor-metrics-*.txt.   tools/metrics.sh > results/refactor-metrics-<date>.txt
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
T=tools
CODE=(':!results' ':!reports' ':!research_notes' ':!docs' ':!*.md' ':!*.txt' ':!tools/metrics.sh')
echo "# refactor metrics  $(git describe --always --dirty)  $(date +%F)"
echo; echo "## size (tokei, code lines)"
command -v tokei > /dev/null && tokei -C -t C,'C Header',C++,OpenCL,Python,Shell,Makefile . -e results -e reports -e research_notes -e third_party 2>/dev/null | grep -E "^ [A-Z]" | sort -u
echo; echo "## cross-directory code edges (tools/depgraph.py --dirs)"; python3 $T/depgraph.py --dirs
echo; echo "## edges breaking repo order (tools/depgraph.py --check)"; python3 $T/depgraph.py --check | tail -n +1 | grep -E '^edges|^  ' | head -20
echo; echo "## kernels (tools/kernels.py)"; python3 $T/kernels.py | tail -1
echo "kernel names defined in >1 file: $(python3 $T/kernels.py --dup | sed -n '/^same name/,/^$/p' | grep -c '^  ')"
echo; echo "## duplication (tools/dup.sh)"; command -v npx > /dev/null && $T/dup.sh "${TMPDIR:-/tmp}/jscpd-metrics" | head -4
echo; echo "## portability"
echo "Makefiles: $(git ls-files '*Makefile' | wc -l)   CMakeLists: $(git ls-files '*CMakeLists.txt' | wc -l)"
echo "copies of lib/kcl.c compiled into other targets: $(grep -lE '\.\./lib/kcl\.c' */Makefile 2>/dev/null | wc -l) (targets linking lib/libkcl.a: $(grep -lE 'lib/libkcl\.a' */Makefile 2>/dev/null | wc -l))"
echo "home paths / host names in code: $(git grep -cE '(~|\$HOME|/home/[a-z]+)/|\b(the 32-thread desktop|gp-gen9|tc-gen9|ssh gp|kp-laptop)\b' -- "${CODE[@]}" | awk -F: '{s+=$2;n++} END{print s+0" lines in "n+0" files"}')"
echo "abspath defines in Makefiles: $(git grep -c 'abspath' -- '*Makefile' | awk -F: '{s+=$2} END{print s+0}')"
echo "fixed device/ocloc target (-device whl, PCI ids): $(git grep -cE '\-device whl|0x3ea0|0x5917|0x1912' -- "${CODE[@]}" | awk -F: '{s+=$2;n++} END{print s+0" lines in "n+0" files"}')"
