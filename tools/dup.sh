#!/bin/bash
# Copy-paste detection (jscpd, pinned) over code that ships: C/OpenCL (.c .h .cl), C++, Python, shell.
# Skips third_party, results, reports/notes, generated kernel tables. Prints the summary; JSON in ${1:-/tmp}/jscpd-report.json.
#   tools/dup.sh [outdir]        (needs node/npx; first run downloads jscpd 4.0.5)
set -eu
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
OUT=${1:-${TMPDIR:-/tmp}/jscpd-$$}; mkdir -p "$OUT"
npx --yes jscpd@4.0.5 --silent --min-tokens 70 --formats-exts "c:c,h,cl;cpp:cpp,hpp;python:py;bash:sh" \
  --ignore "**/third_party/**,**/results/**,**/research_notes/**,**/reports/**,**/build/**,**/kcl_build/**,**/*_kernels.c,**/kernels.c" \
  --reporters json --output "$OUT" . > /dev/null
python3 - "$OUT/jscpd-report.json" <<'PY'
import json, sys, collections
d = json.load(open(sys.argv[1])); t = d["statistics"]["total"]
print(f"lines {t['lines']}  duplicated {t['duplicatedLines']} ({t['percentage']:.1f} %)  clones {t['clones']}")
for k, v in sorted(d["statistics"]["formats"].items()):
    v = v["total"]; print(f"  {k:7s} {v['lines']:6d} lines  {v['duplicatedLines']:5d} dup ({v['percentage']:.1f} %)  {v['clones']} clones")
pairs = collections.Counter()
for c in d["duplicates"]:
    a, b = (c[k]["name"].split("/")[0] if "/" in c[k]["name"] else "." for k in ("firstFile", "secondFile"))
    pairs[tuple(sorted((a, b)))] += c["lines"]
print("duplicated lines by directory pair:")
for p, n in pairs.most_common(12): print(f"  {p[0]:12s} {p[1]:12s} {n:5d}")
PY
