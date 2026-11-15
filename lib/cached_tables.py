#!/usr/bin/env python3
"""Run a generator step through a content-addressed cache (docs/build-knowhow.md: the kernel tables take most of a
clean build and ccache cannot cache ocloc).

usage: cached_tables.py CACHE_DIR --key FILE... [--salt STR...] --out FILE... [--stdout FILE] -- COMMAND...
The key is the sha256 of the salts, the command line (output paths replaced by their names) and the contents of the
key files. On a hit the outputs are copied from CACHE_DIR/<key>/; on a miss COMMAND runs (--stdout FILE: its stdout
goes to FILE) and the outputs are stored. Messages go to stderr. CACHE_DIR empty: always run.
"""
import hashlib, os, shutil, subprocess, sys

argv = sys.argv[1:]
cache = argv.pop(0)
sep = argv.index("--")
opts, cmd = argv[:sep], argv[sep + 1:]
groups = {"--key": [], "--salt": [], "--out": [], "--stdout": []}
cur = []
for a in opts:
    if a in groups: cur = groups[a]
    else: cur.append(a)
outs = groups["--out"]
names = {o: os.path.basename(o) for o in outs}
h = hashlib.sha256()
for s in groups["--salt"]: h.update(b"salt\0" + s.encode() + b"\0")
def norm(a):   # paths of the build tree differ between builds: only names enter the key
    if "=" in a and "/" in a.split("=", 1)[1]: k, v = a.split("=", 1); return k + "=" + os.path.basename(v)
    return os.path.basename(a) if "/" in a else a
for c in cmd: h.update(b"cmd\0" + norm(c).encode() + b"\0")
for k in sorted(groups["--key"], key=lambda p: (os.path.basename(p), p)):
    h.update(b"file\0" + os.path.basename(k).encode() + b"\0")
    with open(k, "rb") as f: h.update(hashlib.sha256(f.read()).digest())
key = h.hexdigest()[:32]
slot = os.path.join(cache, key) if cache else None
if slot and all(os.path.exists(os.path.join(slot, names[o])) for o in outs):
    for o in outs:
        os.makedirs(os.path.dirname(os.path.abspath(o)), exist_ok=True)
        shutil.copyfile(os.path.join(slot, names[o]), o)
    print(f"cached_tables: hit {key} ({', '.join(names.values())})", file=sys.stderr)
    sys.exit(0)
stdout = open(groups["--stdout"][0], "wb") if groups["--stdout"] else None
r = subprocess.run(cmd, stdout=stdout)
if stdout: stdout.close()
if r.returncode:
    for o in outs:
        if stdout and o == stdout.name and os.path.exists(o): os.remove(o)   # no half-written table
    sys.exit(r.returncode)
if slot:
    tmp = slot + f".tmp{os.getpid()}"
    os.makedirs(tmp, exist_ok=True)
    for o in outs: shutil.copyfile(o, os.path.join(tmp, names[o]))
    try: os.rename(tmp, slot)
    except OSError: shutil.rmtree(tmp, ignore_errors=True)   # another build stored it first
