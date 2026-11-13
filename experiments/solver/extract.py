"""Extract the register-allocation instance of attn_gen.py's sched() for several shapes (offline: ocloc only, no GPU).

Reads asm/attn_gen.py (does not modify it). Output: JSON with, per shape, IGC's fixed registers by category and the
counts of values sched() must place. Usage: python3 -I extract.py <repo> <out.json>
"""
import json
import os
import re
import sys
import tempfile

repo, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(repo, "asm"))
import attn_gen as G  # noqa: E402

SHAPES = {
    "qwen": "-DD=128 -DG=6 -DHKV=2 -DRQ=2 -DNK=2",
    "llama": "-DD=64 -DG=4 -DHKV=8 -DRQ=4 -DNK=2",
    "qwen_tokc": "-DD=128 -DG=6 -DHKV=2 -DRQ=2 -DNK=2 -DQTOK -DCAUSAL=1",
    "llama_tokc": "-DD=64 -DG=4 -DHKV=8 -DRQ=4 -DNK=2 -DQTOK -DCAUSAL=1",
    "llama_nk1": "-DD=64 -DG=4 -DHKV=8 -DRQ=4 -DNK=1",
    "qwen_rq4": "-DD=128 -DG=6 -DHKV=2 -DRQ=4 -DNK=2",
}
BASE = "-cl-fast-relaxed-math -cl-mad-enable"

G.check_ocloc()
res = {}
for name, sh in SHAPES.items():
    opts = f"{BASE} {sh}"
    d = tempfile.mkdtemp()
    try:
        G.disassemble(os.path.join(repo, "asm", "attn.cl"), opts, os.path.join(d, "d"))
        t = open(os.path.join(d, "d", f".text.{G.KERNEL}.asm")).read().splitlines()
        pro, loop, epi = G.split_loop(t)
        P = G.params(opts)
        M = G.igc_maps(pro, loop, epi, P)
    except (AssertionError, SystemExit, Exception) as e:  # noqa: BLE001
        res[name] = {"opts": sh, "error": f"{type(e).__name__}: {e}"[:300]}
        print(name, "ERROR", res[name]["error"])
        continue
    q = sorted(set(M["q"].values()))
    o = sorted(set(M["o"].values()))
    lr = sorted(M["l"])
    ptr = sorted({M["kp"][0], M["kp"][0] + 1, M["vp"][0], M["vp"][0] + 1})
    cl = sorted({int(re.match(r"r(\d+)", x).group(1)) for x in (M["cnt"], M["lim"])})
    epi_live, epi_hdr = G.live_in(epi)
    known = {0} | set(q) | set(o) | set(lr) | set(ptr) | set(cl)
    other = sorted((set(epi_live) - set(epi_hdr)) - known)
    busy = known | set(other)
    res[name] = {
        "opts": sh, "P": P, "rows": M["rows"],
        "fixed": {"r0": [0], "q": q, "o": o, "l": lr, "ptr": ptr, "cntlim": cl, "epi_other": other},
        "o_live_after": sorted(set(o) & set(epi_live)), "l_live_after": sorted(set(lr) & set(epi_live)),
        "free_now": sorted(set(range(128)) - busy),
    }
    print(f"{name}: Q {len(q)} O {len(o)} l {len(lr)} ptr {len(ptr)} cnt/lim {len(cl)} epi_other {len(other)} "
          f"free {len(res[name]['free_now'])}")
json.dump(res, open(out, "w"), indent=1)
