import os, re, sys
repo = sys.argv[1]
sys.path.insert(0, os.path.join(repo, "asm"))
import gemm_gen as G, attn_gen as A
for kind in ("h", "hd", "s"):
    pro, loop, epi = G.load(kind)
    h, f = G.accumulators(pro)
    dtype = G.KINDS[kind]["dtype"]
    pa, pb, counter, limit, ptrs = G.analyze(loop, dtype)
    acc = set(h) | {r for x in f for r in (x, x + 1)}
    if dtype == "f": acc |= {r for x in h for r in (x, x + 1)}
    ptr = {r for w in ptrs for r in (w, w + 1)}
    cl = {int(x) for x in re.findall(r"r(\d+)", counter + limit)}
    try:
        live, hdr = A.live_in(epi)
    except Exception as e:
        print(kind, "live_in failed:", repr(e)[:150]); continue
    used = {int(x) for l in epi for x in re.findall(r"\br(\d+)", l.split("//")[0])}
    other_live = set(live) - acc - ptr - cl - {0, 1, 127}
    print(f"{kind}: epilogue mentions {len(used - acc - ptr - cl - {0,1,127})} other regs, live-in other {len(other_live)} {sorted(other_live)}, of them send headers {len(set(hdr) & other_live)}")
