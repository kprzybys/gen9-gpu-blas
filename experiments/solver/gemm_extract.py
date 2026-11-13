import json, os, re, sys
repo = sys.argv[1]
sys.path.insert(0, os.path.join(repo, "asm"))
import gemm_gen as G
out = {}
for kind in ("h", "hd", "s"):
    try:
        pro, loop, epi = G.load(kind)
    except Exception as e:
        out[kind] = {"error": repr(e)[:200]}; continue
    h, f = G.accumulators(pro)
    dtype = G.KINDS[kind]["dtype"]
    pa, pb, counter, limit, ptrs = G.analyze(loop, dtype)
    acc_regs = set(h) | {r for x in f for r in (x, x + 1)} | (set(h) if dtype == "hf" else {r for x in h for r in (x, x+1)})
    ptr_regs = {r for w in ptrs for r in (w, w + 1)}
    cl = {int(x) for x in re.findall(r"r(\d+)", counter + limit)}
    epi_used = {int(x) for l in epi for x in re.findall(r"\br(\d+)", l.split("//")[0])}
    free = G.free_regs(pro, epi, acc_regs | ptr_regs | cl)
    out[kind] = dict(acc_h=len(h), acc_f_pairs=len(f), acc_regs=len(acc_regs), ptr_regs=len(ptr_regs), cntlim=len(cl),
                     epi_only=len(epi_used - acc_regs - ptr_regs - cl - {0, 1, 127}), free=len(free),
                     free_list=free)
    print(kind, {k: v for k, v in out[kind].items() if k != "free_list"})
json.dump(out, open("gemm_instances.json", "w"), indent=1)
