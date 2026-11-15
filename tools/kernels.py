#!/usr/bin/env python3
"""Inventory of OpenCL kernels in tracked .cl files (refactor plan 2026-11-15, stage 1).

For every `kernel void NAME(...)`: where it is defined, which repo the file goes to (tools/repo-map.toml, if present), and which code
files name it in a string literal (clCreateKernel, KER(...), kcl tables, Python). Reports:
  --dup      same name defined in more than one file, and identical bodies under different names/files
  --unused   kernels no code file names (candidates for removal; asm-generated or test-only kernels show up here too)
  --file F   kernels of one .cl file with their users
  (default)  per-file counts: defined / used / unused / duplicated
Names built at run time from printf formats ("trsm_inv%d_%s") count as uses; kernels whose name is made by a macro
(`kernel void F(axpy)`) are skipped. Bodies are compared after stripping comments and whitespace, so formatting-only copies count as identical.
"""
import argparse, collections, hashlib, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depgraph as dg

KDEF = re.compile(r"\b(?:__)?kernel\s+(?:__attribute__\s*\(\(.*?\)\)\s*)*void\s+(\w+)\s*\(", re.S)


def bodies(text):
    """(name, line, normalized body) for every kernel definition in comment-stripped OpenCL C."""
    out = []
    for m in KDEF.finditer(text):
        ls = text.rfind("\n", 0, m.start()) + 1
        prev = text[max(0, ls - 2):ls].rstrip("\n")
        if text[ls:m.start()].lstrip().startswith("#define") or prev.endswith("\\"):
            continue  # kernel template inside a macro (NAME is a parameter); instances are not visible here
        if re.match(r"\s*\w+\s*\)\s*\(", text[m.end():]):
            continue  # `kernel void F(axpy)(...)`: name made by a macro (per-type instances), not a literal name
        i = text.index("{", m.end())
        depth, j = 0, i
        while j < len(text):
            if text[j] == "{": depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0: break
            j += 1
        norm = re.sub(r"\s+", "", text[m.end():j + 1])
        out.append((m.group(1), text.count("\n", 0, m.start()) + 1, norm))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dup", action="store_true")
    ap.add_argument("--unused", action="store_true")
    ap.add_argument("--file")
    a = ap.parse_args()
    rm = dg.RepoMap(os.path.join(dg.ROOT, "tools/repo-map.toml"))
    g = dg.Graph()
    defs = []  # (name, file, line, hash, size)
    for f in g.files:
        if f.endswith(".cl"):
            for name, line, norm in bodies(dg.strip_comments(g.read(f), ".cl")):
                defs.append((name, f, line, hashlib.sha1(norm.encode()).hexdigest()[:12], len(norm)))
    by_name_users = collections.defaultdict(set)  # kernel name -> code files naming it in a string literal
    for f in g.files:
        kind = dg.file_kind(f)
        if kind in (".c", ".h", ".cpp", ".hpp", ".py", ".sh"):
            txt = dg.strip_comments(g.read(f), kind)
            for lit in re.findall(r'"(\w{3,})"|\'(\w{3,})\'', txt):
                n = lit[0] or lit[1]
                if n in g.kernels: by_name_users[n].add(f)
            # printf-built names: "trsm_inv%d_%s" -> trsm_inv\d+_\w+
            for lit in re.findall(r'"([\w%]*%[\w%]*)"', txt):
                if len(re.sub(r"%[-0-9]*l*[dius]|_", "", lit)) < 3: continue  # "%s", "%s_%d": too generic
                rx = re.compile(re.sub(r"%[-0-9]*l*[dius]", r"\\w+", lit) + r"\Z")
                for n in g.kernels:
                    if rx.match(n): by_name_users[n].add(f)
    # "%s_t" built from a used base name (snprintf(n, "%s_t", table[i].name)): the suffixed kernel is used too
    suffixes = set()
    for f in g.files:
        if dg.file_kind(f) in (".c", ".h", ".cpp", ".py"):
            suffixes.update(re.findall(r'"%s(\w{1,8})"', dg.strip_comments(g.read(f), dg.file_kind(f))))
    for n in list(g.kernels):
        for suf in suffixes:
            if n.endswith(suf) and by_name_users.get(n[:-len(suf)]):
                by_name_users[n] |= by_name_users[n[:-len(suf)]]
    names = collections.Counter(n for n, *_ in defs)
    by_hash = collections.defaultdict(list)
    for d in defs: by_hash[d[3]].append(d)

    if a.file:
        for name, f, line, h, size in defs:
            if f == a.file:
                u = sorted(by_name_users.get(name, ()))
                print(f"{f}:{line}\t{name}\t{size:6d}B\tusers: {', '.join(u) or '-'}")
        return 0
    if a.unused:
        for name, f, line, h, size in sorted(defs, key=lambda d: (d[1], d[2])):
            if not by_name_users.get(name):
                print(f"{f}:{line}\t{name}\t[{rm.repo(f)}]")
        return 0
    if a.dup:
        print("same name in several files:")
        for name, n in sorted(names.items()):
            if n > 1:
                locs = [(f, line, h) for nm, f, line, h, s in defs if nm == name]
                same = "identical" if len({h for *_, h in locs}) == 1 else "DIFFERENT bodies"
                print(f"  {name:28s} {same:16s} " + "  ".join(f"{f}:{line}" for f, line, h in locs))
        print("\nidentical bodies under different names (or same name, different files) - size >= 200 B:")
        for h, ds in sorted(by_hash.items(), key=lambda kv: -kv[1][0][4]):
            if len(ds) > 1 and ds[0][4] >= 200 and len({d[0] for d in ds}) > 1:
                print(f"  {ds[0][4]:6d}B  " + "  ".join(f"{n}@{f}:{line}" for n, f, line, *_ in ds))
        return 0
    per = collections.defaultdict(lambda: [0, 0, 0, 0])
    for name, f, line, h, size in defs:
        p = per[f]; p[0] += 1
        if by_name_users.get(name): p[1] += 1
        else: p[2] += 1
        if names[name] > 1: p[3] += 1
    print(f"{'file':34s} {'repo':8s} {'def':>4s} {'used':>4s} {'unused':>6s} {'dupname':>7s}")
    for f, (n, u, x, dn) in sorted(per.items()):
        print(f"{f:34s} {rm.repo(f) or '?':8s} {n:4d} {u:4d} {x:6d} {dn:7d}")
    tot = [sum(v[i] for v in per.values()) for i in range(4)]
    print(f"{'TOTAL':34s} {'':8s} {tot[0]:4d} {tot[1]:4d} {tot[2]:6d} {tot[3]:7d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
