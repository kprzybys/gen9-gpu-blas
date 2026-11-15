#!/usr/bin/env python3
"""Dependency graph between paths of the repo and their target repositories (tools/repo-map.toml).

Edges come from tracked files only (git ls-files), comments stripped:
  path    a path-like token (`../bench/sgemm.cl`, `lib/kcl.c`, `$(abspath ../lib)`) that resolves to a tracked file/dir
  include #include "x.h" (relative to the file, else a unique basename in the repo)
  name    a string literal naming a tracked file by basename (kcl_program("gen_s.bin"), "sgemm.cl")
  import  Python `import x` / `from x import` resolving to a tracked x.py (same dir first, else unique basename)
  kernel  a string literal equal to a `kernel void NAME` defined in a tracked .cl file
  doc     a Markdown link or `backticked` path in a .md file (reported only with --docs; never a violation)

Usage:
  tools/depgraph.py                  summary: repo -> repo edge counts and the edges that break repo-map deps
  tools/depgraph.py --dirs           top-level directory -> directory matrix (e.g. who still uses bench/)
  tools/depgraph.py --edges [GLOB]   every edge whose source or target matches GLOB
  tools/depgraph.py --docs           cross-repo documentation links (dead after the split)
  tools/depgraph.py --list REPO      paths assigned to REPO (input for filter-repo)
  tools/depgraph.py --check          exit 1 if a code edge breaks the allowed repo dependencies
"""
import argparse, collections, os, re, subprocess, sys, tomllib

ROOT = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                      cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
CODE_EXT = {".c", ".h", ".cpp", ".hpp", ".cc", ".cl", ".py", ".sh"}
NAME_EXT = (".cl", ".bin", ".h", ".c", ".cpp", ".py", ".so", ".json", ".npy", ".sh")


def glob_re(pat):
    out, i = "", 0
    while i < len(pat):
        if pat.startswith("**/", i): out += "(?:.*/)?"; i += 3
        elif pat.startswith("**", i): out += ".*"; i += 2
        elif pat[i] == "*": out += "[^/]*"; i += 1
        elif pat[i] == "?": out += "[^/]"; i += 1
        else: out += re.escape(pat[i]); i += 1
    return re.compile(out + r"\Z")


class RepoMap:
    def __init__(self, path):
        cfg = tomllib.load(open(path, "rb"))
        self.deps = {k: set(v.get("deps", [])) for k, v in cfg["repos"].items()}
        self.rules = [(r["repo"], [glob_re(p) for p in r["patterns"]]) for r in cfg["rule"]]

    def repo(self, path):
        for repo, pats in self.rules:
            if any(p.match(path) for p in pats): return repo
        return None

    def allowed(self, a, b):
        return a == b or b in self.deps.get(a, ())


def strip_comments(text, ext):
    if ext in (".c", ".h", ".cpp", ".hpp", ".cc", ".cl"):
        # keep string literals, drop // and /* */ comments
        return re.sub(r'"(?:\\.|[^"\\\n])*"|//[^\n]*|/\*.*?\*/',
                      lambda m: m.group(0) if m.group(0).startswith('"') else " " * 0 + "\n" * m.group(0).count("\n"),
                      text, flags=re.S)
    if ext in (".py", ".sh", "mk", ".toml"):
        out = []
        for line in text.split("\n"):
            q = None
            for i, ch in enumerate(line):
                if q:
                    if ch == q: q = None
                elif ch in "\"'": q = ch
                elif ch == "#" and (i == 0 or line[i - 1] in " \t;"):
                    line = line[:i]; break
            out.append(line)
        text = "\n".join(out)
        if ext == ".py":  # docstrings are documentation, not dependencies
            text = re.sub(r'("""|\'\'\').*?\1', lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
        return text
    return text


def file_kind(path):
    base = os.path.basename(path)
    if base == "Makefile" or base.endswith(".mk"): return "mk"
    return os.path.splitext(base)[1]


class Graph:
    def __init__(self):
        self.files = subprocess.run(["git", "ls-files", "-z"], capture_output=True, cwd=ROOT).stdout.decode().split("\0")
        self.files = [f for f in self.files if f and os.path.isfile(os.path.join(ROOT, f))]
        self.fileset = set(self.files)
        self.dirs = set()
        for f in self.files:
            d = os.path.dirname(f)
            while d: self.dirs.add(d); d = os.path.dirname(d)
        self.by_base = collections.defaultdict(list)
        for f in self.files: self.by_base[os.path.basename(f)].append(f)
        self.kernels = collections.defaultdict(set)
        for f in self.files:
            if f.endswith(".cl"):
                txt = strip_comments(self.read(f), ".cl")
                for m in re.finditer(r"\b(?:__)?kernel\s+(?:__attribute__\s*\(\(.*?\)\)\s*)*void\s+(\w+)\s*\(", txt, re.S):
                    self.kernels[m.group(1)].add(f)
        self.edges = []  # (src, dst, kind, line)
        for f in self.files: self.scan(f)

    def read(self, f):
        try: return open(os.path.join(ROOT, f), encoding="utf-8", errors="replace").read()
        except OSError: return ""

    def resolve(self, src, tok):
        tok = tok.strip("'\"`,;:()[]{}<>")
        tok = re.sub(r"^\$\((?:abspath|realpath)\s+", "", tok).rstrip(")")
        tok = re.sub(r"^\$\(?(?:CURDIR|PWD)\)?/", "", tok)
        if not tok or "$" in tok or tok.startswith(("/", "~", "http")): return None
        cands = [os.path.normpath(os.path.join(os.path.dirname(src), tok))]
        if "/" in tok and not tok.startswith("."): cands.append(os.path.normpath(tok))
        for c in cands:
            if c.startswith(".."): continue
            if c in self.fileset or c in self.dirs: return c
        return None

    def by_name(self, src, name):
        hits = self.by_base.get(name, [])
        same = [h for h in hits if os.path.dirname(h) == os.path.dirname(src)]
        if same: return same[0]
        return hits[0] if len(hits) == 1 else None

    def add(self, src, dst, kind, line):
        if dst and dst != src: self.edges.append((src, dst, kind, line))

    def scan(self, f):
        kind = file_kind(f)
        if kind == ".md":
            txt = self.read(f)
            for m in re.finditer(r"\]\(([^)#\s]+)|`([A-Za-z0-9_.][A-Za-z0-9_./-]*/[A-Za-z0-9_.-]+)`", txt):
                tok = m.group(1) or m.group(2)
                line = txt.count("\n", 0, m.start()) + 1
                self.add(f, self.resolve(f, tok), "doc", line)
            return
        if kind not in CODE_EXT and kind != "mk": return
        txt = strip_comments(self.read(f), kind)
        lines = txt.split("\n")
        for ln, line in enumerate(lines, 1):
            m = re.match(r'\s*#\s*include\s+"([^"]+)"', line)
            if m and kind != "mk":
                self.add(f, self.resolve(f, m.group(1)) or self.by_name(f, os.path.basename(m.group(1))), "include", ln)
                continue
            for tok in re.findall(r"[A-Za-z0-9_.$()/{}~-]*/[A-Za-z0-9_./-]*", line):
                if tok.count("/") == 0 or tok.startswith("//"): continue
                self.add(f, self.resolve(f, tok), "path", ln)
            if kind == ".py":
                m = re.match(r"\s*(?:from\s+([\w.]+)\s+import|import\s+([\w., ]+))", line)
                if m:
                    mods = [m.group(1)] if m.group(1) else [x.strip().split(" ")[0] for x in m.group(2).split(",")]
                    for mod in mods:
                        top = mod.split(".")[0]
                        if top: self.add(f, self.by_name(f, top + ".py"), "import", ln)
            for s in re.findall(r'"([^"\n]{2,120})"|\'([^\'\n]{2,120})\'', line):
                s = s[0] or s[1]
                if s.endswith(NAME_EXT) and "/" not in s:
                    self.add(f, self.by_name(f, s), "name", ln)
                if s in self.kernels and kind != ".cl":
                    for d in sorted(self.kernels[s]): self.add(f, d, "kernel", ln)


def top(p):
    parts = p.split("/")
    return parts[0] if len(parts) > 1 else "."


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default=os.path.join(ROOT, "tools/repo-map.toml"))
    ap.add_argument("--dirs", action="store_true")
    ap.add_argument("--edges", nargs="?", const="**", metavar="GLOB")
    ap.add_argument("--docs", action="store_true")
    ap.add_argument("--list", metavar="REPO")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    rm = RepoMap(a.map)
    if a.list:
        g_files = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=ROOT).stdout.split()
        for f in g_files:
            if rm.repo(f) == a.list: print(f)
        return 0
    g = Graph()
    code = [e for e in g.edges if e[2] != "doc"]
    if a.edges:
        pat = glob_re(a.edges)
        for s, d, k, ln in sorted(set(g.edges)):
            if pat.match(s) or pat.match(d): print(f"{s}:{ln}\t{k}\t-> {d}")
        return 0
    if a.dirs:
        cnt = collections.Counter((top(s), top(d)) for s, d, k, ln in code if top(s) != top(d))
        for (s, d), n in sorted(cnt.items()): print(f"{s:14s} -> {d:14s} {n:4d}")
        return 0
    bad = [(s, d, k, ln) for s, d, k, ln in (g.edges if a.docs else code)
           if not rm.allowed(rm.repo(s), rm.repo(d))]
    if a.docs:
        bad = [e for e in bad if e[2] == "doc"]
    if not a.check or a.docs:
        cnt = collections.Counter((rm.repo(s), rm.repo(d), k) for s, d, k, ln in (g.edges if a.docs else code)
                                  if rm.repo(s) != rm.repo(d))
        print("cross-repo edges (src -> dst kind: count):")
        for (s, d, k), n in sorted(cnt.items()):
            flag = "" if rm.allowed(s, d) else "   <-- not allowed"
            print(f"  {s:8s} -> {d:8s} {k:8s} {n:4d}{flag}")
    print(f"\n{'documentation links' if a.docs else 'edges breaking repo deps'}: {len(bad)}")
    by = collections.defaultdict(list)
    for s, d, k, ln in sorted(set(bad)): by[(s, d)].append((k, ln))
    for (s, d), v in sorted(by.items()):
        print(f"  {s} [{rm.repo(s)}] -> {d} [{rm.repo(d)}]  " + ", ".join(f"{k}:{ln}" for k, ln in v[:6])
              + (" ..." if len(v) > 6 else ""))
    return 1 if (a.check and bad) else 0


if __name__ == "__main__":
    sys.exit(main())
