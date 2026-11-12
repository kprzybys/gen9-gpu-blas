#!/usr/bin/env python3
"""Static check of the library kernels against the Cherryview (Gen8 LP) EU region rules (docs/gen8.md, section 3).

ocloc 24.35 compiles Gen8 only as Broadwell (-device bdw). Cherryview adds rules for instructions with a 64-bit
operand (DF/Q/UQ) or an integer DWord x DWord mul (PRM CHV vol07, printed p.823-824; Mesa elk_eu_validate.c:1736-1830):
  R1a  source and destination horizontal strides equal (in bytes) and a multiple of 8 (scalar source exempt)
  R1b  VStride == Width * HStride; a width-1 region <v;1,0> is reported separately as R1b-w1 (it is qword-strided
       when v*size == 8, but fails Mesa's literal check)
  R1c  equal subregister offsets (scalar source exempt)
  R2   no indirect addressing
  R3   no ARF operand (accumulator, flag, address), no mac, no AccWrEn; R3-mme: madm/math.invm with acc2-9
  R4   no DepCtrl (NoDDClr/NoDDChk)
3-source instructions are skipped, as in Mesa (Align16 on Gen8). Also counts kernels that use scratch (kcl has none).

Usage: python3 scripts/chv_scan.py OUT_DIR       (extra IGC flags via the environment, e.g. IGC_VISAOptions=-noFP64bRegion)
Builds the programs of lib/gen_kernels.py (except the prebuilt Gen9 .bin files) for bdw into OUT_DIR, then scans them.
"""
import ast, collections, glob, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZE = {"q": 8, "uq": 8, "df": 8, "d": 4, "ud": 4, "f": 4, "w": 2, "uw": 2, "hf": 2, "b": 1, "ub": 1, "v": 4, "uv": 4, "vf": 4}
SKIP = ("send", "jmpi", "call", "ret", "if", "else", "endif", "while", "break", "cont", "join", "goto", "nop", "wait", "halt")
INSN = re.compile(r"^(\([^)]*\)\s*)?([a-z_.0-9]+)\s+\((\d+)\|M\d+\)\s*(\([^)]*\)\s*)?(.*)$")
OPND = re.compile(r"^(-|~|\(abs\)|-\(abs\))?(r\[[^\]]*\]|[a-z]+\d*)(?:\.(\d+))?(?:<([\d;,]+)>)?:(\w+)$")
IMM = re.compile(r"^-?(0x[0-9a-fA-F]+|[\d.e+-]+|inf|s?q?nan)(\(.*\))?:\w+$")


def programs():
    tree = ast.parse(open(os.path.join(ROOT, "lib", "gen_kernels.py")).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "PROGRAMS":
            return [p for p in ast.literal_eval(node.value) if p[1] is not None]
    sys.exit("chv_scan: PROGRAMS not found in lib/gen_kernels.py")


def build(out):
    for i, (src, opts) in enumerate(programs()):
        cmd = ["ocloc", "compile", "-q", "-file", os.path.join(ROOT, "bench", src), "-device", "bdw",
               "-output_no_suffix", "-output", f"p{i}", "-out_dir", out] + (["-options", opts] if opts else [])
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"chv_scan: {src} {opts}: {r.stdout}{r.stderr}")
        subprocess.run(["ocloc", "disasm", "-file", os.path.join(out, f"p{i}.bin"), "-device", "bdw",
                        "-dump", os.path.join(out, f"p{i}.d")], capture_output=True, check=True)


def operand(text):
    if IMM.match(text):
        return {"imm": True, "type": text.rsplit(":", 1)[1]}
    m = OPND.match(text)
    if not m:
        return None
    reg, region = m.group(2), m.group(4)
    return {"imm": False, "ind": reg.startswith("r["), "null": reg == "null",
            "arf": not reg.startswith("r[") and reg != "null" and not re.match(r"^r\d+$", reg),
            "sub": int(m.group(3) or 0), "type": m.group(5), "size": SIZE.get(m.group(5), 0),
            "region": [int(x) for x in re.split("[;,]", region)] if region else None}


def check(line):
    """Rule hits for one instruction line (empty set if the rules do not apply), or None if not parsed."""
    m = INSN.match(line)
    if not m or m.group(2).startswith(SKIP):
        return set()
    if ".mme" in line or ".nomme" in line:   # madm / math.invm / math.rsqtm: DF with accumulators acc2-9 (ARF)
        return {"64b:R3-mme"}
    op, rest = m.group(2), m.group(5)
    opts = re.search(r"\{([^}]*)\}\s*$", rest)
    rest = rest[:opts.start()] if opts else rest
    rest = re.sub(r"^(\(\w+\))?f\d\.\d\s+", "", rest)   # conditional modifier
    ops = [operand(t) for t in rest.split()]
    if not ops or None in ops or len(ops) not in (2, 3):  # 3-source (4 operands) skipped
        return None if None in ops else set()
    dst, srcs = ops[0], ops[1:]
    wide = any(o.get("size") == 8 for o in ops if not o["imm"])
    dwmul = op == "mul" and len(srcs) == 2 and all(s["type"] in ("d", "ud") for s in srcs)
    if not (wide or dwmul):
        return set()
    hits = set()
    dst_stride = (dst["region"][0] if dst["region"] else 1) * dst["size"]
    for s in srcs:
        if s["imm"] or s["null"]:
            continue
        if s["ind"]:
            hits.add("R2"); continue
        if s["arf"]:
            hits.add("R3"); continue
        v, w, h = s["region"]
        if v == 0 and w == 1 and h == 0:   # scalar
            continue
        if w == 1 and h == 0:
            hits.add("R1b-w1")
            stride = v * s["size"]
        else:
            stride = h * s["size"]
            if v != w * h:
                hits.add("R1b")
        if stride != dst_stride or stride % 8:
            hits.add("R1a")
        if not (dst["null"] or dst["arf"] or dst["ind"]) and dst["sub"] * dst["size"] != s["sub"] * s["size"]:
            hits.add("R1c")
    if dst["ind"]:
        hits.add("R2")
    if dst["arf"] or op == "mac" or (opts and "AccWrEn" in opts.group(1)):
        hits.add("R3")
    if opts and "NoDD" in opts.group(1):
        hits.add("R4")
    return {f"{'dwmul' if dwmul and not wide else '64b'}:{h}" for h in hits}


def scan(out):
    total, unparsed, per_kernel, example = collections.Counter(), 0, {}, {}
    for f in sorted(glob.glob(os.path.join(out, "p*.d", "*_KernelHeap.asm"))):
        kernel = os.path.basename(f)[:-len("_KernelHeap.asm")]
        hits_k, scratch = collections.Counter(), False
        for line in open(f):
            scratch |= "scratch" in line   # in the disassembler's send comment
            line = line.split("//")[0].strip()
            hits = check(line)
            if hits is None:
                unparsed += 1
                continue
            for h in hits:
                hits_k[h] += 1; total[h] += 1
                example.setdefault(h, f"{kernel}: {line}")
        per_kernel[kernel] = (hits_k, scratch)
    strict = [k for k, (h, _) in per_kernel.items() if not h]
    no_w1 = [k for k, (h, _) in per_kernel.items() if not [x for x in h if not x.endswith("R1b-w1")]]
    print(f"kernels: {len(per_kernel)}; without any hit: {len(strict)}; without hits other than R1b-w1: {len(no_w1)}")
    print("kernels using scratch:", " ".join(k for k, (_, s) in per_kernel.items() if s) or "none")
    for h, n in sorted(total.items()):
        print(f"{n:6d} {h:14s} e.g. {example[h]}")
    print("unparsed instruction lines:", unparsed)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    os.makedirs(sys.argv[1], exist_ok=True)
    build(sys.argv[1])
    scan(sys.argv[1])
