"""Accurate HGEMM (16x16 tile, FP16 sum in chunks of 32 products -> FP32) in hand-written Gen9 assembly.

Template: hgemm_16x16kc from bench/hgemm.cl (asm/hgemm_kc.d). From the template: prologue (pointers, accumulator zeroing)
and epilogue ((half)(f + h) write). The k loop is replaced with our own. Detected automatically from the compiler loop:
pointers A/B (:q registers incremented by a constant step; role from data use in mad, also through `mov :w` copies),
counter and limit (`cmp (lt)`), accumulators (zeroed in the prologue: 16 × hf, 16 × f).
Modes: map_f | map_h | loop  <result.bin>   (loop needs asm/hgemm_kc_map.txt: "f|h i r")

Loop: [8 quarters of k (kl 0..3): 64 mad from buffer q%2, then send of the next quarter into that buffer]
      [carry: f(r) += h(r) (mixed add (8) :f <- :hf, 2 per row), h(r) = 0] [counter += 32]
Banks: B row kl in a register of parity kl (even base), A (layout A3, pack_a3_pad) for even kl
in an odd register and vice versa -> 0 conflicts.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.realpath(__file__))
TEMPLATE = os.path.join(HERE, "hgemm_kc.d")
KERNEL = "hgemm_16x16kc"
MAP_FILE = os.path.join(HERE, "hgemm_kc_map.txt")
KC = 32
W = "(W)     "


def load():
    t = open(os.path.join(TEMPLATE, f".text.{KERNEL}.asm")).read().splitlines()
    w = next(i for i, l in enumerate(t) if re.search(r"^\s+while \(16\|M0\)\s+L\d+", l))
    label = re.search(r"while \(16\|M0\)\s+(L\d+)", t[w]).group(1)
    s = next(i for i, l in enumerate(t) if l.startswith(label + ":"))
    return t[:s], t[s:w + 1], t[w + 1:]


def accumulators(prologue):
    """Zeroings right before the loop (after the last jump of the prologue): 16 × hf (h) and 16 × f (f, 2 GRF each)."""
    last = max(i for i, l in enumerate(prologue) if "jmpi" in l)
    h = [int(m.group(1)) for l in prologue[last:] for m in [re.search(r"mov \(16\|M0\)\s+r(\d+)\.0<1>:hf\s+0\.0:hf", l)] if m]
    f = [int(m.group(1)) for l in prologue[last:] for m in [re.search(r"mov \(16\|M0\)\s+r(\d+)\.0<1>:f\s+0\.0:f", l)] if m]
    assert len(h) == 16 and len(f) == 16, (h, f)
    return h, f


def analyze(loop):
    ptrs = sorted({int(m.group(1)) for l in loop
                  for m in [re.search(r"add \((?:8|1)\|M0\)\s+r(\d+)\.0<1>:q\s+r\1\.0<[^>]*>:q\s+\d+:w", l)] if m})
    usage = {}
    for l in loop:
        m = re.search(r"mad \(16\|M0\)\s+r\d+\.0<1>:hf\s+r\d+\.0<2;1>:hf\s+r(\d+)\.0<2;1>:hf\s+r(\d+)\.\d+<0>:hf", l)
        if m:
            usage.setdefault(int(m.group(1)), "B")
            usage.setdefault(int(m.group(2)), "A")
    # data copies mov (16) :w rD <- rS: the role of rD passes to rS (until fixpoint)
    copies = [(int(m.group(1)), int(m.group(2))) for l in loop
             for m in [re.search(r"mov \(16\|M0\)\s+r(\d+)\.0<1>:w\s+r(\d+)\.0<16;16,1>:w", l)] if m]
    for _ in range(4):
        for d, s in copies:
            if d in usage:
                usage.setdefault(s, usage[d])
    defs, role = {}, {}
    for l in loop:
        m = re.search(r"(mov|add) \((?:1|8)\|M0\)\s+r(\d+)\.0<1>:u?q\s+r(\d+)\.0<[^>]*>:u?q(?:\s+(\S+):)?", l)
        if m:
            src = int(m.group(3))
            off = int(m.group(4), 0) if m.group(4) and m.group(1) == "add" else 0
            base, boff = defs.get(src, (src, 0))
            defs[int(m.group(2))] = (base, boff + off)
            continue
        m = re.search(r"send \(16\|M0\)\s+r(\d+):w\s+r(\d+)\s", l)
        if m:
            base, _ = defs.get(int(m.group(2)), (int(m.group(2)), 0))
            r = int(m.group(1))
            roles = {usage[x] for x in range(r, r + 4) if x in usage}
            if base in ptrs and len(roles) == 1:
                role.setdefault(base, set()).update(roles)
    pa = [w for w, t in role.items() if t == {"A"}]
    pb = [w for w, t in role.items() if t == {"B"}]
    assert len(pa) == 1 and len(pb) == 1, (ptrs, role)
    c = next(re.search(r"cmp \(16\|M0\)\s+\(lt\)f0\.0\s+null<1>:d\s+(r\d+\.\d)<0;1,0>:d\s+(r\d+\.\d)<0;1,0>:d", l)
             for l in loop if "(lt)f0.0" in l)
    return pa[0], pb[0], c.group(1), c.group(2), ptrs


def build(prologue, body, epilogue, result):
    tmp = tempfile.mkdtemp()
    d = os.path.join(tmp, "d")
    shutil.copytree(TEMPLATE, d)
    open(os.path.join(d, f".text.{KERNEL}.asm"), "w").write("\n".join(prologue + body + epilogue) + "\n")
    name = os.path.basename(result)
    subprocess.run(["ocloc", "asm", "-file", name, "-dump", d, "-device", "whl"], check=True, stdout=subprocess.DEVNULL)
    shutil.move(os.path.join(d, name), result)
    shutil.rmtree(tmp)


def allocate(free, size):
    w = set(free)
    for b in range(4, 127, 2):
        if all(r in w for r in range(b, b + size)):
            for r in range(b, b + size):
                free.remove(r)
            return b
    raise SystemExit(f"no block of {size} registers with even base")


def loop(prologue, loop_k, epilogue, h, f):
    pa, pb, counter, limit, ptrs = analyze(loop_k)
    mh, mf = {}, {}
    for l in open(MAP_FILE):
        kind, i, r = l.split()
        (mh if kind == "h" else mf)[int(r)] = (h if kind == "h" else f)[int(i)]
    assert len(mh) == 16 and len(mf) == 16
    busy = set(h) | {r for x in f for r in (x, x + 1)} | {0, 1, 127} | {r for w in ptrs for r in (w, w + 1)}
    busy |= {int(x) for x in re.findall(r"r(\d+)", counter + limit)}
    epilogue_used = {int(x) for l in epilogue for x in re.findall(r"\br(\d+)", l.split("//")[0])}
    free = [r for r in range(2, 127) if r not in busy | epilogue_used]
    buffers = [(allocate(free, 4), allocate(free, 4)) for _ in range(2)]   # (A, B), even bases
    H = free[:4]
    assert len(H) == 4
    out = [f"// --- hand-written accurate HGEMM loop: pa=r{pa} pb=r{pb} counter={counter} limit={limit} buffers={buffers} headers={H}"]
    for b in range(2):
        out.append(f"{W}add (1|M0)  r{H[2 * b]}.0<1>:q  r{pa}.0<0;1,0>:q  {128 * b}:w")
        out.append(f"{W}add (1|M0)  r{H[2 * b + 1]}.0<1>:q  r{pb}.0<0;1,0>:q  {128 * b}:w")

    def load_q(b):
        A, B = buffers[b]
        o = []
        for dst, hd in ((A, H[2 * b]), (B, H[2 * b + 1])):
            o.append(f"{W}send (16|M0)  r{dst}:w  r{hd}  0xC  0x024D0CFF")
            o.append(f"{W}add (1|M0)  r{hd}.0<1>:q  r{hd}.0<0;1,0>:q  256:w")
        return o

    def mads(b):
        A, B = buffers[b]
        o = []
        for kl in range(4):
            for r in range(16):
                g = A + 2 * (r // 8) + (1 if kl % 2 == 0 else 0)
                sub = (r % 8) * 2 + kl // 2
                a = mh[r]
                o.append(f"        mad (16|M0)  r{a}.0<1>:hf  r{a}.0<2;1>:hf  r{B + kl}.0<2;1>:hf  r{g}.{sub}<0>:hf")
        return o

    out += load_q(0) + load_q(1) + ["LKC_LOOP:"]
    for q in range(KC // 4):
        out += mads(q % 2) + load_q(q % 2)
    for r in range(16):   # carry the chunk into FP32 (syntax as in compiler code) and zero h
        out.append(f"        add (8|M0)  r{mf[r]}.0<1>:f  r{mf[r]}.0<8;8,1>:f  r{mh[r]}.0<8;8,1>:hf")
        out.append(f"        add (8|M8)  r{mf[r] + 1}.0<1>:f  r{mf[r] + 1}.0<8;8,1>:f  r{mh[r]}.8<8;8,1>:hf")
        out.append(f"        mov (16|M0)  r{mh[r]}.0<1>:hf  0.0:hf")
    out += [f"{W}add (1|M0)  {counter}<1>:d  {counter}<0;1,0>:d  {KC}:w",
            f"{W}cmp (16|M0)  (lt)f0.0  null<1>:d  {counter}<0;1,0>:d  {limit}<0;1,0>:d",
            "(W&f0.0) jmpi  LKC_LOOP"]
    return out


if __name__ == "__main__":
    mode, result = sys.argv[1], sys.argv[2]
    prologue, loop_k, epilogue = load()
    h, f = accumulators(prologue)
    if mode == "analyze":
        print(analyze(loop_k), h, f)
    elif mode in ("map_f", "map_h"):
        target = f if mode == "map_f" else h
        dtype = "f" if mode == "map_f" else "hf"
        build(prologue, ["LMAP:"] + [f"{W}mov (16|M0)  r{a}.0<1>:{dtype}  {float(i)}:{dtype}" for i, a in enumerate(target)],
               epilogue, result)
    else:
        body = loop(prologue, loop_k, epilogue, h, f)
        build(prologue, body, epilogue, result)
        print(body[0])
        print(f"instructions in body: {len(body)}, mad: {sum('mad (' in l for l in body)}")
