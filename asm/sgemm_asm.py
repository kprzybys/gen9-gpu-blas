"""SGEMM 16x16 (FP32) in hand-written Gen9 assembly: generator of a loop inserted into the compiler template (sgemm_16x16p).

Like hgemm_asm.py, but the pointer, counter and limit registers of the loop are detected automatically from the compiler loop:
  - pointer = a :q register incremented in the loop by 1024 (step k = 16 × 64 B),
  - A or B: data from the send reading this pointer goes into mad as a scalar (src2 -> A) or a vector (src1 -> B),
  - counter and limit: the `cmp (lt)` operands in the loop.
Modes: map <result.bin> | loop <result.bin>   (loop needs asm/sgemm_map.txt from map mode: "i r")

Loop (step k = 16; halves kl 0..7 / 8..15 in buffers X / Y):
  For FP32 a B row is 2 GRF. A bank conflict counts for the first GRF of src1 and the scalar register src2.
  B rows with even kl -> block BE (even base), odd -> block BO (odd base); pack_bf2 reorders the
  rows within the block of 8 (0,2,4,6,1,3,5,7). A (pack_af2): even kl in odd GRFs and vice versa (block A with an even base).
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.realpath(__file__))
TEMPLATE = os.path.join(HERE, "sgemm.d")
KERNEL = "sgemm_16x16p"
MAP_FILE = os.path.join(HERE, "sgemm_map.txt")
W = "(W)     "


def load():
    t = open(os.path.join(TEMPLATE, f".text.{KERNEL}.asm")).read().splitlines()
    w = next(i for i, l in enumerate(t) if re.search(r"^\s+while \(16\|M0\)\s+L\d+", l))
    label = re.search(r"while \(16\|M0\)\s+(L\d+)", t[w]).group(1)
    s = next(i for i, l in enumerate(t) if l.startswith(label + ":"))
    return t[:s], t[s:w + 1], t[w + 1:]


def analyze(loop):
    """Returns (pointer_A, pointer_B, counter, limit) as operand strings, e.g. 'r34.0'."""
    ptrs = sorted({int(m.group(1)) for l in loop
                  for m in [re.search(r"add \(8\|M0\)\s+r(\d+)\.0<1>:q\s+r\1\.0<4;4,1>:q\s+1024:w", l)] if m})
    defs = {}      # header register -> (source, offset), last definition before send
    role = {}
    usage = {}     # data register -> 'A' (scalar in mad) / 'B' (vector in mad)
    for l in loop:
        m = re.search(r"mad \(16\|M0\)\s+r\d+\.0<1>:f\s+r\d+\.0<2;1>:f\s+r(\d+)\.0<2;1>:f\s+r(\d+)\.\d+<0>:f", l)
        if m:
            usage.setdefault(int(m.group(1)), "B")
            usage.setdefault(int(m.group(2)), "A")
    for l in loop:
        m = re.search(r"(mov|add) \(1\|M0\)\s+r(\d+)\.(\d)<1>:u?q\s+r(\d+)\.(\d)<0;1,0>:u?q(?:\s+(\S+):)?", l)
        if m:
            dst = (int(m.group(2)), int(m.group(3)))
            src = (int(m.group(4)), int(m.group(5)))
            off = int(m.group(6), 0) if m.group(6) else 0
            base, boff = defs.get(src, (src, 0))
            defs[dst] = (base, boff + off)
            continue
        m = re.search(r"send \(16\|M0\)\s+r(\d+):w\s+r(\d+)\s", l)
        if m:
            base, _ = defs.get((int(m.group(2)), 0), ((int(m.group(2)), 0), 0))
            r = int(m.group(1))
            kind = next((usage[x] for x in range(r, r + 4) if x in usage), None)
            if base[0] in ptrs and kind:
                role.setdefault(base[0], set()).add(kind)
    pa = [w for w, t in role.items() if t == {"A"}]
    pb = [w for w, t in role.items() if t == {"B"}]
    assert len(pa) == 1 and len(pb) == 1, (ptrs, role)
    c = next(re.search(r"cmp \(16\|M0\)\s+\(lt\)f0\.0\s+null<1>:d\s+(r\d+\.\d)<0;1,0>:d\s+(r\d+\.\d)<0;1,0>:d", l)
             for l in loop if "cmp (16|M0)" in l and "(lt)f0.0" in l)
    return f"r{pa[0]}.0", f"r{pb[0]}.0", c.group(1), c.group(2), ptrs


def accumulators(prologue):
    acc = [int(m.group(1)) for l in prologue for m in [re.search(r"mov \(16\|M0\)\s+r(\d+)\.0<1>:f\s+0\.0:f", l)] if m]
    assert len(acc) == 16, acc
    return acc


def free_regs(epilogue, taken):
    used = {int(x) for l in epilogue for x in re.findall(r"\br(\d+)", l.split("//")[0])}
    return [r for r in range(3, 127) if r not in used | taken]


def allocate(free, size, parity):
    w = set(free)
    for b in range(4 + parity, 127, 2):
        if all(r in w for r in range(b, b + size)):
            for r in range(b, b + size):
                free.remove(r)
            return b
    raise SystemExit(f"no block of {size} registers with parity {parity}")


def build(prologue, body, epilogue, result):
    tmp = tempfile.mkdtemp()
    d = os.path.join(tmp, "d")
    shutil.copytree(TEMPLATE, d)
    open(os.path.join(d, f".text.{KERNEL}.asm"), "w").write("\n".join(prologue + body + epilogue) + "\n")
    name = os.path.basename(result)
    subprocess.run(["ocloc", "asm", "-file", name, "-dump", d, "-device", "whl"], check=True, stdout=subprocess.DEVNULL)
    shutil.move(os.path.join(d, name), result)
    shutil.rmtree(tmp)


def loop(prologue, loop_k, epilogue, acc):
    pa, pb, counter, limit, ptrs = analyze(loop_k)
    acc_map = {}
    for l in open(MAP_FILE):
        i, r = map(int, l.split())
        acc_map[r] = acc[i]
    busy = set(acc) | {r for a in acc for r in (a, a + 1)} | {0, 1, 2, 127}
    busy |= {int(x) for x in re.findall(r"r(\d+)", pa + pb + counter + limit)}
    busy |= {r for w in ptrs for r in (w, w + 1)}
    free = free_regs(epilogue, busy)
    buffers = []
    for _ in range(2):                          # quarters of k alternately in buffers 0 / 1
        A = allocate(free, 8, 0)
        BE = allocate(free, 4, 0)
        BO = allocate(free, 4, 1)
        buffers.append((A, BE, BO))
    H = free[:4]                                 # headers: [buffer][A, B]
    assert len(H) == 4, "no registers for headers"
    out = [f"// --- hand-written SGEMM loop: pa={pa} pb={pb} counter={counter} limit={limit} buffers={buffers} headers={H}"]
    for b in range(2):                          # buffer b starts at quarter b (offset 256*b)
        out.append(f"{W}add (1|M0)  r{H[2 * b]}.0<1>:q  {pa}<0;1,0>:q  {256 * b}:w")
        out.append(f"{W}add (1|M0)  r{H[2 * b + 1]}.0<1>:q  {pb}<0;1,0>:q  {256 * b}:w")

    def load_q(b):
        """Quarter into buffer b: A 2 × 128 B, BE 128 B, BO 128 B; headers advanced by 2 quarters (512 B)."""
        A, BE, BO = buffers[b]
        hA, hB = H[2 * b], H[2 * b + 1]
        o = []
        for dst, h in [(A, hA), (A + 4, hA), (BE, hB), (BO, hB)]:
            o.append(f"{W}send (16|M0)  r{dst}:w  r{h}  0xC  0x024D0CFF")
            o.append(f"{W}add (1|M0)  r{h}.0<1>:q  r{h}.0<0;1,0>:q  128:w")
        o.append(f"{W}add (1|M0)  r{hA}.0<1>:q  r{hA}.0<0;1,0>:q  256:w")
        o.append(f"{W}add (1|M0)  r{hB}.0<1>:q  r{hB}.0<0;1,0>:q  256:w")
        return o

    def mads(b):
        A, BE, BO = buffers[b]
        o = []
        for kl in range(4):
            breg = (BE if kl % 2 == 0 else BO) + 2 * (kl // 2)
            for r in range(16):
                g = A + 2 * (r // 4) + (1 if kl % 2 == 0 else 0)
                sub = (r % 4) * 2 + kl // 2
                a = acc_map[r]
                o.append(f"        mad (16|M0)  r{a}.0<1>:f  r{a}.0<2;1>:f  r{breg}.0<2;1>:f  r{g}.{sub}<0>:f")
        return o

    out += load_q(0) + load_q(1) + ["LSG_LOOP:"]
    for q in range(4):
        out += mads(q % 2) + load_q(q % 2)
    out += [f"{W}add (1|M0)  {counter}<1>:d  {counter}<0;1,0>:d  16:w",
            f"{W}cmp (16|M0)  (lt)f0.0  null<1>:d  {counter}<0;1,0>:d  {limit}<0;1,0>:d",
            "(W&f0.0) jmpi  LSG_LOOP"]
    return out


if __name__ == "__main__":
    mode, result = sys.argv[1], sys.argv[2]
    prologue, loop_k, epilogue = load()
    acc = accumulators(prologue)
    if mode == "analyze":
        print(analyze(loop_k), acc)
    elif mode == "map":
        build(prologue, ["LMAP:"] + [f"{W}mov (16|M0)  r{a}.0<1>:f  {float(i)}:f" for i, a in enumerate(acc)], epilogue, result)
    else:
        body = loop(prologue, loop_k, epilogue, acc)
        build(prologue, body, epilogue, result)
        print(body[0])
        print(f"instructions in body: {len(body)}, mad: {sum('mad (' in l for l in body)}")
