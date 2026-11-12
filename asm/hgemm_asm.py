"""HGEMM 16x32 in hand-written Gen9 assembly: generator of a loop inserted into the compiler template.

Template: kernel hgemm_16x32p from bench/hgemm.cl (disassembled by asm/disassemble.sh into asm/hgemm.d).
From the template we take the prologue (addresses pa = r78.0:q, pb0 = r78.1:q, pb1 = r6.1:q, k counter = r6.1:d, K = r6.0:d,
accumulator zeroing) and the epilogue (C write). The whole k loop (from the loop label to `while`) is replaced with our own code.

Modes:
  python3 -I hgemm_asm.py map  <result.bin>  # no loop: accumulator no. i := i (to read the register -> (row, column) assignment)
  python3 -I hgemm_asm.py loop <result.bin>  # the actual kernel (needs asm/hgemm_map.txt from map mode)

Loop (step k = 16, two halves kl = 0..7 and 8..15, each in a separate register buffer X/Y):
  [256 mad: half 0 from buffer X] [6 send: next half 0 -> X] [256 mad: half 1 from Y] [6 send -> Y] [counter]
No bank conflicts: B registers (row kl) have the parity of kl (even base), A registers for a given kl
have the opposite parity thanks to the A2 layout (pack_a2 in bench/hgemm.cl).
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.realpath(__file__))
TEMPLATE = os.path.join(HERE, "hgemm.d")
KERNEL = "hgemm_16x32p"
MAP_FILE = os.path.join(HERE, "hgemm_map.txt")


def load():
    t = open(os.path.join(TEMPLATE, f".text.{KERNEL}.asm")).read().splitlines()
    w = next(i for i, l in enumerate(t) if re.search(r"^\s+while \(16\|M0\)\s+L\d+", l))
    label = re.search(r"while \(16\|M0\)\s+(L\d+)", t[w]).group(1)
    s = next(i for i, l in enumerate(t) if l.startswith(label + ":"))
    return t[:s], t[s:w + 1], t[w + 1:]


def accumulators(prologue):
    """Registers zeroed in the prologue as hf: these are the tile accumulators (32 of them)."""
    acc = [int(m.group(1)) for l in prologue for m in [re.search(r"mov \(16\|M0\)\s+r(\d+)\.0<1>:hf\s+0\.0:hf", l)] if m]
    assert len(acc) == 32, acc
    return acc


def regs_of(lines):
    return {int(x) for l in lines for x in re.findall(r"\br(\d+)", l.split("//")[0])}


def free_regs(prologue, epilogue, acc):
    """Free in the loop: everything except the accumulators, the epilogue registers (conservatively: every one mentioned),
    r0-r2 (header, local id), r6 (counter, K, pb1), r78 (pa, pb0), r127."""
    taken = set(acc) | regs_of(epilogue) | {0, 1, 2, 6, 78, 127}
    return [r for r in range(3, 127) if r not in taken]


def even_blocks(free_list, count, size):
    """Allocate `count` blocks of `size` consecutive registers, with an even base."""
    w, out = set(free_list), []
    for b in range(4, 127, 2):
        if len(out) == count:
            break
        if all(r in w for r in range(b, b + size)):
            out.append(b)
            w -= set(range(b, b + size))
    assert len(out) == count, f"not enough free blocks: {out}"
    return out, sorted(w)


def build(prologue, body, epilogue, result):
    tmp = tempfile.mkdtemp()
    d = os.path.join(tmp, "d")
    shutil.copytree(TEMPLATE, d)
    open(os.path.join(d, f".text.{KERNEL}.asm"), "w").write("\n".join(prologue + body + epilogue) + "\n")
    name = os.path.basename(result)
    subprocess.run(["ocloc", "asm", "-file", name, "-dump", d, "-device", "whl"], check=True, stdout=subprocess.DEVNULL)
    shutil.move(os.path.join(d, name), result)
    shutil.rmtree(tmp)


W = "(W)     "


def mode_map(prologue, acc):
    """Accumulator i (order from the prologue) := i. After the epilogue C[r][c] = accumulator number."""
    return ["LMAP:"] + [f"{W}mov (16|M0)  r{a}.0<1>:hf  {float(i)}:hf" for i, a in enumerate(acc)]


def mode_loop(prologue, epilogue, acc):
    # assignment (row, column group) -> accumulator register, from the map file
    acc_map = {}
    for l in open(MAP_FILE):
        i, r, c = map(int, l.split())
        acc_map[(r, c)] = acc[i]
    assert len(acc_map) == 32
    free = free_regs(prologue, epilogue, acc)
    # 6 blocks of 8 registers: X: A, B0, B1; Y: A, B0, B1
    bases, free = even_blocks(free, 6, 8)
    XA, XB0, XB1, YA, YB0, YB1 = bases
    assert len(free) >= 12, f"not enough registers for headers: {free}"
    H = free[:12]   # headers: [buffer][stream A,B0,B1][part 0,1]
    hdr = {(b, s, p): H[b * 6 + s * 2 + p] for b in range(2) for s in range(3) for p in range(2)}
    src = {0: "r78.0", 1: "r78.1", 2: "r6.1"}          # pa, pb0, pb1 (64-bit, from the prologue)
    data = {(0, 0): XA, (0, 1): XB0, (0, 2): XB1, (1, 0): YA, (1, 1): YB0, (1, 2): YB1}

    out = [f"// --- hand-written HGEMM loop: X={XA},{XB0},{XB1} Y={YA},{YB0},{YB1} headers={H}"]
    # start addresses: buffer b (half b) = base + 256*b, part p = +128*p
    for (b, s, p), h in hdr.items():
        out.append(f"{W}add (1|M0)  r{h}.0<1>:q  {src[s]}<0;1,0>:q  {256 * b + 128 * p}:w")

    def load_q(b):
        o = []
        for s in range(3):
            for p in range(2):
                o.append(f"{W}send (16|M0)  r{data[(b, s)] + 4 * p}:w  r{hdr[(b, s, p)]}  0xC  0x024D0CFF")
        for s in range(3):
            for p in range(2):
                o.append(f"{W}add (1|M0)  r{hdr[(b, s, p)]}.0<1>:q  r{hdr[(b, s, p)]}.0<0;1,0>:q  512:w")
        return o

    def mads(b):
        A, B0, B1 = data[(b, 0)], data[(b, 1)], data[(b, 2)]
        o = []
        for kl in range(8):
            for r in range(16):
                ga = A + 2 * (r // 4) + (1 if kl % 2 == 0 else 0)
                sub = (r % 4) * 4 + kl // 2
                for c, Bc in ((0, B0), (1, B1)):
                    a = acc_map[(r, c)]
                    o.append(f"        mad (16|M0)  r{a}.0<1>:hf  r{a}.0<2;1>:hf  r{Bc + kl}.0<2;1>:hf  r{ga}.{sub}<0>:hf")
        return o

    out += load_q(0) + load_q(1)
    out += ["LHG_LOOP:"]
    out += mads(0) + load_q(0) + mads(1) + load_q(1)
    out += [f"{W}add (1|M0)  r6.1<1>:d  r6.1<0;1,0>:d  16:w",
            f"{W}cmp (16|M0)  (lt)f0.0  null<1>:d  r6.1<0;1,0>:d  r6.0<0;1,0>:d",
            "(W&f0.0) jmpi  LHG_LOOP"]
    return out


if __name__ == "__main__":
    mode, result = sys.argv[1], sys.argv[2]
    prologue, loop, epilogue = load()
    acc = accumulators(prologue)
    if mode == "map":
        build(prologue, mode_map(prologue, acc), epilogue, result)
    else:
        body = mode_loop(prologue, epilogue, acc)
        build(prologue, body, epilogue, result)
        print(f"instructions in body: {len(body)}, mad: {sum('mad (' in l for l in body)}")
