"""Common generator of hand-written GEMM loops in Gen9 assembly for UHD 620 (replaces hgemm_asm.py, sgemm_asm.py, hgemm_kc_asm.py).

Types (templates in bench/*.cl; all batched: matrix index from the thread index, b = 0 for 1 matrix):
  h  : hgemm_16x32q    tile 16×32, FP16, A2 + Bq,   half-steps of k (2 × 8 kl), 512 mad / k16
  hd : hgemm_16x16kc_b tile 16×16, FP16 -> chunks of 32 -> FP32, A3 + B16, quarters of k, 256 mad / k16
  s  : sgemm_16x16p_b  tile 16×16, FP32, A2f + B2f, quarters of k, 256 mad / k16
Taken automatically from the compiler loop: pointers A and B (:q registers incremented in the loop; role from data use in mad,
also through `mov :w` copies), counter and limit (`cmp (lt)`), accumulators (zeroed before the loop).
Accumulator map (register -> row/column) from a GPU run (map* modes, see maps.sh).

Usage: python3 -I gemm_gen.py <type> analyze|map|map_f|map_h|loop <result.bin>
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.realpath(__file__))
W = "(W)     "
KINDS = {
    # "v" templates: arguments (Ap, Bp, C, M, N, ldc, K, Kfull) -- pieces of data packed once
    "h": dict(file="hgemm", kernel="hgemm_v", dtype="hf"),
    "hd": dict(file="hgemm", kernel="hgemm_kc_v", dtype="hf"),
    "s": dict(file="sgemm", kernel="sgemm_v", dtype="f"),
}
SEND = "0xC  0x024D0CFF"   # a64 aligned OWord block read ×8 (128 B), header 1 GRF


def kind_dir(kind):
    return os.path.join(HERE, f"gen_{kind}.d")


def load(kind):
    k = KINDS[kind]["kernel"]
    t = open(os.path.join(kind_dir(kind), f".text.{k}.asm")).read().splitlines()
    w = next(i for i, l in enumerate(t) if re.search(r"^\s+while \(16\|M0\)\s+L\d+", l))
    label = re.search(r"while \(16\|M0\)\s+(L\d+)", t[w]).group(1)
    s = next(i for i, l in enumerate(t) if l.startswith(label + ":"))
    return t[:s], t[s:w + 1], t[w + 1:]


def accumulators(prologue):
    """Zeroings after the last jump of the prologue: (hf list, f list)."""
    last = max([i for i, l in enumerate(prologue) if "jmpi" in l] + [0])
    h = [int(m.group(1)) for l in prologue[last:] for m in [re.search(r"mov \(16\|M0\)\s+r(\d+)\.0<1>:hf\s+0\.0:hf", l)] if m]
    f = [int(m.group(1)) for l in prologue[last:] for m in [re.search(r"mov \(16\|M0\)\s+r(\d+)\.0<1>:f\s+0\.0:f", l)] if m]
    return h, f


def analyze(loop, dtype):
    ptrs = sorted({int(m.group(1)) for l in loop
                  for m in [re.search(r"add \((?:8|1)\|M0\)\s+r(\d+)\.0<1>:q\s+r\1\.0<[^>]*>:q\s+\d+:w", l)] if m})
    usage = {}
    for l in loop:
        m = re.search(rf"mad \(16\|M0\)\s+r\d+\.0<1>:{dtype}\s+r\d+\.0<2;1>:{dtype}\s+r(\d+)\.0<2;1>:{dtype}\s+r(\d+)\.\d+<0>:{dtype}", l)
        if m:
            usage.setdefault(int(m.group(1)), "B")
            usage.setdefault(int(m.group(2)), "A")
    copies = [(int(m.group(1)), int(m.group(2))) for l in loop
             for m in [re.search(r"mov \(16\|M0\)\s+r(\d+)\.0<1>:w\s+r(\d+)\.0<16;16,1>:w", l)] if m]
    for _ in range(6):
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
            t = {usage[x] for x in range(r, r + 4) if x in usage}
            if base in ptrs and len(t) == 1:
                role.setdefault(base, set()).update(t)
    pa = [w for w, t in role.items() if t == {"A"}]
    pb = [w for w, t in role.items() if t == {"B"}]
    assert len(pa) == 1 and len(pb) == 1, ("pointer detection", ptrs, role)
    c = next(re.search(r"cmp \(16\|M0\)\s+\(lt\)f0\.0\s+null<1>:d\s+(r\d+\.\d)<0;1,0>:d\s+(r\d+\.\d)<0;1,0>:d", l)
             for l in loop if "(lt)f0.0" in l)
    return f"r{pa[0]}.0", f"r{pb[0]}.0", c.group(1), c.group(2), ptrs


def build(kind, prologue, body, epilogue, result):
    k = KINDS[kind]["kernel"]
    tmp = tempfile.mkdtemp()
    d = os.path.join(tmp, "d")
    shutil.copytree(kind_dir(kind), d)
    open(os.path.join(d, f".text.{k}.asm"), "w").write("\n".join(prologue + body + epilogue) + "\n")
    name = os.path.basename(result)
    subprocess.run(["ocloc", "asm", "-file", name, "-dump", d, "-device", "whl"], check=True, stdout=subprocess.DEVNULL)
    shutil.move(os.path.join(d, name), result)
    shutil.rmtree(tmp)


class Registers:
    def __init__(self, free_regs):
        self.w = list(free_regs)

    def block(self, size, parity=0):
        s = set(self.w)
        for b in range(2 + parity, 127, 2):
            if all(r in s for r in range(b, b + size)):
                for r in range(b, b + size):
                    self.w.remove(r)
                return b
        raise SystemExit(f"no block of {size} registers with parity {parity}")

    def one(self):
        return self.w.pop(0)


def free_regs(prologue, epilogue, taken):
    used = {int(x) for l in epilogue for x in re.findall(r"\br(\d+)", l.split("//")[0])}
    return [r for r in range(2, 127) if r not in used | taken | {0, 1, 127}]


def loads(H, cele):
    """targets: list of (destination register, header index); after each send header += 128 B (hardware reads the header at issue)."""
    return [x for dst, h in cele for x in (f"{W}send (16|M0)  r{dst}:w  r{H[h]}  {SEND}",
                                           f"{W}add (1|M0)  r{H[h]}.0<1>:q  r{H[h]}.0<0;1,0>:q  128:w")]


def advance(H, idx, amount):
    return [f"{W}add (1|M0)  r{H[i]}.0<1>:q  r{H[i]}.0<0;1,0>:q  {amount}:w" for i in idx]


def loop_end(counter, limit, step, label):
    return [f"{W}add (1|M0)  {counter}<1>:d  {counter}<0;1,0>:d  {step}:w",
            f"{W}cmp (16|M0)  (lt)f0.0  null<1>:d  {counter}<0;1,0>:d  {limit}<0;1,0>:d",
            f"(W&f0.0) jmpi  {label}"]


def loop_h(prologue, loop_k, epilogue, acc_map):
    """FP16 16×32: half-step h (kl 0..7) in buffer h: A2 8 GRF (pa + 256h), B0 8 GRF (pb + 256h), B1 8 GRF (pb + 512 + 256h).
    Step k16: pa += 512, pb += 1024."""
    pa, pb, counter, limit, ptrs = analyze(loop_k, "hf")
    acc = set(acc_map.values())
    busy = acc | {r for w in ptrs for r in (w, w + 1)} | {int(x) for x in re.findall(r"r(\d+)", counter + limit)}
    R = Registers(free_regs(prologue, epilogue, busy))
    buf = [(R.block(8), R.block(8), R.block(8)) for _ in range(2)]
    H = [R.one() for _ in range(6)]   # [buffer][A, B0, B1]
    out = [f"// --- gemm_gen h: pa={pa} pb={pb} counter={counter} limit={limit} buffers={buf} headers={H}"]
    for b in range(2):
        out += [f"{W}add (1|M0)  r{H[3 * b]}.0<1>:q  {pa}<0;1,0>:q  {256 * b}:w",
                f"{W}add (1|M0)  r{H[3 * b + 1]}.0<1>:q  {pb}<0;1,0>:q  {256 * b}:w",
                f"{W}add (1|M0)  r{H[3 * b + 2]}.0<1>:q  {pb}<0;1,0>:q  {512 + 256 * b}:w"]

    def load_q(b):
        A, B0, B1 = buf[b]
        o = loads(H, [(A, 3 * b), (A + 4, 3 * b), (B0, 3 * b + 1), (B0 + 4, 3 * b + 1), (B1, 3 * b + 2), (B1 + 4, 3 * b + 2)])
        # after 2 × 128 B: A to the next k16 +256, B +768 (skip the second half and the second column group)
        return o + advance(H, [3 * b], 256) + advance(H, [3 * b + 1, 3 * b + 2], 768)

    def mads(b):
        A, B0, B1 = buf[b]
        o = []
        for kl in range(8):
            for r in range(16):
                ga, sub = A + 2 * (r // 4) + (1 if kl % 2 == 0 else 0), (r % 4) * 4 + kl // 2
                for c, Bc in ((0, B0), (1, B1)):
                    a = acc_map[(r, c)]
                    o.append(f"        mad (16|M0)  r{a}.0<1>:hf  r{a}.0<2;1>:hf  r{Bc + kl}.0<2;1>:hf  r{ga}.{sub}<0>:hf")
        return o
    return out + load_q(0) + load_q(1) + ["LGEN:"] + mads(0) + load_q(0) + mads(1) + load_q(1) + loop_end(counter, limit, 16, "LGEN")


def loop_quarters(prologue, loop_k, epilogue, acc_map, dtype, kc=None, acc_map_f=None):
    """16×16, quarters of k (kl 0..3), alternating buffers, read-ahead by 2 quarters.
    hf (hd): A3 4 GRF + B 4 GRF; quarter = 128 B A + 128 B B. f (s): A2f 8 GRF + BE 4 + BO 4; quarter = 256 B + 256 B.
    hd: after kc products carry f += h (mixed add), h = 0."""
    pa, pb, counter, limit, ptrs = analyze(loop_k, dtype)
    acc = set(acc_map.values()) | ({r for x in acc_map.values() for r in (x, x + 1)} if dtype == "f" else set())
    if acc_map_f:
        acc |= {r for x in acc_map_f.values() for r in (x, x + 1)}
    busy = acc | {r for w in ptrs for r in (w, w + 1)} | {int(x) for x in re.findall(r"r(\d+)", counter + limit)}
    R = Registers(free_regs(prologue, epilogue, busy))
    if dtype == "hf":
        buf = [(R.block(4), R.block(4), None) for _ in range(2)]
        qw = 128
    else:
        buf = [(R.block(8), R.block(4, 0), R.block(4, 1)) for _ in range(2)]
        qw = 256
    H = [R.one() for _ in range(4)]   # [buffer][A, B]
    out = [f"// --- gemm_gen {dtype}: pa={pa} pb={pb} counter={counter} limit={limit} buffers={buf} headers={H}"]
    for b in range(2):
        out += [f"{W}add (1|M0)  r{H[2 * b]}.0<1>:q  {pa}<0;1,0>:q  {qw * b}:w",
                f"{W}add (1|M0)  r{H[2 * b + 1]}.0<1>:q  {pb}<0;1,0>:q  {qw * b}:w"]

    def load_q(b):
        A, B, BO = buf[b]
        if dtype == "hf":
            o = loads(H, [(A, 2 * b), (B, 2 * b + 1)])
        else:
            o = loads(H, [(A, 2 * b), (A + 4, 2 * b), (B, 2 * b + 1), (BO, 2 * b + 1)])
        return o + advance(H, [2 * b, 2 * b + 1], 2 * qw - (128 if dtype == "hf" else 256))

    def mads(b):
        A, B, BO = buf[b]
        o = []
        for kl in range(4):
            for r in range(16):
                a = acc_map[r]
                if dtype == "hf":
                    g, sub, breg = A + 2 * (r // 8) + (kl % 2 == 0), (r % 8) * 2 + kl // 2, B + kl
                else:
                    g, sub = A + 2 * (r // 4) + (kl % 2 == 0), (r % 4) * 2 + kl // 2
                    breg = (B if kl % 2 == 0 else BO) + 2 * (kl // 2)
                o.append(f"        mad (16|M0)  r{a}.0<1>:{dtype}  r{a}.0<2;1>:{dtype}  r{breg}.0<2;1>:{dtype}  r{g}.{sub}<0>:{dtype}")
        return o

    step = kc or 16
    out += load_q(0) + load_q(1) + ["LGEN:"]
    for q in range(step // 4):
        out += mads(q % 2) + load_q(q % 2)
    if kc:
        for r in range(16):
            h, f = acc_map[r], acc_map_f[r]
            out += [f"        add (8|M0)  r{f}.0<1>:f  r{f}.0<8;8,1>:f  r{h}.0<8;8,1>:hf",
                    f"        add (8|M8)  r{f + 1}.0<1>:f  r{f + 1}.0<8;8,1>:f  r{h}.8<8;8,1>:hf",
                    f"        mov (16|M0)  r{h}.0<1>:hf  0.0:hf"]
    return out + loop_end(counter, limit, step, "LGEN")


def load_map(kind):
    acc_map, acc_map_f = {}, {}
    for l in open(os.path.join(HERE, f"gen_{kind}_map.txt")):
        p = l.split()
        if kind == "h":
            acc_map[(int(p[1]), int(p[2]))] = int(p[0])
        elif p[0] == "f":
            acc_map_f[int(p[2])] = int(p[1])
        else:
            acc_map[int(p[2])] = int(p[1])
    return acc_map, acc_map_f


if __name__ == "__main__":
    kind, mode = sys.argv[1], sys.argv[2]
    result = sys.argv[3] if len(sys.argv) > 3 else None
    prologue, loop_k, epilogue = load(kind)
    h, f = accumulators(prologue)
    dtype = KINDS[kind]["dtype"]
    if mode == "analyze":
        print(analyze(loop_k, dtype), "hf:", h, "f:", f)
    elif mode.startswith("map"):
        # map (h, s) / map_f / map_h (hd): accumulator no. i := i; the C write will show which register is which row
        target, t = (f, "f") if mode == "map_f" or (mode == "map" and kind == "s") else (h, "hf")
        build(kind, prologue, ["LMAP:"] + [f"{W}mov (16|M0)  r{a}.0<1>:{t}  {float(i)}:{t}" for i, a in enumerate(target)],
               epilogue, result)
    else:
        idx_map, idx_map_f = load_map(kind)
        if kind == "h":
            acc_map = {k: h[v] for k, v in idx_map.items()}
            body = loop_h(prologue, loop_k, epilogue, acc_map)
        elif kind == "hd":
            body = loop_quarters(prologue, loop_k, epilogue, {k: h[v] for k, v in idx_map.items()}, "hf", kc=32,
                                 acc_map_f={k: f[v] for k, v in idx_map_f.items()})
        else:
            body = loop_quarters(prologue, loop_k, epilogue, {k: f[v] for k, v in idx_map.items()}, "f")
        build(kind, prologue, body, epilogue, result)
        print(body[0])
        print(f"instructions: {len(body)}, mad: {sum('mad (' in l for l in body)}")
