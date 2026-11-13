"""Small parser for Gen9 assembly as printed by `ocloc disasm` (IGA syntax): operands and the GRF registers they touch.
Used by attn_gen.py. Only what the attention generator needs: exec size, operand regions, send payload sizes."""
import re

TS = {"b": 1, "ub": 1, "w": 2, "uw": 2, "hf": 2, "d": 4, "ud": 4, "f": 4, "q": 8, "uq": 8, "df": 8}
OPND = re.compile(r"(-?)(r\d+|null|acc\d+)(?:\.(\d+))?(<[^>]*>)?:(\w+)")


def code(line):
    return line.split("//")[0]


def exec_size(line):
    m = re.search(r"\((\d+)\|M\d+\)", line)
    return int(m.group(1)) if m else 1


def opcode(line):
    m = re.match(r"^\s*(?:\([^)]*\))?\s*([a-z][a-z0-9.]*)\s", line)
    return m.group(1) if m else None


def operands(line):
    """[(neg, reg, sub, region, type)] in order dst, src0, src1, src2 (regular instructions)."""
    c = code(line)
    c = re.sub(r"^\s*\([^)]*\)\s*", " ", c)            # predicate
    c = re.sub(r"\((\d+)\|M\d+\)", " ", c, count=1)     # exec size
    c = re.sub(r"\([a-z]+\)f\d\.\d", " ", c)            # conditional modifier
    return [(m.group(1), m.group(2), int(m.group(3) or 0), m.group(4) or "", m.group(5)) for m in OPND.finditer(c)]


def span(op, n):
    """GRF registers touched by operand op at exec size n."""
    _, reg, sub, region, t = op
    if not reg.startswith("r"):
        return []
    r = int(reg[1:])
    ts = TS[t]
    start = r * 32 + sub * ts
    if region.startswith("<0") or (region == "" and n == 1):
        size = ts
    else:
        hs = 1
        m = re.match(r"<(\d+)>$", region)
        if m:
            hs = max(1, int(m.group(1)))
        size = n * ts * hs
    return list(range(start // 32, (start + size - 1) // 32 + 1))


def send_regs(line):
    """(dst regs, src regs) of a send/sends: rd:N GRFs from dst, wr:M(+M2) from src0 (and src1)."""
    m = re.search(r"sends? \(\d+\|M\d+\)\s+(r\d+|null)(?::\w+)?\s+r(\d+)(?:\s+r(\d+))?", line)
    rd = re.search(r"rd:(\d+)", line)
    wr = re.search(r"wr:(\d+)h?(?:\+(\d+))?", line)
    d0 = None if m.group(1) == "null" else int(m.group(1)[1:])
    dst = [] if d0 is None else list(range(d0, d0 + (int(rd.group(1)) if rd else 1)))
    n0 = int(wr.group(1)) if wr else 1
    src = list(range(int(m.group(2)), int(m.group(2)) + n0))
    if m.group(3) and wr and wr.group(2):
        src += list(range(int(m.group(3)), int(m.group(3)) + int(wr.group(2))))
    return dst, src


def defs_uses(line):
    """(registers written, registers read)."""
    op = opcode(line)
    if op is None:
        return [], []
    if op.startswith("send"):
        return send_regs(line)
    ops = operands(line)
    n = exec_size(line)
    if not ops:
        return [], []
    w = span(ops[0], n)
    r = [x for o in ops[1:] for x in span(o, n)]
    if op in ("cmp",) and ops[0][1] == "null":
        w = []
    return w, r


def used_regs(lines):
    s = set()
    for l in lines:
        w, r = defs_uses(l)
        s |= set(w) | set(r)
    return s
