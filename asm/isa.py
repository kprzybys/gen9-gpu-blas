"""Small parser for Gen9 assembly as printed by `ocloc disasm` (IGA syntax): operands and the GRF registers they touch.
Used by attn_gen.py. Only what the attention generator needs: exec size, operand regions, send payload sizes.
check() / check_all() reject every line form this parser does not understand (unknown opcode, indirect or
accumulator operands, unknown regions, sends without the rd:/wr: decoding), so that a register analysis built on
defs_uses() fails loudly instead of silently missing a read or a write (e.g. after an IGC/NEO update)."""
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
    m3 = re.match(r"<(\d+);(\d+),(\d+)>$", region)
    if region.startswith("<0") or (region == "" and n == 1):
        size = ts
    elif m3 and int(m3.group(2)) > 0:                   # <vs;w,hs>: n/w rows of w elements
        vs, w, hs = (int(x) for x in m3.groups())
        size = ((max(n, w) // w) - 1) * vs * ts + (w - 1) * hs * ts + ts
    else:                                               # dst <hs>, ternary <vs;hs> (contiguous here), none
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


# opcode -> number of register/immediate operands (dst + sources) that check() accepts
ARITY = {"mov": 2, "add": 3, "mul": 3, "sel": 3, "cmp": 3, "shl": 3, "shr": 3, "asr": 3, "and": 3, "or": 3,
         "mad": 4, "math.exp": 2, "math.inv": 2}
NOREG = {"jmpi", "while", "break", "illegal", "nop"}     # control flow / padding: no GRF operand allowed
# regions whose GRF footprint span() computes exactly (contiguous rows or scalars)
REGION = re.compile(r"<(?:0|1|2|0;0|2;1|\d+;[1-9]\d*,\d+)>$")
IMM = re.compile(r"(-?(?:\d+(?:\.\d*)?(?:e[+-]?\d+)?|0x[0-9a-fA-F]+|inf|nan)):(?:b|ub|w|uw|d|ud|q|uq|f|hf|df|v|uv|vf)$")
LABEL = re.compile(r"^[A-Za-z_]\w*:$")


def full_writes(line):
    """GRF registers this line overwrites completely and unconditionally (for liveness: only these end a live
    range). Predicated writes, writes of part of a register ((1|M0) :q headers, .2:ud fields) and writes of a
    multi-register span's partial ends are not kills. Non-(W) instructions count as full writes: the kernels run
    with full SIMD8 subgroups and the analysed regions are straight-line (no divergent branch around them)."""
    op = opcode(line)
    if op is None:
        return []
    if op.startswith("send"):
        return send_regs(line)[0]
    pred = re.match(r"^\s*\(([^)]*)\)", code(line))
    if pred and re.search(r"f\d", pred.group(1)):
        return []
    ops = operands(line)
    if not ops or not ops[0][1].startswith("r") or (op == "cmp" and ops[0][1] == "null"):
        return []
    _, reg, sub, region, t = ops[0]
    hs = 1
    m = re.match(r"<(\d+)>$", region)
    if m:
        hs = int(m.group(1))
    if hs != 1:
        return []
    start = int(reg[1:]) * 32 + sub * TS[t]
    end = start + exec_size(line) * TS[t]               # exclusive
    return [r for r in range(start // 32, (end - 1) // 32 + 1) if r * 32 >= start and r * 32 + 32 <= end]


def check(line):
    """None if the line is a form defs_uses() handles exactly, else the reason (a string)."""
    c = code(line).strip()
    if not c or LABEL.match(c):
        return None
    op = opcode(line)
    if op is None:
        return "no opcode"
    if "[" in c or re.search(r"\ba\d+\.", c):
        return "indirect addressing"
    if op in NOREG:
        if re.search(r"\br\d+", c):
            return f"{op} with a GRF operand"
        if op == "jmpi" and not re.search(r"jmpi\s+[A-Za-z_]\w*$", c):
            return "jmpi to a non-label"
        return None
    if op in ("send", "sends"):
        if not re.search(r"sends? \(\d+\|M\d+\)\s+(r\d+|null)(?::\w+)?\s+r\d+(?:\s+r\d+)?\s", line):
            return "send operands not understood"
        if not re.search(r"rd:\d+", line) or not re.search(r"wr:\d+h?(?:\+\d+)?", line):
            return "send without rd:/wr: decoding"
        if op == "sends" and not re.search(r"wr:\d+h?\+\d+", line):
            return "sends without src1 length"
        return None
    if op not in ARITY:
        return f"opcode {op} not known to isa.py"
    if not re.search(r"\((\d+)\|M\d+\)", c):
        return "no exec size"
    body = re.sub(r"^\s*\([^)]*\)\s*", " ", c)
    body = re.sub(r"^\s*" + re.escape(op), " ", body)
    body = re.sub(r"\((\d+)\|M\d+\)", " ", body, count=1)
    body = re.sub(r"\((?:ge|gt|lt|le|eq|ne)\)f\d\.\d", " ", body)
    body = re.sub(r"\{[^}]*\}", " ", body)
    toks = body.split()
    if len(toks) != ARITY[op]:
        return f"{len(toks)} operands, expected {ARITY[op]}"
    for t in toks:
        m = OPND.fullmatch(t)
        if m:
            if m.group(2).startswith("acc"):
                return "accumulator operand"
            if m.group(5) not in TS:
                return f"type {m.group(5)}"
            if m.group(4) and not REGION.match(m.group(4)):
                return f"region {m.group(4)}"
        elif not IMM.match(t):
            return f"operand {t} not understood"
    if OPND.fullmatch(toks[0]) is None:
        return "destination is not a register"
    return None


def check_all(lines, where):
    """Exit with the offending lines if any line in a region the generator analyses is not understood."""
    bad = [(i, l, r) for i, l in enumerate(lines) for r in [check(l)] if r]
    if bad:
        raise SystemExit(f"isa.py does not understand {len(bad)} line(s) in {where} (refusing to analyse):\n" +
                         "\n".join(f"  {i}: {r}: {l.strip()}" for i, l, r in bad[:20]))


def used_regs(lines):
    s = set()
    for l in lines:
        w, r = defs_uses(l)
        s |= set(w) | set(r)
    return s
