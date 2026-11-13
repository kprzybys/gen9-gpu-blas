"""Symbolic equivalence check of a generated GEMM loop against IGC's loop (docs/solvers.md, report "Solvery w tworzeniu
kerneli GPU", recommendation 4).

Both loops run in a symbolic interpreter over the register file (byte level), starting from the same symbolic
registers at the loop entry (the prologue is IGC's in both kernels and does no loads). Memory reads become symbols
M(base, byte offset), where base is the symbolic pointer value; arithmetic builds expression trees in program order.
The loop counter and limit are concrete (counter 0, limit K_TOTAL), so every branch is decided. At the end the
accumulator registers must hold identical expressions: same products of the same memory elements, accumulated in the
same order (= bit-identical results). One normalisation: mad(+0, b, a) == mul(b, a) (they differ only in the sign of a
zero, which cannot change a sum that starts at +0; see the comment in gemm_gen.loop_quarters).

Assumptions (stated, not hidden): ternary operand regions <2;1> / <1> are contiguous and <0> is a scalar, as the
generators and IGC's code use them; every instruction is executed for all exec-size lanes (no masked lanes inside the
loop); a send's lengths come from isa.send_regs (comment or descriptor). Any instruction form the interpreter does not
know stops the check instead of being skipped.
Usage: python3 -I loopcheck.py {h|hd|s|all} [--mutate]
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import gemm_gen as G  # noqa: E402
import isa  # noqa: E402

K_TOTAL = 64
IMM = re.compile(r"\s(-?(?:0x[0-9a-fA-F]+|\d+(?:\.\d*)?(?:e[+-]?\d+)?)):(\w+)\s*(?:\{|$)")
FLOAT = {"f", "hf", "df"}


class Unsupported(Exception):
    pass


class Machine:
    def __init__(self, counter, limit, attn=False, k_total=None):
        self.b = {}                     # absolute byte -> (value, k)
        self.flags = {}
        self.attn = attn
        cr, cs = self.loc(counter)
        lr, ls = self.loc(limit)
        self.write(cr * 32 + cs * 4, 4, ("C", 0))
        self.write(lr * 32 + ls * 4, 4, ("C", K_TOTAL if k_total is None else k_total))

    @staticmethod
    def loc(s):
        m = re.match(r"r(\d+)\.(\d+)", s)
        return int(m.group(1)), int(m.group(2))

    def write(self, a, size, val):
        for k in range(size):
            self.b[a + k] = (val, k)

    def read(self, a, size):
        parts = [self.b.get(a + k, (("Ib", a + k), 0)) for k in range(size)]
        v0, k0 = parts[0]
        if k0 == 0 and all(p == (v0, k) for k, p in enumerate(parts)) and size >= self.size_of(v0):
            return v0
        if all(p[0][0] == "M" and p[1] == 0 for p in parts):
            b0, o0 = parts[0][0][1], parts[0][0][2]
            if all(p[0][1] == b0 and p[0][2] == o0 + k for k, p in enumerate(parts)):
                return ("MV", b0, o0, size)
        if all(p[0][0] == "Ib" for p in parts) and all(p[0][1] == parts[0][0][1] + k for k, p in enumerate(parts)):
            return ("I", parts[0][0][1], size)
        return ("CAT", tuple(parts))

    @staticmethod
    def size_of(v):
        return 1 if v[0] in ("M", "Ib") else 0

    def copy(self, dst, src, size):
        for k in range(size):
            self.b[dst + k] = self.b.get(src + k, (("Ib", src + k), 0))


def elem_addr(op, i, n, dst=False):
    _, reg, sub, region, t = op
    r = int(reg[1:])
    ts = isa.TS[t]
    m3 = re.match(r"<(\d+);(\d+),(\d+)>$", region)
    if region in ("<0>", "<0;1,0>") or region.startswith("<0;"):
        off = 0
    elif m3:
        vs, w, hs = (int(x) for x in m3.groups())
        off = (i // w) * vs + (i % w) * hs
    elif region in ("<1>", "<2;1>", "", "<1;1>"):
        off = i
    elif re.match(r"<(\d+)>$", region):
        off = i * int(region[1:-1])
    else:
        raise Unsupported(f"region {region}")
    return r * 32 + (sub + off) * ts, ts


def imm_of(line):
    m = IMM.search(isa.code(line) + " ")
    if not m:
        return None
    s, t = m.group(1), m.group(2)
    v = float(s) if t in FLOAT else int(s, 0)
    return v, t


def norm_addr(v, imm):
    if v[0] == "addr":
        return ("addr", v[1], v[2] + imm)
    return ("addr", v, imm)


def run(lines, m, max_steps=200000):
    labels = {isa.code(l).strip()[:-1]: i for i, l in enumerate(lines) if isa.LABEL.match(isa.code(l).strip())}
    pc, steps = 0, 0
    loop_start = 0
    while pc < len(lines):
        steps += 1
        if steps > max_steps:
            raise Unsupported("too many steps (loop does not terminate under the concrete counter)")
        l = lines[pc]
        c = isa.code(l).strip()
        op = isa.opcode(l)
        pc += 1
        if not c or isa.LABEL.match(c) or op is None:
            continue
        pred = re.match(r"^\((W)?&?(~)?(f\d\.\d)?\)", c)
        flag = None
        if pred and pred.group(3):
            flag = m.flags.get(pred.group(3))
            if flag is None:
                if not m.attn:
                    raise Unsupported(f"unknown flag {pred.group(3)}: {c}")
                flag = "unknown"            # data-dependent (attention softmax): jumps not taken, other ops executed
            elif pred.group(2):
                flag = not flag
        if op in ("jmpi", "while", "break"):
            target = re.findall(r"\b(L\w+)\b", c)
            taken = True if flag is None else (False if flag == "unknown" else flag)
            if op == "while":
                if taken:
                    pc = labels[target[-1]]
                continue
            if op == "break":
                if taken:
                    return
                continue
            if taken:
                if target[0] not in labels:
                    raise Unsupported(f"jump outside the region: {c}")
                pc = labels[target[0]]
            continue
        if flag is not None and not m.attn:
            raise Unsupported(f"predicated {op}: {c}")
        n = isa.exec_size(l)
        ops = isa.operands(l)
        if op.startswith("send"):
            dst, src = isa.send_regs(l)
            hv = m.read(src[0] * 32, 8)
            base, off = (hv[1], hv[2]) if hv[0] == "addr" else (hv, 0)
            for j in range(len(dst) * 32):
                m.b[dst[0] * 32 + j] = (("M", base, off + j), 0)
            continue
        if op == "cmp":
            cm = re.search(r"\((lt|eq|ne|gt|ge|le)\)(f\d\.\d)", c)
            a, ts = elem_addr(ops[1], 0, n)
            x = m.read(a, ts)
            im = imm_of(l)
            if im is not None:
                y = ("C", im[0])
            else:
                a2, _ = elem_addr(ops[2], 0, n)
                y = m.read(a2, ts)
            if x[0] != "C" or y[0] != "C":
                if not m.attn:
                    raise Unsupported(f"cmp on a symbolic value: {c}")
                m.flags[cm.group(2)] = None
                continue
            r = {"lt": x[1] < y[1], "eq": x[1] == y[1], "ne": x[1] != y[1], "gt": x[1] > y[1],
                 "ge": x[1] >= y[1], "le": x[1] <= y[1]}[cm.group(1)]
            m.flags[cm.group(2)] = r
            continue
        dsto = ops[0]
        dt = dsto[4]
        im = imm_of(l)
        outs = []
        for i in range(n):
            da, dts = elem_addr(dsto, i, n, dst=True)
            srcv = []
            for so in ops[1:]:
                sa, sts = elem_addr(so, i, n)
                v = m.read(sa, sts)
                if so[0] == "-":
                    v = ("neg", v)
                srcv.append((v, so[4], sa, sts))
            if m.attn and (dt in FLOAT or op in ("mad", "sel") or op.startswith("math")) and \
                    not (op == "mov" and isa.TS[(srcv[0][1] if srcv else dt)] == dts and srcv and srcv[0][1] == dt):
                outs.append(("w", da, dts, attn_value(op, srcv, ops[1:], im)))
                continue
            if op == "mov":
                if im is not None:
                    outs.append(("w", da, dts, ("const", dt, float(im[0]) if dt in FLOAT else im[0])))
                elif isa.TS[srcv[0][1]] == dts and (srcv[0][1] == dt or (dt not in FLOAT and srcv[0][1] not in FLOAT)):
                    outs.append(("c", da, srcv[0][2], dts))
                else:
                    outs.append(("w", da, dts, ("cvt", dt, srcv[0][0])))
            elif op == "add":
                if im is not None:
                    v = srcv[0][0]
                    if v[0] == "C":
                        outs.append(("w", da, dts, ("C", v[1] + im[0])))
                    elif dt in ("q", "uq"):
                        outs.append(("w", da, dts, norm_addr(v, im[0])))
                    else:
                        outs.append(("w", da, dts, ("add", dt, v, ("const", im[1], im[0]))))
                else:
                    a, b = srcv[0][0], srcv[1][0]
                    if srcv[1][1] != dt:
                        b = ("cvt", dt, b)
                    if srcv[0][1] != dt:
                        a = ("cvt", dt, a)
                    outs.append(("w", da, dts, ("add", dt, a, b)))
            elif op == "and":
                v = srcv[0][0]
                if v[0] != "C" or im is None:
                    raise Unsupported(f"and on a symbolic value: {c}")
                outs.append(("w", da, dts, ("C", v[1] & im[0])))
            elif op == "mad":
                cc, bb, aa = srcv[0][0], srcv[1][0], srcv[2][0]
                if cc[0] == "const" and cc[2] == 0.0:
                    outs.append(("w", da, dts, ("mul", dt, bb, aa)))
                else:
                    outs.append(("w", da, dts, ("mad", dt, cc, bb, aa)))
            elif op == "mul":
                outs.append(("w", da, dts, ("mul", dt, srcv[0][0], srcv[1][0])))
            else:
                raise Unsupported(f"opcode {op}: {c}")
        if flag == "unknown":           # predicated on data (attention): merge, keeping the value that carries identity
            keep = []
            for o in outs:
                if o[0] == "w":
                    old_v = m.read(o[1], o[2])
                    if old_v[0] in ("chain", "id") and o[3][0] not in ("chain", "id"):
                        continue
                keep.append(o)
            outs = keep
        for o in outs:                  # all lanes read before any lane writes
            if o[0] == "w":
                m.write(o[1], o[2], o[3])
        for o in outs:
            if o[0] == "c":
                pass
        # copies: read all sources first
        cps = [(o[1], [m.b.get(o[2] + k, (("Ib", o[2] + k), 0)) for k in range(o[3])]) for o in outs if o[0] == "c"]
        for da, parts in cps:
            for k, p in enumerate(parts):
                m.b[da + k] = p


# ---- attention mode: products as multisets, identity through softmax (see verify_attn) ----
def ident(v):
    if v[0] == "chain":
        return ("S", tuple(sorted(v[1], key=repr)))
    if v[0] == "id":
        return v[1]
    return v


def scalar_region(region):
    return region in ("<0>", "<0;1,0>") or region.startswith("<0;")


def attn_value(op, srcv, src_ops, im):
    vals = [s[0] for s in srcv]
    if op == "mad":
        c, b, a = vals
        items = c[1] if c[0] == "chain" else ()
        return ("chain", items + ((ident(b), ident(a)),))
    if op == "mov" and im is not None:
        return ("const",)
    cands = [v for v, so in zip(vals, src_ops) if not scalar_region(so[3])] or vals
    for v in cands:
        if v[0] in ("chain", "id"):
            return v                    # exp(S - max), O * alpha, copies of p: keep the identity of the vector operand
    for v in cands:
        if v[0] in ("MV", "I"):
            return ("id", v)
    return ("other",)


def verify_attn(opts, NS, mode="sched", iterations=2, mutate=None, src=None):
    """Attention mode: generated loop (attn_gen sched<NS>/remap) vs IGC's loop. For every O element the multiset of
    (V element, identity of p) must match, where identity of p = multiset of (K element, Q element) of its score.
    Checks the mapping of both mad phases (which register, lane, address meets which); does not check the softmax
    arithmetic or the order (the generated loop uses another order by design)."""
    import tempfile
    import attn_gen as A
    d = tempfile.mkdtemp()
    dd = os.path.join(d, "d")
    A.disassemble(src or os.path.join(HERE, "attn.cl"), opts, dd)
    t = open(os.path.join(dd, f".text.{A.KERNEL}.asm")).read().splitlines()
    shutil_rm(d)
    pro, loop, epi = A.split_loop(t)
    P = A.params(opts)
    if mode == "remap":
        raise Unsupported("remap (variant R on bench/attn.cl) is not covered: igc_maps does not accept that loop form")
    npro, nloop, nepi, _ = A.sched(list(pro), list(loop), list(epi), P, NS)
    if mutate:
        nloop = mutate(list(nloop))
    return verify_attn_parts(pro, loop, epi, npro, nloop, P, iterations)


def verify_attn_parts(pro, loop, epi, npro, nloop, P, iterations=2):
    """verify_attn on already split code (attn_gen.main calls this before assembling)."""
    import collections
    import attn_gen as A
    M = A.igc_maps(pro, loop, epi, P)
    extra = npro[len(pro):] if npro[:len(pro)] == pro else None
    if extra is None:
        raise Unsupported("generated prologue is not IGC's prologue plus additions")
    kt = iterations * 8 * P["NK"]
    mi = Machine(M["cnt"], M["lim"], attn=True, k_total=kt)
    mg = Machine(M["cnt"], M["lim"], attn=True, k_total=kt)
    run(loop, mi)
    run(extra + nloop, mg)
    bad, lost = [], 0
    for r in sorted(set(M["o"].values())):
        for i in range(8):
            vi, vg = mi.read(r * 32 + i * 4, 4), mg.read(r * 32 + i * 4, 4)
            ci = collections.Counter(vi[1]) if vi[0] == "chain" else None
            cg = collections.Counter(vg[1]) if vg[0] == "chain" else None
            if ci is None or cg is None:
                bad.append((r, i, "O is not a product chain"))
                continue
            lost += sum(1 for it in vg[1] if it[1] == ("other",))
            if ci != cg:
                bad.append((r, i, f"{sum((ci - cg).values())} products only in IGC, {sum((cg - ci).values())} only generated"))
    n = len(set(M["o"].values())) * 8
    if bad:
        return False, f"{len(bad)} of {n} O elements differ; first r{bad[0][0]} lane {bad[0][1]}: {bad[0][2]}"
    if lost:
        return False, f"{lost} products whose p lost its score identity (propagation rule too weak)"
    per = len(mi.read(sorted(set(M['o'].values()))[0] * 32, 4)[1])
    return True, (f"{n} O elements: same (V, p) products as IGC's loop, p traced to the same (K, Q) products; "
                  f"{per} products per element over {iterations} iterations (order not compared)")


def shutil_rm(d):
    import shutil
    shutil.rmtree(d, ignore_errors=True)


def acc_values(m, h, f):
    out = {}
    for r in h:
        for i in range(16):
            out[(r, "hf", i)] = m.read(r * 32 + i * 2, 2)
    for r in f:
        for i in range(16):
            out[(r, "f", i)] = m.read(r * 32 + i * 4, 4)
    return out


def short(v, n=160):
    s = repr(v)
    return s if len(s) <= n else s[:n] + "..."


def isomorphic(vi, vg):
    """Same expression shape and order with a consistent bijection between memory elements (IGC's template may assume
    another packed-data layout than the generated loop; the generator's own packing is then the reference layout).
    Returns (ok, message, mapping)."""
    fwd, bwd = {}, {}

    def walk(a, b):
        if type(a) is not tuple or type(b) is not tuple:
            return None if a == b else f"different values {short(a, 80)} / {short(b, 80)}"
        if a[0] == "MV" or b[0] == "MV":
            if a[0] != "MV" or b[0] != "MV" or a[3] != b[3]:
                return f"memory element vs other value: {short(a, 80)} / {short(b, 80)}"
            ka, kb = (a[1], a[2]), (b[1], b[2])
            if fwd.setdefault(ka, kb) != kb or bwd.setdefault(kb, ka) != ka:
                return f"inconsistent mapping: IGC {ka} -> generated {fwd[ka]} and {kb}"
            return None
        if type(a) is not tuple or type(b) is not tuple or len(a) != len(b):
            return None if a == b else f"different values {short(a, 80)} / {short(b, 80)}"
        if a[0] != b[0]:
            return f"different operations {a[0]} / {b[0]}"
        for x, y in zip(a[1:], b[1:]):
            e = walk(x, y)
            if e:
                return e
        return None
    for k in vi:
        e = walk(vi[k], vg[k])
        if e:
            return False, f"r{k[0]} {k[1]} lane {k[2]}: {e}", fwd
    return True, "", fwd


def check(kind, mutate=False):
    pro, loop_k, epi = G.load(kind)
    h, f = G.accumulators(pro)
    idx_map, idx_map_f = G.load_map(kind)
    body = G.gen_body(kind, pro, loop_k, epi, h, f, idx_map, idx_map_f)
    return verify(kind, loop_k, h, f, body, mutate)


def verify(kind, loop_k, h, f, body, mutate=False):
    """(ok, message) for a generated body against IGC's loop_k; used by gemm_gen before it assembles a binary."""
    pa, pb, counter, limit, ptrs = G.analyze(loop_k, G.KINDS[kind]["dtype"])
    if mutate:                       # self-test: swap the B operand register of one mad -> must be detected
        i = next(i for i, l in enumerate(body) if " mad (" in l)
        body = list(body)
        body[i] = re.sub(r"(\s+r)(\d+)(\.\d+<2;1>:\w+\s+r\d+\.\d+<0>)", lambda x: f"{x.group(1)}{int(x.group(2)) + 2}{x.group(3)}",
                         body[i], count=1)
    mi, mg = Machine(counter, limit), Machine(counter, limit)
    for mm in (mi, mg):                 # the prologue zeroes the accumulators (gemm_gen.accumulators finds the movs)
        for r in h:
            for i in range(16):
                mm.write(r * 32 + i * 2, 2, ("const", "hf", 0.0))
        for r in f:
            for i in range(16):
                mm.write(r * 32 + i * 4, 4, ("const", "f", 0.0))
    run(loop_k, mi)
    run(body, mg)
    vi, vg = acc_values(mi, h, f), acc_values(mg, h, f)
    bad = [k for k in vi if vi[k] != vg[k]]
    nterm = sum(1 for k in vi if vi[k][0] in ("mad", "mul", "add"))
    if not bad:
        return True, f"{kind}: {len(vi)} accumulator elements identical to IGC's loop ({nterm} non-trivial), K = {K_TOTAL}"
    ok, msg, fwd = isomorphic(vi, vg)
    if ok:
        return True, (f"{kind}: identical to IGC's loop up to a consistent relabelling of {len(fwd)} memory elements "
                      f"(the template assumes another packed layout), same order, K = {K_TOTAL}")
    k = bad[0]
    return False, (f"{kind}: {len(bad)} of {len(vi)} accumulator elements differ and no consistent relabelling exists: {msg}\n"
                   f"   IGC:       {short(vi[k])}\n   generated: {short(vg[k])}")


def main():
    if sys.argv[1] == "attn":       # attn <sched<NS>|remap> "<build options>" [template.cl]
        mode, opts = sys.argv[2], sys.argv[3]
        src = sys.argv[4] if len(sys.argv) > 4 else None
        ok, msg = verify_attn(opts, int(mode[5:] or 3) if mode.startswith("sched") else 0,
                              "remap" if mode == "remap" else "sched", src=src)
        print(("OK   " if ok else "FAIL ") + msg)
        sys.exit(0 if ok else 1)
    kinds = ["h", "hd", "s"] if sys.argv[1] == "all" else [sys.argv[1]]
    mutate = "--mutate" in sys.argv
    ok_all = True
    for kind in kinds:
        try:
            ok, msg = check(kind, mutate)
        except Unsupported as e:
            ok, msg = False, f"{kind}: cannot check: {e}"
        ok_all &= ok
        print(("OK   " if ok else "FAIL ") + msg)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
