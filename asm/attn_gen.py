"""Hand-patched Gen9 assembly for the fused attention kernel (docs/attention-gpu.md, stage 2b).

Workflow as for the GEMMs (gemm_gen.py): OpenCL C template -> ocloc (whl) -> disassembly -> rewrite of the hot loop
-> ocloc asm -> .bin for clCreateProgramWithBinary (bench/attn.c with CL=<file>.bin).

Modes (variant names as in results/attn-asm-hd530-2026-11-13.txt):

Variant R ("remap", the cheap step): keep IGC's instruction stream for variant A of bench/attn.cl and only change
operands so that no `mad` has src1 and src2 in the same GRF bank (register-number parity):
  - S = Q K^T mads: src1 = K (packed half, 4 GRF per block read, parity alternates every 2 dims), src2 = scalar
    q[r][d] from the lane-split Q registers (loop invariant). Before the loop the Q elements are permuted between
    Q registers of opposite parity (swaps of single floats through a temporary register), so that every element
    sits in a register whose parity differs from the K register it meets; the mads then read the new location.
  - P V mads: src1 = V (same 4-GRF blocks), src2 = scalar p from the exp'd score registers (written in the loop).
    Where p and V would conflict, the mad reads a copy of the p register made in a register of the other parity
    (one `mov (8)` per score register and key block), taken from the registers unused in and after the loop.
The rest (addresses, softmax, branches, prologue, epilogue) is IGC's code unchanged.

Variants S/C ("sched<NS>", template asm/attn.cl): the whole loop body is generated (see sched()); IGC's prologue
and epilogue are kept, and IGC's own loop is only read by a symbolic pass (igc_maps) that finds the registers of
Q[r][c], O[r][c], the K/V pointers, the l sums and the loop counter. Best: sched4 (Qwen), sched2 (Llama).

Usage: python3 -I attn_gen.py remap|plain|poison|sched<NS> <src.cl> "<build options>" <out.bin> [--asm out.asm]
       (plain: IGC's code unchanged, without .spv; poison: one S mad negated, results must be wrong)
       env DEVICE=skl for HD 530 (desktop), default whl (UHD 620)
       python3 -I attn_gen.py analyze <src.cl> "<build options>"
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import isa  # noqa: E402

W = "(W)     "
KERNEL = "k_attn"
# ocloc device: whl = UHD 620 (laptop), skl = HD 530 (desktop); a binary for one is CL_INVALID_BINARY on the other.
DEVICE = os.environ.get("DEVICE", "whl")
# PE_IN_S=1: exp in place, the score registers serve as the even p copies (saves RQ*NK registers)
PE_IN_S = os.environ.get("PE_IN_S", "0") == "1"
# COMPACT=1: l sums in IGC's l registers and row maxima in a temporary's spare lanes (3 registers fewer for Qwen)
COMPACT = os.environ.get("COMPACT", "1") == "1"


def disassemble(src, options, d):
    """OpenCL C -> whl binary -> dump dir d (without .spv, so the driver must use our ISA)."""
    tmp = tempfile.mkdtemp()
    subprocess.run(["ocloc", "compile", "-q", "-file", src, "-device", DEVICE, "-options", options, "-output",
                    os.path.join(tmp, "k")], check=True, stdout=subprocess.DEVNULL)
    b = next(os.path.join(tmp, f) for f in os.listdir(tmp) if f.endswith(".bin"))
    subprocess.run(["ocloc", "disasm", "-file", b, "-dump", d, "-device", DEVICE], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.rmtree(tmp)
    if os.path.exists(os.path.join(d, ".spv")):
        os.remove(os.path.join(d, ".spv"))
    st = os.path.join(d, "sections.txt")
    keep = [l for l in open(st) if not l.startswith(".spv,")]
    open(st, "w").writelines(keep)


def assemble(d, text, out):
    open(os.path.join(d, f".text.{KERNEL}.asm"), "w").write("\n".join(text) + "\n")
    name = os.path.basename(out)
    r = subprocess.run(["ocloc", "asm", "-file", name, "-dump", d, "-device", DEVICE], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"ocloc asm failed ({r.returncode}):\n{r.stdout[-3000:]}{r.stderr[-3000:]}")
    shutil.move(os.path.join(d, name), out)


def split_loop(t):
    """(prologue, loop incl. label and while, epilogue) around the main `while` loop (the one with mads)."""
    for w, l in enumerate(t):
        m = re.search(r"^\s+while \(\d+\|M0\)\s+(L\d+)", l)
        if m:
            s = next(i for i, x in enumerate(t) if x.startswith(m.group(1) + ":"))
            if sum("mad (" in x for x in t[s:w]) > 100:
                return t[:s], t[s:w + 1], t[w + 1:]
    raise SystemExit("no hot loop found")


MAD = re.compile(r"^(\s+mad \(8\|M0\)\s+)(r\d+\.\d+<1>:f)(\s+)(\S+)(\s+)(r(\d+)\.(\d+)<2;1>:hf)(\s+)r(\d+)\.(\d+)<0>:f(.*)$")


def conflicts(loop):
    n = c = 0
    for l in loop:
        m = MAD.match(l)
        if m:
            n += 1
            c += int(m.group(7)) % 2 == int(m.group(10)) % 2
    return n, c


def remap(prologue, loop, epilogue):
    written = set()
    for l in loop:
        written |= set(isa.defs_uses(l)[0])
    free = sorted(set(range(1, 127)) - isa.used_regs(loop + epilogue))   # r0 kept (thread header)
    # --- S mads: Q element -> required parity
    need = {}
    for l in loop:
        m = MAD.match(l)
        if m and int(m.group(10)) not in written:
            k = (int(m.group(10)), int(m.group(11)))
            need.setdefault(k, set()).add(1 - int(m.group(7)) % 2)
    bad = [k for k, v in need.items() if len(v) > 1]
    qregs = sorted({k[0] for k in need})
    for l in loop:
        if not MAD.match(l):
            assert not set(qregs) & set(isa.defs_uses(l)[1]), ("Q register read outside the S mads", l)
    live = set(qregs)
    for l in epilogue:      # after the loop a Q register must be written before it is read
        w, r = isa.defs_uses(l)
        assert not live & set(r), ("Q register read after the loop", l)
        live -= set(w)
    mis = {0: [], 1: []}        # misplaced elements by current register parity
    for k, v in sorted(need.items()):
        if len(v) == 1 and k[0] % 2 != next(iter(v)):
            mis[k[0] % 2].append(k)
    loc = {}
    swaps = list(zip(mis[0], mis[1]))
    for a, b in swaps:
        loc[a], loc[b] = b, a
    tmp = free[0]
    pre = [f"// --- attn_gen remap: Q regs {qregs}, {len(swaps)} swaps (left: {len(mis[0]) - len(swaps)} even, "
           f"{len(mis[1]) - len(swaps)} odd, inconsistent {len(bad)}), temp r{tmp}"]
    for a, b in swaps:
        pre += [f"{W}mov (1|M0)  r{tmp}.0<1>:f  r{a[0]}.{a[1]}<0;1,0>:f",
                f"{W}mov (1|M0)  r{a[0]}.{a[1]}<1>:f  r{b[0]}.{b[1]}<0;1,0>:f",
                f"{W}mov (1|M0)  r{b[0]}.{b[1]}<1>:f  r{tmp}.0<0;1,0>:f"]
    # --- P V mads: copies of p registers in the other parity
    pregs = sorted({int(MAD.match(l).group(10)) for l in loop if MAD.match(l) and int(MAD.match(l).group(10)) in written})
    pool = {0: [r for r in free[1:] if r % 2 == 0], 1: [r for r in free[1:] if r % 2 == 1]}

    def needs_copy(l):
        m = MAD.match(l)
        if m and int(m.group(10)) in written and int(m.group(10)) % 2 == int(m.group(7)) % 2:
            return int(m.group(10))
        return None
    holder = {}     # copy register -> p register whose current value it holds
    out, ncopy = [], 0
    for i, l in enumerate(loop):
        m = MAD.match(l)
        if m:
            q = (int(m.group(10)), int(m.group(11)))
            p = needs_copy(l)
            if q[0] not in written:
                q = loc.get(q, q)
            elif p is not None:
                c = next((r for r, h in holder.items() if h == p), None)
                if c is None:
                    later = {needs_copy(x) for x in loop[i + 1:]}
                    c = next((r for r in pool[1 - p % 2] if holder.get(r) is None or holder[r] not in later), None)
                    if c is None:
                        raise SystemExit(f"no free register of parity {1 - p % 2} for a copy of r{p}")
                    out.append(f"        mov (8|M0)  r{c}.0<1>:f  r{p}.0<8;8,1>:f")
                    holder[c] = p
                    ncopy += 1
                q = (c, q[1])
            l = m.expand(r"\g<1>\g<2>\g<3>\g<4>\g<5>\g<6>\g<9>") + f"r{q[0]}.{q[1]}<0>:f" + m.group(12)
        else:
            w = set(isa.defs_uses(l)[0])
            holder = {r: h for r, h in holder.items() if h not in w and r not in w}
        out.append(l)
    info = f"p registers {pregs}, copy pool {pool}, {ncopy} copies per iteration"
    return prologue + pre, out, epilogue, info



def params(opts):
    return {k: int(v) for k, v in re.findall(r"-D(\w+)=(\d+)", opts)}


def imm(line):
    m = re.search(r"\s(-?\d+):(?:w|uw|d|ud)\b", isa.code(line))
    return int(m.group(1)) if m else None


def igc_maps(pro, loop, epi, P):
    """Symbolic pass over IGC's loop (linear order; branches only in the softmax): which registers hold
    Q[r][c] and O[r][c] (lane l = dim 8c + l), the K/V pointer registers, the l accumulators, counter/limit.
    Row numbers are internal (from the S accumulator chains); Q row r and O row r are the same query row."""
    D, NK, HKV = P["D"], P["NK"], P["HKV"]
    written = set()
    for l in loop:
        written |= set(isa.defs_uses(l)[0])
    val, addr = {}, {}

    def get(r):
        return val.get(r, ("entry", r))

    chains, schain_q, sinfo = 0, {}, {}
    qmap, omap, kbase, vbase = {}, {}, set(), set()
    lregs = set()
    pend_pv = []
    for l in loop:
        op = isa.opcode(l)
        if op is None:
            continue
        n = isa.exec_size(l)
        if op == "send":
            dst, src = isa.send_regs(l)
            base, off = addr.get(src[0], (src[0], 0))
            for gi, r in enumerate(dst):
                val[r] = ("mem", base, off, gi)
            continue
        ops = isa.operands(l)
        if op in ("add", "mov") and ops and ops[0][4] in ("q", "uq"):
            # addresses: kept apart from the data tags, since a (1|M0) :q write leaves the rest of the GRF intact
            sr = int(ops[1][1][1:])
            base, off = addr.get(sr, (sr, 0))
            off += (imm(l) or 0) if op == "add" else 0
            for r in isa.span(ops[0], n):
                addr[r] = (base, off)
                if n > 1:
                    val[r] = ("other", l)
            continue
        if op == "mad" and n == 8:
            d, s0, s1, s2 = ops
            dr, r0, r1, r2 = (int(x[1][1:]) for x in (d, s0, s1, s2))
            mem = get(r1)
            assert mem[0] == "mem", l
            sub1 = s1[2]
            if r2 not in written:                       # S mad: q scalar from a Q register
                if s0[3].startswith("<0"):
                    chains += 1
                    cid = chains
                else:
                    cid = get(r0)
                    assert cid[0] == "S", (l, cid)
                    cid = cid[1]
                val[dr] = ("S", cid)
                k, c = mem[2] // (D * 16), (mem[2] % (D * 16)) // 128
                i = mem[3] * 2 + sub1 // 8
                assert s2[2] == i, ("q lane != dim in chunk", l)
                kbase.add(mem[1])
                schain_q.setdefault(cid, set()).add((r2, c))
                assert sinfo.setdefault(cid, k) == k
            else:                                       # P V mad: p scalar from an exp'd score register
                pv = get(r2)
                assert pv[0] == "P", (l, pv)
                o = get(r0)
                assert o[0] == "entry", (l, o)
                rs = HKV * D * 2
                j, half = mem[2] // rs, (mem[2] % rs) // 128
                c = half * 8 + mem[3] * 2 + sub1 // 8
                assert j == sinfo[pv[1]] * 8 + s2[2], ("key mismatch", l)
                vbase.add(mem[1])
                pend_pv.append((pv[1], c, o[1]))
                val[dr] = o
            continue
        w, rd = isa.defs_uses(l)
        if op == "add" and n == 8 and ops[0][4] == "f":
            a = get(int(ops[1][1][1:]))
            if a[0] == "S" and ops[2][3].startswith("<0"):
                val[int(ops[0][1][1:])] = ("Sm", a[1])
                continue
            if ops[0][1] == ops[1][1] and get(int(ops[1][1][1:]))[0] == "entry":
                lregs.add(int(ops[0][1][1:]))
        if op == "math.exp" and n == 8:
            a = get(int(ops[1][1][1:]))
            if a[0] == "Sm":
                val[int(ops[0][1][1:])] = ("P", a[1])
                continue
        if op == "mul" and n == 8 and ops[0][1] == ops[1][1]:
            continue                                    # rescale in place keeps the identity
        if n > 1:                                       # (1|M0) writes leave the rest of the GRF intact
            for r in w:
                val[r] = ("other", l)
    # rows: chains with the same Q registers
    rows = {}
    for cid, qs in sorted(schain_q.items()):
        key = frozenset(r for r, c in qs)
        rows.setdefault(key, []).append(cid)
    row_of = {cid: ri for ri, (key, cids) in enumerate(rows.items()) for cid in cids}
    for cid, qs in schain_q.items():
        for r, c in qs:
            assert qmap.setdefault((row_of[cid], c), r) == r
    for cid, c, e in pend_pv:
        assert omap.setdefault((row_of[cid], c), e) == e, ("O register changes", cid, c, e)
    for e in set(omap.values()):                        # loop-carried: back in its register at the end
        assert get(e) == ("entry", e), ("O register rotated", e, get(e))
    head = next(l for l in loop if re.search(r"cmp \(\d+\|M0\)\s+\(lt\)", l))
    cnt, lim = re.findall(r"(r\d+\.\d+)<0;1,0>:d", head)[:2]
    assert len(kbase) == 1 and len(vbase) == 1, (kbase, vbase)
    return dict(q=qmap, o=omap, l=sorted(lregs - set(omap.values())), kp=kbase.pop(), vp=vbase.pop(),
                cnt=cnt, lim=lim, rows=len(rows))


def live_in(lines):
    """Registers read before written (linear order)."""
    w_, live = set(), set()
    for l in lines:
        w, r = isa.defs_uses(l)
        live |= set(r) - w_
        w_ |= set(w)
    return live


class Alloc:
    def __init__(self, free):
        self.free = sorted(free)

    def block(self, n, parity=0):
        for b in self.free:
            if b % 2 == parity and all(x in self.free for x in range(b, b + n)):
                for x in range(b, b + n):
                    self.free.remove(x)
                return b
        raise SystemExit(f"no block of {n} registers with parity {parity}; free {self.free}")

    def one(self, parity=None):
        r = next(x for x in self.free if parity is None or x % 2 == parity)
        self.free.remove(r)
        return r


def sched(pro, loop, epi, P, NS):
    """Variant S: the whole loop body written here; IGC's prologue (Q load and scale, pointers, O = 0) and
    epilogue (O stores) kept. Per block of 8*NK keys: D/8 K steps (NK block reads of 128 B, RQ*NK*8 mads) then
    8*NK V steps (D/64 block reads, RQ*D/8 mads) through a ring of NS slots of 8 GRF; the load for step s+NS is
    issued right after the mads of step s (NS-1 steps of mads ahead). Softmax between the phases (one branch per
    row for the rescale). Bank conflicts: slots start at even registers; Q elements are permuted before the loop
    to the parity opposite to their K register; p is kept in an even and an odd copy."""
    D, NK, HKV, RQ = P["D"], P["NK"], P["HKV"], P["RQ"]
    M = igc_maps(pro, loop, epi, P)
    assert M["rows"] == RQ, M["rows"]
    qregs, oregs = set(M["q"].values()), set(M["o"].values())
    kp, vp = M["kp"], M["vp"]
    cnt, lim = M["cnt"], M["lim"]
    busy = {0} | qregs | oregs | set(M["l"]) | {kp, kp + 1, vp, vp + 1}
    busy |= {int(re.match(r"r(\d+)", x).group(1)) for x in (cnt, lim)}
    busy |= live_in(epi)
    if COMPACT:
        assert len(M["l"]) == RQ, M["l"]
    A = Alloc(set(range(128)) - busy)
    nfree = len(A.free)
    slots = [A.block(8, 0) for _ in range(NS)]
    if PE_IN_S:     # exp in place: the even copy of p is the score register itself (saves RQ*NK registers)
        S = {(r, k): A.one(0) for r in range(RQ) for k in range(NK)}
        Pe = S
    else:
        S = {(r, k): A.one() for r in range(RQ) for k in range(NK)}
        Pe = {(r, k): A.one(0) for r in range(RQ) for k in range(NK)}
    Po = {(r, k): A.one(1) for r in range(RQ) for k in range(NK)}
    if COMPACT:     # running sums in IGC's (otherwise idle) l registers, row maxima in T2.4.. (saves RQ + 1)
        L = {r: M["l"][r] for r in range(RQ)}
        T1, T2 = A.one(), A.one()
        Mx, MO = T2, 4
    else:
        L = {r: A.one() for r in range(RQ)}
        Mx, T1, T2 = A.one(), A.one(), A.one()
        MO = 0
    nv = D // 64                                        # V block reads per row
    hK = [A.one() for _ in range(NK)]
    hV = [A.one() for _ in range(nv)]
    rs = HKV * D * 2                                    # V row stride (bytes)
    out = [f"// --- attn_gen sched: NS={NS} slots={slots} S={S} Pe={Pe} Po={Po} L={L} M=r{Mx} headers K{hK} V{hV}; "
           f"IGC: kp=r{kp} vp=r{vp} cnt={cnt} lim={lim} l={M['l']}; {nfree} free registers, {len(A.free)} left"]
    # Q permutation: element (r, c, i) must sit in a register of parity 1 - (i // 2) % 2 (slot bases even).
    need = {(M["q"][(r, c)], i): 1 - (i // 2) % 2 for (r, c) in M["q"] for i in range(8)}
    mis = {0: [], 1: []}
    for k, par in sorted(need.items()):
        if k[0] % 2 != par:
            mis[k[0] % 2].append(k)
    loc = {}
    for a, b in zip(mis[0], mis[1]):
        loc[a], loc[b] = b, a
        out += [f"{W}mov (1|M0)  r{T1}.0<1>:f  r{a[0]}.{a[1]}<0;1,0>:f",
                f"{W}mov (1|M0)  r{a[0]}.{a[1]}<1>:f  r{b[0]}.{b[1]}<0;1,0>:f",
                f"{W}mov (1|M0)  r{b[0]}.{b[1]}<1>:f  r{T1}.0<0;1,0>:f"]
    left = abs(len(mis[0]) - len(mis[1]))
    for r in range(RQ):
        out += [f"{W}mov (1|M0)  r{Mx}.{MO + r}<1>:f  -1e+30:f", f"        mov (8|M0)  r{L[r]}.0<1>:f  0.0:f"]
    for k in range(NK):
        out.append(f"{W}add (1|M0)  r{hK[k]}.0<1>:q  r{kp}.0<0;1,0>:q  {k * D * 16}:w")
    for h in range(nv):
        out.append(f"{W}add (1|M0)  r{hV[h]}.0<1>:q  r{vp}.0<0;1,0>:q  {h * 128}:w")
    nks, nvs = D // 8, 8 * NK
    nsteps = nks + nvs

    def load(s):
        """Load for global step s (mod nsteps) into slot s % NS."""
        s %= nsteps
        b = slots_of[s]
        o = []
        if s < nks:
            for k in range(NK):
                o += [f"{W}send (16|M0)  r{b + 4 * k}:w  r{hK[k]}  0xC  0x024D0CFF",
                      f"{W}add (1|M0)  r{hK[k]}.0<1>:q  r{hK[k]}.0<0;1,0>:q  {128 + (NK - 1) * D * 16 if s == nks - 1 else 128}:w"]
        else:
            for h in range(nv):
                o += [f"{W}send (16|M0)  r{b + 4 * h}:w  r{hV[h]}  0xC  0x024D0CFF",
                      f"{W}add (1|M0)  r{hV[h]}.0<1>:q  r{hV[h]}.0<0;1,0>:q  {rs}:w"]
        return o
    # slot of global step s: s % NS over a continuous step counter -> needs nsteps % NS == 0 for a fixed code body
    assert nsteps % NS == 0, f"steps per block {nsteps} must divide by NS {NS}"
    slots_of = {s: slots[s % NS] for s in range(nsteps)}

    def kmads(c):
        b, o = slots_of[c], []
        for i in range(8):
            for k in range(NK):
                for r in range(RQ):
                    q = loc.get((M["q"][(r, c)], i), (M["q"][(r, c)], i))
                    o.append(f"        mad (8|M0)  r{S[(r, k)]}.0<1>:f  r{S[(r, k)]}.0<2;1>:f  r{b + 4 * k + i // 2}.{(i % 2) * 8}<2;1>:hf  r{q[0]}.{q[1]}<0>:f")
        return o

    def vmads(j):
        b, o = slots_of[nks + j], []
        k, jj = j // 8, j % 8
        for r in range(RQ):
            for c in range(D // 8):
                h, cc = c // 8, c % 8
                vr = b + 4 * h + cc // 2
                p = Po[(r, k)] if vr % 2 == 0 else Pe[(r, k)]
                o.append(f"        mad (8|M0)  r{M['o'][(r, c)]}.0<1>:f  r{M['o'][(r, c)]}.0<2;1>:f  r{vr}.{(cc % 2) * 8}<2;1>:hf  r{p}.{jj}<0>:f")
        return o

    def softmax():
        o = []
        for r in range(RQ):
            sk = [S[(r, k)] for k in range(NK)]
            if NK == 1:
                o.append(f"        mov (8|M0)  r{T1}.0<1>:f  r{sk[0]}.0<8;8,1>:f")
            else:
                o.append(f"        sel (8|M0)  (ge)f0.0  r{T1}.0<1>:f  r{sk[0]}.0<8;8,1>:f  r{sk[1]}.0<8;8,1>:f")
                for k in range(2, NK):
                    o.append(f"        sel (8|M0)  (ge)f0.0  r{T1}.0<1>:f  r{T1}.0<8;8,1>:f  r{sk[k]}.0<8;8,1>:f")
            o += [f"{W}sel (4|M0)  (ge)f0.0  r{T2}.0<1>:f  r{T1}.0<4;4,1>:f  r{T1}.4<4;4,1>:f",
                  f"{W}sel (2|M0)  (ge)f0.0  r{T2}.0<1>:f  r{T2}.0<2;2,1>:f  r{T2}.2<2;2,1>:f",
                  f"{W}sel (1|M0)  (ge)f0.0  r{T2}.0<1>:f  r{T2}.0<0;1,0>:f  r{T2}.1<0;1,0>:f",
                  f"{W}cmp (8|M0)  (gt)f1.0  null<1>:f  r{T2}.0<0;1,0>:f  r{Mx}.{MO + r}<0;1,0>:f",
                  f"(W&~f1.0) jmpi  LATT_SKIP{r}",
                  f"{W}add (1|M0)  r{T2}.1<1>:f  r{Mx}.{MO + r}<0;1,0>:f  -r{T2}.0<0;1,0>:f",
                  f"{W}mov (1|M0)  r{Mx}.{MO + r}<1>:f  r{T2}.0<0;1,0>:f",
                  f"{W}math.exp (1|M0)  r{T2}.1<1>:f  r{T2}.1<0;1,0>:f"]
            o += [f"        mul (8|M0)  r{x}.0<1>:f  r{x}.0<8;8,1>:f  r{T2}.1<0;1,0>:f"
                  for x in [M["o"][(r, c)] for c in range(D // 8)] + [L[r]]]
            o.append(f"LATT_SKIP{r}:")
            for k in range(NK):
                o += [f"        add (8|M0)  r{sk[k]}.0<1>:f  r{sk[k]}.0<8;8,1>:f  -r{Mx}.{MO + r}<0;1,0>:f",
                      f"        math.exp (8|M0)  r{Pe[(r, k)]}.0<1>:f  r{sk[k]}.0<8;8,1>:f"]
            for k in range(NK):
                o += [f"        add (8|M0)  r{L[r]}.0<1>:f  r{L[r]}.0<8;8,1>:f  r{Pe[(r, k)]}.0<8;8,1>:f",
                      f"        mov (8|M0)  r{Po[(r, k)]}.0<1>:f  r{Pe[(r, k)]}.0<8;8,1>:f"]
        return o

    for s in range(NS):
        out += load(s)
    out.append("LATT_LOOP:")
    out += [f"        mov (8|M0)  r{S[k]}.0<1>:f  0.0:f" for k in sorted(S)]
    for s in range(nsteps):
        out += kmads(s) if s < nks else vmads(s - nks)
        out += load(s + NS)
        if s == nks - 1:
            out += softmax()
    out += [f"{W}add (1|M0)  {cnt}<1>:d  {cnt}<0;1,0>:d  {8 * NK}:w",
            f"{W}cmp (8|M0)  (lt)f0.0  null<1>:d  {cnt}<0;1,0>:d  {lim}<0;1,0>:d",
            "(W&f0.0) jmpi  LATT_LOOP"]
    # after the loop: O /= sum(l) here; IGC's l registers := 1/8 per lane so the epilogue divides by 1
    for r in range(RQ):
        out += [f"{W}add (4|M0)  r{T2}.0<1>:f  r{L[r]}.0<4;4,1>:f  r{L[r]}.4<4;4,1>:f",
                f"{W}add (2|M0)  r{T2}.0<1>:f  r{T2}.0<2;2,1>:f  r{T2}.2<2;2,1>:f",
                f"{W}add (1|M0)  r{T2}.0<1>:f  r{T2}.0<0;1,0>:f  r{T2}.1<0;1,0>:f",
                f"{W}math.inv (1|M0)  r{T2}.0<1>:f  r{T2}.0<0;1,0>:f"]
        out += [f"        mul (8|M0)  r{M['o'][(r, c)]}.0<1>:f  r{M['o'][(r, c)]}.0<8;8,1>:f  r{T2}.0<0;1,0>:f" for c in range(D // 8)]
    out += [f"        mov (8|M0)  r{x}.0<1>:f  0.125:f" for x in M["l"]]
    info = f"Q swaps {len(loc) // 2} (unpaired {left}), {nfree} registers free for the loop"
    return pro, out, epi, info


def main():
    mode, src, opts = sys.argv[1], sys.argv[2], sys.argv[3]
    d = tempfile.mkdtemp()
    try:
        dd = os.path.join(d, "d")
        disassemble(src, opts, dd)
        t = open(os.path.join(dd, f".text.{KERNEL}.asm")).read().splitlines()
        pro, loop, epi = split_loop(t)
        n, c = conflicts(loop)
        print(f"IGC loop: {len(loop)} lines, mad {n}, bank conflicts {c}")
        if mode == "analyze":
            return
        if mode == "remap":
            pro, loop, epi, info = remap(pro, loop, epi)
        elif mode.startswith("sched"):        # sched<NS>, e.g. sched3
            pro, loop, epi, info = sched(pro, loop, epi, params(opts), int(mode[5:] or 3))
            print(loop[0][:400])
            print(info)
        elif mode == "poison":      # check that the driver runs our ISA: first S mad with -q -> wrong O
            i = next(i for i, l in enumerate(loop) if MAD.match(l))
            m = MAD.match(loop[i])
            loop[i] = f"{m.group(1)}{m.group(2)}  {m.group(4)}  {m.group(6)}  -r{m.group(10)}.{m.group(11)}<0>:f"
        n, c = conflicts(loop)
        if mode == "remap":
            print([x for x in pro if x.startswith("// --- attn_gen")][0])
        if mode == "remap":
            print(info)
        print(f"new loop: {len(loop)} lines, mad {n}, bank conflicts {c}")
        out = sys.argv[4]
        text = pro + loop + epi
        if "--asm" in sys.argv:
            open(sys.argv[sys.argv.index("--asm") + 1], "w").write("\n".join(text) + "\n")
        assemble(dd, text, out)
    finally:
        shutil.rmtree(d)


if __name__ == "__main__":
    main()
