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

Toolchain pin: the symbolic pass reads IGC's own code, so the generator only runs with the ocloc it was checked
against (OCLOC_VERSION below; `ocloc query OCL_DRIVER_VERSION`). Another version exits with a message;
ATTN_OCLOC_ANY=1 overrides (then re-check the results on the GPU and the expected hashes, `make attn-check`).
Every line of the regions the generator analyses (IGC's loop and epilogue) must be a form asm/isa.py understands
(isa.check); anything else (new opcode, indirect or accumulator operand, unknown region) stops the generator.

Requirements on the host (bench/attn.c checks them; the generated loop does not):
  - n_k > 0 and n_k divisible by 8*NK (one loop iteration = 8*NK keys, no tail handling);
  - n_q * G divisible by RQ (not needed with -DROWVIS=1: rows clamped, grid rounded up), full SIMD8 subgroups:
    work-group size divisible by 8 and global size by the work-group size (the analysis assumes all 8 lanes enabled);
  - ROWVIS=1 (template option, docs/attention-gpu.md section 13): per-row visible key counts. IGC's l register of each
    row enters the loop holding (float) lim_r; igc_maps ties every l register to one row by data flow (the l += sum of
    that row's exp'd scores) and sched() copies lim_r to T2.6.. before zeroing l; the mask of the last block compares
    each row against its own limit (RQ <= 2: rows differ by at most one key, both limits in the last block);
  - K PADDING (sched modes): the ring prefetches the first NS steps of the next key block, also after the last
    block, so the K-transposed buffer Kt is read past the end of the last KV head by up to
    (NS-1)*128 + (NK-1)*D*16 + 128 bytes (Qwen sched4 NK 2 D 128: 2560 B; Llama sched2 NK 2 D 64: 1280 B).
    These bytes are loaded but never used. Since NS <= D/8 (asserted), the bound NK*D*16 bytes (one block of
    8*NK keys: Qwen 4096 B, Llama 2048 B) covers every ring depth: allocate Kt with at least that much extra.
    V is not over-read (V steps never prefetch past the current block).
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
RD4 = "// wr:1h+0, rd:4; hdc.dc1; a64 aligned oword block read x8"   # as ocloc disasm decodes it (isa.send_regs)
# ocloc device: whl = UHD 620 (laptop), skl = HD 530 (desktop); a binary for one is CL_INVALID_BINARY on the other.
DEVICE = os.environ.get("DEVICE", "whl")
# PE_IN_S=1: exp in place, the score registers serve as the even p copies (saves RQ*NK registers)
PE_IN_S = os.environ.get("PE_IN_S", "0") == "1"
# COMPACT=1: l sums in IGC's l registers and row maxima in a temporary's spare lanes (3 registers fewer for Qwen)
COMPACT = os.environ.get("COMPACT", "1") == "1"
# Loop slimming (docs/attention-gpu.md section 12; results/attn-slim-hd530-2026-11-13.txt). Default on (measured gains):
#   SLIM_SMAX row maxima of all RQ rows in one reduction tree and one `.any<RQ>h` branch (Llama/whisper +1.3 %)
#   SLIM_CNT  loop counter test before the softmax, its flag shared by the causal mask and the loop jump
#   SLIM_MUL  first K product of a block as mul instead of zeroing the scores
# Default off (measured slower, kept for the record): SLIM_HDR two send headers per (2|M0) :q add with dst <4>
#   (Qwen -5 %, Llama -3 %: the 2-register QW add loads FPU0 and starves the dual issue, OA FPU1 -10 points).
SLIM_SMAX = os.environ.get("SLIM_SMAX", "1") == "1"
SLIM_CNT = os.environ.get("SLIM_CNT", "1") == "1"
SLIM_MUL = os.environ.get("SLIM_MUL", "0") == "1"
SLIM_HDR = os.environ.get("SLIM_HDR", "0") == "1"
OCLOC_VERSION = "24.35.30872.36"    # NEO release the symbolic pass and the hashes in attn_sha256.txt were checked with


def check_ocloc():
    tmp = tempfile.mkdtemp()            # ocloc query also writes a file named after the query into the cwd
    try:
        r = subprocess.run(["ocloc", "query", "OCL_DRIVER_VERSION"], capture_output=True, text=True, cwd=tmp)
    finally:
        shutil.rmtree(tmp)
    v = r.stdout.strip().splitlines()[-1].strip() if r.returncode == 0 and r.stdout.strip() else "?"
    if v != OCLOC_VERSION:
        msg = (f"attn_gen.py: ocloc reports driver version {v!r}, expected {OCLOC_VERSION} (the version the "
               f"symbolic pass over IGC's code was checked with).")
        if os.environ.get("ATTN_OCLOC_ANY") != "1":
            raise SystemExit(msg + " Set ATTN_OCLOC_ANY=1 to try anyway, then re-check the results.")
        print("WARNING: " + msg, file=sys.stderr)


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


def assert_not_read_after(regs, lines, what):
    """The registers regs (modified in place before/inside the loop) must be written before they are read in
    lines (code after the loop): partial writes do not count as a write (isa.full_writes)."""
    live = set(regs)
    for l in lines:
        w, r = isa.defs_uses(l)
        assert not live & set(r), (f"{what} register read after the loop", sorted(live & set(r)), l)
        live -= set(isa.full_writes(l))


def remap(prologue, loop, epilogue):
    isa.check_all(loop, "IGC's loop")
    isa.check_all(epilogue, "IGC's epilogue")
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
    assert_not_read_after(qregs, epilogue, "Q")     # the swaps below leave Q permuted
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
    """Integer immediate of the line (None if there is none). IGA prints a value that does not fit the signed type
    in hex with the unsigned type: with HKV 20 (V row 2560 B) IGC advances the V pointer by 0x8200:uw = 33280 B,
    which the old decimal-only pattern missed; the address add then counted as +0, the PV mad of key 13 read as
    key 0 and igc_maps stopped on its key assert. Parsing is isa.imm's (hex of a signed type is two's complement)."""
    r = isa.imm(line, ("w", "uw", "d", "ud"))     # asserts on a value wider than its type
    return r[0] if r else None


def igc_maps(pro, loop, epi, P):
    """Symbolic pass over IGC's loop (linear order; branches only in the softmax): which registers hold
    Q[r][c] and O[r][c] (lane l = dim 8c + l), the K/V pointer registers, the l accumulators, counter/limit.
    Row numbers are internal (from the S accumulator chains); Q row r and O row r are the same query row."""
    D, NK, HKV = P["D"], P["NK"], P["HKV"]
    isa.check_all(loop, "IGC's loop")
    isa.check_all(epi, "IGC's epilogue")
    written = set()
    for l in loop:
        written |= set(isa.defs_uses(l)[0])
    val, addr = {}, {}

    def get(r):
        return val.get(r, ("entry", r))

    chains, schain_q, sinfo = 0, {}, {}
    qmap, omap, kbase, vbase = {}, {}, set(), set()
    lregs = {}                                          # l candidate -> accumulating adds per iteration
    ltag = {}                                           # l candidate -> tags of what it accumulates (ROWVIS: its row)
    pend_pv = []
    for l in loop:
        op = isa.opcode(l)
        if op is None:
            continue
        n = isa.exec_size(l)
        if op == "send":
            dst, src = isa.send_regs(l)
            base, off = addr.get(src[0], ((src[0], 0), 0))
            for gi, r in enumerate(dst):
                val[r] = ("mem", base, off, gi)
            continue
        ops = isa.operands(l)
        if op in ("add", "mov") and ops and ops[0][4] in ("q", "uq"):
            # addresses: kept apart from the data tags, since a (1|M0) :q write leaves the rest of the GRF intact
            # base = (register, :q subregister) of the pointer IGC keeps across the loop: IGC may hold it in
            # r6.1:q, not r6.0 (seen with -DHKV_INNER; ignoring the subregister gave a wrong binary, error 1.0)
            sr = int(ops[1][1][1:])
            if sr not in addr:
                assert ops[1][4] in ("q", "uq") or ops[1][2] == 0, ("address taken from a non-:q subregister", l)
            base, off = addr.get(sr, ((sr, ops[1][2]), 0))
            if op == "add":     # an address step must be an immediate the pass can read (no silent +0)
                step = imm(l)
                assert step is not None, ("address add without an integer immediate", l)
                off += step
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
            b = get(int(ops[2][1][1:])) if len(ops) > 2 and ops[2][1].startswith("r") else ("imm",)
            if a[0] in ("P", "Psum") and b[0] in ("P", "Psum") and not ops[1][0] and not ops[2][0]:
                cids = lambda x: x[1] if x[0] == "Psum" else frozenset([x[1]])    # noqa: E731
                val[int(ops[0][1][1:])] = ("Psum", cids(a) | cids(b))   # sum of exp'd score chains (NK 2: p.s0 + p.s1)
                continue
            if ops[0][1] == ops[1][1] and get(int(ops[1][1][1:]))[0] == "entry":
                lr = int(ops[0][1][1:])
                lregs[lr] = lregs.get(lr, 0) + 1
                ltag.setdefault(lr, set()).add(b[:2] if b[0] in ("P", "Psum") else b[:1])
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
    # l sums (heuristic: loop-carried `add (8) rX = rX + ...` :f that is not an O register). Must be exactly one
    # register per row, accumulated once per iteration and read by the epilogue (sum and 1/l).
    lr = sorted(set(lregs) - set(omap.values()))
    assert len(lr) == len(rows), ("l registers: expected one per row", lr, len(rows))
    assert all(lregs[r] == 1 for r in lr), ("l register accumulated more than once per iteration", lregs)
    assert set(lr) <= live_in(epi)[0], ("l register not read by the epilogue", lr)
    # row of each l register (data flow: the l += sum(p) of exactly one row's scores); used by ROWVIS, checked always
    lrow = {}
    for r in lr:
        t = ltag[r]
        if len(t) == 1 and next(iter(t))[0] in ("P", "Psum"):
            x = next(iter(t))
            rs_ = {row_of[c] for c in (x[1] if x[0] == "Psum" else [x[1]])}
            if len(rs_) == 1:
                lrow[r] = rs_.pop()
    if P.get("ROWVIS"):
        assert len(lrow) == len(rows) and sorted(lrow.values()) == list(range(len(rows))), \
            ("ROWVIS: l registers not tied one-to-one to rows", ltag, lrow)
    # loop counter: the only `cmp (lt)` of the loop, `cmp cnt < lim` on two scalar :d registers, after the only
    # write of cnt (`add (1) cnt = cnt + 8*NK`); lim not written in the loop; the loop exits on the negated flag.
    heads = [i for i, l in enumerate(loop) if re.search(r"cmp \(\d+\|M0\)\s+\(lt\)", l)]
    assert len(heads) == 1, ("expected exactly one cmp (lt) in IGC's loop", [loop[i] for i in heads])
    head = loop[heads[0]]
    regs = re.findall(r"(r\d+\.\d+)<0;1,0>:d", head)
    assert len(regs) == 2 and "null" in head, ("cmp (lt) form", head)
    cnt, lim = regs
    flag = re.search(r"\(lt\)(f\d\.\d)", head).group(1)

    def bytes_of(s, ts=4):
        r, sub = (int(x) for x in s[1:].split("."))
        return set(range(r * 32 + sub * ts, r * 32 + sub * ts + ts))

    def writes(l):
        ops = isa.operands(l)
        if isa.opcode(l) in (None, "send", "sends") or not ops or not ops[0][1].startswith("r"):
            return {b for r in isa.defs_uses(l)[0] for b in range(r * 32, r * 32 + 32)}
        _, reg, sub, region, t = ops[0]
        m = re.match(r"<(\d+)>$", region)
        hs = int(m.group(1)) if m else 1
        st = int(reg[1:]) * 32 + sub * isa.TS[t]
        return {st + e * hs * isa.TS[t] + b for e in range(isa.exec_size(l)) for b in range(isa.TS[t])}
    cw = [i for i, l in enumerate(loop) if writes(l) & bytes_of(cnt)]
    lw = [i for i, l in enumerate(loop) if writes(l) & bytes_of(lim)]
    assert len(cw) == 1 and not lw, ("counter/limit written elsewhere in the loop", cnt, lim, cw, lw)
    inc = loop[cw[0]]
    assert re.search(rf"add \(1\|M0\)\s+{re.escape(cnt)}<1>:d\s+{re.escape(cnt)}<0;1,0>:d\s+{8 * NK}:w", inc), \
        ("counter increment is not cnt += 8*NK", inc)
    assert cw[0] < heads[0], ("cmp before the counter increment", inc, head)
    exits = [l for l in loop if re.search(rf"\(~{re.escape(flag)}\)\s+break", l)]
    assert len(exits) == 1, ("loop exit is not `(~flag) break` on the cmp flag", flag)
    assert len(kbase) == 1 and len(vbase) == 1, (kbase, vbase)
    return dict(q=qmap, o=omap, l=lr, lrow=lrow, kp=kbase.pop(), vp=vbase.pop(),
                cnt=cnt, lim=lim, rows=len(rows))


def live_in(lines):
    """(live, headers): registers read before they are completely written (linear order; partial writes such as
    `mov (1) r72.2:ud` do not end a live range, isa.full_writes). headers = the subset whose every such read is as
    the header (src0) of a send after a partial write of that register in lines: IGC sets only the header fields
    the message uses and leaves the rest as found; the loop may use these registers if they are zeroed after it."""
    full, part, live, hdr_ok = set(), set(), set(), {}
    for l in lines:
        _, r = isa.defs_uses(l)
        early = set(r) - full
        live |= early
        if isa.opcode(l) in ("send", "sends"):
            h = isa.send_regs(l)[1][0]
            for x in early:
                hdr_ok[x] = hdr_ok.get(x, True) and x == h and x in part
        else:
            for x in early:
                hdr_ok[x] = False
        w = set(isa.defs_uses(l)[0])
        full |= set(isa.full_writes(l))
        part |= w
    return live, {x for x in live if hdr_ok.get(x)}


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
    busy = {0} | qregs | oregs | set(M["l"]) | {kp[0], kp[0] + 1, vp[0], vp[0] + 1}
    busy |= {int(re.match(r"r(\d+)", x).group(1)) for x in (cnt, lim)}
    epi_live, epi_hdr = live_in(epi)
    busy |= epi_live - epi_hdr          # epilogue send headers: zeroed after the loop instead (see live_in)
    assert NS <= D // 8, ("ring deeper than the K phase: the prefetch would cross into V steps", NS, D // 8)
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
    # SLIM_HDR: header pairs in adjacent registers, advanced together by one (2|M0) :q add with dst <4> (32 B apart);
    # with one V block read per step (D 64) two V headers serve the even and odd key steps (advanced by 2 rows).
    hdr2 = SLIM_HDR and NK == 2
    vpair = SLIM_HDR and nv <= 2

    def pair():
        try:
            b = A.block(2, 0)
        except SystemExit:
            b = A.block(2, 1)
        return [b, b + 1]
    hK = pair() if hdr2 else [A.one() for _ in range(NK)]
    hV = pair() if vpair else [A.one() for _ in range(nv)]
    rs = HKV * D * 2                                    # V row stride (bytes)
    # SLIM_SMAX: RQ consecutive registers that are dead during the row-max reduction (odd p copies, even p copies
    # unless they are the score registers, T1, unallocated): Pe/Po are last read by the previous block's V mads and
    # written only after the reduction; T1 is free after the mask.
    dead = (set(Pe.values()) - set(S.values())) | set(Po.values()) | {T1} | set(A.free)
    Rblk = next(([b + i for i in range(RQ)] for b in range(128) if all(b + i in dead for i in range(RQ))), None)
    if SLIM_SMAX:
        assert Rblk is not None, ("SLIM_SMAX: no", RQ, "consecutive dead registers", sorted(dead))
    out = [f"// --- attn_gen sched: NS={NS} slots={slots} S={S} Pe={Pe} Po={Po} L={L} M=r{Mx} headers K{hK} V{hV}; "
           f"IGC: kp=r{kp[0]}.{kp[1]}:q vp=r{vp[0]}.{vp[1]}:q cnt={cnt} lim={lim} l={M['l']}; {nfree} free registers, {len(A.free)} left"]
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
    # ROWVIS (asm/attn.cl): IGC's l register of row r enters the loop holding (float) lim_r; copy it as an integer into
    # T2.(LV + r) before the l sums are zeroed (T2 dwords: 0-1 reductions, 2 rem, MO.. row maxima, LV.. limits)
    rowvis = P.get("ROWVIS", 0) == 1
    LV = 6
    if rowvis:
        assert P.get("CAUSAL", 0) == 1 and RQ <= 2 and (Mx != T2 or MO + RQ <= LV), ("ROWVIS needs CAUSAL, RQ <= 2", RQ, MO)
        lof = {row: reg for reg, row in M["lrow"].items()}
        out += [f"{W}mov (1|M0)  r{T2}.{LV + r}<1>:d  r{lof[r]}.0<0;1,0>:f" for r in range(RQ)]
    for r in range(RQ):
        out += [f"{W}mov (1|M0)  r{Mx}.{MO + r}<1>:f  -1e+30:f", f"        mov (8|M0)  r{L[r]}.0<1>:f  0.0:f"]
    # Send headers: the A64 oword block read uses only the address in DW0-1 (IGC's own headers set only .0:uq
    # too), but the whole header is zeroed once here so no field holds a prologue leftover; inside the loop only
    # .0:q is updated, the other 6 dwords stay zero.
    for h in hK + hV:
        out.append(f"{W}mov (8|M0)  r{h}.0<1>:ud  0x0:ud")
    for k in range(NK):
        out.append(f"{W}add (1|M0)  r{hK[k]}.0<1>:q  r{kp[0]}.{kp[1]}<0;1,0>:q  {k * D * 16}:w")
    for h in range(len(hV)):    # nv 2: the two halves of a V row; vpair with nv 1: rows j and j + 1
        out.append(f"{W}add (1|M0)  r{hV[h]}.0<1>:q  r{vp[0]}.{vp[1]}<0;1,0>:q  {h * (128 if nv == 2 else rs)}:w")
    nks, nvs = D // 8, 8 * NK
    nsteps = nks + nvs

    def load(s):
        """Load for global step s (mod nsteps) into slot s % NS."""
        s %= nsteps
        b = slots_of[s]
        o = []
        if s < nks:
            step = 128 + (NK - 1) * D * 16 if s == nks - 1 else 128
            if hdr2:
                o += [f"{W}send (16|M0)  r{b + 4 * k}:w  r{hK[k]}  0xC  0x024D0CFF  {RD4}" for k in range(NK)]
                o.append(f"{W}add (2|M0)  r{hK[0]}.0<4>:q  r{hK[0]}.0<4;1,0>:q  {step}:w")
            else:
                for k in range(NK):
                    o += [f"{W}send (16|M0)  r{b + 4 * k}:w  r{hK[k]}  0xC  0x024D0CFF  {RD4}",
                          f"{W}add (1|M0)  r{hK[k]}.0<1>:q  r{hK[k]}.0<0;1,0>:q  {step}:w"]
        elif vpair and nv == 2:
            o += [f"{W}send (16|M0)  r{b + 4 * h}:w  r{hV[h]}  0xC  0x024D0CFF  {RD4}" for h in range(nv)]
            o.append(f"{W}add (2|M0)  r{hV[0]}.0<4>:q  r{hV[0]}.0<4;1,0>:q  {rs}:w")
        elif vpair:                 # nv 1: key step j reads through hV[j % 2]; both advance by 2 rows after odd j
            j = s - nks
            o.append(f"{W}send (16|M0)  r{b}:w  r{hV[j % 2]}  0xC  0x024D0CFF  {RD4}")
            if j % 2:
                o.append(f"{W}add (2|M0)  r{hV[0]}.0<4>:q  r{hV[0]}.0<4;1,0>:q  {2 * rs}:w")
        else:
            for h in range(nv):
                o += [f"{W}send (16|M0)  r{b + 4 * h}:w  r{hV[h]}  0xC  0x024D0CFF  {RD4}",
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
                    if SLIM_MUL and c == 0 and i == 0:  # first product of the block: S = K q, no S = 0 before
                        o.append(f"        mul (8|M0)  r{S[(r, k)]}.0<1>:f  r{b + 4 * k}.0<8;8,1>:hf  r{q[0]}.{q[1]}<0;1,0>:f")
                        continue
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

    def softmax_vec():
        """SLIM_SMAX: the row maxima of all RQ rows in one reduction tree and one branch (section 12). Stage 1 per
        row: R[r] = max over the NK score registers; R are RQ consecutive registers dead at this point (p copies, T1).
        Rows are then reduced together through 2-register regions (<8;4,1>, <4;2,1>, <2;1,0>) into T2.0..RQ-1;
        one cmp (RQ) against the running maxima and one `.any<RQ>h` jump skip the rescale when no row grew. In the
        rescale path each row is tested again (cmp + jmpi per row), so a row whose maximum did not grow is not
        touched (same arithmetic as softmax(): a = exp(m - mn), O and l multiplied only when mn > m)."""
        o, R = [], Rblk
        for r in range(RQ):
            sk = [S[(r, k)] for k in range(NK)]
            if NK == 1:
                o.append(f"        mov (8|M0)  r{R[r]}.0<1>:f  r{sk[0]}.0<8;8,1>:f")
            else:
                o.append(f"        sel (8|M0)  (ge)f0.0  r{R[r]}.0<1>:f  r{sk[0]}.0<8;8,1>:f  r{sk[1]}.0<8;8,1>:f")
                for k in range(2, NK):
                    o.append(f"        sel (8|M0)  (ge)f0.0  r{R[r]}.0<1>:f  r{R[r]}.0<8;8,1>:f  r{sk[k]}.0<8;8,1>:f")
        if RQ == 1:
            o += [f"{W}sel (4|M0)  (ge)f0.0  r{T2}.0<1>:f  r{R[0]}.0<4;4,1>:f  r{R[0]}.4<4;4,1>:f",
                  f"{W}sel (2|M0)  (ge)f0.0  r{T2}.0<1>:f  r{T2}.0<2;2,1>:f  r{T2}.2<2;2,1>:f",
                  f"{W}sel (1|M0)  (ge)f0.0  r{T2}.0<1>:f  r{T2}.0<0;1,0>:f  r{T2}.1<0;1,0>:f"]
        elif RQ == 2:   # R0 lanes: 4 per row, then 2 per row, then T2.0/T2.1
            o += [f"{W}sel (8|M0)  (ge)f0.0  r{R[0]}.0<1>:f  r{R[0]}.0<8;4,1>:f  r{R[0]}.4<8;4,1>:f",
                  f"{W}sel (4|M0)  (ge)f0.0  r{R[0]}.0<1>:f  r{R[0]}.0<4;2,1>:f  r{R[0]}.2<4;2,1>:f",
                  f"{W}sel (2|M0)  (ge)f0.0  r{T2}.0<1>:f  r{R[0]}.0<2;1,0>:f  r{R[0]}.1<2;1,0>:f"]
        else:           # RQ 4: rows 0,1 -> R0 (4 lanes each), rows 2,3 -> R1; then 2 per row in R0; then T2.0..3
            o += [f"{W}sel (8|M0)  (ge)f0.0  r{R[0]}.0<1>:f  r{R[0]}.0<8;4,1>:f  r{R[0]}.4<8;4,1>:f",
                  f"{W}sel (8|M0)  (ge)f0.0  r{R[1]}.0<1>:f  r{R[2]}.0<8;4,1>:f  r{R[2]}.4<8;4,1>:f",
                  f"{W}sel (8|M0)  (ge)f0.0  r{R[0]}.0<1>:f  r{R[0]}.0<4;2,1>:f  r{R[0]}.2<4;2,1>:f",
                  f"{W}sel (4|M0)  (ge)f0.0  r{T2}.0<1>:f  r{R[0]}.0<2;1,0>:f  r{R[0]}.1<2;1,0>:f"]
        v = f"<{RQ};{RQ},1>"
        o += [f"{W}cmp ({RQ}|M0)  (gt)f1.0  null<1>:f  r{T2}.0{v}:f  r{Mx}.{MO}{v}:f",
              f"(W&~f1.0{'.any%dh' % RQ if RQ > 1 else ''}) jmpi  LATT_SKIP",
              f"{W}add ({RQ}|M0)  r{T1}.0<1>:f  r{Mx}.{MO}{v}:f  -r{T2}.0{v}:f",
              f"{W}math.exp ({RQ}|M0)  r{T1}.0<1>:f  r{T1}.0{v}:f"]
        for r in range(RQ):
            o += [f"{W}cmp (1|M0)  (gt)f0.1  null<1>:f  r{T2}.{r}<0;1,0>:f  r{Mx}.{MO + r}<0;1,0>:f",
                  f"(W&~f0.1) jmpi  LATT_RS{r}"]
            o += [f"        mul (8|M0)  r{x}.0<1>:f  r{x}.0<8;8,1>:f  r{T1}.{r}<0;1,0>:f"
                  for x in [M["o"][(r, c)] for c in range(D // 8)] + [L[r]]]
            o.append(f"LATT_RS{r}:")
        o += [f"{W}sel ({RQ}|M0)  (ge)f0.0  r{Mx}.{MO}<1>:f  r{Mx}.{MO}{v}:f  r{T2}.0{v}:f",
              "LATT_SKIP:"]
        for r in range(RQ):
            sk = [S[(r, k)] for k in range(NK)]
            for k in range(NK):
                o += [f"        add (8|M0)  r{sk[k]}.0<1>:f  r{sk[k]}.0<8;8,1>:f  -r{Mx}.{MO + r}<0;1,0>:f",
                      f"        math.exp (8|M0)  r{Pe[(r, k)]}.0<1>:f  r{sk[k]}.0<8;8,1>:f"]
            for k in range(NK):
                o += [f"        add (8|M0)  r{L[r]}.0<1>:f  r{L[r]}.0<8;8,1>:f  r{Pe[(r, k)]}.0<8;8,1>:f",
                      f"        mov (8|M0)  r{Po[(r, k)]}.0<1>:f  r{Pe[(r, k)]}.0<8;8,1>:f"]
        return o

    def softmax():
        if SLIM_SMAX:
            return softmax_vec()
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

    def mask():
        """CAUSAL (template option): lim is the per-subgroup visible key count nv (attn.cl), cnt the first key of the
        current block. A block with rem = nv - cnt < 8*NK keys left is the last, partial one: its score lanes of keys
        >= nv are set to -1e30 (exp -> 0). Full blocks cost 3 scalar instructions (add, cmp, jmpi). Uses T1 (lane
        ids, :w) and T2.2 (rem, :d; read as T2.4:w), both free here (the softmax rewrites them; Mx sits in T2.4:f+ as
        dwords 4..7, not dword 2); flag f1.0 (the softmax sets it again before each use)."""
        if SLIM_CNT:    # cnt already advanced, f1.1 = (cnt < lim) = another block follows -> this one is full
            o = ["(W&f1.1) jmpi  LATT_NOMASK"]
            if not rowvis:  # (rowvis: rem per row below)
                o += [f"{W}add (1|M0)  r{T2}.2<1>:d  {lim}<0;1,0>:d  -{cnt}<0;1,0>:d",
                      f"{W}add (1|M0)  r{T2}.2<1>:d  r{T2}.2<0;1,0>:d  {8 * NK}:w"]
            o.append(f"{W}mov (8|M0)  r{T1}.0<1>:w  0x76543210:v")
        else:
            o = [f"{W}add (1|M0)  r{T2}.2<1>:d  {lim}<0;1,0>:d  -{cnt}<0;1,0>:d",
                 f"{W}cmp (1|M0)  (lt)f1.0  null<1>:d  r{T2}.2<0;1,0>:d  {8 * NK}:w",
                 "(W&~f1.0) jmpi  LATT_NOMASK",
                 f"{W}mov (8|M0)  r{T1}.0<1>:w  0x76543210:v"]
        if rowvis:      # each row against its own limit: rem_r = lim_r - (first key of this block), as above
            for r in range(RQ):
                o.append(f"{W}add (1|M0)  r{T2}.2<1>:d  r{T2}.{LV + r}<0;1,0>:d  -{cnt}<0;1,0>:d")
                if SLIM_CNT:
                    o.append(f"{W}add (1|M0)  r{T2}.2<1>:d  r{T2}.2<0;1,0>:d  {8 * NK}:w")
                for k in range(NK):
                    if k:
                        o.append(f"{W}add (1|M0)  r{T2}.2<1>:d  r{T2}.2<0;1,0>:d  -8:w")
                    o += [f"{W}cmp (8|M0)  (ge)f1.0  null<1>:w  r{T1}.0<8;8,1>:w  r{T2}.4<0;1,0>:w",
                          f"(W&f1.0) mov (8|M0)  r{S[(r, k)]}.0<1>:f  -1e+30:f"]
            o.append("LATT_NOMASK:")
            return o
        for k in range(NK):
            if k:
                o.append(f"{W}add (1|M0)  r{T2}.2<1>:d  r{T2}.2<0;1,0>:d  -8:w")
            o.append(f"{W}cmp (8|M0)  (ge)f1.0  null<1>:w  r{T1}.0<8;8,1>:w  r{T2}.4<0;1,0>:w")
            o += [f"(W&f1.0) mov (8|M0)  r{S[(r, k)]}.0<1>:f  -1e+30:f" for r in range(RQ)]
        o.append("LATT_NOMASK:")
        return o

    causal = P.get("CAUSAL", 0) == 1
    if causal:
        assert Mx != T2 or MO > 2, "rem (T2.2) would overlap the row maxima"
    for s in range(NS):
        out += load(s)
    out.append("LATT_LOOP:")
    if not SLIM_MUL:    # SLIM_MUL: the block's first K step writes S with mul (S = +0 + p and S = p differ only in
        # the sign of a zero: exp(s - m) and the max are the same for +0 and -0)
        out += [f"        mov (8|M0)  r{S[k]}.0<1>:f  0.0:f" for k in sorted(S)]
    for s in range(nsteps):
        out += kmads(s) if s < nks else vmads(s - nks)
        out += load(s + NS)
        if s == nks - 1:
            if SLIM_CNT:    # loop test moved here; flag f1.1 serves the mask and the loop jump (nothing else writes it)
                out += [f"{W}add (1|M0)  {cnt}<1>:d  {cnt}<0;1,0>:d  {8 * NK}:w",
                        f"{W}cmp (1|M0)  (lt)f1.1  null<1>:d  {cnt}<0;1,0>:d  {lim}<0;1,0>:d"]
            out += (mask() if causal else []) + softmax()
    if SLIM_CNT:
        out.append("(W&f1.1) jmpi  LATT_LOOP")
    else:
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
    # epilogue send headers that the loop may have used (live_in): zero, as IGC leaves the unused fields as found
    out += [f"{W}mov (8|M0)  r{x}.0<1>:ud  0x0:ud" for x in sorted(epi_hdr)]
    # Q is left permuted: nothing after the loop (our tail, IGC's epilogue) may read it before writing it
    tail = out[out.index("(W&f1.1) jmpi  LATT_LOOP" if SLIM_CNT else "(W&f0.0) jmpi  LATT_LOOP") + 1:]
    assert_not_read_after(qregs, tail + epi, "Q")
    isa.check_all(out, "the generated loop")
    over = max(s * 128 + k * D * 16 for s in range(NS) for k in range(NK)) + 128
    assert over <= NK * D * 16, (over, NK * D * 16)
    out[0] += f"; Kt over-read {over} B (pad >= {NK * D * 16} B)"
    info = (f"Q swaps {len(loc) // 2} (unpaired {left}), {nfree} registers free for the loop; epilogue headers "
            f"zeroed {sorted(epi_hdr)}; Kt read up to {over} B past the end (host pad >= NK*D*16 = {NK * D * 16} B)")
    return pro, out, epi, info


def main():
    mode, src, opts = sys.argv[1], sys.argv[2], sys.argv[3]
    check_ocloc()
    d = tempfile.mkdtemp()
    try:
        dd = os.path.join(d, "d")
        disassemble(src, opts, dd)
        t = open(os.path.join(dd, f".text.{KERNEL}.asm")).read().splitlines()
        pro, loop, epi = split_loop(t)
        pro0, loop0, epi0 = list(pro), list(loop), list(epi)
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
        if mode.startswith("sched") and os.environ.get("ATTN_POISON") == "1":
            # test binary only (with ATTN_LOOPCHECK=0): first S mad of the generated loop with -q -> O must be wrong
            i = next(i for i, l in enumerate(loop) if MAD.match(l))
            loop[i] = re.sub(r"(\s)(r\d+\.\d+<0>:f)$", r"\1-\2", loop[i])
            assert loop[i].endswith("<0>:f") and " -r" in loop[i], loop[i]
        n, c = conflicts(loop)
        if mode == "remap" or mode.startswith("sched"):   # plain/poison: conflicts expected, not checked
            isa.check_generated(loop, dd, KERNEL, f"attn_gen {mode} {opts}")
        if mode.startswith("sched") and os.environ.get("ATTN_LOOPCHECK", "1") == "1":
            import loopcheck   # noqa: E402  (product-level equivalence with IGC's loop, docs/kernel-guide.md section 3)
            try:
                ok, msg = loopcheck.verify_attn_parts(pro0, loop0, epi0, pro, loop, params(opts))
            except loopcheck.Unsupported as e:
                ok, msg = False, f"cannot check: {e}"
            if not ok:
                raise SystemExit(f"attn_gen {mode} {opts}: generated loop rejected by loopcheck: {msg}")
            print(f"loopcheck: {msg}")
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
