"""Microbenchmarks in Gen9 assembly: inserts a hand-generated loop body into the fma32.d template (kernel k_fma).

Template (asm/fma32.cl → asm/fma32.d): the loop `L560: ... jmpi L560` executed n times (argument n).
The loop may use registers r51–r126. r0–r50 and r127 belong to the template prologue and epilogue.

Usage: python3 -I micro.py <test> <result.bin>   (list of tests: python3 -I micro.py --list)
Prints the number of instructions in the loop body. Measurement: bench/clbench <result.bin> k_fma <global> <local> <n> <ops>.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.realpath(__file__))
TEMPLATE = os.path.join(HERE, "fma32.d")


COMPACT = os.environ.get("COMPACT") == "1"   # COMPACT=1: append {Compacted} (IGA will reject it if not possible)


def mad(dst, s0, s1, s2, t="f", n=16, m="M0"):
    """mad: dst = s1*s2 + s0 (operand order as in IGA). sN are ready operand strings with a region."""
    return f"        mad ({n}|{m})  r{dst}.0<1>:{t}  {reg(s0, 0)}:{t}  {reg(s1, 1)}:{t}  {reg(s2, 2)}:{t}" + ("  {Compacted}" if COMPACT else "")


# IGA region syntax for 3-source Gen9 instructions (checked with iga64 -p=9 and in compiler code):
#   src0, src1: vector <2;1>, broadcast scalar <0;0>;   src2: vector <1>, scalar <0>.
# Generators return a template with a {R} slot, filled in depending on the operand position.
def vec(r):          # SIMD16 FP32 vector operand: 2 registers starting at r
    return (f"r{r}.0", "vec")


def bcast(r, sub=0):  # broadcast scalar from r.sub (like the A element in GEMM)
    return (f"r{r}.{sub}", "bcast")


def reg(op, pos):
    r, kind = op
    if pos == 2:
        return r + ("<1>" if kind == "vec" else "<0>")
    return r + ("<2;1>" if kind == "vec" else "<0;0>")


def banks(src1, src2, count=256, dst0=52, ndst=8, src0=110):
    """Independent mad: results rotate over ndst register pairs, sources constant."""
    out = []
    for i in range(count):
        d = dst0 + 2 * (i % ndst)
        out.append(mad(d, vec(src0) if src0 else bcast(120, 1), src1, src2))
    return out


def gemm(nacc, a=73, bvar=False, count=256):
    """Loop as in GEMM: acc_j += B * A[k]. acc in-place (dst == src0), B vector (src1), A broadcast scalar (src2)."""
    out = []
    for i in range(count):
        acc = 84 + 2 * (i % nacc)
        b = (56 + 4 * ((i // nacc) % 4)) if bvar else 70
        out.append(mad(acc, vec(acc), vec(b), bcast(a, i % 8)))
    return out


def compiler(compact):
    """Loop body copied from the template (IGC code), with or without {Compacted}."""
    src = open(os.path.join(TEMPLATE, ".text.k_fma.asm")).read()
    m = re.search(r"L560:\n(.*?)\n\(W&f0\.0\) jmpi", src, re.S)
    lines = [l for l in m.group(1).splitlines() if l.strip() and "mad" in l]
    return [l if compact else l.replace("{Compacted}", "") for l in lines]


def bank(r):
    return ((r & 0x40) >> 5) | (r & 1)


TESTS = {
    # name: (description, loop body generator)
    "bank_different_parity": ("src1=r70 (bank 0/1), src2=r73 (bank 1/0): different banks in both halves",
                              lambda: banks(vec(70), vec(73))),
    "bank_same": ("src1=r70, src2=r72: both even in the lower half -> same bank in both halves",
                     lambda: banks(vec(70), vec(72))),
    "bank_different_halves": ("src1=r70 (lo), src2=r100 (hi): banks 0/1 vs 2/3, no conflict",
                          lambda: banks(vec(70), vec(100))),
    "bank_hi_same": ("src1=r100, src2=r102: both in the upper half, even -> conflict",
                        lambda: banks(vec(100), vec(102))),
    "bank_same_register": ("src1=src2=r70: same register (does the EU skip the second read?)",
                             lambda: banks(vec(70), vec(70))),
    "bcast_conflict": ("src1=r70 vector, src2=r72.0 broadcast scalar (same bank) - GEMM pattern",
                       lambda: banks(vec(70), bcast(72))),
    "bcast_no_conflict": ("src1=r70 vector, src2=r73.0 broadcast scalar (different bank)",
                            lambda: banks(vec(70), bcast(73))),
    "bcast_both_scalar": ("src1=r70.0, src2=r72.0: both broadcast (as in the compiler template)",
                         lambda: banks(bcast(70), bcast(72))),
    "src0_same_bank": ("src0=r72 (same bank as src2? no: as src1 r70), src1=r70, src2=r101 - does src0 matter",
                          lambda: banks(vec(70), vec(101), src0=72)),
    "parity_71_100": ("src1=r71 (odd, lo), src2=r100 (even, hi): does only parity decide",
                    lambda: banks(vec(71), vec(100))),
    "parity_70_101": ("src1=r70 (even, lo), src2=r101 (odd, hi)",
                    lambda: banks(vec(70), vec(101))),
    "parity_71_73": ("src1=r71, src2=r73: both odd, lo",
                   lambda: banks(vec(71), vec(73))),
    "parity_70_78": ("src1=r70, src2=r78: even, distance of 8 registers",
                   lambda: banks(vec(70), vec(78))),
    "parity_70_75": ("src1=r70, src2=r75: different parity, distance of 5",
                   lambda: banks(vec(70), vec(75))),
    "dst8": ("like bank_different_parity, results rotate over 8 pairs (default)",
             lambda: banks(vec(70), vec(73), ndst=8)),
    "dst4": ("results rotate over 4 pairs (closer to WAW)", lambda: banks(vec(70), vec(73), ndst=4)),
    "dst2": ("results rotate over 2 pairs", lambda: banks(vec(70), vec(73), ndst=2)),
    "dst16_hi": ("results over 16 pairs in r80-r111? no: r84+, sources r70/r73 + src0 r120",
                 lambda: banks(vec(70), vec(73), dst0=84, ndst=12, src0=120)),
    "m0_m16": ("M0/M16 pairs as in compiler code (SIMD32), different banks",
               lambda: [mad(52 + 2 * (i // 2 % 8), vec(110), vec(70 if i % 2 == 0 else 74), vec(73 if i % 2 == 0 else 77),
                            m="M0" if i % 2 == 0 else "M16") for i in range(256)]),

    "s0scalar_v_b": ("src0 scalar, src1 vector r70, src2 scalar r73.0: like compiler code (1 vector per instr.)",
                   lambda: banks(vec(70), bcast(73), src0=None)),
    "s0scalar_v_v": ("src0 scalar, src1 vector r70, src2 vector r73: 2 vectors per instr.",
                   lambda: banks(vec(70), vec(73), src0=None)),
    "gemm_acc16": ("GEMM: acc_i = acc_i + B(r70 vector)*A(r73.k scalar), 16 accumulators (r84-r115), in-place",
                   lambda: gemm(16)),
    "gemm_acc8": ("as above, 8 accumulators", lambda: gemm(8)),
    "gemm_acc16_conflict": ("GEMM with A in the same bank as B (r72.k)", lambda: gemm(16, a=72)),
    "gemm_acc16_bvar": ("GEMM: B changes every 16 instructions (4 B registers alternating: r56,r60,r64,r68 even), A r73.k",
                        lambda: gemm(16, bvar=True)),
    "compiler_full": ("exactly the compiler loop (8 chains, a/b scalars), but without {Compacted}",
                         lambda: compiler(False)),
    "compiler_compact": ("exactly the compiler loop with {Compacted}", lambda: compiler(True)),
}


def build(body, result):
    tmp = tempfile.mkdtemp()
    d = os.path.join(tmp, "d")
    shutil.copytree(TEMPLATE, d)
    asm_path = os.path.join(d, ".text.k_fma.asm")
    src = open(asm_path).read()
    # Replace the WHOLE loop body between "L560:" and "jmpi L560" with our code plus the loop control.
    # Note: the compiler also interleaves mad AFTER the loop counter (before cmp/jmpi), so it is not enough
    # to cut the code up to "add r8.1". We keep only: counter add, 2× cmp, jmpi.
    m = re.search(r"L560:\n(.*?)\n(\(W&f0\.0\) jmpi\s+L560[^\n]*)", src, re.S)
    if not m:
        sys.exit("loop L560 not found in the template")
    control = [l for l in m.group(1).splitlines() if re.search(r"r8\.1<1>:d|cmp \(", l)]
    assert len(control) == 3, control
    src = src[: m.start(1)] + "\n".join(body + control) + "\n" + src[m.start(2):]
    open(asm_path, "w").write(src)
    name = os.path.basename(result)
    subprocess.run(["ocloc", "asm", "-file", name, "-dump", d, "-device", "whl"],
                   check=True, stdout=subprocess.DEVNULL)
    shutil.move(os.path.join(d, name), result)
    shutil.rmtree(tmp)


if __name__ == "__main__":
    if sys.argv[1] == "--list":
        for k, (desc, _) in TESTS.items():
            print(f"{k:24s} {desc}")
        sys.exit()
    test, result = sys.argv[1], sys.argv[2]
    body = TESTS[test][1]()
    build(body, result)
    print(len(body))
