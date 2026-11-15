"""Check of the patch-token reader in gen_kernels.py (needed for Gen8, where ocloc emits only patch tokens).

Builds kernels.c for Gen9.5 twice -- from zebin (.ze_info) and from the patch-token format -- and compares the kernels
whose ISA is byte-identical in both builds (for the others the compiler produced different code, so the descriptors may
legitimately differ). Accepted differences, everything else is an error:
  - both unsupported (the two formats name the reasons differently);
  - a pointer argument stateless in one ([off, 8, 1, -1, -1, -1]) and stateful with its address at the same offset in the
    other ([-1, 0, 1, bti, off, -1]): kcl writes the address in both cases; the binding-table size may then differ;
  - patch tokens declare scratch (unsupported) where zebin lists it only in per_thread_memory_buffers, which kcl's zebin
    reader does not check (on Gen9 these kernels contain no scratch messages; on bdw they do);
  - Intel_Symbol_Table_Void_Program (an empty helper entry of zebin).
Run: python3 check_patchtokens.py <work dir>   (needs ocloc)
"""
import os
import re
import subprocess
import sys

work = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
ARG = re.compile(r"\{(-?\d+),(-?\d+),(-?\d+),(-?\d+),(-?\d+),(-?\d+)\}")


def build(target):
    out = os.path.join(work, f"kernels_{target}.c")
    with open(out, "w") as f:
        subprocess.run([sys.executable, os.path.join(here, "gen_kernels.py"), os.path.join(here, "..", "bench"),
                        os.path.join(work, target), target], stdout=f, check=True)
    text = open(out).read()
    isa = dict(re.findall(r"static const uint8_t (p\d+_k\d+_isa)\[\d+\] = \{\n(.*?)\};", text, re.S))
    descs = {}
    for prog, body in re.findall(r"static const struct kcl_kernel_desc (p\d+)_kernels\[\] = \{\n(.*?)\n\};", text, re.S):
        for line in body.split("\n"):
            m = re.match(r'\s*\{"(\w+)", (p\d+_k\d+_isa), (.*), "([^"]*)"\},?$', line)
            descs[(prog, m.group(1))] = (isa[m.group(2)], m.group(3), m.group(4))
    return descs


def normalise(desc):
    """Stateless pointer -> the stateful form with the same address offset; drop the binding-table size (last field)."""
    def arg(m):
        off, size, ptr, bti, addr, boff = (int(x) for x in m.groups())
        if ptr and bti < 0 and size == 8 and addr < 0:
            return f"{{-1,0,1,BTI,{off},{boff}}}"
        if ptr and bti >= 0:
            return f"{{-1,0,1,BTI,{addr},{boff}}}"
        return m.group(0)
    return ARG.sub(arg, desc.rsplit(",", 1)[0])


os.makedirs(work, exist_ok=True)
a, b = build("whl"), build("whl-pt")
counts = {"identical": 0, "accepted": 0, "different ISA": 0, "errors": 0}
for key in sorted(set(a) | set(b)):
    if key[1] == "Intel_Symbol_Table_Void_Program":
        continue
    if key not in a or key not in b:
        print(f"ERROR {key}: only in {'zebin' if key in a else 'patch tokens'}")
        counts["errors"] += 1
        continue
    (isa_a, d_a, u_a), (isa_b, d_b, u_b) = a[key], b[key]
    if isa_a != isa_b:
        counts["different ISA"] += 1
    elif d_a == d_b and bool(u_a) == bool(u_b):
        counts["identical"] += 1
    elif (u_a and u_b) or (normalise(d_a) == normalise(d_b) and bool(u_a) == bool(u_b)) or \
            (not u_a and u_b == "per_thread_scratch_size"):
        counts["accepted"] += 1
    else:
        print(f"ERROR {key}:\n  zebin: {d_a} \"{u_a}\"\n  ptok:  {d_b} \"{u_b}\"")
        counts["errors"] += 1
print(", ".join(f"{v} {k}" for k, v in counts.items()))
sys.exit(1 if counts["errors"] else 0)
