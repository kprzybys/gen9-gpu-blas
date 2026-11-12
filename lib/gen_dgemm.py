"""Assembles bench/dgemm.cl: bench/dgemm_core.cl (packing, main FP64 kernel) + helper kernels from bench/sgemm.cl
(batch and triangular epilog, TRSM, expansion, merging, block inverses) with float -> double substitution and
name suffixes _f -> _d (_fb -> _db, _trif -> _trid).

Run: python3 gen_dgemm.py ../bench/sgemm.cl ../bench/dgemm_core.cl > ../bench/dgemm.cl   (the Makefile does this)
"""
import re
import sys

sg = open(sys.argv[1]).read()
core = open(sys.argv[2]).read()


def extract(name):
    """Text of the __kernel function with the given name (with attributes on the previous line), up to the matching closing brace."""
    m = re.search(r"(__kernel[^{;]*?\b" + name + r"\s*\()", sg)
    if not m:
        sys.exit(f"gen_dgemm: kernel {name} not found")
    start = sg.rfind("\n", 0, m.start()) + 1
    if sg[start:m.start()].strip() == "" and sg[:start].rstrip().endswith("))"):   # attribute on a separate line
        start = sg.rfind("\n", 0, start - 1) + 1
    i, depth = sg.index("{", m.end()), 0
    for j in range(i, len(sg)):
        depth += {"{": 1, "}": -1}.get(sg[j], 0)
        if depth == 0:
            return sg[start:j + 1] + "\n"
    sys.exit(f"gen_dgemm: unclosed {name}")


names = ["kpg_epilog_fb", "kpg_epilog_trif", "trsm_L_f", "trsm_Y_f", "trsm_zapisz_f", "gen9blas_pelna_f", "gen9blas_sklej_f",
         "trsm_inv0_f", "trsm_inv1_f", "trsm_inv2_f", "trsm_inv3_f"]
macros = "".join(l + "\n" for l in sg.splitlines() if l.startswith(("#define INV_L", "#define INV_X")))
text = macros + "".join(extract(n) for n in names)
text = re.sub(r"\bfloat\b", "double", text)
text = re.sub(r"(\d+\.\d+)f\b", r"\1", text)
text = re.sub(r"\b(\w+)_fb\b", r"\1_db", text)
text = re.sub(r"\b(\w+)_trif\b", r"\1_trid", text)
text = re.sub(r"\b(\w+)_f\b", r"\1_d", text)
# NOTE: the two header comments below are emitted into bench/dgemm.cl, which must stay byte-identical until the .cl sources are translated.
print("// Wygenerowane przez lib/gen_dgemm.py -- nie edytować (źródła: dgemm_core.cl, sgemm.cl).")
print(core)
print("// ---------- z sgemm.cl (float -> double) ----------")
print(text)
