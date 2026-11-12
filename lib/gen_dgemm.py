"""Składa bench/dgemm.cl: bench/dgemm_core.cl (pakowanie, główny kernel FP64) + kernele pomocnicze z bench/sgemm.cl
(epilog wsadowy i trójkątny, TRSM, rozwijanie, sklejanie, odwrotności bloków) z podstawieniem float -> double i
przyrostków nazw _f -> _d (_fb -> _db, _trif -> _trid).

Uruchom: python3 gen_dgemm.py ../bench/sgemm.cl ../bench/dgemm_core.cl > ../bench/dgemm.cl   (robi to Makefile)
"""
import re
import sys

sg = open(sys.argv[1]).read()
rdzen = open(sys.argv[2]).read()


def wytnij(nazwa):
    """Tekst funkcji __kernel o danej nazwie (z atrybutami w poprzedniej linii), do pasującego nawiasu klamrowego."""
    m = re.search(r"(__kernel[^{;]*?\b" + nazwa + r"\s*\()", sg)
    if not m:
        sys.exit(f"gen_dgemm: brak kernela {nazwa}")
    poczatek = sg.rfind("\n", 0, m.start()) + 1
    if sg[poczatek:m.start()].strip() == "" and sg[:poczatek].rstrip().endswith("))"):   # atrybut w osobnej linii
        poczatek = sg.rfind("\n", 0, poczatek - 1) + 1
    i, glebokosc = sg.index("{", m.end()), 0
    for j in range(i, len(sg)):
        glebokosc += {"{": 1, "}": -1}.get(sg[j], 0)
        if glebokosc == 0:
            return sg[poczatek:j + 1] + "\n"
    sys.exit(f"gen_dgemm: niedomknięty {nazwa}")


nazwy = ["kpg_epilog_fb", "kpg_epilog_trif", "trsm_L_f", "trsm_Y_f", "trsm_zapisz_f", "gen9blas_pelna_f", "gen9blas_sklej_f",
         "trsm_inv0_f", "trsm_inv1_f", "trsm_inv2_f", "trsm_inv3_f"]
makra = "".join(l + "\n" for l in sg.splitlines() if l.startswith(("#define INV_L", "#define INV_X")))
tekst = makra + "".join(wytnij(n) for n in nazwy)
tekst = re.sub(r"\bfloat\b", "double", tekst)
tekst = re.sub(r"(\d+\.\d+)f\b", r"\1", tekst)
tekst = re.sub(r"\b(\w+)_fb\b", r"\1_db", tekst)
tekst = re.sub(r"\b(\w+)_trif\b", r"\1_trid", tekst)
tekst = re.sub(r"\b(\w+)_f\b", r"\1_d", tekst)
print("// Wygenerowane przez lib/gen_dgemm.py -- nie edytować (źródła: dgemm_core.cl, sgemm.cl).")
print(rdzen)
print("// ---------- z sgemm.cl (float -> double) ----------")
print(tekst)
