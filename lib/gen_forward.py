"""Generuje gen9blas_blas_forward.c: skoki do BLAS na CPU (OpenBLAS) dla symboli BLAS, których gen9blas_blas.c nie implementuje.

Uruchom: python3 gen_forward.py blas_symbols.txt gen9blas_blas.c [gen9blas_blas_d.inc ...] > gen9blas_blas_forward.c   (robi to Makefile)
blas_symbols.txt: eksporty systemowego libblas.so.3 (nm -D --defined-only, OpenBLAS 0.3, 393 symbole: CBLAS
i nazwy fortranowe). Funkcje zaimplementowane w gen9blas_blas.c są oznaczone makrem BLAS_API. Każdy pozostały symbol to
`jmp *gen9blas_cel[i](%rip)` -- argumenty i wynik przechodzą bez zmian, sygnatury nie są potrzebne. Adresy gen9blas_cel
wypełnia gen9blas_blas.c (konstruktor) przez dlsym z biblioteki CPU.
"""
import re
import sys

symbole = [l.strip() for l in open(sys.argv[1]) if l.strip()]
wlasne = set()
for plik in sys.argv[2:]:
    wlasne |= set(re.findall(r"^BLAS_API\b[^(]*?\b(\w+)\s*\(", open(plik).read(), re.M))
brak = wlasne - set(symbole)
if brak:
    sys.exit(f"gen_przekaz: funkcje spoza listy BLAS: {sorted(brak)}")
przekaz = [s for s in symbole if s not in wlasne]
print("// Wygenerowane przez gen_forward.py -- nie edytować. Skoki do BLAS na CPU dla funkcji spoza gen9blas_blas.c.")
print(f"// Własne: {len(wlasne)}, przekazywane: {len(przekaz)}.")
print(f"#define GEN9BLAS_NPRZEKAZ {len(przekaz)}")
print("__attribute__((visibility(\"hidden\"))) void *gen9blas_cel[GEN9BLAS_NPRZEKAZ];")
print("__attribute__((visibility(\"hidden\"))) const int gen9blas_nprzekaz = GEN9BLAS_NPRZEKAZ;")
print("__attribute__((visibility(\"hidden\"))) const char *const gen9blas_cel_nazwy[GEN9BLAS_NPRZEKAZ] = {")
for s in przekaz:
    print(f'    "{s}",')
print("};")
print("__asm__(\".text\\n\"")
for i, s in enumerate(przekaz):
    print(f'        ".globl {s}\\n.type {s},@function\\n{s}: jmp *gen9blas_cel+{8 * i}(%rip)\\n"')
print(");")
