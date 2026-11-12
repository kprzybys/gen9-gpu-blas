"""Generuje gen9blas_blas_d.inc: funkcje D (FP64) warstwy BLAS z kodu funkcji S w gen9blas_blas.c.

Uruchom: python3 gen_blas_d.py gen9blas_blas.c > gen9blas_blas_d.inc   (robi to Makefile)
Fragment od „POCZĄTEK TYPU” do „KONIEC TYPU” (bez bloków „tylko S”): funkcje static i typedef dostają przyrostek _d
(ten sam plik źródłowy -- bez kolizji), float -> double, GEN9BLAS_S -> GEN9BLAS_D, nazwy eksportowane i napisy s -> d
(cblas_sgemm -> cblas_dgemm, sgemm_ -> dgemm_, isamax -> idamax).
"""
import re
import sys

s = open(sys.argv[1]).read()
t = s[s.index("// ==== POCZĄTEK TYPU"):s.index("// ==== KONIEC TYPU")]
t = re.sub(r"// ---- tylko S: początek.*?// ---- tylko S: koniec\n", "", t, flags=re.S)
statyczne = set(re.findall(r"^static\b[^(=;]*?\b(\w+)\s*\(", t, re.M)) | set(re.findall(r"^typedef\b.*?\b(\w+);", t, re.M)) \
    | set(re.findall(r"^typedef\b[^;]*?\(\*(\w+)\)", t, re.M))
for n in sorted(statyczne, key=len, reverse=True):
    t = re.sub(r"\b" + n + r"\b", n + "_d", t)
t = re.sub(r"\bfloat\b", "double", t)
t = t.replace("GEN9BLAS_S", "GEN9BLAS_D")
t = re.sub(r"\bcblas_s(\w+)", r"cblas_d\1", t)
t = re.sub(r"\bcblas_isamax\b", "cblas_idamax", t)
t = re.sub(r"\bs(gemm|symm|syrk|syr2k|trmm|trsm|gemv|ger|symv|trmv|trsv|syr|syr2|gbmv|sbmv|tbmv|tbsv|spmv|tpmv|tpsv|spr|spr2)_\b", r"d\1_", t)
t = re.sub(r'"s([a-z0-9]+_?)"', r'"d\1"', t)
t = t.replace('"isamax"', '"idamax"')
t = t.replace("// ==== POCZĄTEK TYPU", "// ==== POCZĄTEK TYPU (wersja D)")
print("// Wygenerowane przez gen_blas_d.py z gen9blas_blas.c -- nie edytować.")
print(t)
