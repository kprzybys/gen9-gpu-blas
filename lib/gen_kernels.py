"""Generuje kernels.c: kod ISA i opisy wszystkich jąder libgen9blas dla warstwy kcl (bez NEO, bezpośrednio przez i915).

Uruchom (robi to Makefile): python3 gen_kernels.py <katalog bench> <katalog roboczy> > kernels.c
Programy jak w gen9blas.c: źródła OpenCL C kompilowane `ocloc -device whl` z tymi samymi opcjami co dawniej w NEO
(clBuildProgram) oraz binarki z naszego asemblera (gen_*.bin). Dla każdego jądra z .ze_info: SIMD, rozmiary danych
wspólnych i na wątek, offsety argumentów (wartość / wskaźnik stateless / stateful z binding table), SLM, bariera,
pamięć prywatna, stałe programu (.bss.const / .data.const). Jądra z nieobsługiwanymi cechami (obrazy, sampler, scratch)
dostają flagę `nieobslugiwane` -- kcl odmówi ich utworzenia zamiast uruchomić z niepełnym stanem.
"""
import os
import struct
import subprocess
import sys

import yaml

BENCH, ROB = sys.argv[1], sys.argv[2]
os.makedirs(ROB, exist_ok=True)

# (plik, opcje) -- dokładnie jak wywołania program_zr / program_bin w gen9blas.c
PROGRAMY = [
    ("hgemm.cl", ""), ("sgemm.cl", ""), ("dgemm.cl", ""),
    ("level12.cl", "-cl-std=CL3.0 -DPOL"), ("level12.cl", "-cl-std=CL3.0"), ("level12.cl", "-cl-std=CL3.0 -DDBL"),
    ("gen_h.bin", None), ("gen_hd.bin", None), ("gen_s.bin", None),
]
WARTOSCI = ("arg_byvalue",)
SPECJALNE = {"local_size": "ls", "enqueued_local_size": "els", "global_id_offset": "gid", "global_size": "gs",
             "num_work_groups": "ng"}


def sekcje(sciezka):
    d = open(sciezka, "rb").read()
    shoff, = struct.unpack_from("<Q", d, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", d, 0x3A)
    sek = [struct.unpack_from("<IIQQQQIIQQ", d, shoff + i * shentsize) for i in range(shnum)]
    stroff = sek[shstrndx][4]
    wynik = {}
    for s in sek:
        nazwa = d[stroff + s[0]:d.index(b"\0", stroff + s[0])].decode()
        typ, off, size = s[1], s[4], s[5]
        wynik[nazwa] = (typ, d[off:off + size] if typ != 8 else b"", size)   # 8 = SHT_NOBITS
    return wynik


def c_bajty(nazwa, b):
    linie = [f"static const uint8_t {nazwa}[{max(len(b), 1)}] = {{"]
    for i in range(0, len(b), 32):
        linie.append("    " + ",".join(str(x) for x in b[i:i + 32]) + ",")
    if not b:
        linie.append("    0,")
    linie.append("};")
    return "\n".join(linie)


print("// Wygenerowane przez lib/gen_kernels.py -- nie edytować.")
print('#include "kcl_desc.h"\n')
progi = []
for pi, (plik, opcje) in enumerate(PROGRAMY):
    baza = f"p{pi}"
    if opcje is None:
        binarka = os.path.join(BENCH, plik)
    else:
        binarka = os.path.join(ROB, baza + ".bin")
        cmd = ["ocloc", "compile", "-file", os.path.join(BENCH, plik), "-device", "whl", "-output_no_suffix",
               "-output", baza, "-out_dir", ROB]
        if opcje:
            cmd += ["-options", opcje]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"gen_kernele: ocloc {plik} {opcje}: {r.stdout}{r.stderr}")
    dump = os.path.join(ROB, baza + ".d")
    subprocess.run(["rm", "-rf", dump])
    subprocess.run(["ocloc", "disasm", "-file", binarka, "-device", "whl", "-dump", dump], capture_output=True)
    info = yaml.safe_load(open(os.path.join(dump, ".ze_info")))
    sek = sekcje(binarka)
    stale = b""
    for nazwa in (".data.const", ".bss.const"):
        if nazwa in sek:
            typ, dane, size = sek[nazwa]
            stale = dane if dane else bytes(size)
    print(c_bajty(f"{baza}_const", stale))
    opisy = []
    for ki, k in enumerate(info["kernels"]):
        n, env = k["name"], k["execution_env"]
        tekst = sek.get(".text." + n)
        if not tekst:
            continue
        isa = tekst[1]
        nieobs = []
        # spill_size: kompilator odkłada rejestry do pamięci scratch, której kcl nie konfiguruje (MEDIA_VFE_STATE) --
        # takie jądro liczyło po cichu źle (chudy GEMM MT = 8, 2026-11-12); teraz błąd przy kcl_CreateKernel.
        for klucz in ("per_thread_scratch_size", "spill_size", "has_stack_calls", "inline_data_payload_size"):
            if env.get(klucz):
                nieobs.append(klucz)
        bti = {b["arg_index"]: b["bti_value"] for b in k.get("binding_table_indices", [])}
        arg = {}   # indeks -> [off, size, wsk, bti, off_adr, off_boff]
        spec = {v: [] for v in SPECJALNE.values()}
        inne = {"const_base": -1, "private_base_stateless": -1, "work_dimensions": -1}
        koniec = 0
        for a in k.get("payload_arguments", []):
            t = a["arg_type"]
            koniec = max(koniec, a["offset"] + a["size"])
            if t in ("arg_bypointer", "arg_byvalue"):
                e = arg.setdefault(a["arg_index"], [-1, 0, 0, -1, -1, -1])
                if t == "arg_bypointer":
                    e[2] = 1
                    tryb, prz = a.get("addrmode"), a.get("addrspace")
                    if prz not in ("global", "constant"):
                        nieobs.append(f"arg{a['arg_index']}:{prz}")
                    if tryb == "stateful":
                        e[3] = bti[a["arg_index"]]
                    elif tryb == "stateless":
                        e[0], e[1] = a["offset"], a["size"]
                    else:
                        nieobs.append(f"arg{a['arg_index']}:{tryb}")
                else:
                    if e[0] >= 0:   # wartość w kilku kawałkach (struktury) -- nieobsługiwane
                        nieobs.append(f"arg{a['arg_index']}:kawalki")
                    e[0], e[1] = a["offset"], a["size"]
            elif t == "buffer_address":
                arg.setdefault(a["arg_index"], [-1, 0, 1, -1, -1, -1])[4] = a["offset"]
            elif t == "buffer_offset":
                arg.setdefault(a["arg_index"], [-1, 0, 1, -1, -1, -1])[5] = a["offset"]
            elif t in SPECJALNE:
                spec[SPECJALNE[t]].append(a["offset"])
            elif t in inne:
                inne[t] = a["offset"]
            else:
                nieobs.append(t)
        nargs = max(arg) + 1 if arg else 0
        for i in range(nargs):
            arg.setdefault(i, [-1, 0, 0, -1, -1, -1])
        pt = 0
        lid = 0
        for a in k.get("per_thread_payload_arguments", []):
            if a["arg_type"] == "local_id":
                lid = 1
                pt += a["size"]
            else:
                nieobs.append("per_thread:" + a["arg_type"])
        if nargs > 24:
            nieobs.append("ponad 24 argumenty")
        print(c_bajty(f"{baza}_k{ki}_isa", isa))
        def trzy(xs):
            return "{" + ",".join(str(x) for x in (xs + [-1, -1, -1])[:3]) + "}"
        args = ",".join("{" + ",".join(str(x) for x in arg[i]) + "}" for i in range(nargs)) or "{-1,0,0,-1,-1,-1}"
        opisy.append(f'{{"{n}", {baza}_k{ki}_isa, {len(isa)}, {env["simd_size"]}, {(koniec + 31) // 32 * 32}, {pt}, '
                     f'{lid}, {nargs}, {{{args}}}, {trzy(spec["ls"])}, {trzy(spec["els"])}, {trzy(spec["gid"])}, '
                     f'{trzy(spec["gs"])}, {trzy(spec["ng"])}, {inne["work_dimensions"]}, {inne["const_base"]}, '
                     f'{inne["private_base_stateless"]}, {env.get("private_size", 0)}, {env.get("slm_size", 0)}, '
                     f'{1 if env.get("barrier_count") else 0}, {max(bti.values()) + 1 if bti else 0}, '
                     f'"{" ".join(nieobs)}"}}')
    print(f"static const struct kcl_jadro_opis {baza}_jadra[] = {{\n    " + ",\n    ".join(opisy) + "\n};")
    progi.append(f'{{"{plik}", {"NULL" if opcje is None else chr(34) + opcje + chr(34)}, {baza}_jadra, '
                 f'{len(opisy)}, {baza}_const, {len(stale)}}}')
print("const struct kcl_program_opis kcl_programy[] = {\n    " + ",\n    ".join(progi) + "\n};")
print(f"const int kcl_nprogramow = {len(progi)};")
