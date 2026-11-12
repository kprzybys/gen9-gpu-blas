"""Z binarki zebin (ocloc) robi nagłówek C: kod ISA jąder i opis ich danych (offsety argumentów z .ze_info).

Uruchom: python3 kernels.py <plik.bin> <ze_info> > kernels.h   (ze_info: z `ocloc disasm -dump <katalog>`)
Obsługiwane: wskaźniki global stateless (adres 64-bit) i stateful (binding table + buffer_address/buffer_offset),
argumenty przez wartość, local_size, enqueued_local_size, global_id_offset, private_base_stateless (zero), lokalne ID.
Inne (scratch, SLM, obrazy, sampler) -> błąd, żeby nie uruchomić jądra z niepełnym stanem.
"""
import struct
import sys

import yaml

zebin, zeinfo = sys.argv[1], sys.argv[2]
tylko = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None   # opcjonalnie: lista jąder
d = open(zebin, "rb").read()
shoff, = struct.unpack_from("<Q", d, 0x28)
shentsize, shnum, shstrndx = struct.unpack_from("<HHH", d, 0x3A)
sek = [struct.unpack_from("<IIQQQQIIQQ", d, shoff + i * shentsize) for i in range(shnum)]
stroff = sek[shstrndx][4]
teksty = {}
for s in sek:
    nazwa = d[stroff + s[0]:d.index(b"\0", stroff + s[0])].decode()
    if nazwa.startswith(".text."):
        teksty[nazwa[6:]] = d[s[4]:s[4] + s[5]]

info = yaml.safe_load(open(zeinfo))
print("// Wygenerowane przez experiments/no-neo/kernels.py -- nie edytować.")
print("#pragma once\n#include <stdint.h>\n#include \"gen9i.h\"\n")
for k in info["kernels"]:
    n, env = k["name"], k["execution_env"]
    if n not in teksty or (tylko and n not in tylko):
        continue
    for klucz in ("per_thread_scratch_size", "slm_size", "barrier_count"):
        if env.get(klucz):
            sys.exit(f"kernels.py: {n}: {klucz} = {env[klucz]} -- jeszcze nieobsługiwane")
    bti = {b["arg_index"]: b["bti_value"] for b in k.get("binding_table_indices", [])}
    arg = {}          # indeks -> [offset, rozmiar, bti, offset buffer_address, offset buffer_offset]
    spec = {"local_size": [], "enqueued_local_size": [], "global_id_offset": [], "private_base_stateless": []}
    koniec = 0
    for a in k.get("payload_arguments", []):
        t = a["arg_type"]
        if t in ("arg_bypointer", "arg_byvalue"):
            i = a["arg_index"]
            e = arg.setdefault(i, [-1, 0, -1, -1, -1])
            if t == "arg_bypointer" and a.get("addrmode") == "stateful":
                if a.get("addrspace") != "global":
                    sys.exit(f"kernels.py: {n}: argument {i}: {a.get('addrspace')} -- jeszcze nieobsługiwane")
                e[2] = bti[i]
            elif t == "arg_bypointer" and a.get("addrmode") != "stateless":
                sys.exit(f"kernels.py: {n}: argument {i}: tryb {a.get('addrmode')}")
            else:
                e[0], e[1] = a["offset"], a["size"]
        elif t == "buffer_address":
            arg.setdefault(a["arg_index"], [-1, 0, -1, -1, -1])[3] = a["offset"]
        elif t == "buffer_offset":
            arg.setdefault(a["arg_index"], [-1, 0, -1, -1, -1])[4] = a["offset"]
        elif t in spec:
            spec[t].append(a["offset"])
        else:
            sys.exit(f"kernels.py: {n}: nieobsługiwany argument {t}")
        koniec = max(koniec, a["offset"] + a["size"])
    pt = sum(a["size"] for a in k.get("per_thread_payload_arguments", []))
    isa = teksty[n]
    print(f"static const uint8_t {n}_isa[{len(isa)}] = {{")
    for i in range(0, len(isa), 24):
        print("    " + ", ".join(f"0x{b:02x}" for b in isa[i:i + 24]) + ",")
    print("};")
    if sorted(arg) != list(range(len(arg))):
        sys.exit(f"kernels.py: {n}: argumenty nie po kolei: {sorted(arg)}")
    offs = ", ".join("{" + ", ".join(str(x) for x in arg[i]) + "}" for i in sorted(arg))
    def lista(xs):
        return "{" + ", ".join(str(x) for x in (xs + [-1, -1, -1])[:3]) + "}"
    print(f"static const struct kpgi_jadro {n}_jadro = {{ \"{n}\", {n}_isa, sizeof {n}_isa, {env['simd_size']}, "
          f"{env.get('grf_count', 128)}, {(koniec + 31) // 32 * 32}, {pt}, {len(arg)}, {{ {offs} }}, "
          f"{lista(spec['local_size'])}, {lista(spec['enqueued_local_size'])}, {lista(spec['global_id_offset'])}, "
          f"{max(bti.values()) + 1 if bti else 0} }};\n")
