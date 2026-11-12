"""Wycina sekcję .text.<jądro> z binarki zebin (ELF z ocloc) do pliku surowego ISA."""
import struct
import sys

d = open(sys.argv[1], "rb").read()
shoff, = struct.unpack_from("<Q", d, 0x28)
shentsize, shnum, shstrndx = struct.unpack_from("<HHH", d, 0x3A)
sek = [struct.unpack_from("<IIQQQQIIQQ", d, shoff + i * shentsize) for i in range(shnum)]
stroff = sek[shstrndx][4]
for s in sek:
    nazwa = d[stroff + s[0]:d.index(b"\0", stroff + s[0])].decode()
    if nazwa == ".text." + sys.argv[2]:
        open(sys.argv[3], "wb").write(d[s[4]:s[4] + s[5]])
        print(f"{nazwa}: {s[5]} B")
        break
else:
    sys.exit("brak sekcji")
