"""Cuts the .text.<kernel> section out of a zebin binary (ELF from ocloc) into a raw ISA file."""
import struct
import sys

d = open(sys.argv[1], "rb").read()
shoff, = struct.unpack_from("<Q", d, 0x28)
shentsize, shnum, shstrndx = struct.unpack_from("<HHH", d, 0x3A)
sections = [struct.unpack_from("<IIQQQQIIQQ", d, shoff + i * shentsize) for i in range(shnum)]
stroff = sections[shstrndx][4]
for s in sections:
    name = d[stroff + s[0]:d.index(b"\0", stroff + s[0])].decode()
    if name == ".text." + sys.argv[2]:
        open(sys.argv[3], "wb").write(d[s[4]:s[4] + s[5]])
        print(f"{name}: {s[5]} B")
        break
else:
    sys.exit("section not found")
