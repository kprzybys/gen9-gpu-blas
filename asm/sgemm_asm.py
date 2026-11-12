"""SGEMM 16x16 (FP32) w ręcznym asemblerze Gen9: generator pętli wstawianej do szablonu kompilatora (sgemm_16x16p).

Jak hgemm_asm.py, ale rejestry wskaźników, licznika i granicy pętli są rozpoznawane automatycznie z pętli kompilatora:
  - wskaźnik = rejestr :q zwiększany w pętli o 1024 (krok k = 16 × 64 B),
  - A czy B: dane z send-a czytającego ten wskaźnik trafiają do mad jako skalar (src2 -> A) albo wektor (src1 -> B),
  - licznik i granica: operandy `cmp (lt)` w pętli.
Tryby: mapa <wynik.bin> | petla <wynik.bin>   (petla wymaga asm/sgemm_map.txt z trybu mapa: "i r")

Pętla (krok k = 16; połówki kl 0..7 / 8..15 w buforach X / Y):
  Dla FP32 wiersz B to 2 GRF. Konflikt banku liczy się dla pierwszego GRF src1 i rejestru skalara src2.
  Wiersze B o parzystym kl -> blok BE (baza parzysta), nieparzyste -> blok BO (baza nieparzysta); pack_bf2 przestawia
  wiersze w bloku 8 (0,2,4,6,1,3,5,7). A (pack_af2): kl parzyste w GRF nieparzystych i odwrotnie (blok A z bazą parzystą).
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

TU = os.path.dirname(os.path.realpath(__file__))
SZABLON = os.path.join(TU, "sgemm.d")
KERNEL = "sgemm_16x16p"
MAPA = os.path.join(TU, "sgemm_map.txt")
W = "(W)     "


def wczytaj():
    t = open(os.path.join(SZABLON, f".text.{KERNEL}.asm")).read().splitlines()
    w = next(i for i, l in enumerate(t) if re.search(r"^\s+while \(16\|M0\)\s+L\d+", l))
    etyk = re.search(r"while \(16\|M0\)\s+(L\d+)", t[w]).group(1)
    s = next(i for i, l in enumerate(t) if l.startswith(etyk + ":"))
    return t[:s], t[s:w + 1], t[w + 1:]


def analiza(petla):
    """Zwraca (wskaźnik_A, wskaźnik_B, licznik, granica) jako napisy operandów, np. 'r34.0'."""
    wsk = sorted({int(m.group(1)) for l in petla
                  for m in [re.search(r"add \(8\|M0\)\s+r(\d+)\.0<1>:q\s+r\1\.0<4;4,1>:q\s+1024:w", l)] if m})
    defs = {}      # rejestr nagłówka -> (źródło, offset), ostatnia definicja przed send
    rola = {}
    uzycie = {}    # rejestr danych -> 'A' (skalar w mad) / 'B' (wektor w mad)
    for l in petla:
        m = re.search(r"mad \(16\|M0\)\s+r\d+\.0<1>:f\s+r\d+\.0<2;1>:f\s+r(\d+)\.0<2;1>:f\s+r(\d+)\.\d+<0>:f", l)
        if m:
            uzycie.setdefault(int(m.group(1)), "B")
            uzycie.setdefault(int(m.group(2)), "A")
    for l in petla:
        m = re.search(r"(mov|add) \(1\|M0\)\s+r(\d+)\.(\d)<1>:u?q\s+r(\d+)\.(\d)<0;1,0>:u?q(?:\s+(\S+):)?", l)
        if m:
            dst = (int(m.group(2)), int(m.group(3)))
            src = (int(m.group(4)), int(m.group(5)))
            off = int(m.group(6), 0) if m.group(6) else 0
            base, boff = defs.get(src, (src, 0))
            defs[dst] = (base, boff + off)
            continue
        m = re.search(r"send \(16\|M0\)\s+r(\d+):w\s+r(\d+)\s", l)
        if m:
            base, _ = defs.get((int(m.group(2)), 0), ((int(m.group(2)), 0), 0))
            r = int(m.group(1))
            typ = next((uzycie[x] for x in range(r, r + 4) if x in uzycie), None)
            if base[0] in wsk and typ:
                rola.setdefault(base[0], set()).add(typ)
    pa = [w for w, t in rola.items() if t == {"A"}]
    pb = [w for w, t in rola.items() if t == {"B"}]
    assert len(pa) == 1 and len(pb) == 1, (wsk, rola)
    c = next(re.search(r"cmp \(16\|M0\)\s+\(lt\)f0\.0\s+null<1>:d\s+(r\d+\.\d)<0;1,0>:d\s+(r\d+\.\d)<0;1,0>:d", l)
             for l in petla if "cmp (16|M0)" in l and "(lt)f0.0" in l)
    return f"r{pa[0]}.0", f"r{pb[0]}.0", c.group(1), c.group(2), wsk


def akumulatory(prolog):
    acc = [int(m.group(1)) for l in prolog for m in [re.search(r"mov \(16\|M0\)\s+r(\d+)\.0<1>:f\s+0\.0:f", l)] if m]
    assert len(acc) == 16, acc
    return acc


def wolne(epilog, zajete):
    uzyte = {int(x) for l in epilog for x in re.findall(r"\br(\d+)", l.split("//")[0])}
    return [r for r in range(3, 127) if r not in uzyte | zajete]


def przydziel(wol, rozmiar, parzystosc):
    w = set(wol)
    for b in range(4 + parzystosc, 127, 2):
        if all(r in w for r in range(b, b + rozmiar)):
            for r in range(b, b + rozmiar):
                wol.remove(r)
            return b
    raise SystemExit(f"brak bloku {rozmiar} rejestrów o parzystości {parzystosc}")


def zbuduj(prolog, cialo, epilog, wynik):
    tmp = tempfile.mkdtemp()
    d = os.path.join(tmp, "d")
    shutil.copytree(SZABLON, d)
    open(os.path.join(d, f".text.{KERNEL}.asm"), "w").write("\n".join(prolog + cialo + epilog) + "\n")
    nazwa = os.path.basename(wynik)
    subprocess.run(["ocloc", "asm", "-file", nazwa, "-dump", d, "-device", "whl"], check=True, stdout=subprocess.DEVNULL)
    shutil.move(os.path.join(d, nazwa), wynik)
    shutil.rmtree(tmp)


def petla(prolog, petla_k, epilog, acc):
    pa, pb, licz, gran, wsk = analiza(petla_k)
    mapa = {}
    for l in open(MAPA):
        i, r = map(int, l.split())
        mapa[r] = acc[i]
    zaj = set(acc) | {r for a in acc for r in (a, a + 1)} | {0, 1, 2, 127}
    zaj |= {int(x) for x in re.findall(r"r(\d+)", pa + pb + licz + gran)}
    zaj |= {r for w in wsk for r in (w, w + 1)}
    wol = wolne(epilog, zaj)
    bufory = []
    for _ in range(2):                          # ćwiartki k na zmianę w buforach 0 / 1
        A = przydziel(wol, 8, 0)
        BE = przydziel(wol, 4, 0)
        BO = przydziel(wol, 4, 1)
        bufory.append((A, BE, BO))
    H = wol[:4]                                 # nagłówki: [bufor][A, B]
    assert len(H) == 4, "brak rejestrów na nagłówki"
    out = [f"// --- ręczna pętla SGEMM: pa={pa} pb={pb} licznik={licz} granica={gran} bufory={bufory} naglowki={H}"]
    for b in range(2):                          # bufor b zaczyna od ćwiartki b (offset 256*b)
        out.append(f"{W}add (1|M0)  r{H[2 * b]}.0<1>:q  {pa}<0;1,0>:q  {256 * b}:w")
        out.append(f"{W}add (1|M0)  r{H[2 * b + 1]}.0<1>:q  {pb}<0;1,0>:q  {256 * b}:w")

    def laduj(b):
        """Ćwiartka do bufora b: A 2 × 128 B, BE 128 B, BO 128 B; nagłówki przesunięte o 2 ćwiartki (512 B)."""
        A, BE, BO = bufory[b]
        hA, hB = H[2 * b], H[2 * b + 1]
        o = []
        for dst, h in [(A, hA), (A + 4, hA), (BE, hB), (BO, hB)]:
            o.append(f"{W}send (16|M0)  r{dst}:w  r{h}  0xC  0x024D0CFF")
            o.append(f"{W}add (1|M0)  r{h}.0<1>:q  r{h}.0<0;1,0>:q  128:w")
        o.append(f"{W}add (1|M0)  r{hA}.0<1>:q  r{hA}.0<0;1,0>:q  256:w")
        o.append(f"{W}add (1|M0)  r{hB}.0<1>:q  r{hB}.0<0;1,0>:q  256:w")
        return o

    def mady(b):
        A, BE, BO = bufory[b]
        o = []
        for kl in range(4):
            breg = (BE if kl % 2 == 0 else BO) + 2 * (kl // 2)
            for r in range(16):
                g = A + 2 * (r // 4) + (1 if kl % 2 == 0 else 0)
                sub = (r % 4) * 2 + kl // 2
                a = mapa[r]
                o.append(f"        mad (16|M0)  r{a}.0<1>:f  r{a}.0<2;1>:f  r{breg}.0<2;1>:f  r{g}.{sub}<0>:f")
        return o

    out += laduj(0) + laduj(1) + ["LSG_PETLA:"]
    for q in range(4):
        out += mady(q % 2) + laduj(q % 2)
    out += [f"{W}add (1|M0)  {licz}<1>:d  {licz}<0;1,0>:d  16:w",
            f"{W}cmp (16|M0)  (lt)f0.0  null<1>:d  {licz}<0;1,0>:d  {gran}<0;1,0>:d",
            "(W&f0.0) jmpi  LSG_PETLA"]
    return out


if __name__ == "__main__":
    tryb, wynik = sys.argv[1], sys.argv[2]
    prolog, petla_k, epilog = wczytaj()
    acc = akumulatory(prolog)
    if tryb == "analiza":
        print(analiza(petla_k), acc)
    elif tryb == "mapa":
        zbuduj(prolog, ["LMAPA:"] + [f"{W}mov (16|M0)  r{a}.0<1>:f  {float(i)}:f" for i, a in enumerate(acc)], epilog, wynik)
    else:
        cialo = petla(prolog, petla_k, epilog, acc)
        zbuduj(prolog, cialo, epilog, wynik)
        print(cialo[0])
        print(f"instrukcji w ciele: {len(cialo)}, mad: {sum('mad (' in l for l in cialo)}")
