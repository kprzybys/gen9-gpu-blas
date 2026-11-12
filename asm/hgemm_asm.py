"""HGEMM 16x32 w ręcznym asemblerze Gen9: generator pętli wstawianej do szablonu kompilatora.

Szablon: kernel hgemm_16x32p z bench/hgemm.cl (rozłożony przez asm/disassemble.sh do asm/hgemm.d).
Z szablonu bierzemy prolog (adresy pa = r78.0:q, pb0 = r78.1:q, pb1 = r6.1:q, licznik k = r6.1:d, K = r6.0:d,
zerowanie akumulatorów) i epilog (zapis C). Całą pętlę k (od etykiety pętli do `while`) zastępujemy własnym kodem.

Tryby:
  python3 -I hgemm_asm.py mapa  <wynik.bin>  # bez pętli: akumulator nr i := i (do odczytu przypisania rejestr -> (wiersz, kolumna))
  python3 -I hgemm_asm.py petla <wynik.bin>  # właściwy kernel (wymaga asm/hgemm_map.txt z trybu mapa)

Pętla (krok k = 16, dwie połówki kl = 0..7 i 8..15, każda w osobnym buforze rejestrów X/Y):
  [256 mad: połówka 0 z bufora X] [6 send: następna połówka 0 -> X] [256 mad: połówka 1 z Y] [6 send -> Y] [licznik]
Brak konfliktów banków: rejestry B (wiersz kl) mają parzystość kl (baza parzysta), rejestry A dla danego kl
mają parzystość przeciwną dzięki układowi A2 (pack_a2 w bench/hgemm.cl).
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

TU = os.path.dirname(os.path.realpath(__file__))
SZABLON = os.path.join(TU, "hgemm.d")
KERNEL = "hgemm_16x32p"
MAPA = os.path.join(TU, "hgemm_map.txt")


def wczytaj():
    t = open(os.path.join(SZABLON, f".text.{KERNEL}.asm")).read().splitlines()
    w = next(i for i, l in enumerate(t) if re.search(r"^\s+while \(16\|M0\)\s+L\d+", l))
    etyk = re.search(r"while \(16\|M0\)\s+(L\d+)", t[w]).group(1)
    s = next(i for i, l in enumerate(t) if l.startswith(etyk + ":"))
    return t[:s], t[s:w + 1], t[w + 1:]


def akumulatory(prolog):
    """Rejestry zerowane w prologu jako hf: to akumulatory kafelka (32 sztuki)."""
    acc = [int(m.group(1)) for l in prolog for m in [re.search(r"mov \(16\|M0\)\s+r(\d+)\.0<1>:hf\s+0\.0:hf", l)] if m]
    assert len(acc) == 32, acc
    return acc


def rejestry(linie):
    return {int(x) for l in linie for x in re.findall(r"\br(\d+)", l.split("//")[0])}


def wolne(prolog, epilog, acc):
    """Wolne w pętli: wszystko poza akumulatorami, rejestrami epilogu (zachowawczo: każdy wymieniony),
    r0-r2 (nagłówek, local id), r6 (licznik, K, pb1), r78 (pa, pb0), r127."""
    zajete = set(acc) | rejestry(epilog) | {0, 1, 2, 6, 78, 127}
    return [r for r in range(3, 127) if r not in zajete]


def bloki_parzyste(wolne_r, ile, rozmiar):
    """Przydziel `ile` bloków po `rozmiar` kolejnych rejestrów, z parzystą bazą."""
    w, out = set(wolne_r), []
    for b in range(4, 127, 2):
        if len(out) == ile:
            break
        if all(r in w for r in range(b, b + rozmiar)):
            out.append(b)
            w -= set(range(b, b + rozmiar))
    assert len(out) == ile, f"za malo wolnych blokow: {out}"
    return out, sorted(w)


def zbuduj(prolog, cialo, epilog, wynik):
    tmp = tempfile.mkdtemp()
    d = os.path.join(tmp, "d")
    shutil.copytree(SZABLON, d)
    open(os.path.join(d, f".text.{KERNEL}.asm"), "w").write("\n".join(prolog + cialo + epilog) + "\n")
    nazwa = os.path.basename(wynik)
    subprocess.run(["ocloc", "asm", "-file", nazwa, "-dump", d, "-device", "whl"], check=True, stdout=subprocess.DEVNULL)
    shutil.move(os.path.join(d, nazwa), wynik)
    shutil.rmtree(tmp)


W = "(W)     "


def tryb_mapa(prolog, acc):
    """Akumulator i (kolejność z prologu) := i. Po epilogu C[r][c] = numer akumulatora."""
    return ["LMAPA:"] + [f"{W}mov (16|M0)  r{a}.0<1>:hf  {float(i)}:hf" for i, a in enumerate(acc)]


def tryb_petla(prolog, epilog, acc):
    # przypisanie (wiersz, grupa kolumn) -> rejestr akumulatora, z pliku mapy
    mapa = {}
    for l in open(MAPA):
        i, r, c = map(int, l.split())
        mapa[(r, c)] = acc[i]
    assert len(mapa) == 32
    wol = wolne(prolog, epilog, acc)
    # 6 bloków po 8 rejestrów: X: A, B0, B1; Y: A, B0, B1
    bazy, wol = bloki_parzyste(wol, 6, 8)
    XA, XB0, XB1, YA, YB0, YB1 = bazy
    assert len(wol) >= 12, f"za malo rejestrow na naglowki: {wol}"
    H = wol[:12]   # nagłówki: [bufor][strumień A,B0,B1][część 0,1]
    hdr = {(b, s, p): H[b * 6 + s * 2 + p] for b in range(2) for s in range(3) for p in range(2)}
    src = {0: "r78.0", 1: "r78.1", 2: "r6.1"}          # pa, pb0, pb1 (64-bit, z prologu)
    dane = {(0, 0): XA, (0, 1): XB0, (0, 2): XB1, (1, 0): YA, (1, 1): YB0, (1, 2): YB1}

    out = [f"// --- ręczna pętla HGEMM: X={XA},{XB0},{XB1} Y={YA},{YB0},{YB1} naglowki={H}"]
    # adresy początkowe: bufor b (połówka b) = baza + 256*b, część p = +128*p
    for (b, s, p), h in hdr.items():
        out.append(f"{W}add (1|M0)  r{h}.0<1>:q  {src[s]}<0;1,0>:q  {256 * b + 128 * p}:w")

    def laduj(b):
        o = []
        for s in range(3):
            for p in range(2):
                o.append(f"{W}send (16|M0)  r{dane[(b, s)] + 4 * p}:w  r{hdr[(b, s, p)]}  0xC  0x024D0CFF")
        for s in range(3):
            for p in range(2):
                o.append(f"{W}add (1|M0)  r{hdr[(b, s, p)]}.0<1>:q  r{hdr[(b, s, p)]}.0<0;1,0>:q  512:w")
        return o

    def mady(b):
        A, B0, B1 = dane[(b, 0)], dane[(b, 1)], dane[(b, 2)]
        o = []
        for kl in range(8):
            for r in range(16):
                ga = A + 2 * (r // 4) + (1 if kl % 2 == 0 else 0)
                sub = (r % 4) * 4 + kl // 2
                for c, Bc in ((0, B0), (1, B1)):
                    a = mapa[(r, c)]
                    o.append(f"        mad (16|M0)  r{a}.0<1>:hf  r{a}.0<2;1>:hf  r{Bc + kl}.0<2;1>:hf  r{ga}.{sub}<0>:hf")
        return o

    out += laduj(0) + laduj(1)
    out += ["LHG_PETLA:"]
    out += mady(0) + laduj(0) + mady(1) + laduj(1)
    out += [f"{W}add (1|M0)  r6.1<1>:d  r6.1<0;1,0>:d  16:w",
            f"{W}cmp (16|M0)  (lt)f0.0  null<1>:d  r6.1<0;1,0>:d  r6.0<0;1,0>:d",
            "(W&f0.0) jmpi  LHG_PETLA"]
    return out


if __name__ == "__main__":
    tryb, wynik = sys.argv[1], sys.argv[2]
    prolog, petla, epilog = wczytaj()
    acc = akumulatory(prolog)
    if tryb == "mapa":
        zbuduj(prolog, tryb_mapa(prolog, acc), epilog, wynik)
    else:
        cialo = tryb_petla(prolog, epilog, acc)
        zbuduj(prolog, cialo, epilog, wynik)
        print(f"instrukcji w ciele: {len(cialo)}, mad: {sum('mad (' in l for l in cialo)}")
