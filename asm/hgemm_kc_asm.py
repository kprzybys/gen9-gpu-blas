"""HGEMM dokładny (kafelek 16x16, suma FP16 porcjami po 32 iloczyny -> FP32) w ręcznym asemblerze Gen9.

Szablon: hgemm_16x16kc z bench/hgemm.cl (asm/hgemm_kc.d). Z szablonu: prolog (wskaźniki, zerowanie akumulatorów)
i epilog (zapis (half)(f + h)). Pętlę k zastępujemy własną. Rozpoznawane automatycznie z pętli kompilatora:
wskaźniki A/B (rejestry :q zwiększane o stały krok; rola z użycia danych w mad, także przez kopie `mov :w`),
licznik i granica (`cmp (lt)`), akumulatory (zerowane w prologu: 16 × hf, 16 × f).
Tryby: mapa_f | mapa_h | petla  <wynik.bin>   (petla wymaga asm/hgemm_kc_map.txt: "f|h i r")

Pętla: [8 ćwiartek k (kl 0..3): 64 mad z bufora q%2, potem send następnej ćwiartki do tego bufora]
       [przeniesienie: f(r) += h(r) (mieszane add (8) :f <- :hf, 2 na wiersz), h(r) = 0] [licznik += 32]
Banki: B wiersz kl w rejestrze o parzystości kl (baza parzysta), A (układ A3, pack_a3_pad) dla kl parzystego
w rejestrze nieparzystym i odwrotnie -> 0 konfliktów.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

TU = os.path.dirname(os.path.realpath(__file__))
SZABLON = os.path.join(TU, "hgemm_kc.d")
KERNEL = "hgemm_16x16kc"
MAPA = os.path.join(TU, "hgemm_kc_map.txt")
KC = 32
W = "(W)     "


def wczytaj():
    t = open(os.path.join(SZABLON, f".text.{KERNEL}.asm")).read().splitlines()
    w = next(i for i, l in enumerate(t) if re.search(r"^\s+while \(16\|M0\)\s+L\d+", l))
    etyk = re.search(r"while \(16\|M0\)\s+(L\d+)", t[w]).group(1)
    s = next(i for i, l in enumerate(t) if l.startswith(etyk + ":"))
    return t[:s], t[s:w + 1], t[w + 1:]


def akumulatory(prolog):
    """Zerowania tuż przed pętlą (po ostatnim skoku prologu): 16 × hf (h) i 16 × f (f, po 2 GRF)."""
    ost = max(i for i, l in enumerate(prolog) if "jmpi" in l)
    h = [int(m.group(1)) for l in prolog[ost:] for m in [re.search(r"mov \(16\|M0\)\s+r(\d+)\.0<1>:hf\s+0\.0:hf", l)] if m]
    f = [int(m.group(1)) for l in prolog[ost:] for m in [re.search(r"mov \(16\|M0\)\s+r(\d+)\.0<1>:f\s+0\.0:f", l)] if m]
    assert len(h) == 16 and len(f) == 16, (h, f)
    return h, f


def analiza(petla):
    wsk = sorted({int(m.group(1)) for l in petla
                  for m in [re.search(r"add \((?:8|1)\|M0\)\s+r(\d+)\.0<1>:q\s+r\1\.0<[^>]*>:q\s+\d+:w", l)] if m})
    uzycie = {}
    for l in petla:
        m = re.search(r"mad \(16\|M0\)\s+r\d+\.0<1>:hf\s+r\d+\.0<2;1>:hf\s+r(\d+)\.0<2;1>:hf\s+r(\d+)\.\d+<0>:hf", l)
        if m:
            uzycie.setdefault(int(m.group(1)), "B")
            uzycie.setdefault(int(m.group(2)), "A")
    # kopie danych mov (16) :w rD <- rS: rola rD przechodzi na rS (do skutku)
    kopie = [(int(m.group(1)), int(m.group(2))) for l in petla
             for m in [re.search(r"mov \(16\|M0\)\s+r(\d+)\.0<1>:w\s+r(\d+)\.0<16;16,1>:w", l)] if m]
    for _ in range(4):
        for d, s in kopie:
            if d in uzycie:
                uzycie.setdefault(s, uzycie[d])
    defs, rola = {}, {}
    for l in petla:
        m = re.search(r"(mov|add) \((?:1|8)\|M0\)\s+r(\d+)\.0<1>:u?q\s+r(\d+)\.0<[^>]*>:u?q(?:\s+(\S+):)?", l)
        if m:
            src = int(m.group(3))
            off = int(m.group(4), 0) if m.group(4) and m.group(1) == "add" else 0
            base, boff = defs.get(src, (src, 0))
            defs[int(m.group(2))] = (base, boff + off)
            continue
        m = re.search(r"send \(16\|M0\)\s+r(\d+):w\s+r(\d+)\s", l)
        if m:
            base, _ = defs.get(int(m.group(2)), (int(m.group(2)), 0))
            r = int(m.group(1))
            typ = {uzycie[x] for x in range(r, r + 4) if x in uzycie}
            if base in wsk and len(typ) == 1:
                rola.setdefault(base, set()).update(typ)
    pa = [w for w, t in rola.items() if t == {"A"}]
    pb = [w for w, t in rola.items() if t == {"B"}]
    assert len(pa) == 1 and len(pb) == 1, (wsk, rola)
    c = next(re.search(r"cmp \(16\|M0\)\s+\(lt\)f0\.0\s+null<1>:d\s+(r\d+\.\d)<0;1,0>:d\s+(r\d+\.\d)<0;1,0>:d", l)
             for l in petla if "(lt)f0.0" in l)
    return pa[0], pb[0], c.group(1), c.group(2), wsk


def zbuduj(prolog, cialo, epilog, wynik):
    tmp = tempfile.mkdtemp()
    d = os.path.join(tmp, "d")
    shutil.copytree(SZABLON, d)
    open(os.path.join(d, f".text.{KERNEL}.asm"), "w").write("\n".join(prolog + cialo + epilog) + "\n")
    nazwa = os.path.basename(wynik)
    subprocess.run(["ocloc", "asm", "-file", nazwa, "-dump", d, "-device", "whl"], check=True, stdout=subprocess.DEVNULL)
    shutil.move(os.path.join(d, nazwa), wynik)
    shutil.rmtree(tmp)


def przydziel(wol, rozmiar):
    w = set(wol)
    for b in range(4, 127, 2):
        if all(r in w for r in range(b, b + rozmiar)):
            for r in range(b, b + rozmiar):
                wol.remove(r)
            return b
    raise SystemExit(f"brak bloku {rozmiar} rejestrów z parzystą bazą")


def petla(prolog, petla_k, epilog, h, f):
    pa, pb, licz, gran, wsk = analiza(petla_k)
    mh, mf = {}, {}
    for l in open(MAPA):
        rodz, i, r = l.split()
        (mh if rodz == "h" else mf)[int(r)] = (h if rodz == "h" else f)[int(i)]
    assert len(mh) == 16 and len(mf) == 16
    zaj = set(h) | {r for x in f for r in (x, x + 1)} | {0, 1, 127} | {r for w in wsk for r in (w, w + 1)}
    zaj |= {int(x) for x in re.findall(r"r(\d+)", licz + gran)}
    uzyte_epilog = {int(x) for l in epilog for x in re.findall(r"\br(\d+)", l.split("//")[0])}
    wol = [r for r in range(2, 127) if r not in zaj | uzyte_epilog]
    bufory = [(przydziel(wol, 4), przydziel(wol, 4)) for _ in range(2)]   # (A, B), bazy parzyste
    H = wol[:4]
    assert len(H) == 4
    out = [f"// --- ręczna pętla HGEMM dokładny: pa=r{pa} pb=r{pb} licznik={licz} granica={gran} bufory={bufory} naglowki={H}"]
    for b in range(2):
        out.append(f"{W}add (1|M0)  r{H[2 * b]}.0<1>:q  r{pa}.0<0;1,0>:q  {128 * b}:w")
        out.append(f"{W}add (1|M0)  r{H[2 * b + 1]}.0<1>:q  r{pb}.0<0;1,0>:q  {128 * b}:w")

    def laduj(b):
        A, B = bufory[b]
        o = []
        for dst, hd in ((A, H[2 * b]), (B, H[2 * b + 1])):
            o.append(f"{W}send (16|M0)  r{dst}:w  r{hd}  0xC  0x024D0CFF")
            o.append(f"{W}add (1|M0)  r{hd}.0<1>:q  r{hd}.0<0;1,0>:q  256:w")
        return o

    def mady(b):
        A, B = bufory[b]
        o = []
        for kl in range(4):
            for r in range(16):
                g = A + 2 * (r // 8) + (1 if kl % 2 == 0 else 0)
                sub = (r % 8) * 2 + kl // 2
                a = mh[r]
                o.append(f"        mad (16|M0)  r{a}.0<1>:hf  r{a}.0<2;1>:hf  r{B + kl}.0<2;1>:hf  r{g}.{sub}<0>:hf")
        return o

    out += laduj(0) + laduj(1) + ["LKC_PETLA:"]
    for q in range(KC // 4):
        out += mady(q % 2) + laduj(q % 2)
    for r in range(16):   # przeniesienie porcji do FP32 (składnia jak w kodzie kompilatora) i wyzerowanie h
        out.append(f"        add (8|M0)  r{mf[r]}.0<1>:f  r{mf[r]}.0<8;8,1>:f  r{mh[r]}.0<8;8,1>:hf")
        out.append(f"        add (8|M8)  r{mf[r] + 1}.0<1>:f  r{mf[r] + 1}.0<8;8,1>:f  r{mh[r]}.8<8;8,1>:hf")
        out.append(f"        mov (16|M0)  r{mh[r]}.0<1>:hf  0.0:hf")
    out += [f"{W}add (1|M0)  {licz}<1>:d  {licz}<0;1,0>:d  {KC}:w",
            f"{W}cmp (16|M0)  (lt)f0.0  null<1>:d  {licz}<0;1,0>:d  {gran}<0;1,0>:d",
            "(W&f0.0) jmpi  LKC_PETLA"]
    return out


if __name__ == "__main__":
    tryb, wynik = sys.argv[1], sys.argv[2]
    prolog, petla_k, epilog = wczytaj()
    h, f = akumulatory(prolog)
    if tryb == "analiza":
        print(analiza(petla_k), h, f)
    elif tryb in ("mapa_f", "mapa_h"):
        cel = f if tryb == "mapa_f" else h
        typ = "f" if tryb == "mapa_f" else "hf"
        zbuduj(prolog, ["LMAPA:"] + [f"{W}mov (16|M0)  r{a}.0<1>:{typ}  {float(i)}:{typ}" for i, a in enumerate(cel)],
               epilog, wynik)
    else:
        cialo = petla(prolog, petla_k, epilog, h, f)
        zbuduj(prolog, cialo, epilog, wynik)
        print(cialo[0])
        print(f"instrukcji w ciele: {len(cialo)}, mad: {sum('mad (' in l for l in cialo)}")
