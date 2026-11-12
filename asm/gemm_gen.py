"""Wspólny generator ręcznych pętli GEMM w asemblerze Gen9 dla UHD 620 (zastępuje hgemm_asm.py, sgemm_asm.py, hgemm_kc_asm.py).

Rodzaje (szablony w bench/*.cl; wszystkie wsadowe: numer macierzy z numeru wątku, dla 1 macierzy b = 0):
  h  : hgemm_16x32q    kafelek 16×32, FP16, A2 + Bq,   pół-kroki k (2 × 8 kl), 512 mad / k16
  hd : hgemm_16x16kc_b kafelek 16×16, FP16 -> porcje 32 -> FP32, A3 + B16, ćwiartki k, 256 mad / k16
  s  : sgemm_16x16p_b  kafelek 16×16, FP32, A2f + B2f, ćwiartki k, 256 mad / k16
Automatycznie z pętli kompilatora: wskaźniki A i B (rejestry :q zwiększane w pętli; rola z użycia danych w mad,
także przez kopie `mov :w`), licznik i granica (`cmp (lt)`), akumulatory (zerowane przed pętlą).
Mapa akumulatorów (rejestr -> wiersz/kolumna) z uruchomienia na GPU (tryb mapa*, zob. gemm_build.sh).

Użycie: python3 -I gemm_gen.py <rodzaj> analiza|mapa|mapa_f|mapa_h|petla <wynik.bin>
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

TU = os.path.dirname(os.path.realpath(__file__))
W = "(W)     "
RODZAJE = {
    # szablony „v”: argumenty (Ap, Bp, C, M, N, ldc, K, Kfull) -- fragmenty danych spakowanych raz
    "h": dict(plik="hgemm", kernel="hgemm_v", typ="hf"),
    "hd": dict(plik="hgemm", kernel="hgemm_kc_v", typ="hf"),
    "s": dict(plik="sgemm", kernel="sgemm_v", typ="f"),
}
SEND = "0xC  0x024D0CFF"   # a64 aligned OWord block read ×8 (128 B), nagłówek 1 GRF


def katalog(rodz):
    return os.path.join(TU, f"gen_{rodz}.d")


def wczytaj(rodz):
    k = RODZAJE[rodz]["kernel"]
    t = open(os.path.join(katalog(rodz), f".text.{k}.asm")).read().splitlines()
    w = next(i for i, l in enumerate(t) if re.search(r"^\s+while \(16\|M0\)\s+L\d+", l))
    etyk = re.search(r"while \(16\|M0\)\s+(L\d+)", t[w]).group(1)
    s = next(i for i, l in enumerate(t) if l.startswith(etyk + ":"))
    return t[:s], t[s:w + 1], t[w + 1:]


def akumulatory(prolog):
    """Zerowania po ostatnim skoku prologu: (lista hf, lista f)."""
    ost = max([i for i, l in enumerate(prolog) if "jmpi" in l] + [0])
    h = [int(m.group(1)) for l in prolog[ost:] for m in [re.search(r"mov \(16\|M0\)\s+r(\d+)\.0<1>:hf\s+0\.0:hf", l)] if m]
    f = [int(m.group(1)) for l in prolog[ost:] for m in [re.search(r"mov \(16\|M0\)\s+r(\d+)\.0<1>:f\s+0\.0:f", l)] if m]
    return h, f


def analiza(petla, typ):
    wsk = sorted({int(m.group(1)) for l in petla
                  for m in [re.search(r"add \((?:8|1)\|M0\)\s+r(\d+)\.0<1>:q\s+r\1\.0<[^>]*>:q\s+\d+:w", l)] if m})
    uzycie = {}
    for l in petla:
        m = re.search(rf"mad \(16\|M0\)\s+r\d+\.0<1>:{typ}\s+r\d+\.0<2;1>:{typ}\s+r(\d+)\.0<2;1>:{typ}\s+r(\d+)\.\d+<0>:{typ}", l)
        if m:
            uzycie.setdefault(int(m.group(1)), "B")
            uzycie.setdefault(int(m.group(2)), "A")
    kopie = [(int(m.group(1)), int(m.group(2))) for l in petla
             for m in [re.search(r"mov \(16\|M0\)\s+r(\d+)\.0<1>:w\s+r(\d+)\.0<16;16,1>:w", l)] if m]
    for _ in range(6):
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
            t = {uzycie[x] for x in range(r, r + 4) if x in uzycie}
            if base in wsk and len(t) == 1:
                rola.setdefault(base, set()).update(t)
    pa = [w for w, t in rola.items() if t == {"A"}]
    pb = [w for w, t in rola.items() if t == {"B"}]
    assert len(pa) == 1 and len(pb) == 1, ("rozpoznanie wskaźników", wsk, rola)
    c = next(re.search(r"cmp \(16\|M0\)\s+\(lt\)f0\.0\s+null<1>:d\s+(r\d+\.\d)<0;1,0>:d\s+(r\d+\.\d)<0;1,0>:d", l)
             for l in petla if "(lt)f0.0" in l)
    return f"r{pa[0]}.0", f"r{pb[0]}.0", c.group(1), c.group(2), wsk


def zbuduj(rodz, prolog, cialo, epilog, wynik):
    k = RODZAJE[rodz]["kernel"]
    tmp = tempfile.mkdtemp()
    d = os.path.join(tmp, "d")
    shutil.copytree(katalog(rodz), d)
    open(os.path.join(d, f".text.{k}.asm"), "w").write("\n".join(prolog + cialo + epilog) + "\n")
    nazwa = os.path.basename(wynik)
    subprocess.run(["ocloc", "asm", "-file", nazwa, "-dump", d, "-device", "whl"], check=True, stdout=subprocess.DEVNULL)
    shutil.move(os.path.join(d, nazwa), wynik)
    shutil.rmtree(tmp)


class Rejestry:
    def __init__(self, wolne):
        self.w = list(wolne)

    def blok(self, rozmiar, parz=0):
        s = set(self.w)
        for b in range(2 + parz, 127, 2):
            if all(r in s for r in range(b, b + rozmiar)):
                for r in range(b, b + rozmiar):
                    self.w.remove(r)
                return b
        raise SystemExit(f"brak bloku {rozmiar} rejestrów o parzystości {parz}")

    def jeden(self):
        return self.w.pop(0)


def wolne(prolog, epilog, zajete):
    uz = {int(x) for l in epilog for x in re.findall(r"\br(\d+)", l.split("//")[0])}
    return [r for r in range(2, 127) if r not in uz | zajete | {0, 1, 127}]


def ladowanie(H, cele):
    """cele: lista (rejestr docelowy, indeks nagłówka); po każdym send nagłówek += 128 B (sprzęt czyta nagłówek przy wydaniu)."""
    return [x for dst, h in cele for x in (f"{W}send (16|M0)  r{dst}:w  r{H[h]}  {SEND}",
                                           f"{W}add (1|M0)  r{H[h]}.0<1>:q  r{H[h]}.0<0;1,0>:q  128:w")]


def przesun(H, idx, ile):
    return [f"{W}add (1|M0)  r{H[i]}.0<1>:q  r{H[i]}.0<0;1,0>:q  {ile}:w" for i in idx]


def koniec(licz, gran, krok, etyk):
    return [f"{W}add (1|M0)  {licz}<1>:d  {licz}<0;1,0>:d  {krok}:w",
            f"{W}cmp (16|M0)  (lt)f0.0  null<1>:d  {licz}<0;1,0>:d  {gran}<0;1,0>:d",
            f"(W&f0.0) jmpi  {etyk}"]


def petla_h(prolog, petla_k, epilog, mapa):
    """FP16 16×32: pół-krok h (kl 0..7) w buforze h: A2 8 GRF (pa + 256h), B0 8 GRF (pb + 256h), B1 8 GRF (pb + 512 + 256h).
    Krok k16: pa += 512, pb += 1024."""
    pa, pb, licz, gran, wsk = analiza(petla_k, "hf")
    acc = set(mapa.values())
    zaj = acc | {r for w in wsk for r in (w, w + 1)} | {int(x) for x in re.findall(r"r(\d+)", licz + gran)}
    R = Rejestry(wolne(prolog, epilog, zaj))
    buf = [(R.blok(8), R.blok(8), R.blok(8)) for _ in range(2)]
    H = [R.jeden() for _ in range(6)]   # [bufor][A, B0, B1]
    out = [f"// --- gemm_gen h: pa={pa} pb={pb} licznik={licz} granica={gran} bufory={buf} naglowki={H}"]
    for b in range(2):
        out += [f"{W}add (1|M0)  r{H[3 * b]}.0<1>:q  {pa}<0;1,0>:q  {256 * b}:w",
                f"{W}add (1|M0)  r{H[3 * b + 1]}.0<1>:q  {pb}<0;1,0>:q  {256 * b}:w",
                f"{W}add (1|M0)  r{H[3 * b + 2]}.0<1>:q  {pb}<0;1,0>:q  {512 + 256 * b}:w"]

    def laduj(b):
        A, B0, B1 = buf[b]
        o = ladowanie(H, [(A, 3 * b), (A + 4, 3 * b), (B0, 3 * b + 1), (B0 + 4, 3 * b + 1), (B1, 3 * b + 2), (B1 + 4, 3 * b + 2)])
        # po 2 × 128 B: A do następnego k16 +256, B +768 (pomijamy drugą połówkę i drugą grupę kolumn)
        return o + przesun(H, [3 * b], 256) + przesun(H, [3 * b + 1, 3 * b + 2], 768)

    def mady(b):
        A, B0, B1 = buf[b]
        o = []
        for kl in range(8):
            for r in range(16):
                ga, sub = A + 2 * (r // 4) + (1 if kl % 2 == 0 else 0), (r % 4) * 4 + kl // 2
                for c, Bc in ((0, B0), (1, B1)):
                    a = mapa[(r, c)]
                    o.append(f"        mad (16|M0)  r{a}.0<1>:hf  r{a}.0<2;1>:hf  r{Bc + kl}.0<2;1>:hf  r{ga}.{sub}<0>:hf")
        return o
    return out + laduj(0) + laduj(1) + ["LGEN:"] + mady(0) + laduj(0) + mady(1) + laduj(1) + koniec(licz, gran, 16, "LGEN")


def petla_cwiartki(prolog, petla_k, epilog, mapa, typ, kc=None, mapa_f=None):
    """16×16, ćwiartki k (kl 0..3), bufory na zmianę, odczyt z wyprzedzeniem o 2 ćwiartki.
    hf (hd): A3 4 GRF + B 4 GRF; ćwiartka = 128 B A + 128 B B. f (s): A2f 8 GRF + BE 4 + BO 4; ćwiartka = 256 B + 256 B.
    hd: po kc iloczynach przeniesienie f += h (mieszane add), h = 0."""
    pa, pb, licz, gran, wsk = analiza(petla_k, typ)
    acc = set(mapa.values()) | ({r for x in mapa.values() for r in (x, x + 1)} if typ == "f" else set())
    if mapa_f:
        acc |= {r for x in mapa_f.values() for r in (x, x + 1)}
    zaj = acc | {r for w in wsk for r in (w, w + 1)} | {int(x) for x in re.findall(r"r(\d+)", licz + gran)}
    R = Rejestry(wolne(prolog, epilog, zaj))
    if typ == "hf":
        buf = [(R.blok(4), R.blok(4), None) for _ in range(2)]
        cw = 128
    else:
        buf = [(R.blok(8), R.blok(4, 0), R.blok(4, 1)) for _ in range(2)]
        cw = 256
    H = [R.jeden() for _ in range(4)]   # [bufor][A, B]
    out = [f"// --- gemm_gen {typ}: pa={pa} pb={pb} licznik={licz} granica={gran} bufory={buf} naglowki={H}"]
    for b in range(2):
        out += [f"{W}add (1|M0)  r{H[2 * b]}.0<1>:q  {pa}<0;1,0>:q  {cw * b}:w",
                f"{W}add (1|M0)  r{H[2 * b + 1]}.0<1>:q  {pb}<0;1,0>:q  {cw * b}:w"]

    def laduj(b):
        A, B, BO = buf[b]
        if typ == "hf":
            o = ladowanie(H, [(A, 2 * b), (B, 2 * b + 1)])
        else:
            o = ladowanie(H, [(A, 2 * b), (A + 4, 2 * b), (B, 2 * b + 1), (BO, 2 * b + 1)])
        return o + przesun(H, [2 * b, 2 * b + 1], 2 * cw - (128 if typ == "hf" else 256))

    def mady(b):
        A, B, BO = buf[b]
        o = []
        for kl in range(4):
            for r in range(16):
                a = mapa[r]
                if typ == "hf":
                    g, sub, breg = A + 2 * (r // 8) + (kl % 2 == 0), (r % 8) * 2 + kl // 2, B + kl
                else:
                    g, sub = A + 2 * (r // 4) + (kl % 2 == 0), (r % 4) * 2 + kl // 2
                    breg = (B if kl % 2 == 0 else BO) + 2 * (kl // 2)
                o.append(f"        mad (16|M0)  r{a}.0<1>:{typ}  r{a}.0<2;1>:{typ}  r{breg}.0<2;1>:{typ}  r{g}.{sub}<0>:{typ}")
        return o

    krok = kc or 16
    out += laduj(0) + laduj(1) + ["LGEN:"]
    for q in range(krok // 4):
        out += mady(q % 2) + laduj(q % 2)
    if kc:
        for r in range(16):
            h, f = mapa[r], mapa_f[r]
            out += [f"        add (8|M0)  r{f}.0<1>:f  r{f}.0<8;8,1>:f  r{h}.0<8;8,1>:hf",
                    f"        add (8|M8)  r{f + 1}.0<1>:f  r{f + 1}.0<8;8,1>:f  r{h}.8<8;8,1>:hf",
                    f"        mov (16|M0)  r{h}.0<1>:hf  0.0:hf"]
    return out + koniec(licz, gran, krok, "LGEN")


def wczytaj_mape(rodz):
    mapa, mapa_f = {}, {}
    for l in open(os.path.join(TU, f"gen_{rodz}_mapa.txt")):
        p = l.split()
        if rodz == "h":
            mapa[(int(p[1]), int(p[2]))] = int(p[0])
        elif p[0] == "f":
            mapa_f[int(p[2])] = int(p[1])
        else:
            mapa[int(p[2])] = int(p[1])
    return mapa, mapa_f


if __name__ == "__main__":
    rodz, tryb = sys.argv[1], sys.argv[2]
    wynik = sys.argv[3] if len(sys.argv) > 3 else None
    prolog, petla_k, epilog = wczytaj(rodz)
    h, f = akumulatory(prolog)
    typ = RODZAJE[rodz]["typ"]
    if tryb == "analiza":
        print(analiza(petla_k, typ), "hf:", h, "f:", f)
    elif tryb.startswith("mapa"):
        # mapa (h, s) / mapa_f / mapa_h (hd): akumulator nr i := i; zapis C pokaże, który rejestr to który wiersz
        cel, t = (f, "f") if tryb == "mapa_f" or (tryb == "mapa" and rodz == "s") else (h, "hf")
        zbuduj(rodz, prolog, ["LMAPA:"] + [f"{W}mov (16|M0)  r{a}.0<1>:{t}  {float(i)}:{t}" for i, a in enumerate(cel)],
               epilog, wynik)
    else:
        mapa_i, mapa_fi = wczytaj_mape(rodz)
        if rodz == "h":
            mapa = {k: h[v] for k, v in mapa_i.items()}
            cialo = petla_h(prolog, petla_k, epilog, mapa)
        elif rodz == "hd":
            cialo = petla_cwiartki(prolog, petla_k, epilog, {k: h[v] for k, v in mapa_i.items()}, "hf", kc=32,
                                   mapa_f={k: f[v] for k, v in mapa_fi.items()})
        else:
            cialo = petla_cwiartki(prolog, petla_k, epilog, {k: f[v] for k, v in mapa_i.items()}, "f")
        zbuduj(rodz, prolog, cialo, epilog, wynik)
        print(cialo[0])
        print(f"instrukcji: {len(cialo)}, mad: {sum('mad (' in l for l in cialo)}")
