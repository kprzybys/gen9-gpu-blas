"""Mikrotesty w asemblerze Gen9: wstawia ręcznie wygenerowane ciało pętli do szablonu fma32.d (kernel k_fma).

Szablon (asm/fma32.cl → asm/fma32.d): pętla `L560: ... jmpi L560` wykonywana n razy (argument n).
W pętli wolno używać rejestrów r51–r126. r0–r50 i r127 należą do prologu i epilogu szablonu.

Użycie: python3 -I micro.py <test> <wynik.bin>   (lista testów: python3 -I micro.py --lista)
Wypisuje liczbę instrukcji w ciele pętli. Pomiar: bench/clbench <wynik.bin> k_fma <global> <local> <n> <ops>.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

TU = os.path.dirname(os.path.realpath(__file__))
SZABLON = os.path.join(TU, "fma32.d")


KOMPAKT = os.environ.get("KOMPAKT") == "1"   # KOMPAKT=1: dopisz {Compacted} (IGA odrzuci, jeśli się nie da)


def mad(dst, s0, s1, s2, t="f", n=16, m="M0"):
    """mad: dst = s1*s2 + s0 (kolejność operandów jak w IGA). sN to gotowe napisy operandów z regionem."""
    return f"        mad ({n}|{m})  r{dst}.0<1>:{t}  {reg(s0, 0)}:{t}  {reg(s1, 1)}:{t}  {reg(s2, 2)}:{t}" + ("  {Compacted}" if KOMPAKT else "")


# Składnia regionów IGA dla 3-źródłowych instrukcji Gen9 (sprawdzona iga64 -p=9 i w kodzie kompilatora):
#   src0, src1: wektor <2;1>, skalar rozgłaszany <0;0>;   src2: wektor <1>, skalar <0>.
# Generatory zwracają szablon z miejscem {R}, wypełniany zależnie od pozycji operandu.
def vec(r):          # operand wektorowy SIMD16 FP32: 2 rejestry od r
    return (f"r{r}.0", "vec")


def bcast(r, sub=0):  # skalar rozgłaszany z r.sub (jak element A w GEMM)
    return (f"r{r}.{sub}", "bcast")


def reg(op, poz):
    r, rodzaj = op
    if poz == 2:
        return r + ("<1>" if rodzaj == "vec" else "<0>")
    return r + ("<2;1>" if rodzaj == "vec" else "<0;0>")


def banki(src1, src2, ile=256, dst0=52, ndst=8, src0=110):
    """Niezależne mad: wyniki rotują po ndst parach rejestrów, źródła stałe."""
    out = []
    for i in range(ile):
        d = dst0 + 2 * (i % ndst)
        out.append(mad(d, vec(src0) if src0 else bcast(120, 1), src1, src2))
    return out


def gemm(nacc, a=73, bvar=False, ile=256):
    """Pętla jak w GEMM: acc_j += B * A[k]. acc in-place (dst == src0), B wektor (src1), A skalar rozgłaszany (src2)."""
    out = []
    for i in range(ile):
        acc = 84 + 2 * (i % nacc)
        b = (56 + 4 * ((i // nacc) % 4)) if bvar else 70
        out.append(mad(acc, vec(acc), vec(b), bcast(a, i % 8)))
    return out


def kompilator(kompakt):
    """Ciało pętli skopiowane z szablonu (kod IGC), z {Compacted} albo bez."""
    src = open(os.path.join(SZABLON, ".text.k_fma.asm")).read()
    m = re.search(r"L560:\n(.*?)\n\(W&f0\.0\) jmpi", src, re.S)
    linie = [l for l in m.group(1).splitlines() if l.strip() and "mad" in l]
    return [l if kompakt else l.replace("{Compacted}", "") for l in linie]


def bank(r):
    return ((r & 0x40) >> 5) | (r & 1)


TESTY = {
    # nazwa: (opis, generator ciała pętli)
    "bank_rozne_parzystosc": ("src1=r70 (bank 0/1), src2=r73 (bank 1/0): różne banki w obu połówkach",
                              lambda: banki(vec(70), vec(73))),
    "bank_ten_sam": ("src1=r70, src2=r72: oba parzyste w dolnej połowie -> ten sam bank w obu połówkach",
                     lambda: banki(vec(70), vec(72))),
    "bank_rozne_polowy": ("src1=r70 (lo), src2=r100 (hi): banki 0/1 vs 2/3, brak konfliktu",
                          lambda: banki(vec(70), vec(100))),
    "bank_hi_ten_sam": ("src1=r100, src2=r102: oba w górnej połowie, parzyste -> konflikt",
                        lambda: banki(vec(100), vec(102))),
    "bank_ten_sam_rejestr": ("src1=src2=r70: ten sam rejestr (czy EU pomija drugi odczyt?)",
                             lambda: banki(vec(70), vec(70))),
    "bcast_konflikt": ("src1=r70 wektor, src2=r72.0 skalar rozgłaszany (ten sam bank) - wzorzec GEMM",
                       lambda: banki(vec(70), bcast(72))),
    "bcast_bez_konfliktu": ("src1=r70 wektor, src2=r73.0 skalar rozgłaszany (inny bank)",
                            lambda: banki(vec(70), bcast(73))),
    "bcast_oba_skalar": ("src1=r70.0, src2=r72.0: oba rozgłaszane (jak w szablonie kompilatora)",
                         lambda: banki(bcast(70), bcast(72))),
    "src0_ten_sam_bank": ("src0=r72 (ten sam bank co src2? nie: co src1 r70), src1=r70, src2=r101 - czy src0 ma znaczenie",
                          lambda: banki(vec(70), vec(101), src0=72)),
    "parz_71_100": ("src1=r71 (nieparzysty, lo), src2=r100 (parzysty, hi): czy decyduje tylko parzystość",
                    lambda: banki(vec(71), vec(100))),
    "parz_70_101": ("src1=r70 (parzysty, lo), src2=r101 (nieparzysty, hi)",
                    lambda: banki(vec(70), vec(101))),
    "parz_71_73": ("src1=r71, src2=r73: oba nieparzyste, lo",
                   lambda: banki(vec(71), vec(73))),
    "parz_70_78": ("src1=r70, src2=r78: parzyste, odległość 8 rejestrów",
                   lambda: banki(vec(70), vec(78))),
    "parz_70_75": ("src1=r70, src2=r75: różna parzystość, odległość 5",
                   lambda: banki(vec(70), vec(75))),
    "dst8": ("jak bank_rozne_parzystosc, wyniki rotują po 8 parach (domyślnie)",
             lambda: banki(vec(70), vec(73), ndst=8)),
    "dst4": ("wyniki rotują po 4 parach (bliżej WAW)", lambda: banki(vec(70), vec(73), ndst=4)),
    "dst2": ("wyniki rotują po 2 parach", lambda: banki(vec(70), vec(73), ndst=2)),
    "dst16_hi": ("wyniki po 16 parach w r80-r111? nie: r84+, źródła r70/r73 + src0 r120",
                 lambda: banki(vec(70), vec(73), dst0=84, ndst=12, src0=120)),
    "m0_m16": ("pary M0/M16 jak w kodzie kompilatora (SIMD32), różne banki",
               lambda: [mad(52 + 2 * (i // 2 % 8), vec(110), vec(70 if i % 2 == 0 else 74), vec(73 if i % 2 == 0 else 77),
                            m="M0" if i % 2 == 0 else "M16") for i in range(256)]),

    "s0skal_v_b": ("src0 skalar, src1 wektor r70, src2 skalar r73.0: jak kod kompilatora (1 wektor na instr.)",
                   lambda: banki(vec(70), bcast(73), src0=None)),
    "s0skal_v_v": ("src0 skalar, src1 wektor r70, src2 wektor r73: 2 wektory na instr.",
                   lambda: banki(vec(70), vec(73), src0=None)),
    "gemm_acc16": ("GEMM: acc_i = acc_i + B(r70 wektor)*A(r73.k skalar), 16 akumulatorów (r84-r115), in-place",
                   lambda: gemm(16)),
    "gemm_acc8": ("jak wyżej, 8 akumulatorów", lambda: gemm(8)),
    "gemm_acc16_konfl": ("GEMM z A w tym samym banku co B (r72.k)", lambda: gemm(16, a=72)),
    "gemm_acc16_bvar": ("GEMM: B zmienia się co 16 instrukcji (4 rejestry B na zmianę: r56,r60,r64,r68 parzyste), A r73.k",
                        lambda: gemm(16, bvar=True)),
    "kompilator_pelne": ("dokładnie pętla kompilatora (8 łańcuchów, a/b skalary), ale bez {Compacted}",
                         lambda: kompilator(False)),
    "kompilator_kompakt": ("dokładnie pętla kompilatora z {Compacted}", lambda: kompilator(True)),
}


def zbuduj(cialo, wynik):
    tmp = tempfile.mkdtemp()
    d = os.path.join(tmp, "d")
    shutil.copytree(SZABLON, d)
    asm_path = os.path.join(d, ".text.k_fma.asm")
    src = open(asm_path).read()
    # Zastąp CAŁE ciało pętli między "L560:" a "jmpi L560" naszym kodem plus sterowaniem pętli.
    # Uwaga: kompilator przeplata mad także ZA licznikiem pętli (przed cmp/jmpi), więc nie wystarczy
    # wyciąć kodu do "add r8.1". Zostawiamy tylko: add licznika, 2× cmp, jmpi.
    m = re.search(r"L560:\n(.*?)\n(\(W&f0\.0\) jmpi\s+L560[^\n]*)", src, re.S)
    if not m:
        sys.exit("nie znaleziono pętli L560 w szablonie")
    sterowanie = [l for l in m.group(1).splitlines() if re.search(r"r8\.1<1>:d|cmp \(", l)]
    assert len(sterowanie) == 3, sterowanie
    src = src[: m.start(1)] + "\n".join(cialo + sterowanie) + "\n" + src[m.start(2):]
    open(asm_path, "w").write(src)
    nazwa = os.path.basename(wynik)
    subprocess.run(["ocloc", "asm", "-file", nazwa, "-dump", d, "-device", "whl"],
                   check=True, stdout=subprocess.DEVNULL)
    shutil.move(os.path.join(d, nazwa), wynik)
    shutil.rmtree(tmp)


if __name__ == "__main__":
    if sys.argv[1] == "--lista":
        for k, (opis, _) in TESTY.items():
            print(f"{k:24s} {opis}")
        sys.exit()
    test, wynik = sys.argv[1], sys.argv[2]
    cialo = TESTY[test][1]()
    zbuduj(cialo, wynik)
    print(len(cialo))
