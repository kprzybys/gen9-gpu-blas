"""CP-SAT model of the register allocation in attn_gen.py sched() (Gen9: 128 GRF of 32 B).

Per instance (from extract.py) and configuration (NS, PE_IN_S, COMPACT, relocation of IGC's Q/O/l registers):
place NS ring slots (8 contiguous GRF starting at an even register), the score registers S, the p copies (even Pe,
odd Po), running sums, temporaries and send headers, next to IGC's registers. Zero bank conflicts require: slots at
even bases (the generator's design), Pe even / Po odd, and as many even as odd Q registers (each Q register holds 4
elements that need each parity; elements are swapped pairwise between an even and an odd Q register).
With relocation, Q/O/l may move (cost: Q 1 mov before the loop, O and l 2 movs: before and after). The objective is
the number of movs. Also reports whether the generator's greedy first-fit (slots first) succeeds.
Usage: python3 solve.py instances.json [workers]  -> results.json and a table on stdout.
"""
import itertools
import json
import multiprocessing as mp
import sys

from ortools.sat.python import cp_model

N = 128


def needs(P, NS, pe_in_s, compact):
    RQ, NK, D = P["RQ"], P["NK"], P["D"]
    s = RQ * NK
    singles = []                                   # (name, parity or None)
    singles += [("S", 0 if pe_in_s else None)] * s
    if not pe_in_s:
        singles += [("Pe", 0)] * s
    singles += [("Po", 1)] * s
    if not compact:
        singles += [("L", None)] * RQ
    singles += [("tmp", None)] * (2 if compact else 3)
    singles += [("hK", None)] * NK + [("hV", None)] * (D // 64)
    return singles


def greedy(inst, NS, pe_in_s, compact):
    """attn_gen.Alloc order: slots (lowest even base with 8 free), then singles in sched() order."""
    free = sorted(inst["free_now"])
    for _ in range(NS):
        b = next((b for b in free if b % 2 == 0 and all(x in free for x in range(b, b + 8))), None)
        if b is None:
            return False
        for x in range(b, b + 8):
            free.remove(x)
    for _, par in needs(inst["P"], NS, pe_in_s, compact):
        r = next((x for x in free if par is None or x % 2 == par), None)
        if r is None:
            return False
        free.remove(r)
    return True


def solve(args):
    name, inst, NS, pe_in_s, compact, reloc, tlimit = args
    P = inst["P"]
    m = cp_model.CpModel()
    fx = inst["fixed"]
    unmovable = set(fx["r0"]) | set(fx["ptr"]) | set(fx["cntlim"]) | set(fx["epi_other"])
    movable = [("q", r, 1) for r in fx["q"]] + [("o", r, 2) for r in fx["o"]] + [("l", r, 2) for r in fx["l"]]
    if not reloc:
        unmovable |= {r for _, r, _ in movable}
        movable = []
    if compact and len(fx["l"]) != P["RQ"]:
        return dict(name=name, NS=NS, pe=pe_in_s, compact=compact, reloc=reloc, status="N/A (COMPACT needs RQ l regs)")
    intervals, starts = [], {}
    for r in sorted(unmovable):
        intervals.append(m.NewFixedSizeIntervalVar(r, 1, f"fix{r}"))

    def single(tag, parity, hint=None):
        if parity is None:
            x = m.NewIntVar(1, N - 1, tag)
        else:
            y = m.NewIntVar(0, (N - 1 - parity) // 2, tag + "_h")
            x = m.NewIntVar(1, N - 1, tag)
            m.Add(x == 2 * y + parity)
        if hint is not None:
            m.AddHint(x, hint)
        intervals.append(m.NewIntervalVar(x, 1, x + 1, tag + "_i"))
        return x

    cost = []
    qpar = []
    for kind, r, c in movable:
        x = single(f"{kind}{r}", None, r)
        moved = m.NewBoolVar(f"mv_{kind}{r}")
        m.Add(x != r).OnlyEnforceIf(moved)
        m.Add(x == r).OnlyEnforceIf(moved.Not())
        cost.append(c * moved)
        if kind == "q":
            p = m.NewIntVar(0, 1, f"par{r}")
            h = m.NewIntVar(0, N // 2, f"half{r}")
            m.Add(x == 2 * h + p)
            qpar.append(p)
    if not reloc:
        nodd = sum(r % 2 for r in fx["q"])
        balanced = 2 * nodd == len(fx["q"])
    else:
        m.Add(2 * sum(qpar) == len(fx["q"]))
        balanced = True
    for i in range(NS):
        y = m.NewIntVar(0, (N - 8) // 2, f"slot{i}_h")
        b = m.NewIntVar(0, N - 8, f"slot{i}")
        m.Add(b == 2 * y)
        intervals.append(m.NewIntervalVar(b, 8, b + 8, f"slot{i}_i"))
        starts[f"slot{i}"] = b
    for j, (tag, par) in enumerate(needs(P, NS, pe_in_s, compact)):
        starts[f"{tag}{j}"] = single(f"{tag}{j}", par)
    m.AddNoOverlap(intervals)
    m.Minimize(sum(cost) if cost else 0)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = tlimit
    s.parameters.num_workers = 8
    st = s.Solve(m)
    res = dict(name=name, NS=NS, pe=pe_in_s, compact=compact, reloc=reloc, status=s.StatusName(st),
               q_balanced_now=balanced, greedy=greedy(inst, NS, pe_in_s, compact) if not reloc else None,
               time=round(s.WallTime(), 2))
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        res["movs"] = int(s.ObjectiveValue())
        res["slots"] = sorted(s.Value(starts[f"slot{i}"]) for i in range(NS))
    return res


def main():
    insts = {k: v for k, v in json.load(open(sys.argv[1])).items() if "error" not in v}
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    jobs = [(n, inst, NS, pe, cp, rl, 60.0) for n, inst in insts.items()
            for NS, pe, cp, rl in itertools.product(range(1, 9), (0, 1), (0, 1), (0, 1))]
    with mp.Pool(workers) as pool:
        out = pool.map(solve, jobs)
    json.dump(out, open("results.json", "w"), indent=1)
    for n, inst in insts.items():
        P = inst["P"]
        nks, nvs = P["D"] // 8, 8 * P["NK"]
        print(f"\n== {n} ({inst['opts']}); free now {len(inst['free_now'])}; generator rules: NS <= {nks}, "
              f"{nks + nvs} % NS == 0")
        print("NS | gen ok | config (PE_IN_S,COMPACT) | IGC regs fixed: solver / greedy | relocatable: solver (movs)")
        for NS in range(1, 9):
            gen_ok = NS <= nks and (nks + nvs) % NS == 0
            for pe, cp in itertools.product((0, 1), (0, 1)):
                f = next(r for r in out if r["name"] == n and r["NS"] == NS and r["pe"] == pe and r["compact"] == cp and not r["reloc"])
                g = next(r for r in out if r["name"] == n and r["NS"] == NS and r["pe"] == pe and r["compact"] == cp and r["reloc"])
                fs = f["status"] if "N/A" in f["status"] else f"{f['status']} / {'ok' if f['greedy'] else 'FAIL'}"
                gs = g["status"] + (f" ({g['movs']})" if "movs" in g else "")
                print(f"{NS:2} | {'yes' if gen_ok else 'no ':3}    | ({pe},{cp})                     | {fs:28} | {gs}")


if __name__ == "__main__":
    main()
