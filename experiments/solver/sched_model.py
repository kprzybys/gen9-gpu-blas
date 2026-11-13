"""Loop-schedule model of the attention kernel (attn_gen.py sched<NS>) and a CP-SAT search for better send placement.

Model (one hardware thread, in-order issue, one instruction per issue slot; Leff = effective read latency in this
thread's issue slots, i.e. what the other threads do not hide; calibrated against measurements):
  block of 8*NK keys = D/8 K steps (RQ*NK*8 mads, NK sends of 4 GRF each) + softmax overhead + 8*NK V steps
  (RQ*D/8 mads, D/64 sends of 4 GRF each). A send (with its header add) costs 2 slots. A step cannot start its first
  mad on a unit before send_start + Leff. Registers: a unit of 4 GRF is busy from its send to the last mad reading it.
Schedules:
  current(NS): ring of NS slots of 8 GRF (2 units); the sends of step g are issued right after the mads of step g-NS
               (simulated, steady state).
  free(U):     CP-SAT places every send anywhere before its step (order of each address stream kept), with at most U
               units busy at any time (same register budget at U = 2*NS); minimises the makespan of B blocks.
Usage: python3 sched_model.py calib|sweep [workers]
"""
import itertools
import json
import multiprocessing as mp
import sys

from ortools.sat.python import cp_model

SHAPES = {"qwen": dict(D=128, NK=2, RQ=2, other=77), "llama": dict(D=64, NK=2, RQ=4, other=154)}
B = 4                       # blocks simulated; steady state = blocks 1..B-2


def steps(S):
    D, NK, RQ = S["D"], S["NK"], S["RQ"]
    nks, nvs, nv = D // 8, 8 * NK, D // 64
    out = []                # (length, [(stream, first_off, last_off)], overhead_before)
    for c in range(nks):
        L = RQ * NK * 8
        out.append((L, [(("K", k), k * RQ, (7 * NK + k) * RQ + RQ) for k in range(NK)], 0))
    for j in range(nvs):
        L = RQ * D // 8
        out.append((L, [(("V", h), 8 * h, (RQ - 1) * nks + 8 * h + 8) for h in range(nv)], S["other"] if j == 0 else 0))
    return out


def current(S, NS, Leff):
    """Steady-state slots per block of the generator's scheme (simulation, in-order issue with stalls)."""
    st = steps(S) * B
    n = len(st)
    t = 0
    ready = {}                                  # (g, stream) -> time data ready
    issue_after = {}                            # step index -> list of g whose sends follow it
    for g in range(n):
        issue_after.setdefault(g - NS, []).append(g)
    starts = []
    for g0 in issue_after.get(-NS, []) + [x for k in range(-NS + 1, 0) for x in issue_after.get(k, [])]:
        for (stream, _, _) in st[g0][1]:
            ready[(g0, stream)] = t + Leff
            t += 2
    for g, (L, uses, ov) in enumerate(st):
        t += ov
        start = t
        for stream, first, _ in uses:
            start = max(start, ready[(g, stream)] - first)
        starts.append(start)
        t = start + L
        for g2 in issue_after.get(g, []):
            for stream, _, _ in st[g2][1]:
                ready[(g2, stream)] = t + Leff
                t += 2
    per = len(st) // B
    return (starts[(B - 1) * per] - starts[per]) / (B - 2)


def free(S, U, Leff, tlimit=120, workers=8):
    st = steps(S) * B
    m = cp_model.CpModel()
    H = sum(L + ov for L, _, ov in st) * 2 + 4 * Leff + 1000
    issue, occ, dem = [], [], []
    sstart = []
    prev_end = None
    sends_by_stream = {}
    for g, (L, uses, ov) in enumerate(st):
        s = m.NewIntVar(0, H, f"s{g}")
        e = m.NewIntVar(0, H, f"e{g}")
        m.Add(e == s + L)
        issue.append(m.NewIntervalVar(s, L, e, f"st{g}"))
        if prev_end is not None:
            m.Add(s >= prev_end + ov)
        prev_end = e
        sstart.append(s)
        for stream, first, last in uses:
            x = m.NewIntVar(0, H, f"x{g}{stream}")
            issue.append(m.NewIntervalVar(x, 2, x + 2, f"snd{g}{stream}"))
            m.Add(s + first >= x + Leff)
            end_use = m.NewIntVar(0, H + L, f"u{g}{stream}")
            m.Add(end_use == s + last)
            occ.append(m.NewIntervalVar(x, m.NewIntVar(0, H, f"d{g}{stream}"), end_use, f"o{g}{stream}"))
            dem.append(1)
            sends_by_stream.setdefault(stream, []).append(x)
    for xs in sends_by_stream.values():
        for a, b in zip(xs, xs[1:]):
            m.Add(b >= a + 2)
    m.AddNoOverlap(issue)
    m.AddCumulative(occ, dem, U)
    m.Minimize(prev_end)
    sol = cp_model.CpSolver()
    sol.parameters.max_time_in_seconds = tlimit
    sol.parameters.num_workers = workers
    r = sol.Solve(m)
    if r not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, sol.StatusName(r)
    per = len(st) // B
    return (sol.Value(sstart[(B - 1) * per]) - sol.Value(sstart[per])) / (B - 2), sol.StatusName(r)


def slots_ideal(S):
    return sum(L + ov + 2 * len(u) for L, u, ov in steps(S))


def calib():
    S = SHAPES["qwen"]
    ideal = slots_ideal(S)
    print(f"qwen ideal slots/block {ideal} (measured loop 1229 instructions); measured NS4/NS2 = 295.5/287.7 = 1.027,"
          f" NS4 efficiency vs issue bound 320 GFLOPS = 0.923, NS2 = 0.899")
    for Leff in range(0, 401, 20):
        c2, c4 = current(S, 2, Leff), current(S, 4, Leff)
        print(f"Leff {Leff:3}: NS2 {c2:7.1f} (eff {ideal / c2:.3f})  NS4 {c4:7.1f} (eff {ideal / c4:.3f})  NS4/NS2 {c2 / c4:.3f}")


def job(a):
    name, U, Leff = a
    S = SHAPES[name]
    t, status = free(S, U, Leff)
    return dict(shape=name, U=U, Leff=Leff, free=t, status=status)


def sweep(workers):
    jobs = []
    for name, Leffs in (("qwen", (100, 140, 180, 220)), ("llama", (100, 140, 180, 220))):
        for U in range(2, 11):
            for Leff in Leffs:
                jobs.append((name, U, Leff))
    with mp.Pool(workers) as p:
        res = p.map(job, jobs)
    for r in res:
        S = SHAPES[r["shape"]]
        NS = r["U"] // 2
        r["ideal"] = slots_ideal(S)
        r["current_same_regs"] = current(S, NS, r["Leff"]) if r["U"] % 2 == 0 and NS >= 1 else None
    json.dump(res, open("sched_results.json", "w"), indent=1)
    for r in res:
        cur = r["current_same_regs"]
        gain = f"{cur / r['free'] - 1:+.1%}" if cur and r["free"] else "-"
        fe = f"{r['ideal'] / r['free']:.3f}" if r["free"] else "-"
        ce = f"{r['ideal'] / cur:.3f}" if cur else "-"
        print(f"{r['shape']:5} U={r['U']:2} ({4 * r['U']:2} GRF) Leff={r['Leff']:3}: free eff {fe} [{r['status']}]  "
              f"current NS={r['U'] // 2 if r['U'] % 2 == 0 else '-'} eff {ce}  gain {gain}")


if __name__ == "__main__":
    if sys.argv[1] == "calib":
        calib()
    else:
        sweep(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
