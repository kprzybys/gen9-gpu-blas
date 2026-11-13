"""Multithreaded latency-hiding model of the attention loop on one Gen9 EU (discrete-event, cycle level).

Why: a single-thread schedule model could not reproduce the measured prefetch-depth effect (docs/solvers.md section 5).
Here 7 hardware threads share one EU: 2 FPU pipes (a SIMD8 FP32 mad occupies a pipe for 2 clocks, 4 lanes per clock;
other ALU instructions c_other clocks), one instruction per thread per clock, in-order per thread; sends go to the
subslice data port: a FIFO with bw bytes per clock per EU (64 B/clk per subslice / 8 EUs = 8 at full load), then a
fixed latency L. A mad waits for the data of its step. Peak = 24 EU x 16 flop/clk.
Instruction stream per thread and block (Qwen shape D 128, NK 2, RQ 2, the measured C4 loop: 1229 instructions,
1024 mad, 64 send): D/8 K steps (32 mad, 2 sends of 128 B), softmax/other overhead, 16 V steps (32 mad, 2 sends).
Schemes: sends of step g issued after the mads of step g-NS (NS 1 = right before use, like IGC's template).
Measured (results/attn-asm-hd530-2026-11-13.txt, HD 530 1.0 GHz, GFLOPS): template 230.8, NS2 287.7, NS4 295.5.
Usage: python3 mt_model.py sweep [workers]   |   python3 mt_model.py one L bw c_other
"""
import itertools
import json
import multiprocessing as mp
import sys

MEAS = {1: 230.8, 2: 287.7, 4: 295.5}
PEAK = 384.0            # 24 EU x 16 flop/clk x 1.0 GHz


def stream(NS, other=77, D=128, NK=2, RQ=2, blocks=6):
    nks, nvs = D // 8, 8 * NK
    steps = [("K", RQ * NK * 8, NK) for _ in range(nks)] + [("V", RQ * D // 8, D // 64) for _ in range(nvs)]
    per = len(steps)
    total = per * blocks
    after = {}
    for g in range(total):
        after.setdefault(g - NS, []).append(g)
    prog = []                                   # (kind, step) ; kind in mad / alu / send
    for g0 in sorted(x for k in range(-NS, 0) for x in after.get(k, [])):
        prog += [("send", g0)] * steps[g0 % per][2] + [("alu", None)] * steps[g0 % per][2]
    for g in range(total):
        kind, nm, ns = steps[g % per]
        if kind == "V" and g % per == nks:
            prog += [("alu", None)] * other
        prog += [("mad", g)] * nm
        for g2 in after.get(g, []):
            prog += [("send", g2)] * steps[g2 % per][2] + [("alu", None)] * steps[g2 % per][2]
    return prog, blocks


def simulate(NS, L, bw, c_other, threads=7, maxclk=400000):
    prog, blocks = stream(NS)
    nsends = {}
    for k, g in prog:
        if k == "send":
            nsends[g] = nsends.get(g, 0) + 1
    pc = [0] * threads
    ready_at = [0] * threads                    # thread may issue at this clock (in-order, 1 per clock)
    arrived = [dict() for _ in range(threads)]  # step -> sends arrived
    pending = []                                # (arrival clock, thread, step)
    port_free = 0.0
    pipes = [0, 0]                              # clock each FPU pipe becomes free
    mads = 0
    rr = 0
    t = 0
    done = 0
    while done < threads and t < maxclk:
        # deliver data
        if pending:
            keep = []
            for a, th, g in pending:
                if a <= t:
                    arrived[th][g] = arrived[th].get(g, 0) + 1
                else:
                    keep.append((a, th, g))
            pending = keep
        issued_fpu = 0
        for i in range(threads):
            th = (rr + i) % threads
            if pc[th] >= len(prog) or ready_at[th] > t:
                continue
            kind, g = prog[pc[th]]
            if kind == "send":
                start = max(t, port_free)
                port_free = start + 128.0 / bw
                pending.append((start + 128.0 / bw + L, th, g))
            else:
                if kind == "mad" and arrived[th].get(g, 0) < nsends[g]:
                    continue                    # waits for its data
                p = 0 if pipes[0] <= pipes[1] else 1
                if pipes[p] > t or issued_fpu >= 2:
                    continue
                pipes[p] = t + (2 if kind == "mad" else c_other)
                issued_fpu += 1
                if kind == "mad":
                    mads += 1
            pc[th] += 1
            ready_at[th] = t + 1
            if pc[th] >= len(prog):
                done += 1
        rr = (rr + 1) % threads
        t += 1
    # steady state is not separated: 6 blocks per thread, start-up included (same for all schemes)
    flops = mads * 8 * 2
    gflops = flops / t * 24 * 1.0                 # flop per clock per EU x 24 EU x 1 GHz
    return gflops, t


def job(a):
    L, bw, c_other = a
    r = {NS: simulate(NS, L, bw, c_other)[0] for NS in (1, 2, 4)}
    err = max(abs(r[NS] / MEAS[NS] - 1) for NS in r)
    return dict(L=L, bw=bw, c_other=c_other, pred=r, max_rel_err=err,
                ratio_ns4_ns2=r[4] / r[2], ratio_ns2_tmpl=r[2] / r[1])


def main():
    if sys.argv[1] == "one":
        L, bw, c = float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
        print(job((L, bw, c)))
        return
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    grid = list(itertools.product((80, 120, 150, 200, 250, 300, 400, 500, 600), (4, 5, 6, 7, 8, 10, 12, 16, 1000), (1, 2)))
    with mp.Pool(workers) as p:
        res = p.map(job, grid)
    res.sort(key=lambda r: r["max_rel_err"])
    json.dump(res, open("mt_results.json", "w"), indent=1)
    print(f"measured: template {MEAS[1]}, NS2 {MEAS[2]}, NS4 {MEAS[4]}  (NS4/NS2 {MEAS[4] / MEAS[2]:.3f}, "
          f"NS2/template {MEAS[2] / MEAS[1]:.3f})")
    for r in res[:15]:
        p = r["pred"]
        print(f"L {r['L']:4} bw {r['bw']:5} c_other {r['c_other']}: tmpl {p[1]:6.1f} NS2 {p[2]:6.1f} NS4 {p[4]:6.1f} | "
              f"NS4/NS2 {r['ratio_ns4_ns2']:.3f} NS2/tmpl {r['ratio_ns2_tmpl']:.3f} | max err {r['max_rel_err']:.1%}")


if __name__ == "__main__":
    main()
