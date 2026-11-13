"""CP-SAT register budget for an accurate HGEMM loop (FP16 products into FP16 chunk sums, carried into FP32) with a
tile of R rows x 32 columns on Gen9 (128 GRF), in the style of asm/gemm_gen.py (kinds h and hd).

Placed values: hf chunk accumulators R x 2 GRF (one GRF = 16 halves = one row x 16 columns), FP32 accumulators R x 2
pairs of 2 contiguous GRF, per buffer an A block (R*kl*2 B) and two B blocks (16 columns x kl halves = kl GRF each),
all blocks starting at an even register (the gemm_gen layout that gives zero bank conflicts), send headers.
Fixed: r0, r1, r127 (gemm_gen.free_regs), 2 pointers of 2 GRF, counter/limit 1 GRF, E registers live into the
epilogue (E measured on the existing kernels: 17 = every register the epilogue mentions, as free_regs blocks today;
9 = live-in only; 7 = live-in minus send headers zeroed after the loop, as attn_gen does).
Usage: python3 gemm_solve.py [workers]
"""
import itertools
import multiprocessing as mp
import sys

from ortools.sat.python import cp_model

N = 128


def solve(cfg):
    R, kl, nbuf, shared_hdr, E, f_even = cfg
    a_regs = R * kl * 2 // 32
    if a_regs < 1:
        return cfg, "N/A", None
    m = cp_model.CpModel()
    iv = []
    fixed = [0, 1, 127] + [2, 3, 4, 5] + [6] + list(range(7, 7 + E))   # positions are arbitrary: only the count matters
    for r in fixed:
        iv.append(m.NewFixedSizeIntervalVar(r, 1, f"fix{r}"))

    def block(name, size, even):
        if even:
            y = m.NewIntVar(0, (N - size) // 2, name + "h")
            b = m.NewIntVar(0, N - size, name)
            m.Add(b == 2 * y)
        else:
            b = m.NewIntVar(0, N - size, name)
        iv.append(m.NewIntervalVar(b, size, b + size, name + "i"))
        return b

    for r in range(R):
        for c in range(2):
            block(f"h{r}_{c}", 1, False)
            block(f"f{r}_{c}", 2, f_even)
    for b in range(nbuf):
        block(f"A{b}", a_regs, True)
        block(f"B0_{b}", kl, True)
        block(f"B1_{b}", kl, True)
    for h in range(3 if shared_hdr else 3 * nbuf):
        block(f"hdr{h}", 1, False)
    m.AddNoOverlap(iv)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = 60
    s.parameters.num_workers = 8
    st = s.Solve(m)
    need = 3 + 4 + 1 + E + R * 2 + R * 4 + nbuf * (a_regs + 2 * kl) + (3 if shared_hdr else 3 * nbuf)
    return cfg, s.StatusName(st), need


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    cfgs = list(itertools.product((16, 8), (2, 4, 8), (1, 2), (False, True), (17, 9, 7), (False, True)))
    with mp.Pool(workers) as p:
        res = p.map(solve, cfgs)
    print("R  kl buf hdr      E   f-pair | need/128 | CP-SAT")
    for (R, kl, nbuf, sh, E, fe), st, need in res:
        print(f"{R:2} {kl:2} {nbuf:3} {'shared' if sh else 'perbuf':7} {E:3} {'even' if fe else 'any ':4} | "
              f"{need if need else '-':>4}/128 | {st}")


if __name__ == "__main__":
    main()
