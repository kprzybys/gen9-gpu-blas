"""Hybrid LAPACK model, steps 1-2 (results/lapack-hybrid-model-2026-11-13.txt):
(1) gen9blas rank-nb trailing update shapes: SGEMM C(m x m) -= A(m x nb) B(nb x m) and SSYRK lower C -= A A^T, beta = 1,
    ordinary NumPy arrays (userptr, as a LAPACK caller would pass them);
(2) OpenBLAS (scipy.linalg.lapack, system libopenblas pthread) spotrf / sgetrf / dpotrf / dgetrf / dposv baselines.
Every line carries package power (RAPL), package temperature and the thread count. Power states: 'sustained' = after
>= 35 s of continuous load (past the 32 s PL1 window), 'turbo' = first call after >= 30 s idle.
Run: python3 bench/lapack_baseline.py [part1|part2|turbo|all]"""
import glob, os, sys, time
import numpy as np

RAPL = "/sys/class/powercap/intel-rapl:0/energy_uj"


def energy():
    with open(RAPL) as f:
        return int(f.read())


def temp():
    for h in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            if open(h + "/name").read().strip() == "coretemp":
                return int(open(h + "/temp1_input").read()) / 1000
        except OSError:
            pass
    return float("nan")


def timed(fn):
    e0, t0 = energy(), time.perf_counter()
    r = fn()
    t1, e1 = time.perf_counter(), energy()
    return r, t1 - t0, (e1 - e0) / 1e6 / (t1 - t0)


def threads():
    return os.environ.get("OPENBLAS_NUM_THREADS", f"default({os.cpu_count()})")


def part1():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
    import gen9blas as g
    rng = np.random.default_rng(1)
    print("# part 1: gen9blas rank-nb update, FP32, beta = 1, ordinary NumPy arrays; GFLOPS: GEMM 2 m^2 nb, SYRK m^2 nb")
    for m in (2048, 4096, 8192):
        C = rng.standard_normal((m, m), dtype=np.float32)
        for nb in (256, 512):
            A = rng.standard_normal((m, nb), dtype=np.float32)
            B = np.ascontiguousarray(A.T)
            for name, fn, fl in (("sgemm", lambda: g.gemm(A, B, alpha=-1.0, beta=1.0, C=C), 2.0 * m * m * nb),
                                 ("ssyrk", lambda: g.syrk(A, uplo="L", alpha=-1.0, beta=1.0, C=C), 1.0 * m * m * nb)):
                fn(); fn()   # first use: pinning of the arrays
                ts, ws = [], []
                for _ in range(5):
                    _, t, w = timed(fn)
                    ts.append(t); ws.append(w)
                t = min(ts)
                print(f"{name} m={m:5d} nb={nb:3d}  {t*1e3:8.2f} ms  {fl/t/1e9:6.1f} GFLOPS  (median {fl/np.median(ts)/1e9:6.1f})"
                      f"  package {np.median(ws):5.1f} W  {temp():.0f} C", flush=True)


def spd(n, dt, rng):
    X = rng.standard_normal((n, n)).astype(dt)
    return (X @ X.T / n + np.eye(n, dtype=dt)).astype(dt)


def part2(state, sizes=(2048, 4096, 8192)):
    from scipy.linalg import lapack as L
    rng = np.random.default_rng(2)
    print(f"# part 2: OpenBLAS via scipy.linalg.lapack, threads {threads()}, state {state}")
    for n in sizes:
        S64 = spd(n, np.float64, rng); S32 = S64.astype(np.float32)
        G64 = rng.standard_normal((n, n)); G32 = G64.astype(np.float32)
        b = rng.standard_normal((n, 1))
        cases = (("spotrf", lambda: L.spotrf(S32, lower=1), n**3 / 3), ("dpotrf", lambda: L.dpotrf(S64, lower=1), n**3 / 3),
                 ("sgetrf", lambda: L.sgetrf(G32), 2 * n**3 / 3), ("dgetrf", lambda: L.dgetrf(G64), 2 * n**3 / 3),
                 ("dposv", lambda: L.dposv(S64, b, lower=1), n**3 / 3))
        reps = 3 if n <= 4096 else 1
        for name, fn, fl in cases:
            if state == "turbo":
                time.sleep(30)
            ts, ws = [], []
            for _ in range(1 if state == "turbo" else reps):
                r, t, w = timed(fn)
                assert r[-1] == 0, (name, r[-1])
                ts.append(t); ws.append(w)
            t = min(ts)
            print(f"{name} n={n:5d}  {t*1e3:9.1f} ms  {fl/t/1e9:6.1f} GFLOPS  package {np.median(ws):5.1f} W  {temp():.0f} C", flush=True)


def heat(seconds=35):
    from scipy.linalg import lapack as L
    S = spd(4096, np.float32, np.random.default_rng(3))
    t0 = time.time()
    while time.time() - t0 < seconds:
        L.spotrf(S, lower=1)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    pl = [open(f"/sys/class/powercap/intel-rapl:0/constraint_{i}_{k}").read().strip() for i in (0, 1) for k in ("name", "power_limit_uw", "time_window_us")]
    print(f"# RAPL constraints: {pl}; start temp {temp():.0f} C")
    if what in ("part1", "all"):
        part1()
    if what in ("part2", "all"):
        heat(); part2("sustained")
    if what in ("turbo",):
        part2("turbo", sizes=(4096,))
