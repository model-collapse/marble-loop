"""Protected evaluator for toy-pairwise. Writes {out}/score.json."""
import argparse
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODES = {"quick": {"n": 120, "d": 16, "reps": 3}, "full": {"n": 240, "d": 32, "reps": 5}}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def as_matrix(m, n):
    rows = [list(map(float, r)) for r in (m.tolist() if hasattr(m, "tolist") else m)]
    if len(rows) != n or any(len(r) != n for r in rows):
        raise ValueError(f"expected {n}x{n} result")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES, default="quick")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = MODES[a.mode]
    rng = random.Random(1234)
    X = [[rng.uniform(-1, 1) for _ in range(cfg["d"])] for _ in range(cfg["n"])]

    base = load(os.path.join(HERE, "baseline_impl.py"), "baseline_impl")
    sol = load(os.path.join(os.getcwd(), "solution.py"), "solution")

    expected = base.pairwise_distances([r[:] for r in X])
    got = as_matrix(sol.pairwise_distances([r[:] for r in X]), cfg["n"])
    max_err = max(abs(e - g) for er, gr in zip(expected, got) for e, g in zip(er, gr))
    correct = bool(math.isfinite(max_err) and max_err < 1e-6)

    def timeit(fn):
        ts = []
        for _ in range(cfg["reps"]):
            Xc = [r[:] for r in X]  # fresh input each run
            t = time.perf_counter()
            fn(Xc)
            ts.append(time.perf_counter() - t)
        return statistics.median(ts)

    t_base = timeit(base.pairwise_distances)
    t_sol = timeit(sol.pairwise_distances)
    res = {"correct": correct, "max_abs_err": max_err, "runtime_s": t_sol, "baseline_runtime_s": t_base,
           "speedup": t_base / t_sol if t_sol > 0 else None, "mode": a.mode}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "score.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
