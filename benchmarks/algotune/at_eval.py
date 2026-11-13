"""MARBLE evaluator for one AlgoTune task (runs with the AlgoTune venv's python, -I mode).

Protected file: copied into each generated task as eval/at_eval.py.

Design (integrity):
  * The driver never imports candidate code. Each timed run happens in a fresh *worker* process:
    one for the reference `Task.solve`, one for the candidate `Solver.solve`, interleaved.
  * Within a worker, every problem is timed exactly once (after a warm-up on a *train* problem),
    so memoising answers across calls cannot inflate the score.
  * Candidate solutions are sent back as JSON (AlgoTune's DatasetEncoder) and verified with
    `Task.is_solution` in the driver, outside the candidate's process. The driver runs with -I and
    without the candidate directory on sys.path, so decoding cannot import candidate modules.
Score: geometric mean over problems of (reference time / candidate time), using the minimum time
over repeats for each problem. Writes {out}/score.json; exits 0 whenever it could write a score.
"""
import argparse
import copy
import glob
import importlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time

MODES = {"quick": {"n_problems": 10, "repeats": 3}, "full": {"n_problems": None, "repeats": 5}}
WORKER_TIMEOUT_S = 3600


def add_root(root):
    if root not in sys.path:
        sys.path.insert(0, root)


def task_instance(root, task):
    add_root(root)
    importlib.import_module(f"AlgoTuneTasks.{task}.{task}")
    from AlgoTuneTasks.base import TASK_REGISTRY
    return TASK_REGISTRY[task]()


def split_file(data_dir, task, split):
    hits = sorted(glob.glob(os.path.join(data_dir, f"{task}_*_{split}.jsonl")))
    if not hits:
        raise FileNotFoundError(f"no {split} split for {task} in {data_dir}")
    return hits[0]


def load_problems(root, data_dir, task, split, n=None):
    add_root(root)
    from AlgoTuner.utils.streaming_json import stream_jsonl
    out = []
    for rec in stream_jsonl(split_file(data_dir, task, split)):
        out.append(rec["problem"])
        if n is not None and len(out) >= n:
            break
    return out


# --------------------------------------------------------------------------- worker


def worker(a):
    problems = load_problems(a.root, a.data, a.task, "test", a.n)
    warm = load_problems(a.root, a.data, a.task, "train", 1)[0]
    if a.impl == "ref":
        t = task_instance(a.root, a.task)
        solve = t.solve
    else:
        add_root(a.root)
        sol_dir = os.path.dirname(os.path.abspath(a.solver))
        sys.path.insert(0, sol_dir)
        spec = importlib.util.spec_from_file_location("solver", a.solver)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        s = mod.Solver()
        solve = s.solve
    solve(copy.deepcopy(warm))  # warm-up: imports, JIT, caches of *other* inputs only
    times, sols, errors = [], [], []
    for p in problems:
        pc = copy.deepcopy(p)
        t0 = time.perf_counter()
        try:
            r = solve(pc)
            errors.append(None)
        except Exception as e:  # candidate bug on this problem
            r = None
            errors.append(f"{type(e).__name__}: {e}"[:500])
        times.append(time.perf_counter() - t0)
        sols.append(r)
    from AlgoTuner.utils.serialization import DatasetEncoder
    with open(a.result, "w") as f:
        json.dump({"times": times, "errors": errors, "solutions": sols if a.emit_solutions else None},
                  f, cls=DatasetEncoder)
    return 0


# --------------------------------------------------------------------------- driver


def run_worker(a, impl, emit, tmpdir, rep):
    res = os.path.join(tmpdir, f"{impl}_{rep}.json")
    cmd = [sys.executable, "-I", os.path.abspath(__file__), "--worker", "--impl", impl, "--root", a.root,
           "--data", a.data, "--task", a.task, "--solver", os.path.abspath(a.solver), "--result", res]
    if a.n is not None:
        cmd += ["--n", str(a.n)]
    if emit:
        cmd += ["--emit-solutions"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=WORKER_TIMEOUT_S)
    if p.returncode != 0 or not os.path.exists(res):
        raise RuntimeError(f"{impl} worker failed (exit {p.returncode}): {p.stderr[-3000:]}")
    add_root(a.root)
    from AlgoTuner.utils.serialization import dataset_decoder
    with open(res) as f:
        return json.load(f, object_hook=lambda d: dataset_decoder(d, base_dir=tmpdir))


def write(out, res):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "score.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "per_problem"}, default=str))


def driver(a):
    cfg = MODES[a.mode]
    a.n = cfg["n_problems"]
    res = {"task": a.task, "mode": a.mode, "correct": False, "speedup": None}
    t_start = time.time()
    task = task_instance(a.root, a.task)
    problems = load_problems(a.root, a.data, a.task, "test", a.n)
    res["n_problems"] = len(problems)
    ref_t = [math.inf] * len(problems)
    cand_t = [math.inf] * len(problems)
    invalid, cand_errors = [], []
    with tempfile.TemporaryDirectory(prefix="marble-at-") as tmp:
        for rep in range(cfg["repeats"]):
            verify = rep == 0 or a.mode == "full"
            try:
                cand = run_worker(a, "cand", verify, tmp, rep)
            except (RuntimeError, subprocess.TimeoutExpired) as e:
                res["error"] = f"candidate crashed: {e}"[:4000]
                write(a.out, res)
                return 0
            ref = run_worker(a, "ref", False, tmp, rep)
            for i, (tr, tc) in enumerate(zip(ref["times"], cand["times"])):
                ref_t[i] = min(ref_t[i], tr)
                cand_t[i] = min(cand_t[i], tc)
            for i, err in enumerate(cand["errors"]):
                if err and i not in [e[0] for e in cand_errors]:
                    cand_errors.append((i, err))
            if verify:
                for i, (p, s) in enumerate(zip(problems, cand["solutions"])):
                    if cand["errors"][i] is not None:
                        continue
                    try:
                        ok = bool(task.is_solution(copy.deepcopy(p), s))
                    except Exception as e:
                        ok = False
                        cand_errors.append((i, f"is_solution raised {type(e).__name__}: {e}"[:300]))
                    if not ok and i not in invalid:
                        invalid.append(i)
    bad = sorted(set(invalid) | {i for i, _ in cand_errors})
    res["invalid_problems"] = bad[:50]
    res["n_invalid"] = len(bad)
    res["candidate_errors"] = [f"#{i}: {e}" for i, e in cand_errors[:5]]
    ratios = [r / c for r, c in zip(ref_t, cand_t) if c > 0 and math.isfinite(r) and math.isfinite(c)]
    res["ref_time_total_s"] = sum(t for t in ref_t if math.isfinite(t))
    res["cand_time_total_s"] = sum(t for t in cand_t if math.isfinite(t))
    res["correct"] = len(bad) == 0 and len(ratios) == len(problems)
    if res["correct"]:
        res["speedup"] = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
        res["speedup_mean"] = sum(ratios) / len(ratios)
        res["speedup_min"] = min(ratios)
    else:
        res["error"] = f"{len(bad)}/{len(problems)} problems invalid or raised"
    res["per_problem"] = [{"ref_s": r, "cand_s": c} for r, c in zip(ref_t, cand_t)]
    res["eval_seconds"] = time.time() - t_start
    write(a.out, res)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="AlgoTune checkout")
    ap.add_argument("--data", required=True, help="directory with <task>_*_{train,test}.jsonl")
    ap.add_argument("--task", required=True)
    ap.add_argument("--solver", default="solver.py")
    ap.add_argument("--mode", choices=MODES, default="quick")
    ap.add_argument("--out")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--impl", choices=["ref", "cand"])
    ap.add_argument("--result")
    ap.add_argument("--n", type=int)
    ap.add_argument("--emit-solutions", action="store_true")
    a = ap.parse_args()
    if a.worker:
        return worker(a)
    if not a.out:
        ap.error("--out is required")
    return driver(a)


if __name__ == "__main__":
    sys.exit(main())
