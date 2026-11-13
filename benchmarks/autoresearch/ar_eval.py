"""MARBLE evaluator for karpathy/autoresearch (protected; copied to eval/ar_eval.py).

Runs `train.py` (the editable file) with the autoresearch venv, measures wall-clock time from the
outside, parses the summary block, and writes {out}/score.json.

Integrity (M0): prepare.py (time budget, data, evaluate_bpb) is protected and checksum-verified by
MARBLE; the run is rejected if wall-clock time or the reported training time exceed the budget.
The model still computes its own loss inside evaluate_bpb, so val_bpb is *self-reported*
(score.json: "trust": "self_reported"). Planned fix (M1): protected evaluator recomputes val_bpb
from logits of the saved weights.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

KEYS = {"val_bpb": float, "training_seconds": float, "total_seconds": float, "peak_vram_mb": float,
        "mfu_percent": float, "total_tokens_M": float, "num_steps": int, "num_params_M": float, "depth": int}


def time_budget(prepare_path):
    m = re.search(r"^TIME_BUDGET\s*=\s*([0-9.]+)", open(prepare_path).read(), re.M)
    if not m:
        raise RuntimeError("TIME_BUDGET not found in prepare.py")
    return float(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True, help="autoresearch venv python")
    ap.add_argument("--out", required=True)
    ap.add_argument("--startup-allowance", type=float, default=600.0,
                    help="seconds allowed beyond TIME_BUDGET for compile + final eval")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    budget = time_budget("prepare.py")
    res = {"correct": False, "val_bpb": None, "time_budget_s": budget, "trust": "self_reported"}
    limit = budget + a.startup_allowance
    log_path = os.path.join(a.out, "train.log")
    t0 = time.time()
    with open(log_path, "w") as log:
        p = subprocess.Popen([a.python, "train.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        tail = []
        for line in p.stdout:  # stream through so the launcher sees progress
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            tail.append(line)
            tail = tail[-400:]
            if time.time() - t0 > limit:
                p.kill()
                break
        p.wait()
    wall = time.time() - t0
    res["wall_seconds"], res["returncode"] = wall, p.returncode
    text = "".join(tail)
    summary = text.split("\n---\n")[-1] if "\n---\n" in text else ""
    for k, typ in KEYS.items():
        m = re.search(rf"^{k}:\s*([-+0-9.eE]+)", summary, re.M)
        if m:
            res[k] = typ(m.group(1))
    problems = []
    if p.returncode != 0:
        problems.append(f"train.py exited with {p.returncode}")
    if wall > limit:
        problems.append(f"wall clock {wall:.0f}s exceeded budget+allowance {limit:.0f}s")
    v = res.get("val_bpb")
    if v is None or not math.isfinite(v) or v <= 0:
        problems.append("no valid val_bpb in output")
    ts = res.get("training_seconds")
    if ts is None or ts > budget * 1.05 + 5:
        problems.append(f"reported training_seconds {ts} exceeds TIME_BUDGET {budget:.0f}s")
    if ts is not None and wall < ts:
        problems.append(f"reported training_seconds {ts:.0f}s > measured wall clock {wall:.0f}s")
    if not res.get("num_steps"):
        problems.append("num_steps missing or 0")
    res["correct"] = not problems
    if problems:
        res["error"] = "; ".join(problems)
        res["log_tail"] = text[-3000:]
    with open(os.path.join(a.out, "score.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "log_tail"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
