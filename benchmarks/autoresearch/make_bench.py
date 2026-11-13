#!/usr/bin/env python3
"""Generate the MARBLE autoresearch benchmark from a karpathy/autoresearch checkout.

Usage:
  AUTORESEARCH_ROOT=/path/to/autoresearch python make_bench.py [--device-batch-size 16] [--out generated/]

The baseline train.py is the upstream file; only DEVICE_BATCH_SIZE may be overridden so the
default config fits smaller GPUs (upstream 128 targets an 80 GB H100; 16 fits a 24 GB A10G).
Run `uv run prepare.py` in the checkout once beforehand (data + tokenizer in ~/.cache/autoresearch).
"""
import argparse
import os
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

YAML = """\
schema_version: 1
name: {name}
description: >
  karpathy/autoresearch: edit train.py (model, optimizer, hyperparameters, training loop) to reach
  the lowest validation bits-per-byte after a fixed {budget:.0f}-second training budget on one GPU.
  prepare.py (data, tokenizer, time budget, evaluate_bpb) is protected. Baseline: upstream train.py
  @ {commit} with DEVICE_BATCH_SIZE={dbs}. Note: val_bpb is self-reported by train.py (see eval/ar_eval.py).
repo: repo
editable: ["train.py"]
protected: ["prepare.py", "eval/**"]
env:
  PYTHONDONTWRITEBYTECODE: "1"
evaluate:
  train:
    cmd: "${{AUTORESEARCH_ROOT}}/.venv/bin/python eval/ar_eval.py --python ${{AUTORESEARCH_ROOT}}/.venv/bin/python --out {{out}}"
    timeout: 1500
    resources: {{gpu: 1, cpu: 4}}
default_mode: train
score:
  file: "{{out}}/score.json"
  key: val_bpb
  direction: minimize
  extra_keys: [training_seconds, wall_seconds, peak_vram_mb, num_steps, mfu_percent, num_params_M, trust]
correctness: {{key: correct, equals: true}}
noise:
  repeats: 2
  min_rel_improvement: 0.002
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--autoresearch", default=os.environ.get("AUTORESEARCH_ROOT"))
    ap.add_argument("--device-batch-size", type=int, default=None, help="override DEVICE_BATCH_SIZE (upstream: 128)")
    ap.add_argument("--out", default=str(HERE / "generated"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if not a.autoresearch:
        ap.error("set --autoresearch or AUTORESEARCH_ROOT")
    root = Path(a.autoresearch)
    train = (root / "train.py").read_text()
    m = re.search(r"^DEVICE_BATCH_SIZE = (\d+)", train, re.M)
    if not m:
        raise SystemExit("DEVICE_BATCH_SIZE not found in train.py")
    dbs = a.device_batch_size or int(m.group(1))
    if a.device_batch_size:
        train = re.sub(r"^DEVICE_BATCH_SIZE = \d+", f"DEVICE_BATCH_SIZE = {dbs}", train, count=1, flags=re.M)
    budget = float(re.search(r"^TIME_BUDGET\s*=\s*([0-9.]+)", (root / "prepare.py").read_text(), re.M).group(1))
    commit = "unknown"
    head = root / ".git" / "HEAD"
    if head.exists():
        ref = head.read_text().strip()
        commit = ((root / ".git" / ref[5:]).read_text().strip() if ref.startswith("ref: ") and (root / ".git" / ref[5:]).exists() else ref)[:7]
    name = f"ar-dbs{dbs}"
    dest = Path(a.out) / name
    if dest.exists():
        if not a.force:
            print(f"skip {dest} (exists; use --force)")
            return 0
        shutil.rmtree(dest)
    (dest / "repo" / "eval").mkdir(parents=True)
    (dest / "repo" / "train.py").write_text(train)
    shutil.copy2(root / "prepare.py", dest / "repo" / "prepare.py")
    shutil.copy2(HERE / "ar_eval.py", dest / "repo" / "eval" / "ar_eval.py")
    (dest / "benchmark.yaml").write_text(YAML.format(name=name, budget=budget, commit=commit, dbs=dbs))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
