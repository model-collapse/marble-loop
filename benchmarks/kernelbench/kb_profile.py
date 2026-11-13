"""Profiling driver: run ModelNew forward passes on the reference inputs (wrap with nsys / ncu).

Protected file: copied into each generated problem as eval/kb_profile.py.
"""
import argparse
import importlib.util
import sys

import torch


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--which", choices=["new", "ref"], default="new")
    a = ap.parse_args()
    ref = load(a.ref, "reference")
    mod = load(a.kernel, "model_new") if a.which == "new" else ref
    cls = mod.ModelNew if a.which == "new" else ref.Model
    torch.manual_seed(0)
    model = cls(*ref.get_init_inputs()).cuda()
    inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in ref.get_inputs()]
    with torch.no_grad():
        for _ in range(a.warmup):
            model(*inputs)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("marble_profile")
        for _ in range(a.iters):
            model(*inputs)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    print(f"ran {a.iters} iterations of {a.which}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
