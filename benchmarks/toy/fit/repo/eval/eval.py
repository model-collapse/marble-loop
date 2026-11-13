"""Protected evaluator for toy-fit. The target function and validation set live only here."""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np


def target(x):
    return np.sin(3 * x) + 0.5 * x ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    x_tr = rng.uniform(-2, 2, 200)
    y_tr = target(x_tr) + rng.normal(0, 0.1, x_tr.shape)
    x_va = np.linspace(-2, 2, 400)
    y_va = target(x_va)

    spec = importlib.util.spec_from_file_location("model", os.path.join(os.getcwd(), "model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    predict = mod.fit(x_tr.copy(), y_tr.copy())
    p_va = np.asarray(predict(x_va.copy()), dtype=float)
    p_tr = np.asarray(predict(x_tr.copy()), dtype=float)
    valid = bool(p_va.shape == y_va.shape and np.all(np.isfinite(p_va)))
    res = {"valid": valid,
           "val_mse": float(np.mean((p_va - y_va) ** 2)) if valid else None,
           "train_mse": float(np.mean((p_tr - y_tr) ** 2)) if valid else None}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "score.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
