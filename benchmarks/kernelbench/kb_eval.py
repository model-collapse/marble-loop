"""MARBLE evaluator for one KernelBench problem (runs inside the KernelBench venv).

Protected file: copied into each generated problem as eval/kb_eval.py.
Writes {out}/score.json with: correct, speedup (vs eager PyTorch reference), runtimes, static-check
results and KernelBench's excessive-speedup flag. Correctness is KernelBench's own check. The score
`speedup` uses MARBLE timing (reference and candidate interleaved in rounds, equal warm-up, median),
because KernelBench's mean over a fixed candidate-then-reference order is biased by first-call outliers
(e.g. 0.68x for an identical copy of max-pool 2D on A10G). KernelBench's numbers are kept as kb_*. Always exits 0 when it could write a score file,
so that MARBLE classifies a broken kernel as "incorrect" rather than "failed".
"""
import argparse
import json
import os
import sys
import time
import traceback

MODES = {
    "quick": {"correct_trials": 3, "perf_trials": 30, "compile_ref": False, "rounds": 4, "per_round": 10},
    "full": {"correct_trials": 5, "perf_trials": 100, "compile_ref": True, "rounds": 8, "per_round": 25},
}
WARMUP = 10
ARCH_BY_CAPABILITY = {7: {0: "Volta", 5: "Turing"}, 8: {0: "Ampere", 6: "Ampere", 7: "Ampere", 9: "Ada"},
                      9: {0: "Hopper"}, 10: {0: "Blackwell"}, 12: {0: "Blackwell"}}


def gpu_arch(torch):
    major, minor = torch.cuda.get_device_capability(0)
    arch = ARCH_BY_CAPABILITY.get(major, {}).get(minor)
    if arch is None:
        raise RuntimeError(f"unknown GPU capability {major}.{minor}")
    return arch


def marble_timing(kb_eval, timing, ref_src, kernel_src, backend, dtype, device, build_dir, rounds, per_round):
    """Interleaved, equal-warm-up timing of reference vs candidate; returns median-based stats (ms).

    Inputs always come from the *reference* get_inputs(); the candidate file cannot change them.
    Integer tensors keep their dtype; floating tensors are cast to the eval precision.
    """
    import statistics

    import torch

    def prep(x):
        if not isinstance(x, torch.Tensor):
            return x
        x = x.to(dtype=dtype) if x.is_floating_point() else x
        return x.to(device=device)

    context = {}
    Model, get_init_inputs, get_inputs = kb_eval.load_original_model_and_inputs(ref_src, context)
    kb_eval.set_seed(42)
    init = [prep(x) for x in get_init_inputs()]
    if backend.lower() in ("triton", "tilelang", "cute"):
        ModelNew, _tmp = kb_eval.load_custom_model_with_tempfile(kernel_src, entry_point="ModelNew")
    else:
        ModelNew = kb_eval.load_custom_model(kernel_src, {}, build_dir)
    kb_eval.set_seed(42)
    ref_model = Model(*init).to(device=device, dtype=dtype)
    kb_eval.set_seed(42)
    new_model = ModelNew(*init).to(device=device, dtype=dtype)
    kb_eval.set_seed(42)
    inputs = [prep(x) for x in get_inputs()]
    tfn = timing.get_timing_function("cuda_event")
    ref_t, new_t = [], []
    with torch.no_grad():
        for m in (ref_model, new_model):
            tfn(m, inputs, num_warmup=WARMUP, num_trials=2, verbose=False, device=device)
        for r in range(rounds):
            order = [(ref_model, ref_t), (new_model, new_t)]
            if r % 2:
                order.reverse()
            for m, acc in order:
                acc.extend(tfn(m, inputs, num_warmup=2, num_trials=per_round, verbose=False, device=device))
    med_ref, med_new = statistics.median(ref_t), statistics.median(new_t)
    return {
        "runtime_ms": med_new, "ref_ms": med_ref, "speedup": med_ref / med_new if med_new > 0 else None,
        "runtime_min_ms": min(new_t), "ref_min_ms": min(ref_t),
        "runtime_iqr_ms": statistics.quantiles(new_t, n=4)[2] - statistics.quantiles(new_t, n=4)[0],
        "trials_each": len(new_t),
    }


def write(out, res):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "score.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in res.items() if k not in ("metadata",)}, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--mode", choices=MODES, default="quick")
    ap.add_argument("--backend", default="cuda")
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-cache", help="shared build-cache root; builds are keyed by source hash + GPU arch")
    ap.add_argument("--no-static-check", action="store_true",
                    help="calibration of the pure-PyTorch baseline only (MARBLE 'reference' mode)")
    a = ap.parse_args()
    cfg = MODES[a.mode]
    res = {"correct": False, "compiled": False, "speedup": None, "mode": a.mode, "backend": a.backend,
           "precision": a.precision}
    t0 = time.time()

    ref_src = open(a.ref).read()
    kernel_src = open(a.kernel).read()

    import torch
    from kernelbench import eval as kb_eval
    from kernelbench import utils as kb_utils
    from kernelbench.kernel_static_checker import validate_kernel_static
    from kernelbench import timing
    from kernelbench.timing import measure_ref_program_time

    ok, errors, warnings = validate_kernel_static(kernel_src, backend=a.backend, precision=a.precision)
    res["static_ok"], res["static_errors"], res["static_warnings"] = bool(ok), list(errors), list(warnings)
    res["static_check_skipped"] = a.no_static_check
    if not ok and not a.no_static_check:
        res["error"] = "static check failed"
        write(a.out, res)
        return 0

    arch = gpu_arch(torch)
    kb_utils.set_gpu_arch([arch])
    device = torch.device("cuda:0")
    res.update(gpu=torch.cuda.get_device_name(0), gpu_arch=arch)
    if a.build_cache:
        import hashlib
        key = hashlib.sha256(f"{arch}|{a.backend}|{a.precision}|{torch.__version__}\n{kernel_src}".encode()).hexdigest()[:20]
        build_dir = os.path.join(a.build_cache, key)
        res["build_cache_hit"] = os.path.isdir(build_dir)
    else:
        build_dir = os.path.join(a.out, "build")
    os.makedirs(build_dir, exist_ok=True)
    try:
        r = kb_eval.eval_kernel_against_ref(
            original_model_src=ref_src, custom_model_src=kernel_src, measure_performance=True,
            timing_method="cuda_event", num_correct_trials=cfg["correct_trials"],
            num_perf_trials=cfg["perf_trials"], build_dir=build_dir, device=device,
            backend=a.backend, precision=kb_eval.get_torch_dtype_from_string(a.precision),
        )
    except Exception as e:  # compile/runtime errors in candidate code
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-4000:]
        write(a.out, res)
        return 0
    if r is None:
        res["error"] = "KernelBench returned no result (build lock or internal error)"
        write(a.out, res)
        return 0

    res["compiled"], res["metadata"] = bool(r.compiled), r.metadata
    res["correct"] = bool(r.compiled and r.correctness)
    res["excessive_speedup"] = bool(r.metadata.get("excessive_speedup", False))
    if res["correct"] and r.runtime > 0 and r.ref_runtime > 0:
        # KernelBench's own (mean-based, fixed-order) numbers, kept for comparability with published results
        res.update(kb_runtime=r.runtime, kb_runtime_std=r.runtime_stats.get("std"), kb_ref_eager=r.ref_runtime,
                   kb_speedup=r.ref_runtime / r.runtime)
        try:
            mt = marble_timing(kb_eval, timing, ref_src, kernel_src, a.backend,
                               kb_eval.get_torch_dtype_from_string(a.precision), device,
                               build_dir, cfg["rounds"], cfg["per_round"])
        except Exception as e:
            res["correct"] = False
            res["error"] = f"MARBLE timing failed: {type(e).__name__}: {e}"
            res["traceback"] = traceback.format_exc()[-4000:]
            write(a.out, res)
            return 0
        res.update(runtime=mt["runtime_ms"], ref_eager=mt["ref_ms"], speedup=mt["speedup"],
                   runtime_min=mt["runtime_min_ms"], ref_min=mt["ref_min_ms"],
                   runtime_iqr=mt["runtime_iqr_ms"], timing_trials=mt["trials_each"],
                   timing="marble: interleaved rounds, equal warm-up, median")
        if cfg["compile_ref"]:
            comp = measure_ref_program_time(
                ref_arch_name="ref", ref_arch_src=ref_src, num_trials=cfg["perf_trials"],
                use_torch_compile=True, timing_method="cuda_event", device=device,
                precision=a.precision,
            )
            res["ref_compile"] = comp["mean"]
            res["speedup_vs_compile"] = comp["mean"] / res["runtime"]
    elif not res["correct"]:
        res["error"] = "compile failed" if not r.compiled else "output mismatch vs reference"
    res["eval_seconds"] = time.time() - t0
    write(a.out, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
