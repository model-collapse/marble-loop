"""Resource planning (DESIGN.md §4): probe hardware, calibrate, estimate, write resources.yaml.

`marble run plan <benchmark>` produces:
  resource_plan.md   human-readable plan (what will run where, how long, how much)
  resources.yaml     machine-readable plan; edit it (or tell Claude how) and pass it to
                     `marble run start --resources resources.yaml`
resources.yaml (schema v1):
  schema_version: 1
  budget: {usd: 20, wall_hours: 2}
  models: {planner: ..., implementer: ..., ...}          # optional overrides
  slots:                                                  # where evals run (launcher: local only in v1)
    - {name: gpu0, launcher: local, gpus: [0], cpus: 24, cpu_ids: []}
  run: {max_iterations: 20, repeats: 3, ...}              # optional RunConfig overrides
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
import yaml

from .launcher import Slot
from .runstate import RunConfig
from .spec import BenchmarkSpec, SpecError

SCHEMA_VERSION = 1
DEFAULT_ITER_COST_USD = 0.18      # median agent cost per iteration over the M1 live runs
DEFAULT_AGENT_S = 120.0           # analyst + planner + implementer + reflector wall time (M1 median-ish)
DEBUG_PROB = 0.3                  # share of candidates that need a debug attempt
RUN_KEYS = {"max_iterations", "max_no_improve", "repeats", "debug_attempts", "eval_mode", "analyze", "confirm_mode",
            "confirm_repeats", "search", "c_uct", "max_children", "stagnation_n", "time_penalty_w",
            "lesson_test_every", "inspector", "stall_s", "max_smoke_s"}


# --------------------------------------------------------------------------- probe


def probe() -> dict[str, Any]:
    info: dict[str, Any] = {"cpus": os.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
                            "disk_free_gb": round(shutil.disk_usage(".").free / 2**30, 1), "gpus": []}
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,utilization.gpu",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
            for line in out.strip().splitlines():
                i, name, mem, util = [x.strip() for x in line.split(",")]
                info["gpus"].append({"index": int(i), "name": name, "memory_gb": round(int(mem) / 1024, 1),
                                     "util_pct": int(util)})
        except (subprocess.SubprocessError, ValueError):
            pass
    info["claude_cli"] = shutil.which("claude") is not None
    return info


# --------------------------------------------------------------------------- estimates


def past_iteration_costs(runs_root: Path, benchmark_name: str) -> list[float]:
    """Agent $ per finished iteration in earlier runs of the same benchmark (for better estimates)."""
    return [c for c, _ in past_iterations(runs_root, benchmark_name)]


def past_iterations(runs_root: Path, benchmark_name: str) -> list[tuple[float, float]]:
    """(agent $, wall seconds) per finished iteration, one entry per earlier run of the same benchmark."""
    out = []
    for cfg_path in Path(runs_root).glob("*/config.json"):
        try:
            cfg = json.loads(cfg_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if Path(cfg.get("benchmark", "")).name != benchmark_name:
            continue
        ev = cfg_path.parent / "events.jsonl"
        if not ev.exists():
            continue
        usd, nodes, created, verdict_t = 0.0, set(), {}, {}
        for line in ev.read_text().splitlines():
            e = json.loads(line)
            if e["type"] == "cost":
                usd += e.get("usd", 0.0)
            elif e["type"] == "node_created":
                created[e["node"]] = e["t"]
            elif e["type"] == "verdict":
                nodes.add(e["node"])
                verdict_t[e["node"]] = e["t"]
        if nodes:
            secs = [verdict_t[n] - created[n] for n in nodes if n in created]
            out.append((usd / len(nodes), statistics.median(secs) if secs else 0.0))
    return out


@dataclass
class Estimate:
    eval_s: float
    iteration_s: float
    iteration_usd: float
    iterations_by_time: int
    iterations_by_budget: int

    @property
    def iterations(self) -> int:
        return max(0, min(self.iterations_by_time, self.iterations_by_budget))


def estimate(spec: BenchmarkSpec, *, eval_s: float, repeats: int, hours: float, budget_usd: float,
             analyze_s: float = 0.0, iter_usd: float | None = None, confirm_s: float = 0.0,
             iter_s: float | None = None) -> Estimate:
    candidate_eval = eval_s * repeats * (1 + DEBUG_PROB)
    model_s = candidate_eval + DEFAULT_AGENT_S + analyze_s * 0.5 + confirm_s * 0.3   # analyses are cached per node
    it_s = max(model_s, iter_s) if iter_s else model_s   # earlier runs include compile/agent overheads
    it_usd = iter_usd if iter_usd is not None else DEFAULT_ITER_COST_USD
    return Estimate(eval_s=eval_s, iteration_s=it_s, iteration_usd=it_usd,
                    iterations_by_time=int(hours * 3600 / it_s + 1e-9) if it_s > 0 else 0,
                    iterations_by_budget=int(budget_usd / it_usd + 1e-9) if it_usd > 0 else 0)


# --------------------------------------------------------------------------- plan


def _g(x) -> str:
    return "-" if x is None else f"{x:.4g}"


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.2f}%"


def make_plan(spec: BenchmarkSpec, *, hw: dict, calibration: dict, hours: float, budget_usd: float,
              gpus: list[int] | None = None, past_costs: list[float] | None = None,
              analyze_s: float = 0.0, past_seconds: list[float] | None = None) -> dict[str, Any]:
    mode_name = next((m for m in calibration["modes"]), spec.default_mode)
    cal = calibration["modes"][mode_name]
    eval_s = float(cal["duration_mean_s"] or 0.0)
    repeats = spec.noise.repeats
    needs_gpu = spec.mode(None).resources.gpu > 0
    hw_gpus = [g["index"] for g in hw.get("gpus", [])]
    use_gpus = [g for g in (gpus if gpus is not None else hw_gpus) if g in hw_gpus]
    if needs_gpu and not use_gpus:
        raise SpecError("benchmark needs a GPU but none is available/selected")
    ncpu = hw.get("cpus") or 1
    if needs_gpu:
        g = use_gpus[0]
        per = max(1, ncpu // max(1, len(use_gpus)))
        slots = [{"name": f"gpu{g}", "launcher": "local", "gpus": [g], "cpus": per, "cpu_ids": []}]
    else:
        per = min(ncpu, max(spec.mode(None).resources.cpu, 8))
        slots = [{"name": "cpu0", "launcher": "local", "gpus": [], "cpus": per, "cpu_ids": list(range(per))}]
    iter_usd = statistics.median(past_costs) if past_costs else None
    past_s = [x for x in (past_seconds or []) if x > 0]
    est = estimate(spec, eval_s=eval_s, repeats=repeats, hours=hours, budget_usd=budget_usd,
                   analyze_s=analyze_s, iter_usd=iter_usd, iter_s=statistics.median(past_s) if past_s else None)
    sug = cal.get("suggestions", {})
    timeout_ok = sug.get("suggested_timeout_s") is None or spec.mode(None).timeout >= sug["suggested_timeout_s"] / 3
    rc = RunConfig(benchmark=str(spec.bench_dir))
    resources = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": str(spec.bench_dir),
        "budget": {"usd": budget_usd, "wall_hours": hours},
        "models": dict(rc.models),
        "slots": slots,
        "run": {"max_iterations": max(1, est.iterations), "repeats": repeats},
    }
    idle_gpus = [g for g in use_gpus if g not in sum((s["gpus"] for s in slots), [])]
    notes = []
    if idle_gpus:
        notes.append(f"GPUs {idle_gpus} stay free for other work (one search tree uses one eval slot); "
                     "run another benchmark on them in parallel if you like.")
    if not timeout_ok:
        notes.append(f"eval timeout {spec.mode(None).timeout:.0f}s looks tight vs calibration; consider raising it")
    if est.iterations_by_budget < est.iterations_by_time:
        notes.append("the $ budget, not the wall clock, limits the number of iterations")
    elif est.iterations_by_time < est.iterations_by_budget:
        notes.append("the wall clock, not the $ budget, limits the number of iterations")
    gpu_desc = ", ".join(f"{g['index']}: {g['name']} {g['memory_gb']} GB" for g in hw.get("gpus", [])) or "none"
    md = [
        f"# MARBLE resource plan: {spec.name}", "",
        "## Hardware", f"- CPUs: {hw.get('cpus')}, RAM: {hw.get('ram_gb')} GB, free disk: {hw.get('disk_free_gb')} GB",
        f"- GPUs: {gpu_desc}", f"- Claude CLI available: {hw.get('claude_cli')}", "",
        "## Calibration (baseline)",
        f"- mode `{mode_name}`: {cal['status']}, {spec.score.key} {_g(cal.get('score_mean'))} ± {_g(cal.get('score_std'))} "
        f"(relative noise {_pct(cal.get('score_rel_std'))}), {eval_s:.1f} s per eval",
        f"- analyze tools: {analyze_s:.1f} s per analysis" if spec.analyze else "- analyze tools: none", "",
        "## Plan",
        f"- eval slot(s): {', '.join(s['name'] for s in slots)}; repeats per candidate: {repeats}",
        f"- one iteration ≈ {est.iteration_s / 60:.1f} min and ≈ ${est.iteration_usd:.2f} "
        f"({'from ' + str(len(past_costs or [])) + ' earlier run(s)' if iter_usd is not None else 'default estimate'})",
        f"- budget: {hours:g} h and ${budget_usd:g} → about **{est.iterations} iterations** "
        f"(time allows {est.iterations_by_time}, money allows {est.iterations_by_budget})",
        f"- models: " + ", ".join(f"{k}={v.split('.')[-1]}" for k, v in rc.models.items()), "",
    ]
    if notes:
        md += ["## Notes"] + [f"- {n}" for n in notes] + [""]
    md += ["## How to use", "Edit `resources.yaml` (or tell Claude how the resources may be used), then run:",
           "```", f"marble run start {spec.bench_dir} --resources resources.yaml --supervise", "```"]
    return {"resources": resources, "markdown": "\n".join(md) + "\n", "estimate": est.__dict__ | {"iterations": est.iterations}}


# --------------------------------------------------------------------------- load


def load_resources(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise SpecError(f"{path}: expected resources.yaml with schema_version {SCHEMA_VERSION}")
    slots = data.get("slots") or []
    if not slots:
        raise SpecError(f"{path}: at least one slot is required")
    for s in slots:
        if s.get("launcher", "local") != "local":
            raise SpecError(f"{path}: slot {s.get('name')}: only launcher 'local' is supported in v1")
        if not isinstance(s.get("name"), str):
            raise SpecError(f"{path}: every slot needs a name")
    unknown = set(data.get("run") or {}) - RUN_KEYS
    if unknown:
        raise SpecError(f"{path}: unknown run setting(s) {sorted(unknown)}")
    return data


def slots_from(data: dict) -> list[Slot]:
    return [Slot(name=s["name"], gpus=tuple(s.get("gpus") or ()), cpus=int(s.get("cpus") or 1),
                 cpu_ids=tuple(s.get("cpu_ids") or ())) for s in data["slots"]]


def apply_to_config(data: dict, cfg: RunConfig) -> RunConfig:
    b = data.get("budget") or {}
    if "usd" in b:
        cfg.budget_usd = float(b["usd"])
    if "wall_hours" in b:
        cfg.wall_hours = float(b["wall_hours"])
    cfg.models.update(data.get("models") or {})
    for k, v in (data.get("run") or {}).items():
        setattr(cfg, k, v)
    return cfg
