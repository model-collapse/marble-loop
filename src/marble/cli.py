"""Command-line interface: `marble bench validate|calibrate|eval`."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from . import __version__
from .calibrate import calibrate
from .cleanroom import IntegrityError, check_candidate, list_repo_files, match_any, protected_manifest
from .evaluate import run_repeats
from .launcher import LocalLauncher, default_slots
from .spec import SpecError, load_spec, validate_spec


def _launcher(args) -> LocalLauncher:
    gpus = None
    if args.gpus is not None:
        gpus = [] if args.gpus in ("", "none") else [int(g) for g in args.gpus.split(",")]
    slots = default_slots(gpus=gpus, cpu_slots=args.cpu_slots, cpus_per_slot=args.cpus_per_slot, pin=args.pin)
    return LocalLauncher(slots)


def _default_out(spec_name: str, kind: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("runs") / f"{spec_name}-{kind}-{stamp}"


def _print(obj, as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, indent=2, default=str))


def cmd_validate(args) -> int:
    spec = load_spec(args.benchmark)
    warnings = validate_spec(spec)
    files = list_repo_files(spec.repo)
    manifest = protected_manifest(spec)
    editable = [f for f in files if match_any(f, spec.editable)]
    info = {
        "name": spec.name, "repo": str(spec.repo), "files": len(files), "editable": editable,
        "protected": sorted(manifest), "modes": {m: c.timeout for m, c in spec.evaluate.items()},
        "default_mode": spec.default_mode, "score": f"{spec.score.key} ({spec.score.direction})",
        "warnings": warnings,
    }
    if args.candidate:
        try:
            d = check_candidate(spec, Path(args.candidate))
            info["candidate_changes"] = d.changed
        except IntegrityError as e:
            info["candidate_error"] = str(e)
    if args.json:
        _print(info, True)
    else:
        print(f"✓ {spec.name}: valid (schema v{spec.schema_version})")
        print(f"  repo:      {spec.repo}  ({len(files)} files)")
        print(f"  editable:  {len(editable)} file(s): {', '.join(editable[:8])}{' …' if len(editable) > 8 else ''}")
        print(f"  protected: {len(manifest)} file(s), SHA-256 manifest computed")
        print(f"  modes:     " + ", ".join(f"{m} (timeout {c.timeout:.0f}s{', default' if m == spec.default_mode else ''})"
                                         for m, c in spec.evaluate.items()))
        print(f"  score:     {spec.score.key} → {spec.score.direction}"
              + (f"; gate {spec.correctness.key} == {spec.correctness.equals!r}" if spec.correctness else ""))
        for w in warnings:
            print(f"  ! {w}")
        if "candidate_changes" in info:
            print(f"  candidate: OK, changes {info['candidate_changes']}")
        if "candidate_error" in info:
            print(f"  candidate: REJECTED: {info['candidate_error']}")
    return 1 if "candidate_error" in info else 0


def cmd_eval(args) -> int:
    spec = load_spec(args.benchmark)
    validate_spec(spec)
    launcher = _launcher(args)
    out = Path(args.out) if args.out else _default_out(spec.name, "eval")
    repeats = args.repeats or spec.noise.repeats

    def progress(i, r):
        if not args.json:
            s = f"{r.score:.6g}" if r.score is not None else "-"
            print(f"  [{i + 1}/{repeats}] {r.status:<15} score={s:<12} {r.duration_s:7.1f}s"
                  + (f"  {r.error}" if r.error else ""), flush=True)

    if not args.json:
        print(f"Evaluating {spec.name} [{args.mode or spec.default_mode}] x{repeats}"
              + (f" candidate={args.candidate}" if args.candidate else " (baseline)") + f" → {out}")
    summary = run_repeats(spec, launcher=launcher, out_dir=out, repeats=repeats, mode=args.mode,
                          candidate=Path(args.candidate) if args.candidate else None,
                          keep_workdir=args.keep_workdir, on_result=progress)
    stats = summary.stats()
    (out / "summary.json").write_text(json.dumps(stats, indent=2))
    if args.json:
        _print(stats, True)
    else:
        if stats["n_ok"]:
            std = f" ± {stats['score_std']:.4g}" if stats["n_ok"] > 1 else ""
            print(f"{stats['status']}: {spec.score.key} = {stats['score_mean']:.6g}{std} "
                  f"({stats['n_ok']}/{stats['n']} ok, mean {stats['duration_mean_s']:.1f}s)")
        else:
            print(f"{stats['status']}: no successful run; see {out}")
    return 0 if stats["status"] == "ok" else 2


def cmd_calibrate(args) -> int:
    spec = load_spec(args.benchmark)
    validate_spec(spec)
    launcher = _launcher(args)
    out = Path(args.out) if args.out else _default_out(spec.name, "calibrate")
    calib_modes = [m for m, c in spec.evaluate.items() if c.calibration_only]
    modes = args.mode or calib_modes or [spec.default_mode]

    def progress(m, i, r):
        if not args.json:
            s = f"{r.score:.6g}" if r.score is not None else "-"
            print(f"  {m} [{i + 1}/{args.repeats}] {r.status:<15} score={s:<12} {r.duration_s:7.1f}s"
                  + (f"  {r.error}" if r.error else ""), flush=True)

    if not args.json:
        print(f"Calibrating {spec.name}: modes={modes} repeats={args.repeats} → {out}")
    t0 = time.monotonic()
    report = calibrate(spec, launcher=launcher, out_dir=out, modes=modes, repeats=args.repeats, on_result=progress)
    if args.json:
        _print(report, True)
    else:
        for m, st in report["modes"].items():
            sug = st["suggestions"]
            score = (f"{st['score_mean']:.6g} ± {st['score_std']:.3g} (rel {st['score_rel_std']:.2%})"
                     if st["score_mean"] is not None and st["score_rel_std"] is not None
                     else f"{st['score_mean']}")
            print(f"{m}: {st['status']}  score {score}  duration {st['duration_mean_s']:.1f}s "
                  f"(max {st['duration_max_s']:.1f}s)")
            if sug:
                print(f"   expected {sug.get('expected_duration_s')}s, soft limit {sug.get('soft_limit_s')}s, "
                      f"suggested timeout {sug.get('suggested_timeout_s')}s, "
                      f"min_rel_improvement {sug.get('suggested_min_rel_improvement')}")
                if "warning" in sug:
                    print(f"   ! {sug['warning']}")
        print(f"wrote {out / 'calibration.json'} ({time.monotonic() - t0:.0f}s)")
    return 0 if report["status"] == "ok" else 2


def cmd_run_start(args) -> int:
    from .runstate import RunConfig
    from .search import Controller

    out = Path(args.out) if args.out else None
    if out is None:
        spec = load_spec(args.benchmark)
        out = _default_out(spec.name, "run")
    cfg = RunConfig(benchmark=str(Path(args.benchmark).resolve()), budget_usd=args.budget_usd,
                    wall_hours=args.hours, max_iterations=args.max_iters, max_no_improve=args.max_no_improve,
                    repeats=args.repeats, debug_attempts=args.debug_attempts, eval_mode=args.mode,
                    analyze=not args.no_analyze)
    for kv in args.model or []:
        role, _, model = kv.partition("=")
        if role not in cfg.models or not model:
            raise SpecError(f"--model expects role=model-id with role in {sorted(cfg.models)}")
        cfg.models[role] = model
    cfg.inspector = args.inspector
    launcher = None
    if args.resources:
        from . import resources as R

        data = R.load_resources(Path(args.resources))
        R.apply_to_config(data, cfg)
        launcher = LocalLauncher(R.slots_from(data))
    if (out / "tree.sqlite").exists():
        print(f"✗ {out} already contains a run; use `marble run resume {out}`", file=sys.stderr)
        return 1
    ctl = Controller(out, launcher or _launcher(args), cfg)
    if args.supervise:
        return _supervise(out, args)
    best = ctl.run()
    print(f"\nbest: node #{best.id} {ctl.spec.score.key}={best.score:.6g}; report: {out / 'report.md'}")
    return 0


def cmd_run_plan(args) -> int:
    import yaml

    from . import resources as R
    from .evaluate import run_tool

    spec = load_spec(args.benchmark)
    validate_spec(spec)
    out = Path(args.out) if args.out else _default_out(spec.name, "plan")
    out.mkdir(parents=True, exist_ok=True)
    hw = R.probe()
    launcher = _launcher(args)
    calib_modes = [m for m, c in spec.evaluate.items() if c.calibration_only] or [spec.default_mode]
    print(f"Planning {spec.name}: probing hardware, calibrating baseline ({calib_modes[0]}, x{args.repeats}) …", flush=True)
    cal = calibrate(spec, launcher=launcher, out_dir=out / "calibration", modes=calib_modes[:1], repeats=args.repeats)
    if cal["status"] != "ok":
        print(f"✗ baseline calibration failed: {cal['modes']}", file=sys.stderr)
        return 2
    analyze_s = 0.0
    for tool in spec.analyze if not args.no_analyze else []:
        r = run_tool(spec, tool, launcher=launcher, out_dir=out / "analyze" / tool.name)
        print(f"  analyze tool {tool.name}: {r['status']} {r.get('duration_s', 0):.0f}s", flush=True)
        analyze_s += r.get("duration_s", 0.0) or 0.0
    gpus = None if args.gpus is None else ([] if args.gpus in ("", "none") else [int(g) for g in args.gpus.split(",")])
    past = R.past_iterations(Path(args.runs_root), spec.bench_dir.name)
    plan = R.make_plan(spec, hw=hw, calibration=cal, hours=args.hours, budget_usd=args.budget_usd, gpus=gpus,
                       past_costs=[c for c, _ in past], past_seconds=[t for _, t in past], analyze_s=analyze_s)
    (out / "resources.yaml").write_text(yaml.safe_dump(plan["resources"], sort_keys=False))
    (out / "resource_plan.md").write_text(plan["markdown"])
    (out / "hardware.json").write_text(json.dumps(hw, indent=2))
    print(plan["markdown"])
    print(f"wrote {out / 'resources.yaml'} and {out / 'resource_plan.md'}")
    return 0


def cmd_run_report(args) -> int:
    from . import report as rep

    p = rep.write_report(Path(args.run), stop=None)
    print(p.read_text() if not args.quiet else str(p))
    return 0


def cmd_run_control(args) -> int:
    from . import report as rep

    action = "stop" if args.stop else "pause" if args.pause else "run" if args.resume else None
    data = rep.write_control(Path(args.run), action=action, note=args.note, clear_notes=args.clear_notes)
    print(json.dumps(data, indent=2))
    return 0


def cmd_run_resume(args) -> int:
    from . import report as rep
    from .search import Controller

    if rep.read_control(Path(args.run)).get("action") in ("stop", "pause"):
        rep.write_control(Path(args.run), action="run")

    from .runstate import RunState

    st = RunState(Path(args.run))
    cfg = st.load_config()
    for attr, val in (("budget_usd", args.budget_usd), ("max_iterations", args.max_iters), ("wall_hours", args.hours),
                      ("max_no_improve", args.max_no_improve)):
        if val is not None:
            setattr(cfg, attr, val)
    ctl = Controller(Path(args.run), _launcher(args), cfg)
    best = ctl.run()
    print(f"\nbest: node #{best.id} {ctl.spec.score.key}={best.score:.6g}; report: {Path(args.run) / 'report.md'}")
    return 0


def _slot_args(args) -> list[str]:
    a = []
    if args.gpus is not None:
        a += ["--gpus", args.gpus]
    a += ["--cpu-slots", str(args.cpu_slots)]
    if args.cpus_per_slot:
        a += ["--cpus-per-slot", str(args.cpus_per_slot)]
    if args.pin:
        a += ["--pin"]
    return a


def _supervise(run_dir: Path, args) -> int:
    from .supervisor import supervise

    return supervise(run_dir, _slot_args(args), max_restarts=args.max_restarts,
                     heartbeat_stale_s=args.heartbeat_stale_s, progress_stale_s=args.progress_stale_s)


def cmd_run_supervise(args) -> int:
    return _supervise(Path(args.run), args)


def cmd_run_status(args) -> int:
    from .runstate import RunState

    st = RunState(Path(args.run))
    cfg = st.load_config()
    nodes = st.nodes()
    best = st.best()
    print(f"run {Path(args.run).resolve()}  benchmark {cfg.benchmark}")
    from .inspector import IncidentLog

    incs = IncidentLog(Path(args.run)).all()
    print(f"cost ${st.total_cost():.2f} / ${cfg.budget_usd:.2f}   nodes {len(nodes)}   lessons {len(st.lessons())}"
          f"   incidents {len(incs)}")
    if best:
        print(f"best: node #{best.id} score {best.score:.6g}")
    for n in nodes[-args.last:]:
        s = f"{n.score:.6g}" if n.score is not None else "-"
        print(f"  #{n.id:<3} parent {str(n.parent):<4} {n.kind:<8} {n.status:<10} {n.eval_status or '-':<14} "
              f"score {s:<12} {'★' if n.is_best else ('+' if n.improved else '')}  ${n.cost_usd:.2f}")
    stopped = [e for e in st.events() if e["type"] == "run_stopped"]
    if stopped:
        print(f"last stop: {stopped[-1].get('reason')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="marble", description="MARBLE: Monitored Analyze-Refine Budgeted Loop Engine")
    p.add_argument("--version", action="version", version=f"marble {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    bench = sub.add_parser("bench", help="benchmark adapter tools").add_subparsers(dest="bench_cmd", required=True)

    def common(sp, launch=True):
        sp.add_argument("benchmark", help="path to benchmark.yaml (or its directory)")
        sp.add_argument("--json", action="store_true", help="machine-readable output")
        if launch:
            sp.add_argument("--out", help="output directory (default: runs/<name>-<kind>-<time>)")
            sp.add_argument("--gpus", help="comma-separated GPU ids for slots, or 'none' (default: all detected)")
            sp.add_argument("--cpu-slots", type=int, default=1, help="number of CPU-only slots (default 1)")
            sp.add_argument("--cpus-per-slot", type=int, help="CPUs per slot (default: even split)")
            sp.add_argument("--pin", action="store_true", help="pin each slot to its own CPU range")

    v = bench.add_parser("validate", help="check benchmark.yaml against the repo")
    common(v, launch=False)
    v.add_argument("--candidate", help="also check that a candidate dir only changes editable files")
    v.set_defaults(func=cmd_validate)

    e = bench.add_parser("eval", help="evaluate the baseline or a candidate")
    common(e)
    e.add_argument("--mode", help="eval mode (default: the spec's default_mode)")
    e.add_argument("--candidate", help="directory with a modified copy of the repo")
    e.add_argument("--repeats", type=int, help="number of runs (default: noise.repeats)")
    e.add_argument("--keep-workdir", action="store_true", help="keep clean-room copies for debugging")
    e.set_defaults(func=cmd_eval)

    c = bench.add_parser("calibrate", help="measure baseline duration and score noise")
    common(c)
    c.add_argument("--mode", action="append", help="mode(s) to calibrate (repeatable; default: default_mode)")
    c.add_argument("--repeats", type=int, default=3)
    c.set_defaults(func=cmd_calibrate)

    mcp = sub.add_parser("mcp", help="run the MARBLE MCP server on stdio (used by the Claude Code plugin)")
    mcp.add_argument("--runs-root", default=os.environ.get("MARBLE_RUNS_ROOT", "runs"))
    mcp.set_defaults(func=lambda a: __import__("marble.mcp_server", fromlist=["serve"]).serve(Path(a.runs_root)))

    hk = sub.add_parser("hook", help="Claude Code hook handlers (used by the plugin)")
    hk.add_argument("name", choices=["session-start", "guard-runs", "guard"])
    hk.set_defaults(func=lambda a: __import__("marble.hooks", fromlist=["main"]).main([a.name]))

    run = sub.add_parser("run", help="automated research runs").add_subparsers(dest="run_cmd", required=True)

    def slots(sp):
        sp.add_argument("--gpus", help="comma-separated GPU ids for slots, or 'none' (default: all detected)")
        sp.add_argument("--cpu-slots", type=int, default=1)
        sp.add_argument("--cpus-per-slot", type=int)
        sp.add_argument("--pin", action="store_true")

    def sup_opts(sp):
        sp.add_argument("--max-restarts", type=int, default=3)
        sp.add_argument("--heartbeat-stale-s", type=float, default=120.0)
        sp.add_argument("--progress-stale-s", type=float, default=3 * 3600.0)

    rs = run.add_parser("start", help="start a greedy analyze-refine run")
    rs.add_argument("benchmark")
    rs.add_argument("--out", help="run directory (default: runs/<name>-run-<time>)")
    rs.add_argument("--budget-usd", type=float, default=20.0)
    rs.add_argument("--hours", type=float, default=2.0)
    rs.add_argument("--max-iters", type=int, default=20)
    rs.add_argument("--max-no-improve", type=int, default=6)
    rs.add_argument("--repeats", type=int, help="eval repeats per candidate (default: noise.repeats)")
    rs.add_argument("--debug-attempts", type=int, default=3)
    rs.add_argument("--mode", help="eval mode for candidates (default: the spec's default_mode)")
    rs.add_argument("--no-analyze", action="store_true", help="skip the benchmark's analyze tools")
    rs.add_argument("--model", action="append", help="role=model-id override (repeatable)")
    rs.add_argument("--inspector", choices=["llm", "rule", "off"], default="llm", help="job watchdog investigations")
    rs.add_argument("--supervise", action="store_true", help="run under the supervisor (restarts on crash/hang)")
    rs.add_argument("--resources", help="resources.yaml from `marble run plan` (slots, budget, models, run settings)")
    sup_opts(rs)
    slots(rs)
    rs.set_defaults(func=cmd_run_start)

    rp = run.add_parser("plan", help="probe hardware, calibrate, and propose resources.yaml + resource_plan.md")
    rp.add_argument("benchmark")
    rp.add_argument("--out", help="plan directory (default: runs/<name>-plan-<time>)")
    rp.add_argument("--hours", type=float, default=2.0)
    rp.add_argument("--budget-usd", type=float, default=20.0)
    rp.add_argument("--repeats", type=int, default=2, help="baseline calibration repeats")
    rp.add_argument("--no-analyze", action="store_true", help="do not time the analyze tools")
    rp.add_argument("--runs-root", default="runs", help="earlier runs used for cost estimates")
    slots(rp)
    rp.set_defaults(func=cmd_run_plan)

    rr = run.add_parser("resume", help="continue a stopped run (same config)")
    rr.add_argument("run")
    rr.add_argument("--budget-usd", type=float, help="new total budget (default: unchanged)")
    rr.add_argument("--max-iters", type=int, help="new total iteration limit")
    rr.add_argument("--hours", type=float, help="wall-clock limit for this session")
    rr.add_argument("--max-no-improve", type=int)
    slots(rr)
    rr.set_defaults(func=cmd_run_resume)

    sv = run.add_parser("supervise", help="(re)start a run's controller and restart it on crash/hang")
    sv.add_argument("run")
    sup_opts(sv)
    slots(sv)
    sv.set_defaults(func=cmd_run_supervise)

    rpt = run.add_parser("report", help="(re)build report.md from the run's state")
    rpt.add_argument("run")
    rpt.add_argument("--quiet", action="store_true", help="only print the path")
    rpt.set_defaults(func=cmd_run_report)

    rc = run.add_parser("control", help="stop / pause / resume a running controller, or pin a note for the planner")
    rc.add_argument("run")
    g = rc.add_mutually_exclusive_group()
    g.add_argument("--stop", action="store_true", help="stop after the current iteration")
    g.add_argument("--pause", action="store_true")
    g.add_argument("--resume", action="store_true")
    rc.add_argument("--note", help="hint added to every planner prompt")
    rc.add_argument("--clear-notes", action="store_true")
    rc.set_defaults(func=cmd_run_control)

    rt = run.add_parser("status", help="show a run's nodes and spend")
    rt.add_argument("run")
    rt.add_argument("--last", type=int, default=30)
    rt.set_defaults(func=cmd_run_status)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except SpecError as e:
        print(f"✗ invalid benchmark: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
