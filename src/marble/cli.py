"""Command-line interface: `marble bench validate|calibrate|eval`."""

from __future__ import annotations

import argparse
import json
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
