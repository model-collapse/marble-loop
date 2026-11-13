"""Calibration: run the unmodified baseline k times to measure duration and score noise.

The output feeds the resource planner (DESIGN.md §4) and the Inspector's expected durations (§6.1).
"""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .cleanroom import protected_manifest
from .evaluate import EvalResult, run_repeats
from .launcher import LocalLauncher
from .spec import BenchmarkSpec

SOFT_FACTOR = 1.5      # investigate when a job runs this much longer than expected
TIMEOUT_FACTOR = 3.0   # suggested hard timeout relative to the slowest baseline run
NOISE_SIGMAS = 2.0     # improvements must exceed this many relative std devs


def suggest(stats: dict[str, Any], spec_timeout: float, min_rel_improvement: float) -> dict[str, Any]:
    mean, std, mx = stats["duration_mean_s"], stats["duration_std_s"] or 0.0, stats["duration_max_s"]
    out: dict[str, Any] = {}
    if mean:
        out["expected_duration_s"] = round(mean, 2)
        out["soft_limit_s"] = round(max(SOFT_FACTOR * mean, mean + 3 * std), 2)
        out["suggested_timeout_s"] = round(max(TIMEOUT_FACTOR * mx, mx + 60), 1)
        if spec_timeout < SOFT_FACTOR * mx:
            out["warning"] = (f"timeout {spec_timeout:.0f}s is < {SOFT_FACTOR}x the slowest baseline run "
                              f"({mx:.1f}s); raise it")
    rel = stats.get("score_rel_std")
    if rel is not None:
        out["score_rel_std"] = rel
        out["suggested_min_rel_improvement"] = round(max(min_rel_improvement, NOISE_SIGMAS * rel), 6)
    return out


def calibrate(
    spec: BenchmarkSpec,
    *,
    launcher: LocalLauncher,
    out_dir: Path,
    modes: list[str] | None = None,
    repeats: int = 3,
    on_result: Callable[[str, int, EvalResult], None] | None = None,
) -> dict[str, Any]:
    modes = modes or [spec.default_mode]
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = protected_manifest(spec)
    report: dict[str, Any] = {
        "benchmark": spec.name,
        "marble_version": __version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "slots": [s.name for s in launcher.slots],
        "repeats": repeats,
        "protected_manifest": manifest,
        "modes": {},
    }
    t0 = time.monotonic()
    for m in modes:
        summary = run_repeats(
            spec, launcher=launcher, out_dir=out_dir / m, repeats=repeats, mode=m, manifest=manifest,
            on_result=(lambda i, r, m=m: on_result(m, i, r)) if on_result else None,
        )
        stats = summary.stats()
        stats["errors"] = [r.error for r in summary.results if r.error]
        stats["suggestions"] = suggest(stats, spec.mode(m).timeout, spec.noise.min_rel_improvement)
        report["modes"][m] = stats
    report["total_s"] = round(time.monotonic() - t0, 2)
    report["status"] = "ok" if all(v["status"] == "ok" for v in report["modes"].values()) else "failed"
    (out_dir / "calibration.json").write_text(json.dumps(report, indent=2))
    return report
