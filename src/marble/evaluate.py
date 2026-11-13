"""Run a benchmark's evaluation command in a clean room and read the score."""

from __future__ import annotations

import json
import math
import shutil
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .cleanroom import IntegrityError, prepare_cleanroom, protected_manifest
from .launcher import JobHandle, LocalLauncher, Sample
from .spec import BenchmarkSpec, SpecError, render

STATUSES = ("ok", "incorrect", "failed", "timeout", "cancelled", "invalid_output", "integrity_error")


@dataclass
class EvalResult:
    status: str
    mode: str
    score: float | None = None
    correct: bool | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    error: str | None = None
    out_dir: str = ""
    job: dict | None = None
    changed_files: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalSummary:
    mode: str
    results: list[EvalResult]

    @property
    def scores(self) -> list[float]:
        return [r.score for r in self.results if r.ok and r.score is not None]

    @property
    def status(self) -> str:
        bad = [r.status for r in self.results if not r.ok]
        return "ok" if not bad else bad[0]

    def stats(self) -> dict[str, Any]:
        s = self.scores
        d = [r.duration_s for r in self.results]
        mean = statistics.fmean(s) if s else None
        std = statistics.stdev(s) if len(s) > 1 else 0.0 if s else None
        return {
            "mode": self.mode,
            "status": self.status,
            "n": len(self.results),
            "n_ok": len(s),
            "score_mean": mean,
            "score_std": std,
            "score_rel_std": (std / abs(mean)) if (s and mean) else None,
            "score_min": min(s) if s else None,
            "score_max": max(s) if s else None,
            "duration_mean_s": statistics.fmean(d) if d else None,
            "duration_std_s": statistics.stdev(d) if len(d) > 1 else 0.0,
            "duration_max_s": max(d) if d else None,
            "statuses": [r.status for r in self.results],
        }


def _lookup(data: Any, dotted: str) -> Any:
    cur = data
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(dotted)
    return cur


def _as_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    return None


def run_eval(
    spec: BenchmarkSpec,
    *,
    launcher: LocalLauncher,
    out_dir: Path,
    mode: str | None = None,
    candidate: Path | None = None,
    manifest: dict[str, str] | None = None,
    keep_workdir: bool = False,
    on_sample: Callable[[JobHandle, Sample], None] | None = None,
) -> EvalResult:
    mode_name = mode or spec.default_mode
    cmd_spec = spec.mode(mode_name)
    if cmd_spec.calibration_only and candidate is not None:
        raise SpecError(f"eval mode '{mode_name}' is calibration_only and cannot evaluate a candidate")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_out = out_dir / "out"
    eval_out.mkdir(exist_ok=True)
    manifest = manifest if manifest is not None else protected_manifest(spec)
    t0 = time.monotonic()

    def finish(res: EvalResult) -> EvalResult:
        res.duration_s = res.duration_s or (time.monotonic() - t0)
        res.out_dir = str(out_dir)
        (out_dir / "result.json").write_text(json.dumps(res.to_dict(), indent=2, default=str))
        return res

    workdir = out_dir / "workdir"
    if workdir.exists():
        shutil.rmtree(workdir)
    try:
        room = prepare_cleanroom(spec, candidate=candidate, dest=workdir, manifest=manifest)
    except IntegrityError as e:
        return finish(EvalResult(status="integrity_error", mode=mode_name, error=str(e)))

    try:
        values = {"out": eval_out, "workdir": room.path, "bench_dir": spec.bench_dir}
        cmd = render(cmd_spec.cmd, **values)
        env = {k: render(v, **values) for k, v in spec.env.items()}
        job = launcher.run(
            cmd, cwd=room.path, resources=cmd_spec.resources, timeout=cmd_spec.timeout,
            log_path=out_dir / "eval.log", env=env, on_sample=on_sample,
        )
        res = EvalResult(status="ok", mode=mode_name, duration_s=job.duration_s, job=job.to_dict(),
                         changed_files=room.diff.changed)

        # Candidate code runs during evaluation: re-check that it did not alter protected files.
        after = protected_manifest(spec, room.path)
        if after != manifest:
            changed = sorted(k for k in set(after) | set(manifest) if after.get(k) != manifest.get(k))
            res.status, res.error = "integrity_error", f"protected files modified during eval: {changed[:10]}"
            return finish(res)
        if job.timed_out:
            res.status, res.error = "timeout", f"exceeded timeout of {cmd_spec.timeout:.0f}s"
            return finish(res)
        if job.cancelled:
            res.status, res.error = "cancelled", job.cancel_reason
            return finish(res)
        if job.returncode != 0:
            res.status, res.error = "failed", f"exit code {job.returncode}; see {out_dir / 'eval.log'}"
            return finish(res)

        score_path = Path(render(spec.score.file, **values))
        if not score_path.is_absolute():
            score_path = room.path / score_path
        try:
            data = json.loads(score_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            res.status, res.error = "invalid_output", f"cannot read score file {score_path}: {e}"
            return finish(res)
        res.metrics = {"raw": data}

        for k in spec.score.extra_keys:
            try:
                res.metrics[k] = _lookup(data, k)
            except KeyError:
                res.metrics[k] = None

        if spec.correctness is not None:
            try:
                cval = _lookup(data, spec.correctness.key)
            except KeyError:
                res.status, res.error = "invalid_output", f"correctness key '{spec.correctness.key}' missing"
                return finish(res)
            res.correct = cval == spec.correctness.equals
            if not res.correct:
                res.status, res.error = "incorrect", f"{spec.correctness.key}={cval!r}"
                return finish(res)

        try:
            score = _as_float(_lookup(data, spec.score.key))
        except KeyError:
            score = None
        if score is None:
            res.status, res.error = "invalid_output", f"score key '{spec.score.key}' missing or not a finite number"
            return finish(res)
        res.score = score
        return finish(res)
    finally:
        if not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


def run_repeats(
    spec: BenchmarkSpec,
    *,
    launcher: LocalLauncher,
    out_dir: Path,
    repeats: int | None = None,
    mode: str | None = None,
    candidate: Path | None = None,
    manifest: dict[str, str] | None = None,
    keep_workdir: bool = False,
    stop_on_failure: bool = True,
    on_result: Callable[[int, EvalResult], None] | None = None,
) -> EvalSummary:
    repeats = repeats or spec.noise.repeats
    manifest = manifest if manifest is not None else protected_manifest(spec)
    mode_name = mode or spec.default_mode
    results = []
    for i in range(repeats):
        r = run_eval(spec, launcher=launcher, out_dir=Path(out_dir) / f"rep{i}", mode=mode_name,
                     candidate=candidate, manifest=manifest, keep_workdir=keep_workdir)
        results.append(r)
        if on_result:
            on_result(i, r)
        if stop_on_failure and r.status in ("failed", "timeout", "integrity_error", "invalid_output"):
            break
    return EvalSummary(mode=mode_name, results=results)


def run_tool(
    spec: BenchmarkSpec,
    tool,  # spec.AnalyzeTool
    *,
    launcher: LocalLauncher,
    out_dir: Path,
    candidate: Path | None = None,
    manifest: dict[str, str] | None = None,
    timeout: float | None = None,
    max_chars: int = 12_000,
) -> dict[str, Any]:
    """Run an analyze tool in a clean room of `candidate`; returns status + truncated output."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "out").mkdir(exist_ok=True)
    workdir = out_dir / "workdir"
    if workdir.exists():
        shutil.rmtree(workdir)
    try:
        room = prepare_cleanroom(spec, candidate=candidate, dest=workdir, manifest=manifest)
    except IntegrityError as e:
        return {"tool": tool.name, "status": "integrity_error", "output": str(e)}
    try:
        values = {"out": out_dir / "out", "workdir": room.path, "bench_dir": spec.bench_dir}
        env = {k: render(v, **values) for k, v in spec.env.items()}
        job = launcher.run(render(tool.cmd, **values), cwd=room.path, resources=tool.resources,
                           timeout=min(tool.timeout, timeout or tool.timeout), log_path=out_dir / "tool.log", env=env)
        text = (out_dir / "tool.log").read_text(errors="replace")
        if len(text) > max_chars:  # keep head and tail: summaries are usually at the end
            text = text[: max_chars // 3] + "\n... [truncated] ...\n" + text[-2 * max_chars // 3:]
        status = "ok" if job.ok else ("timeout" if job.timed_out else f"exit {job.returncode}")
        return {"tool": tool.name, "status": status, "duration_s": round(job.duration_s, 1), "output": text}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
