"""Benchmark specification (`benchmark.yaml`): schema, loading, validation.

A benchmark is a baseline repo plus a declarative contract:
which files the agent may edit, which are protected, how to evaluate, how to read the score.

Placeholders available in commands: {out} (per-eval output dir), {workdir} (clean-room copy of
the repo), {bench_dir} (directory holding benchmark.yaml).
Environment variables are interpolated at load time: ${VAR} or ${VAR:-default}.
"""

from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 1
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_ALLOWED_PLACEHOLDERS = {"out", "workdir", "bench_dir"}


class SpecError(ValueError):
    """Raised for an invalid benchmark specification."""


@dataclass
class Resources:
    gpu: int = 0
    cpu: int = 1

    @classmethod
    def parse(cls, raw: Any, where: str) -> "Resources":
        raw = raw or {}
        _expect_mapping(raw, where)
        _reject_unknown(raw, {"gpu", "cpu"}, where)
        gpu, cpu = raw.get("gpu", 0), raw.get("cpu", 1)
        if not isinstance(gpu, int) or gpu < 0:
            raise SpecError(f"{where}.gpu must be a non-negative integer")
        if not isinstance(cpu, int) or cpu < 1:
            raise SpecError(f"{where}.cpu must be a positive integer")
        return cls(gpu=gpu, cpu=cpu)


@dataclass
class CommandSpec:
    """A command run inside the clean-room copy (evaluate or analyze)."""

    cmd: str
    timeout: float
    resources: Resources = field(default_factory=Resources)
    expected_duration: float | None = None  # None = "auto" (from calibration)
    calibration_only: bool = False  # mode may only evaluate the unmodified baseline

    @classmethod
    def parse(cls, raw: Any, where: str) -> "CommandSpec":
        _expect_mapping(raw, where)
        _reject_unknown(raw, {"cmd", "timeout", "resources", "expected_duration", "calibration_only"}, where)
        cmd = raw.get("cmd")
        if not isinstance(cmd, str) or not cmd.strip():
            raise SpecError(f"{where}.cmd must be a non-empty string")
        _check_placeholders(cmd, f"{where}.cmd")
        timeout = raw.get("timeout")
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise SpecError(f"{where}.timeout must be a positive number of seconds")
        expected = raw.get("expected_duration", "auto")
        if expected == "auto" or expected is None:
            expected = None
        elif not isinstance(expected, (int, float)) or expected <= 0:
            raise SpecError(f"{where}.expected_duration must be 'auto' or a positive number")
        elif expected > timeout:
            raise SpecError(f"{where}.expected_duration ({expected}) exceeds timeout ({timeout})")
        calib = raw.get("calibration_only", False)
        if not isinstance(calib, bool):
            raise SpecError(f"{where}.calibration_only must be true or false")
        return cls(
            cmd=cmd,
            timeout=float(timeout),
            resources=Resources.parse(raw.get("resources"), f"{where}.resources"),
            expected_duration=None if expected is None else float(expected),
            calibration_only=calib,
        )


@dataclass
class AnalyzeTool(CommandSpec):
    name: str = ""


@dataclass
class ScoreSpec:
    file: str
    key: str
    direction: str  # "maximize" | "minimize"
    extra_keys: list[str] = field(default_factory=list)

    def better(self, a: float, b: float) -> bool:
        """True if score a is strictly better than b."""
        return a > b if self.direction == "maximize" else a < b


@dataclass
class CorrectnessSpec:
    key: str
    equals: Any = True


@dataclass
class NoiseSpec:
    repeats: int = 1
    min_rel_improvement: float = 0.0


@dataclass
class BenchmarkSpec:
    name: str
    bench_dir: Path
    repo: Path
    editable: list[str]
    protected: list[str]
    evaluate: dict[str, CommandSpec]
    default_mode: str
    score: ScoreSpec
    correctness: CorrectnessSpec | None
    noise: NoiseSpec
    description: str = ""
    setup: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    analyze: list[AnalyzeTool] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def mode(self, name: str | None) -> CommandSpec:
        name = name or self.default_mode
        if name not in self.evaluate:
            raise SpecError(f"unknown eval mode '{name}'; available: {sorted(self.evaluate)}")
        return self.evaluate[name]


# --------------------------------------------------------------------------- loading


def interpolate_env(value: Any, env: dict[str, str] | None = None) -> Any:
    """Recursively replace ${VAR} / ${VAR:-default} in strings."""
    env = os.environ if env is None else env
    if isinstance(value, str):

        def repl(m: re.Match) -> str:
            var, default = m.group(1), m.group(2)
            if var in env:
                return env[var]
            if default is not None:
                return default
            raise SpecError(f"environment variable ${{{var}}} is not set and has no default")

        return _ENV_RE.sub(repl, value)
    if isinstance(value, list):
        return [interpolate_env(v, env) for v in value]
    if isinstance(value, dict):
        return {k: interpolate_env(v, env) for k, v in value.items()}
    return value


def load_spec(path: str | os.PathLike, env: dict[str, str] | None = None) -> BenchmarkSpec:
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "benchmark.yaml"
    if not path.is_file():
        raise SpecError(f"benchmark file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise SpecError("benchmark.yaml must contain a mapping at top level")
    if env is None:
        env = dict(os.environ)
        # Default interpreter for adapters: the Python running MARBLE (override with MARBLE_PYTHON).
        env.setdefault("MARBLE_PYTHON", sys.executable)
    raw = interpolate_env(raw, env)
    return parse_spec(raw, bench_dir=path.parent)


def parse_spec(raw: dict, bench_dir: Path) -> BenchmarkSpec:
    top = {
        "schema_version", "name", "description", "repo", "setup", "env", "editable", "protected",
        "analyze", "evaluate", "default_mode", "score", "correctness", "noise",
    }
    _reject_unknown(raw, top, "benchmark")
    version = raw.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise SpecError(f"unsupported schema_version {version} (this MARBLE supports {SCHEMA_VERSION})")

    name = raw.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.\-]*", name or ""):
        raise SpecError("name must be a non-empty identifier ([A-Za-z0-9_.-])")

    repo = raw.get("repo")
    if not isinstance(repo, str):
        raise SpecError("repo must be a path to the baseline code")
    repo_path = (bench_dir / repo).resolve()

    editable = _str_list(raw.get("editable"), "editable", required=True)
    protected = _str_list(raw.get("protected", []), "protected")

    ev_raw = raw.get("evaluate")
    _expect_mapping(ev_raw, "evaluate")
    if not ev_raw:
        raise SpecError("evaluate must define at least one mode (e.g. quick)")
    # Allow the short form `evaluate: {cmd: ..., timeout: ...}` = single mode "default".
    if "cmd" in ev_raw:
        ev_raw = {"default": ev_raw}
    evaluate = {m: CommandSpec.parse(v, f"evaluate.{m}") for m, v in ev_raw.items()}
    default_mode = raw.get("default_mode") or ("quick" if "quick" in evaluate else next(iter(evaluate)))
    if default_mode not in evaluate:
        raise SpecError(f"default_mode '{default_mode}' is not one of {sorted(evaluate)}")
    if evaluate[default_mode].calibration_only:
        raise SpecError(f"default_mode '{default_mode}' is calibration_only; it cannot score candidates")

    analyze = []
    for i, a in enumerate(raw.get("analyze") or []):
        _expect_mapping(a, f"analyze[{i}]")
        a = dict(a)
        tool_name = a.pop("name", None)
        if not isinstance(tool_name, str) or not tool_name:
            raise SpecError(f"analyze[{i}].name is required")
        base = CommandSpec.parse(a, f"analyze[{i}]")
        analyze.append(AnalyzeTool(**base.__dict__, name=tool_name))

    sc = raw.get("score")
    _expect_mapping(sc, "score")
    _reject_unknown(sc, {"file", "key", "direction", "extra_keys"}, "score")
    if sc.get("direction") not in ("maximize", "minimize"):
        raise SpecError("score.direction must be 'maximize' or 'minimize'")
    for k in ("file", "key"):
        if not isinstance(sc.get(k), str) or not sc[k]:
            raise SpecError(f"score.{k} is required")
    _check_placeholders(sc["file"], "score.file")
    score = ScoreSpec(
        file=sc["file"], key=sc["key"], direction=sc["direction"],
        extra_keys=_str_list(sc.get("extra_keys", []), "score.extra_keys"),
    )

    correctness = None
    if raw.get("correctness") is not None:
        c = raw["correctness"]
        _expect_mapping(c, "correctness")
        _reject_unknown(c, {"key", "equals"}, "correctness")
        if not isinstance(c.get("key"), str):
            raise SpecError("correctness.key is required")
        correctness = CorrectnessSpec(key=c["key"], equals=c.get("equals", True))

    n = raw.get("noise") or {}
    _expect_mapping(n, "noise")
    _reject_unknown(n, {"repeats", "min_rel_improvement"}, "noise")
    repeats = n.get("repeats", 1)
    if not isinstance(repeats, int) or repeats < 1:
        raise SpecError("noise.repeats must be a positive integer")
    mri = n.get("min_rel_improvement", 0.0)
    if not isinstance(mri, (int, float)) or mri < 0 or math.isnan(mri):
        raise SpecError("noise.min_rel_improvement must be >= 0")

    env = raw.get("env") or {}
    _expect_mapping(env, "env")
    env = {str(k): str(v) for k, v in env.items()}

    setup = raw.get("setup")
    if setup is not None and not isinstance(setup, str):
        raise SpecError("setup must be a shell command string")

    return BenchmarkSpec(
        name=name, bench_dir=bench_dir, repo=repo_path, editable=editable, protected=protected,
        evaluate=evaluate, default_mode=default_mode, score=score, correctness=correctness,
        noise=NoiseSpec(repeats=repeats, min_rel_improvement=float(mri)),
        description=str(raw.get("description", "")), setup=setup, env=env, analyze=analyze,
        schema_version=version,
    )


# --------------------------------------------------------------------------- validation


def validate_spec(spec: BenchmarkSpec) -> list[str]:
    """Check the spec against the repo on disk. Returns warnings; raises SpecError on errors."""
    from .cleanroom import list_repo_files, match_any  # local import: avoid cycle

    warnings: list[str] = []
    if not spec.repo.is_dir():
        raise SpecError(f"repo directory not found: {spec.repo}")
    files = list_repo_files(spec.repo)
    if not files:
        raise SpecError(f"repo is empty: {spec.repo}")

    if not any(match_any(f, spec.editable) for f in files):
        raise SpecError(f"editable globs match no files in the baseline repo: {spec.editable}")
    for g in spec.editable:
        if not any(match_any(f, [g]) for f in files):
            # Allowed: a glob may only permit *new* files (e.g. "src/kernels/**").
            warnings.append(f"editable glob matches no baseline files (only new files): {g}")
    for g in spec.protected:
        if not any(match_any(f, [g]) for f in files):
            raise SpecError(f"protected glob matches no files: {g}")
    overlap = [f for f in files if match_any(f, spec.editable) and match_any(f, spec.protected)]
    if overlap:
        raise SpecError(f"files are both editable and protected: {overlap[:5]}")
    if not spec.protected:
        warnings.append("no protected files: the evaluator itself is not checksum-guarded")
    if spec.correctness is None:
        warnings.append("no correctness gate: any score is accepted")
    if spec.noise.repeats == 1:
        warnings.append("noise.repeats=1: improvements cannot be separated from noise")
    if spec.default_mode != "quick" and len(spec.evaluate) > 1:
        warnings.append(f"default_mode is '{spec.default_mode}', not 'quick'")
    return warnings


# --------------------------------------------------------------------------- helpers


def _expect_mapping(v: Any, where: str) -> None:
    if not isinstance(v, dict):
        raise SpecError(f"{where} must be a mapping")


def _reject_unknown(d: dict, allowed: set[str], where: str) -> None:
    unknown = set(d) - allowed
    if unknown:
        raise SpecError(f"{where}: unknown field(s) {sorted(unknown)}; allowed: {sorted(allowed)}")


def _str_list(v: Any, where: str, required: bool = False) -> list[str]:
    if v is None:
        v = []
    if not isinstance(v, list) or not all(isinstance(x, str) and x for x in v):
        raise SpecError(f"{where} must be a list of non-empty strings")
    if required and not v:
        raise SpecError(f"{where} must not be empty")
    return list(v)


def _check_placeholders(s: str, where: str) -> None:
    # Only single-brace {name} placeholders are ours; '{{' / '}}' escape literal braces.
    for name in re.findall(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})", s):
        if name not in _ALLOWED_PLACEHOLDERS:
            raise SpecError(f"{where}: unknown placeholder {{{name}}}; allowed: {sorted(_ALLOWED_PLACEHOLDERS)}")


def render(template: str, **values: Any) -> str:
    """Fill {out}/{workdir}/{bench_dir}; '{{' and '}}' become literal braces."""
    out = re.sub(
        r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})",
        lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0),
        template,
    )
    return out.replace("{{", "{").replace("}}", "}")
