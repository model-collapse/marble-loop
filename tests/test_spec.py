import textwrap

import pytest

from marble.cleanroom import match_any
from marble.spec import SpecError, interpolate_env, load_spec, render, validate_spec

from .conftest import make_bench


def test_load_and_validate(tmp_path):
    bench = make_bench(tmp_path, "write(score=1.0, correct=True)\n")
    spec = load_spec(bench)
    assert spec.name == "t-bench"
    assert spec.default_mode == "quick"
    assert spec.score.better(2.0, 1.0)
    assert validate_spec(spec) == ["editable glob matches no baseline files (only new files): extra/**"]


def test_minimize_direction(tmp_path):
    spec = load_spec(make_bench(tmp_path, "write(score=1.0, correct=True)\n", direction="minimize"))
    assert spec.score.better(1.0, 2.0) and not spec.score.better(2.0, 1.0)


def test_env_interpolation():
    assert interpolate_env("${A}/x", {"A": "/r"}) == "/r/x"
    assert interpolate_env("${B:-dflt}", {}) == "dflt"
    assert interpolate_env({"k": ["${A}"]}, {"A": "v"}) == {"k": ["v"]}
    with pytest.raises(SpecError, match="not set"):
        interpolate_env("${MISSING_VAR_X}", {})


def test_render_placeholders_and_escapes():
    assert render("run {out} --j '{{\"a\": 1}}'", out="/o") == "run /o --j '{\"a\": 1}'"


@pytest.mark.parametrize("mutation,msg", [
    (("editable: [\"solution.py\", \"extra/**\"]", "editable: []"), "editable must not be empty"),
    (("direction: maximize", "direction: up"), "direction"),
    (("timeout: 30", "timeout: -1"), "timeout"),
    (("name: t-bench", "name: \"bad name!\""), "name"),
    (("noise:", "bogus: 1\nnoise:"), "unknown field"),
    (("{out}/score.json", "{nope}/score.json"), "placeholder"),
])
def test_invalid_specs(tmp_path, mutation, msg):
    bench = make_bench(tmp_path, "write(score=1.0, correct=True)\n")
    y = bench / "benchmark.yaml"
    old, new = mutation
    assert old in y.read_text()
    y.write_text(y.read_text().replace(old, new, 1))
    with pytest.raises(SpecError, match=msg):
        validate_spec(load_spec(bench))


def test_validate_rejects_overlap_and_unmatched(tmp_path):
    bench = make_bench(tmp_path, "write(score=1.0, correct=True)\n")
    y = bench / "benchmark.yaml"
    y.write_text(y.read_text().replace('protected: ["eval/**"]', 'protected: ["eval/**", "solution.py"]'))
    with pytest.raises(SpecError, match="both editable and protected"):
        validate_spec(load_spec(bench))
    y.write_text(y.read_text().replace('"solution.py"]', '"nothing_here.py"]'))
    with pytest.raises(SpecError, match="matches no files"):
        validate_spec(load_spec(bench))


def test_short_form_single_mode(tmp_path):
    bench = make_bench(tmp_path, "write(score=1.0, correct=True)\n")
    y = bench / "benchmark.yaml"
    txt = y.read_text()
    y.write_text(txt.replace("evaluate:\n  quick:\n    cmd", "evaluate:\n    cmd").replace("\n    timeout", "\n  timeout")
                 .replace('    cmd: "', '  cmd: "'))
    spec = load_spec(bench)
    assert list(spec.evaluate) == ["default"] and spec.default_mode == "default"


@pytest.mark.parametrize("path,pattern,expected", [
    ("solution.py", "*.py", True),
    ("src/a.py", "*.py", False),          # '*' does not cross directories
    ("src/a.py", "**/*.py", True),
    ("a.py", "**/*.py", True),
    ("eval/x/y.json", "eval/**", True),
    ("evalx/y.json", "eval/**", False),
    ("kernel_1.cu", "kernel_?.cu", True),
])
def test_globs(path, pattern, expected):
    assert match_any(path, [pattern]) is expected


def test_example_benchmarks_validate():
    from .conftest import ROOT
    for b in ("toy/pairwise", "toy/fit"):
        spec = load_spec(ROOT / "benchmarks" / b)
        validate_spec(spec)
