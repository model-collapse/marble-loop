import sys
import textwrap
from pathlib import Path

import pytest

from marble.launcher import LocalLauncher, Slot

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


@pytest.fixture
def launcher():
    return LocalLauncher([Slot("cpu0", cpus=2), Slot("cpu1", cpus=2)], sample_interval=0.1)


def make_bench(tmp_path: Path, eval_body: str, *, solution: str = "VALUE = 1.0\n", extra_yaml: str = "",
               timeout: float = 30, direction: str = "maximize", correctness: bool = True) -> Path:
    """Create a minimal benchmark whose evaluator runs `eval_body` (Python) and writes score.json."""
    bench = tmp_path / "bench"
    (bench / "repo" / "eval").mkdir(parents=True)
    (bench / "repo" / "solution.py").write_text(solution)
    (bench / "repo" / "README.md").write_text("not editable\n")
    (bench / "repo" / "eval" / "eval.py").write_text(textwrap.dedent("""\
        import json, os, sys, time
        sys.path.insert(0, os.getcwd())
        out = sys.argv[1]
        os.makedirs(out, exist_ok=True)
        def write(**kw):
            with open(os.path.join(out, "score.json"), "w") as f:
                json.dump(kw, f)
        """) + textwrap.dedent(eval_body))
    lines = [
        "schema_version: 1",
        "name: t-bench",
        "repo: repo",
        'editable: ["solution.py", "extra/**"]',
        'protected: ["eval/**"]',
        "evaluate:",
        "  quick:",
        f'    cmd: "{PY} eval/eval.py {{out}}"',
        f"    timeout: {timeout}",
        f'score: {{file: "{{out}}/score.json", key: score, direction: {direction}}}',
    ]
    if correctness:
        lines.append("correctness: {key: correct, equals: true}")
    lines.append("noise: {repeats: 2, min_rel_improvement: 0.01}")
    (bench / "benchmark.yaml").write_text("\n".join(lines) + "\n" + extra_yaml)
    return bench
