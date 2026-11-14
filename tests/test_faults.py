"""Fault-injection suite (DESIGN.md §11.1): every fault must be detected and recovered automatically."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from marble import inspector as insp
from marble import search
from marble.runstate import RunConfig
from marble.search import Controller

from .conftest import PY, ROOT, make_bench
from .fakes import PLAN, Scripted

# The evaluator calls solution.run() and scores its return value; the candidate decides how it behaves.
EVAL = """
import solution
v = solution.run()
write(score=float(v), correct=True)
"""
BASE = "def run():\n    return 1.0\n"
GOOD = "def run():\n    return 2.0\n"
HANG = "import time\n\ndef run():\n    time.sleep(600)\n    return 9.0\n"
BUSY_LOOP = "def run():\n    while True:\n        pass\n"
SLOW_ACTIVE = ("import sys, time\n\ndef run():\n    for i in range(30):\n        print('step', i, flush=True)\n"
               "        time.sleep(0.1)\n    return 2.0\n")
BROKEN = "def run():\n    raise ValueError('bad shape (3, 4)')\n"
BROKEN_AGAIN = "def run():\n    raise ValueError('bad shape (5, 6)')  # 'fixed'\n"


@pytest.fixture(autouse=True)
def fast_inspector(monkeypatch):
    monkeypatch.setattr(insp, "SOFT_MIN_EXTRA_S", 0.5)


def _ctl(tmp_path, launcher, *, inspector="rule", stall_s=1.5, max_iterations=1, timeout=60, **cfg):
    bench = make_bench(tmp_path, EVAL, solution=BASE, timeout=timeout)
    y = bench / "benchmark.yaml"
    y.write_text(y.read_text().replace("noise: {repeats: 2", "noise: {repeats: 1"))
    c = RunConfig(benchmark=str(bench), budget_usd=2.0, max_iterations=max_iterations, analyze=False,
                  inspector=inspector, stall_s=stall_s, confirm_mode=None, **cfg)
    return Controller(tmp_path / "run", launcher, c, log=lambda s: None)


def _incidents(ctl, kind=None):
    return [i for i in ctl.incidents.all() if kind is None or i["kind"] == kind]


def test_hanging_eval_is_killed_and_debugged(tmp_path, launcher, monkeypatch):
    fake = Scripted([HANG, GOOD])
    monkeypatch.setattr(search, "run_agent", fake)
    ctl = _ctl(tmp_path, launcher)
    t = time.monotonic()
    best = ctl.run()
    assert time.monotonic() - t < 40                         # nowhere near the 60 s hard timeout x attempts
    inc = _incidents(ctl, "stalled")
    assert inc and inc[0]["action"] == "kill_and_debug"
    evals = [e for e in ctl.state.events() if e["type"] == "eval" and e["node"] == 2]
    assert [e["status"] for e in evals] == ["cancelled", "ok"]
    dbg = [p for r, p in fake.prompts if r == "debugger"][0]
    assert "Inspector (stalled)" in dbg                         # the debugger sees the inspector's diagnosis
    assert best.id == 2


def test_slow_but_active_eval_is_extended_not_killed(tmp_path, launcher, monkeypatch):
    monkeypatch.setattr(search, "run_agent", Scripted([SLOW_ACTIVE]))
    ctl = _ctl(tmp_path, launcher)
    best = ctl.run()
    assert ctl.state.get(2).eval_status == "ok" and best.id == 2
    actions = [i["action"] for i in _incidents(ctl, "slow")]
    assert actions[0] == "auto_extend"
    assert not _incidents(ctl, "stalled") and not any("kill" in a for a in actions)


def test_busy_infinite_loop_killed_by_llm_inspector(tmp_path, launcher, monkeypatch):
    fake = Scripted([BUSY_LOOP, GOOD],
                    inspector={"diagnosis": "100% CPU, no output for the whole run: infinite loop",
                               "action": "kill_and_debug", "lesson": "busy loops show CPU but no output"})
    monkeypatch.setattr(search, "run_agent", fake)
    ctl = _ctl(tmp_path, launcher, inspector="llm")
    best = ctl.run()
    assert "inspector" in fake.calls
    inc = [i for i in _incidents(ctl) if i["action"] == "kill_and_debug"]
    assert inc and "infinite loop" in inc[0]["detail"]
    assert best.id == 2 and ctl.state.get(2).eval_status == "ok"
    assert any(e["role"] == "inspector" for e in ctl.state.events() if e["type"] == "cost")


def test_repeated_failure_stops_debugging_early(tmp_path, launcher, monkeypatch):
    fake = Scripted([BROKEN, BROKEN_AGAIN, GOOD, GOOD])
    monkeypatch.setattr(search, "run_agent", fake)
    ctl = _ctl(tmp_path, launcher, debug_attempts=3)
    ctl.run()
    assert fake.calls.count("debugger") == 1                   # not 3: the 'fix' left the same failure
    inc = _incidents(ctl, "repeated_failure")
    assert inc and inc[0]["action"] == "stop debugging"
    assert ctl.state.get(2).status == "buggy"


def test_duplicate_code_is_not_reevaluated(tmp_path, launcher, monkeypatch):
    same = "def run():\n    return 1.0  # same value, different text\n"
    fake = Scripted([same, same])
    monkeypatch.setattr(search, "run_agent", fake)
    ctl = _ctl(tmp_path, launcher, max_iterations=2, max_no_improve=5, search="greedy")  # greedy re-expands the root
    ctl.run()
    n3 = ctl.state.get(3)
    assert n3.status == "duplicate" and "node #2" in n3.error
    assert not [e for e in ctl.state.events() if e["type"] == "eval" and e["node"] == 3]
    assert _incidents(ctl, "duplicate_code")


def test_repeated_idea_triggers_replan(tmp_path, launcher, monkeypatch):
    other = json.loads(json.dumps(PLAN))
    other["hypotheses"][0].update(family="caching", idea="memoise intermediate results with an lru cache layer")
    fake = Scripted([BASE.replace("1.0", "1.0 + 0"), GOOD], plans=[PLAN, PLAN, other])
    monkeypatch.setattr(search, "run_agent", fake)
    ctl = _ctl(tmp_path, launcher, max_iterations=2, max_no_improve=5)
    ctl.run()
    assert fake.calls.count("planner") == 3                    # iteration 2 re-planned once
    assert _incidents(ctl, "repeated_idea")
    assert "caching" in ctl.state.get(3).plan


def test_supervisor_restarts_crashed_controller(tmp_path):
    bench = tmp_path / "toyfit"
    shutil.copytree(ROOT / "benchmarks/toy/fit", bench)
    poly = "import numpy as np\n\ndef fit(x, y):\n    c = np.polyfit(x, y, 6)\n    return lambda xs: np.polyval(c, np.asarray(xs))\n"
    run = tmp_path / "run"
    from marble.runstate import RunState
    st = RunState(run)
    st.save_config(RunConfig(benchmark=str(bench), budget_usd=1.0, max_iterations=2, analyze=False, inspector="rule"))
    env = dict(os.environ, PYTHONPATH=str(ROOT), MARBLE_FAKE_AGENT="tests.fakes:subprocess_agent",
               MARBLE_FAKE_CODES=json.dumps([poly, poly.replace(", 6)", ", 8)")]), MARBLE_FAKE_TARGET="model.py",
               MARBLE_FAKE_CRASH_MARKER=str(tmp_path / "crashed"))
    p = subprocess.run([PY, "-m", "marble.cli", "run", "supervise", str(run), "--gpus", "none", "--max-restarts", "2"],
                       env=env, capture_output=True, text=True, timeout=300, cwd=str(ROOT))
    assert p.returncode == 0, p.stdout + p.stderr
    assert (tmp_path / "crashed").exists()                      # the controller really died once
    st = RunState(run)
    nodes = st.nodes()
    assert nodes[1].status == "aborted" and "interrupted" in nodes[1].error
    assert any(n.improved for n in nodes[2:])                  # the restarted controller finished the work
    kinds = [json.loads(l)["kind"] for l in (run / "incidents.jsonl").read_text().splitlines()]
    assert "controller_restart" in kinds and "interrupted" in kinds
    assert sum(1 for e in st.events() if e["type"] == "supervisor_start") == 2


def test_failure_signature_ignores_numbers_paths_and_source_lines():
    a = 'Traceback...\n  File "/tmp/a/try0/solution.py", line 2\n    raise ValueError("x")  # v1\nValueError: bad shape (3, 4)\n'
    b = 'Traceback...\n  File "/tmp/b/try1/solution.py", line 9\n    raise ValueError("x")  # v2\nValueError: bad shape (5, 6)\n'
    c = 'Traceback...\nKeyError: missing key\n'
    assert insp.failure_signature("exit code 1; see /x/try0/eval.log", a) == insp.failure_signature("exit code 1; see /y/try1/eval.log", b)
    assert insp.failure_signature("exit code 1", a) != insp.failure_signature("exit code 1", c)


def test_plan_similarity():
    assert insp.plan_similarity("vectorize the distance loop with numpy", "vectorize the distance loop with numpy") == 1.0
    assert insp.plan_similarity("vectorize the distance loop with numpy", "write a fused cuda kernel for gelu") < 0.2
