"""M1 loop tests with a scripted fake agent (no API calls)."""

import json
import shutil
from pathlib import Path

import pytest

from marble import search
from marble.agents import AgentResult
from marble.hooks import check_edit
from marble.runstate import Node, RunConfig, RunState
from marble.search import Controller, improvement
from marble.spec import load_spec
from marble.workspace import Workspace

from .conftest import ROOT

POLY_MODEL = '''import numpy as np


def fit(x, y):
    c = np.polyfit(x, y, 6)
    return lambda xs: np.polyval(c, np.asarray(xs))
'''
BROKEN_MODEL = "def fit(x, y):\n    return lambda xs: undefined_name\n"


# ------------------------------------------------------------------ unit pieces


def test_workspace_commit_revert_diff(tmp_path):
    base = tmp_path / "base"
    (base / "eval").mkdir(parents=True)
    (base / "solution.py").write_text("A = 1\n")
    (base / "eval" / "e.py").write_text("x\n")
    ws = Workspace(tmp_path / "run")
    root = ws.init(base)
    wt = ws.add_worktree(1, root)
    (wt / "solution.py").write_text("A = 2\n")
    (wt / "eval" / "e.py").write_text("hacked\n")
    (wt / "junk.txt").write_text("j\n")
    assert ws.enforce_editable(wt, ["solution.py"]) == ["eval/e.py", "junk.txt"]
    assert (wt / "eval" / "e.py").read_text() == "x\n" and not (wt / "junk.txt").exists()
    sha = ws.commit(wt, "change")
    assert sha and sha != root
    assert "-A = 1" in ws.diff(root, sha) and "+A = 2" in ws.diff(root, sha)
    assert ws.commit(wt, "nothing") is None
    ws.remove_worktree(1)
    assert not wt.exists()


@pytest.mark.parametrize("path,ok", [("solution.py", True), ("eval/e.py", False), ("/etc/passwd", False),
                                     ("../outside.py", False), ("sub/new.py", False)])
def test_guard_check_edit(tmp_path, path, ok):
    payload = {"tool_name": "Write", "tool_input": {"file_path": path}, "cwd": str(tmp_path)}
    reason = check_edit(payload, str(tmp_path), ["solution.py"])
    assert (reason is None) is ok
    assert check_edit({"tool_name": "Read", "tool_input": {"file_path": "/etc/passwd"}}, str(tmp_path), []) is None


def test_improvement_rule():
    spec = load_spec(ROOT / "benchmarks/toy/fit")             # minimize, min_rel_improvement 0.01
    best = Node(id=1, parent=None, sha=None, kind="root", score=1.0, score_rel_std=0.0)
    assert improvement(spec, best, Node(id=2, parent=1, sha=None, kind="improve", score=0.9, score_rel_std=0.0))[0]
    assert not improvement(spec, best, Node(id=3, parent=1, sha=None, kind="improve", score=0.995, score_rel_std=0.0))[0]
    # noise widens the threshold: 4% gain is not enough with 3% relative std
    assert not improvement(spec, best, Node(id=4, parent=1, sha=None, kind="improve", score=0.96, score_rel_std=0.03))[0]


# ------------------------------------------------------------------ fake agent loop


class FakeAgent:
    """Scripted stand-in for marble.agents.run_agent."""

    def __init__(self, implement_sequence):
        self.implement_sequence = list(implement_sequence)
        self.calls = []

    def __call__(self, call, log_dir):
        self.calls.append(call.role)
        out = {
            "analyst": {"summary": "linear fit underfits a curved target", "bottlenecks": [{"what": "bias", "evidence": "val_mse 0.80"}],
                        "opportunities": ["higher-degree polynomial"]},
            "planner": {"hypotheses": [{"id": "h1", "family": "model capacity", "idea": "degree-6 polynomial", "expected_gain": "large"}],
                        "chosen": "h1", "plan": "replace OLS line with np.polyfit degree 6", "files": ["model.py"], "cites": []},
            "reflector": {"title": "capacity", "change": "poly6", "outcome": "improved", "cause": "success",
                          "evidence": "mse drop", "key_lesson": "match model capacity to target curvature", "applies_when": "underfitting"},
        }.get(call.role)
        if call.role in ("implementer", "debugger"):
            content, extra = self.implement_sequence.pop(0)
            (Path(call.cwd) / "model.py").write_text(content)
            for rel, txt in extra.items():  # simulate a misbehaving agent writing elsewhere via Bash
                (Path(call.cwd) / rel).parent.mkdir(parents=True, exist_ok=True)
                (Path(call.cwd) / rel).write_text(txt)
            out = ({"summary": "wrote model", "files_changed": ["model.py"], "self_check": "none"} if call.role == "implementer"
                   else {"diagnosis": "NameError", "fix": "use polyfit", "files_changed": ["model.py"]})
        return AgentResult(role=call.role, ok=True, output=out, text=json.dumps(out), cost_usd=0.01, duration_s=0.01)


@pytest.fixture
def toy_fit_copy(tmp_path):
    dst = tmp_path / "toyfit"
    shutil.copytree(ROOT / "benchmarks/toy/fit", dst)
    return dst


def _controller(tmp_path, bench, launcher, **cfg):
    c = RunConfig(benchmark=str(bench), budget_usd=1.0, max_iterations=cfg.pop("max_iterations", 1), **cfg)
    return Controller(tmp_path / "run", launcher, c, log=lambda s: None)


def test_loop_improves_toy_fit(tmp_path, launcher, toy_fit_copy, monkeypatch):
    fake = FakeAgent([(POLY_MODEL, {})])
    monkeypatch.setattr(search, "run_agent", fake)
    ctl = _controller(tmp_path, toy_fit_copy, launcher)
    best = ctl.run()
    nodes = ctl.state.nodes()
    assert [n.kind for n in nodes] == ["root", "improve"]
    assert best.id == 2 and best.improved == 1 and best.score < nodes[0].score * 0.5
    assert fake.calls == ["analyst", "planner", "implementer", "reflector"]
    assert len(ctl.state.lessons()) == 1
    assert abs(ctl.state.total_cost() - 0.04) < 1e-9
    assert (tmp_path / "run" / "report.md").exists() and (tmp_path / "run" / "best.patch").exists()
    assert "np.polyfit" in (tmp_path / "run" / "best.patch").read_text()


def test_loop_reverts_non_editable_and_debugs(tmp_path, launcher, toy_fit_copy, monkeypatch):
    # First attempt is broken AND writes to the evaluator; the debugger then fixes it.
    fake = FakeAgent([(BROKEN_MODEL, {"eval/eval.py": "print('hacked')\n"}), (POLY_MODEL, {})])
    monkeypatch.setattr(search, "run_agent", fake)
    ctl = _controller(tmp_path, toy_fit_copy, launcher)
    best = ctl.run()
    ev = ctl.state.events()
    assert any(e["type"] == "reverted_non_editable" and e["files"] == ["eval/eval.py"] for e in ev)
    evals = [e for e in ev if e["type"] == "eval" and e["node"] == 2]
    assert [e["status"] for e in evals] == ["failed", "ok"]          # try0 failed, try1 after debug ok
    assert "debugger" in fake.calls and best.id == 2
    # the evaluator in the pristine benchmark is untouched
    assert "hacked" not in (toy_fit_copy / "repo/eval/eval.py").read_text()


def test_budget_stops_run(tmp_path, launcher, toy_fit_copy, monkeypatch):
    monkeypatch.setattr(search, "run_agent", FakeAgent([(POLY_MODEL, {})] * 5))
    ctl = _controller(tmp_path, toy_fit_copy, launcher, max_iterations=5)
    ctl.cfg.budget_usd = 0.52   # one iteration costs $0.04; stop when < $0.50 remains
    ctl.state.save_config(ctl.cfg)
    ctl.run()
    stops = [e for e in ctl.state.events() if e["type"] == "run_stopped"]
    assert stops and "budget" in stops[-1]["reason"]


# ------------------------------------------------------------------ confirmation in full mode

VEC_SOLUTION = '''import numpy as np


def pairwise_distances(X):
    A = np.asarray(X, dtype=np.float64)
    sq = np.einsum("ij,ij->i", A, A)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * A @ A.T, 0.0))
'''
# Fast and correct only at the quick-mode size (n=120); wrong at the full-mode size (n=240).
OVERFIT_SOLUTION = VEC_SOLUTION.replace(
    "    return np.sqrt(", "    if len(X) != 120:\n        return np.zeros((len(X), len(X)))\n    return np.sqrt(")


class FakePairwise(FakeAgent):
    def __call__(self, call, log_dir):
        if call.role == "implementer":
            self.calls.append(call.role)
            (Path(call.cwd) / "solution.py").write_text(self.implement_sequence.pop(0)[0])
            out = {"summary": "vectorized", "files_changed": ["solution.py"], "self_check": "none"}
            return AgentResult(role=call.role, ok=True, output=out, cost_usd=0.01)
        return super().__call__(call, log_dir)


@pytest.fixture
def pairwise_copy(tmp_path):
    dst = tmp_path / "pw"
    shutil.copytree(ROOT / "benchmarks/toy/pairwise", dst)
    return dst


def test_new_best_confirmed_in_full_mode(tmp_path, launcher, pairwise_copy, monkeypatch):
    monkeypatch.setattr(search, "run_agent", FakePairwise([(VEC_SOLUTION, {})]))
    ctl = _controller(tmp_path, pairwise_copy, launcher, analyze=False)
    best = ctl.run()
    assert best.id == 2 and best.full_status == "ok" and best.full_score > 10
    assert "confirmed in full" in (tmp_path / "run" / "report.md").read_text()


def test_quick_only_solution_rejected_by_full_mode(tmp_path, launcher, pairwise_copy, monkeypatch):
    monkeypatch.setattr(search, "run_agent", FakePairwise([(OVERFIT_SOLUTION, {})]))
    ctl = _controller(tmp_path, pairwise_copy, launcher, analyze=False)
    best = ctl.run()
    cand = ctl.state.get(2)
    assert cand.eval_status == "ok" and cand.score > 10          # looked great in quick mode
    assert cand.full_status == "incorrect" and not cand.improved  # caught by the full-mode check
    assert best.id == 1
    v = [e for e in ctl.state.events() if e["type"] == "verdict"][-1]
    assert "not confirmed" in v["verdict"]


def test_resume_continues_from_existing_tree(tmp_path, launcher, toy_fit_copy, monkeypatch):
    better = POLY_MODEL.replace("6)", "8)")
    monkeypatch.setattr(search, "run_agent", FakeAgent([(POLY_MODEL, {}), (better, {})]))
    ctl = _controller(tmp_path, toy_fit_copy, launcher)
    ctl.run()
    assert len(ctl.state.nodes()) == 2
    cfg = ctl.state.load_config()
    cfg.max_iterations = 2
    ctl2 = Controller(tmp_path / "run", launcher, cfg, log=lambda s: None)
    ctl2.run()
    nodes = ctl2.state.nodes()
    assert [n.kind for n in nodes] == ["root", "improve", "improve"]   # no second root, no re-init
    assert nodes[2].parent == 2                                          # expanded the previous best
    assert sum(1 for e in ctl2.state.events() if e["type"] == "run_stopped") == 2
