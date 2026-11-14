"""Replay a recorded live run (real Claude agent outputs) without API calls: the run must reproduce exactly."""
import json
import shutil

import pytest

from marble import replay
from marble.runstate import RunConfig
from marble.search import Controller

from .conftest import ROOT

FIXTURE = ROOT / "tests" / "fixtures" / "replay-toyfit"


@pytest.fixture
def replay_env(monkeypatch):
    monkeypatch.setenv("MARBLE_FAKE_AGENT", "marble.replay:agent")
    monkeypatch.setenv("MARBLE_REPLAY_FIXTURE", str(FIXTURE))
    monkeypatch.setattr(replay, "_AGENT", None)


def test_recorded_run_replays_exactly(tmp_path, launcher, replay_env):
    meta = json.loads((FIXTURE / "meta.json").read_text())
    bench = tmp_path / "fit"
    shutil.copytree(ROOT / "benchmarks/toy/fit", bench)
    cfg = RunConfig(benchmark=str(bench), budget_usd=2.0, max_iterations=meta["nodes"] - 1, max_no_improve=99)
    ctl = Controller(tmp_path / "run", launcher, cfg, log=lambda s: None)
    best = ctl.run()
    assert [n.kind for n in ctl.state.nodes()] == meta["kinds"]
    assert best.id == meta["best_node"] and best.score == pytest.approx(meta["best_score"], rel=1e-9)
    assert ctl.state.total_cost() == 0.0                          # no API calls
    assert not replay._AGENT.queues or all(not q for q in replay._AGENT.queues.values())


def test_export_roundtrip(tmp_path, launcher, replay_env):
    bench = tmp_path / "fit"
    shutil.copytree(ROOT / "benchmarks/toy/fit", bench)
    ctl = Controller(tmp_path / "run", launcher, RunConfig(benchmark=str(bench), budget_usd=2.0, max_iterations=1),
                     log=lambda s: None)
    ctl.run()
    meta = replay.export(tmp_path / "run", tmp_path / "fx")
    calls = [json.loads(l) for l in (tmp_path / "fx" / "calls.jsonl").read_text().splitlines()]
    impl = [c for c in calls if c["role"] == "implementer"]
    assert impl and "model.py" in impl[0]["files"] and meta["best_node"] == 2
