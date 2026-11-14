"""MCP server (real subprocess over stdio) and plugin hook handlers."""

import json
import os
import shutil
import subprocess
import sys
import time

import pytest

from marble import search
from marble.runstate import RunConfig
from marble.search import Controller

from .conftest import PY, ROOT
from .test_loop import POLY_MODEL, FakeAgent


@pytest.fixture
def finished_run(tmp_path, launcher, monkeypatch):
    bench = tmp_path / "toyfit"
    shutil.copytree(ROOT / "benchmarks/toy/fit", bench)
    monkeypatch.setattr(search, "run_agent", FakeAgent([(POLY_MODEL, {})]))
    runs = tmp_path / "runs"
    ctl = Controller(runs / "r1", launcher, RunConfig(benchmark=str(bench), budget_usd=1.0, max_iterations=1),
                     log=lambda s: None)
    ctl.run()
    return runs, bench


class MCP:
    def __init__(self, runs_root):
        self.p = subprocess.Popen([PY, "-m", "marble.cli", "mcp", "--runs-root", str(runs_root)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(ROOT))
        self.i = 0

    def req(self, method, params=None):
        self.i += 1
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.i, "method": method, "params": params or {}}) + "\n")
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

    def call(self, name, **args):
        r = self.req("tools/call", {"name": name, "arguments": args})["result"]
        return r["isError"], (json.loads(r["content"][0]["text"]) if not r["isError"] else r["content"][0]["text"])

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=10)


def test_mcp_protocol_and_tools(finished_run):
    runs, _ = finished_run
    m = MCP(runs)
    try:
        init = m.req("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}})
        assert init["result"]["serverInfo"]["name"] == "marble" and init["result"]["capabilities"]["tools"] is not None
        m.p.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")   # no reply expected
        assert m.req("ping")["result"] == {}
        names = [t["name"] for t in m.req("tools/list")["result"]["tools"]]
        assert {"marble_list_runs", "marble_tree", "marble_diff", "marble_control"} <= set(names)

        err, lr = m.call("marble_list_runs")
        assert not err and lr["runs"][0]["run"] == "r1" and lr["runs"][0]["best"]["node"] == 2
        err, tree = m.call("marble_tree", run="r1")
        assert [n["kind"] for n in tree] == ["root", "draft"] and tree[1]["best"]
        err, node = m.call("marble_node", run="r1", node=2)
        assert node["plan"]["chosen"] == "h1" and "new best" in node["verdict"]
        err, diff = m.call("marble_diff", run="r1", node=2, against="root")
        assert "np.polyfit" in diff["diff"]
        err, ls = m.call("marble_lessons", run="r1")
        assert ls and ls[0]["type"] == "solution_pos"
        err, rep = m.call("marble_report", run="r1")
        assert "val_mse" in rep["markdown"]
        err, ctl = m.call("marble_control", run="r1", action="pause", note="try float32")
        assert ctl["action"] == "pause" and ctl["notes"] == ["try float32"]
        assert json.loads((runs / "r1" / "control.json").read_text())["action"] == "pause"
        err, msg = m.call("marble_node", run="nope", node=1)
        assert err and "no MARBLE run" in msg
        unknown = m.req("tools/call", {"name": "rm_rf", "arguments": {}})
        assert "error" in unknown
        assert "error" in m.req("bogus/method")
    finally:
        m.close()


def _hook(name, payload, env):
    return subprocess.run([PY, "-m", "marble.cli", "hook", name], input=json.dumps(payload), capture_output=True,
                          text=True, env={**os.environ, **env}, cwd=str(ROOT), timeout=60)


def test_session_start_hook_lists_runs(finished_run):
    runs, _ = finished_run
    r = _hook("session-start", {"hook_event_name": "SessionStart"}, {"MARBLE_RUNS_ROOT": str(runs)})
    ctx = json.loads(r.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "r1" in ctx and "toyfit" in ctx


def test_guard_runs_hook(finished_run):
    runs, bench = finished_run
    env = {"MARBLE_RUNS_ROOT": str(runs)}
    edit = lambda path: {"tool_name": "Edit", "tool_input": {"file_path": str(path)}, "cwd": str(ROOT)}
    r = _hook("guard-runs", edit(runs / "r1" / "tree.sqlite"), env)
    assert r.returncode == 2 and "managed by MARBLE" in r.stderr
    assert _hook("guard-runs", edit(runs / "r1" / "control.json"), env).returncode == 0
    # benchmark files are only protected while a run is alive (fresh heartbeat)
    assert _hook("guard-runs", edit(bench / "repo" / "eval" / "eval.py"), env).returncode == 0
    (runs / "r1" / "heartbeat").write_text(json.dumps({"t": time.time()}))
    from marble.runstate import RunState
    RunState(runs / "r1").event("supervisor_start", pid=1, restart=0)
    r = _hook("guard-runs", edit(bench / "repo" / "eval" / "eval.py"), env)
    assert r.returncode == 2 and "is live" in r.stderr
    assert _hook("guard-runs", {"tool_name": "Read", "tool_input": {"file_path": str(bench)}}, env).returncode == 0
