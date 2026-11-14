"""Record / replay of agent behaviour for deterministic CI (DESIGN.md §9, "deterministic tests").

Export a finished run into a compact fixture:
    marble replay export runs/<run> tests/fixtures/<name>
      calls.jsonl  one line per agent call, in the order they happened:
                   {"role", "node", "output"}            (structured output of the call)
                   + "files": {path: content | null}     (implementer/debugger: the exact change made)
      meta.json    benchmark name, number of nodes, best score, cost (for assertions)
Replay it without API calls:
    MARBLE_FAKE_AGENT=marble.replay:agent MARBLE_REPLAY_FIXTURE=tests/fixtures/<name> marble run start ...
Each role's calls are answered in recorded order; implementer/debugger calls write the recorded files.
Replays are exact for deterministic benchmarks (same scores => same search path).
"""

from __future__ import annotations

import json
import os
import subprocess
from collections import defaultdict, deque
from pathlib import Path

from .agents import AgentResult
from .runstate import RunState

CODE_ROLES = ("implementer", "debugger")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True, check=True).stdout


def export(run_dir: Path, out_dir: Path) -> dict:
    run_dir, out_dir = Path(run_dir).resolve(), Path(out_dir)
    st = RunState(run_dir)
    nodes = {n.id: n for n in st.nodes()}
    repo = run_dir / "repo"
    logs = []
    for p in (run_dir / "nodes").glob("*/agents/*.json"):
        role = p.name.rsplit("-", 1)[0]
        logs.append((p.stat().st_mtime, p, role, int(p.parent.parent.name)))
    logs.sort()
    commits: dict[int, deque] = {}
    calls = []
    for _, p, role, nid in logs:
        try:
            raw = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        entry = {"role": role, "node": nid, "output": raw.get("structured_output")}
        if role in CODE_ROLES:
            n = nodes.get(nid)
            if nid not in commits:
                parent = nodes.get(n.parent) if n and n.parent is not None else None
                shas = (_git(repo, "rev-list", "--reverse", f"{parent.sha}..{n.sha}").split()
                        if n and n.sha and parent and parent.sha else [])
                commits[nid] = deque(shas)
            files = {}
            if commits[nid]:
                c = commits[nid].popleft()
                for line in _git(repo, "diff", "--name-status", "--no-renames", f"{c}^", c).splitlines():
                    status, path = line.split("\t", 1)
                    files[path] = None if status == "D" else _git(repo, "show", f"{c}:{path}")
            entry["files"] = files
        calls.append(entry)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    best = st.best()
    meta = {"benchmark": Path(st.load_config().benchmark).name, "nodes": len(nodes), "calls": len(calls),
            "best_node": best.id if best else None, "best_score": best.score if best else None,
            "recorded_cost_usd": round(st.total_cost(), 4),
            "kinds": [n.kind for n in nodes.values()]}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


class ReplayAgent:
    def __init__(self, fixture: Path):
        self.queues: dict[str, deque] = defaultdict(deque)
        for line in (Path(fixture) / "calls.jsonl").read_text().splitlines():
            c = json.loads(line)
            self.queues[c["role"]].append(c)

    def __call__(self, call, log_dir) -> AgentResult:
        q = self.queues.get(call.role)
        if not q:
            return AgentResult(role=call.role, ok=False, error=f"replay: no recorded {call.role} call left")
        c = q.popleft()
        if call.role in CODE_ROLES:
            for path, content in (c.get("files") or {}).items():
                target = Path(call.cwd) / path
                if content is None:
                    target.unlink(missing_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content)
        out = c.get("output")
        return AgentResult(role=call.role, ok=out is not None, output=out, text=json.dumps(out), cost_usd=0.0,
                           error=None if out is not None else "replay: recorded call had no output")


_AGENT: ReplayAgent | None = None


def agent(call, log_dir) -> AgentResult:
    """Entry point for MARBLE_FAKE_AGENT=marble.replay:agent (fixture from $MARBLE_REPLAY_FIXTURE)."""
    global _AGENT
    if _AGENT is None:
        _AGENT = ReplayAgent(Path(os.environ["MARBLE_REPLAY_FIXTURE"]))
    return _AGENT(call, log_dir)
