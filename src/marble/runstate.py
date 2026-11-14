"""Run state on disk: node table (SQLite), append-only event log, cost ledger, config.

Layout of a run directory:
  config.json      run configuration (budget, models, ...)
  tree.sqlite      nodes: one row per candidate
  events.jsonl     append-only log of every step (source of truth for replay/audit)
  repo/            git repo (workspace.py); nodes/<id>/ per-node artefacts
  lessons.jsonl    lessons written by the Reflector (M1: flat list; lifecycle comes in M3)
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
  id INTEGER PRIMARY KEY,
  parent INTEGER,
  sha TEXT,
  kind TEXT,            -- root | improve | debug
  status TEXT,          -- created | evaluated | buggy | no_change | aborted
  eval_status TEXT,     -- ok | incorrect | failed | timeout | ... (marble.evaluate)
  score REAL,
  score_std REAL,
  score_rel_std REAL,
  n_ok INTEGER,
  full_score REAL,      -- confirmation eval (e.g. 'full' mode) for candidate bests
  full_status TEXT,
  family TEXT,          -- normalised approach family of the chosen hypothesis
  plan TEXT,            -- planner summary (JSON)
  summary TEXT,         -- implementer/debugger summary
  error TEXT,
  is_best INTEGER DEFAULT 0,
  improved INTEGER DEFAULT 0,
  cost_usd REAL DEFAULT 0,
  eval_seconds REAL DEFAULT 0,
  created REAL,
  finished REAL
);
"""


@dataclass
class Node:
    id: int
    parent: int | None
    sha: str | None
    kind: str
    status: str = "created"
    eval_status: str | None = None
    score: float | None = None
    score_std: float | None = None
    score_rel_std: float | None = None
    n_ok: int | None = None
    full_score: float | None = None
    full_status: str | None = None
    family: str | None = None
    plan: str | None = None
    summary: str | None = None
    error: str | None = None
    is_best: int = 0
    improved: int = 0
    cost_usd: float = 0.0
    eval_seconds: float = 0.0
    created: float = field(default_factory=time.time)
    finished: float | None = None


_NODE_FIELDS = [f.name for f in fields(Node)]


@dataclass
class RunConfig:
    benchmark: str
    budget_usd: float = 20.0
    wall_hours: float = 2.0
    max_iterations: int = 20
    max_no_improve: int = 6
    repeats: int | None = None          # None = spec.noise.repeats
    debug_attempts: int = 3
    eval_mode: str | None = None        # None = spec.default_mode
    confirm_mode: str | None = "auto"   # re-check quick-mode bests here; "auto" = 'full' if defined; None = off
    confirm_repeats: int = 1
    analyze: bool = True
    analyze_timeout: float | None = None
    models: dict[str, str] = field(default_factory=lambda: {
        "analyst": "us.anthropic.claude-sonnet-5-5",
        "planner": "us.anthropic.claude-opus-5-5",
        "implementer": "us.anthropic.claude-sonnet-5-5",
        "debugger": "us.anthropic.claude-sonnet-5-5",
        "reflector": "us.anthropic.claude-haiku-5-5",
        "inspector": "us.anthropic.claude-haiku-5-5",
    })
    role_budget_usd: dict[str, float] = field(default_factory=lambda: {
        "analyst": 1.0, "planner": 1.5, "implementer": 3.0, "debugger": 2.0, "reflector": 0.3, "inspector": 0.15,
    })
    role_timeout_s: dict[str, float] = field(default_factory=lambda: {
        "analyst": 600, "planner": 600, "implementer": 1500, "debugger": 1200, "reflector": 300,
    })
    lessons_in_prompt: int = 12
    inspector: str = "llm"              # llm | rule | off  (job watchdog investigations)
    stall_s: float = 180.0              # no output + no CPU + idle GPU this long = stalled
    repeat_failure_limit: int = 2       # same failure signature this many times in a row -> stop debugging
    plan_similarity_max: float = 0.8
    max_smoke_s: float = 120.0
    search: str = "mcts"                # mcts | greedy
    c_uct: float = 1.0
    max_children: int = 2               # N_i: improve attempts per valid node
    stagnation_n: int = 3               # n_s: re-open the root (draft) after this many candidates without a new best
    time_penalty_w: float = -0.07       # w in R = G * (t/L)^w
    seed: int = 0
    lesson_test_every: int = 5          # every n-th iteration tests one unverified positive lesson (0 = off)          # agents may run the evaluator as a smoke test only if it is this fast    # re-plan once if the chosen idea is this similar to an earlier one

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        base = cls(benchmark=d["benchmark"])
        for k, v in d.items():
            if k in ("models", "role_budget_usd", "role_timeout_s") and isinstance(v, dict):
                getattr(base, k).update(v)
            elif hasattr(base, k):
                setattr(base, k, v)
        return base


class RunState:
    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir).resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(str(self.dir / "tree.sqlite"), check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    # ------------------------------------------------------------------ config
    def save_config(self, cfg: RunConfig) -> None:
        (self.dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    def load_config(self) -> RunConfig:
        return RunConfig.from_dict(json.loads((self.dir / "config.json").read_text()))

    # ------------------------------------------------------------------ events
    def event(self, type_: str, **data: Any) -> None:
        rec = {"t": time.time(), "type": type_, **data}
        with self._lock, open(self.dir / "events.jsonl", "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def events(self) -> list[dict]:
        p = self.dir / "events.jsonl"
        if not p.exists():
            return []
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

    # ------------------------------------------------------------------ nodes
    def new_node(self, parent: int | None, kind: str, sha: str | None = None) -> Node:
        with self._lock:
            cur = self.db.execute(
                "INSERT INTO nodes (parent, sha, kind, status, created) VALUES (?, ?, ?, 'created', ?)",
                (parent, sha, kind, time.time()))
            nid = cur.lastrowid
        n = self.get(nid)
        self.event("node_created", node=nid, parent=parent, kind=kind)
        return n

    def update(self, node: Node) -> None:
        cols = [c for c in _NODE_FIELDS if c != "id"]
        with self._lock:
            self.db.execute(f"UPDATE nodes SET {', '.join(f'{c}=?' for c in cols)} WHERE id=?",
                            [getattr(node, c) for c in cols] + [node.id])

    def get(self, nid: int) -> Node:
        row = self.db.execute(f"SELECT {', '.join(_NODE_FIELDS)} FROM nodes WHERE id=?", (nid,)).fetchone()
        if row is None:
            raise KeyError(nid)
        return Node(**dict(zip(_NODE_FIELDS, row)))

    def nodes(self) -> list[Node]:
        rows = self.db.execute(f"SELECT {', '.join(_NODE_FIELDS)} FROM nodes ORDER BY id").fetchall()
        return [Node(**dict(zip(_NODE_FIELDS, r))) for r in rows]

    def best(self) -> Node | None:
        row = self.db.execute(f"SELECT {', '.join(_NODE_FIELDS)} FROM nodes WHERE is_best=1 ORDER BY id DESC LIMIT 1").fetchone()
        return Node(**dict(zip(_NODE_FIELDS, row))) if row else None

    def set_best(self, node: Node) -> None:
        with self._lock:
            self.db.execute("UPDATE nodes SET is_best=0")
            self.db.execute("UPDATE nodes SET is_best=1 WHERE id=?", (node.id,))
        node.is_best = 1
        self.event("new_best", node=node.id, score=node.score)

    # ------------------------------------------------------------------ cost
    def add_cost(self, role: str, usd: float, node: int | None = None, **extra: Any) -> None:
        self.event("cost", role=role, usd=usd, node=node, **extra)

    def total_cost(self) -> float:
        return sum(e.get("usd", 0.0) for e in self.events() if e["type"] == "cost")

    # ------------------------------------------------------------------ lessons
    def add_lesson(self, lesson: dict) -> dict:
        p = self.dir / "lessons.jsonl"
        n = sum(1 for _ in open(p)) if p.exists() else 0
        lesson = {"id": f"L{n + 1:04d}", "created": time.time(), **lesson}
        with self._lock, open(p, "a") as f:
            f.write(json.dumps(lesson, default=str) + "\n")
        self.event("lesson", id=lesson["id"], node=lesson.get("node"))
        return lesson

    def lessons(self) -> list[dict]:
        """Lessons that passed admission (lesson store table if present; legacy lessons.jsonl otherwise)."""
        try:
            from .lessons import FIELDS
            rows = self.db.execute(f"SELECT {', '.join(FIELDS)} FROM lessons WHERE status != 'rejected' ORDER BY created").fetchall()
            return [dict(zip(FIELDS, r)) for r in rows]
        except sqlite3.OperationalError:
            pass
        p = self.dir / "lessons.jsonl"
        if not p.exists():
            return []
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
