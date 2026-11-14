"""Run report (report.md) and run control (control.json).

The report is generated from on-disk state only, so it can be rebuilt at any time
(`marble run report <run>`, the MCP server, or the controller when it stops).

control.json lets an interactive session steer a running controller (checked before every iteration):
  {"action": "stop"}                 stop after the current iteration
  {"action": "pause"}                wait (heartbeat continues) until the action changes
  {"action": "run"}                  resume after a pause
  {"notes": ["..."]}                 human hints added to every planner prompt (pinned)
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .inspector import IncidentLog
from .lessons import LessonStore
from .runstate import Node, RunState


# --------------------------------------------------------------------------- control


def read_control(run_dir: Path) -> dict[str, Any]:
    p = Path(run_dir) / "control.json"
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_control(run_dir: Path, *, action: str | None = None, note: str | None = None,
                  clear_notes: bool = False) -> dict[str, Any]:
    if action not in (None, "stop", "pause", "run"):
        raise ValueError("action must be one of stop | pause | run")
    data = read_control(run_dir)
    if action:
        data["action"] = action
    notes = [] if clear_notes else list(data.get("notes") or [])
    if note:
        notes.append(str(note)[:1000])
    data["notes"] = notes[-10:]
    data["updated"] = time.time()
    p = Path(run_dir) / "control.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(p)
    return data


# --------------------------------------------------------------------------- summary


def run_summary(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    st = RunState(run_dir)
    cfg = st.load_config()
    nodes = st.nodes()
    best = st.best()
    ev = st.events()
    hb = run_dir / "heartbeat"
    hb_age = time.time() - hb.stat().st_mtime if hb.exists() else None
    stopped = [e for e in ev if e["type"] == "run_stopped"]
    started = [e for e in ev if e["type"] in ("node_created", "supervisor_start")]
    alive = hb_age is not None and hb_age < 60 and (not stopped or (started and started[-1]["t"] > stopped[-1]["t"]))
    root = nodes[0] if nodes else None
    return {
        "run": str(run_dir), "benchmark": Path(cfg.benchmark).name, "alive": bool(alive),
        "heartbeat_age_s": None if hb_age is None else round(hb_age),
        "cost_usd": round(st.total_cost(), 3), "budget_usd": cfg.budget_usd, "nodes": len(nodes),
        "baseline": root.score if root else None,
        "best": {"node": best.id, "score": best.score, "full_score": best.full_score} if best else None,
        "last_stop": stopped[-1].get("reason") if stopped else None,
        "control": read_control(run_dir),
        "search": cfg.search,
    }


def node_detail(run_dir: Path, nid: int) -> dict[str, Any]:
    st = RunState(run_dir)
    n = st.get(nid)
    d = Path(run_dir) / "nodes" / f"{nid:04d}"
    verdicts = [e for e in st.events() if e["type"] == "verdict" and e["node"] == nid]
    evals = [e for e in st.events() if e["type"] == "eval" and e["node"] == nid]
    lessons = [l for l in LessonStore(st.db).all() if l["node"] == nid]
    out = {k: getattr(n, k) for k in ("id", "parent", "kind", "status", "eval_status", "score", "score_std",
                                       "full_score", "full_status", "family", "summary", "error", "is_best",
                                       "improved", "cost_usd")}
    out["plan"] = json.loads(n.plan) if n.plan else None
    out["verdict"] = verdicts[-1]["verdict"] if verdicts else None
    out["evals"] = [{k: e.get(k) for k in ("tag", "mode", "status", "score_mean", "score_std", "duration_mean_s")} for e in evals]
    out["diffstat"] = (d / "diffstat.txt").read_text() if (d / "diffstat.txt").exists() else None
    out["lessons"] = [{k: l[k] for k in ("id", "type", "status", "title", "claim")} for l in lessons]
    return out


# --------------------------------------------------------------------------- report


def _idea(n: Node) -> str:
    if not n.plan:
        return ""
    try:
        p = json.loads(n.plan)
    except json.JSONDecodeError:
        return ""
    ch = next((h for h in p.get("hypotheses", []) if h.get("id") == p.get("chosen")), {})
    return f"[{ch.get('family', '')}] {ch.get('idea', '')}"


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.6g}"


def score_key_of(benchmark: str) -> str:
    """Score key from benchmark.yaml without env interpolation (works without KERNELBENCH_ROOT etc.)."""
    import yaml
    p = Path(benchmark)
    p = p / "benchmark.yaml" if p.is_dir() else p
    try:
        return str(yaml.safe_load(p.read_text())["score"]["key"])
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        return "score"


def build_report(run_dir: Path, score_key: str | None = None, stop: str | None = None) -> str:
    run_dir = Path(run_dir).resolve()
    st = RunState(run_dir)
    cfg = st.load_config()
    score_key = score_key or score_key_of(cfg.benchmark)
    nodes = st.nodes()
    if not nodes:
        return f"# MARBLE run report\n\n(no nodes yet in {run_dir})\n"
    ev = st.events()
    root, best = nodes[0], st.best()
    lessons = LessonStore(st.db)
    incs = IncidentLog(run_dir).all()
    stop = stop or next((e.get("reason") for e in reversed(ev) if e["type"] == "run_stopped"), None)
    t0, t1 = ev[0]["t"], ev[-1]["t"]
    by_role: dict[str, float] = defaultdict(float)
    for e in ev:
        if e["type"] == "cost":
            by_role[e["role"]] += e.get("usd", 0.0)
    L = [f"# MARBLE run report: {Path(cfg.benchmark).name}", "",
         f"- run: `{run_dir}`", f"- search: {cfg.search}; stop reason: {stop}",
         f"- wall time: {(t1 - t0) / 60:.0f} min; cost: ${st.total_cost():.2f} of ${cfg.budget_usd:.2f} "
         f"({', '.join(f'{k} ${v:.2f}' for k, v in sorted(by_role.items(), key=lambda kv: -kv[1]))})",
         f"- nodes: {len(nodes)}; new bests: {sum(1 for e in ev if e['type'] == 'new_best') - 1}",
         f"- baseline: {score_key} = {_fmt(root.score)}",
         f"- **best: node #{best.id}, {score_key} = {_fmt(best.score)}**"
         + (f" (confirmation eval: {_fmt(best.full_score)})" if best.full_score is not None else ""), ""]
    # best path
    path, cur = [], best
    while cur is not None:
        path.append(cur)
        cur = st.get(cur.parent) if cur.parent is not None else None
    L += ["## Path to the best", ""]
    for n in reversed(path):
        L.append(f"1. #{n.id} {n.kind}: {_fmt(n.score)} {_idea(n)[:140]}")
    # timeline of bests
    bests = [e for e in ev if e["type"] == "new_best"]
    if len(bests) > 1:
        L += ["", "## Best score over time", "", "| min | node | " + score_key + " |", "|---|---|---|"]
        for e in bests:
            L.append(f"| {(e['t'] - t0) / 60:.0f} | #{e['node']} | {_fmt(e.get('score'))} |")
    # all nodes
    L += ["", "## Nodes", "", "| node | parent | kind | status | eval | score | confirm | beat parent | cost $ | idea |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n in nodes:
        conf = "" if not n.full_status else (_fmt(n.full_score) if n.full_status == "ok" else n.full_status)
        L.append(f"| {n.id}{' ★' if n.is_best else ''} | {n.parent if n.parent is not None else '-'} | {n.kind} | "
                 f"{n.status} | {n.eval_status or '-'} | {_fmt(n.score)} | {conf} | {'yes' if n.improved else ''} | "
                 f"{n.cost_usd:.2f} | {_idea(n)[:90].replace('|', '/')} |")
    if best.id != root.id and (run_dir / "best.patch").exists():
        L += ["", "Best change vs baseline: `best.patch`"]
    # lessons
    ls = [l for l in lessons.all() if l["status"] != "observation"]
    if ls:
        L += ["", "## Lessons", "", "| id | type | status | trust | node | title |", "|---|---|---|---|---|---|"]
        for l in ls:
            p = l["trust_a"] / (l["trust_a"] + l["trust_b"])
            L.append(f"| {l['id']} | {l['type']} | {l['status']} | {p:.2f} | {l['node']} | {str(l['title'])[:100].replace('|', '/')} |")
        L.append(f"\n({sum(1 for l in lessons.all() if l['status'] == 'observation')} observations within noise not shown)")
    if incs:
        L += ["", f"## Inspector incidents ({len(incs)})", "", "| node | kind | action | detail |", "|---|---|---|---|"]
        for i in incs:
            L.append(f"| {i.get('node') or '-'} | {i['kind']} | {i['action']} | {str(i['detail'])[:160].replace('|', '/')} |")
    notes = read_control(run_dir).get("notes") or []
    if notes:
        L += ["", "## Human notes (control.json)", ""] + [f"- {n}" for n in notes]
    return "\n".join(L) + "\n"


def write_report(run_dir: Path, score_key: str | None = None, stop: str | None = None) -> Path:
    p = Path(run_dir) / "report.md"
    p.write_text(build_report(run_dir, score_key, stop))
    return p
