"""Claude Code hooks used by MARBLE worker agents.

`python -m marble.hooks guard` is installed as a PreToolUse hook for Edit/Write/MultiEdit/NotebookEdit.
It reads the hook payload on stdin and blocks (exit code 2, reason on stderr) any edit whose target
is outside the node worktree or does not match the benchmark's editable globs.
Configuration comes from the environment set by marble.agents:
  MARBLE_WORKTREE  absolute path of the node worktree
  MARBLE_EDITABLE  JSON list of editable globs (relative to the worktree)
Edits made through Bash are not intercepted here; the controller reverts any non-editable change
before committing (workspace.enforce_editable) and the clean room rejects them at evaluation.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .cleanroom import match_any

EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}


def check_edit(payload: dict, worktree: str, editable: list[str]) -> str | None:
    """Return a denial reason, or None if the edit is allowed."""
    if payload.get("tool_name") not in EDIT_TOOLS:
        return None
    ti = payload.get("tool_input") or {}
    target = ti.get("file_path") or ti.get("notebook_path")
    if not target:
        return "edit without a file path"
    root = Path(worktree).resolve()
    p = Path(target)
    if not p.is_absolute():
        p = Path(payload.get("cwd") or root) / p
    p = p.resolve()
    try:
        rel = p.relative_to(root).as_posix()
    except ValueError:
        return f"{target} is outside the node workspace {root}; only files in the workspace may be edited"
    if not match_any(rel, editable):
        return (f"{rel} is not editable. Editable files (globs): {editable}. Evaluator and protected files "
                f"must not be changed; put new code in an editable location.")
    return None


def guard() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        print("MARBLE guard: invalid hook payload", file=sys.stderr)
        return 2
    worktree = os.environ.get("MARBLE_WORKTREE")
    try:
        editable = json.loads(os.environ.get("MARBLE_EDITABLE", "[]"))
    except json.JSONDecodeError:
        editable = []
    if not worktree:
        return 0  # not running under MARBLE: allow
    reason = check_edit(payload, worktree, editable)
    if reason:
        print(f"MARBLE: edit blocked: {reason}", file=sys.stderr)
        return 2
    return 0


# --------------------------------------------------------------------------- plugin hooks (interactive sessions)


def _runs_root() -> Path:
    if os.environ.get("MARBLE_RUNS_ROOT"):
        return Path(os.environ["MARBLE_RUNS_ROOT"])
    base = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    return Path(base) / "runs"


def _alive_runs(root: Path) -> list[tuple[Path, dict]]:
    from . import report
    out = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        if (d / "tree.sqlite").exists():
            try:
                s = report.run_summary(d)
            except Exception:
                continue
            out.append((d, s))
    return out


def session_start() -> int:
    """SessionStart: tell the session which MARBLE runs exist (alive first)."""
    try:
        json.load(sys.stdin)
    except json.JSONDecodeError:
        pass
    runs = _alive_runs(_runs_root())
    if not runs:
        print(json.dumps({}))
        return 0
    lines = []
    for d, s in sorted(runs, key=lambda r: (not r[1]["alive"], r[0].name))[:8]:
        best = s["best"] or {}
        lines.append(f"- {d.name}: {s['benchmark']}, {'RUNNING' if s['alive'] else 'stopped'}, best "
                     f"{best.get('score')} (baseline {s['baseline']}), ${s['cost_usd']:.2f}/{s['budget_usd']:.0f}"
                     + (f", last stop: {s['last_stop']}" if not s["alive"] and s["last_stop"] else ""))
    ctx = ("MARBLE runs in " + str(_runs_root()) + " (use the marble MCP tools or `/marble status` for details):\n"
           + "\n".join(lines))
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": ctx}}))
    return 0


def guard_runs() -> int:
    """PreToolUse (interactive): protect run state and benchmarks that have a live run."""
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    if payload.get("tool_name") not in EDIT_TOOLS:
        return 0
    ti = payload.get("tool_input") or {}
    target = ti.get("file_path") or ti.get("notebook_path")
    if not target:
        return 0
    p = Path(target)
    if not p.is_absolute():
        p = Path(payload.get("cwd") or os.getcwd()) / p
    p = p.resolve()
    for d, s in _alive_runs(_runs_root()):
        d = d.resolve()
        if p.is_relative_to(d) and p.name != "control.json":
            print(f"MARBLE: {p} belongs to run {d.name}; run state is managed by MARBLE. Use `marble run control` "
                  "or the marble_control MCP tool to steer a run.", file=sys.stderr)
            return 2
        if not s["alive"]:
            continue
        try:
            import yaml
            cfg = json.loads((d / "config.json").read_text())
            bench = Path(cfg["benchmark"])
            bench = bench if bench.is_dir() else bench.parent
            raw = yaml.safe_load((bench / "benchmark.yaml").read_text())
            repo = (bench / raw["repo"]).resolve()
        except Exception:
            continue
        if p.is_relative_to(repo) or p == (bench / "benchmark.yaml").resolve():
            print(f"MARBLE: run {d.name} is live on benchmark {bench.name}; editing its baseline or evaluator now "
                  "would invalidate the run. Stop the run first (`marble run control --stop`).", file=sys.stderr)
            return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cmds = {"guard": guard, "session-start": session_start, "guard-runs": guard_runs}
    if argv[:1] and argv[0] in cmds:
        return cmds[argv[0]]()
    print(f"usage: python -m marble.hooks {{{'|'.join(cmds)}}}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
