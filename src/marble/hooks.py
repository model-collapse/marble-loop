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


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["guard"]:
        return guard()
    print("usage: python -m marble.hooks guard", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
