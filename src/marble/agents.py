"""Run one role agent as an isolated, headless Claude Code worker (`claude -p`).

Isolation per call:
  * fresh session, no session persistence, no user/project settings, no MCP servers, no slash commands
  * explicit tool list (--tools) and allowlist; permission mode `dontAsk` denies anything else
  * PreToolUse hook (marble.hooks guard) blocks edits outside the node's editable globs
  * per-call $ cap (--max-budget-usd) and wall-clock timeout (whole process group killed)
  * structured output validated against a JSON schema (--json-schema)
Every call's prompt, system prompt and raw JSON response are written to the node's directory.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

READ_TOOLS = ["Read", "Grep", "Glob"]
EDIT_TOOLS = ["Read", "Grep", "Glob", "Edit", "Write", "Bash"]


@dataclass
class AgentCall:
    role: str
    model: str
    prompt: str
    system: str
    cwd: Path
    tools: list[str] = field(default_factory=lambda: list(READ_TOOLS))
    schema: dict | None = None
    max_budget_usd: float = 1.0
    timeout_s: float = 600.0
    editable: list[str] | None = None   # set => install the edit guard for this worktree
    env: dict[str, str] = field(default_factory=dict)
    add_dirs: list[Path] = field(default_factory=list)


@dataclass
class AgentResult:
    role: str
    ok: bool
    output: Any = None          # structured output (dict) if a schema was given, else text
    text: str = ""
    cost_usd: float = 0.0
    duration_s: float = 0.0
    num_turns: int = 0
    error: str | None = None
    log_path: str = ""


def claude_binary() -> str:
    exe = os.environ.get("MARBLE_CLAUDE") or shutil.which("claude")
    if not exe:
        raise RuntimeError("claude CLI not found (install Claude Code or set MARBLE_CLAUDE)")
    return exe


def _settings(call: AgentCall) -> dict:
    s: dict = {"includeCoAuthoredBy": False}
    if call.editable is not None:
        guard = f"{shlex.quote(sys.executable)} -m marble.hooks guard"
        s["hooks"] = {"PreToolUse": [{"matcher": "Edit|Write|MultiEdit|NotebookEdit",
                                      "hooks": [{"type": "command", "command": guard, "timeout": 30}]}]}
    return s


def build_command(call: AgentCall) -> list[str]:
    cmd = [
        claude_binary(), "-p",
        "--model", call.model,
        "--output-format", "json",
        "--no-session-persistence",
        "--strict-mcp-config",
        "--setting-sources", "",
        "--disable-slash-commands",
        "--permission-mode", "dontAsk",
        "--tools", ",".join(call.tools) if call.tools else "",
        "--max-budget-usd", f"{max(call.max_budget_usd, 0.01):.2f}",
        "--append-system-prompt", call.system,
        "--settings", json.dumps(_settings(call)),
    ]
    if call.tools:
        cmd += ["--allowedTools", ",".join(call.tools)]
    if call.schema is not None:
        cmd += ["--json-schema", json.dumps(call.schema)]
    for d in call.add_dirs:
        cmd += ["--add-dir", str(d)]
    return cmd


def run_agent(call: AgentCall, log_dir: Path) -> AgentResult:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{call.role}-{int(time.time() * 1000) % 10_000_000:07d}"
    (log_dir / f"{stem}.prompt.md").write_text(f"# system (appended)\n\n{call.system}\n\n# prompt\n\n{call.prompt}\n")
    env = dict(os.environ)
    env.update(call.env)
    # The guard hook runs `python -m marble.hooks`: make sure this package is importable.
    src_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = src_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if call.editable is not None:
        env["MARBLE_WORKTREE"] = str(Path(call.cwd).resolve())
        env["MARBLE_EDITABLE"] = json.dumps(call.editable)
    t0 = time.monotonic()
    raw_path = log_dir / f"{stem}.json"
    try:
        proc = subprocess.Popen(build_command(call), cwd=str(call.cwd), env=env, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            out, err = proc.communicate(call.prompt, timeout=call.timeout_s)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            out, err = proc.communicate()
            raw_path.write_text(out or "")
            return AgentResult(role=call.role, ok=False, error=f"agent timed out after {call.timeout_s:.0f}s",
                               duration_s=time.monotonic() - t0, log_path=str(raw_path))
    except OSError as e:
        return AgentResult(role=call.role, ok=False, error=f"cannot start claude: {e}")
    raw_path.write_text(out or "")
    if err:
        (log_dir / f"{stem}.stderr.txt").write_text(err)
    dur = time.monotonic() - t0
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return AgentResult(role=call.role, ok=False, duration_s=dur, log_path=str(raw_path),
                           error=f"unparseable agent output (exit {proc.returncode}): {(out or err)[-1500:]}")
    res = AgentResult(
        role=call.role, ok=not data.get("is_error", False) and data.get("subtype") == "success",
        text=data.get("result") or "", cost_usd=float(data.get("total_cost_usd") or 0.0),
        duration_s=dur, num_turns=int(data.get("num_turns") or 0), log_path=str(raw_path),
    )
    if call.schema is not None:
        res.output = data.get("structured_output")
        if res.output is None and res.ok:
            res.ok, res.error = False, "agent finished without structured output"
    else:
        res.output = res.text
    if not res.ok and res.error is None:
        res.error = f"agent ended with {data.get('subtype')}: {str(data.get('result') or '')[:800]}"
    return res


def _kill_group(proc: subprocess.Popen) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            continue
