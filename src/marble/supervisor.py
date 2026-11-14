"""Supervisor: keep a run alive across controller crashes or hangs (DESIGN.md §6.2, last row).

Runs the controller as a child process (`marble run resume <run>`) and restarts it when
  * it exits with a non-zero code (crash, OOM-kill, ...), or
  * its heartbeat file is older than `heartbeat_stale_s` (process frozen), or
  * events.jsonl has not changed for `progress_stale_s` (alive but making no progress).
Exit code 0 from the child means the run stopped normally (budget, iterations, ...): done.
Every restart is recorded as an incident; the controller's own recovery closes interrupted nodes.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from .inspector import Incident, IncidentLog
from .runstate import RunState


def _age(path: Path) -> float | None:
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return None


def _kill(p: subprocess.Popen) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:
            return
        try:
            p.wait(timeout=15)
            return
        except subprocess.TimeoutExpired:
            continue


def supervise(run_dir: Path, child_args: list[str] | None = None, *, max_restarts: int = 3,
              heartbeat_stale_s: float = 120.0, progress_stale_s: float = 3 * 3600.0, poll_s: float = 5.0,
              env: dict[str, str] | None = None, log: Callable[[str], None] = print) -> int:
    run_dir = Path(run_dir).resolve()
    state = RunState(run_dir)
    incidents = IncidentLog(run_dir, state.event)
    hb, events = run_dir / "heartbeat", run_dir / "events.jsonl"
    restarts = 0
    while True:
        cmd = [sys.executable, "-m", "marble.cli", "run", "resume", str(run_dir), *(child_args or [])]
        child_env = dict(os.environ)
        child_env.update(env or {})
        with open(run_dir / "controller.log", "ab") as out:
            p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True, env=child_env)
        state.event("supervisor_start", pid=p.pid, restart=restarts)
        log(f"supervisor: controller pid {p.pid} (restart {restarts}/{max_restarts})")
        t0, why = time.time(), None
        while True:
            rc = p.poll()
            if rc is not None:
                why = None if rc == 0 else f"controller exited with code {rc}"
                break
            up = time.time() - t0
            hb_age, ev_age = _age(hb), _age(events)
            if up > heartbeat_stale_s and (hb_age is None or hb_age > heartbeat_stale_s):
                why = f"heartbeat stale ({hb_age if hb_age is None else round(hb_age)} s)"
            elif up > progress_stale_s and ev_age is not None and ev_age > progress_stale_s:
                why = f"no progress: events.jsonl unchanged for {ev_age:.0f} s"
            if why:
                _kill(p)
                break
            time.sleep(poll_s)
        if why is None:
            state.event("supervisor_done", restarts=restarts)
            log("supervisor: run finished")
            return 0
        restarts += 1
        incidents.add(Incident("controller_restart", None, why,
                               action="restart" if restarts <= max_restarts else "give up"))
        log(f"supervisor: {why}")
        if restarts > max_restarts:
            state.event("supervisor_gave_up", restarts=restarts - 1, reason=why)
            log("supervisor: too many restarts; giving up")
            return 1
        time.sleep(min(30.0, 2.0 * restarts))
