"""Inspector: expectations, watchdogs and loop-level detectors (DESIGN.md §6).

Job watchdog (runs inside the launcher's monitor thread via `on_sample`):
  * soft limit = max(SOFT_FACTOR x expected, expected + SOFT_MIN_EXTRA_S); the hard timeout always applies.
  * past the soft limit and still *active* (CPU, GPU or fresh output) -> one automatic extension;
  * *stalled* (no output for `stall_s`, CPU ~0, GPU ~idle) or past the extended limit -> investigation:
    evidence is collected and an Investigator (LLM, or a rule fallback) returns
    extend | kill_and_debug | kill_and_abandon. At most one LLM extension per job.
Loop-level detectors (called by the controller):
  * repeated identical failure signature  -> stop debugging that node
  * candidate code identical to an earlier node (oscillation) -> reject before evaluation
  * plan too similar to an earlier one -> ask the Planner for a different idea (once)
  * budget / wall pace -> do not start an iteration that cannot finish
Every detection is appended to incidents.jsonl and to the event log.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import psutil

SOFT_FACTOR = 1.5
SOFT_MIN_EXTRA_S = 60.0
AUTO_EXTEND_FACTOR = 2.0       # active-but-slow: soft limit becomes this x expected (once)
STALL_S = 180.0                # no output + no CPU + idle GPU for this long = stalled
STALL_CPU_PCT = 5.0
STALL_GPU_PCT = 5.0
MAX_LLM_EXTENSION_S = 1800.0


# --------------------------------------------------------------------------- expectations


def expected_from_events(events: list[dict], kind: str, key: str, min_samples: int = 3,
                         pct: float = 0.9) -> float | None:
    """Rolling percentile of past durations, e.g. kind='cost' key=role for agents."""
    xs = [e["seconds"] for e in events if e.get("type") == kind and e.get("role", e.get("tool")) == key
          and isinstance(e.get("seconds"), (int, float))]
    if len(xs) < min_samples:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(pct * len(xs)))]


# --------------------------------------------------------------------------- incidents


@dataclass
class Incident:
    kind: str                 # slow | stalled | killed | extended | repeated_failure | duplicate_code | ...
    node: int | None
    detail: str
    action: str = "none"
    evidence: dict = field(default_factory=dict)
    t: float = field(default_factory=time.time)


class IncidentLog:
    def __init__(self, run_dir: Path, event_fn: Callable[..., None] | None = None):
        self.path = Path(run_dir) / "incidents.jsonl"
        self._event = event_fn
        self._lock = threading.Lock()

    def add(self, inc: Incident) -> Incident:
        rec = asdict(inc)
        with self._lock, open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        if self._event:
            self._event("incident", kind=inc.kind, node=inc.node, detail=inc.detail, action=inc.action)
        return inc

    def all(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]


# --------------------------------------------------------------------------- evidence


def collect_evidence(handle, log_tail_chars: int = 4000) -> dict[str, Any]:
    """Unprivileged evidence about a running job (no ptrace needed)."""
    ev: dict[str, Any] = {"elapsed_s": round(handle.elapsed(), 1), "timeout_s": handle.timeout}
    procs = []
    for p in handle._tree():
        try:
            with p.oneshot():
                wchan = ""
                try:
                    wchan = Path(f"/proc/{p.pid}/wchan").read_text().strip()
                except OSError:
                    pass
                procs.append({"pid": p.pid, "cmd": " ".join(p.cmdline())[:200], "status": p.status(),
                              "threads": p.num_threads(), "rss_mb": round(p.memory_info().rss / 2**20, 1),
                              "cpu_s": round(sum(p.cpu_times()[:2]), 1), "wchan": wchan})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    ev["processes"] = procs
    recent = list(handle.samples)[-12:]
    ev["recent_samples"] = [{"t": round(s.t, 1), "cpu%": round(s.cpu_percent, 1), "rss_mb": round(s.rss_bytes / 2**20),
                             "since_output_s": round(s.seconds_since_output, 1), "gpu%": s.gpu_util} for s in recent]
    try:
        txt = handle.log_path.read_text(errors="replace")
        ev["log_tail"] = txt[-log_tail_chars:]
        ev["log_bytes"] = len(txt)
    except OSError:
        ev["log_tail"] = ""
    return ev


# --------------------------------------------------------------------------- investigators


@dataclass
class Verdict:
    action: str                # extend | kill_and_debug | kill_and_abandon
    diagnosis: str
    extend_by_s: float = 0.0
    lesson: str = ""


INSPECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string"},
        "action": {"type": "string", "enum": ["extend", "kill_and_debug", "kill_and_abandon"]},
        "extend_by_s": {"type": "number", "description": "seconds to extend (only for extend)"},
        "lesson": {"type": "string", "description": "a reusable inspection lesson, or empty"},
    },
    "required": ["diagnosis", "action"],
}


def rule_investigator(evidence: dict, context: dict) -> Verdict:
    """Deterministic fallback: extend if clearly working, otherwise kill and let the Debugger look."""
    samples = evidence.get("recent_samples", [])
    busy = any((s["cpu%"] or 0) > 20 or any((g or 0) > 20 for g in (s.get("gpu%") or [])) for s in samples)
    fresh = any(s["since_output_s"] < 30 for s in samples)
    if (busy or fresh) and context.get("reason") == "slow":
        return Verdict("extend", "rule: process is still active (CPU/GPU busy or producing output)",
                       extend_by_s=float(context.get("expected_s") or 300.0))
    return Verdict("kill_and_debug", f"rule: {context.get('reason')} with no sign of progress")


def make_llm_investigator(run_agent_fn, *, model: str, cwd: Path, log_dir: Path, budget_usd: float = 0.15,
                          timeout_s: float = 180.0, cost_cb: Callable[[float], None] | None = None):
    """Inspector agent: cheap model, no tools, evidence in the prompt."""
    from .agents import AgentCall

    def investigate(evidence: dict, context: dict) -> Verdict:
        prompt = (
            "A benchmark job may be stuck or far slower than expected. Decide what to do.\n\n"
            f"## Context\n{json.dumps(context, indent=2, default=str)}\n\n"
            f"## Evidence\n{json.dumps(evidence, indent=2, default=str)[:12000]}\n\n"
            "If the candidate's code is shown, read it: an obvious infinite loop, deadlock or wait that can never "
            "finish means kill, regardless of CPU/GPU load (a spinning GPU kernel keeps both GPU and the waiting "
            "CPU thread at 100%). Choose: `extend` only if there is clear evidence of progress (CPU/GPU busy doing useful work, log "
            "advancing) and give extend_by_s; `kill_and_debug` if it looks hung, deadlocked, in an infinite loop or "
            "thrashing (the Debugger will get your diagnosis); `kill_and_abandon` if the approach itself is hopeless "
            "(e.g. far too slow by design). Write a short reusable lesson if one applies.")
        call = AgentCall(role="inspector", model=model, prompt=prompt,
                         system="You are the Inspector of MARBLE, an automated research loop. You diagnose stuck jobs.",
                         cwd=cwd, tools=[], schema=INSPECTOR_SCHEMA, max_budget_usd=budget_usd, timeout_s=timeout_s)
        res = run_agent_fn(call, log_dir)
        if cost_cb:
            cost_cb(res.cost_usd)
        if not res.ok or not isinstance(res.output, dict):
            v = rule_investigator(evidence, context)
            v.diagnosis = f"(inspector unavailable: {res.error}) {v.diagnosis}"
            return v
        o = res.output
        return Verdict(action=o["action"], diagnosis=o.get("diagnosis", ""),
                       extend_by_s=float(o.get("extend_by_s") or 0.0), lesson=o.get("lesson", ""))

    return investigate


# --------------------------------------------------------------------------- job watchdog


class JobWatch:
    """`on_sample` callback for one launcher job."""

    def __init__(self, *, node: int | None, label: str, expected_s: float | None, incidents: IncidentLog,
                 investigator: Callable[[dict, dict], Verdict], stall_s: float = STALL_S,
                 context: dict | None = None):
        self.node, self.label, self.expected = node, label, expected_s
        self.context = context or {}   # e.g. the candidate's changed code, shown to the investigator
        self.incidents, self.investigator, self.stall_s = incidents, investigator, stall_s
        self.soft = (max(SOFT_FACTOR * expected_s, expected_s + SOFT_MIN_EXTRA_S) if expected_s else None)
        self.auto_extended = False
        self.llm_extended = False
        self.kill_reason: str | None = None
        self.abandon = False
        self._busy = False
        self._lock = threading.Lock()

    def __call__(self, handle, sample) -> None:
        if self.kill_reason or self._busy or not sample.alive:
            return
        reason = None
        if self._stalled(handle):
            reason = "stalled"
        elif self.soft is not None and sample.t > self.soft:
            if not self.auto_extended and self._active(handle):
                self.auto_extended = True
                old, self.soft = self.soft, max(self.soft, AUTO_EXTEND_FACTOR * self.expected + SOFT_MIN_EXTRA_S)
                self.incidents.add(Incident("slow", self.node, f"{self.label}: {sample.t:.0f}s > soft limit {old:.0f}s "
                                            f"(expected {self.expected:.0f}s); still active → extended to {self.soft:.0f}s",
                                            action="auto_extend"))
                return
            reason = "slow"
        if reason is None:
            return
        with self._lock:
            if self._busy:
                return
            self._busy = True
        threading.Thread(target=self._investigate, args=(handle, reason), daemon=True).start()

    def _active(self, handle) -> bool:
        recent = list(handle.samples)[-5:]
        return any(s.cpu_percent > 20 or s.seconds_since_output < 30 or any((g or 0) > 20 for g in (s.gpu_util or []))
                   for s in recent)

    def _stalled(self, handle) -> bool:
        recent = [s for s in handle.samples if s.t >= handle.elapsed() - self.stall_s]
        if not recent or handle.elapsed() < self.stall_s or recent[-1].seconds_since_output < self.stall_s:
            return False
        gpu_seen = [g for s in recent for g in (s.gpu_util or [])]
        return (max(s.cpu_percent for s in recent) < STALL_CPU_PCT
                and (not gpu_seen or max(gpu_seen) < STALL_GPU_PCT))

    def _investigate(self, handle, reason: str) -> None:
        try:
            ev = collect_evidence(handle)
            ctx = {"job": self.label, "node": self.node, "reason": reason, "expected_s": self.expected,
                   "soft_limit_s": self.soft, "hard_timeout_s": handle.timeout,
                   "already_extended": self.auto_extended or self.llm_extended, **self.context}
            if self.llm_extended:
                v = Verdict("kill_and_debug", "already extended once by the inspector; limit reached")
            else:
                v = self.investigator(ev, ctx)
            if v.action == "extend" and not self.llm_extended:
                self.llm_extended = True
                cap = max(3.0 * self.expected, 120.0) if self.expected else MAX_LLM_EXTENSION_S
                ext = min(max(v.extend_by_s, 60.0), cap, MAX_LLM_EXTENSION_S)
                self.soft = handle.elapsed() + ext
                self.incidents.add(Incident(reason, self.node, f"{self.label}: {v.diagnosis}", action=f"extend {ext:.0f}s",
                                            evidence={"samples": ev.get("recent_samples")}))
                return
            action = "kill_and_abandon" if v.action == "kill_and_abandon" else "kill_and_debug"
            self.kill_reason = f"Inspector ({reason}): {v.diagnosis}"
            self.incidents.add(Incident(reason, self.node, f"{self.label}: {v.diagnosis}", action=action,
                                        evidence={"samples": ev.get("recent_samples"),
                                                  "processes": ev.get("processes"), "lesson": v.lesson}))
            self.abandon = action == "kill_and_abandon"
            handle.cancel(self.kill_reason)
        finally:
            self._busy = False


# --------------------------------------------------------------------------- loop-level detectors

_NUM = re.compile(r"0x[0-9a-fA-F]+|\d+(\.\d+)?")
_PATH = re.compile(r"(/[\w.\-]+)+")


def failure_signature(error: str | None, log_tail: str = "") -> str:
    """Normalise an error (numbers, addresses, paths removed) and hash it."""
    text = (error or "")
    # Only exception lines ("ValueError: ...", "torch.OutOfMemoryError: ...", "error: ..."), not source lines,
    # so a 'fix' that changes code text but not the failure keeps the same signature.
    lines = [l.strip() for l in log_tail.splitlines()
             if re.match(r"^\s*([\w.]+(Error|Exception|Exit|Interrupt)\b|error:|fatal error)", l)]
    text += "\n" + "\n".join(lines[-3:])
    norm = _NUM.sub("N", _PATH.sub("P", text)).strip().lower()
    return hashlib.sha1(norm.encode()).hexdigest()[:12]


def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]+", s.lower()) if len(w) > 2}


def plan_similarity(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def chosen_text(plan: dict) -> str:
    ch = next((h for h in plan.get("hypotheses", []) if h.get("id") == plan.get("chosen")), {})
    return f"{ch.get('family', '')} {ch.get('idea', '')} {plan.get('plan', '')}"


@dataclass
class Pace:
    """Average cost and wall time per completed iteration."""
    iterations: int
    usd: float
    seconds: float

    def per_iter(self) -> tuple[float, float]:
        if self.iterations <= 0:
            return 0.0, 0.0
        return self.usd / self.iterations, self.seconds / self.iterations


def median_or_none(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None
