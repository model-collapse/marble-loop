"""Scripted fake agents usable in-process (monkeypatch) or in a subprocess (MARBLE_FAKE_AGENT)."""

import json
import os
from pathlib import Path

from marble.agents import AgentResult

PLAN = {"hypotheses": [{"id": "h1", "family": "fix", "idea": "rewrite the solution", "expected_gain": "large"}],
        "chosen": "h1", "plan": "rewrite solution.py", "files": ["solution.py"], "cites": []}
LESSON = {"title": "t", "change": "c", "outcome": "improved", "cause": "success", "evidence": "e",
          "key_lesson": "k", "applies_when": "a"}


class Scripted:
    """implementer/debugger outputs are popped from `codes` (written to `target`); others are canned.

    `plans` (optional) is a list of planner outputs used in order (last one repeats).
    `inspector` (optional) is the inspector verdict dict to return.
    """

    def __init__(self, codes, target="solution.py", plans=None, inspector=None, on_call=None):
        self.codes, self.target = list(codes), target
        self.plans = list(plans or [PLAN])
        self.inspector = inspector or {"diagnosis": "looks hung", "action": "kill_and_debug", "lesson": ""}
        self.calls, self.prompts = [], []
        self.on_call = on_call

    def __call__(self, call, log_dir):
        self.calls.append(call.role)
        self.prompts.append((call.role, call.prompt))
        if self.on_call:
            self.on_call(call)
        if call.role in ("implementer", "debugger"):
            if self.codes:
                (Path(call.cwd) / self.target).write_text(self.codes.pop(0))
            out = ({"summary": "s", "files_changed": [self.target], "self_check": "-"} if call.role == "implementer"
                   else {"diagnosis": "d", "fix": "f", "files_changed": [self.target]})
        elif call.role == "planner":
            out = self.plans.pop(0) if len(self.plans) > 1 else self.plans[0]
        elif call.role == "analyst":
            out = {"summary": "s", "bottlenecks": [], "opportunities": []}
        elif call.role == "reflector":
            out = LESSON
        elif call.role == "inspector":
            out = self.inspector
        else:
            out = {}
        return AgentResult(role=call.role, ok=True, output=out, text=json.dumps(out), cost_usd=0.01, duration_s=0.01)


# ---- subprocess entry point (MARBLE_FAKE_AGENT=tests.fakes:subprocess_agent) -------------------------------

_SUB = None


def subprocess_agent(call, log_dir):
    """Reads codes from $MARBLE_FAKE_CODES (JSON list). If $MARBLE_FAKE_CRASH_MARKER does not exist yet, the first
    implementer call creates it and kills the controller process (simulates a crash mid-iteration)."""
    global _SUB
    if _SUB is None:
        _SUB = Scripted(json.loads(os.environ["MARBLE_FAKE_CODES"]), target=os.environ.get("MARBLE_FAKE_TARGET", "model.py"))
    marker = os.environ.get("MARBLE_FAKE_CRASH_MARKER")
    if call.role == "implementer" and marker and not Path(marker).exists():
        Path(marker).write_text("crashed once")
        os._exit(137)
    return _SUB(call, log_dir)
