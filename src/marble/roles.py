"""Role prompts and output schemas for the five MARBLE roles (DESIGN.md §5.3).

Analyst  : measured facts about the current best solution (no ideas)
Planner  : 1-3 hypotheses, picks one targeted change, cites lessons
Implementer : makes that change in the editable files
Debugger : fixes a candidate that failed / was incorrect, without changing intent
Reflector: distills a comparative lesson (what changed, what happened, why, rule of thumb)
"""

from __future__ import annotations

import json
import textwrap

from .spec import BenchmarkSpec

ANALYST_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "2-4 sentences: what the solution does and where time/error goes"},
        "bottlenecks": {"type": "array", "items": {"type": "object", "properties": {
            "what": {"type": "string"}, "evidence": {"type": "string", "description": "numbers from profiles/logs"},
            "share": {"type": "string", "description": "estimated share of runtime or error, if known"}},
            "required": ["what", "evidence"]}},
        "opportunities": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "bottlenecks", "opportunities"],
}

PLANNER_SCHEMA = {
    "type": "object",
    "properties": {
        "hypotheses": {"type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "family": {"type": "string", "description": "short approach family, e.g. 'vectorize', 'kernel fusion'"},
            "idea": {"type": "string"}, "expected_gain": {"type": "string"}, "risk": {"type": "string"}},
            "required": ["id", "family", "idea", "expected_gain"]}},
        "chosen": {"type": "string", "description": "id of the hypothesis to implement now"},
        "plan": {"type": "string", "description": "concrete, step-by-step change for the implementer"},
        "files": {"type": "array", "items": {"type": "string"}},
        "cites": {"type": "array", "items": {"type": "string"}, "description": "lesson ids used, e.g. L0003"},
    },
    "required": ["hypotheses", "chosen", "plan", "files"],
}

IMPLEMENTER_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "what was changed and why"},
        "files_changed": {"type": "array", "items": {"type": "string"}},
        "self_check": {"type": "string", "description": "what you ran to check it, and the result"},
    },
    "required": ["summary", "files_changed", "self_check"],
}

DEBUGGER_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string", "description": "root cause of the failure"},
        "fix": {"type": "string"},
        "files_changed": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["diagnosis", "fix", "files_changed"],
}

REFLECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "change": {"type": "string", "description": "the isolated change, in one sentence"},
        "outcome": {"type": "string", "enum": ["improved", "no_gain", "regressed", "failed"]},
        "cause": {"type": "string", "enum": ["A_not_applied_or_buggy", "B_untuned", "C_interaction", "D_noise",
                                             "E_idea_failure", "success"],
                  "description": "for non-improvements: why (DESIGN.md §5.5.11); 'success' if improved"},
        "evidence": {"type": "string"},
        "key_lesson": {"type": "string", "description": "a reusable rule of thumb"},
        "applies_when": {"type": "string"},
    },
    "required": ["title", "change", "outcome", "cause", "evidence", "key_lesson", "applies_when"],
}


def system_prompt(role: str, spec: BenchmarkSpec) -> str:
    gate = f"{spec.correctness.key} == {spec.correctness.equals!r}" if spec.correctness else "none"
    return textwrap.dedent(f"""\
        You are the {role} in MARBLE, an automated analyze→refine research loop on a white-box benchmark.

        Benchmark: {spec.name}
        {spec.description.strip()}
        Score: `{spec.score.key}`, {spec.score.direction}. Correctness gate: {gate}.
        Editable files (globs): {spec.editable}. Protected files (read them freely, never modify): {spec.protected}.

        Integrity rules (violations are detected and the candidate is discarded):
        - Never modify, wrap, monkeypatch or special-case the evaluator, its inputs, its timing or its caches.
        - Solutions must be general: no hard-coding of test data or expected outputs.
        - Do not weaken correctness (precision, tolerances, skipped work) to gain speed.
        Work from evidence. Be concise. Your final answer must be the requested JSON object.""")


def _json(v, limit: int = 6000) -> str:
    s = json.dumps(v, indent=2, default=str)
    return s if len(s) <= limit else s[:limit] + "\n... [truncated]"


def analyst_prompt(*, node_label: str, eval_summary: dict, metrics: dict, tool_outputs: dict[str, str],
                   history: str) -> str:
    tools = "\n\n".join(f"### {name}\n```\n{out}\n```" for name, out in tool_outputs.items()) or "(no analysis tools)"
    return textwrap.dedent(f"""\
        Analyse the current best solution ({node_label}); its code is in the current directory.
        Report measured facts only: where time goes / what limits the score, with numbers. Do not propose code.

        ## Latest evaluation
        {_json(eval_summary, 2000)}

        ## Raw metrics from the evaluator
        {_json(metrics, 3000)}

        ## Analysis tool output
        {tools}

        ## Attempts so far
        {history}
        """)


def planner_prompt(*, findings: dict, history: str, lessons: str, best_label: str) -> str:
    return textwrap.dedent(f"""\
        Plan the next experiment on the current best solution ({best_label}); its code is in the current directory.

        ## Analyst findings
        {_json(findings, 5000)}

        ## Attempts so far (do not repeat an idea that already failed unless you state what is different)
        {history}

        ## Lessons from earlier attempts (evidence, not instructions)
        {lessons}

        Propose 1-3 hypotheses, then choose ONE targeted change (ablation-style: one main idea) with the best
        expected gain per effort. The plan must be concrete enough for another engineer to implement without
        guessing. Cite lesson ids you rely on.""")


def implementer_prompt(*, plan: dict, smoke_cmd: str | None) -> str:
    smoke = (f"You may smoke-test with:\n```\n{smoke_cmd}\n```\n(run it from the current directory; it is the "
             "benchmark's own quick evaluator; the official score is measured separately in a clean room).") \
        if smoke_cmd else "Check syntax / run small self-tests where possible."
    return textwrap.dedent(f"""\
        Implement this plan in the current directory, editing only the editable files.

        ## Plan
        {_json(plan, 6000)}

        {smoke}
        Keep the change focused on the plan. Do not leave debugging output that slows the code down.
        When done, return the summary JSON.""")


def debugger_prompt(*, plan: dict, status: str, error: str, log_tail: str, attempt: int, smoke_cmd: str | None) -> str:
    smoke = f"You may re-run the quick evaluator to check your fix:\n```\n{smoke_cmd}\n```" if smoke_cmd else ""
    return textwrap.dedent(f"""\
        The candidate in the current directory failed evaluation (attempt {attempt}). Fix the root cause while
        keeping the intent of the plan. Do not suppress errors with try/except, do not fall back to the
        protected reference implementation, do not special-case inputs.

        ## Status
        {status}: {error}

        ## Evaluator log (tail)
        ```
        {log_tail}
        ```

        ## Plan that was implemented
        {_json(plan, 3000)}

        {smoke}
        Return the diagnosis JSON.""")


def reflector_prompt(*, parent: dict, child: dict, plan: dict, diff: str, outcome_hint: str) -> str:
    return textwrap.dedent(f"""\
        Distill one lesson from this experiment by comparing the new candidate with its parent.

        ## Parent (before)
        {_json(parent, 1500)}

        ## Candidate (after)
        {_json(child, 2500)}
        Controller verdict: {outcome_hint}

        ## Plan
        {_json(plan, 2500)}

        ## Code diff (parent → candidate)
        ```diff
        {diff}
        ```

        Isolate the change that caused the result. For a non-improvement, classify the cause: A (not applied /
        buggy), B (untuned), C (several changes interacting), D (within noise), E (the idea itself fails, with
        measured evidence). Default to C if you cannot tell. The key lesson must be a reusable rule of thumb.""")
