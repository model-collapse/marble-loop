"""M1 controller: greedy analyze→plan→implement→evaluate→debug→reflect loop.

Greedy = always expand the current best node (budget-aware MCTS replaces this in M3).
One iteration:
  1. Analyst   - analyze tools are run on the best node in a clean room, the Analyst turns them into facts
                 (cached per node: re-expanding the same best node reuses its findings)
  2. Planner   - picks one targeted change, given findings, attempt history and recent lessons
  3. Implementer edits a fresh worktree of the best node; non-editable changes are reverted; commit
  4. Evaluate  - clean-room eval with repeats; on failure the Debugger gets up to `debug_attempts` tries
  5. Compare   - noise-aware: relative gain must exceed max(min_rel_improvement, 2 x rel. std)
  6. Reflector - comparative lesson (parent vs candidate, with the diff)
Stops on: iteration limit, $ budget, wall clock, or `max_no_improve` iterations without improvement.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from . import roles
from .agents import EDIT_TOOLS, READ_TOOLS, AgentCall, AgentResult, run_agent
from .cleanroom import protected_manifest
from .evaluate import EvalSummary, run_repeats, run_tool
from .launcher import LocalLauncher
from .runstate import Node, RunConfig, RunState
from .spec import BenchmarkSpec, load_spec, render, validate_spec
from .workspace import Workspace

FAILED_STATUSES = {"failed", "incorrect", "invalid_output", "timeout"}


def _full(n: Node) -> str:
    if not n.full_status:
        return ""
    return f"{n.full_score:.6g}" if n.full_status == "ok" and n.full_score is not None else n.full_status


class BudgetExhausted(Exception):
    pass


def improvement(spec: BenchmarkSpec, best: Node, cand: Node) -> tuple[bool, float, float]:
    """(is_improvement, relative_gain, threshold)."""
    if cand.score is None or best.score is None:
        return False, 0.0, 0.0
    denom = abs(best.score) or 1.0
    gain = (cand.score - best.score) / denom
    if spec.score.direction == "minimize":
        gain = -gain
    thr = max(spec.noise.min_rel_improvement, 2.0 * max(best.score_rel_std or 0.0, cand.score_rel_std or 0.0))
    return gain > thr, gain, thr


class Controller:
    def __init__(self, run_dir: Path, launcher: LocalLauncher, cfg: RunConfig | None = None,
                 log: Callable[[str], None] = print):
        self.dir = Path(run_dir).resolve()
        self.state = RunState(self.dir)
        if cfg is not None:
            self.state.save_config(cfg)
        self.cfg = self.state.load_config()
        self.spec = load_spec(self.cfg.benchmark)
        validate_spec(self.spec)
        self.ws = Workspace(self.dir)
        self.launcher = launcher
        self.log = log
        self.t_start = time.monotonic()
        self.manifest_path = self.dir / "protected_manifest.json"

    # ------------------------------------------------------------------ helpers
    def node_dir(self, nid: int) -> Path:
        d = self.dir / "nodes" / f"{nid:04d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def manifest(self) -> dict[str, str]:
        return json.loads(self.manifest_path.read_text())

    def remaining_usd(self) -> float:
        return self.cfg.budget_usd - self.state.total_cost()

    def elapsed_h(self) -> float:
        return (time.monotonic() - self.t_start) / 3600.0

    def _gpu_env(self) -> dict[str, str]:
        # Agents may smoke-test on the GPU that evals use; the loop is sequential so it is free then.
        mode = self.spec.mode(self.cfg.eval_mode)
        slot = next((s for s in self.launcher.slots if s.fits(mode.resources)), self.launcher.slots[0])
        env = {k: v for k, v in slot.env().items() if k != "MARBLE_SLOT"}
        return env

    def agent(self, role: str, node: Node, *, prompt: str, cwd: Path, tools: list[str], schema: dict,
              editable: bool = False) -> AgentResult:
        remaining = self.remaining_usd()
        if remaining < 0.05:
            raise BudgetExhausted(f"budget exhausted before {role} (remaining ${remaining:.2f})")
        call = AgentCall(
            role=role, model=self.cfg.models[role], prompt=prompt, system=roles.system_prompt(role, self.spec),
            cwd=cwd, tools=tools, schema=schema,
            max_budget_usd=min(self.cfg.role_budget_usd.get(role, 1.0), remaining),
            timeout_s=self.cfg.role_timeout_s.get(role, 600),
            editable=self.spec.editable if editable else None, env=self._gpu_env(),
        )
        self.log(f"    · {role} ({call.model.split('.')[-1]}) …")
        res = run_agent(call, self.node_dir(node.id) / "agents")
        self.state.add_cost(role, res.cost_usd, node=node.id, turns=res.num_turns, ok=res.ok,
                            seconds=round(res.duration_s, 1), error=res.error)
        node.cost_usd = (node.cost_usd or 0.0) + res.cost_usd
        self.log(f"      {'ok' if res.ok else 'FAILED'}  ${res.cost_usd:.3f}  {res.duration_s:.0f}s  {res.num_turns} turns"
                 + (f"  {res.error}" if res.error else ""))
        return res

    def confirm_mode(self) -> str | None:
        m = self.cfg.confirm_mode
        if m in (None, "", "none", "off"):
            return None
        if m == "auto":
            m = "full" if "full" in self.spec.evaluate else None
        if m is None or m == (self.cfg.eval_mode or self.spec.default_mode):
            return None
        return m

    def evaluate(self, node: Node, candidate: Path | None, *, mode: str | None, tag: str,
                 record: bool = True, repeats: int | None = None) -> EvalSummary:
        out = self.node_dir(node.id) / "eval" / tag
        if out.exists():
            shutil.rmtree(out)
        summary = run_repeats(self.spec, launcher=self.launcher, out_dir=out, repeats=repeats or self.cfg.repeats,
                              mode=mode, candidate=candidate, manifest=self.manifest())
        st = summary.stats()
        (out / "summary.json").write_text(json.dumps(st, indent=2))
        first_bad = next((r for r in summary.results if not r.ok), None)
        node.eval_seconds = (node.eval_seconds or 0.0) + sum(r.duration_s for r in summary.results)
        if record:
            node.eval_status = st["status"]
            node.score, node.score_std, node.score_rel_std, node.n_ok = (
                st["score_mean"], st["score_std"], st["score_rel_std"], st["n_ok"])
            node.error = first_bad.error if first_bad else None
        self.state.event("eval", node=node.id, tag=tag, **{k: st[k] for k in ("status", "score_mean", "score_std", "n_ok", "duration_mean_s")})
        s = f"{st['score_mean']:.6g}" if st["score_mean"] is not None else "-"
        err = first_bad.error if first_bad else None
        self.log(f"    · eval[{tag}] {st['status']}  {self.spec.score.key}={s}"
                 + (f" ± {st['score_std']:.3g}" if st["n_ok"] > 1 else "") + (f"  ({err})" if err else ""))
        return summary

    def history(self, limit: int = 25) -> str:
        rows = []
        for n in self.state.nodes()[-limit:]:
            idea = ""
            if n.plan:
                try:
                    p = json.loads(n.plan)
                    ch = next((h for h in p.get("hypotheses", []) if h.get("id") == p.get("chosen")), {})
                    idea = f"[{ch.get('family', '?')}] {ch.get('idea', '')}"[:160]
                except json.JSONDecodeError:
                    idea = n.plan[:160]
            s = f"{n.score:.6g}" if n.score is not None else "-"
            verdict = "BEST" if n.is_best else ("improved" if n.improved else n.status)
            full = f" full={_full(n)}" if n.full_status else ""
            rows.append(f"#{n.id} (from #{n.parent}) {n.kind:<7} {n.eval_status or '-':<14} score={s:<12}{full} {verdict:<9} {idea}")
        return "\n".join(rows) or "(none yet)"

    def lessons_text(self) -> str:
        ls = self.state.lessons()[-self.cfg.lessons_in_prompt:]
        if not ls:
            return "(none yet)"
        return "\n".join(f"- {l['id']} [{l.get('outcome')}/{l.get('cause')}] {l.get('title')}: {l.get('key_lesson')} "
                         f"(applies when: {l.get('applies_when')})" for l in ls)

    def smoke_cmd(self) -> str | None:
        mode = self.spec.mode(self.cfg.eval_mode)
        cmd = render(mode.cmd, out="/tmp/marble-smoke", workdir=".", bench_dir=self.spec.bench_dir)
        env = " ".join(f"{k}={json.dumps(render(v, out='/tmp/marble-smoke', workdir='.', bench_dir=self.spec.bench_dir))}"
                       for k, v in self.spec.env.items())
        return f"{env} {cmd}".strip()

    # ------------------------------------------------------------------ run
    def start(self) -> Node:
        """Initialise the workspace and evaluate the baseline as the root node (idempotent)."""
        if self.state.nodes():
            return self.state.best()
        self.log(f"Starting run {self.dir.name} on {self.spec.name}")
        self.manifest_path.write_text(json.dumps(protected_manifest(self.spec), indent=2))
        root_sha = self.ws.init(self.spec.repo)
        root = self.state.new_node(None, "root", sha=root_sha)
        calib = [m for m, c in self.spec.evaluate.items() if c.calibration_only]
        mode = calib[0] if calib else self.cfg.eval_mode
        self.log(f"  root #{root.id}: evaluating baseline (mode {mode or self.spec.default_mode})")
        self.evaluate(root, None, mode=mode, tag="baseline")
        root.status, root.finished = "evaluated", time.time()
        self.state.update(root)
        if root.eval_status != "ok":
            raise RuntimeError(f"baseline evaluation failed: {root.eval_status}: {root.error}")
        self.state.set_best(root)
        return root

    def run(self) -> Node:
        self.start()
        no_improve, it = 0, sum(1 for n in self.state.nodes() if n.kind != "root")
        stop = None
        while stop is None:
            if it >= self.cfg.max_iterations:
                stop = f"iteration limit ({self.cfg.max_iterations})"
            elif self.elapsed_h() >= self.cfg.wall_hours:
                stop = f"wall clock limit ({self.cfg.wall_hours} h)"
            elif no_improve >= self.cfg.max_no_improve:
                stop = f"no improvement in {no_improve} iterations"
            elif self.remaining_usd() < 0.5:
                stop = f"budget nearly exhausted (${self.remaining_usd():.2f} left)"
            if stop:
                break
            it += 1
            best = self.state.best()
            self.log(f"\n[iter {it}] expanding best #{best.id} ({self.spec.score.key}={best.score:.6g}); "
                     f"spent ${self.state.total_cost():.2f}/{self.cfg.budget_usd:.0f}, {self.elapsed_h() * 60:.0f} min")
            try:
                improved = self.iteration(best)
            except BudgetExhausted as e:
                stop = str(e)
                break
            no_improve = 0 if improved else no_improve + 1
        self.state.event("run_stopped", reason=stop, cost=self.state.total_cost())
        self.log(f"\nStopped: {stop}")
        best = self.state.best()
        self.write_report(stop)
        return best

    def iteration(self, parent: Node) -> bool:
        node = self.state.new_node(parent.id, "improve")
        nd = self.node_dir(node.id)
        wt = self.ws.add_worktree(node.id, parent.sha)
        try:
            return self._iteration(parent, node, nd, wt)
        except BudgetExhausted:
            node.status, node.finished = "aborted", time.time()
            node.error = "budget exhausted"
            self.state.update(node)
            raise
        finally:
            self.ws.remove_worktree(node.id)

    def _iteration(self, parent: Node, node: Node, nd: Path, wt: Path) -> bool:
        # 1. analysis (cached per parent node)
        findings = self.findings_for(parent, node, wt)
        # 2. plan
        pr = self.agent("planner", node, cwd=wt, tools=READ_TOOLS, schema=roles.PLANNER_SCHEMA,
                        prompt=roles.planner_prompt(findings=findings, history=self.history(),
                                                    lessons=self.lessons_text(), best_label=f"node #{parent.id}"))
        if not pr.ok:
            return self._abort(node, f"planner failed: {pr.error}")
        plan = pr.output
        node.plan = json.dumps(plan)
        (nd / "plan.json").write_text(json.dumps(plan, indent=2))
        chosen = next((h for h in plan["hypotheses"] if h["id"] == plan["chosen"]), plan["hypotheses"][0])
        self.log(f"    plan: [{chosen.get('family')}] {chosen.get('idea')[:150]}")
        # 3. implement
        ir = self.agent("implementer", node, cwd=wt, tools=EDIT_TOOLS, schema=roles.IMPLEMENTER_SCHEMA, editable=True,
                        prompt=roles.implementer_prompt(plan=plan, smoke_cmd=self.smoke_cmd()))
        if not ir.ok:
            return self._abort(node, f"implementer failed: {ir.error}")
        node.summary = ir.output.get("summary")
        if not self._commit(node, wt, f"node {node.id}: {chosen.get('idea', '')[:60]}"):
            node.status, node.finished = "no_change", time.time()
            self.state.update(node)
            self.log("    implementer made no change")
            self.reflect(parent, node, plan, outcome="failed: no change made")
            return False
        # 4. evaluate + debug loop
        summary = self.evaluate(node, wt, mode=self.cfg.eval_mode, tag="try0")
        attempt = 0
        while node.eval_status in FAILED_STATUSES and attempt < self.cfg.debug_attempts:
            attempt += 1
            bad = next(r for r in summary.results if not r.ok)
            log_tail = self._log_tail(Path(bad.out_dir) / "eval.log")
            dr = self.agent("debugger", node, cwd=wt, tools=EDIT_TOOLS, schema=roles.DEBUGGER_SCHEMA, editable=True,
                            prompt=roles.debugger_prompt(plan=plan, status=bad.status, error=bad.error or "",
                                                         log_tail=log_tail, attempt=attempt, smoke_cmd=self.smoke_cmd()))
            if not dr.ok:
                break
            self.state.event("debug", node=node.id, attempt=attempt, diagnosis=dr.output.get("diagnosis"))
            if not self._commit(node, wt, f"node {node.id}: debug {attempt}"):
                break
            summary = self.evaluate(node, wt, mode=self.cfg.eval_mode, tag=f"try{attempt}")
        node.status = "evaluated" if node.eval_status == "ok" else "buggy"
        # 5. compare
        is_imp, gain, thr = improvement(self.spec, parent, node) if node.eval_status == "ok" else (False, 0.0, 0.0)
        confirm_note = ""
        cm = self.confirm_mode()
        if is_imp and cm:
            # Quick mode can over- or under-state a gain (problem subset, fewer trials): re-check before
            # accepting a new best. Compare full-vs-full when the parent has a full score, else vs its score.
            st = self.evaluate(node, wt, mode=cm, tag="confirm", record=False, repeats=self.cfg.confirm_repeats).stats()
            node.full_status, node.full_score = st["status"], st["score_mean"]
            if st["status"] != "ok":
                is_imp = False
                confirm_note = f"; rejected: {cm} eval {st['status']}"
            else:
                ref = Node(id=parent.id, parent=None, sha=None, kind="ref",
                           score=parent.full_score if parent.full_score is not None else parent.score,
                           score_rel_std=parent.score_rel_std)
                cand = Node(id=node.id, parent=None, sha=None, kind="cand", score=node.full_score,
                            score_rel_std=node.score_rel_std)
                ok_full, gain_full, thr_full = improvement(self.spec, ref, cand)
                confirm_note = f"; {cm}: {node.full_score:.6g} ({gain_full:+.2%})"
                if not ok_full:
                    is_imp = False
                    confirm_note += f" below threshold {thr_full:.2%} → rejected"
        node.improved = int(is_imp)
        node.finished = time.time()
        self.state.update(node)
        if is_imp:
            self.state.set_best(node)
            verdict = f"improved: {gain:+.2%} (threshold {thr:.2%}){confirm_note}"
        elif confirm_note:
            verdict = f"quick-mode gain {gain:+.2%} not confirmed{confirm_note}"
        elif node.eval_status == "ok":
            verdict = f"no significant gain: {gain:+.2%} (threshold {thr:.2%})"
        else:
            verdict = f"failed: {node.eval_status} after {attempt} debug attempt(s): {node.error}"
        self.log(f"    → {verdict}")
        self.state.event("verdict", node=node.id, improved=is_imp, gain=gain, threshold=thr, verdict=verdict)
        # 6. reflect
        self.reflect(parent, node, plan, outcome=verdict)
        return is_imp

    # ------------------------------------------------------------------ steps
    def findings_for(self, parent: Node, node: Node, wt: Path) -> dict:
        cache = self.node_dir(parent.id) / "findings.json"
        if cache.exists():
            return json.loads(cache.read_text())
        tool_outputs = {}
        if self.cfg.analyze:
            for tool in self.spec.analyze:
                self.log(f"    · analyze tool '{tool.name}' on #{parent.id} …")
                r = run_tool(self.spec, tool, launcher=self.launcher, out_dir=self.node_dir(parent.id) / "analyze" / tool.name,
                             candidate=wt, manifest=self.manifest(), timeout=self.cfg.analyze_timeout)
                self.state.event("analyze_tool", node=parent.id, tool=tool.name, status=r["status"], seconds=r.get("duration_s"))
                tool_outputs[f"{tool.name} ({r['status']})"] = r["output"]
        eval_summary, metrics = self._latest_eval(parent)
        ar = self.agent("analyst", node, cwd=wt, tools=READ_TOOLS, schema=roles.ANALYST_SCHEMA,
                        prompt=roles.analyst_prompt(node_label=f"node #{parent.id}", eval_summary=eval_summary,
                                                    metrics=metrics, tool_outputs=tool_outputs, history=self.history()))
        findings = ar.output if ar.ok else {"summary": f"(analysis unavailable: {ar.error})", "bottlenecks": [], "opportunities": []}
        if ar.ok:
            cache.write_text(json.dumps(findings, indent=2))
        return findings

    def reflect(self, parent: Node, node: Node, plan: dict, outcome: str) -> None:
        child = self.state.get(node.id)
        diff = self.ws.diff(parent.sha, child.sha) if child.sha and parent.sha and child.sha != parent.sha else "(no change)"
        brief = lambda n: {"id": n.id, "score": n.score, "score_std": n.score_std, "eval_status": n.eval_status,
                           "error": n.error, "summary": n.summary}
        try:
            rr = self.agent("reflector", node, cwd=self.dir, tools=[], schema=roles.REFLECTOR_SCHEMA,
                            prompt=roles.reflector_prompt(parent=brief(parent), child=brief(child), plan=plan,
                                                          diff=diff[:20000], outcome_hint=outcome))
        except BudgetExhausted:
            return
        self.state.update(node)  # persist reflector cost
        if rr.ok:
            lesson = self.state.add_lesson({"node": node.id, "parent": parent.id, **rr.output})
            self.log(f"    lesson {lesson['id']}: {lesson.get('title')}")

    def _commit(self, node: Node, wt: Path, msg: str) -> bool:
        reverted = self.ws.enforce_editable(wt, self.spec.editable)
        if reverted:
            self.state.event("reverted_non_editable", node=node.id, files=reverted)
            self.log(f"    reverted non-editable changes: {reverted[:5]}")
        sha = self.ws.commit(wt, msg)
        if sha:
            node.sha = sha
            self.state.update(node)
            (self.node_dir(node.id) / "diffstat.txt").write_text(self.ws.diffstat(self.state.get(node.parent).sha, sha))
        return sha is not None  # True only if a new commit was made

    def _abort(self, node: Node, why: str) -> bool:
        node.status, node.error, node.finished = "aborted", why, time.time()
        self.state.update(node)
        self.state.event("aborted", node=node.id, error=why)
        self.log(f"    aborted: {why}")
        return False

    def _latest_eval(self, node: Node) -> tuple[dict, dict]:
        evdir = self.node_dir(node.id) / "eval"
        tags = sorted(evdir.glob("*/summary.json"), key=lambda p: p.stat().st_mtime) if evdir.exists() else []
        if not tags:
            return {}, {}
        summ = json.loads(tags[-1].read_text())
        res = sorted(tags[-1].parent.glob("rep*/result.json"))
        metrics = json.loads(res[0].read_text()).get("metrics", {}).get("raw", {}) if res else {}
        return summ, metrics

    @staticmethod
    def _log_tail(path: Path, n: int = 6000) -> str:
        try:
            t = path.read_text(errors="replace")
        except OSError:
            return "(no log)"
        return t[-n:]

    # ------------------------------------------------------------------ report
    def write_report(self, stop: str | None) -> Path:
        nodes = self.state.nodes()
        root = nodes[0]
        best = self.state.best()
        lines = [f"# MARBLE run report: {self.spec.name}", "",
                 f"- stop reason: {stop}", f"- cost: ${self.state.total_cost():.2f} of ${self.cfg.budget_usd:.2f}",
                 f"- nodes: {len(nodes)} ({sum(1 for n in nodes if n.improved)} improvements)",
                 f"- baseline: {self.spec.score.key} = {root.score:.6g}",
                 f"- best: node #{best.id}, {self.spec.score.key} = {best.score:.6g}"
                 + (f" (confirmed in {self.confirm_mode()}: {best.full_score:.6g})" if best.full_score is not None else ""), "",
                 "| node | parent | kind | status | eval | score | confirm | improved | cost $ | idea |",
                 "|---|---|---|---|---|---|---|---|---|---|"]
        for n in nodes:
            idea = ""
            if n.plan:
                p = json.loads(n.plan)
                ch = next((h for h in p["hypotheses"] if h["id"] == p["chosen"]), {})
                idea = f"[{ch.get('family', '')}] {ch.get('idea', '')}"[:90].replace("|", "/")
            s = f"{n.score:.6g}" if n.score is not None else "-"
            lines.append(f"| {n.id}{' ★' if n.is_best else ''} | {n.parent if n.parent is not None else '-'} | {n.kind} | "
                         f"{n.status} | {n.eval_status or '-'} | {s} | {_full(n)} | {'yes' if n.improved else ''} | {n.cost_usd:.2f} | {idea} |")
        if best.id != root.id:
            (self.dir / "best.patch").write_text(self.ws.diff(root.sha, best.sha, max_chars=10**7))
            lines += ["", f"Best change vs baseline: `best.patch`"]
        p = self.dir / "report.md"
        p.write_text("\n".join(lines) + "\n")
        return p
