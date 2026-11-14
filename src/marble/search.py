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

import importlib
import json
import os
import shutil
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import random

from . import inspector as insp
from . import policy
from . import roles
from .agents import EDIT_TOOLS, READ_TOOLS, AgentCall, AgentResult, run_agent
from .cleanroom import protected_manifest
from .evaluate import EvalSummary, run_repeats, run_tool
from .launcher import LocalLauncher
from .runstate import Node, RunConfig, RunState
from .spec import BenchmarkSpec, load_spec, render, validate_spec
from .workspace import Workspace

FAILED_STATUSES = {"failed", "incorrect", "invalid_output", "timeout", "cancelled"}
HEARTBEAT_S = 10.0


def _agent_fn():
    """The worker runner; MARBLE_FAKE_AGENT=module:callable substitutes a scripted agent (tests, CI)."""
    spec = os.environ.get("MARBLE_FAKE_AGENT")
    if spec:
        mod, _, attr = spec.partition(":")
        return getattr(importlib.import_module(mod), attr)
    return run_agent


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
        self.incidents = insp.IncidentLog(self.dir, self.state.event)
        self._last_watches: list[insp.JobWatch] = []
        if self.cfg.inspector == "llm":
            self.investigator = insp.make_llm_investigator(
                lambda call, log_dir: _agent_fn()(call, log_dir), model=self.cfg.models.get("inspector", "us.anthropic.claude-haiku-5-5"),
                cwd=self.dir, log_dir=self.dir / "inspector",
                budget_usd=self.cfg.role_budget_usd.get("inspector", 0.15),
                cost_cb=lambda usd: self.state.add_cost("inspector", usd))
        else:
            self.investigator = insp.rule_investigator

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
        res = _agent_fn()(call, self.node_dir(node.id) / "agents")
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

    def expected_eval_s(self, mode: str | None) -> float | None:
        """Expected duration of one eval in `mode`: the calibrated baseline duration (root node).

        The root may have been timed in a calibration-only mode that mirrors the default mode
        (e.g. KernelBench 'reference' for 'quick'); that duration is used for the default mode too."""
        mode = mode or self.spec.default_mode
        base = [e for e in self.state.events() if e["type"] == "eval" and e.get("tag") == "baseline"]
        if not base or not base[0].get("duration_mean_s"):
            return None
        base_mode = base[0].get("mode") or self.spec.default_mode
        if mode == base_mode or (self.spec.evaluate.get(base_mode) and self.spec.evaluate[base_mode].calibration_only
                                 and mode == self.spec.default_mode):
            return float(base[0]["duration_mean_s"])
        return None

    def evaluate(self, node: Node, candidate: Path | None, *, mode: str | None, tag: str,
                 record: bool = True, repeats: int | None = None) -> EvalSummary:
        out = self.node_dir(node.id) / "eval" / tag
        if out.exists():
            shutil.rmtree(out)
        expected = self.expected_eval_s(mode) if node.kind != "root" else None
        code = self._changed_code(node) if node.kind != "root" else ""
        self._last_watches = []

        def watch(i: int):
            if self.cfg.inspector == "off":
                return None
            w = insp.JobWatch(node=node.id, label=f"eval[{tag}] rep{i} of node #{node.id}", expected_s=expected,
                              incidents=self.incidents, investigator=self.investigator, stall_s=self.cfg.stall_s,
                              context={"candidate_code": code})
            self._last_watches.append(w)
            return w

        summary = run_repeats(self.spec, launcher=self.launcher, out_dir=out, repeats=repeats or self.cfg.repeats,
                              mode=mode, candidate=candidate, manifest=self.manifest(), on_sample_factory=watch)
        st = summary.stats()
        (out / "summary.json").write_text(json.dumps(st, indent=2))
        first_bad = next((r for r in summary.results if not r.ok), None)
        node.eval_seconds = (node.eval_seconds or 0.0) + sum(r.duration_s for r in summary.results)
        if record:
            node.eval_status = st["status"]
            node.score, node.score_std, node.score_rel_std, node.n_ok = (
                st["score_mean"], st["score_std"], st["score_rel_std"], st["n_ok"])
            node.error = first_bad.error if first_bad else None
        self.state.event("eval", node=node.id, tag=tag, mode=mode or self.spec.default_mode,
                         **{k: st[k] for k in ("status", "score_mean", "score_std", "n_ok", "duration_mean_s")})
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
    def recover(self) -> None:
        """After a crash/restart: close iterations that were interrupted mid-way."""
        for n in self.state.nodes():
            if n.status == "created" and n.kind != "root":
                n.status, n.error, n.finished = "aborted", "interrupted (controller restarted)", time.time()
                self.state.update(n)
                self.ws.remove_worktree(n.id)
                self.incidents.add(insp.Incident("interrupted", n.id, "iteration interrupted by a controller restart",
                                                 action="aborted node"))

    def _heartbeat(self, stop: threading.Event) -> None:
        hb = self.dir / "heartbeat"
        while not stop.is_set():
            hb.write_text(json.dumps({"t": time.time(), "pid": os.getpid()}))
            stop.wait(HEARTBEAT_S)

    def start(self) -> Node:
        """Initialise the workspace and evaluate the baseline as the root node (idempotent)."""
        if self.state.nodes():
            self.recover()
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
        hb_stop = threading.Event()
        threading.Thread(target=self._heartbeat, args=(hb_stop,), daemon=True).start()
        try:
            return self._run()
        finally:
            hb_stop.set()

    def pace(self) -> insp.Pace:
        done = [n for n in self.state.nodes() if n.kind != "root" and n.finished]
        secs = sum(max(0.0, (n.finished or 0) - (n.created or 0)) for n in done)
        return insp.Pace(len(done), sum(n.cost_usd or 0.0 for n in done), secs)

    def _run(self) -> Node:
        self.start()
        no_improve, it = 0, sum(1 for n in self.state.nodes() if n.kind != "root")
        stop = None
        while stop is None:
            usd_it, s_it = self.pace().per_iter()
            if it >= self.cfg.max_iterations:
                stop = f"iteration limit ({self.cfg.max_iterations})"
            elif self.elapsed_h() >= self.cfg.wall_hours:
                stop = f"wall clock limit ({self.cfg.wall_hours} h)"
            elif no_improve >= self.cfg.max_no_improve:
                stop = f"no improvement in {no_improve} iterations"
            elif self.remaining_usd() < 0.5:
                stop = f"budget nearly exhausted (${self.remaining_usd():.2f} left)"
            elif usd_it and self.remaining_usd() < 0.8 * usd_it:
                stop = f"budget pace: ${self.remaining_usd():.2f} left < typical iteration ${usd_it:.2f}"
            elif s_it and (self.cfg.wall_hours - self.elapsed_h()) * 3600 < 0.8 * s_it:
                stop = f"wall pace: {(self.cfg.wall_hours - self.elapsed_h()) * 60:.0f} min left < typical iteration {s_it / 60:.0f} min"
            if stop:
                break
            it += 1
            best = self.state.best()
            parent, action = self.select()
            self.log(f"\n[iter {it}] {action} from #{parent.id}"
                     + (f" ({self.spec.score.key}={parent.score:.6g})" if parent.score is not None else "")
                     + f"; best #{best.id}={best.score:.6g}; spent ${self.state.total_cost():.2f}/{self.cfg.budget_usd:.0f}, "
                     f"{self.elapsed_h() * 60:.0f} min")
            try:
                improved = self.iteration(parent, action)
            except BudgetExhausted as e:
                stop = str(e)
                break
            no_improve = 0 if improved else no_improve + 1
        self.state.event("run_stopped", reason=stop, cost=self.state.total_cost())
        self.log(f"\nStopped: {stop}")
        best = self.state.best()
        self.write_report(stop)
        return best

    # ------------------------------------------------------------------ search policy
    def _durations(self) -> dict[int, float]:
        d: dict[int, float] = {}
        for e in self.state.events():
            if e["type"] == "eval" and e.get("status") == "ok" and e.get("tag") != "confirm" and e.get("duration_mean_s"):
                d[e["node"]] = float(e["duration_mean_s"])
        return d

    def _new_best_ids(self) -> set[int]:
        return {e["node"] for e in self.state.events() if e["type"] == "new_best"}

    def select(self) -> tuple[Node, str]:
        if self.cfg.search == "greedy":
            return self.state.best(), "improve"
        pc = policy.PolicyConfig(c_uct=self.cfg.c_uct, max_children=self.cfg.max_children,
                                 stagnation_n=self.cfg.stagnation_n, time_penalty_w=self.cfg.time_penalty_w)
        nid, action = policy.select(self.state.nodes(), direction=self.spec.score.direction, durations=self._durations(),
                                    timeout_s=self.spec.mode(self.cfg.eval_mode).timeout, cfg=pc,
                                    new_best_ids=self._new_best_ids())
        return self.state.get(nid), action

    def _families(self) -> dict[int, str]:
        return {n.id: n.family for n in self.state.nodes() if n.family}

    def directive(self, parent: Node, action: str, node: Node) -> str:
        if action != "draft":
            return roles.improve_directive(f"node #{parent.id}")
        stats = policy.family_stats(self.state.nodes(), self._families())
        fam = policy.choose_family(stats, random.Random(self.cfg.seed * 7919 + node.id))
        self.state.event("draft_family", node=node.id, family=fam, stats=stats)
        return roles.draft_directive(fam, stats)

    def iteration(self, parent: Node, action: str = "improve") -> bool:
        node = self.state.new_node(parent.id, action)
        nd = self.node_dir(node.id)
        wt = self.ws.add_worktree(node.id, parent.sha)
        try:
            return self._iteration(parent, node, nd, wt, action)
        except BudgetExhausted:
            node.status, node.finished = "aborted", time.time()
            node.error = "budget exhausted"
            self.state.update(node)
            raise
        finally:
            self.ws.remove_worktree(node.id)

    def _iteration(self, parent: Node, node: Node, nd: Path, wt: Path, action: str = "improve") -> bool:
        # 1. analysis (cached per parent node)
        findings = self.findings_for(parent, node, wt)
        # 2. plan
        directive = self.directive(parent, action, node)
        pr = self.agent("planner", node, cwd=wt, tools=READ_TOOLS, schema=roles.PLANNER_SCHEMA,
                        prompt=roles.planner_prompt(findings=findings, history=self.history(),
                                                    lessons=self.lessons_text(), best_label=f"node #{parent.id}",
                                                    directive=directive))
        if not pr.ok:
            return self._abort(node, f"planner failed: {pr.error}")
        plan = pr.output
        similar = self._similar_plan(plan, exclude=node.id)
        if similar is not None:
            sim_node, sim = similar
            self.incidents.add(insp.Incident("repeated_idea", node.id, f"plan {sim:.0%} similar to node #{sim_node.id} "
                                             f"({sim_node.status}/{sim_node.eval_status}); asking for a different idea",
                                             action="re-plan"))
            note = (f"\n\nIMPORTANT: your previous answer repeated the idea of node #{sim_node.id}, which ended as "
                    f"'{sim_node.status}/{sim_node.eval_status}' (score {sim_node.score}). Propose a clearly different idea, "
                    "or explain concretely what is different this time.")
            pr2 = self.agent("planner", node, cwd=wt, tools=READ_TOOLS, schema=roles.PLANNER_SCHEMA,
                             prompt=roles.planner_prompt(findings=findings, history=self.history(),
                                                         lessons=self.lessons_text(), best_label=f"node #{parent.id}",
                                                         directive=directive) + note)
            if pr2.ok:
                plan = pr2.output
        node.plan = json.dumps(plan)
        (nd / "plan.json").write_text(json.dumps(plan, indent=2))
        chosen = next((h for h in plan["hypotheses"] if h["id"] == plan["chosen"]), plan["hypotheses"][0])
        node.family = policy.normalize_family(chosen.get("family"))
        self.log(f"    plan: [{chosen.get('family')}] {chosen.get('idea')[:150]}")
        # 3. implement
        ir = self.agent("implementer", node, cwd=wt, tools=EDIT_TOOLS, schema=roles.IMPLEMENTER_SCHEMA, editable=True,
                        prompt=roles.implementer_prompt(plan=plan, smoke_cmd=self.smoke_cmd(),
                                                        expected_s=self.expected_eval_s(self.cfg.eval_mode),
                                                        max_smoke_s=self.cfg.max_smoke_s))
        if not ir.ok:
            return self._abort(node, f"implementer failed: {ir.error}")
        node.summary = ir.output.get("summary")
        if not self._commit(node, wt, f"node {node.id}: {chosen.get('idea', '')[:60]}"):
            node.status, node.finished = "no_change", time.time()
            self.state.update(node)
            self.log("    implementer made no change")
            self.reflect(parent, node, plan, outcome="failed: no change made")
            return False
        dup = self._duplicate_of(node)
        if dup is not None:
            node.status, node.finished = "duplicate", time.time()
            node.error = f"code identical to node #{dup.id}"
            self.state.update(node)
            self.incidents.add(insp.Incident("duplicate_code", node.id, f"candidate code is identical to node #{dup.id} "
                                             f"({dup.eval_status}, score {dup.score}); not re-evaluated", action="rejected"))
            self.log(f"    → duplicate of node #{dup.id}; not evaluated")
            self.state.event("verdict", node=node.id, improved=False, gain=0.0, threshold=0.0,
                             verdict=f"duplicate of node #{dup.id}")
            return False
        # 4. evaluate + debug loop
        summary = self.evaluate(node, wt, mode=self.cfg.eval_mode, tag="try0")
        attempt, signatures = 0, []
        while node.eval_status in FAILED_STATUSES and attempt < self.cfg.debug_attempts:
            bad = next(r for r in summary.results if not r.ok)
            log_tail = self._log_tail(Path(bad.out_dir) / "eval.log")
            if any(w.abandon for w in self._last_watches):
                self.log("    inspector: abandon this approach; no debugging")
                break
            signatures.append(insp.failure_signature(bad.error, log_tail))
            k = self.cfg.repeat_failure_limit
            if len(signatures) >= k and len(set(signatures[-k:])) == 1 and attempt > 0:
                self.incidents.add(insp.Incident("repeated_failure", node.id, f"same failure {k}x in a row after debugging "
                                                 f"(signature {signatures[-1]}): {(bad.error or '')[:200]}",
                                                 action="stop debugging"))
                self.log(f"    same failure {k}x in a row; stop debugging this node")
                break
            attempt += 1
            dr = self.agent("debugger", node, cwd=wt, tools=EDIT_TOOLS, schema=roles.DEBUGGER_SCHEMA, editable=True,
                            prompt=roles.debugger_prompt(plan=plan, status=bad.status, error=bad.error or "",
                                                         log_tail=log_tail, attempt=attempt, smoke_cmd=self.smoke_cmd(),
                                                         expected_s=self.expected_eval_s(self.cfg.eval_mode),
                                                         max_smoke_s=self.cfg.max_smoke_s))
            if not dr.ok:
                break
            self.state.event("debug", node=node.id, attempt=attempt, diagnosis=dr.output.get("diagnosis"))
            if not self._commit(node, wt, f"node {node.id}: debug {attempt}"):
                break
            summary = self.evaluate(node, wt, mode=self.cfg.eval_mode, tag=f"try{attempt}")
        node.status = "evaluated" if node.eval_status == "ok" else "buggy"
        # 5. compare: vs parent (tree statistics) and vs the global best (new best?)
        ok = node.eval_status == "ok"
        beat_parent, gain, thr = improvement(self.spec, parent, node) if ok else (False, 0.0, 0.0)
        best = self.state.best()
        beat_best, gain_b, thr_b = improvement(self.spec, best, node) if ok else (False, 0.0, 0.0)
        confirm_note = ""
        cm = self.confirm_mode()
        if beat_best and cm:
            # Quick mode can over- or under-state a gain (problem subset, fewer trials): re-check before
            # accepting a new best. Compare full-vs-full when the best has a full score, else vs its score.
            st = self.evaluate(node, wt, mode=cm, tag="confirm", record=False, repeats=self.cfg.confirm_repeats).stats()
            node.full_status, node.full_score = st["status"], st["score_mean"]
            if st["status"] != "ok":
                beat_best = beat_parent = False
                confirm_note = f"; rejected: {cm} eval {st['status']}"
            else:
                ref = Node(id=best.id, parent=None, sha=None, kind="ref",
                           score=best.full_score if best.full_score is not None else best.score,
                           score_rel_std=best.score_rel_std)
                cand = Node(id=node.id, parent=None, sha=None, kind="cand", score=node.full_score,
                            score_rel_std=node.score_rel_std)
                ok_full, gain_full, thr_full = improvement(self.spec, ref, cand)
                confirm_note = f"; {cm}: {node.full_score:.6g} ({gain_full:+.2%})"
                if not ok_full:
                    beat_best = False
                    confirm_note += f" below threshold {thr_full:.2%} → not a new best"
        node.improved = int(beat_parent)
        node.finished = time.time()
        self.state.update(node)
        if beat_best:
            self.state.set_best(node)
            verdict = f"new best: {gain_b:+.2%} vs best #{best.id} (threshold {thr_b:.2%}){confirm_note}"
        elif confirm_note:
            verdict = f"quick-mode gain {gain_b:+.2%} vs best not confirmed{confirm_note}"
        elif ok and beat_parent:
            verdict = f"improved on parent #{parent.id}: {gain:+.2%}, but not on best #{best.id} ({gain_b:+.2%})"
        elif ok:
            verdict = f"no significant gain: {gain:+.2%} vs parent (threshold {thr:.2%})"
        else:
            verdict = f"failed: {node.eval_status} after {attempt} debug attempt(s): {node.error}"
        is_imp = beat_best
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

    def _changed_code(self, node: Node, max_chars: int = 6000) -> str:
        """Diff of the candidate against its parent (what the Inspector should read when a job looks stuck)."""
        if not node.sha or node.parent is None:
            return ""
        parent = self.state.get(node.parent)
        return self.ws.diff(parent.sha, node.sha, max_chars=max_chars) if parent.sha else ""

    def _tree(self, sha: str | None) -> str | None:
        return self.ws.git("rev-parse", f"{sha}^{{tree}}").strip() if sha else None

    def _duplicate_of(self, node: Node) -> Node | None:
        """Earlier node whose code tree is identical to this node's (oscillation / repeated attempt)."""
        tree = self._tree(node.sha)
        for n in self.state.nodes():
            if n.id != node.id and n.sha and self._tree(n.sha) == tree:
                return n
        return None

    def _similar_plan(self, plan: dict, exclude: int) -> tuple[Node, float] | None:
        text = insp.chosen_text(plan)
        best = None
        for n in self.state.nodes():
            if n.id == exclude or not n.plan:
                continue
            try:
                sim = insp.plan_similarity(text, insp.chosen_text(json.loads(n.plan)))
            except json.JSONDecodeError:
                continue
            if sim >= self.cfg.plan_similarity_max and (best is None or sim > best[1]):
                best = (n, sim)
        return best

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
        incs = self.incidents.all()
        if incs:
            lines += ["", f"## Inspector incidents ({len(incs)})", "", "| node | kind | action | detail |", "|---|---|---|---|"]
            for i in incs:
                lines.append(f"| {i.get('node') or '-'} | {i['kind']} | {i['action']} | {str(i['detail'])[:160].replace('|', '/')} |")
        p = self.dir / "report.md"
        p.write_text("\n".join(lines) + "\n")
        return p
