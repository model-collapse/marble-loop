"""Search policy: budget-aware MCTS over candidate nodes (DESIGN.md §5.1; MARS §4.4).

Pure functions over the node table, so they can be tested without agents.

Actions: `draft`  - a new approach family started from the baseline (root)
         `improve` - a targeted change to a valid node
(debug attempts happen inside a node; failed nodes are terminal.)

Reward  R(v) = G(v) * clip(t/L, eps, 1)^w / eps^w        (w < 0 favours fast candidates; R in [0, 1])
  G(v) = direction-aware min-max normalised score over valid nodes (0.5 if all equal); 0 if not valid.
  t    = mean eval duration of v, L = timeout of the eval mode.
Q(v) = mean reward over the evaluated nodes of v's subtree; N(v) = number of evaluated nodes in it.
UCT(child) = Q + c * sqrt(ln N(parent) / N(child)).
Expansion: root is expandable when it has no children or the best has not improved in the last
`stagnation_n` finished candidates; a valid node while it has < `max_children` children.
Draft family choice: Thompson sampling over families (Beta from improved / not-improved children)
plus a "new family" arm, so a family with a bad first try keeps a real chance (§5.5.11).
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass

from .runstate import Node

EPS = 0.01
NEW_FAMILY = None


@dataclass
class PolicyConfig:
    c_uct: float = 1.0
    max_children: int = 2
    stagnation_n: int = 3
    time_penalty_w: float = -0.07


def valid(n: Node) -> bool:
    return n.eval_status == "ok" and n.score is not None and n.status in ("evaluated",)


def children_map(nodes: list[Node]) -> dict[int, list[Node]]:
    ch: dict[int, list[Node]] = defaultdict(list)
    for n in nodes:
        if n.parent is not None:
            ch[n.parent].append(n)
    return ch


def rewards(nodes: list[Node], direction: str, durations: dict[int, float], timeout_s: float,
            w: float) -> dict[int, float]:
    vs = [n for n in nodes if valid(n)]
    if not vs:
        return {n.id: 0.0 for n in nodes}
    scores = [n.score for n in vs]
    lo, hi = min(scores), max(scores)
    out = {}
    for n in nodes:
        if not valid(n):
            out[n.id] = 0.0
            continue
        if hi == lo:
            g = 0.5
        else:
            g = (n.score - lo) / (hi - lo)
            if direction == "minimize":
                g = 1.0 - g
        t = durations.get(n.id)
        frac = min(1.0, max(EPS, (t / timeout_s) if (t and timeout_s) else EPS))
        out[n.id] = g * (frac ** w) / (EPS ** w)
    return out


def subtree_stats(root_id: int, ch: dict[int, list[Node]], by_id: dict[int, Node],
                  r: dict[int, float]) -> dict[int, tuple[int, float]]:
    """node id -> (N evaluated nodes in subtree, sum of rewards)."""
    stats: dict[int, tuple[int, float]] = {}

    def visit(nid: int) -> tuple[int, float]:
        n = by_id[nid]
        cnt, tot = (1, r[nid]) if (n.eval_status is not None and n.status not in ("created",)) else (0, 0.0)
        for c in ch.get(nid, []):
            cn, ct = visit(c.id)
            cnt, tot = cnt + cn, tot + ct
        stats[nid] = (cnt, tot)
        return cnt, tot

    visit(root_id)
    return stats


def select(nodes: list[Node], *, direction: str, durations: dict[int, float], timeout_s: float,
           cfg: PolicyConfig, new_best_ids: set[int] | None = None) -> tuple[int, str]:
    """Return (node id to expand, action)."""
    by_id = {n.id: n for n in nodes}
    root = next(n for n in nodes if n.kind == "root")
    ch = children_map(nodes)
    if not ch.get(root.id):
        return root.id, "draft"
    done = [x for x in nodes if x.kind != "root" and x.finished]
    recent = done[-cfg.stagnation_n:]
    if len(done) >= cfg.stagnation_n and not any(x.id in (new_best_ids or set()) for x in recent):
        return root.id, "draft"
    r = rewards(nodes, direction, durations, timeout_s, cfg.time_penalty_w)
    st = subtree_stats(root.id, ch, by_id, r)

    def expandable(n: Node) -> bool:
        return n.kind != "root" and valid(n) and len(ch.get(n.id, [])) < cfg.max_children

    def has_frontier(n: Node) -> bool:
        return expandable(n) or any(has_frontier(c) for c in ch.get(n.id, []))

    v = root
    while True:
        if v is not root and expandable(v):
            return v.id, "improve"
        cand = [c for c in ch.get(v.id, []) if has_frontier(c)]
        if not cand:
            return root.id, "draft"          # nothing left below: re-open the root
        n_parent = max(1, st[v.id][0])

        def uct(c: Node) -> float:
            cnt, tot = st[c.id]
            if cnt == 0:
                return math.inf
            return tot / cnt + cfg.c_uct * math.sqrt(math.log(n_parent + 1) / cnt)

        v = max(cand, key=lambda c: (uct(c), -c.id))


def family_stats(nodes: list[Node], families: dict[int, str]) -> dict[str, tuple[int, int]]:
    """family -> (successes, failures); success = the node improved on its parent."""
    st: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for n in nodes:
        f = families.get(n.id)
        if not f or n.kind == "root" or not n.finished:
            continue
        st[f][0 if n.improved else 1] += 1
    return {k: (v[0], v[1]) for k, v in st.items()}


def choose_family(stats: dict[str, tuple[int, int]], rng: random.Random, new_prior: tuple[float, float] = (1.0, 1.0)
                  ) -> str | None:
    """Thompson sampling over known families plus a 'new family' arm (returns None for new)."""
    best, best_v = NEW_FAMILY, rng.betavariate(*new_prior)
    for fam, (s, f) in stats.items():
        v = rng.betavariate(1.0 + s, 1.0 + f)
        if v > best_v:
            best, best_v = fam, v
    return best


def normalize_family(f: str | None) -> str:
    return " ".join((f or "").lower().replace("/", " ").replace("-", " ").split())[:60]
