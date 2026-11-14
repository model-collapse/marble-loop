import random
import time

from marble import policy
from marble.runstate import Node

T = time.time()


def N(id, parent, kind="improve", score=None, status="evaluated", eval_status="ok", improved=0, is_best=0):
    return Node(id=id, parent=parent, sha=f"s{id}", kind=kind, status=status, eval_status=eval_status, score=score,
                improved=improved, is_best=is_best, finished=T + id)


def root(score=1.0):
    return N(1, None, kind="root", score=score)


PC = policy.PolicyConfig(max_children=2, stagnation_n=3)


def sel(nodes, durations=None, new_best=None, cfg=PC):
    return policy.select(nodes, direction="maximize", durations=durations or {}, timeout_s=100.0, cfg=cfg,
                         new_best_ids=new_best or set())


def test_reward_time_penalty_and_direction():
    nodes = [root(1.0), N(2, 1, score=2.0), N(3, 1, score=2.0)]
    r = policy.rewards(nodes, "maximize", {1: 10, 2: 1.0, 3: 50.0}, 100.0, -0.07)
    assert r[2] > r[3] > 0                       # same score, faster node gets more reward
    assert r[1] == 0.0                            # worst score normalises to 0
    assert max(r.values()) <= 1.0 + 1e-9
    rm = policy.rewards([root(5.0), N(2, 1, score=1.0)], "minimize", {}, 100.0, -0.07)
    assert rm[2] > rm[1]                          # lower is better
    bad = policy.rewards([root(), N(2, 1, score=None, status="buggy", eval_status="failed")], "maximize", {}, 100.0, -0.07)
    assert bad[2] == 0.0


def test_select_draft_then_improve():
    assert sel([root()]) == (1, "draft")
    assert sel([root(), N(2, 1, kind="draft", score=2.0)], new_best={2}) == (2, "improve")


def test_failed_nodes_are_terminal_and_root_reopens():
    nodes = [root(), N(2, 1, kind="draft", status="buggy", eval_status="failed")]
    assert sel(nodes) == (1, "draft")


def test_children_cap_moves_selection_down_the_tree():
    nodes = [root(), N(2, 1, kind="draft", score=2.0, is_best=0), N(3, 2, score=3.0), N(4, 2, score=1.5)]
    pid, action = sel(nodes, new_best={3})
    assert action == "improve" and pid in (3, 4)   # node 2 is full (2 children)
    assert pid == 3                                 # higher reward wins with equal visits


def test_stagnation_reopens_root():
    nodes = [root(), N(2, 1, kind="draft", score=2.0), N(3, 2, score=1.9), N(4, 2, score=1.8), N(5, 3, score=1.7)]
    assert sel(nodes, new_best={2}) == (1, "draft")          # last 3 candidates produced no new best
    assert sel(nodes, new_best={5})[1] == "improve"


def test_choose_family_thompson():
    rng = random.Random(0)
    stats = {"vectorize": (5, 0), "caching": (0, 5)}
    picks = [policy.choose_family(stats, rng) for _ in range(500)]
    assert picks.count("vectorize") > 300
    assert picks.count("caching") > 0              # a failed family keeps a non-zero chance
    assert picks.count(None) > 0                    # and so does a brand-new family


def test_normalize_family():
    assert policy.normalize_family("  Kernel-Fusion / CUDA ") == "kernel fusion cuda"
