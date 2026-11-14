import json

import pytest
import yaml

from marble import resources as R
from marble.runstate import RunConfig
from marble.spec import SpecError, load_spec

from .conftest import ROOT

CAL = {"modes": {"quick": {"status": "ok", "duration_mean_s": 10.0, "score_mean": 1.0, "score_std": 0.01,
                           "score_rel_std": 0.01, "suggestions": {"suggested_timeout_s": 60}}}}
HW = {"cpus": 16, "ram_gb": 64, "disk_free_gb": 100, "gpus": [{"index": 0, "name": "X", "memory_gb": 24, "util_pct": 0}],
      "claude_cli": True}


def test_estimate_limits_by_time_and_budget():
    spec = load_spec(ROOT / "benchmarks/toy/pairwise")
    e = R.estimate(spec, eval_s=10, repeats=3, hours=1, budget_usd=1.0, iter_usd=0.2)
    assert e.iterations_by_budget == 5 and e.iterations == min(e.iterations_by_time, 5)
    slow = R.estimate(spec, eval_s=10, repeats=3, hours=1, budget_usd=100, iter_s=600)
    assert slow.iteration_s == 600 and slow.iterations == 6          # earlier runs override the model


def test_make_plan_cpu_benchmark_and_roundtrip(tmp_path):
    spec = load_spec(ROOT / "benchmarks/toy/pairwise")
    plan = R.make_plan(spec, hw=HW, calibration=CAL, hours=2, budget_usd=10, past_costs=[0.1, 0.3])
    res = plan["resources"]
    assert res["slots"][0]["name"] == "cpu0" and res["slots"][0]["cpu_ids"]
    assert "iterations" in plan["markdown"] and plan["estimate"]["iteration_usd"] == 0.2
    p = tmp_path / "resources.yaml"
    p.write_text(yaml.safe_dump(res))
    data = R.load_resources(p)
    cfg = R.apply_to_config(data, RunConfig(benchmark="x"))
    assert cfg.budget_usd == 10 and cfg.wall_hours == 2 and cfg.max_iterations == res["run"]["max_iterations"]
    assert R.slots_from(data)[0].cpus == res["slots"][0]["cpus"]


def test_gpu_benchmark_without_gpu_is_rejected(tmp_path):
    from marble.spec import parse_spec
    raw = yaml.safe_load((ROOT / "benchmarks/toy/pairwise/benchmark.yaml").read_text())
    raw["evaluate"]["quick"]["resources"] = {"gpu": 1}
    raw["evaluate"]["quick"]["cmd"] = "python x {out}"
    raw["evaluate"]["full"]["cmd"] = "python x {out}"
    raw["analyze"] = []
    spec = parse_spec(raw, bench_dir=ROOT / "benchmarks/toy/pairwise")
    with pytest.raises(SpecError, match="needs a GPU"):
        R.make_plan(spec, hw={**HW, "gpus": []}, calibration=CAL, hours=1, budget_usd=1)


@pytest.mark.parametrize("mut,msg", [
    (lambda d: d.update(schema_version=9), "schema_version"),
    (lambda d: d.update(slots=[]), "at least one slot"),
    (lambda d: d["slots"][0].update(launcher="slurm"), "only launcher 'local'"),
    (lambda d: d.update(run={"bogus": 1}), "unknown run setting"),
])
def test_load_resources_validation(tmp_path, mut, msg):
    d = {"schema_version": 1, "budget": {"usd": 1}, "slots": [{"name": "cpu0", "launcher": "local", "cpus": 2}], "run": {}}
    mut(d)
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(SpecError, match=msg):
        R.load_resources(p)


def test_past_iterations_reads_earlier_runs(tmp_path):
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"benchmark": "/x/pairwise"}))
    ev = [{"type": "node_created", "node": 2, "t": 100.0}, {"type": "cost", "usd": 0.3},
          {"type": "verdict", "node": 2, "t": 160.0}]
    (run / "events.jsonl").write_text("\n".join(json.dumps(e) for e in ev))
    assert R.past_iterations(tmp_path / "runs", "pairwise") == [(0.3, 60.0)]
    assert R.past_iterations(tmp_path / "runs", "other") == []
