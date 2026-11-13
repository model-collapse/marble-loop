import json
import shutil

import pytest

from marble.calibrate import calibrate
from marble.evaluate import run_eval, run_repeats
from marble.spec import load_spec

from .conftest import ROOT, make_bench

OK_EVAL = """
import solution
write(score=solution.VALUE, correct=True)
"""


def _eval(spec, launcher, tmp_path, candidate=None):
    return run_eval(spec, launcher=launcher, out_dir=tmp_path / "ev", candidate=candidate)


def test_ok(tmp_path, launcher):
    spec = load_spec(make_bench(tmp_path, OK_EVAL))
    r = _eval(spec, launcher, tmp_path)
    assert r.status == "ok" and r.score == 1.0 and r.correct is True
    saved = json.loads((tmp_path / "ev" / "result.json").read_text())
    assert saved["status"] == "ok"
    assert not (tmp_path / "ev" / "workdir").exists()      # clean room removed by default


def test_candidate_score(tmp_path, launcher):
    spec = load_spec(make_bench(tmp_path, OK_EVAL))
    cand = tmp_path / "cand"
    shutil.copytree(spec.repo, cand)
    (cand / "solution.py").write_text("VALUE = 3.5\n")
    r = _eval(spec, launcher, tmp_path, candidate=cand)
    assert r.status == "ok" and r.score == 3.5 and r.changed_files == ["solution.py"]


@pytest.mark.parametrize("body,status", [
    ("write(score=1.0, correct=False)\n", "incorrect"),
    ("raise SystemExit(4)\n", "failed"),
    ("pass\n", "invalid_output"),                                  # no score file
    ("write(score=float('nan'), correct=True)\n", "invalid_output"),
    ("write(score='fast', correct=True)\n", "invalid_output"),
    ("write(correct=True)\n", "invalid_output"),
    ("write(score=1.0)\n", "invalid_output"),                      # correctness key missing
    ("time.sleep(60)\n", "timeout"),
])
def test_failure_statuses(tmp_path, launcher, body, status):
    spec = load_spec(make_bench(tmp_path, body, timeout=1.5))
    assert _eval(spec, launcher, tmp_path).status == status


def test_integrity_error_for_bad_candidate(tmp_path, launcher):
    spec = load_spec(make_bench(tmp_path, OK_EVAL))
    cand = tmp_path / "cand"
    shutil.copytree(spec.repo, cand)
    (cand / "eval" / "eval.py").write_text("write(score=1e9, correct=True)\n")
    r = _eval(spec, launcher, tmp_path, candidate=cand)
    assert r.status == "integrity_error" and "protected" in r.error


def test_candidate_tampering_during_eval_detected(tmp_path, launcher):
    # The evaluator imports candidate code; the candidate rewrites the evaluator on import.
    spec = load_spec(make_bench(tmp_path, OK_EVAL))
    cand = tmp_path / "cand"
    shutil.copytree(spec.repo, cand)
    (cand / "solution.py").write_text(
        "import os\nopen(os.path.join(os.getcwd(), 'eval', 'eval.py'), 'a').write('# pwned\\n')\nVALUE = 9.0\n")
    r = _eval(spec, launcher, tmp_path, candidate=cand)
    assert r.status == "integrity_error" and "during eval" in r.error


def test_repeats_and_stats(tmp_path, launcher):
    body = "import random\nwrite(score=1.0 + random.random() * 0.01, correct=True)\n"
    spec = load_spec(make_bench(tmp_path, body))
    s = run_repeats(spec, launcher=launcher, out_dir=tmp_path / "rep", repeats=3)
    st = s.stats()
    assert st["status"] == "ok" and st["n_ok"] == 3
    assert 1.0 <= st["score_mean"] <= 1.01 and st["score_std"] > 0
    assert st["score_rel_std"] < 0.01


def test_repeats_stop_on_failure(tmp_path, launcher):
    spec = load_spec(make_bench(tmp_path, "raise SystemExit(1)\n"))
    s = run_repeats(spec, launcher=launcher, out_dir=tmp_path / "rep", repeats=5)
    assert len(s.results) == 1 and s.status == "failed"


def test_calibrate_writes_report(tmp_path, launcher):
    spec = load_spec(make_bench(tmp_path, OK_EVAL))
    rep = calibrate(spec, launcher=launcher, out_dir=tmp_path / "cal", repeats=2)
    assert rep["status"] == "ok"
    q = rep["modes"]["quick"]
    assert q["n_ok"] == 2 and q["suggestions"]["expected_duration_s"] > 0
    assert q["suggestions"]["suggested_timeout_s"] >= q["suggestions"]["soft_limit_s"]
    assert (tmp_path / "cal" / "calibration.json").exists()
    assert "eval/eval.py" in rep["protected_manifest"]


@pytest.mark.parametrize("bench,expect", [("toy/pairwise", "maximize"), ("toy/fit", "minimize")])
def test_toy_benchmarks_end_to_end(tmp_path, launcher, bench, expect):
    spec = load_spec(ROOT / "benchmarks" / bench)
    assert spec.score.direction == expect
    r = run_eval(spec, launcher=launcher, out_dir=tmp_path / "ev")
    assert r.status == "ok", r.error
    if bench == "toy/pairwise":
        assert 0.5 < r.score < 2.0             # baseline vs itself ~ 1x
    else:
        assert 0.5 < r.score < 1.0             # linear fit of a curved target


CALIB_MODE = """  reference:
    cmd: "{py} eval/eval.py {{out}}"
    timeout: 30
    calibration_only: true
"""


def _with_calib_mode(bench):
    import sys
    y = bench / "benchmark.yaml"
    txt = y.read_text()
    y.write_text(txt.replace("score:", CALIB_MODE.format(py=sys.executable) + "score:", 1))
    return bench


def test_calibration_only_mode_refuses_candidates(tmp_path, launcher):
    from marble.spec import SpecError
    spec = load_spec(_with_calib_mode(make_bench(tmp_path, OK_EVAL)))
    assert spec.evaluate["reference"].calibration_only
    assert run_eval(spec, launcher=launcher, out_dir=tmp_path / "a", mode="reference").status == "ok"
    cand = tmp_path / "cand"
    shutil.copytree(spec.repo, cand)
    with pytest.raises(SpecError, match="calibration_only"):
        run_eval(spec, launcher=launcher, out_dir=tmp_path / "b", mode="reference", candidate=cand)


def test_default_mode_cannot_be_calibration_only(tmp_path):
    from marble.spec import SpecError
    bench = _with_calib_mode(make_bench(tmp_path, OK_EVAL))
    y = bench / "benchmark.yaml"
    y.write_text(y.read_text() + "default_mode: reference\n")
    with pytest.raises(SpecError, match="calibration_only"):
        load_spec(bench)
