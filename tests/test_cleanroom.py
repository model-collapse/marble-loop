import shutil

import pytest

from marble.cleanroom import IntegrityError, check_candidate, prepare_cleanroom, protected_manifest
from marble.spec import load_spec

from .conftest import make_bench


@pytest.fixture
def spec(tmp_path):
    return load_spec(make_bench(tmp_path, "write(score=1.0, correct=True)\n"))


def _candidate(spec, tmp_path):
    cand = tmp_path / "cand"
    shutil.copytree(spec.repo, cand)
    return cand


def test_baseline_cleanroom_has_all_files(spec):
    with prepare_cleanroom(spec) as room:
        assert (room.path / "solution.py").read_text() == "VALUE = 1.0\n"
        assert (room.path / "eval" / "eval.py").exists()
        assert room.diff.changed == []


def test_editable_change_is_applied(spec, tmp_path):
    cand = _candidate(spec, tmp_path)
    (cand / "solution.py").write_text("VALUE = 2.0\n")
    (cand / "extra").mkdir()
    (cand / "extra" / "helper.py").write_text("X = 1\n")
    with prepare_cleanroom(spec, candidate=cand) as room:
        assert (room.path / "solution.py").read_text() == "VALUE = 2.0\n"
        assert (room.path / "extra" / "helper.py").exists()
        assert room.diff.changed == ["extra/helper.py", "solution.py"]


def test_caches_are_ignored(spec, tmp_path):
    cand = _candidate(spec, tmp_path)
    (cand / "__pycache__").mkdir()
    (cand / "__pycache__" / "x.pyc").write_bytes(b"\0")
    assert check_candidate(spec, cand).changed == []


@pytest.mark.parametrize("attack", ["modify_protected", "add_protected", "delete_protected", "non_editable"])
def test_integrity_violations_rejected(spec, tmp_path, attack):
    cand = _candidate(spec, tmp_path)
    if attack == "modify_protected":
        (cand / "eval" / "eval.py").write_text("write(score=1e9, correct=True)\n")
    elif attack == "add_protected":
        (cand / "eval" / "sitecustomize.py").write_text("print('hi')\n")
    elif attack == "delete_protected":
        (cand / "eval" / "eval.py").unlink()
    else:
        (cand / "README.md").write_text("changed\n")
    with pytest.raises(IntegrityError):
        prepare_cleanroom(spec, candidate=cand)


def test_tampered_pristine_detected_against_trusted_manifest(spec):
    trusted = protected_manifest(spec)
    (spec.repo / "eval" / "eval.py").write_text("# tampered after the run started\n")
    with pytest.raises(IntegrityError, match="trusted manifest"):
        prepare_cleanroom(spec, manifest=trusted)
