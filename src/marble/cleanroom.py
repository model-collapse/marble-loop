"""Clean-room evaluation copies and integrity checks.

Every evaluation runs on a fresh copy of the *pristine* baseline repo. From a candidate only files
matching the `editable` globs are taken; any other difference is an integrity violation.
Protected files are verified by SHA-256 after the copy is assembled.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .spec import BenchmarkSpec

IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache", ".marble", ".ipynb_checkpoints"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}


class IntegrityError(RuntimeError):
    """A candidate touched files it may not touch, or protected files were altered."""


# --------------------------------------------------------------------------- globbing


@lru_cache(maxsize=512)
def _glob_regex(pattern: str) -> re.Pattern:
    """Glob -> regex with path semantics: '*' and '?' stop at '/', '**' crosses directories."""
    i, out = 0, []
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("".join(out) + r"\Z")


def match_any(relpath: str, patterns: list[str]) -> bool:
    return any(_glob_regex(p).match(relpath) for p in patterns)


# --------------------------------------------------------------------------- file listing


def _ignored(name: str, is_dir: bool) -> bool:
    if is_dir:
        return name in IGNORED_DIRS
    return any(name.endswith(s) for s in IGNORED_SUFFIXES)


def list_repo_files(root: Path) -> list[str]:
    """Relative POSIX paths of all regular files under root, skipping caches and VCS dirs."""
    root = Path(root)
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not _ignored(d, True))
        for fn in sorted(filenames):
            if _ignored(fn, False):
                continue
            p = Path(dirpath) / fn
            if p.is_symlink() or not p.is_file():
                continue
            files.append(p.relative_to(root).as_posix())
    return files


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def protected_manifest(spec: BenchmarkSpec, root: Path | None = None) -> dict[str, str]:
    root = Path(root or spec.repo)
    return {f: sha256_file(root / f) for f in list_repo_files(root) if match_any(f, spec.protected)}


# --------------------------------------------------------------------------- candidate diff


@dataclass
class CandidateDiff:
    modified: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    @property
    def changed(self) -> list[str]:
        return sorted(self.modified + self.added + self.deleted)


def diff_candidate(pristine: Path, candidate: Path) -> CandidateDiff:
    base = set(list_repo_files(pristine))
    cand = set(list_repo_files(candidate))
    d = CandidateDiff(added=sorted(cand - base), deleted=sorted(base - cand))
    for f in sorted(base & cand):
        a, b = pristine / f, candidate / f
        if a.stat().st_size != b.stat().st_size or sha256_file(a) != sha256_file(b):
            d.modified.append(f)
    return d


def check_candidate(spec: BenchmarkSpec, candidate: Path) -> CandidateDiff:
    """Raise IntegrityError if the candidate changes anything outside `editable`."""
    d = diff_candidate(spec.repo, Path(candidate))
    bad = [f for f in d.changed if not match_any(f, spec.editable)]
    touched_protected = [f for f in d.changed if match_any(f, spec.protected)]
    if touched_protected:
        raise IntegrityError(f"candidate modifies protected files: {touched_protected[:10]}")
    if bad:
        raise IntegrityError(f"candidate changes non-editable files: {bad[:10]}")
    return d


# --------------------------------------------------------------------------- clean room


@dataclass
class Cleanroom:
    path: Path
    diff: CandidateDiff
    manifest: dict[str, str]
    _owned_tmp: Path | None = None

    def cleanup(self) -> None:
        if self._owned_tmp is not None:
            shutil.rmtree(self._owned_tmp, ignore_errors=True)

    def __enter__(self) -> "Cleanroom":
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()


def _copy_ignore(_dir: str, names: list[str]) -> set[str]:
    return {n for n in names if _ignored(n, os.path.isdir(os.path.join(_dir, n)))}


def prepare_cleanroom(
    spec: BenchmarkSpec,
    candidate: Path | None = None,
    dest: Path | None = None,
    manifest: dict[str, str] | None = None,
) -> Cleanroom:
    """Assemble pristine repo + candidate's editable files in a fresh directory and verify it.

    `manifest` is the trusted protected-file manifest (e.g. recorded at run start). If omitted it is
    computed from the pristine repo now.
    """
    manifest = manifest if manifest is not None else protected_manifest(spec)
    diff = check_candidate(spec, candidate) if candidate is not None else CandidateDiff()

    owned = None
    if dest is None:
        owned = Path(tempfile.mkdtemp(prefix=f"marble-{spec.name}-"))
        dest = owned / "repo"
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(f"clean-room destination already exists: {dest}")
    shutil.copytree(spec.repo, dest, ignore=_copy_ignore, symlinks=False)

    if candidate is not None:
        for f in diff.modified + diff.added:
            (dest / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(candidate) / f, dest / f)
        for f in diff.deleted:
            (dest / f).unlink()

    actual = protected_manifest(spec, dest)
    if actual != manifest:
        changed = sorted(k for k in set(actual) | set(manifest) if actual.get(k) != manifest.get(k))
        if owned:
            shutil.rmtree(owned, ignore_errors=True)
        raise IntegrityError(f"protected files differ from trusted manifest: {changed[:10]}")
    return Cleanroom(path=dest, diff=diff, manifest=manifest, _owned_tmp=owned)
