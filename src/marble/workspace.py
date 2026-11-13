"""Per-run git workspace: one commit per node, one worktree per node being worked on.

The run repo starts with a single commit = the pristine baseline. Every candidate is a commit whose
parent is the node it was derived from, so `git diff parent child` is the exact change a lesson
talks about. Worktrees are created on demand and pruned when no longer needed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .cleanroom import IGNORED_DIRS, IGNORED_SUFFIXES, list_repo_files, match_any

GIT_ENV = {
    "GIT_AUTHOR_NAME": "MARBLE", "GIT_AUTHOR_EMAIL": "marble@localhost",
    "GIT_COMMITTER_NAME": "MARBLE", "GIT_COMMITTER_EMAIL": "marble@localhost",
    "GIT_TERMINAL_PROMPT": "0",
}


class WorkspaceError(RuntimeError):
    pass


class Workspace:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir).resolve()
        self.repo = self.run_dir / "repo"
        self.nodes_dir = self.run_dir / "nodes"

    # ------------------------------------------------------------------ git plumbing
    def git(self, *args: str, cwd: Path | None = None, check: bool = True) -> str:
        env = dict(os.environ)
        env.update(GIT_ENV)
        p = subprocess.run(["git", *args], cwd=str(cwd or self.repo), env=env, capture_output=True, text=True)
        if check and p.returncode != 0:
            raise WorkspaceError(f"git {' '.join(args)} failed: {p.stderr.strip()[:2000]}")
        return p.stdout

    # ------------------------------------------------------------------ lifecycle
    def init(self, baseline: Path) -> str:
        """Create the run repo from the pristine baseline; returns the root commit sha."""
        if self.repo.exists():
            raise WorkspaceError(f"workspace already initialised: {self.repo}")
        self.repo.mkdir(parents=True)
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        self.git("init", "-q", "-b", "marble")
        # Ignore caches via info/exclude (a .gitignore would itself be a non-editable change).
        excl = self.repo / ".git" / "info" / "exclude"
        excl.parent.mkdir(parents=True, exist_ok=True)
        excl.write_text("".join(f"{d}/\n" for d in sorted(IGNORED_DIRS)) +
                        "".join(f"*{s}\n" for s in sorted(IGNORED_SUFFIXES)))
        for rel in list_repo_files(baseline):
            dst = self.repo / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(baseline) / rel, dst)
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", "baseline")
        return self.head()

    def head(self, cwd: Path | None = None) -> str:
        return self.git("rev-parse", "HEAD", cwd=cwd).strip()

    def worktree_path(self, node_id: int) -> Path:
        return self.nodes_dir / f"{node_id:04d}" / "wt"

    def add_worktree(self, node_id: int, base_sha: str) -> Path:
        wt = self.worktree_path(node_id)
        if wt.exists():
            self.remove_worktree(node_id)
        wt.parent.mkdir(parents=True, exist_ok=True)
        self.git("worktree", "add", "-q", "--detach", "-f", str(wt), base_sha)
        return wt

    def remove_worktree(self, node_id: int) -> None:
        wt = self.worktree_path(node_id)
        if wt.exists():
            self.git("worktree", "remove", "--force", str(wt), check=False)
            shutil.rmtree(wt, ignore_errors=True)
        self.git("worktree", "prune", check=False)

    # ------------------------------------------------------------------ changes
    def changed_files(self, wt: Path) -> list[str]:
        """Files changed in the worktree relative to its HEAD (tracked + untracked, cache files ignored)."""
        out = self.git("status", "--porcelain", "-uall", "--no-renames", cwd=wt)
        files = []
        for line in out.splitlines():
            path = line[3:]
            if path.startswith('"') and path.endswith('"'):
                path = path[1:-1]
            files.append(path)
        return sorted(set(files))

    def enforce_editable(self, wt: Path, editable: list[str]) -> list[str]:
        """Revert every change outside the editable globs; returns the reverted paths."""
        reverted = []
        for f in self.changed_files(wt):
            if match_any(f, editable):
                continue
            tracked = self.git("ls-files", "--error-unmatch", f, cwd=wt, check=False).strip()
            if tracked:
                self.git("checkout", "-q", "HEAD", "--", f, cwd=wt)
            else:
                p = wt / f
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)
            reverted.append(f)
        return reverted

    def commit(self, wt: Path, message: str) -> str | None:
        """Commit all changes in the worktree; returns the new sha, or None if nothing changed."""
        if not self.changed_files(wt):
            return None
        self.git("add", "-A", cwd=wt)
        self.git("commit", "-q", "-m", message, cwd=wt)
        return self.head(cwd=wt)

    def diff(self, a: str, b: str, max_chars: int = 40_000) -> str:
        d = self.git("diff", "--no-color", "-U3", a, b)
        if len(d) > max_chars:
            d = d[:max_chars] + f"\n... [diff truncated, {len(d) - max_chars} more chars]\n"
        return d

    def diffstat(self, a: str, b: str) -> str:
        return self.git("diff", "--stat", a, b).strip()

    def export(self, sha: str, dest: Path) -> Path:
        """Materialise a commit as a plain directory (used as an eval candidate)."""
        dest = Path(dest)
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        archive = subprocess.run(["git", "archive", sha], cwd=str(self.repo), capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)
        return dest
