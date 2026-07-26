"""Independent workspaces for bounded pytest repair attempts."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class AttemptWorkspaceManager:
    """Creates clean Git worktrees, with a copy fallback for non-Git unit fixtures."""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        self._temporary = tempfile.TemporaryDirectory(prefix="firstcoder-pytest-attempts-")
        self.root = Path(self._temporary.name)
        self.git_backed = _is_git_worktree(self.project_root)
        if self.git_backed:
            self.source_repo = self.project_root
            self.base_commit = _git_stdout(["rev-parse", "HEAD"], self.project_root)
            dirty = _git_stdout(["status", "--porcelain"], self.project_root)
            if dirty:
                self.close()
                raise RuntimeError(
                    "pytest-fix requires a clean Git worktree so attempts can be isolated without "
                    "overwriting user changes"
                )
        else:
            self.source_repo = self.root / "baseline"
            shutil.copytree(
                self.project_root,
                self.source_repo,
                ignore=shutil.ignore_patterns(".git", ".firstcoder", "__pycache__", ".pytest_cache"),
            )
            _initialize_snapshot_repo(self.source_repo)
            self.base_commit = _git_stdout(["rev-parse", "HEAD"], self.source_repo)
        self._worktrees: list[Path] = []

    def create(self, number: int) -> Path:
        destination = self.root / f"attempt-{number}"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(destination), str(self.base_commit)],
            cwd=self.source_repo,
            check=True,
            text=True,
            capture_output=True,
        )
        self._worktrees.append(destination)
        return destination

    def close(self) -> None:
        for worktree in reversed(getattr(self, "_worktrees", [])):
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=self.source_repo,
                check=False,
                text=True,
                capture_output=True,
            )
        self._worktrees = []
        self._temporary.cleanup()

    def __enter__(self) -> "AttemptWorkspaceManager":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def apply_selected_files(
    source: Path,
    destination: Path,
    *,
    editable_paths: list[str],
) -> None:
    """Copy only the explicitly editable result set back to the user's clean tree."""

    for relative in editable_paths:
        source_path = source / relative
        destination_path = destination / relative
        if source_path.is_file():
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
        elif destination_path.is_file() or destination_path.is_symlink():
            destination_path.unlink()


def _is_git_worktree(root: Path) -> bool:
    completed = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=root,
        text=True,
        capture_output=True,
    )
    return completed.returncode == 0 and completed.stdout.strip() == "true"


def _git_stdout(args: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def _initialize_snapshot_repo(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, text=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "firstcoder@local"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "FirstCoder Snapshot"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-m", "FirstCoder attempt baseline"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )
