from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


IGNORED_DIRS = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache"}


def copy_fixture(fixture: Path, workspace: Path) -> None:
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(fixture, workspace)


def snapshot_files(root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if should_ignore_workspace_path(root, path):
            continue
        rel = path.relative_to(root).as_posix()
        snapshot[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def read_text_files(root: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if should_ignore_workspace_path(root, path):
            continue
        try:
            data[path.relative_to(root).as_posix()] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return data


def should_ignore_workspace_path(root: Path, path: Path) -> bool:
    return any(part in IGNORED_DIRS for part in path.relative_to(root).parts)


def changed_files(before: dict[str, str], after: dict[str, str]) -> list[str]:
    keys = sorted(set(before) | set(after))
    return [key for key in keys if before.get(key) != after.get(key)]


def path_exists(root: Path, rel_path: str) -> bool:
    try:
        candidate = (root / rel_path).resolve()
        candidate.relative_to(root.resolve())
    except ValueError:
        return False
    return candidate.exists()
