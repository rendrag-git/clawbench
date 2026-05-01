from __future__ import annotations

import json
from pathlib import Path

from .models import SuiteManifest, TaskSpec


def load_suite(path: Path) -> SuiteManifest:
    return _load_suite(path, seen=set())


def _load_suite(path: Path, seen: set[Path]) -> SuiteManifest:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"cyclic suite include: {path}")
    seen.add(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks: list[TaskSpec] = []
    source_paths = [path]
    for include in data.get("include_suites", []):
        included = _load_suite(path.parent / include, seen)
        tasks.extend(included.tasks)
        source_paths.extend(included.source_paths)
    tasks.extend(_load_tasks(data.get("tasks", [])))
    seen.remove(path)
    return SuiteManifest(suite_id=data["suite_id"], tasks=tasks, root=path.parent, source_paths=source_paths)


def _load_tasks(items: list[dict]) -> list[TaskSpec]:
    return [
        TaskSpec(
            task_id=item["task_id"],
            task_type=item["task_type"],
            fixture=item["fixture"],
            prompt=item["prompt"],
            expected=item.get("expected", {}),
            verify_command=item.get("verify_command", []),
            context_sizes=item.get("context_sizes", []),
            max_changed_files=item.get("max_changed_files", 6),
            tags=item.get("tags", []),
        )
        for item in items
    ]


def load_model_specs(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return data["models"]


def load_model_manifest_scope(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    scope = data.get("manifest_scope")
    return scope if isinstance(scope, dict) else {}
