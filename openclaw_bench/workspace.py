from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


IGNORED_DIRS = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache"}
OPENCLAW_SEED_FILES = ("AGENTS.md", "SOUL.md", "TOOLS.md", "IDENTITY.md", "USER.md", "HEARTBEAT.md")
OPENCLAW_STATE_PATH = Path(".openclaw") / "workspace-state.json"
OPENCLAW_SEED_TIMESTAMP = "2026-01-01T00:00:00.000Z"


def copy_fixture(fixture: Path, workspace: Path) -> None:
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(fixture, workspace)


def seed_openclaw_workspace_files(
    workspace: Path,
    *,
    agent_id: str = "bench",
    task_id: str = "unknown-task",
    model_id: str = "unknown-model",
) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    seed_context = {
        "agent_id": agent_id,
        "task_id": task_id,
        "model_id": model_id,
    }
    seed_payloads = _openclaw_seed_payloads(seed_context)
    for rel_path in OPENCLAW_SEED_FILES:
        content = seed_payloads[rel_path]
        path = workspace / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    state_path = workspace / OPENCLAW_STATE_PATH
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "setupCompletedAt": OPENCLAW_SEED_TIMESTAMP,
                "benchSeededAt": OPENCLAW_SEED_TIMESTAMP,
                "benchAgentId": agent_id,
                "benchTaskId": task_id,
                "benchModelId": model_id,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


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


def _openclaw_seed_payloads(context: dict[str, str]) -> dict[str, str]:
    agent_id = context["agent_id"]
    task_id = context["task_id"]
    model_id = context["model_id"]
    return {
        "AGENTS.md": (
            "# Benchclaw Agent Notes\n\n"
            f"- Agent id: {agent_id}\n"
            f"- Current benchmark task: {task_id}\n"
            f"- Routed model: {model_id}\n"
            "- This workspace is already onboarded for automated benchmark evaluation.\n"
            "- Do not start interactive onboarding or look for BOOTSTRAP.md.\n"
            "- Inspect repository files with tools before answering task questions.\n"
            "- When the task asks for JSON, return only the requested JSON object.\n"
        ),
        "SOUL.md": (
            "# Soul\n\n"
            "Benchclaw is a repo-focused OpenClaw agent used for local model evaluation.\n"
            "It is quiet, evidence-seeking, and practical: inspect files first, avoid guesses, "
            "cite real paths, and keep answers concise.\n"
        ),
        "TOOLS.md": (
            "# Tool Preferences\n\n"
            "- Use file-reading and shell tools when repository evidence is needed.\n"
            "- Prefer targeted inspection over broad directory sweeps.\n"
            "- Do not fabricate file paths, commands, or test results.\n"
            "- Keep code edits scoped to the requested task and verify with the configured checks.\n"
        ),
        "IDENTITY.md": (
            "# Identity\n\n"
            f"Name: Benchclaw {agent_id}\n"
            "Role: OpenClaw benchmark workspace agent\n"
            f"Assignment: {task_id}\n"
            f"Model route under test: {model_id}\n"
            "Operating mode: non-interactive, already onboarded, tool-first repository work\n"
        ),
        "USER.md": (
            "# User Preferences\n\n"
            "The user is evaluating OpenClaw local model behavior. They value verified output, "
            "minimal churn, and direct answers. Treat benchmark prompts as authoritative and do "
            "not perform setup rituals unless explicitly asked.\n"
        ),
        "HEARTBEAT.md": (
            "# Heartbeat\n\n"
            f"Status: ready for benchmark task {task_id}\n"
            "Workspace onboarding: complete\n"
            "Bootstrap ritual: intentionally disabled for this benchmark workspace\n"
        ),
    }
