from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Callable

from .preflight import PreflightCheck, ensure_openclaw_gateway


DEFAULT_PROFILE = "benchclaw"
DEFAULT_AGENT = "bench"
DEFAULT_START_PORT = 19191
STARTER_SUITE = "openclaw-agent-discovery-smoke.example.json"
LOCAL_MODEL_MANIFEST = "vllm-gptoss-smoke.example.json"
API_MODEL_MANIFEST = "api-providers.example.json"
PROVIDER_CHOICES = {"local", "api", "both"}


@dataclass(frozen=True)
class QuickstartPaths:
    bench_root: Path
    results_root: Path
    workspace_root: Path
    fixtures_root: Path
    manifest_dir: Path
    config_path: Path
    suite_path: Path
    model_config_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class QuickstartInitResult:
    profile: str
    providers: str
    port: int
    paths: QuickstartPaths
    openclaw_cli: str | None
    existing_profiles: list[str]
    validation: PreflightCheck | None


def default_bench_root() -> Path:
    return Path.home() / "openclaw-bench"


def normalize_provider_selection(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {"a": "api", "apis": "api", "api-key": "api", "api-keys": "api", "all": "both"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in PROVIDER_CHOICES:
        raise ValueError("providers must be one of: local, api, both")
    return normalized


def prompt_provider_selection(input_fn: Callable[[str], str] = input) -> str:
    prompt = "Provider mode [local/api/both] (local): "
    answer = input_fn(prompt).strip()
    return normalize_provider_selection(answer or "local")


def init_quickstart(
    *,
    providers: str,
    project_root: Path,
    bench_root: Path | None = None,
    profile: str = DEFAULT_PROFILE,
    agent: str = DEFAULT_AGENT,
    port: int | None = None,
    home: Path | None = None,
    force: bool = False,
    reuse_existing: bool = False,
    validate: bool = True,
) -> QuickstartInitResult:
    provider_mode = normalize_provider_selection(providers)
    root = (bench_root or default_bench_root()).expanduser()
    home_root = (home or Path.home()).expanduser()
    selected_port = port or choose_safe_port(DEFAULT_START_PORT)
    paths = _quickstart_paths(root, home_root, profile)

    for path in (paths.results_root, paths.workspace_root, paths.fixtures_root, paths.manifest_dir, paths.config_path.parent):
        path.mkdir(parents=True, exist_ok=True)

    config_exists = paths.config_path.exists()
    if config_exists and not force and not reuse_existing:
        raise ValueError(f"{paths.config_path} already exists; pass --force to overwrite the isolated {profile} profile")
    if config_exists and reuse_existing and not force:
        _check_reusable_metadata(paths.metadata_path, provider_mode)

    suite_payload = _read_asset_json(project_root, "manifests", STARTER_SUITE)
    model_payload = _selected_model_manifest(provider_mode, project_root)
    paths.suite_path.write_text(json.dumps(suite_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths.model_config_path.write_text(json.dumps(model_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _copy_asset_tree(project_root, ("fixtures", "discovery_repo"), paths.fixtures_root / "discovery_repo")

    config = _openclaw_config(provider_mode, project_root, selected_port, agent)
    if not config_exists or force:
        paths.config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "providers": provider_mode,
        "port": selected_port,
        "agent": agent,
        "results_root": str(paths.results_root),
        "workspace_root": str(paths.workspace_root),
        "fixtures_root": str(paths.fixtures_root),
        "suite": str(paths.suite_path),
        "model_config": str(paths.model_config_path),
        "oauth_note": "OAuth-backed providers are bring-your-own-auth for this phase; configure them in the benchclaw profile before running.",
    }
    paths.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    validation = validate_benchclaw_config(profile, home_root) if validate and shutil.which("openclaw") else None
    if validation is not None and validation.status == "fail":
        raise ValueError(f"generated OpenClaw config did not validate: {validation.notes}")

    return QuickstartInitResult(
        profile=profile,
        providers=provider_mode,
        port=selected_port,
        paths=paths,
        openclaw_cli=shutil.which("openclaw"),
        existing_profiles=detect_existing_profiles(home_root),
        validation=validation,
    )


def start_benchclaw_gateway(profile: str = DEFAULT_PROFILE, timeout_s: int = 60) -> PreflightCheck:
    return ensure_openclaw_gateway(profile, None, timeout_s=timeout_s)


def stop_benchclaw_gateway(profile: str = DEFAULT_PROFILE, timeout_s: int = 30) -> PreflightCheck:
    cmd = ["openclaw", "--profile", profile, "gateway", "stop"]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return PreflightCheck("openclaw_gateway_stop", "fail", str(exc))
    output = f"{proc.stdout}\n{proc.stderr}".strip()
    if proc.returncode != 0:
        return PreflightCheck("openclaw_gateway_stop", "fail", _trim_output(output))
    return PreflightCheck("openclaw_gateway_stop", "pass", _trim_output(output) or f"stopped {profile} gateway")


def validate_benchclaw_config(profile: str, home: Path | None = None, timeout_s: int = 15) -> PreflightCheck:
    env = os.environ.copy()
    if home is not None:
        env["HOME"] = str(home)
    cmd = ["openclaw", "--profile", profile, "config", "validate"]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_s, check=False, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return PreflightCheck("openclaw_profile_config", "fail", str(exc))
    output = f"{proc.stdout}\n{proc.stderr}".strip()
    if proc.returncode == 0:
        return PreflightCheck("openclaw_profile_config", "pass", output or f"profile {profile} validates")
    return PreflightCheck("openclaw_profile_config", "fail", _trim_output(output))


def detect_existing_profiles(home: Path | None = None) -> list[str]:
    root = (home or Path.home()).expanduser()
    profiles = []
    for path in sorted(root.glob(".openclaw*")):
        if not path.is_dir():
            continue
        config = path / "openclaw.json"
        if not config.exists():
            continue
        if path.name == ".openclaw":
            profiles.append("default")
        elif path.name.startswith(".openclaw-"):
            profiles.append(path.name.removeprefix(".openclaw-"))
    return profiles


def choose_safe_port(start: int = DEFAULT_START_PORT, limit: int = 200) -> int:
    for port in range(start, start + limit):
        if _port_available(port):
            return port
    raise ValueError(f"no free loopback port found in range {start}-{start + limit - 1}")


def quickstart_run_args(paths: QuickstartPaths, *, profile: str = DEFAULT_PROFILE, agent: str = DEFAULT_AGENT) -> dict[str, str]:
    return {
        "suite": str(paths.suite_path),
        "model_config": str(paths.model_config_path),
        "out": str(paths.results_root),
        "workspace_root": str(paths.workspace_root),
        "fixtures_root": str(paths.fixtures_root),
        "openclaw_profile": profile,
        "openclaw_agent": agent,
    }


def _quickstart_paths(bench_root: Path, home: Path, profile: str) -> QuickstartPaths:
    manifest_dir = bench_root / "manifests"
    return QuickstartPaths(
        bench_root=bench_root,
        results_root=bench_root / "results",
        workspace_root=bench_root / "workspaces" / "quickstart",
        fixtures_root=bench_root / "fixtures",
        manifest_dir=manifest_dir,
        config_path=home / f".openclaw-{profile}" / "openclaw.json",
        suite_path=manifest_dir / "starter-suite.json",
        model_config_path=manifest_dir / "starter-models.json",
        metadata_path=bench_root / "quickstart.json",
    )


def _check_reusable_metadata(metadata_path: Path, providers: str) -> None:
    if not metadata_path.exists():
        return
    try:
        metadata = _read_json(metadata_path)
    except (OSError, json.JSONDecodeError):
        return
    previous = metadata.get("providers")
    if previous and previous != providers:
        raise ValueError(
            f"existing quickstart metadata was generated for providers={previous}; "
            f"pass --force to regenerate it for providers={providers}"
        )


def _selected_model_manifest(providers: str, project_root: Path) -> dict:
    models = []
    source_manifests = []
    if providers in {"local", "both"}:
        local = _read_asset_json(project_root, "manifests", LOCAL_MODEL_MANIFEST)
        models.extend(local["models"])
        source_manifests.append(LOCAL_MODEL_MANIFEST)
    if providers in {"api", "both"}:
        api = _read_asset_json(project_root, "manifests", API_MODEL_MANIFEST)
        models.extend(api["models"])
        source_manifests.append(API_MODEL_MANIFEST)
    return {
        "manifest_scope": {
            "portability": "quickstart_generated",
            "notes": "Generated by oc-bench init. API keys stay in environment variables; OAuth providers are bring-your-own-auth in this phase.",
            "source_manifests": source_manifests,
        },
        "models": models,
    }


def _openclaw_config(providers: str, project_root: Path, port: int, agent: str) -> dict:
    provider_config = {}
    if providers in {"local", "both"}:
        provider_config["vllm"] = _read_asset_json(project_root, "openclaw-config", "vllm-provider-smoke.example.json")
    if providers in {"api", "both"}:
        provider_config["openai"] = _read_asset_json(project_root, "openclaw-config", "openai-provider.example.json")
        provider_config["anthropic"] = _read_asset_json(project_root, "openclaw-config", "anthropic-provider.example.json")

    default_model = _default_route_model(providers)
    return {
        "commands": {
            "native": "auto",
            "nativeSkills": "auto",
            "restart": True,
            "ownerDisplay": "raw",
        },
        "env": {
            "vars": {
                "VLLM_API_KEY": "vllm-local",
            }
        },
        "gateway": {
            "mode": "local",
            "bind": "loopback",
            "port": port,
            "auth": {"mode": "none"},
            "tailscale": {"mode": "off"},
        },
        "models": {
            "providers": provider_config,
        },
        "agents": {
            "defaults": {
                "model": default_model,
                "params": {"maxTokens": 256},
            },
            "list": [
                {
                    "id": agent,
                    "model": default_model,
                    "tools": {"profile": "coding"},
                }
            ],
        },
    }


def _default_route_model(providers: str) -> str:
    if providers in {"local", "both"}:
        return "vllm/gpt-oss-20b-nvfp4-smoke"
    return "openai/gpt-4.1"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_asset_json(project_root: Path, *parts: str) -> dict:
    source = project_root.joinpath(*parts)
    if source.exists():
        return _read_json(source)
    asset = resources.files("openclaw_bench").joinpath("quickstart_assets", *parts)
    return json.loads(asset.read_text(encoding="utf-8"))


def _copy_asset_tree(project_root: Path, parts: tuple[str, ...], destination: Path) -> None:
    source = project_root.joinpath(*parts)
    if destination.exists():
        shutil.rmtree(destination)
    if source.exists():
        shutil.copytree(source, destination)
        return
    asset = resources.files("openclaw_bench").joinpath("quickstart_assets", *parts)
    _copy_traversable_tree(asset, destination)


def _copy_traversable_tree(source, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        if child.name == "__pycache__" or child.name.endswith(".pyc"):
            continue
        target = destination / child.name
        if child.is_dir():
            _copy_traversable_tree(child, target)
        else:
            target.write_text(child.read_text(encoding="utf-8"), encoding="utf-8")


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _trim_output(output: str, limit: int = 1000) -> str:
    stripped = output.strip()
    return stripped[-limit:] if len(stripped) > limit else stripped
