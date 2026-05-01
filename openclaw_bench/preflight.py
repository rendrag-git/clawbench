from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .backend import OpenClawBackend
from .container import DEFAULT_GATEWAY_PORT, gateway_run_command
from .models import ModelSpec, SuiteManifest, TaskSpec
from .workspace import OPENCLAW_SEED_FILES, copy_fixture, seed_openclaw_workspace_files


OPENCLAW_PINNED_VERSION = "2026.4.27"
OPENCLAW_BLOCKED_VERSION = "2026.4.29"


@dataclass
class PreflightCheck:
    name: str
    status: str
    notes: str = ""

    def to_row(self) -> dict:
        return {"name": self.name, "status": self.status, "notes": self.notes}


@dataclass
class PreflightResult:
    checks: list[PreflightCheck]

    @property
    def ok(self) -> bool:
        return all(check.status != "fail" for check in self.checks)

    def to_json(self) -> str:
        return json.dumps({"ok": self.ok, "checks": [check.to_row() for check in self.checks]}, indent=2, sort_keys=True)


def run_preflight(
    suite: SuiteManifest,
    models: list[ModelSpec],
    backend_name: str,
    out_dir: Path,
    workspace_root: Path,
    fixtures_root: Path,
    openclaw_profile: str,
    openclaw_local: bool,
    openclaw_agent: str = "main",
    openclaw_container: str | None = None,
    openclaw_workspace_agents: bool = False,
    ensure_gateway: bool = False,
    gateway_timeout_s: int = 60,
    smoke_turn: bool = False,
    agent_smoke_turn: bool = False,
    smoke_timeout_s: int = 60,
) -> PreflightResult:
    checks: list[PreflightCheck] = []
    checks.extend(_check_suite(suite, fixtures_root))
    checks.append(_check_output_path(out_dir, "output_path"))
    checks.append(_check_output_path(workspace_root, "workspace_root"))
    if backend_name == "openclaw" and openclaw_container:
        checks.append(_check_container_path(workspace_root, openclaw_container, "container_workspace_root"))
    checks.extend(_check_models(models, backend_name, smoke_turn or agent_smoke_turn))
    if backend_name == "openclaw":
        checks.extend(
            _check_openclaw(
                openclaw_profile,
                openclaw_local,
                openclaw_container,
                ensure_gateway=ensure_gateway,
                gateway_timeout_s=gateway_timeout_s,
            )
        )
        if smoke_turn:
            checks.extend(_check_openclaw_model_routes(models, openclaw_profile, openclaw_local, smoke_timeout_s, openclaw_container))
        if agent_smoke_turn:
            checks.extend(
                _check_openclaw_agent_routes(
                    suite=suite,
                    models=models,
                    profile=openclaw_profile,
                    agent=openclaw_agent,
                    local=openclaw_local,
                    workspace_agents=openclaw_workspace_agents,
                    container=openclaw_container,
                    workspace_root=workspace_root,
                    fixtures_root=fixtures_root,
                    timeout_s=smoke_timeout_s,
                )
            )
    return PreflightResult(checks=checks)


def render_text(result: PreflightResult) -> str:
    lines = [f"preflight={'ok' if result.ok else 'failed'}"]
    for check in result.checks:
        suffix = f" - {check.notes}" if check.notes else ""
        lines.append(f"{check.status.upper()} {check.name}{suffix}")
    return "\n".join(lines)


def _check_suite(suite: SuiteManifest, fixtures_root: Path) -> list[PreflightCheck]:
    checks = [PreflightCheck("suite_manifest", "pass", f"{suite.suite_id}: {len(suite.tasks)} tasks")]
    task_ids = set()
    for task in suite.tasks:
        if task.task_id in task_ids:
            checks.append(PreflightCheck(f"task:{task.task_id}", "fail", "duplicate task_id"))
            continue
        task_ids.add(task.task_id)
        fixture = fixtures_root / task.fixture
        if not fixture.exists():
            checks.append(PreflightCheck(f"fixture:{task.task_id}", "fail", f"missing {fixture}"))
        elif not fixture.is_dir():
            checks.append(PreflightCheck(f"fixture:{task.task_id}", "fail", f"not a directory: {fixture}"))
        else:
            checks.append(PreflightCheck(f"fixture:{task.task_id}", "pass", str(fixture)))
            checks.extend(_check_task_expected_paths(task, fixture))
            min_fixture_chars = task.expected.get("min_fixture_chars")
            if isinstance(min_fixture_chars, int):
                fixture_chars = _fixture_text_chars(fixture)
                if fixture_chars < min_fixture_chars:
                    checks.append(
                        PreflightCheck(
                            f"fixture_size:{task.task_id}",
                            "fail",
                            f"fixture has {fixture_chars} text chars; expected at least {min_fixture_chars}",
                        )
                    )
                else:
                    checks.append(PreflightCheck(f"fixture_size:{task.task_id}", "pass", f"{fixture_chars} text chars"))
    return checks


def _check_task_expected_paths(task: object, fixture: Path) -> list[PreflightCheck]:
    expected = getattr(task, "expected", {})
    task_id = getattr(task, "task_id", "unknown-task")
    task_type = getattr(task, "task_type", "")
    if not isinstance(expected, dict):
        return [PreflightCheck(f"expected:{task_id}", "fail", "expected must be an object")]
    if task_type == "workspace_discovery":
        required = ["test_command", "routes_file", "schema_file"]
        checks = [_check_required_expected_keys(task_id, expected, required)]
        checks.extend(_check_fixture_file_refs(task_id, fixture, expected, ["routes_file", "schema_file"]))
        return checks
    if task_type == "repo_read_only":
        checks = [_check_required_expected_keys(task_id, expected, ["answer", "evidence_files"])]
        checks.extend(_check_fixture_file_refs(task_id, fixture, expected, ["answer"]))
        evidence_files = expected.get("evidence_files")
        if isinstance(evidence_files, list):
            checks.extend(_check_fixture_paths(task_id, fixture, evidence_files, "evidence_files"))
        else:
            checks.append(PreflightCheck(f"expected_paths:{task_id}:evidence_files", "fail", "evidence_files must be a list"))
        return checks
    if task_type == "workspace_needle":
        required = ["needle", "distractor", "source_file", "target_file"]
        checks = [_check_required_expected_keys(task_id, expected, required)]
        checks.extend(_check_fixture_file_refs(task_id, fixture, expected, ["source_file", "target_file"]))
        return checks
    if task_type == "action_gate_triage":
        checks = [_check_required_expected_keys(task_id, expected, ["decision", "evidence_files", "preserved_files", "max_tool_calls"])]
        evidence_files = expected.get("evidence_files")
        preserved_files = expected.get("preserved_files")
        if isinstance(evidence_files, list):
            checks.extend(_check_fixture_paths(task_id, fixture, evidence_files, "evidence_files"))
        else:
            checks.append(PreflightCheck(f"expected_paths:{task_id}:evidence_files", "fail", "evidence_files must be a list"))
        if isinstance(preserved_files, list):
            checks.extend(_check_fixture_paths(task_id, fixture, preserved_files, "preserved_files"))
        else:
            checks.append(PreflightCheck(f"expected_paths:{task_id}:preserved_files", "fail", "preserved_files must be a list"))
        return checks
    if task_type == "agents_soul_adherence":
        checks = [_check_required_expected_keys(task_id, expected, ["changed_files", "target_file", "policy_files", "forbidden_changed_files", "behavior_checks"])]
        checks.extend(_check_fixture_file_refs(task_id, fixture, expected, ["target_file"]))
        policy_files = expected.get("policy_files")
        if isinstance(policy_files, list):
            fixture_policy_files = [path for path in policy_files if path not in OPENCLAW_SEED_FILES]
            checks.extend(_check_fixture_paths(task_id, fixture, fixture_policy_files, "policy_files"))
        else:
            checks.append(PreflightCheck(f"expected_paths:{task_id}:policy_files", "fail", "policy_files must be a list"))
        return checks
    if task_type == "format_drift_under_length":
        required = [
            "decision",
            "owner",
            "risk_count",
            "trail_length",
            "checksum",
            "final_file",
            "source_file",
            "trail_files",
            "required_keys",
            "min_tool_calls",
            "max_tool_calls",
            "max_response_chars",
        ]
        checks = [_check_required_expected_keys(task_id, expected, required)]
        checks.extend(_check_fixture_file_refs(task_id, fixture, expected, ["source_file", "final_file"]))
        trail_files = expected.get("trail_files")
        if isinstance(trail_files, list):
            checks.extend(_check_fixture_paths(task_id, fixture, trail_files, "trail_files"))
        else:
            checks.append(PreflightCheck(f"expected_paths:{task_id}:trail_files", "fail", "trail_files must be a list"))
        return checks
    return []


def _check_required_expected_keys(task_id: str, expected: dict, keys: list[str]) -> PreflightCheck:
    missing = [key for key in keys if key not in expected]
    if missing:
        return PreflightCheck(f"expected:{task_id}", "fail", "missing " + ", ".join(missing))
    return PreflightCheck(f"expected:{task_id}", "pass", "required expected fields present")


def _check_fixture_file_refs(task_id: str, fixture: Path, expected: dict, keys: list[str]) -> list[PreflightCheck]:
    paths = [expected.get(key) for key in keys if isinstance(expected.get(key), str)]
    return _check_fixture_paths(task_id, fixture, paths, ",".join(keys))


def _check_fixture_paths(task_id: str, fixture: Path, paths: list[object], label: str) -> list[PreflightCheck]:
    invalid = [path for path in paths if not isinstance(path, str) or not path]
    missing = [path for path in paths if isinstance(path, str) and path and not (fixture / path).is_file()]
    if invalid:
        return [PreflightCheck(f"expected_paths:{task_id}:{label}", "fail", "expected path values must be non-empty strings")]
    if missing:
        return [PreflightCheck(f"expected_paths:{task_id}:{label}", "fail", "missing " + ", ".join(missing[:8]))]
    if paths:
        return [PreflightCheck(f"expected_paths:{task_id}:{label}", "pass", f"{len(paths)} path(s) exist")]
    return []


def _check_output_path(path: Path, name: str) -> PreflightCheck:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".openclaw-bench-preflight-", dir=path, delete=True) as handle:
            handle.write(b"")
    except OSError as exc:
        return PreflightCheck(name, "fail", str(exc))
    return PreflightCheck(name, "pass", str(path))


def _check_container_path(path: Path, container: str, name: str) -> PreflightCheck:
    cmd = ["docker", "exec", container, "test", "-d", str(path), "-a", "-w", str(path)]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return PreflightCheck(name, "fail", f"{path} is not visible in {container}: {exc}")
    if proc.returncode == 0:
        return PreflightCheck(name, "pass", f"{path} is visible and writable inside {container}")
    output = f"{proc.stdout}\n{proc.stderr}".strip()
    notes = output or f"{path} is missing or not writable inside {container}"
    return PreflightCheck(name, "fail", _trim_output(notes))


def _fixture_text_chars(fixture: Path) -> int:
    total = 0
    for path in fixture.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        try:
            total += len(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue
    return total


def _check_models(models: list[ModelSpec], backend_name: str, smoke_turn: bool = False) -> list[PreflightCheck]:
    if not models:
        return [PreflightCheck("models", "fail", "no models configured")]
    checks = [PreflightCheck("models", "pass", f"{len(models)} model/context/KV cells")]
    for model in models:
        label = f"model:{model.served_model_name}:{model.kv_cache_dtype}:ctx{model.context_limit}"
        if model.support_status in {"unsupported", "unsupported_kv_dtype", "false"}:
            checks.append(PreflightCheck(label, "fail", "marked unsupported"))
            continue
        if model.provider_type in {"api", "subscription"}:
            checks.append(_check_api_model(label, model, smoke_turn))
            continue
        if backend_name == "openclaw":
            if model.api_env and not os.environ.get(model.api_env):
                checks.append(PreflightCheck(label, "fail", f"missing environment variable {model.api_env}"))
                continue
            if model.serve_command and not model.health_check_url:
                checks.append(PreflightCheck(label, "fail", "serve_command requires health_check_url for live readiness"))
            elif model.health_check_url and _looks_openai_compatible_health_url(model.health_check_url) and not model.api_base:
                checks.append(PreflightCheck(label, "fail", "OpenAI-compatible health_check_url requires api_base so chat route probing can run"))
            elif model.serve_command and not _serve_command_available(model.serve_command):
                checks.append(PreflightCheck(label, "fail", f"serve command not found: {model.serve_command[0]}"))
            elif model.health_check_url:
                checks.append(PreflightCheck(label, "pass", "live readiness will be checked with health_check_url"))
            elif model.support_status == "validated_external":
                checks.append(PreflightCheck(label, "warn", "external readiness is marked validated but has no health_check_url"))
            elif not model.serve_command:
                checks.append(PreflightCheck(label, "fail", "local live model needs serve_command+health_check_url or health_check_url for an external endpoint"))
            else:
                checks.append(PreflightCheck(label, "pass", model.expected_support))
        else:
            checks.append(PreflightCheck(label, "pass", "simulator backend"))
    return checks


def _check_api_model(label: str, model: ModelSpec, smoke_turn: bool = False) -> PreflightCheck:
    if model.api_env and not os.environ.get(model.api_env):
        return PreflightCheck(label, "fail", f"missing environment variable {model.api_env}")
    if not model.api_env:
        return PreflightCheck(label, "fail", "api/subscription model needs api_env or validated_external support_status")
    if smoke_turn:
        return PreflightCheck(label, "pass", "provider credentials exist; OpenClaw smoke will run")
    if model.support_status == "validated_external":
        return PreflightCheck(label, "warn", "provider credentials exist, but no smoke turn was run by preflight")
    return PreflightCheck(label, "fail", "provider env is present, but live OpenClaw smoke still needs to run before certification")


def _serve_command_available(command: list[str]) -> bool:
    if not command:
        return False
    return shutil.which(command[0]) is not None


def _looks_openai_compatible_health_url(health_check_url: str) -> bool:
    return health_check_url.rstrip("/").endswith("/v1/models")


def _check_openclaw(
    profile: str,
    local: bool,
    container: str | None = None,
    ensure_gateway: bool = False,
    gateway_timeout_s: int = 60,
) -> list[PreflightCheck]:
    checks = []
    if container:
        docker = shutil.which("docker")
        if docker is None:
            return [PreflightCheck("openclaw_cli", "fail", "docker executable not found for --openclaw-container")]
        checks.append(PreflightCheck("openclaw_cli", "pass", f"docker exec {container} openclaw"))
    else:
        executable = shutil.which("openclaw")
        if executable is None:
            return [PreflightCheck("openclaw_cli", "fail", "openclaw executable not found")]
        checks.append(PreflightCheck("openclaw_cli", "pass", executable))
    version_check = check_openclaw_version(container)
    checks.append(version_check)
    if version_check.status == "fail" and not local:
        return checks
    if local:
        checks.append(_check_profile_config(profile, container, local=True))
        checks.append(PreflightCheck("openclaw_gateway", "warn", "--openclaw-local skips gateway requirement"))
        return checks
    if ensure_gateway:
        checks.append(ensure_openclaw_gateway(profile, container, timeout_s=gateway_timeout_s))
        checks.append(_check_profile_config(profile, container, local=False))
        return checks
    checks.append(_check_profile_config(profile, container, local=False))
    if checks[-1].status == "fail" and checks[-1].name == "openclaw_profile_config":
        return checks
    checks.append(_check_gateway(profile, container))
    return checks


def _check_profile_config(profile: str, container: str | None, local: bool) -> PreflightCheck:
    if container:
        return _check_container_config(profile, container)
    config_path = _openclaw_config_path(profile)
    if config_path.exists():
        return PreflightCheck("openclaw_profile_config", "pass", str(config_path))
    if local:
        return PreflightCheck("openclaw_profile_config", "warn", f"missing {config_path}; --openclaw-local may rely on shell provider keys")
    return PreflightCheck("openclaw_profile_config", "fail", f"missing {config_path}")


def _check_container_config(profile: str, container: str) -> PreflightCheck:
    cmd = [*_openclaw_cmd(container), "--profile", profile, "config", "validate"]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return PreflightCheck("openclaw_profile_config", "fail", str(exc))
    output = f"{proc.stdout}\n{proc.stderr}".strip()
    if proc.returncode == 0:
        return PreflightCheck("openclaw_profile_config", "pass", f"profile {profile} validates inside {container}")
    return PreflightCheck("openclaw_profile_config", "fail", _trim_output(output))


def check_openclaw_version(container: str | None = None) -> PreflightCheck:
    cmd = [*_openclaw_cmd(container), "--version"]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return PreflightCheck("openclaw_version", "fail", str(exc))
    output = f"{proc.stdout}\n{proc.stderr}".strip()
    if proc.returncode != 0:
        return PreflightCheck("openclaw_version", "fail", _trim_output(output))
    version = _parse_openclaw_version(output)
    if version is None:
        return PreflightCheck("openclaw_version", "fail", f"could not parse OpenClaw version from: {_trim_output(output)}")
    if version != OPENCLAW_PINNED_VERSION:
        return PreflightCheck(
            "openclaw_version",
            "fail",
            (
                f"OpenClaw {version} does not match pinned version {OPENCLAW_PINNED_VERSION}; "
                f"{OPENCLAW_BLOCKED_VERSION} is currently blocked for bench runs. "
                f"Use npm install -g openclaw@{OPENCLAW_PINNED_VERSION}."
            ),
        )
    return PreflightCheck("openclaw_version", "pass", f"OpenClaw {version} matches pinned version {OPENCLAW_PINNED_VERSION}")


def _parse_openclaw_version(output: str) -> str | None:
    for token in output.replace("(", " ").split():
        parts = token.split(".")
        if len(parts) != 3:
            continue
        try:
            tuple(int(part) for part in parts)
        except ValueError:
            continue
        return token
    return None


def _check_openclaw_model_routes(
    models: list[ModelSpec],
    profile: str,
    local: bool,
    timeout_s: int,
    container: str | None = None,
) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    seen: dict[str, PreflightCheck] = {}
    for model in models:
        route_model = model.openclaw_route_model
        label = f"openclaw_route:{route_model}:{model.kv_cache_dtype}:ctx{model.context_limit}"
        if model.support_status in {"unsupported", "unsupported_kv_dtype", "false"}:
            checks.append(PreflightCheck(label, "warn", "skipped route smoke for unsupported model cell"))
            continue
        if model.api_env and not os.environ.get(model.api_env):
            check = PreflightCheck(label, "fail", f"skipped route smoke; missing environment variable {model.api_env}")
            checks.append(check)
            seen.setdefault(route_model, check)
            continue
        if model.provider_type == "local" and model.serve_command:
            checks.append(PreflightCheck(label, "warn", "skipped route smoke; model server is started during run"))
            continue
        if route_model in seen:
            previous = seen[route_model]
            checks.append(PreflightCheck(label, previous.status, f"same OpenClaw route model already smoke-tested: {previous.status}"))
            continue
        check = _run_openclaw_route_smoke(label, route_model, profile, local, timeout_s, container)
        checks.append(check)
        seen[route_model] = check
    return checks


def _check_openclaw_agent_routes(
    suite: SuiteManifest,
    models: list[ModelSpec],
    profile: str,
    agent: str,
    local: bool,
    workspace_agents: bool,
    container: str | None,
    workspace_root: Path,
    fixtures_root: Path,
    timeout_s: int,
) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    backend = OpenClawBackend(
        profile=profile,
        agent=agent,
        local=local,
        workspace_agents=workspace_agents,
        container=container,
    )
    for model in models:
        route_model = model.openclaw_route_model
        label = f"openclaw_agent:{route_model}:{model.kv_cache_dtype}:ctx{model.context_limit}"
        if model.support_status in {"unsupported", "unsupported_kv_dtype", "false"}:
            checks.append(PreflightCheck(label, "warn", "skipped agent smoke for unsupported model cell"))
            continue
        if model.api_env and not os.environ.get(model.api_env):
            checks.append(PreflightCheck(label, "fail", f"skipped agent smoke; missing environment variable {model.api_env}"))
            continue
        task = _first_task_for_context(suite, model.context_limit)
        if task is None:
            checks.append(PreflightCheck(label, "warn", f"skipped agent smoke; suite has no task for ctx{model.context_limit}"))
            continue
        fixture = fixtures_root / task.fixture
        workspace = workspace_root / "_preflight_agent_smoke" / _safe_path_part(f"{model.comparison_key}-{model.kv_cache_dtype}-ctx{model.context_limit}")
        try:
            copy_fixture(fixture, workspace)
            seed_openclaw_workspace_files(
                workspace,
                agent_id=f"{agent}-preflight",
                task_id=task.task_id,
                model_id=model.served_model_name,
            )
        except OSError as exc:
            checks.append(PreflightCheck(label, "fail", f"agent smoke workspace setup failed: {exc}"))
            continue
        session_id = f"preflight-agent-smoke-{_safe_path_part(model.comparison_key)[:48]}"
        response = backend.run(model, task, workspace, session_id, timeout_s)
        if response.error:
            checks.append(PreflightCheck(label, "fail", _trim_output(f"{response.error}: {response.raw.get('stderr', '') or response.text}")))
            continue
        telemetry_error = _agent_smoke_telemetry_error(response)
        if telemetry_error:
            checks.append(PreflightCheck(label, "fail", telemetry_error))
            continue
        checks.append(PreflightCheck(label, "pass", "OpenClaw agent smoke succeeded with certification telemetry"))
    return checks


def _agent_smoke_telemetry_error(response: BackendResponse) -> str:
    missing = []
    if response.tool_calls <= 0:
        missing.append("tool_calls")
    if response.files_read <= 0:
        missing.append("files_read")
    if response.duplicate_file_reads is None or response.duplicate_file_reads < 0:
        missing.append("duplicate_file_reads")
    if response.time_to_first_relevant_file_s is None or response.time_to_first_relevant_file_s < 0:
        missing.append("time_to_first_relevant_file_s")
    if missing:
        return "OpenClaw agent smoke succeeded but missing certification telemetry: " + ", ".join(missing)
    return ""


def _first_task_for_context(suite: SuiteManifest, context_limit: int) -> TaskSpec | None:
    for task in suite.tasks:
        if not task.context_sizes or context_limit in task.context_sizes:
            return task
    return None


def _run_openclaw_route_smoke(
    label: str,
    route_model: str,
    profile: str,
    local: bool,
    timeout_s: int,
    container: str | None = None,
) -> PreflightCheck:
    cmd = [
        *_openclaw_cmd(container),
        "--profile",
        profile,
        "infer",
        "model",
        "run",
        "--model",
        route_model,
        "--prompt",
        "Reply with exactly: ok",
        "--json",
    ]
    cmd.append("--local" if local else "--gateway")
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return PreflightCheck(label, "fail", f"OpenClaw route smoke failed: {exc}")
    output = f"{proc.stdout}\n{proc.stderr}".strip()
    if proc.returncode != 0:
        return PreflightCheck(label, "fail", _trim_output(output))
    if not output:
        return PreflightCheck(label, "fail", "OpenClaw route smoke returned no output")
    return PreflightCheck(label, "pass", "OpenClaw model route smoke succeeded")


def _openclaw_config_path(profile: str) -> Path:
    if profile in {"", "default"}:
        return Path.home() / ".openclaw" / "openclaw.json"
    return Path.home() / f".openclaw-{profile}" / "openclaw.json"


def _check_gateway(profile: str, container: str | None = None) -> PreflightCheck:
    cmd = [*_openclaw_cmd(container), "--profile", profile, "gateway", "status"]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return PreflightCheck("openclaw_gateway", "fail", str(exc))
    output = f"{proc.stdout}\n{proc.stderr}"
    if proc.returncode != 0:
        return PreflightCheck("openclaw_gateway", "fail", output.strip()[-1000:])
    if "Connectivity probe: failed" in output:
        return PreflightCheck("openclaw_gateway", "fail", _compact_gateway_status(output))
    if "Connectivity probe: ok" in output:
        return PreflightCheck("openclaw_gateway", "pass", _compact_gateway_status(output))
    if "Runtime: stopped" in output:
        return PreflightCheck("openclaw_gateway", "fail", _compact_gateway_status(output))
    return PreflightCheck("openclaw_gateway", "pass", _compact_gateway_status(output))


def ensure_openclaw_gateway(profile: str, container: str | None = None, timeout_s: int = 60) -> PreflightCheck:
    current = _check_gateway(profile, container)
    if current.status == "pass":
        return PreflightCheck("openclaw_gateway", "pass", _join_gateway_notes("already running", current.notes))

    start = _start_gateway_process(profile, container, timeout_s)
    start_output = start.notes
    if start.status == "fail":
        return PreflightCheck(
            "openclaw_gateway",
            "fail",
            _trim_output(f"gateway auto-start failed: {start_output}; previous status: {current.notes}"),
        )

    deadline = time.monotonic() + max(timeout_s, 0)
    after = _check_gateway(profile, container)
    while after.status != "pass" and time.monotonic() < deadline:
        time.sleep(1)
        after = _check_gateway(profile, container)
    if after.status == "pass":
        return PreflightCheck("openclaw_gateway", "pass", _join_gateway_notes("started bench gateway", after.notes))
    return PreflightCheck(
        "openclaw_gateway",
        "fail",
        _trim_output(f"gateway auto-start ran but status is still {after.status}: {after.notes}; start output: {start_output}"),
    )


def stop_openclaw_gateway(profile: str, timeout_s: int = 30) -> PreflightCheck:
    foreground = _stop_foreground_gateway(profile, timeout_s)
    if foreground.status == "pass":
        return PreflightCheck("openclaw_gateway_stop", "pass", foreground.notes)

    cmd = ["openclaw", "--profile", profile, "gateway", "stop"]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return PreflightCheck("openclaw_gateway_stop", "fail", str(exc))
    output = f"{proc.stdout}\n{proc.stderr}".strip()
    if proc.returncode != 0:
        notes = _trim_output(output)
        if foreground.status == "fail":
            notes = _join_gateway_notes(notes, foreground.notes)
        return PreflightCheck("openclaw_gateway_stop", "fail", notes)
    if foreground.status == "fail":
        return PreflightCheck("openclaw_gateway_stop", "fail", foreground.notes)
    return PreflightCheck("openclaw_gateway_stop", "pass", _trim_output(output) or f"stopped {profile} gateway")


def _compact_gateway_status(output: str) -> str:
    lines = []
    for line in output.splitlines():
        if any(marker in line for marker in ("Runtime:", "Connectivity probe:", "Config (cli):", "Probe target:", "Port ")):
            lines.append(line.strip())
    return "; ".join(lines[-6:])


def _trim_output(output: str, limit: int = 1000) -> str:
    return output[-limit:] if len(output) > limit else output


def _join_gateway_notes(prefix: str, notes: str) -> str:
    return f"{prefix}; {notes}" if notes else prefix


def _openclaw_cmd(container: str | None = None) -> list[str]:
    if not container:
        return ["openclaw"]
    return ["docker", "exec", container, "openclaw"]


def _start_gateway_process(profile: str, container: str | None = None, timeout_s: int = 60) -> PreflightCheck:
    if container:
        start_cmd = _gateway_start_cmd(profile, container)
        try:
            start = subprocess.run(start_cmd, text=True, capture_output=True, timeout=max(timeout_s, 1), check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return PreflightCheck("openclaw_gateway_start", "fail", str(exc))
        output = f"{start.stdout}\n{start.stderr}".strip()
        if start.returncode != 0:
            return PreflightCheck("openclaw_gateway_start", "fail", _trim_output(output))
        return PreflightCheck("openclaw_gateway_start", "pass", _trim_output(output) or "started detached gateway in container")

    start_cmd = _gateway_start_cmd(profile, None)
    try:
        process = subprocess.Popen(
            start_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        return PreflightCheck("openclaw_gateway_start", "fail", str(exc))
    try:
        _gateway_pid_path(profile).write_text(str(process.pid), encoding="utf-8")
    except OSError as exc:
        return PreflightCheck("openclaw_gateway_start", "fail", f"launched pid {process.pid} but could not write pid file: {exc}")
    return PreflightCheck("openclaw_gateway_start", "pass", f"launched foreground gateway pid {process.pid}")


def _gateway_start_cmd(profile: str, container: str | None = None) -> list[str]:
    if container:
        return ["docker", "exec", "-d", container, "sh", "-lc", gateway_run_command(profile, DEFAULT_GATEWAY_PORT)]
    return ["openclaw", "--profile", profile, "gateway", "--dev", "--verbose", "run"]


def _stop_foreground_gateway(profile: str, timeout_s: int = 30) -> PreflightCheck:
    pid_path = _gateway_pid_path(profile)
    if not pid_path.exists():
        return PreflightCheck("openclaw_gateway_foreground_stop", "warn", f"no foreground pid file at {pid_path}")
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as exc:
        return PreflightCheck("openclaw_gateway_foreground_stop", "fail", f"invalid foreground pid file {pid_path}: {exc}")
    if not _process_exists(pid):
        _unlink_pid_file(pid_path)
        return PreflightCheck("openclaw_gateway_foreground_stop", "pass", f"foreground gateway pid {pid} was already stopped")
    if not _pid_matches_gateway(pid):
        return PreflightCheck("openclaw_gateway_foreground_stop", "fail", f"pid {pid} does not look like a {profile} OpenClaw gateway")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _unlink_pid_file(pid_path)
        return PreflightCheck("openclaw_gateway_foreground_stop", "pass", f"foreground gateway pid {pid} was already stopped")
    except OSError as exc:
        return PreflightCheck("openclaw_gateway_foreground_stop", "fail", str(exc))

    deadline = time.monotonic() + max(timeout_s, 0)
    while time.monotonic() < deadline:
        if not _process_exists(pid):
            _unlink_pid_file(pid_path)
            return PreflightCheck("openclaw_gateway_foreground_stop", "pass", f"stopped foreground gateway pid {pid}")
        time.sleep(0.2)
    return PreflightCheck("openclaw_gateway_foreground_stop", "fail", f"foreground gateway pid {pid} did not stop within {timeout_s}s")


def _gateway_pid_path(profile: str) -> Path:
    return Path(tempfile.gettempdir()) / f"openclaw-bench-gateway-{_safe_path_part(profile)}.pid"


def _pid_matches_gateway(pid: int) -> bool:
    cmdline_path = Path("/proc") / str(pid) / "cmdline"
    try:
        cmdline = cmdline_path.read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
    except FileNotFoundError:
        return False
    except OSError:
        return False
    return "openclaw" in cmdline


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return True


def _unlink_pid_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _safe_path_part(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)
    return safe.strip("-") or "model"
