from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .models import ModelSpec


DEFAULT_OPENCLAW_IMAGE = "clawdaddy/openclaw:business-smoke-2026.4.27"
DEFAULT_GATEWAY_TOKEN = "oc-bench-token"
DEFAULT_GATEWAY_PORT = 19091


@dataclass(frozen=True)
class ContainerEnsureResult:
    name: str
    status: str
    notes: str

    def to_row(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "notes": self.notes}


def runtime_kind_target(runtime: str) -> tuple[str, str]:
    if ":" not in runtime:
        return "docker", runtime
    kind, _, target = runtime.partition(":")
    kind = kind.strip().lower()
    target = target.strip()
    if kind not in {"docker", "incus"} or not target:
        raise ValueError(f"unsupported OpenClaw runtime {runtime!r}; use docker:<container> or incus:<instance>")
    return kind, target


def runtime_exec_prefix(runtime: str) -> list[str]:
    kind, target = runtime_kind_target(runtime)
    if kind == "incus":
        return ["incus", "exec", target, "--"]
    return ["docker", "exec", target]


def ensure_openclaw_container(
    *,
    container: str,
    image: str,
    profile: str,
    project_root: Path,
    bench_root: Path,
    workspace_root: Path,
    container_home: Path | None = None,
    gateway_port: int = DEFAULT_GATEWAY_PORT,
    gateway_token: str | None = None,
    models: list[ModelSpec] | None = None,
    timeout_s: int = 60,
) -> ContainerEnsureResult:
    """Ensure the isolated OpenClaw bench container exists and is running."""

    token = gateway_token or os.environ.get("OPENCLAW_GATEWAY_TOKEN") or DEFAULT_GATEWAY_TOKEN
    home = container_home or bench_root / "container-home"
    for path in (bench_root, home, workspace_root):
        path.mkdir(parents=True, exist_ok=True)

    inspect = _run(["docker", "inspect", "-f", "{{.State.Running}}", container], timeout_s=10)
    if inspect.returncode == 0:
        running = inspect.stdout.strip().lower() == "true"
        if running:
            return ContainerEnsureResult("openclaw_container", "pass", f"{container} already running")
        started = _run(["docker", "start", container], timeout_s=timeout_s)
        if started.returncode == 0:
            return ContainerEnsureResult("openclaw_container", "pass", f"{container} started")
        return ContainerEnsureResult("openclaw_container", "fail", _trim_output(started.stderr or started.stdout))

    run = _run(
        _docker_run_command(
            container=container,
            image=image,
            profile=profile,
            project_root=project_root,
            bench_root=bench_root,
            workspace_root=workspace_root,
            container_home=home,
            gateway_port=gateway_port,
            gateway_token=token,
            models=models or [],
        ),
        timeout_s=timeout_s,
    )
    if run.returncode == 0:
        return ContainerEnsureResult("openclaw_container", "pass", f"{container} created from {image}")
    return ContainerEnsureResult("openclaw_container", "fail", _trim_output(run.stderr or run.stdout))


def _docker_run_command(
    *,
    container: str,
    image: str,
    profile: str,
    project_root: Path,
    bench_root: Path,
    workspace_root: Path,
    container_home: Path,
    gateway_port: int,
    gateway_token: str,
    models: list[ModelSpec],
) -> list[str]:
    env = _container_env(models, gateway_token)
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        container,
        "--network",
        "host",
        "--health-cmd",
        f"openclaw --profile {shlex.quote(profile)} gateway status 2>/dev/null | grep -q 'Connectivity probe: ok'",
        "--health-interval",
        "30s",
        "--health-timeout",
        "5s",
        "--health-start-period",
        "30s",
        "--health-retries",
        "3",
    ]
    for host_path, container_path in _mounts(project_root, bench_root, workspace_root, container_home):
        cmd.extend(["-v", f"{host_path}:{container_path}"])
    for key, value in env.items():
        cmd.extend(["-e", f"{key}={value}"])
    cmd.extend(
        [
            image,
            "sh",
            "-lc",
            "sleep infinity",
        ]
    )
    return cmd


def _container_env(models: list[ModelSpec], gateway_token: str) -> dict[str, str]:
    env = {
        "HOME": "/home/ubuntu",
        "OPENCLAW_LOG_LEVEL": "debug",
        "OPENCLAW_GATEWAY_TOKEN": gateway_token,
        "VLLM_API_KEY": os.environ.get("VLLM_API_KEY", "vllm-local"),
    }
    for model in models:
        if model.api_env and model.api_env in os.environ:
            env[model.api_env] = os.environ[model.api_env]
    return env


def _mounts(project_root: Path, bench_root: Path, workspace_root: Path, container_home: Path) -> list[tuple[Path, Path]]:
    mounts = [
        (project_root.resolve(), project_root.resolve()),
        (container_home.resolve(), Path("/home/ubuntu")),
        (bench_root.resolve(), bench_root.resolve()),
    ]
    resolved_workspace = workspace_root.resolve()
    if not _is_relative_to(resolved_workspace, project_root.resolve()) and not _is_relative_to(resolved_workspace, bench_root.resolve()):
        mounts.append((resolved_workspace, resolved_workspace))
    deduped: list[tuple[Path, Path]] = []
    seen: set[tuple[str, str]] = set()
    for host_path, container_path in mounts:
        key = (str(host_path), str(container_path))
        if key not in seen:
            seen.add(key)
            deduped.append((host_path, container_path))
    return deduped


def gateway_run_command(profile: str, gateway_port: int = DEFAULT_GATEWAY_PORT) -> str:
    return (
        "exec openclaw "
        f"--profile {shlex.quote(profile)} "
        "gateway "
        f"--port {int(gateway_port)} "
        "--bind loopback "
        "--auth token "
        '--token "$OPENCLAW_GATEWAY_TOKEN" '
        "--allow-unconfigured "
        "--verbose "
        "--ws-log compact "
        "--raw-stream "
        "--raw-stream-path /home/ubuntu/openclaw-bench/oc-bench-raw-stream.jsonl "
        "run"
    )


def _run(cmd: list[str], timeout_s: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_s, check=False)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _trim_output(output: str, limit: int = 1000) -> str:
    stripped = output.strip()
    return stripped[-limit:] if len(stripped) > limit else stripped
