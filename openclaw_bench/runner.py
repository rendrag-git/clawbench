from __future__ import annotations

import difflib
import hashlib
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from importlib import metadata
from pathlib import Path

from .backend import AgentBackend
from .models import AttemptResult, ModelSpec, SuiteManifest, TaskSpec
from .reporting import write_reports
from .scoring import json_valid, run_verify_command, score_task
from .serve import serve_model
from .telemetry import GpuTelemetrySampler, apply_gpu_telemetry, sample_nvidia_smi
from .workspace import changed_files, copy_fixture, read_text_files, seed_openclaw_workspace_files, snapshot_files


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    suite: SuiteManifest
    models: list[ModelSpec]
    kv_modes: list[str]
    contexts: list[int]
    concurrencies: list[int]
    out_dir: Path
    workspace_root: Path
    fixtures_root: Path
    backend_name: str
    suite_path: Path | None = None
    model_config_path: Path | None = None
    openclaw_profile: str = "bench"
    openclaw_agent: str = "main"
    openclaw_local: bool = False
    openclaw_container: str | None = None
    ensure_openclaw_gateway: bool = True
    openclaw_gateway_ensure: dict[str, str] | None = None
    openclaw_gateway_timeout_s: int = 60
    openclaw_workspace_agents: bool = False
    thinking: str | None = None
    timeout_s: int = 300
    openclaw_smoke_timeout_s: int = 60


class BenchmarkRunner:
    def __init__(self, backend: AgentBackend) -> None:
        self.backend = backend

    def run(self, config: RunConfig) -> list[AttemptResult]:
        raw_dir = config.out_dir / "raw"
        patch_dir = config.out_dir / "patches"
        raw_dir.mkdir(parents=True, exist_ok=True)
        patch_dir.mkdir(parents=True, exist_ok=True)
        config.workspace_root.mkdir(parents=True, exist_ok=True)

        server = {
            "run_id": config.run_id,
            "started": False,
            "hardware": sample_nvidia_smi().to_row(),
            "models": [model.__dict__ for model in config.models],
            "support_probes": [],
            "serve_results": [],
            "throughput_probes": [],
            "notes": "Server lifecycle is external unless model serve_command is configured.",
        }
        (config.out_dir / "config.json").write_text(
            json.dumps(
                {
                    "run_id": config.run_id,
                    "suite_id": config.suite.suite_id,
                    "models": [model.__dict__ for model in config.models],
                    "kv_modes": config.kv_modes,
                    "contexts": config.contexts,
                    "concurrencies": config.concurrencies,
                    "workspace_root": str(config.workspace_root),
                    "fixtures_root": str(config.fixtures_root),
                    "backend": config.backend_name,
                    "openclaw_profile": config.openclaw_profile,
                    "openclaw_agent": config.openclaw_agent,
                    "openclaw_local": config.openclaw_local,
                    "openclaw_container": config.openclaw_container,
                    "ensure_openclaw_gateway": config.ensure_openclaw_gateway,
                    "openclaw_gateway_ensure": config.openclaw_gateway_ensure,
                    "openclaw_gateway_timeout_s": config.openclaw_gateway_timeout_s,
                    "openclaw_workspace_agents": config.openclaw_workspace_agents,
                    "thinking": config.thinking,
                    "timeout_s": config.timeout_s,
                    "openclaw_smoke_timeout_s": config.openclaw_smoke_timeout_s,
                    "provenance": _run_provenance(config),
                    "runtime": _runtime_provenance(config),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        results: list[AttemptResult] = []
        for model in config.models:
            if model.support_status in {"unsupported", "unsupported_kv_dtype", "false"}:
                server["support_probes"].append(
                    {
                        "model": model.served_model_name,
                        "hardware_profile": model.hardware_profile,
                        "kv_cache_dtype": model.kv_cache_dtype,
                        "context_limit": model.context_limit,
                        "status": "failed",
                        "failure_type": "unsupported_kv_dtype",
                    }
                )
                results.extend(self._load_failure_results(config, model, "unsupported_kv_dtype"))
                continue
            if config.backend_name == "simulator":
                model = replace(model, support_status="simulator")
            with serve_model(model, timeout_s=min(config.timeout_s, 120)) as server_state:
                if server_state.load_success and config.backend_name == "openclaw":
                    smoke = self.backend.smoke(model, timeout_s=config.openclaw_smoke_timeout_s)
                    server_state.route_probe["openclaw_route"] = {
                        "success": smoke.error is None,
                        "error": smoke.error,
                        "timed_out": smoke.timed_out,
                        "raw": smoke.raw,
                    }
                    if smoke.error:
                        server_state.load_success = False
                        server_state.failure_type = smoke.error
                        server_state.request_errors += smoke.request_errors or 1
                        server_state.notes = _join_notes(server_state.notes, f"OpenClaw route smoke failed: {smoke.error}")
                    else:
                        server_state.notes = _join_notes(server_state.notes, "OpenClaw route smoke succeeded.")
                server_row = server_state.to_row()
                server_row["comparison_id"] = model.comparison_key
                server_row["provider_type"] = model.provider_type
                server_row["hardware_profile"] = model.hardware_profile
                server_row["weight_quant"] = model.weight_quant
                server_row["context_limit"] = model.context_limit
                server["serve_results"].append(server_row)
                throughput_probe = _throughput_probe_row(model, server_state.route_probe)
                if throughput_probe:
                    server["throughput_probes"].append(throughput_probe)
                server["support_probes"].append(
                    {
                        "model": model.served_model_name,
                        "hardware_profile": model.hardware_profile,
                        "kv_cache_dtype": model.kv_cache_dtype,
                        "context_limit": model.context_limit,
                        "status": "supported" if server_state.load_success else "failed",
                        "failure_type": server_state.failure_type,
                    }
                )
                if not server_state.load_success:
                    results.extend(
                        self._load_failure_results(
                            config,
                            model,
                            server_state.failure_type or "model_load_failed",
                            request_errors=server_state.request_errors,
                        )
                    )
                    continue
                for concurrency in config.concurrencies:
                    results.extend(self._run_matrix_cell(config, model, concurrency, raw_dir, patch_dir))
        write_reports(config.out_dir, results, server)
        return results

    def _load_failure_results(self, config: RunConfig, model: ModelSpec, failure_type: str, request_errors: int = 0) -> list[AttemptResult]:
        results: list[AttemptResult] = []
        for concurrency in config.concurrencies:
            for worker_index in range(concurrency):
                for task in config.suite.tasks:
                    if task.context_sizes and model.context_limit not in task.context_sizes:
                        continue
                    results.append(
                        AttemptResult(
                            run_id=config.run_id,
                            model=model.model_id,
                            served_model_name=model.served_model_name,
                            comparison_id=model.comparison_key,
                            backend=config.backend_name,
                            provider_type=model.provider_type,
                            hardware_profile=model.hardware_profile,
                            weight_quant=model.weight_quant,
                            kv_cache_dtype=model.kv_cache_dtype,
                            context_limit=model.context_limit,
                            concurrency=concurrency,
                            task_id=task.task_id,
                            task_type=task.task_type,
                            task_tags=task.tags,
                            workspace_id=_workspace_id(model, concurrency, worker_index, task),
                            status="fail",
                            score=0.0,
                            wall_time_s=0.0,
                            ttft_s=None,
                            tool_calls=0,
                            files_read=0,
                            duplicate_file_reads=None,
                            time_to_first_relevant_file_s=None,
                            files_changed=0,
                            changed_files=[],
                            tests_passed=False,
                            json_valid=False,
                            hallucinated_paths=0,
                            oom=failure_type in {"oom_on_load", "oom_during_run"},
                            timeout=failure_type in {"server_timeout", "openclaw_timeout"},
                            request_errors=request_errors,
                            failure_type=failure_type,
                            notes=_load_failure_note(failure_type),
                        )
                    )
        return results

    def _run_matrix_cell(
        self,
        config: RunConfig,
        model: ModelSpec,
        concurrency: int,
        raw_dir: Path,
        patch_dir: Path,
    ) -> list[AttemptResult]:
        attempts = []
        for worker_index in range(concurrency):
            for task in config.suite.tasks:
                if task.context_sizes and model.context_limit not in task.context_sizes:
                    continue
                attempts.append((worker_index, task))

        results: list[AttemptResult] = []
        with GpuTelemetrySampler() as telemetry:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [
                    pool.submit(self._run_attempt, config, model, concurrency, worker_index, task, raw_dir, patch_dir)
                    for worker_index, task in attempts
                ]
                for future in as_completed(futures):
                    results.append(future.result())
        gpu_telemetry = telemetry.result()
        for result in results:
            apply_gpu_telemetry(result, gpu_telemetry)
        return results

    def _run_attempt(
        self,
        config: RunConfig,
        model: ModelSpec,
        concurrency: int,
        worker_index: int,
        task: TaskSpec,
        raw_dir: Path,
        patch_dir: Path,
    ) -> AttemptResult:
        workspace_id = _workspace_id(model, concurrency, worker_index, task)
        workspace = config.workspace_root / workspace_id
        fixture = config.fixtures_root / task.fixture
        copy_fixture(fixture, workspace)
        seed_openclaw_workspace_files(
            workspace,
            agent_id=f"{config.openclaw_agent}-{worker_index:03d}",
            task_id=task.task_id,
            model_id=model.served_model_name,
        )
        before_hashes = snapshot_files(workspace)
        before_text = read_text_files(workspace)
        session_id = _session_id(config.run_id, workspace_id, worker_index, task)

        start = time.monotonic()
        response = self.backend.run(model, task, workspace, session_id, config.timeout_s)
        wall_time_s = time.monotonic() - start

        after_hashes = snapshot_files(workspace)
        changed = changed_files(before_hashes, after_hashes)
        tests_passed, verify_output = run_verify_command(workspace, task.verify_command)
        score, failure_type, hallucinated, notes = score_task(task, workspace, response, changed, tests_passed)
        if response.error:
            failure_type = response.error
            notes = _join_notes(notes, f"OpenClaw error: {response.error}")
        status = "pass" if score == 1.0 and tests_passed and not response.error and hallucinated == 0 else "fail"
        raw_payload = {
            "run_id": config.run_id,
            "task": task.task_id,
            "task_type": task.task_type,
            "workspace_id": workspace_id,
            "workspace": str(workspace),
            "session_id": session_id,
            "model": {
                "model": model.model_id,
                "served_model_name": model.served_model_name,
                "comparison_id": model.comparison_key,
                "backend": config.backend_name,
                "provider_type": model.provider_type,
                "hardware_profile": model.hardware_profile,
                "weight_quant": model.weight_quant,
                "kv_cache_dtype": model.kv_cache_dtype,
                "context_limit": model.context_limit,
                "concurrency": concurrency,
            },
            "response": response.raw,
            "text": response.text,
            "verify_output": verify_output,
            "notes": notes,
        }
        raw_path = raw_dir / f"{workspace_id}.json"
        raw_path.write_text(json.dumps(raw_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._write_patch(patch_dir / f"{workspace_id}.diff", before_text, read_text_files(workspace))

        return AttemptResult(
            run_id=config.run_id,
            model=model.model_id,
            served_model_name=model.served_model_name,
            comparison_id=model.comparison_key,
            backend=config.backend_name,
            provider_type=model.provider_type,
            hardware_profile=model.hardware_profile,
            weight_quant=model.weight_quant,
            kv_cache_dtype=model.kv_cache_dtype,
            context_limit=model.context_limit,
            concurrency=concurrency,
            task_id=task.task_id,
            task_type=task.task_type,
            task_tags=task.tags,
            workspace_id=workspace_id,
            status=status,
            score=score,
            wall_time_s=wall_time_s,
            ttft_s=response.ttft_s,
            tool_calls=response.tool_calls,
            files_read=response.files_read,
            duplicate_file_reads=response.duplicate_file_reads,
            time_to_first_relevant_file_s=response.time_to_first_relevant_file_s,
            files_changed=len(changed),
            changed_files=changed,
            tests_passed=tests_passed,
            json_valid=json_valid(response),
            hallucinated_paths=hallucinated,
            oom=False,
            timeout=response.timed_out,
            request_errors=response.request_errors,
            failure_type=failure_type,
            notes=notes or verify_output[-500:],
        )

    def _write_patch(self, path: Path, before: dict[str, str], after: dict[str, str]) -> None:
        lines: list[str] = []
        for rel_path in sorted(set(before) | set(after)):
            if before.get(rel_path) == after.get(rel_path):
                continue
            before_lines = before.get(rel_path, "").splitlines(keepends=True)
            after_lines = after.get(rel_path, "").splitlines(keepends=True)
            lines.extend(difflib.unified_diff(before_lines, after_lines, fromfile=f"a/{rel_path}", tofile=f"b/{rel_path}"))
        path.write_text("".join(lines), encoding="utf-8")


def reset_output_dir(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)


def _workspace_id(model: ModelSpec, concurrency: int, worker_index: int, task: TaskSpec) -> str:
    return (
        f"{_safe_id(model.served_model_name)}-{_safe_id(model.hardware_profile)}-{_safe_id(model.kv_cache_dtype)}-"
        f"ctx{model.context_limit}-conc{concurrency}-worker-{worker_index:03d}-{task.task_id}"
    )


def _session_id(run_id: str, workspace_id: str, worker_index: int, task: TaskSpec) -> str:
    digest = hashlib.sha1(workspace_id.encode("utf-8")).hexdigest()[:12]
    run_part = _safe_id(run_id)[:32].strip("-") or "run"
    task_part = _safe_id(task.task_id)[:24].strip("-") or "task"
    return f"{run_part}-w{worker_index:03d}-{task_part}-{digest}"


def _run_provenance(config: RunConfig) -> dict:
    task_rows = [
        {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "fixture": task.fixture,
            "prompt": task.prompt,
            "expected": task.expected,
            "verify_command": task.verify_command,
            "context_sizes": task.context_sizes,
            "max_changed_files": task.max_changed_files,
            "tags": task.tags,
        }
        for task in config.suite.tasks
    ]
    model_rows = [model.__dict__ for model in config.models]
    return {
        "schema_version": 1,
        "suite_id": config.suite.suite_id,
        "suite_digest": _json_digest(task_rows),
        "task_count": len(task_rows),
        "model_source": "model_config" if config.model_config_path is not None else "aliases",
        "model_matrix_digest": _json_digest(model_rows),
        "input_files": _input_file_provenance(config),
        "fixture_digests": {
            task.fixture: _fixture_digest(config.fixtures_root / task.fixture)
            for task in config.suite.tasks
        },
    }


def _json_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fixture_digest(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(str(path.name).encode("utf-8"))
    if not path.exists():
        digest.update(b"\0missing")
        return digest.hexdigest()
    for item in sorted(path.rglob("*")):
        if not item.is_file() or "__pycache__" in item.parts:
            continue
        rel = item.relative_to(path).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _input_file_provenance(config: RunConfig) -> list[dict[str, str]]:
    rows = []
    source_paths = config.suite.source_paths or ([config.suite_path] if config.suite_path is not None else [])
    for index, source_path in enumerate(source_paths):
        rows.append(_file_provenance("suite" if index == 0 else "suite_include", source_path))
    if config.model_config_path is not None:
        rows.append(_file_provenance("model_config", config.model_config_path))
    return rows


def _file_provenance(role: str, path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "role": role,
        "path": str(path),
        "digest": _file_digest(resolved),
    }


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        digest.update(path.read_bytes())
    except OSError:
        digest.update(b"\0missing")
    return digest.hexdigest()


def _runtime_provenance(config: RunConfig) -> dict:
    payload = {
        "schema_version": 1,
        "python_version": sys.version.split()[0],
        "harness_version": _harness_version(),
    }
    if config.backend_name == "openclaw":
        payload["openclaw"] = _openclaw_runtime(config)
    return payload


def _harness_version() -> str:
    try:
        return metadata.version("openclaw-local-model-bench")
    except metadata.PackageNotFoundError:
        return "editable-or-uninstalled"


def _openclaw_runtime(config: RunConfig) -> dict:
    cmd = _openclaw_version_cmd(config.openclaw_container)
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "cmd": cmd,
            "status": "fail",
            "version": "",
            "error": str(exc),
        }
    output = (proc.stdout or proc.stderr).strip()
    return {
        "cmd": cmd,
        "status": "pass" if proc.returncode == 0 and bool(output) else "fail",
        "version": output.splitlines()[0] if output else "",
        "returncode": proc.returncode,
    }


def _openclaw_version_cmd(container: str | None) -> list[str]:
    if container:
        return ["docker", "exec", container, "openclaw", "--version"]
    return ["openclaw", "--version"]


def _join_notes(*parts: str) -> str:
    return " ".join(part for part in parts if part)


def _load_failure_note(failure_type: str) -> str:
    if failure_type in {"model_route_failed", "openclaw_timeout"}:
        return "Model server loaded, but OpenClaw route smoke failed; task was not attempted."
    if failure_type == "serve_probe_failed":
        return "Model server health passed, but serve probe failed; task was not attempted."
    return "Model server did not load; task was not attempted."


def _throughput_probe_row(model: ModelSpec, route_probe: dict) -> dict | None:
    if not route_probe.get("success"):
        return None
    return {
        "model": model.served_model_name,
        "comparison_id": model.comparison_key,
        "provider_type": model.provider_type,
        "hardware_profile": model.hardware_profile,
        "weight_quant": model.weight_quant,
        "kv_cache_dtype": model.kv_cache_dtype,
        "context_limit": model.context_limit,
        "endpoint": route_probe.get("endpoint"),
        "prompt_chars": route_probe.get("prompt_chars"),
        "wall_time_s": route_probe.get("wall_time_s"),
        "completion_tokens": route_probe.get("completion_tokens"),
        "total_tokens": route_probe.get("total_tokens"),
        "tokens_per_s": route_probe.get("tokens_per_s"),
        "sample_count": route_probe.get("sample_count"),
        "tokens_per_s_p50": route_probe.get("tokens_per_s_p50"),
        "tokens_per_s_p95": route_probe.get("tokens_per_s_p95"),
        "wall_time_p50_s": route_probe.get("wall_time_p50_s"),
        "wall_time_p95_s": route_probe.get("wall_time_p95_s"),
    }
