from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_TASK_TYPES = {
    "workspace_discovery",
    "multi_file_bug_trace",
    "patch_execution",
    "workspace_needle",
    "instruction_retention",
    "repo_read_only",
    "repo_code_edit",
}


EXTERNAL_PROVIDER_TYPES = {"api", "subscription"}
REQUIRED_LOCAL_KV_MODES = {"fp8", "turboquant_k8v4", "turboquant_k3v4_nc"}
REQUIRED_CONTEXTS = {4096, 8192, 16384, 32768, 65536}
REQUIRED_CONCURRENCIES = {1, 2, 4, 8, 16, 32, 64}
REQUIRED_EXTERNAL_CONTEXTS = {4096, 8192, 32768}
REQUIRED_EXTERNAL_CONCURRENCIES = {1, 4, 16}
REQUIRED_LOCAL_SETUP_TASK_TYPES = {"patch_execution", "instruction_retention"}
REPRESENTATIVE_CONCURRENCY_TASK_TYPES = {"patch_execution", "instruction_retention", "repo_code_edit", "multi_file_bug_trace"}
MIN_LOCAL_HARDWARE_PROFILES = 2
MAX_P95_TOOL_CALLS = 80
MAX_P95_FILES_READ = 80
MAX_P95_DUPLICATE_FILE_READS = 20
MAX_P95_TIME_TO_FIRST_RELEVANT_FILE_S = 120.0


NOT_ATTEMPTED_FAILURE_TYPES = {
    "model_load_failed",
    "unsupported_kv_dtype",
    "oom_on_load",
    "server_timeout",
    "model_route_failed",
    "openclaw_timeout",
    "serve_probe_failed",
    "context_window_exceeded",
    "incomplete_result",
    "tool_parser_missing",
    "model_override_unauthorized",
    "openclaw_embedded_fallback",
    "openclaw_agent_setup_failed",
}


@dataclass(frozen=True)
class CertificationCheck:
    name: str
    status: str
    notes: str = ""

    def to_row(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "notes": self.notes}


@dataclass(frozen=True)
class CertificationResult:
    checks: list[CertificationCheck]

    @property
    def ok(self) -> bool:
        return all(check.status != "fail" for check in self.checks)

    def to_json(self) -> str:
        return json.dumps(
            {
                "ok": self.ok,
                "summary": self.summary(),
                "checks": [check.to_row() for check in self.checks],
            },
            indent=2,
            sort_keys=True,
        )

    def summary(self) -> dict[str, Any]:
        counts = _status_counts(self.checks)
        failed = [check.name for check in self.checks if check.status == "fail"]
        warnings = [check.name for check in self.checks if check.status == "warn"]
        return {
            "pass": counts.get("pass", 0),
            "warn": counts.get("warn", 0),
            "fail": counts.get("fail", 0),
            "failed_checks": failed,
            "warning_checks": warnings,
        }

    def nonpassing(self) -> "CertificationResult":
        return CertificationResult([check for check in self.checks if check.status != "pass"])


def certify_run_dirs(run_dirs: list[Path]) -> CertificationResult:
    checks: list[CertificationCheck] = []
    attempts: list[dict[str, Any]] = []
    servers: list[dict[str, Any]] = []

    if not run_dirs:
        return CertificationResult([CertificationCheck("run_dirs", "fail", "no run directories supplied")])

    for run_dir in run_dirs:
        run_checks, run_attempts, server = _load_run_dir(run_dir)
        checks.extend(run_checks)
        attempts.extend(run_attempts)
        if server is not None:
            servers.append(server)

    if attempts:
        checks.extend(_coverage_checks(attempts, servers))
    else:
        checks.append(CertificationCheck("attempts", "fail", "no benchmark attempts loaded"))
    return CertificationResult(checks)


def render_certification_text(result: CertificationResult, failures_only: bool = False) -> str:
    lines = [f"certification={'ok' if result.ok else 'failed'}"]
    summary = result.summary()
    lines.append(f"summary=pass:{summary['pass']} warn:{summary['warn']} fail:{summary['fail']}")
    checks = result.nonpassing().checks if failures_only else result.checks
    for check in checks:
        suffix = f" - {check.notes}" if check.notes else ""
        lines.append(f"{check.status.upper()} {check.name}{suffix}")
    return "\n".join(lines)


def _status_counts(checks: list[CertificationCheck]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
    return counts


def _load_run_dir(run_dir: Path) -> tuple[list[CertificationCheck], list[dict[str, Any]], dict[str, Any] | None]:
    checks: list[CertificationCheck] = []
    attempts: list[dict[str, Any]] = []
    server: dict[str, Any] | None = None
    required_files = ["config.json", "attempts.jsonl", "failures.jsonl", "summary.json", "server.json"]
    for filename in required_files:
        path = run_dir / filename
        checks.append(
            CertificationCheck(
                f"artifact:{run_dir.name}:{filename}",
                "pass" if path.exists() else "fail",
                str(path),
            )
        )
    if not (run_dir / "attempts.jsonl").exists():
        return checks, attempts, server

    try:
        attempts = _read_jsonl(run_dir / "attempts.jsonl")
    except (OSError, json.JSONDecodeError) as exc:
        checks.append(CertificationCheck(f"parse:{run_dir.name}:attempts", "fail", str(exc)))
        attempts = []
    else:
        status = "pass" if attempts else "fail"
        checks.append(CertificationCheck(f"parse:{run_dir.name}:attempts", status, f"{len(attempts)} rows"))
        if attempts:
            attempted_rows = [row for row in attempts if _requires_task_artifacts(row)]
            checks.extend(_artifact_dir_checks(run_dir, "raw", "*.json", len(attempted_rows), parse_json=True))
            checks.extend(_artifact_dir_checks(run_dir, "patches", "*.diff", len(attempted_rows), parse_json=False))
            checks.extend(_attempt_artifact_binding_checks(run_dir, attempted_rows))

    for filename in ("summary.json", "server.json", "config.json"):
        path = run_dir / filename
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            checks.append(CertificationCheck(f"parse:{run_dir.name}:{filename}", "fail", str(exc)))
            continue
        checks.append(CertificationCheck(f"parse:{run_dir.name}:{filename}", "pass", "valid JSON"))
        if filename == "server.json" and isinstance(payload, dict):
            server = payload
        if filename == "config.json" and isinstance(payload, dict):
            checks.extend(_config_provenance_checks(run_dir, payload))
            checks.extend(_config_runtime_checks(run_dir, payload))
            checks.extend(_config_gateway_ensure_checks(run_dir, payload))
            checks.extend(_config_context_checks(run_dir, payload))
    return checks, attempts, server


def _config_provenance_checks(run_dir: Path, config: dict[str, Any]) -> list[CertificationCheck]:
    provenance = config.get("provenance")
    label = f"config:{run_dir.name}:provenance"
    if not isinstance(provenance, dict):
        return [CertificationCheck(label, "fail", "missing provenance block")]
    fixture_digests = provenance.get("fixture_digests")
    errors = []
    if provenance.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not isinstance(provenance.get("suite_id"), str) or not provenance.get("suite_id"):
        errors.append("suite_id missing")
    if not _looks_sha256(provenance.get("suite_digest")):
        errors.append("suite_digest missing/invalid")
    if not _looks_sha256(provenance.get("model_matrix_digest")):
        errors.append("model_matrix_digest missing/invalid")
    if provenance.get("model_source") not in {"aliases", "model_config"}:
        errors.append("model_source missing/invalid")
    if not isinstance(provenance.get("task_count"), int) or provenance.get("task_count") <= 0:
        errors.append("task_count missing/invalid")
    input_files = provenance.get("input_files")
    if not isinstance(input_files, list) or not input_files:
        errors.append("input_files missing/invalid")
    elif any(not _valid_input_file_provenance(row) for row in input_files):
        errors.append("input_files must include role, path, and sha256 digest")
    elif not _input_file_roles_valid(input_files, str(provenance.get("model_source"))):
        errors.append("input_files missing required suite/model_config role")
    if not isinstance(fixture_digests, dict) or not fixture_digests:
        errors.append("fixture_digests missing/invalid")
    elif any(not isinstance(name, str) or not _looks_sha256(value) for name, value in fixture_digests.items()):
        errors.append("fixture_digests must map fixture names to sha256 digests")
    return [
        CertificationCheck(
            label,
            "pass" if not errors else "fail",
            f"{provenance.get('suite_id')} task_count={provenance.get('task_count')}"
            if not errors
            else "; ".join(errors),
        )
    ]


def _valid_input_file_provenance(row: Any) -> bool:
    return (
        isinstance(row, dict)
        and isinstance(row.get("role"), str)
        and bool(row.get("role"))
        and isinstance(row.get("path"), str)
        and bool(row.get("path"))
        and _looks_sha256(row.get("digest"))
    )


def _input_file_roles_valid(input_files: list[Any], model_source: str) -> bool:
    roles = {row.get("role") for row in input_files if isinstance(row, dict)}
    if "suite" not in roles:
        return False
    if model_source == "model_config" and "model_config" not in roles:
        return False
    return True


def _config_gateway_ensure_checks(run_dir: Path, config: dict[str, Any]) -> list[CertificationCheck]:
    label = f"config:{run_dir.name}:openclaw_gateway_ensure"
    if config.get("backend") != "openclaw" or config.get("openclaw_local") is True:
        return []
    if config.get("ensure_openclaw_gateway") is False:
        return [CertificationCheck(label, "warn", "gateway auto-ensure was disabled for this run")]
    ensure = config.get("openclaw_gateway_ensure")
    if not isinstance(ensure, dict):
        return [CertificationCheck(label, "fail", "missing gateway ensure result for non-local OpenClaw run")]
    errors = []
    if ensure.get("name") != "openclaw_gateway":
        errors.append("name must be openclaw_gateway")
    if ensure.get("status") != "pass":
        errors.append(f"status must be pass, got {ensure.get('status')!r}")
    if not isinstance(ensure.get("notes"), str) or not ensure.get("notes"):
        errors.append("notes missing")
    return [
        CertificationCheck(
            label,
            "pass" if not errors else "fail",
            ensure.get("notes", "") if not errors else "; ".join(errors),
        )
    ]


def _config_runtime_checks(run_dir: Path, config: dict[str, Any]) -> list[CertificationCheck]:
    runtime = config.get("runtime")
    label = f"config:{run_dir.name}:runtime"
    if not isinstance(runtime, dict):
        return [CertificationCheck(label, "fail", "missing runtime block")]
    errors = []
    if runtime.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not isinstance(runtime.get("python_version"), str) or not runtime.get("python_version"):
        errors.append("python_version missing")
    if not isinstance(runtime.get("harness_version"), str) or not runtime.get("harness_version"):
        errors.append("harness_version missing")
    if config.get("backend") == "openclaw":
        openclaw = runtime.get("openclaw")
        if not isinstance(openclaw, dict):
            errors.append("openclaw runtime missing")
        else:
            if openclaw.get("status") != "pass":
                errors.append(f"openclaw status must be pass, got {openclaw.get('status')!r}")
            if not isinstance(openclaw.get("version"), str) or not openclaw.get("version"):
                errors.append("openclaw version missing")
            if not isinstance(openclaw.get("cmd"), list) or "openclaw" not in [str(part) for part in openclaw.get("cmd", [])]:
                errors.append("openclaw cmd missing/invalid")
            if openclaw.get("returncode") != 0:
                errors.append(f"openclaw returncode must be 0, got {openclaw.get('returncode')!r}")
    return [
        CertificationCheck(
            label,
            "pass" if not errors else "fail",
            f"python={runtime.get('python_version')} harness={runtime.get('harness_version')}"
            if not errors
            else "; ".join(errors),
        )
    ]


def _config_context_checks(run_dir: Path, config: dict[str, Any]) -> list[CertificationCheck]:
    checks: list[CertificationCheck] = []
    models = config.get("models")
    if not isinstance(models, list):
        return checks
    for index, model in enumerate(models):
        if not isinstance(model, dict) or model.get("provider_type") != "local":
            continue
        serve_command = model.get("serve_command")
        context_limit = _as_int(model.get("context_limit"))
        if not isinstance(serve_command, list) or context_limit is None:
            continue
        max_model_len = _serve_arg_int(serve_command, "--max-model-len")
        label = f"config:{run_dir.name}:model-{index}:max_model_len"
        if max_model_len is None:
            checks.append(CertificationCheck(label, "warn", "serve_command has no --max-model-len"))
            continue
        checks.append(
            CertificationCheck(
                label,
                "pass" if max_model_len >= context_limit else "fail",
                f"--max-model-len={max_model_len}, context_limit={context_limit}",
            )
        )
    return checks


def _artifact_dir_checks(run_dir: Path, dirname: str, pattern: str, expected_count: int, parse_json: bool) -> list[CertificationCheck]:
    artifact_dir = run_dir / dirname
    if not artifact_dir.is_dir():
        return [CertificationCheck(f"artifact_dir:{run_dir.name}:{dirname}", "fail", f"missing {artifact_dir}")]
    files = sorted(artifact_dir.glob(pattern))
    checks = [
        CertificationCheck(
            f"artifact_count:{run_dir.name}:{dirname}",
            "pass" if len(files) == expected_count else "fail",
            f"{len(files)} artifact(s) for {expected_count} attempt(s)",
        )
    ]
    if parse_json:
        errors = []
        for path in files:
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"{path.name}: {exc}")
        checks.append(
            CertificationCheck(
                f"artifact_parse:{run_dir.name}:{dirname}",
                "pass" if not errors else "fail",
                "; ".join(errors[:3]) if errors else f"{len(files)} valid JSON artifact(s)",
            )
        )
    return checks


def _attempt_artifact_binding_checks(run_dir: Path, attempted_rows: list[dict[str, Any]]) -> list[CertificationCheck]:
    if not attempted_rows:
        return []
    checks: list[CertificationCheck] = []
    raw_dir = run_dir / "raw"
    patch_dir = run_dir / "patches"
    if not raw_dir.is_dir() or not patch_dir.is_dir():
        return []

    workspace_ids = [str(row.get("workspace_id") or "") for row in attempted_rows]
    missing_workspace_ids = [str(row.get("task_id") or index) for index, row in enumerate(attempted_rows) if not row.get("workspace_id")]
    duplicate_workspace_ids = sorted(_duplicates(workspace_id for workspace_id in workspace_ids if workspace_id))
    expected_raw = {f"{workspace_id}.json" for workspace_id in workspace_ids if workspace_id}
    expected_patches = {f"{workspace_id}.diff" for workspace_id in workspace_ids if workspace_id}
    actual_raw = {path.name for path in raw_dir.glob("*.json")}
    actual_patches = {path.name for path in patch_dir.glob("*.diff")}

    identity_errors = []
    if missing_workspace_ids:
        identity_errors.append("missing workspace_id for " + ", ".join(missing_workspace_ids[:8]))
    if duplicate_workspace_ids:
        identity_errors.append("duplicate workspace_id " + ", ".join(duplicate_workspace_ids[:8]))
    checks.append(
        CertificationCheck(
            f"artifact_binding:{run_dir.name}:attempt_identity",
            "pass" if not identity_errors else "fail",
            "; ".join(identity_errors) if identity_errors else f"{len(workspace_ids)} attempt workspace id(s)",
        )
    )

    raw_mismatch = _artifact_set_mismatch(expected_raw, actual_raw)
    patch_mismatch = _artifact_set_mismatch(expected_patches, actual_patches)
    checks.append(
        CertificationCheck(
            f"artifact_binding:{run_dir.name}:raw_names",
            "pass" if not raw_mismatch else "fail",
            raw_mismatch or f"{len(expected_raw)} raw artifact(s) match attempts",
        )
    )
    checks.append(
        CertificationCheck(
            f"artifact_binding:{run_dir.name}:patch_names",
            "pass" if not patch_mismatch else "fail",
            patch_mismatch or f"{len(expected_patches)} patch artifact(s) match attempts",
        )
    )

    metadata_errors = []
    for row in attempted_rows:
        workspace_id = str(row.get("workspace_id") or "")
        if not workspace_id:
            continue
        raw_path = raw_dir / f"{workspace_id}.json"
        if not raw_path.exists():
            continue
        try:
            raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            metadata_errors.append(f"{raw_path.name}: {exc}")
            continue
        metadata_errors.extend(_raw_artifact_metadata_errors(raw_path.name, row, raw_payload))
    checks.append(
        CertificationCheck(
            f"artifact_binding:{run_dir.name}:raw_metadata",
            "pass" if not metadata_errors else "fail",
            "; ".join(metadata_errors[:8]) if metadata_errors else f"{len(attempted_rows)} raw artifact(s) match row metadata",
        )
    )
    return checks


def _duplicates(values: Any) -> set[str]:
    seen: set[str] = set()
    duplicated: set[str] = set()
    for value in values:
        text = str(value)
        if text in seen:
            duplicated.add(text)
        seen.add(text)
    return duplicated


def _artifact_set_mismatch(expected: set[str], actual: set[str]) -> str:
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    notes = []
    if missing:
        notes.append("missing " + ", ".join(missing[:8]))
    if unexpected:
        notes.append("unexpected " + ", ".join(unexpected[:8]))
    return "; ".join(notes)


def _raw_artifact_metadata_errors(filename: str, row: dict[str, Any], raw_payload: dict[str, Any]) -> list[str]:
    errors = []
    expected_scalars = {
        "workspace_id": row.get("workspace_id"),
        "task": row.get("task_id"),
        "task_type": row.get("task_type"),
    }
    for key, expected in expected_scalars.items():
        if raw_payload.get(key) != expected:
            errors.append(f"{filename}: {key}={raw_payload.get(key)!r} expected {expected!r}")
    model_payload = raw_payload.get("model")
    if not isinstance(model_payload, dict):
        errors.append(f"{filename}: missing model metadata")
        return errors
    expected_model = {
        "served_model_name": row.get("served_model_name"),
        "backend": row.get("backend"),
        "provider_type": row.get("provider_type"),
        "hardware_profile": row.get("hardware_profile"),
        "weight_quant": row.get("weight_quant"),
        "kv_cache_dtype": row.get("kv_cache_dtype"),
        "context_limit": row.get("context_limit"),
        "concurrency": row.get("concurrency"),
    }
    for key, expected in expected_model.items():
        if model_payload.get(key) != expected:
            errors.append(f"{filename}: model.{key}={model_payload.get(key)!r} expected {expected!r}")
    errors.extend(_raw_response_provenance_errors(filename, row, raw_payload.get("response")))
    return errors


def _raw_response_provenance_errors(filename: str, row: dict[str, Any], response: Any) -> list[str]:
    if not isinstance(response, dict):
        return [f"{filename}: response provenance missing"]
    backend = str(row.get("backend") or "")
    if backend == "simulator":
        if response.get("simulated") is not True:
            return [f"{filename}: simulator response missing simulated=true"]
        return []
    cmd = response.get("cmd")
    if not isinstance(cmd, list) or not any(str(part) == "openclaw" for part in cmd):
        return [f"{filename}: live response missing OpenClaw command provenance"]
    if response.get("simulated") is True:
        return [f"{filename}: live response was marked simulated"]
    if row.get("status") == "pass" and response.get("returncode") not in {0, None}:
        return [f"{filename}: passing live response had nonzero returncode {response.get('returncode')!r}"]
    return []


def _requires_task_artifacts(row: dict[str, Any]) -> bool:
    wall_time_s = _as_float(row.get("wall_time_s")) or 0.0
    if (
        row.get("failure_type") in NOT_ATTEMPTED_FAILURE_TYPES
        and _as_int(row.get("files_read")) == 0
        and _as_int(row.get("tool_calls")) == 0
        and wall_time_s == 0.0
    ):
        return False
    return True


def _coverage_checks(attempts: list[dict[str, Any]], servers: list[dict[str, Any]]) -> list[CertificationCheck]:
    checks: list[CertificationCheck] = []
    live_attempts = [row for row in attempts if row.get("backend") != "simulator"]
    passed = [row for row in live_attempts if row.get("status") == "pass"]
    providers = {str(row.get("provider_type", "")) for row in live_attempts}
    backends = {str(row.get("backend", "")) for row in attempts}
    task_types = {str(row.get("task_type", "")) for row in live_attempts}
    passed_task_types = {str(row.get("task_type", "")) for row in passed}
    local_passed_task_types = {str(row.get("task_type", "")) for row in passed if row.get("provider_type") == "local"}
    local_setups = {
        (str(row.get("weight_quant", "")), str(row.get("kv_cache_dtype", "")))
        for row in live_attempts
        if row.get("provider_type") == "local"
    }
    local_setups_by_kv = {
        str(row.get("kv_cache_dtype", "")): [
            candidate
            for candidate in live_attempts
            if candidate.get("provider_type") == "local" and str(candidate.get("kv_cache_dtype", "")) == str(row.get("kv_cache_dtype", ""))
        ]
        for row in live_attempts
        if row.get("provider_type") == "local"
    }
    local_passes = [row for row in passed if row.get("provider_type") == "local"]
    local_passing_rows_by_kv = {
        kv_mode: [
            row
            for row in local_passes
            if str(row.get("kv_cache_dtype", "")) == kv_mode
        ]
        for kv_mode in {str(row.get("kv_cache_dtype", "")) for row in local_passes}
    }
    local_passing_needle_rows = [
        row
        for row in passed
        if row.get("provider_type") == "local" and row.get("task_type") == "workspace_needle"
    ]
    local_passing_needle_rows_by_kv = {
        kv_mode: [
            row
            for row in local_passing_needle_rows
            if str(row.get("kv_cache_dtype", "")) == kv_mode
        ]
        for kv_mode in {str(row.get("kv_cache_dtype", "")) for row in local_passing_needle_rows}
    }
    contexts = {_as_int(row.get("context_limit")) for row in live_attempts}
    contexts.discard(None)
    concurrencies = {_as_int(row.get("concurrency")) for row in live_attempts}
    concurrencies.discard(None)
    local_contexts = {_as_int(row.get("context_limit")) for row in live_attempts if row.get("provider_type") == "local"}
    local_contexts.discard(None)
    local_passing_needle_contexts = {_as_int(row.get("context_limit")) for row in local_passing_needle_rows}
    local_passing_needle_contexts.discard(None)
    local_concurrencies = {_as_int(row.get("concurrency")) for row in local_passes}
    local_concurrencies.discard(None)
    local_kv_modes = {str(row.get("kv_cache_dtype", "")) for row in live_attempts if row.get("provider_type") == "local"}
    local_hardware_profiles = {
        str(row.get("hardware_profile") or "default")
        for row in live_attempts
        if row.get("provider_type") == "local"
    }
    external_passes = [row for row in passed if row.get("provider_type") in EXTERNAL_PROVIDER_TYPES]
    passed_external_providers = {str(row.get("provider_type", "")) for row in external_passes}
    external_passed_task_types = {
        provider: {str(row.get("task_type", "")) for row in external_passes if row.get("provider_type") == provider}
        for provider in EXTERNAL_PROVIDER_TYPES
    }
    external_contexts = {
        provider: {
            _as_int(row.get("context_limit"))
            for row in live_attempts
            if row.get("provider_type") == provider
        }
        for provider in EXTERNAL_PROVIDER_TYPES
    }
    external_concurrencies = {
        provider: {
            _as_int(row.get("concurrency"))
            for row in live_attempts
            if row.get("provider_type") == provider
        }
        for provider in EXTERNAL_PROVIDER_TYPES
    }
    for values in external_contexts.values():
        values.discard(None)
    for values in external_concurrencies.values():
        values.discard(None)

    checks.append(
        CertificationCheck(
            "live_backend",
            "pass" if any(backend != "simulator" for backend in backends) else "fail",
            f"backends={', '.join(sorted(backends))}",
        )
    )
    checks.append(
        CertificationCheck(
            "local_provider_rows",
            "pass" if "local" in providers else "fail",
            f"providers={', '.join(sorted(providers))}",
        )
    )
    checks.append(
        CertificationCheck(
            "api_or_subscription_rows",
            "pass" if EXTERNAL_PROVIDER_TYPES <= providers else "fail",
            _missing_note(EXTERNAL_PROVIDER_TYPES - providers) if EXTERNAL_PROVIDER_TYPES - providers else f"providers={', '.join(sorted(providers))}",
        )
    )
    checks.append(
        CertificationCheck(
            "local_setup_exploration",
            "pass" if REQUIRED_LOCAL_KV_MODES <= local_kv_modes else "fail",
            _missing_note(REQUIRED_LOCAL_KV_MODES - local_kv_modes)
            if REQUIRED_LOCAL_KV_MODES - local_kv_modes
            else f"{len(local_setups)} local weight/KV setup(s)",
        )
    )
    checks.append(_local_hardware_setup_check(local_hardware_profiles, servers))
    checks.append(
        CertificationCheck(
            "required_task_types",
            "pass" if REQUIRED_TASK_TYPES <= task_types else "fail",
            _missing_note(REQUIRED_TASK_TYPES - task_types),
        )
    )
    checks.append(
        CertificationCheck(
            "required_task_types_passed",
            "pass" if REQUIRED_TASK_TYPES <= passed_task_types else "fail",
            _missing_note(REQUIRED_TASK_TYPES - passed_task_types),
        )
    )
    checks.append(
        CertificationCheck(
            "local_required_task_types_passed",
            "pass" if REQUIRED_TASK_TYPES <= local_passed_task_types else "fail",
            _missing_note(REQUIRED_TASK_TYPES - local_passed_task_types),
        )
    )
    checks.append(
        CertificationCheck(
            "local_task_success",
            "pass" if local_passes else "fail",
            f"{len(local_passes)} passing local attempt(s)",
        )
    )
    checks.append(_local_setup_representative_task_check(local_passes))
    checks.append(
        CertificationCheck(
            "api_or_subscription_task_success",
            "pass" if EXTERNAL_PROVIDER_TYPES <= passed_external_providers else "fail",
            _missing_note(EXTERNAL_PROVIDER_TYPES - passed_external_providers)
            if EXTERNAL_PROVIDER_TYPES - passed_external_providers
            else f"{len(external_passes)} passing API/subscription attempt(s)",
        )
    )
    checks.append(_external_required_task_types_check(external_passed_task_types))
    checks.append(_external_context_coverage_check(external_contexts))
    checks.append(_external_concurrency_coverage_check(external_concurrencies))
    checks.append(
        CertificationCheck(
            "baseline_context",
            "pass" if 4096 in local_passing_needle_contexts else "fail",
            f"local_passing_needle_contexts={_format_ints(local_passing_needle_contexts)}",
        )
    )
    checks.append(
        CertificationCheck(
            "long_context",
            "pass" if any(context >= 32768 for context in local_passing_needle_contexts) else "fail",
            f"local_passing_needle_contexts={_format_ints(local_passing_needle_contexts)}",
        )
    )
    checks.append(
        CertificationCheck(
            "context_sweep",
            "pass" if REQUIRED_CONTEXTS <= local_passing_needle_contexts else "fail",
            _missing_int_note(REQUIRED_CONTEXTS - local_passing_needle_contexts)
            if REQUIRED_CONTEXTS - local_passing_needle_contexts
            else f"local_passing_needle_contexts={_format_ints(local_passing_needle_contexts)}",
        )
    )
    checks.append(_local_context_ceiling_check(local_contexts))
    checks.append(_local_setup_context_sweep_check(local_passing_needle_rows_by_kv))
    checks.append(_local_fp8_pairing_check(live_attempts))
    checks.append(
        CertificationCheck(
            "single_and_pool_concurrency",
            "pass" if {1, 4} <= local_concurrencies else "fail",
            f"local_concurrency={_format_ints(local_concurrencies)}",
        )
    )
    checks.append(_local_setup_concurrency_sweep_check(local_passing_rows_by_kv))
    checks.append(_local_concurrency_representative_task_check(local_passes))
    checks.append(
        CertificationCheck(
            "concurrency_sweep",
            "pass" if REQUIRED_CONCURRENCIES <= local_concurrencies else "fail",
            _missing_int_note(REQUIRED_CONCURRENCIES - local_concurrencies)
            if REQUIRED_CONCURRENCIES - local_concurrencies
            else f"local_concurrency={_format_ints(local_concurrencies)}",
        )
    )
    checks.append(
        CertificationCheck(
            "stress_concurrency",
            "pass" if any(level >= 64 for level in local_concurrencies) else "fail",
            f"local_concurrency={_format_ints(local_concurrencies)}",
        )
    )
    checks.append(_local_hardware_pairing_check(local_passes))
    checks.append(_route_probe_check(servers))
    checks.extend(_tool_file_efficiency_checks(passed))
    checks.extend(_server_evidence_checks(live_attempts, servers))
    if providers & EXTERNAL_PROVIDER_TYPES:
        checks.append(_external_route_probe_check(live_attempts, servers))
    return checks


def _server_evidence_checks(attempts: list[dict[str, Any]], servers: list[dict[str, Any]]) -> list[CertificationCheck]:
    checks: list[CertificationCheck] = []
    missing_hardware = []
    for index, server in enumerate(servers):
        hardware = server.get("hardware")
        if not isinstance(hardware, dict) or not isinstance(hardware.get("available"), bool) or not isinstance(hardware.get("devices"), list):
            missing_hardware.append(str(index))
    checks.append(
        CertificationCheck(
            "hardware_inventory",
            "pass" if servers and not missing_hardware else "fail",
            "server.json includes hardware inventory"
            if servers and not missing_hardware
            else f"missing/invalid hardware in server index(es): {', '.join(missing_hardware) or 'all'}",
        )
    )

    live_cells = {_model_cell_key(row) for row in attempts}
    live_cells.discard(None)
    server_model_cells = {
        _model_cell_key(model)
        for server in servers
        for model in server.get("models", [])
        if isinstance(model, dict)
    }
    server_model_cells.discard(None)
    missing_server_model_cells = live_cells - server_model_cells
    checks.append(
        CertificationCheck(
            "server_model_cell_evidence",
            "pass" if live_cells and not missing_server_model_cells else "fail",
            f"{len(server_model_cells & live_cells)}/{len(live_cells)} live model cell(s) represented in server.json"
            if live_cells and not missing_server_model_cells
            else "missing " + "; ".join(_format_model_cell(cell) for cell in sorted(missing_server_model_cells, key=_format_model_cell)[:8]),
        )
    )

    successful_direct_routes: set[tuple[str, str, str, str, str, int | None]] = set()
    throughput_routes: set[tuple[str, str, str, str, str, int | None]] = set()
    invalid_throughput = []
    for server in servers:
        for result in server.get("serve_results", []):
            if not isinstance(result, dict):
                continue
            route_probe = result.get("route_probe")
            if isinstance(route_probe, dict) and route_probe.get("success") is True:
                route_key = _model_cell_key(result)
                if route_key is not None:
                    successful_direct_routes.add(route_key)
        for probe in server.get("throughput_probes", []):
            if isinstance(probe, dict):
                route_key = _model_cell_key(probe)
                if route_key is not None:
                    if _throughput_probe_valid(probe):
                        throughput_routes.add(route_key)
                    else:
                        invalid_throughput.append(_format_model_cell(route_key))
    missing_throughput = successful_direct_routes - throughput_routes
    throughput_errors = []
    if missing_throughput:
        throughput_errors.append("missing " + "; ".join(_format_model_cell(cell) for cell in sorted(missing_throughput, key=_format_model_cell)[:8]))
    if invalid_throughput:
        throughput_errors.append("invalid " + "; ".join(sorted(invalid_throughput)[:8]))
    checks.append(
        CertificationCheck(
            "throughput_probe_evidence",
            "pass" if successful_direct_routes and not throughput_errors else "fail",
            f"{len(throughput_routes & successful_direct_routes)}/{len(successful_direct_routes)} direct route probe(s) include throughput"
            if successful_direct_routes and not throughput_errors
            else "; ".join(throughput_errors) if throughput_errors else "no successful direct route probes with throughput evidence",
        )
    )

    passing_cells = {
        _model_cell_key(row)
        for row in attempts
        if row.get("status") == "pass" and row.get("provider_type") == "local"
    }
    passing_cells.discard(None)
    missing_passing_routes = passing_cells - successful_direct_routes
    checks.append(
        CertificationCheck(
            "route_probe_cell_evidence",
            "pass" if passing_cells and not missing_passing_routes else "fail",
            f"{len(successful_direct_routes & passing_cells)}/{len(passing_cells)} passing local model cell(s) include successful direct route probes"
            if passing_cells and not missing_passing_routes
            else "missing " + "; ".join(_format_model_cell(cell) for cell in sorted(missing_passing_routes, key=_format_model_cell)[:8])
            if missing_passing_routes
            else "no passing local model cells",
        )
    )

    local_passes = [row for row in attempts if row.get("provider_type") == "local" and row.get("status") == "pass"]
    missing_resource_rows = [
        str(row.get("task_id") or row.get("served_model_name") or index)
        for index, row in enumerate(local_passes)
        if _as_float(row.get("peak_vram_mb")) is None or _as_float(row.get("gpu_utilization_pct")) is None
    ]
    checks.append(
        CertificationCheck(
            "local_resource_telemetry",
            "pass" if local_passes and not missing_resource_rows else "fail",
            f"{len(local_passes)} passing local row(s) include GPU telemetry"
            if local_passes and not missing_resource_rows
            else "missing peak_vram_mb/gpu_utilization_pct for " + ", ".join(missing_resource_rows[:8])
            if missing_resource_rows
            else "no passing local attempts",
        )
    )
    return checks


def _throughput_probe_valid(probe: dict[str, Any]) -> bool:
    prompt_chars = _as_int(probe.get("prompt_chars"))
    wall_time_s = _as_float(probe.get("wall_time_s"))
    completion_tokens = _as_int(probe.get("completion_tokens"))
    total_tokens = _as_int(probe.get("total_tokens"))
    tokens_per_s = _as_float(probe.get("tokens_per_s"))
    sample_count = _as_int(probe.get("sample_count"))
    tokens_per_s_p50 = _as_float(probe.get("tokens_per_s_p50"))
    tokens_per_s_p95 = _as_float(probe.get("tokens_per_s_p95"))
    return (
        prompt_chars is not None
        and prompt_chars > 0
        and wall_time_s is not None
        and wall_time_s > 0
        and completion_tokens is not None
        and completion_tokens > 0
        and total_tokens is not None
        and total_tokens >= completion_tokens
        and tokens_per_s is not None
        and tokens_per_s > 0
        and sample_count is not None
        and sample_count >= 3
        and tokens_per_s_p50 is not None
        and tokens_per_s_p50 > 0
        and tokens_per_s_p95 is not None
        and tokens_per_s_p95 > 0
    )


def _model_cell_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, int | None] | None:
    model = str(row.get("served_model_name") or row.get("model") or "")
    if not model:
        return None
    return (
        model,
        str(row.get("provider_type") or ""),
        str(row.get("hardware_profile") or "default"),
        str(row.get("weight_quant") or ""),
        str(row.get("kv_cache_dtype") or ""),
        _as_int(row.get("context_limit")),
    )


def _format_model_cell(cell: tuple[str, str, str, str, str, int | None]) -> str:
    return "/".join(str(part) for part in cell)


def _tool_file_efficiency_checks(passed: list[dict[str, Any]]) -> list[CertificationCheck]:
    missing_telemetry = [
        str(row.get("task_id") or index)
        for index, row in enumerate(passed)
        if _as_int(row.get("tool_calls")) is None
        or _as_int(row.get("files_read")) is None
        or _as_int(row.get("duplicate_file_reads")) is None
        or _as_float(row.get("time_to_first_relevant_file_s")) is None
        or _as_int(row.get("tool_calls")) <= 0
        or _as_int(row.get("files_read")) <= 0
        or (_as_int(row.get("duplicate_file_reads")) or 0) < 0
        or (_as_float(row.get("time_to_first_relevant_file_s")) or 0.0) < 0
    ]
    checks = [
        CertificationCheck(
            "tool_file_efficiency_telemetry",
            "pass" if passed and not missing_telemetry else "fail",
            f"{len(passed)} passing live row(s) include tool/file telemetry"
            if passed and not missing_telemetry
            else "missing/invalid tool_calls, files_read, duplicate_file_reads, or time_to_first_relevant_file_s for " + ", ".join(missing_telemetry[:8])
            if missing_telemetry
            else "no passing live attempts",
        )
    ]
    over_budget = []
    for key, rows in sorted(_group_efficiency_rows(passed).items()):
        tool_p95 = _percentile([float(_as_int(row.get("tool_calls")) or 0) for row in rows], 0.95)
        files_p95 = _percentile([float(_as_int(row.get("files_read")) or 0) for row in rows], 0.95)
        duplicate_p95 = _percentile([float(_as_int(row.get("duplicate_file_reads")) or 0) for row in rows], 0.95)
        first_relevant_p95 = _percentile([float(_as_float(row.get("time_to_first_relevant_file_s")) or 0.0) for row in rows], 0.95)
        if (
            tool_p95 > MAX_P95_TOOL_CALLS
            or files_p95 > MAX_P95_FILES_READ
            or duplicate_p95 > MAX_P95_DUPLICATE_FILE_READS
            or first_relevant_p95 > MAX_P95_TIME_TO_FIRST_RELEVANT_FILE_S
        ):
            over_budget.append(
                f"{'/'.join(key)} tools={tool_p95:g}/{MAX_P95_TOOL_CALLS} "
                f"files={files_p95:g}/{MAX_P95_FILES_READ} "
                f"dupes={duplicate_p95:g}/{MAX_P95_DUPLICATE_FILE_READS} "
                f"first_file={first_relevant_p95:g}/{MAX_P95_TIME_TO_FIRST_RELEVANT_FILE_S:g}"
            )
    checks.append(
        CertificationCheck(
            "tool_file_efficiency_budget",
            "pass" if passed and not over_budget else "fail",
            "p95 tool/file counts within budget"
            if passed and not over_budget
            else "; ".join(over_budget[:8]) if over_budget else "no passing live attempts",
        )
    )
    return checks


def _group_efficiency_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("provider_type") or ""), str(row.get("task_type") or ""))
        groups.setdefault(key, []).append(row)
    return groups


def _local_setup_context_sweep_check(local_setups_by_kv: dict[str, list[dict[str, Any]]]) -> CertificationCheck:
    missing = []
    for kv_mode in sorted(REQUIRED_LOCAL_KV_MODES):
        contexts = {_as_int(row.get("context_limit")) for row in local_setups_by_kv.get(kv_mode, [])}
        contexts.discard(None)
        missing_contexts = REQUIRED_CONTEXTS - contexts
        if missing_contexts:
            missing.append(f"{kv_mode}:{_format_ints(missing_contexts)}")
    if missing:
        return CertificationCheck("local_setup_context_sweep", "fail", "missing " + "; ".join(missing))
    return CertificationCheck("local_setup_context_sweep", "pass", "all required local KV modes cover required contexts")


def _local_context_ceiling_check(local_contexts: set[int | None]) -> CertificationCheck:
    numeric_contexts = {context for context in local_contexts if isinstance(context, int)}
    if not numeric_contexts:
        return CertificationCheck("local_context_ceiling", "fail", "no local context_limit rows")
    observed = max(numeric_contexts)
    required = max(REQUIRED_CONTEXTS)
    if observed < required:
        return CertificationCheck(
            "local_context_ceiling",
            "fail",
            f"maximum local context_limit={observed}; certification requires local coverage through {required}, so an 8k-only vLLM endpoint cannot certify the full local sweep",
        )
    return CertificationCheck(
        "local_context_ceiling",
        "pass",
        f"maximum local context_limit={observed}",
    )


def _local_hardware_setup_check(local_hardware_profiles: set[str], servers: list[dict[str, Any]]) -> CertificationCheck:
    server_profiles = {
        str(model.get("hardware_profile") or "default")
        for server in servers
        for model in server.get("models", [])
        if isinstance(model, dict) and model.get("provider_type") == "local"
    }
    missing_artifacts = local_hardware_profiles - server_profiles
    if len(local_hardware_profiles) < MIN_LOCAL_HARDWARE_PROFILES:
        return CertificationCheck(
            "local_hardware_setup_exploration",
            "fail",
            f"need at least {MIN_LOCAL_HARDWARE_PROFILES} local hardware/setup profile(s); found {', '.join(sorted(local_hardware_profiles)) or 'none'}",
        )
    if missing_artifacts:
        return CertificationCheck(
            "local_hardware_setup_exploration",
            "fail",
            "missing server model artifact profile(s): " + ", ".join(sorted(missing_artifacts)),
        )
    return CertificationCheck(
        "local_hardware_setup_exploration",
        "pass",
        f"hardware_profiles={', '.join(sorted(local_hardware_profiles))}",
    )


def _local_setup_concurrency_sweep_check(local_setups_by_kv: dict[str, list[dict[str, Any]]]) -> CertificationCheck:
    missing = []
    for kv_mode in sorted(REQUIRED_LOCAL_KV_MODES):
        concurrencies = {_as_int(row.get("concurrency")) for row in local_setups_by_kv.get(kv_mode, [])}
        concurrencies.discard(None)
        missing_concurrencies = REQUIRED_CONCURRENCIES - concurrencies
        if missing_concurrencies:
            missing.append(f"{kv_mode}:{_format_ints(missing_concurrencies)}")
    if missing:
        return CertificationCheck("local_setup_concurrency_sweep", "fail", "missing " + "; ".join(missing))
    return CertificationCheck("local_setup_concurrency_sweep", "pass", "all required local KV modes cover required concurrency levels")


def _local_setup_representative_task_check(local_passes: list[dict[str, Any]]) -> CertificationCheck:
    missing = []
    for kv_mode in sorted(REQUIRED_LOCAL_KV_MODES):
        task_types = {
            str(row.get("task_type", ""))
            for row in local_passes
            if str(row.get("kv_cache_dtype", "")) == kv_mode
        }
        missing_tasks = REQUIRED_LOCAL_SETUP_TASK_TYPES - task_types
        if missing_tasks:
            missing.append(f"{kv_mode}:{', '.join(sorted(missing_tasks))}")
    if missing:
        return CertificationCheck("local_setup_representative_tasks", "fail", "missing " + "; ".join(missing))
    return CertificationCheck("local_setup_representative_tasks", "pass", "all required local KV modes passed patch and instruction tasks")


def _local_concurrency_representative_task_check(local_passes: list[dict[str, Any]]) -> CertificationCheck:
    missing = []
    for level in sorted(REQUIRED_CONCURRENCIES):
        task_types = {
            str(row.get("task_type", ""))
            for row in local_passes
            if _as_int(row.get("concurrency")) == level
        }
        if not (task_types & REPRESENTATIVE_CONCURRENCY_TASK_TYPES):
            missing.append(str(level))
    if missing:
        return CertificationCheck("local_concurrency_representative_tasks", "fail", "missing representative patch/instruction rows at concurrency " + ", ".join(missing))
    return CertificationCheck("local_concurrency_representative_tasks", "pass", "all required concurrency levels include a representative patch/instruction task")


def _local_hardware_pairing_check(local_passes: list[dict[str, Any]]) -> CertificationCheck:
    profiles_by_cell: dict[tuple[str, str, str, int | None, int | None], set[str]] = {}
    for row in local_passes:
        key = (
            str(row.get("comparison_id") or row.get("model") or row.get("served_model_name") or ""),
            str(row.get("weight_quant", "")),
            str(row.get("kv_cache_dtype", "")),
            _as_int(row.get("context_limit")),
            _as_int(row.get("concurrency")),
        )
        profiles_by_cell.setdefault(key, set()).add(str(row.get("hardware_profile") or "default"))
    paired = {
        key: profiles
        for key, profiles in profiles_by_cell.items()
        if len(profiles) >= MIN_LOCAL_HARDWARE_PROFILES
    }
    paired_kv_modes = {key[2] for key in paired}
    missing_kv_modes = REQUIRED_LOCAL_KV_MODES - paired_kv_modes
    if missing_kv_modes:
        if not paired:
            return CertificationCheck(
                "local_hardware_setup_pairing",
                "fail",
                "no same model/weight/KV/context/concurrency cell has multiple hardware profiles",
            )
        return CertificationCheck(
            "local_hardware_setup_pairing",
            "fail",
            "missing paired hardware profiles for local KV mode(s): " + ", ".join(sorted(missing_kv_modes)),
        )
    key, profiles = sorted(paired.items(), key=lambda item: ("/".join(str(part) for part in item[0]), sorted(item[1])))[0]
    return CertificationCheck(
        "local_hardware_setup_pairing",
        "pass",
        f"paired KV modes={', '.join(sorted(paired_kv_modes))}; example {'/'.join(str(part) for part in key)} profiles={', '.join(sorted(profiles))}",
    )


def _external_required_task_types_check(external_passed_task_types: dict[str, set[str]]) -> CertificationCheck:
    missing = []
    for provider in sorted(EXTERNAL_PROVIDER_TYPES):
        missing_task_types = REQUIRED_TASK_TYPES - external_passed_task_types.get(provider, set())
        if missing_task_types:
            missing.append(f"{provider}:{', '.join(sorted(missing_task_types))}")
    if missing:
        return CertificationCheck("api_or_subscription_required_task_types_passed", "fail", "missing " + "; ".join(missing))
    return CertificationCheck("api_or_subscription_required_task_types_passed", "pass", "API and subscription providers passed all required task types")


def _external_context_coverage_check(external_contexts: dict[str, set[int | None]]) -> CertificationCheck:
    missing = []
    for provider in sorted(EXTERNAL_PROVIDER_TYPES):
        contexts = {context for context in external_contexts.get(provider, set()) if context is not None}
        missing_contexts = REQUIRED_EXTERNAL_CONTEXTS - contexts
        if missing_contexts:
            missing.append(f"{provider}:{_format_ints(missing_contexts)}")
    if missing:
        return CertificationCheck("api_or_subscription_context_coverage", "fail", "missing " + "; ".join(missing))
    return CertificationCheck("api_or_subscription_context_coverage", "pass", "API and subscription providers cover baseline and long-context cells")


def _external_concurrency_coverage_check(external_concurrencies: dict[str, set[int | None]]) -> CertificationCheck:
    missing = []
    for provider in sorted(EXTERNAL_PROVIDER_TYPES):
        concurrencies = {level for level in external_concurrencies.get(provider, set()) if level is not None}
        missing_levels = REQUIRED_EXTERNAL_CONCURRENCIES - concurrencies
        if missing_levels:
            missing.append(f"{provider}:{_format_ints(missing_levels)}")
    if missing:
        return CertificationCheck("api_or_subscription_concurrency_coverage", "fail", "missing " + "; ".join(missing))
    return CertificationCheck("api_or_subscription_concurrency_coverage", "pass", "API and subscription providers cover single, pooled, and higher-concurrency cells")


def _local_fp8_pairing_check(attempts: list[dict[str, Any]]) -> CertificationCheck:
    local_rows = [row for row in attempts if row.get("provider_type") == "local"]
    passing_fp8_keys = {
        _comparison_cell_key(row)
        for row in local_rows
        if row.get("kv_cache_dtype") == "fp8" and row.get("status") == "pass"
    }
    missing = []
    for row in local_rows:
        kv_mode = str(row.get("kv_cache_dtype", ""))
        if kv_mode in {"", "fp8"}:
            continue
        key = _comparison_cell_key(row)
        if key not in passing_fp8_keys:
            missing.append(f"{kv_mode}:{'/'.join(str(part) for part in key)}")
    if missing:
        return CertificationCheck("local_fp8_pairing", "fail", "missing passing fp8 baseline for " + "; ".join(sorted(missing)[:8]))
    return CertificationCheck("local_fp8_pairing", "pass", "all non-FP8 local rows have matching passing FP8 baselines")


def _comparison_cell_key(row: dict[str, Any]) -> tuple[str, str, str, str, int | None, int | None]:
    return (
        str(row.get("comparison_id") or row.get("model") or row.get("served_model_name") or ""),
        str(row.get("backend", "")),
        str(row.get("hardware_profile") or "default"),
        str(row.get("weight_quant", "")),
        _as_int(row.get("context_limit")),
        _as_int(row.get("concurrency")),
    )


def _route_probe_check(servers: list[dict[str, Any]]) -> CertificationCheck:
    serve_results = [
        result
        for server in servers
        for result in server.get("serve_results", [])
        if isinstance(result, dict)
    ]
    if not serve_results:
        return CertificationCheck("route_probes", "fail", "no live serve results")
    direct_success = 0
    direct_total = 0
    openclaw_success = 0
    openclaw_total = 0
    for result in serve_results:
        route_probe = result.get("route_probe")
        if not isinstance(route_probe, dict):
            continue
        if "success" in route_probe:
            direct_total += 1
            if route_probe.get("success") is True:
                direct_success += 1
        openclaw_route = route_probe.get("openclaw_route")
        if isinstance(openclaw_route, dict):
            openclaw_total += 1
            if openclaw_route.get("success") is True:
                openclaw_success += 1
    direct_ok = direct_success == direct_total
    if openclaw_total:
        return CertificationCheck(
            "route_probes",
            "pass" if direct_ok and openclaw_success == openclaw_total else "fail",
            f"direct={direct_success}/{direct_total}, openclaw={openclaw_success}/{openclaw_total}",
        )
    return CertificationCheck(
        "route_probes",
        "pass" if direct_total and direct_ok else "fail",
        f"direct={direct_success}/{direct_total}, openclaw=0/0",
    )


def _external_route_probe_check(attempts: list[dict[str, Any]], servers: list[dict[str, Any]]) -> CertificationCheck:
    external_models = {
        str(row.get("served_model_name"))
        for row in attempts
        if row.get("provider_type") in EXTERNAL_PROVIDER_TYPES
    }
    external_models.discard("")
    successful_models = set()
    for server in servers:
        for result in server.get("serve_results", []):
            if not isinstance(result, dict) or str(result.get("model")) not in external_models:
                continue
            route_probe = result.get("route_probe")
            if not isinstance(route_probe, dict):
                continue
            openclaw_route = route_probe.get("openclaw_route")
            if isinstance(openclaw_route, dict) and openclaw_route.get("success") is True:
                successful_models.add(str(result.get("model")))
    return CertificationCheck(
        "api_or_subscription_route_probes",
        "pass" if external_models and external_models <= successful_models else "fail",
        f"{len(successful_models & external_models)}/{len(external_models)} external model route probe(s)",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise json.JSONDecodeError("JSONL row is not an object", line, 0)
        rows.append(parsed)
    return rows


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _looks_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _serve_arg_int(command: list[Any], flag: str) -> int | None:
    for index, item in enumerate(command):
        if item == flag and index + 1 < len(command):
            return _as_int(command[index + 1])
        if isinstance(item, str) and item.startswith(f"{flag}="):
            return _as_int(item.split("=", 1)[1])
    return None


def _format_ints(values: set[int | None]) -> str:
    return ",".join(str(value) for value in sorted(value for value in values if value is not None))


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * pct)))
    return ordered[index]


def _missing_note(missing: set[str]) -> str:
    if not missing:
        return "all required task types covered"
    return "missing " + ", ".join(sorted(missing))


def _missing_int_note(missing: set[int]) -> str:
    if not missing:
        return "all required values covered"
    return "missing " + ", ".join(str(value) for value in sorted(missing))
