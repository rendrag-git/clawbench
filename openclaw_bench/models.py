from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


FAILURE_TYPES = {
    "model_load_failed",
    "unsupported_kv_dtype",
    "oom_on_load",
    "oom_during_run",
    "openclaw_timeout",
    "server_timeout",
    "bad_json",
    "wrong_file",
    "hallucinated_file",
    "wrong_needle",
    "test_failed",
    "patch_unrelated",
    "instruction_violation",
    "tool_loop",
    "tool_parser_missing",
    "context_window_exceeded",
    "incomplete_result",
    "model_override_unauthorized",
    "openclaw_embedded_fallback",
    "openclaw_agent_setup_failed",
    "model_route_failed",
    "serve_probe_failed",
    "unknown",
}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    served_model_name: str
    openclaw_model_name: str | None = None
    comparison_id: str | None = None
    provider_type: str = "local"
    hardware_profile: str = "default"
    weight_quant: str = "unknown"
    kv_cache_dtype: str = "fp8"
    context_limit: int = 32768
    serve_args: list[str] = field(default_factory=list)
    serve_command: list[str] = field(default_factory=list)
    serve_env: dict[str, str] = field(default_factory=dict)
    health_check_url: str | None = None
    expected_support: str = ""
    support_status: str = "unknown"
    api_base: str | None = None
    api_env: str | None = None

    @classmethod
    def from_alias(cls, alias: str, kv_cache_dtype: str, context_limit: int) -> "ModelSpec":
        return cls(
            model_id=alias,
            served_model_name=alias,
            kv_cache_dtype=kv_cache_dtype,
            context_limit=context_limit,
        )

    @classmethod
    def from_mapping(cls, data: dict[str, Any], kv_cache_dtype: str | None = None, context_limit: int | None = None) -> "ModelSpec":
        merged = dict(data)
        if kv_cache_dtype is not None:
            merged["kv_cache_dtype"] = kv_cache_dtype
        if context_limit is not None:
            merged["context_limit"] = context_limit
        return cls(**merged)

    @property
    def openclaw_route_model(self) -> str:
        return self.openclaw_model_name or self.served_model_name

    @property
    def comparison_key(self) -> str:
        return self.comparison_id or self.model_id or self.served_model_name


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    task_type: str
    fixture: str
    prompt: str
    expected: dict[str, Any]
    verify_command: list[str] = field(default_factory=list)
    context_sizes: list[int] = field(default_factory=list)
    max_changed_files: int = 6
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SuiteManifest:
    suite_id: str
    tasks: list[TaskSpec]
    root: Path
    source_paths: list[Path] = field(default_factory=list)


@dataclass
class BackendResponse:
    text: str
    json_output: dict[str, Any] | None
    raw: dict[str, Any]
    tool_calls: int = 0
    files_read: int = 0
    duplicate_file_reads: int | None = None
    time_to_first_relevant_file_s: float | None = None
    ttft_s: float | None = None
    request_errors: int = 0
    timed_out: bool = False
    error: str | None = None


@dataclass
class AttemptResult:
    run_id: str
    model: str
    served_model_name: str
    backend: str
    provider_type: str
    hardware_profile: str
    weight_quant: str
    kv_cache_dtype: str
    context_limit: int
    concurrency: int
    task_id: str
    task_type: str
    task_tags: list[str]
    workspace_id: str
    status: str
    score: float
    wall_time_s: float
    ttft_s: float | None
    tool_calls: int
    files_read: int
    duplicate_file_reads: int | None
    time_to_first_relevant_file_s: float | None
    files_changed: int
    changed_files: list[str]
    tests_passed: bool
    json_valid: bool
    hallucinated_paths: int
    oom: bool
    timeout: bool
    peak_vram_mb: float | None = None
    gpu_utilization_pct: float | None = None
    request_errors: int = 0
    failure_type: str | None = None
    notes: str = ""
    comparison_id: str | None = None
    run_index: int = 0  # 0-indexed seed within (model, KV, context, concurrency, task) cell when --runs-per-task > 1

    def to_row(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model": self.model,
            "served_model_name": self.served_model_name,
            "comparison_id": self.comparison_id or self.model or self.served_model_name,
            "backend": self.backend,
            "provider_type": self.provider_type,
            "hardware_profile": self.hardware_profile,
            "weight_quant": self.weight_quant,
            "kv_cache_dtype": self.kv_cache_dtype,
            "context_limit": self.context_limit,
            "concurrency": self.concurrency,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "task_tags": self.task_tags,
            "workspace_id": self.workspace_id,
            "status": self.status,
            "score": self.score,
            "wall_time_s": round(self.wall_time_s, 3),
            "ttft_s": None if self.ttft_s is None else round(self.ttft_s, 3),
            "tool_calls": self.tool_calls,
            "files_read": self.files_read,
            "duplicate_file_reads": self.duplicate_file_reads,
            "time_to_first_relevant_file_s": None if self.time_to_first_relevant_file_s is None else round(self.time_to_first_relevant_file_s, 3),
            "files_changed": self.files_changed,
            "changed_files": self.changed_files,
            "tests_passed": self.tests_passed,
            "json_valid": self.json_valid,
            "hallucinated_paths": self.hallucinated_paths,
            "oom": self.oom,
            "timeout": self.timeout,
            "peak_vram_mb": self.peak_vram_mb,
            "gpu_utilization_pct": self.gpu_utilization_pct,
            "request_errors": self.request_errors,
            "failure_type": self.failure_type,
            "notes": self.notes,
            "run_index": self.run_index,
        }
