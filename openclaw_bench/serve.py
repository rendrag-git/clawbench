from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .models import ModelSpec
from .backend import _looks_like_context_window_error, _looks_like_tool_parser_error
from .telemetry import GpuTelemetrySampler, apply_gpu_telemetry


SERVE_PROBE_SAMPLES = 3


@dataclass
class ServerState:
    model: str
    kv_cache_dtype: str
    started: bool
    load_success: bool
    load_time_s: float
    pid: int | None = None
    peak_vram_mb: float | None = None
    gpu_utilization_pct: float | None = None
    request_errors: int = 0
    failure_type: str | None = None
    notes: str = ""
    route_probe: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict:
        return {
            "model": self.model,
            "kv_cache_dtype": self.kv_cache_dtype,
            "started": self.started,
            "load_success": self.load_success,
            "load_time_s": round(self.load_time_s, 3),
            "pid": self.pid,
            "peak_vram_mb": self.peak_vram_mb,
            "gpu_utilization_pct": self.gpu_utilization_pct,
            "request_errors": self.request_errors,
            "failure_type": self.failure_type,
            "notes": self.notes,
            "route_probe": self.route_probe,
        }


@contextmanager
def serve_model(model: ModelSpec, timeout_s: int = 120) -> Iterator[ServerState]:
    start = time.monotonic()
    if model.api_env and not os.environ.get(model.api_env):
        yield ServerState(
            model=model.served_model_name,
            kv_cache_dtype=model.kv_cache_dtype,
            started=False,
            load_success=False,
            load_time_s=0.0,
            failure_type="model_route_failed",
            notes=f"Missing environment variable {model.api_env}.",
        )
        return
    if not model.serve_command:
        if model.health_check_url:
            state = ServerState(
                model=model.served_model_name,
                kv_cache_dtype=model.kv_cache_dtype,
                started=False,
                load_success=False,
                load_time_s=0.0,
            )
            state.load_success, state.failure_type, state.notes = _wait_for_existing_server(model, timeout_s)
            if not state.load_success and state.failure_type in {"server_timeout", "model_route_failed", "serve_probe_failed"}:
                state.request_errors += 1
            if state.load_success:
                state.load_success, state.failure_type, route_notes = _smoke_model_route(model, timeout_s)
                if not state.load_success:
                    state.request_errors += 1
                state.notes = _join_notes(state.notes, route_notes)
            if state.load_success:
                state.load_success, state.failure_type, probe_notes, state.route_probe = _probe_model_route(model, timeout_s)
                if not state.load_success:
                    state.request_errors += 1
                state.notes = _join_notes(state.notes, probe_notes)
            state.load_time_s = time.monotonic() - start
            yield state
            return
        if not _allows_unverified_endpoint(model):
            yield ServerState(
                model=model.served_model_name,
                kv_cache_dtype=model.kv_cache_dtype,
                started=False,
                load_success=False,
                load_time_s=0.0,
                failure_type="model_load_failed",
                notes="No serve_command, health_check_url, or explicit support_status was configured.",
            )
            return
        yield ServerState(
            model=model.served_model_name,
            kv_cache_dtype=model.kv_cache_dtype,
            started=False,
            load_success=True,
            load_time_s=0.0,
            notes="No serve_command configured; using explicit external/API readiness assumption.",
        )
        return

    with GpuTelemetrySampler() as telemetry:
        env = _serve_env()
        env.update(model.serve_env)
        proc = subprocess.Popen(model.serve_command, env=env, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        state = ServerState(
            model=model.served_model_name,
            kv_cache_dtype=model.kv_cache_dtype,
            started=True,
            load_success=False,
            load_time_s=0.0,
            pid=proc.pid,
        )
        try:
            state.load_success, state.failure_type, state.notes = _wait_for_server(proc, model, timeout_s)
            if not state.load_success and state.failure_type in {"server_timeout", "model_route_failed", "serve_probe_failed"}:
                state.request_errors += 1
            if state.load_success:
                state.load_success, state.failure_type, route_notes = _smoke_model_route(model, timeout_s)
                if not state.load_success:
                    state.request_errors += 1
                state.notes = _join_notes(state.notes, route_notes)
            if state.load_success:
                state.load_success, state.failure_type, probe_notes, state.route_probe = _probe_model_route(model, timeout_s)
                if not state.load_success:
                    state.request_errors += 1
                state.notes = _join_notes(state.notes, probe_notes)
            state.load_time_s = time.monotonic() - start
            apply_gpu_telemetry(state, telemetry.result())
            yield state
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
            if proc.stderr is not None:
                proc.stderr.close()
            apply_gpu_telemetry(state, telemetry.result())


def _wait_for_server(proc: subprocess.Popen, model: ModelSpec, timeout_s: int) -> tuple[bool, str | None, str]:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stderr = proc.stderr.read() if proc.stderr else ""
            return False, _classify_load_failure(stderr), stderr[-4000:]
        if not model.health_check_url:
            return False, "model_load_failed", "Process started but no health_check_url was configured."
        success, failure_type, notes = _check_health_url(model)
        if success or failure_type:
            return success, failure_type, notes
        last_error = notes
        time.sleep(0.5)
    detail = f" Last error: {last_error}" if last_error else ""
    return False, "server_timeout", f"Timed out waiting for model server health check.{detail}"


def _classify_load_failure(stderr: str) -> str:
    text = stderr.lower()
    if "out of memory" in text or "cuda oom" in text or "oom" in text or "cannot allocate memory" in text:
        return "oom_on_load"
    if "unsupported" in text and ("kv" in text or "cache" in text or "dtype" in text or "quant" in text):
        return "unsupported_kv_dtype"
    if "kv cache dtype" in text and ("not support" in text or "not implemented" in text):
        return "unsupported_kv_dtype"
    return "model_load_failed"


def _serve_env() -> dict[str, str]:
    env = os.environ.copy()
    vllm_bin_dir = "/home/ubuntu/.venvs/vllm/bin"
    if _path_exists(vllm_bin_dir) and vllm_bin_dir not in env.get("PATH", "").split(":"):
        env["PATH"] = f"{vllm_bin_dir}:{env.get('PATH', '')}"
    cuda_home = "/usr/local/cuda"
    if _path_exists(Path(cuda_home, "bin", "nvcc")):
        env.setdefault("CUDA_HOME", cuda_home)
        cuda_bin = f"{cuda_home}/bin"
        if cuda_bin not in env.get("PATH", "").split(":"):
            env["PATH"] = f"{cuda_bin}:{env.get('PATH', '')}"
        cuda_lib = f"{cuda_home}/lib64"
        if cuda_lib not in env.get("LD_LIBRARY_PATH", "").split(":"):
            env["LD_LIBRARY_PATH"] = f"{cuda_lib}:{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else cuda_lib
    return env


def _path_exists(path: str | Path) -> bool:
    try:
        return Path(path).exists()
    except OSError:
        return False


def _wait_for_existing_server(model: ModelSpec, timeout_s: int) -> tuple[bool, str | None, str]:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        success, failure_type, notes = _check_health_url(model)
        if success or failure_type:
            return success, failure_type, notes
        last_error = notes
        time.sleep(0.5)
    detail = f" Last error: {last_error}" if last_error else ""
    return False, "server_timeout", f"Timed out waiting for model server health check.{detail}"


def _check_health_url(model: ModelSpec) -> tuple[bool, str | None, str]:
    if not model.health_check_url:
        return False, "model_load_failed", "No health_check_url was configured."
    headers = {}
    if model.api_env and os.environ.get(model.api_env):
        headers["Authorization"] = f"Bearer {os.environ[model.api_env]}"
    request = urllib.request.Request(model.health_check_url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            if 200 <= response.status < 300:
                return True, None, f"Health check returned HTTP {response.status}."
            return False, "server_timeout", f"Health check returned HTTP {response.status}."
    except urllib.error.HTTPError as exc:
        return False, "server_timeout", f"Health check returned HTTP {exc.code}."
    except (urllib.error.URLError, TimeoutError) as exc:
        return False, None, str(exc)


def _smoke_model_route(model: ModelSpec, timeout_s: int) -> tuple[bool, str | None, str]:
    if not model.api_base:
        return True, None, ""
    endpoint = model.api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model.served_model_name,
        "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
        "temperature": 0,
        "max_tokens": 4,
    }
    deadline = time.monotonic() + min(timeout_s, 30)
    last_error = ""
    while time.monotonic() < deadline:
        try:
            status, response_body, _ = _post_chat_completion(model, endpoint, payload, request_timeout_s=5)
            parsed = json.loads(response_body)
            choices = parsed.get("choices") if isinstance(parsed, dict) else None
            if not choices:
                return False, "model_route_failed", "Route smoke response did not include chat completion choices."
            return True, None, f"Route smoke returned HTTP {status} from {endpoint}."
        except urllib.error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            failure_type = _classify_route_failure(response_body, "model_route_failed")
            return False, failure_type, f"Route smoke returned HTTP {exc.code}: {response_body[-500:]}"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            time.sleep(0.5)
    detail = f" Last error: {last_error}" if last_error else ""
    return False, "model_route_failed", f"Timed out waiting for model route smoke at {endpoint}.{detail}"


def _probe_model_route(model: ModelSpec, timeout_s: int) -> tuple[bool, str | None, str, dict[str, Any]]:
    if not model.api_base:
        return True, None, "", {}
    endpoint = model.api_base.rstrip("/") + "/chat/completions"
    prompt = _build_probe_prompt(model.context_limit)
    payload = {
        "model": model.served_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 32,
    }
    samples = []
    for _ in range(SERVE_PROBE_SAMPLES):
        ok, failure_type, note, sample = _probe_model_route_once(model, endpoint, payload, timeout_s)
        if not ok:
            return False, failure_type, note, {}
        samples.append(sample)
    summary = _summarize_probe_samples(samples)
    route_probe = {
        "success": True,
        "endpoint": endpoint,
        "http_status": samples[-1]["http_status"],
        "prompt_chars": len(prompt),
        "sample_count": len(samples),
        "samples": samples,
        **summary,
    }
    return True, None, f"Serve probe completed {len(samples)} sample(s) against OpenAI-compatible route.", route_probe


def _probe_model_route_once(
    model: ModelSpec,
    endpoint: str,
    payload: dict[str, Any],
    timeout_s: int,
) -> tuple[bool, str | None, str, dict[str, Any]]:
    try:
        status, response_body, wall_time_s = _post_chat_completion(
            model,
            endpoint,
            payload,
            request_timeout_s=min(timeout_s, 30),
        )
        parsed = json.loads(response_body)
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        failure_type = _classify_route_failure(response_body, "serve_probe_failed")
        return False, failure_type, f"Serve probe returned HTTP {exc.code}: {response_body[-500:]}", {}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return False, "serve_probe_failed", f"Serve probe failed: {exc}", {}
    choices = parsed.get("choices") if isinstance(parsed, dict) else None
    if not choices:
        return False, "serve_probe_failed", "Serve probe response did not include chat completion choices.", {}
    usage = parsed.get("usage") if isinstance(parsed, dict) else None
    completion_tokens = _usage_int(usage, "completion_tokens")
    total_tokens = _usage_int(usage, "total_tokens")
    if completion_tokens == 0:
        completion_tokens = len(_choice_text(choices).split())
    return True, None, "", {
        "http_status": status,
        "wall_time_s": round(wall_time_s, 3),
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "tokens_per_s": round(completion_tokens / wall_time_s, 3) if wall_time_s > 0 and completion_tokens else None,
    }


def _summarize_probe_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    wall_times = [float(sample["wall_time_s"]) for sample in samples if isinstance(sample.get("wall_time_s"), (int, float))]
    completion_tokens = [int(sample["completion_tokens"]) for sample in samples if isinstance(sample.get("completion_tokens"), int)]
    total_tokens = [int(sample["total_tokens"]) for sample in samples if isinstance(sample.get("total_tokens"), int)]
    token_rates = [float(sample["tokens_per_s"]) for sample in samples if isinstance(sample.get("tokens_per_s"), (int, float))]
    return {
        "wall_time_s": round(sum(wall_times), 3) if wall_times else None,
        "wall_time_p50_s": _percentile(wall_times, 0.50),
        "wall_time_p95_s": _percentile(wall_times, 0.95),
        "completion_tokens": max(completion_tokens) if completion_tokens else 0,
        "total_tokens": max(total_tokens) if total_tokens else 0,
        "tokens_per_s": _percentile(token_rates, 0.50),
        "tokens_per_s_p50": _percentile(token_rates, 0.50),
        "tokens_per_s_p95": _percentile(token_rates, 0.95),
    }


def _post_chat_completion(
    model: ModelSpec,
    endpoint: str,
    payload: dict[str, Any],
    request_timeout_s: int,
) -> tuple[int, str, float]:
    headers = {"Content-Type": "application/json"}
    if model.api_env and os.environ.get(model.api_env):
        headers["Authorization"] = f"Bearer {os.environ[model.api_env]}"
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=request_timeout_s) as response:
        response_body = response.read().decode("utf-8", errors="replace")
        wall_time_s = time.monotonic() - start
        return response.status, response_body, wall_time_s


def _classify_route_failure(response_body: str, default_error: str) -> str:
    text = response_body.lower()
    if _looks_like_context_window_error(text):
        return "context_window_exceeded"
    if _looks_like_tool_parser_error(text):
        return "tool_parser_missing"
    return default_error


def _build_probe_prompt(context_limit: int) -> str:
    target_chars = max(256, min(context_limit // 2, 4096))
    prefix = "OpenClaw local model serve probe. Preserve this marker: OC_BENCH_PROBE. "
    filler = "alpha beta gamma delta epsilon zeta eta theta iota kappa "
    repeated = (filler * ((target_chars // len(filler)) + 1))[: max(0, target_chars - len(prefix))]
    return prefix + repeated + "\nReply with one short sentence that includes OC_BENCH_PROBE."


def _usage_int(usage: Any, key: str) -> int:
    if isinstance(usage, dict) and isinstance(usage.get(key), int):
        return usage[key]
    return 0


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * pct)))
    return round(ordered[index], 3)


def _choice_text(choices: Any) -> str:
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    text = first.get("text")
    return text if isinstance(text, str) else ""


def _allows_unverified_endpoint(model: ModelSpec) -> bool:
    if model.support_status == "simulator":
        return True
    if model.provider_type in {"api", "subscription"} and model.api_env:
        return bool(os.environ.get(model.api_env))
    return False


def _join_notes(*parts: str) -> str:
    return " ".join(part for part in parts if part)
