from __future__ import annotations

import json
import os
import signal
import subprocess
from contextlib import suppress
from pathlib import Path
from typing import Protocol

from .models import BackendResponse, ModelSpec, TaskSpec


class AgentBackend(Protocol):
    def smoke(self, model: ModelSpec, timeout_s: int) -> BackendResponse:
        ...

    def run(self, model: ModelSpec, task: TaskSpec, workspace: Path, session_id: str, timeout_s: int) -> BackendResponse:
        ...


class OpenClawBackend(AgentBackend):
    def __init__(
        self,
        profile: str = "bench",
        agent: str = "main",
        local: bool = False,
        thinking: str | None = None,
        workspace_agents: bool = False,
        container: str | None = None,
    ) -> None:
        self.profile = profile
        self.agent = agent
        self.local = local
        self.thinking = thinking
        self.workspace_agents = workspace_agents
        self.container = container

    def _cmd(self, cwd: Path | None = None) -> list[str]:
        if not self.container:
            return ["openclaw"]
        cmd = ["docker", "exec"]
        if cwd is not None:
            cmd.extend(["-w", str(cwd)])
        cmd.extend([self.container, "openclaw"])
        return cmd

    def smoke(self, model: ModelSpec, timeout_s: int) -> BackendResponse:
        cmd = [
            *self._cmd(),
            "--profile",
            self.profile,
            "infer",
            "model",
            "run",
            "--model",
            model.openclaw_route_model,
            "--prompt",
            "Reply with exactly: ok",
            "--json",
        ]
        cmd.append("--local" if self.local else "--gateway")
        return _run_openclaw_command(cmd, None, timeout_s, "model_route_failed")

    def run(self, model: ModelSpec, task: TaskSpec, workspace: Path, session_id: str, timeout_s: int) -> BackendResponse:
        agent = self.agent
        if self.workspace_agents:
            agent = _agent_id(session_id)
            setup = _ensure_workspace_agent(
                self.profile,
                agent,
                workspace,
                model.openclaw_route_model,
                timeout_s=min(timeout_s, 60),
                container=self.container,
            )
            if setup is not None:
                return setup

        cmd = [
            *self._cmd(workspace),
            "--profile",
            self.profile,
            "agent",
            "--agent",
            agent,
            "--session-id",
            session_id,
            "--message",
            task.prompt,
            "--timeout",
            str(timeout_s),
            "--json",
        ]
        if not self.workspace_agents:
            cmd.extend(["--model", model.openclaw_route_model])
        if self.local:
            cmd.append("--local")
        if self.thinking:
            cmd.extend(["--thinking", self.thinking])
        cwd = None if self.container else workspace
        return _run_openclaw_command(cmd, cwd, timeout_s + 5, "unknown")


class SimulatorBackend(AgentBackend):
    """Deterministic backend used to certify the harness without a live LLM."""

    def smoke(self, model: ModelSpec, timeout_s: int) -> BackendResponse:
        del timeout_s
        return BackendResponse(
            text="ok",
            json_output={"text": "ok"},
            raw={"simulated": True, "model": model.served_model_name},
        )

    def run(self, model: ModelSpec, task: TaskSpec, workspace: Path, session_id: str, timeout_s: int) -> BackendResponse:
        del model, timeout_s
        if task.task_type == "workspace_discovery":
            payload = {
                "test_command": "python -m unittest discover -s tests",
                "routes_file": "api/routes.py",
                "schema_file": "db/schema.py",
            }
            return BackendResponse(
                text=json.dumps(payload),
                json_output=payload,
                raw={"simulated": True, "session_id": session_id},
                tool_calls=3,
                files_read=4,
                duplicate_file_reads=0,
                time_to_first_relevant_file_s=0.2,
            )

        if task.task_type == "repo_read_only":
            payload = {
                "answer": task.expected.get("answer"),
                "evidence_files": task.expected.get("evidence_files", []),
            }
            return BackendResponse(
                text=json.dumps(payload),
                json_output=payload,
                raw={"simulated": True, "session_id": session_id},
                tool_calls=3,
                files_read=len(payload["evidence_files"]),
                duplicate_file_reads=0,
                time_to_first_relevant_file_s=0.2,
            )

        if task.task_type == "repo_code_edit":
            target = workspace / "services" / "orders.py"
            target.write_text(
                "def create_order(payload):\n"
                "    return {\n"
                "        \"order_id\": payload[\"order_id\"],\n"
                "        \"status\": \"created\",\n"
                "    }\n\n\n"
                "def order_status(order_id):\n"
                "    status = \"shipped\" if str(order_id).startswith(\"SHIP-\") else \"processing\"\n"
                "    return {\n"
                "        \"order_id\": order_id,\n"
                "        \"status\": status,\n"
                "    }\n",
                encoding="utf-8",
            )
            return BackendResponse(
                text="Updated services/orders.py to report shipped status for shipped order ids.",
                json_output=None,
                raw={"simulated": True, "session_id": session_id},
                tool_calls=5,
                files_read=4,
                duplicate_file_reads=1,
                time_to_first_relevant_file_s=0.4,
            )

        if task.task_type == "multi_file_bug_trace":
            target = workspace / "app" / "discounts.py"
            target.write_text(
                "def vip_discount_rate(customer):\n"
                "    if customer.get(\"tier\") == \"vip\":\n"
                "        return 0.10\n"
                "    return 0.0\n",
                encoding="utf-8",
            )
            return BackendResponse(
                text="Bug path: tests/test_discount.py -> app/routes.py -> app/discounts.py. Fixed VIP discount rate.",
                json_output=None,
                raw={"simulated": True, "session_id": session_id},
                tool_calls=5,
                files_read=5,
                duplicate_file_reads=1,
                time_to_first_relevant_file_s=0.5,
            )

        if task.task_type == "patch_execution":
            target = workspace / "app" / "slug.py"
            target.write_text(
                "import re\n\n\n"
                "def slugify(value):\n"
                "    value = value.strip().lower()\n"
                "    value = re.sub(r\"[^a-z0-9]+\", \"-\", value)\n"
                "    return value.strip(\"-\")\n",
                encoding="utf-8",
            )
            return BackendResponse(
                text="Implemented slugify in app/slug.py.",
                json_output=None,
                raw={"simulated": True, "session_id": session_id},
                tool_calls=4,
                files_read=3,
                duplicate_file_reads=0,
                time_to_first_relevant_file_s=0.3,
            )

        if task.task_type == "workspace_needle":
            token = _read_needle(workspace)
            health = workspace / "app" / "health.py"
            health.write_text(f"def health():\n    return {{\"status\": \"ok\", \"token\": \"{token}\"}}\n", encoding="utf-8")
            return BackendResponse(
                text=f"Found BENCHMARK_NEEDLE_TOKEN in app/config_notes.py and updated app/health.py.",
                json_output=None,
                raw={"simulated": True, "token": token, "session_id": session_id},
                tool_calls=7,
                files_read=8,
                duplicate_file_reads=2,
                time_to_first_relevant_file_s=0.8,
            )

        if task.task_type == "instruction_retention":
            target = workspace / "app" / "service.py"
            target.write_text(
                "from app.helpers import normalize_status\n\n\n"
                "def render_status(value):\n"
                "    return {\"status\": normalize_status(value)}\n",
                encoding="utf-8",
            )
            payload = {"changed": ["app/service.py"], "used_existing_helper": True}
            return BackendResponse(
                text=json.dumps(payload),
                json_output=payload,
                raw={"simulated": True, "session_id": session_id},
                tool_calls=4,
                files_read=4,
                duplicate_file_reads=0,
                time_to_first_relevant_file_s=0.3,
            )

        return BackendResponse(text="", json_output=None, raw={"simulated": True, "session_id": session_id}, error="unknown_task")


def make_backend(
    name: str,
    profile: str = "bench",
    agent: str = "main",
    local: bool = False,
    thinking: str | None = None,
    workspace_agents: bool = False,
    container: str | None = None,
) -> AgentBackend:
    if name == "simulator":
        return SimulatorBackend()
    if name == "openclaw":
        return OpenClawBackend(
            profile=profile,
            agent=agent,
            local=local,
            thinking=thinking,
            workspace_agents=workspace_agents,
            container=container,
        )
    raise ValueError(f"unknown backend: {name}")


def _ensure_workspace_agent(
    profile: str,
    agent: str,
    workspace: Path,
    model_ref: str,
    timeout_s: int,
    container: str | None = None,
) -> BackendResponse | None:
    agent_dir = workspace.parent / ".openclaw-agent-state" / agent
    cmd = [
        *(_container_openclaw_cmd(container) if container else ["openclaw"]),
        "--profile",
        profile,
        "agents",
        "add",
        agent,
        "--workspace",
        str(workspace),
        "--agent-dir",
        str(agent_dir),
        "--model",
        model_ref,
        "--non-interactive",
        "--json",
    ]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        return BackendResponse(
            text=_as_text(exc.stdout),
            json_output=None,
            raw={"cmd": cmd, "timeout": True, "stderr": _as_text(exc.stderr)},
            request_errors=1,
            timed_out=True,
            error="openclaw_agent_setup_failed",
        )
    output = f"{proc.stdout}\n{proc.stderr}"
    if proc.returncode == 0 or "already exists" in output.lower():
        return None
    return BackendResponse(
        text=proc.stdout,
        json_output=None,
        raw={"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr},
        request_errors=1,
        error="openclaw_agent_setup_failed",
    )


def _container_openclaw_cmd(container: str) -> list[str]:
    return ["docker", "exec", container, "openclaw"]


def _agent_id(session_id: str) -> str:
    safe = "".join(char if char.isalnum() or char == "-" else "-" for char in session_id.lower())
    safe = safe.strip("-") or "attempt"
    return f"bench-{safe[:56]}".rstrip("-")


def _run_openclaw_command(cmd: list[str], cwd: Path | None, timeout_s: int, default_error: str) -> BackendResponse:
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(proc)
        stdout = _as_text(exc.stdout)
        stderr = _as_text(exc.stderr)
        if not stdout and not stderr:
            stdout, stderr = _communicate_after_timeout(proc)
        error = _classify_openclaw_error(stdout, stderr, "openclaw_timeout")
        return BackendResponse(
            text=stdout,
            json_output=None,
            raw={"cmd": cmd, "timeout": True, "stderr": stderr},
            request_errors=1,
            timed_out=error == "openclaw_timeout",
            error=error,
        )

    parsed = _parse_json_object(stdout)
    text = _extract_text(parsed) if parsed else stdout
    error = _classify_openclaw_error(stdout, stderr, "") or _classify_parsed_openclaw_error(parsed) or None
    if proc.returncode != 0 and not error:
        error = default_error
    request_errors = _extract_int(parsed, "request_errors") if parsed else 0
    if proc.returncode != 0 or error:
        request_errors = max(1, request_errors)
    return BackendResponse(
        text=text,
        json_output=parsed if isinstance(parsed, dict) else None,
        raw={"cmd": cmd, "returncode": proc.returncode, "stdout": stdout, "stderr": stderr},
        tool_calls=_extract_int(parsed, "tool_calls") if parsed else 0,
        files_read=_extract_int(parsed, "files_read") if parsed else 0,
        duplicate_file_reads=_extract_duplicate_file_reads(parsed) if parsed else None,
        time_to_first_relevant_file_s=_extract_float(parsed, "time_to_first_relevant_file_s") if parsed else None,
        ttft_s=_extract_float(parsed, "ttft_s") if parsed else None,
        request_errors=request_errors,
        timed_out=False,
        error=error,
    )


def _terminate_process_group(proc: subprocess.Popen) -> None:
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)
    with suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=5)


def _communicate_after_timeout(proc: subprocess.Popen) -> tuple[str, str]:
    try:
        stdout, stderr = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        return "", ""
    return _as_text(stdout), _as_text(stderr)


def _parse_json_object(output: str) -> dict | None:
    with suppress(json.JSONDecodeError):
        parsed = json.loads(output)
        return parsed if isinstance(parsed, dict) else None
    start = output.find("{")
    end = output.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(output[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _extract_text(parsed: dict) -> str:
    for key in ("text", "message", "output", "final", "response"):
        value = parsed.get(key)
        if isinstance(value, str):
            return value
    return json.dumps(parsed, sort_keys=True)


def _classify_openclaw_error(stdout: str, stderr: str, default_error: str = "unknown") -> str:
    text = f"{stdout}\n{stderr}".lower()
    if _looks_like_context_window_error(text):
        return "context_window_exceeded"
    if "provider/model overrides are not authorized" in text:
        return "model_override_unauthorized"
    if "incomplete terminal response" in text or "incomplete_result" in text:
        return "incomplete_result"
    if "embedded fallback" in text or "fallbackfrom" in text and "gateway" in text:
        return "openclaw_embedded_fallback"
    if _looks_like_tool_parser_error(text):
        return "tool_parser_missing"
    if "unsupported" in text and ("kv" in text or "cache" in text or "dtype" in text):
        return "unsupported_kv_dtype"
    if "unknown model" in text or "model_not_found" in text or "model not found" in text:
        return "model_route_failed"
    if "out of memory" in text or "cuda oom" in text or "oom" in text:
        return "oom_during_run"
    if "timeout" in text or "timed out" in text:
        return "server_timeout"
    if "tool loop" in text or "too many tool" in text:
        return "tool_loop"
    return default_error


def _classify_parsed_openclaw_error(parsed: dict | None) -> str:
    if not isinstance(parsed, dict):
        return ""
    meta = parsed.get("meta")
    if isinstance(meta, dict):
        error = meta.get("error")
        if isinstance(error, dict):
            error_text = "\n".join(str(value) for value in (error.get("kind"), error.get("message")) if value)
            classified = _classify_openclaw_error("", error_text, "")
            if classified:
                return classified
        if meta.get("fallbackFrom") == "gateway":
            return "openclaw_embedded_fallback"
    payloads = parsed.get("payloads")
    if isinstance(payloads, list):
        payload_text = "\n".join(
            str(payload.get("text"))
            for payload in payloads
            if isinstance(payload, dict) and isinstance(payload.get("text"), str)
        )
        classified = _classify_openclaw_error(payload_text, "", "")
        if classified:
            return classified
    return ""


def _looks_like_context_window_error(text: str) -> bool:
    return (
        "maximum context length" in text
        or "context length" in text and ("exceed" in text or "requested" in text)
        or "context window" in text and "exceed" in text
        or "input length" in text and "exceed" in text
        or "prompt contains" in text and "output tokens" in text
    )


def _looks_like_tool_parser_error(text: str) -> bool:
    return (
        "tool-call-parser" in text
        or "tool call parser" in text
        or "tool parser" in text
        or '"auto" tool choice requires' in text
        or "'auto' tool choice requires" in text
    )


def _extract_int(parsed: dict, key: str) -> int:
    value = parsed.get(key)
    if isinstance(value, int):
        return value
    metrics = parsed.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get(key), int):
        return metrics[key]
    return 0


def _extract_float(parsed: dict, key: str) -> float | None:
    value = parsed.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    metrics = parsed.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get(key), (int, float)):
        return float(metrics[key])
    return None


def _extract_duplicate_file_reads(parsed: dict) -> int | None:
    explicit = _extract_optional_int(parsed, "duplicate_file_reads")
    if explicit is not None:
        return explicit
    paths = _extract_file_read_paths(parsed)
    if not paths:
        return None
    return len(paths) - len(set(paths))


def _extract_optional_int(parsed: dict, key: str) -> int | None:
    value = parsed.get(key)
    if isinstance(value, int):
        return value
    metrics = parsed.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get(key), int):
        return metrics[key]
    return None


def _extract_file_read_paths(parsed: dict) -> list[str]:
    candidates = [
        parsed.get("file_reads"),
        parsed.get("files_read_paths"),
        parsed.get("read_files"),
    ]
    metrics = parsed.get("metrics")
    if isinstance(metrics, dict):
        candidates.extend([metrics.get("file_reads"), metrics.get("files_read_paths"), metrics.get("read_files")])
    paths: list[str] = []
    for candidate in candidates:
        if isinstance(candidate, list):
            for item in candidate:
                if isinstance(item, str):
                    paths.append(item)
                elif isinstance(item, dict) and isinstance(item.get("path"), str):
                    paths.append(item["path"])
    return paths


def _read_needle(workspace: Path) -> str:
    notes = (workspace / "app" / "config_notes.py").read_text(encoding="utf-8")
    for line in notes.splitlines():
        if line.startswith("BENCHMARK_NEEDLE_TOKEN"):
            return line.split("=", 1)[1].strip().strip("\"'")
    raise RuntimeError("needle token not found")
