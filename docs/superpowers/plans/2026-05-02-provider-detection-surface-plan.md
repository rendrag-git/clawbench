# Provider Detection Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the inspect-first provider detection surface for `oc-bench init` so vLLM/Ollama/llama.cpp/LM Studio endpoints are auto-discovered, configured into OpenClaw correctly, and verified before benchmarks run — making the bench shippable across Mac/Linux/Windows native and containerized OC setups without per-environment debugging.

**Architecture:** New `openclaw_bench/providers/` package with a runtime-aware probe layer (`probes.py`), a detection cascade (`detect.py`) that scans existing OC profiles before port-probing with a 30s/provider budget, and per-provider modules (`vllm.py` full; `ollama.py`, `llamacpp.py`, `lmstudio.py` detect-only stubs). Reuses `quickstart.py`'s existing `_vllm_provider_config`, `_external_vllm_model`, `_openclaw_route_context` helpers — those encode bug fixes (16k context floor, meta fields, plugin entries, `chatTemplateKwargs.enable_thinking=false`) that must not be re-introduced. New `oc-bench preflight` command wraps the four verification gates. `oc-bench init` is extended to call detection automatically with `--no-detect` and `--oc-runtime` escape hatches.

**Tech Stack:** Python 3.10+, `unittest`, `urllib.request` (no new deps), `subprocess` for shell-out probes (`incus exec`, `docker exec`, `ssh`), existing `openclaw_bench` package layout.

**Spec reference:** `docs/superpowers/specs/2026-05-02-provider-detection-surface-design.md`

---

## File Structure

### Create

| Path | Responsibility |
|---|---|
| `openclaw_bench/providers/__init__.py` | Package marker; re-exports `ProbeResult`, `Probe`, `LocalProbe`, `IncusExecProbe`, `DockerExecProbe`, `SSHProbe`, `ProviderCandidate`, `DetectionReport`, `run_detection`. |
| `openclaw_bench/providers/probes.py` | `ProbeResult` dataclass, `Probe` Protocol, four concrete probe classes (`LocalProbe`, `IncusExecProbe`, `DockerExecProbe`, `SSHProbe`). Each probe exposes `http_get(url, timeout_s) -> ProbeResult`. |
| `openclaw_bench/providers/detect.py` | `ProviderCandidate`, `DetectionReport`, `run_detection()` cascade, `_scan_existing_oc_profiles()`, `_port_probe()`, `derive_probes_for_profile()`. |
| `openclaw_bench/providers/vllm.py` | `detect()`, `generate_route_config()` (delegates to `quickstart._vllm_provider_config`), `parameter_shaping()` for Qwen + GPT-OSS rules. |
| `openclaw_bench/providers/ollama.py` | `detect()` only; `generate_route_config()` raises `NotImplementedError`. |
| `openclaw_bench/providers/llamacpp.py` | Same shape as `ollama.py`. |
| `openclaw_bench/providers/lmstudio.py` | Same shape as `ollama.py`. |
| `tests/test_providers_probes.py` | Probe unit tests (HTTP success/error/timeout, subprocess shell-out shape). |
| `tests/test_providers_detect.py` | Cascade unit tests (zero/one/many candidates, already-known scan, mismatch finding, runtime auto-derive). |
| `tests/test_providers_vllm.py` | vLLM module unit tests (detect from response fixture, generator parity with `quickstart._vllm_provider_config`, parameter shaping for Qwen vs GPT-OSS). |
| `tests/test_providers_ollama.py` | Ollama detect tests; assert `generate_route_config` raises. |
| `tests/test_providers_llamacpp.py` | llama.cpp detect tests; assert `generate_route_config` raises. |
| `tests/test_providers_lmstudio.py` | LM Studio detect tests; assert `generate_route_config` raises. |
| `tests/test_providers_preflight.py` | `oc-bench preflight` CLI integration test with stubbed gates. |
| `tests/test_providers_live.py` | Live integration test gated by `OC_BENCH_LIVE=1`; hits real GPT-OSS endpoint via configured probe. |
| `tests/fixtures/provider_responses/vllm_v1_models.json` | Captured/realistic `/v1/models` response shape. |
| `tests/fixtures/provider_responses/ollama_api_tags.json` | Captured/realistic `/api/tags` response shape. |
| `tests/fixtures/provider_responses/llamacpp_v1_models.json` | Captured/realistic llama.cpp `/v1/models` response shape. |
| `tests/fixtures/provider_responses/lmstudio_v1_models.json` | Captured/realistic LM Studio `/v1/models` response shape. |

### Modify

| Path | Change |
|---|---|
| `openclaw_bench/cli.py` | Add `preflight` subcommand; extend `init` with `--no-detect` and `--oc-runtime` flags; route detection results into `init_quickstart`. |
| `openclaw_bench/quickstart.py` | Accept an optional `ProviderCandidate` (when detection ran) in `init_quickstart`; keep env-var defaults as the `--no-detect` path. |
| `openclaw_bench/preflight.py` | Add public `run_verification_gates(profile, candidate, *, container, route_model, home)` returning a `PreflightResult` aggregating the four gates; reuse existing private helpers. |

---

## Task 1: ProbeResult dataclass + LocalProbe HTTP GET

**Files:**
- Create: `openclaw_bench/providers/__init__.py`
- Create: `openclaw_bench/providers/probes.py`
- Test: `tests/test_providers_probes.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_providers_probes.py
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from openclaw_bench.providers.probes import LocalProbe, ProbeResult


class _Handler(BaseHTTPRequestHandler):
    payload = {"object": "list", "data": []}

    def do_GET(self):  # noqa: N802
        body = json.dumps(self.payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args, **_kwargs):  # silence stderr noise
        return


def _serve_once():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


class LocalProbeTests(unittest.TestCase):
    def test_http_get_returns_success_result_with_body(self):
        server = _serve_once()
        try:
            host, port = server.server_address
            url = f"http://{host}:{port}/v1/models"
            result = LocalProbe().http_get(url, timeout_s=2.0)
        finally:
            server.shutdown()

        self.assertIsInstance(result, ProbeResult)
        self.assertTrue(result.ok)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(json.loads(result.body), {"object": "list", "data": []})
        self.assertEqual(result.probe_name, "host")
        self.assertIsNone(result.error)
```

- [ ] **Step 2: Run test, verify fail**

```bash
cd /home/ubuntu/projects/openclaw-local-model-bench
python3 -m unittest tests.test_providers_probes -v
```

Expected: `ModuleNotFoundError: No module named 'openclaw_bench.providers'`.

- [ ] **Step 3: Create the package and minimal probe module**

```python
# openclaw_bench/providers/__init__.py
from .probes import LocalProbe, Probe, ProbeResult

__all__ = ["LocalProbe", "Probe", "ProbeResult"]
```

```python
# openclaw_bench/providers/probes.py
from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    status_code: int | None
    body: str
    probe_name: str
    error: str | None


class Probe(Protocol):
    name: str

    def http_get(self, url: str, *, timeout_s: float) -> ProbeResult: ...


class LocalProbe:
    name = "host"

    def http_get(self, url: str, *, timeout_s: float) -> ProbeResult:
        request = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                body = response.read().decode("utf-8", errors="replace")
                return ProbeResult(
                    ok=True,
                    status_code=response.status,
                    body=body,
                    probe_name=self.name,
                    error=None,
                )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            return ProbeResult(
                ok=False,
                status_code=exc.code,
                body=body,
                probe_name=self.name,
                error=f"http_{exc.code}",
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return ProbeResult(
                ok=False,
                status_code=None,
                body="",
                probe_name=self.name,
                error=str(exc),
            )
```

- [ ] **Step 4: Run test, verify pass**

```bash
python3 -m unittest tests.test_providers_probes -v
```

Expected: `OK` with 1 test.

- [ ] **Step 5: Add a timeout regression test**

Append to `tests/test_providers_probes.py`:

```python
class LocalProbeTimeoutTests(unittest.TestCase):
    def test_http_get_returns_failure_when_endpoint_unreachable(self):
        # Reserved test address per RFC 5737; routes nowhere fast enough to hit timeout.
        result = LocalProbe().http_get("http://192.0.2.1:18080/v1/models", timeout_s=0.5)
        self.assertFalse(result.ok)
        self.assertIsNone(result.status_code)
        self.assertIsNotNone(result.error)
```

- [ ] **Step 6: Run timeout test**

```bash
python3 -m unittest tests.test_providers_probes -v
```

Expected: `OK` with 2 tests.

- [ ] **Step 7: Commit**

```bash
git add openclaw_bench/providers/__init__.py openclaw_bench/providers/probes.py tests/test_providers_probes.py
git commit -m "$(cat <<'EOF'
Add Probe protocol and LocalProbe with HTTP GET

Foundation for the provider-detection cascade. LocalProbe runs probes
from the bench's Python process; later commits add subprocess-backed
probes for containerized and remote OpenClaw runtimes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: IncusExecProbe + DockerExecProbe + SSHProbe

**Files:**
- Modify: `openclaw_bench/providers/probes.py`
- Modify: `openclaw_bench/providers/__init__.py`
- Test: `tests/test_providers_probes.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_providers_probes.py`:

```python
import subprocess
from unittest.mock import patch

from openclaw_bench.providers.probes import (
    DockerExecProbe,
    IncusExecProbe,
    SSHProbe,
)


class IncusExecProbeTests(unittest.TestCase):
    def test_http_get_shells_out_to_incus_exec_curl(self):
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='HTTP_STATUS:200\n{"object":"list","data":[]}',
            stderr="",
        )
        with patch("openclaw_bench.providers.probes.subprocess.run", return_value=completed) as run:
            result = IncusExecProbe("oc-stack").http_get(
                "http://10.68.198.1:8000/v1/models", timeout_s=2.0
            )

        cmd = run.call_args.args[0]
        self.assertEqual(cmd[0:4], ["incus", "exec", "oc-stack", "--"])
        self.assertIn("curl", cmd)
        self.assertIn("--max-time", cmd)
        self.assertIn("2", cmd)
        self.assertIn("http://10.68.198.1:8000/v1/models", cmd)
        self.assertTrue(result.ok)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.probe_name, "incus:oc-stack")

    def test_http_get_returns_failure_when_curl_fails(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=7, stdout="", stderr="connection refused"
        )
        with patch("openclaw_bench.providers.probes.subprocess.run", return_value=completed):
            result = IncusExecProbe("oc-stack").http_get(
                "http://10.68.198.1:8000/v1/models", timeout_s=2.0
            )
        self.assertFalse(result.ok)
        self.assertIsNone(result.status_code)
        self.assertIn("connection refused", result.error or "")


class DockerExecProbeTests(unittest.TestCase):
    def test_http_get_shells_out_to_docker_exec_curl(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout='HTTP_STATUS:200\n{"data":[]}', stderr=""
        )
        with patch("openclaw_bench.providers.probes.subprocess.run", return_value=completed) as run:
            DockerExecProbe("oc").http_get("http://172.17.0.1:11434/api/tags", timeout_s=1.5)
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[0:4], ["docker", "exec", "oc", "curl"])


class SSHProbeTests(unittest.TestCase):
    def test_http_get_shells_out_via_ssh(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout='HTTP_STATUS:200\n{"data":[]}', stderr=""
        )
        with patch("openclaw_bench.providers.probes.subprocess.run", return_value=completed) as run:
            SSHProbe("ubuntu@oc-host").http_get(
                "http://127.0.0.1:8000/v1/models", timeout_s=3.0
            )
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[0], "ssh")
        self.assertIn("ubuntu@oc-host", cmd)
        self.assertIn("curl", " ".join(cmd))
```

- [ ] **Step 2: Run tests, verify fail**

```bash
python3 -m unittest tests.test_providers_probes -v
```

Expected: `ImportError: cannot import name 'IncusExecProbe'`.

- [ ] **Step 3: Implement the three subprocess probes**

Append to `openclaw_bench/providers/probes.py`:

```python
import subprocess


def _curl_args(url: str, timeout_s: float) -> list[str]:
    return [
        "curl",
        "--silent",
        "--show-error",
        "--max-time",
        str(int(max(1, round(timeout_s)))),
        "--write-out",
        "HTTP_STATUS:%{http_code}\\n",
        url,
    ]


def _parse_curl_stdout(stdout: str) -> tuple[int | None, str]:
    status: int | None = None
    body_lines: list[str] = []
    for line in stdout.splitlines():
        if line.startswith("HTTP_STATUS:"):
            try:
                status = int(line.removeprefix("HTTP_STATUS:").strip())
            except ValueError:
                status = None
            continue
        body_lines.append(line)
    return status, "\n".join(body_lines)


def _result_from_curl(name: str, completed: subprocess.CompletedProcess) -> ProbeResult:
    if completed.returncode != 0:
        return ProbeResult(
            ok=False,
            status_code=None,
            body="",
            probe_name=name,
            error=(completed.stderr or completed.stdout or f"curl exit {completed.returncode}").strip(),
        )
    status, body = _parse_curl_stdout(completed.stdout)
    ok = status is not None and 200 <= status < 400
    return ProbeResult(
        ok=ok,
        status_code=status,
        body=body,
        probe_name=name,
        error=None if ok else f"http_{status}" if status else "missing_status_line",
    )


class IncusExecProbe:
    def __init__(self, instance: str) -> None:
        self.instance = instance
        self.name = f"incus:{instance}"

    def http_get(self, url: str, *, timeout_s: float) -> ProbeResult:
        cmd = ["incus", "exec", self.instance, "--"] + _curl_args(url, timeout_s)
        return self._run(cmd, timeout_s)

    def _run(self, cmd: list[str], timeout_s: float) -> ProbeResult:
        try:
            completed = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s + 5
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return ProbeResult(False, None, "", self.name, str(exc))
        return _result_from_curl(self.name, completed)


class DockerExecProbe:
    def __init__(self, container: str) -> None:
        self.container = container
        self.name = f"docker:{container}"

    def http_get(self, url: str, *, timeout_s: float) -> ProbeResult:
        cmd = ["docker", "exec", self.container] + _curl_args(url, timeout_s)
        try:
            completed = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s + 5
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return ProbeResult(False, None, "", self.name, str(exc))
        return _result_from_curl(self.name, completed)


class SSHProbe:
    def __init__(self, target: str) -> None:
        self.target = target
        self.name = f"ssh:{target}"

    def http_get(self, url: str, *, timeout_s: float) -> ProbeResult:
        cmd = ["ssh", "-o", "BatchMode=yes", self.target] + _curl_args(url, timeout_s)
        try:
            completed = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s + 5
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return ProbeResult(False, None, "", self.name, str(exc))
        return _result_from_curl(self.name, completed)
```

Update `openclaw_bench/providers/__init__.py`:

```python
from .probes import (
    DockerExecProbe,
    IncusExecProbe,
    LocalProbe,
    Probe,
    ProbeResult,
    SSHProbe,
)

__all__ = [
    "DockerExecProbe",
    "IncusExecProbe",
    "LocalProbe",
    "Probe",
    "ProbeResult",
    "SSHProbe",
]
```

- [ ] **Step 4: Run tests, verify pass**

```bash
python3 -m unittest tests.test_providers_probes -v
```

Expected: `OK` with 5 tests.

- [ ] **Step 5: Commit**

```bash
git add openclaw_bench/providers/__init__.py openclaw_bench/providers/probes.py tests/test_providers_probes.py
git commit -m "$(cat <<'EOF'
Add Incus/Docker/SSH probes for split OpenClaw runtimes

Subprocess-backed probes shell out via `incus exec`, `docker exec`, or
`ssh` so detection runs from the network namespace OpenClaw will route
from. Bench profiles configured for containerized OC pick these up
automatically; SSH stays opt-in via --oc-runtime.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Provider response fixtures (data only)

**Files:**
- Create: `tests/fixtures/provider_responses/vllm_v1_models.json`
- Create: `tests/fixtures/provider_responses/ollama_api_tags.json`
- Create: `tests/fixtures/provider_responses/llamacpp_v1_models.json`
- Create: `tests/fixtures/provider_responses/lmstudio_v1_models.json`

These are realistic shapes derived from each provider's documented `/v1/models` (or `/api/tags`) response. Used by detection tests; per-provider modules parse them.

- [ ] **Step 1: Write the vLLM fixture**

```json
// tests/fixtures/provider_responses/vllm_v1_models.json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-oss-20b",
      "object": "model",
      "created": 1714600000,
      "owned_by": "vllm",
      "max_model_len": 131072,
      "permission": [],
      "root": "gpt-oss-20b",
      "parent": null
    }
  ]
}
```

- [ ] **Step 2: Write the Ollama fixture**

```json
// tests/fixtures/provider_responses/ollama_api_tags.json
{
  "models": [
    {
      "name": "llama3.1:8b",
      "modified_at": "2026-04-15T10:00:00Z",
      "size": 4920753740,
      "digest": "sha256:abc",
      "details": {
        "format": "gguf",
        "family": "llama",
        "parameter_size": "8B",
        "quantization_level": "Q4_K_M"
      }
    },
    {
      "name": "qwen3:8b",
      "modified_at": "2026-04-20T09:00:00Z",
      "size": 5100000000,
      "digest": "sha256:def",
      "details": {
        "format": "gguf",
        "family": "qwen",
        "parameter_size": "8B",
        "quantization_level": "Q4_K_M"
      }
    }
  ]
}
```

- [ ] **Step 3: Write the llama.cpp fixture**

```json
// tests/fixtures/provider_responses/llamacpp_v1_models.json
{
  "object": "list",
  "data": [
    {
      "id": "default",
      "object": "model",
      "created": 1714600000,
      "owned_by": "llamacpp"
    }
  ]
}
```

- [ ] **Step 4: Write the LM Studio fixture**

```json
// tests/fixtures/provider_responses/lmstudio_v1_models.json
{
  "object": "list",
  "data": [
    {
      "id": "qwen2.5-7b-instruct",
      "object": "model",
      "created": 1714600000,
      "owned_by": "organization-owner"
    }
  ]
}
```

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/provider_responses/
git commit -m "$(cat <<'EOF'
Add captured provider response fixtures for detection tests

Realistic /v1/models and /api/tags payloads for vLLM, Ollama,
llama.cpp, and LM Studio. Fixtures back the detection-cascade unit
tests so we don't depend on a live runtime in CI.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: ProviderCandidate + DetectionReport types

**Files:**
- Create: `openclaw_bench/providers/detect.py`
- Modify: `openclaw_bench/providers/__init__.py`
- Test: `tests/test_providers_detect.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_providers_detect.py
import unittest

from openclaw_bench.providers.detect import DetectionReport, ProviderCandidate


class ProviderCandidateTests(unittest.TestCase):
    def test_candidate_carries_provider_url_and_models(self):
        candidate = ProviderCandidate(
            provider="vllm",
            base_url="http://10.68.198.1:8000/v1",
            models=["gpt-oss-20b"],
            probe_results={},
            source="port_probe",
        )
        self.assertEqual(candidate.provider, "vllm")
        self.assertEqual(candidate.base_url, "http://10.68.198.1:8000/v1")
        self.assertEqual(candidate.models, ["gpt-oss-20b"])
        self.assertEqual(candidate.source, "port_probe")


class DetectionReportTests(unittest.TestCase):
    def test_report_is_immutable_and_holds_candidates_and_findings(self):
        report = DetectionReport(candidates=(), findings=())
        self.assertEqual(report.candidates, ())
        self.assertEqual(report.findings, ())
```

- [ ] **Step 2: Run tests, verify fail**

```bash
python3 -m unittest tests.test_providers_detect -v
```

Expected: `ModuleNotFoundError: No module named 'openclaw_bench.providers.detect'`.

- [ ] **Step 3: Implement the dataclasses**

```python
# openclaw_bench/providers/detect.py
from __future__ import annotations

from dataclasses import dataclass, field

from .probes import ProbeResult


@dataclass(frozen=True)
class ProviderCandidate:
    provider: str
    base_url: str
    models: list[str]
    probe_results: dict[str, ProbeResult]
    source: str  # "already_known" | "port_probe"


@dataclass(frozen=True)
class DetectionReport:
    candidates: tuple[ProviderCandidate, ...] = field(default_factory=tuple)
    findings: tuple[str, ...] = field(default_factory=tuple)
```

Update `openclaw_bench/providers/__init__.py`:

```python
from .detect import DetectionReport, ProviderCandidate
from .probes import (
    DockerExecProbe,
    IncusExecProbe,
    LocalProbe,
    Probe,
    ProbeResult,
    SSHProbe,
)

__all__ = [
    "DetectionReport",
    "DockerExecProbe",
    "IncusExecProbe",
    "LocalProbe",
    "Probe",
    "ProbeResult",
    "ProviderCandidate",
    "SSHProbe",
]
```

- [ ] **Step 4: Run tests, verify pass**

```bash
python3 -m unittest tests.test_providers_detect -v
```

Expected: `OK` with 2 tests.

- [ ] **Step 5: Commit**

```bash
git add openclaw_bench/providers/detect.py openclaw_bench/providers/__init__.py tests/test_providers_detect.py
git commit -m "$(cat <<'EOF'
Add ProviderCandidate and DetectionReport types

Immutable dataclasses for the detection cascade output. Each candidate
records its provider, base URL, served model ids, and per-probe
results so later steps can surface mismatches between host and OC
runtime.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Already-configured scan from existing OC profiles

**Files:**
- Modify: `openclaw_bench/providers/detect.py`
- Test: `tests/test_providers_detect.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_providers_detect.py`:

```python
import json
import tempfile
from pathlib import Path

from openclaw_bench.providers.detect import scan_existing_oc_profiles


class ScanExistingProfilesTests(unittest.TestCase):
    def test_scan_returns_configured_providers_from_profile_jsons(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            profile_dir = home / ".openclaw-bench"
            profile_dir.mkdir()
            (profile_dir / "openclaw.json").write_text(
                json.dumps(
                    {
                        "models": {
                            "providers": {
                                "vllm": {
                                    "baseUrl": "http://10.68.198.1:8000/v1",
                                    "api": "openai-completions",
                                    "models": [{"id": "gpt-oss-20b"}],
                                }
                            }
                        }
                    }
                )
            )
            entries = scan_existing_oc_profiles(home)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].provider, "vllm")
        self.assertEqual(entries[0].base_url, "http://10.68.198.1:8000/v1")
        self.assertEqual(entries[0].models, ["gpt-oss-20b"])
        self.assertEqual(entries[0].source, "already_known")

    def test_scan_skips_directories_without_openclaw_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / ".openclaw-empty").mkdir()
            entries = scan_existing_oc_profiles(home)
        self.assertEqual(entries, [])

    def test_scan_handles_unknown_provider_keys_gracefully(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            profile_dir = home / ".openclaw-bench"
            profile_dir.mkdir()
            (profile_dir / "openclaw.json").write_text(
                json.dumps({"models": {"providers": {"made-up": {"baseUrl": "x"}}}})
            )
            entries = scan_existing_oc_profiles(home)
        self.assertEqual(entries, [])
```

- [ ] **Step 2: Run tests, verify fail**

```bash
python3 -m unittest tests.test_providers_detect -v
```

Expected: `ImportError: cannot import name 'scan_existing_oc_profiles'`.

- [ ] **Step 3: Implement the scan**

Append to `openclaw_bench/providers/detect.py`:

```python
import json
from pathlib import Path

KNOWN_PROVIDERS: tuple[str, ...] = ("vllm", "ollama", "llamacpp", "lmstudio")


def scan_existing_oc_profiles(home: Path) -> list[ProviderCandidate]:
    home = Path(home).expanduser()
    candidates: list[ProviderCandidate] = []
    for profile_dir in sorted(home.glob(".openclaw*")):
        config_path = profile_dir / "openclaw.json"
        if not config_path.is_file():
            continue
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        providers = (((payload or {}).get("models") or {}).get("providers") or {})
        for key, block in providers.items():
            if key not in KNOWN_PROVIDERS:
                continue
            base_url = (block or {}).get("baseUrl")
            if not isinstance(base_url, str):
                continue
            model_ids: list[str] = []
            for entry in (block or {}).get("models") or []:
                model_id = (entry or {}).get("id") if isinstance(entry, dict) else None
                if isinstance(model_id, str):
                    model_ids.append(model_id)
            candidates.append(
                ProviderCandidate(
                    provider=key,
                    base_url=base_url,
                    models=model_ids,
                    probe_results={},
                    source="already_known",
                )
            )
    return candidates
```

- [ ] **Step 4: Run tests, verify pass**

```bash
python3 -m unittest tests.test_providers_detect -v
```

Expected: `OK` with 5 tests.

- [ ] **Step 5: Commit**

```bash
git add openclaw_bench/providers/detect.py tests/test_providers_detect.py
git commit -m "$(cat <<'EOF'
Add cheap-first-pass scan of existing OpenClaw profile configs

Reads ~/.openclaw-*/openclaw.json files for already-configured
providers before any port probes. Free signal: if the user already has
OC routing to a runtime, detection re-uses the configured base URL
instead of rediscovering it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Port probe with 30s/provider budget

**Files:**
- Modify: `openclaw_bench/providers/detect.py`
- Test: `tests/test_providers_detect.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_providers_detect.py`:

```python
from unittest.mock import MagicMock

from openclaw_bench.providers.detect import port_probe_provider
from openclaw_bench.providers.probes import ProbeResult


def _ok(body: str, name: str = "host") -> ProbeResult:
    return ProbeResult(ok=True, status_code=200, body=body, probe_name=name, error=None)


def _fail(name: str = "host") -> ProbeResult:
    return ProbeResult(
        ok=False, status_code=None, body="", probe_name=name, error="connection refused"
    )


VLLM_BODY = '{"object":"list","data":[{"id":"gpt-oss-20b","object":"model"}]}'
OLLAMA_BODY = '{"models":[{"name":"llama3.1:8b"}]}'


class PortProbeProviderTests(unittest.TestCase):
    def test_vllm_probe_returns_candidate_for_first_responding_port(self):
        probe = MagicMock()
        probe.name = "host"
        probe.http_get.side_effect = [_fail(), _ok(VLLM_BODY)]
        candidate = port_probe_provider("vllm", [probe], total_timeout_s=30.0)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.provider, "vllm")
        self.assertEqual(candidate.models, ["gpt-oss-20b"])
        self.assertEqual(candidate.source, "port_probe")

    def test_ollama_probe_uses_api_tags_endpoint(self):
        probe = MagicMock()
        probe.name = "host"
        probe.http_get.return_value = _ok(OLLAMA_BODY)
        candidate = port_probe_provider("ollama", [probe], total_timeout_s=30.0)
        url = probe.http_get.call_args.args[0]
        self.assertIn("/api/tags", url)
        self.assertIsNotNone(candidate)
        self.assertIn("llama3.1:8b", candidate.models)

    def test_returns_none_when_all_ports_fail(self):
        probe = MagicMock()
        probe.name = "host"
        probe.http_get.return_value = _fail()
        candidate = port_probe_provider("vllm", [probe], total_timeout_s=30.0)
        self.assertIsNone(candidate)

    def test_total_budget_caps_probe_count(self):
        # Each probe takes 6s of fake budget; with 12s total we should call at most twice.
        probe = MagicMock()
        probe.name = "host"

        def slow(*_args, **_kwargs):
            return _fail()

        probe.http_get.side_effect = slow
        port_probe_provider(
            "vllm", [probe], total_timeout_s=12.0, per_probe_timeout_s=6.0
        )
        self.assertLessEqual(probe.http_get.call_count, 2)
```

- [ ] **Step 2: Run tests, verify fail**

```bash
python3 -m unittest tests.test_providers_detect -v
```

Expected: `ImportError: cannot import name 'port_probe_provider'`.

- [ ] **Step 3: Implement the per-provider port probe**

Append to `openclaw_bench/providers/detect.py`:

```python
import time

PROVIDER_ENDPOINTS: dict[str, tuple[tuple[int, str], ...]] = {
    "vllm": (
        (8000, "/v1/models"),
        (8001, "/v1/models"),
        (8002, "/v1/models"),
        (8003, "/v1/models"),
        (8080, "/v1/models"),
    ),
    "llamacpp": (
        (8080, "/v1/models"),
        (8000, "/v1/models"),
    ),
    "ollama": (
        (11434, "/api/tags"),
    ),
    "lmstudio": (
        (1234, "/v1/models"),
    ),
}


def _parse_models_from_body(provider: str, body: str) -> list[str]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return []
    if provider == "ollama":
        models = payload.get("models") or []
        return [m.get("name") for m in models if isinstance(m, dict) and isinstance(m.get("name"), str)]
    data = payload.get("data") or []
    return [m.get("id") for m in data if isinstance(m, dict) and isinstance(m.get("id"), str)]


def port_probe_provider(
    provider: str,
    probes: list,
    *,
    total_timeout_s: float = 30.0,
    per_probe_timeout_s: float = 5.0,
) -> ProviderCandidate | None:
    endpoints = PROVIDER_ENDPOINTS.get(provider, ())
    if not endpoints:
        return None
    deadline = time.monotonic() + total_timeout_s
    for port, path in endpoints:
        if time.monotonic() >= deadline:
            break
        url = f"http://127.0.0.1:{port}{path}"
        primary = probes[0]
        result = primary.http_get(url, timeout_s=per_probe_timeout_s)
        if not result.ok:
            continue
        models = _parse_models_from_body(provider, result.body)
        probe_results = {primary.name: result}
        for extra in probes[1:]:
            extra_result = extra.http_get(url, timeout_s=per_probe_timeout_s)
            probe_results[extra.name] = extra_result
        base_url = url[: -len(path)] + ("/v1" if path.startswith("/v1") else "")
        return ProviderCandidate(
            provider=provider,
            base_url=base_url,
            models=models,
            probe_results=probe_results,
            source="port_probe",
        )
    return None
```

- [ ] **Step 4: Run tests, verify pass**

```bash
python3 -m unittest tests.test_providers_detect -v
```

Expected: `OK` with 9 tests.

- [ ] **Step 5: Commit**

```bash
git add openclaw_bench/providers/detect.py tests/test_providers_detect.py
git commit -m "$(cat <<'EOF'
Add per-provider port probe with 30s budget

Walks each provider's well-known endpoints (vLLM/llama.cpp/LM Studio
on /v1/models, Ollama on /api/tags) until one responds or the 30s
total budget per provider is exhausted. Hangs cap at the per-probe
timeout so a stuck port can't burn the budget.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Detection cascade orchestration

**Files:**
- Modify: `openclaw_bench/providers/detect.py`
- Modify: `openclaw_bench/providers/__init__.py`
- Test: `tests/test_providers_detect.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_providers_detect.py`:

```python
from openclaw_bench.providers.detect import run_detection


class RunDetectionTests(unittest.TestCase):
    def test_already_known_short_circuits_port_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            profile_dir = home / ".openclaw-bench"
            profile_dir.mkdir()
            (profile_dir / "openclaw.json").write_text(
                json.dumps(
                    {
                        "models": {
                            "providers": {
                                "vllm": {
                                    "baseUrl": "http://10.68.198.1:8000/v1",
                                    "models": [{"id": "gpt-oss-20b"}],
                                }
                            }
                        }
                    }
                )
            )
            probe = MagicMock()
            probe.name = "host"
            probe.http_get.return_value = _ok(VLLM_BODY)
            report = run_detection(
                providers=["vllm"], probes=[probe], home=home
            )
        self.assertEqual(len(report.candidates), 1)
        self.assertEqual(report.candidates[0].source, "already_known")
        probe.http_get.assert_not_called()

    def test_port_probe_runs_when_nothing_already_known(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            probe = MagicMock()
            probe.name = "host"
            probe.http_get.return_value = _ok(VLLM_BODY)
            report = run_detection(
                providers=["vllm"], probes=[probe], home=home
            )
        self.assertEqual(len(report.candidates), 1)
        self.assertEqual(report.candidates[0].source, "port_probe")
        self.assertEqual(report.candidates[0].models, ["gpt-oss-20b"])

    def test_zero_candidates_yields_empty_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            probe = MagicMock()
            probe.name = "host"
            probe.http_get.return_value = _fail()
            report = run_detection(
                providers=["vllm", "ollama"], probes=[probe], home=home
            )
        self.assertEqual(report.candidates, ())

    def test_host_runtime_mismatch_emits_finding(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            host = MagicMock()
            host.name = "host"
            host.http_get.return_value = _ok(VLLM_BODY, name="host")
            runtime = MagicMock()
            runtime.name = "incus:oc-stack"
            runtime.http_get.return_value = _fail(name="incus:oc-stack")
            report = run_detection(
                providers=["vllm"], probes=[host, runtime], home=home
            )
        self.assertTrue(
            any(f.startswith("reachable_from_host_not_runtime") for f in report.findings),
            msg=f"findings={report.findings}",
        )
```

- [ ] **Step 2: Run tests, verify fail**

```bash
python3 -m unittest tests.test_providers_detect -v
```

Expected: `ImportError: cannot import name 'run_detection'`.

- [ ] **Step 3: Implement the cascade**

Append to `openclaw_bench/providers/detect.py`:

```python
def run_detection(
    *,
    providers: list[str],
    probes: list,
    home: Path,
    per_provider_timeout_s: float = 30.0,
) -> DetectionReport:
    candidates: list[ProviderCandidate] = []
    findings: list[str] = []

    already_known = scan_existing_oc_profiles(home)
    known_by_provider = {c.provider: c for c in already_known}

    for provider in providers:
        if provider in known_by_provider:
            candidates.append(known_by_provider[provider])
            continue
        candidate = port_probe_provider(
            provider, probes, total_timeout_s=per_provider_timeout_s
        )
        if candidate is None:
            continue
        candidates.append(candidate)
        for probe_name, result in candidate.probe_results.items():
            if probe_name == probes[0].name:
                continue
            primary_ok = candidate.probe_results[probes[0].name].ok
            if primary_ok and not result.ok:
                findings.append(
                    f"reachable_from_host_not_runtime:{provider}@{candidate.base_url} "
                    f"(probe={probe_name})"
                )

    return DetectionReport(candidates=tuple(candidates), findings=tuple(findings))
```

Update `openclaw_bench/providers/__init__.py`:

```python
from .detect import (
    DetectionReport,
    ProviderCandidate,
    port_probe_provider,
    run_detection,
    scan_existing_oc_profiles,
)
from .probes import (
    DockerExecProbe,
    IncusExecProbe,
    LocalProbe,
    Probe,
    ProbeResult,
    SSHProbe,
)

__all__ = [
    "DetectionReport",
    "DockerExecProbe",
    "IncusExecProbe",
    "LocalProbe",
    "Probe",
    "ProbeResult",
    "ProviderCandidate",
    "SSHProbe",
    "port_probe_provider",
    "run_detection",
    "scan_existing_oc_profiles",
]
```

- [ ] **Step 4: Run tests, verify pass**

```bash
python3 -m unittest tests.test_providers_detect -v
```

Expected: `OK` with 13 tests.

- [ ] **Step 5: Commit**

```bash
git add openclaw_bench/providers/detect.py openclaw_bench/providers/__init__.py tests/test_providers_detect.py
git commit -m "$(cat <<'EOF'
Add detection cascade with host-vs-runtime mismatch findings

Cascade scans existing OC profiles first, falls back to port probes,
and surfaces a reachable_from_host_not_runtime finding when the
primary probe sees a candidate but a secondary (runtime-side) probe
cannot reach it. That is exactly the GPT-OSS UFW failure shape.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Runtime auto-derive from OC profile

**Files:**
- Modify: `openclaw_bench/providers/detect.py`
- Modify: `openclaw_bench/providers/__init__.py`
- Test: `tests/test_providers_detect.py`

OpenClaw `2026.4.27` profile JSON does not always name a runtime in a parseable field — that risk is recorded in the spec. Derivation reads the bench profile's `gateway.runtime` field if present (forward-compatible), and otherwise checks for repo-local conventions like a `container` hint in the bench's quickstart metadata.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_providers_detect.py`:

```python
from openclaw_bench.providers.detect import derive_probes_for_profile
from openclaw_bench.providers.probes import IncusExecProbe, LocalProbe


class DeriveProbesForProfileTests(unittest.TestCase):
    def test_returns_local_probe_only_for_native_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            profile_dir = home / ".openclaw-bench"
            profile_dir.mkdir()
            (profile_dir / "openclaw.json").write_text(json.dumps({"gateway": {"mode": "local"}}))
            probes = derive_probes_for_profile("bench", home=home)
        self.assertEqual(len(probes), 1)
        self.assertIsInstance(probes[0], LocalProbe)

    def test_returns_local_plus_incus_when_runtime_field_names_incus(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            profile_dir = home / ".openclaw-bench"
            profile_dir.mkdir()
            (profile_dir / "openclaw.json").write_text(
                json.dumps({"gateway": {"runtime": {"kind": "incus", "instance": "oc-stack"}}})
            )
            probes = derive_probes_for_profile("bench", home=home)
        self.assertEqual(len(probes), 2)
        self.assertIsInstance(probes[0], LocalProbe)
        self.assertIsInstance(probes[1], IncusExecProbe)
        self.assertEqual(probes[1].instance, "oc-stack")

    def test_explicit_oc_runtime_overrides_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            probes = derive_probes_for_profile(
                "bench", home=home, oc_runtime_override="incus:oc-stack"
            )
        self.assertEqual(len(probes), 2)
        self.assertIsInstance(probes[1], IncusExecProbe)
        self.assertEqual(probes[1].instance, "oc-stack")

    def test_unknown_override_kind_raises_clear_error(self):
        with self.assertRaises(ValueError) as cm:
            derive_probes_for_profile("bench", home=Path("/tmp"), oc_runtime_override="weird")
        self.assertIn("--oc-runtime", str(cm.exception))
```

- [ ] **Step 2: Run tests, verify fail**

```bash
python3 -m unittest tests.test_providers_detect -v
```

Expected: `ImportError: cannot import name 'derive_probes_for_profile'`.

- [ ] **Step 3: Implement the auto-derive**

Append to `openclaw_bench/providers/detect.py`:

```python
from .probes import DockerExecProbe, IncusExecProbe, LocalProbe, Probe, SSHProbe


def derive_probes_for_profile(
    profile: str,
    *,
    home: Path,
    oc_runtime_override: str | None = None,
) -> list[Probe]:
    probes: list[Probe] = [LocalProbe()]
    if oc_runtime_override:
        probes.append(_probe_from_override(oc_runtime_override))
        return probes
    config_path = Path(home).expanduser() / f".openclaw-{profile}" / "openclaw.json"
    if config_path.is_file():
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        runtime = (((payload or {}).get("gateway") or {}).get("runtime") or {})
        kind = runtime.get("kind") if isinstance(runtime, dict) else None
        if kind == "incus":
            instance = runtime.get("instance")
            if isinstance(instance, str):
                probes.append(IncusExecProbe(instance))
        elif kind == "docker":
            container = runtime.get("container")
            if isinstance(container, str):
                probes.append(DockerExecProbe(container))
    return probes


def _probe_from_override(spec: str) -> Probe:
    if ":" not in spec:
        raise ValueError(
            f"--oc-runtime expects 'kind:target' (incus:<instance>, docker:<container>, ssh:<user@host>); got '{spec}'"
        )
    kind, _, target = spec.partition(":")
    kind = kind.strip().lower()
    target = target.strip()
    if kind == "incus":
        return IncusExecProbe(target)
    if kind == "docker":
        return DockerExecProbe(target)
    if kind == "ssh":
        return SSHProbe(target)
    raise ValueError(
        f"--oc-runtime kind '{kind}' not supported; use incus|docker|ssh"
    )
```

Update `openclaw_bench/providers/__init__.py` exports to include `derive_probes_for_profile`:

```python
from .detect import (
    DetectionReport,
    ProviderCandidate,
    derive_probes_for_profile,
    port_probe_provider,
    run_detection,
    scan_existing_oc_profiles,
)
```

And add `"derive_probes_for_profile"` to `__all__`.

- [ ] **Step 4: Run tests, verify pass**

```bash
python3 -m unittest tests.test_providers_detect -v
```

Expected: `OK` with 17 tests.

- [ ] **Step 5: Commit**

```bash
git add openclaw_bench/providers/detect.py openclaw_bench/providers/__init__.py tests/test_providers_detect.py
git commit -m "$(cat <<'EOF'
Add runtime auto-derive from OpenClaw profile config

Reads gateway.runtime from ~/.openclaw-<profile>/openclaw.json to
auto-pick Incus or Docker probes when the user has a containerized
OC, with --oc-runtime override for SSH or future runtimes. Native
installs get a single LocalProbe and skip the doubled traffic.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: vLLM provider module

**Files:**
- Create: `openclaw_bench/providers/vllm.py`
- Modify: `openclaw_bench/providers/__init__.py`
- Test: `tests/test_providers_vllm.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_providers_vllm.py
import json
import unittest
from pathlib import Path

from openclaw_bench.providers import ProviderCandidate
from openclaw_bench.providers.vllm import (
    detect,
    generate_route_config,
    parameter_shaping,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "provider_responses"


def _candidate(model_id: str = "gpt-oss-20b", base_url: str = "http://10.68.198.1:8000/v1") -> ProviderCandidate:
    return ProviderCandidate(
        provider="vllm",
        base_url=base_url,
        models=[model_id],
        probe_results={},
        source="port_probe",
    )


class VllmDetectTests(unittest.TestCase):
    def test_detect_parses_model_ids_from_v1_models_response(self):
        body = (FIXTURES / "vllm_v1_models.json").read_text()
        models = detect(body)
        self.assertEqual(models, ["gpt-oss-20b"])

    def test_detect_returns_empty_for_invalid_body(self):
        self.assertEqual(detect("not json"), [])


class VllmGenerateRouteConfigTests(unittest.TestCase):
    def test_generated_config_matches_quickstart_helper(self):
        from openclaw_bench.quickstart import VllmEndpoint, _vllm_provider_config

        endpoint = VllmEndpoint(
            base_url="http://10.68.198.1:8000/v1",
            model="gpt-oss-20b",
            context=131072,
            max_tokens=512,
        )
        expected = _vllm_provider_config(endpoint)
        actual = generate_route_config(_candidate(), context=131072, max_tokens=512)
        self.assertEqual(actual, expected)


class VllmParameterShapingTests(unittest.TestCase):
    def test_qwen_model_disables_thinking(self):
        params = parameter_shaping(_candidate(model_id="qwen3.5-4b"))
        self.assertFalse(params["chatTemplateKwargs"]["enable_thinking"])
        self.assertNotIn("extra_body", params)

    def test_gpt_oss_model_sets_reasoning_effort_low(self):
        params = parameter_shaping(_candidate(model_id="gpt-oss-20b"))
        self.assertEqual(params["extra_body"]["reasoning_effort"], "low")
        self.assertFalse(params["chatTemplateKwargs"]["enable_thinking"])
```

- [ ] **Step 2: Run tests, verify fail**

```bash
python3 -m unittest tests.test_providers_vllm -v
```

Expected: `ModuleNotFoundError: No module named 'openclaw_bench.providers.vllm'`.

- [ ] **Step 3: Implement the vLLM module**

```python
# openclaw_bench/providers/vllm.py
from __future__ import annotations

import json

from ..quickstart import VllmEndpoint, _vllm_provider_config
from .detect import ProviderCandidate


def detect(body: str) -> list[str]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return []
    data = payload.get("data") or []
    return [
        entry.get("id")
        for entry in data
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    ]


def generate_route_config(
    candidate: ProviderCandidate,
    *,
    context: int,
    max_tokens: int,
) -> dict:
    if not candidate.models:
        raise ValueError("vllm candidate has no models; cannot generate route config")
    endpoint = VllmEndpoint(
        base_url=candidate.base_url,
        model=candidate.models[0],
        context=context,
        max_tokens=max_tokens,
    )
    return _vllm_provider_config(endpoint)


def parameter_shaping(candidate: ProviderCandidate) -> dict:
    if not candidate.models:
        raise ValueError("vllm candidate has no models; cannot shape parameters")
    model_id = candidate.models[0]
    params: dict = {"chatTemplateKwargs": {"enable_thinking": False}}
    if model_id.startswith("gpt-oss"):
        params["extra_body"] = {"reasoning_effort": "low"}
    return params
```

Update `openclaw_bench/providers/__init__.py` to export the module surface:

```python
from . import vllm  # noqa: F401
```

Append `"vllm"` to `__all__`.

- [ ] **Step 4: Run tests, verify pass**

```bash
python3 -m unittest tests.test_providers_vllm -v
```

Expected: `OK` with 5 tests.

- [ ] **Step 5: Commit**

```bash
git add openclaw_bench/providers/vllm.py openclaw_bench/providers/__init__.py tests/test_providers_vllm.py
git commit -m "$(cat <<'EOF'
Add vLLM provider module: detect, generate, parameter shaping

Reuses quickstart._vllm_provider_config so generated route configs
inherit the 16k context floor, meta fields, and plugin entries that
OpenClaw 2026.4.27 requires. Parameter shaping encodes the two
known runtime gotchas: chatTemplateKwargs.enable_thinking=false for
Qwen-class models and extra_body.reasoning_effort='low' for GPT-OSS.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Ollama detect-only stub

**Files:**
- Create: `openclaw_bench/providers/ollama.py`
- Modify: `openclaw_bench/providers/__init__.py`
- Test: `tests/test_providers_ollama.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_providers_ollama.py
import unittest
from pathlib import Path

from openclaw_bench.providers import ProviderCandidate
from openclaw_bench.providers.ollama import detect, generate_route_config


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "provider_responses"


class OllamaDetectTests(unittest.TestCase):
    def test_detect_parses_model_names_from_api_tags(self):
        body = (FIXTURES / "ollama_api_tags.json").read_text()
        models = detect(body)
        self.assertIn("llama3.1:8b", models)
        self.assertIn("qwen3:8b", models)


class OllamaGenerateRouteConfigTests(unittest.TestCase):
    def test_generate_route_config_raises_not_implemented(self):
        candidate = ProviderCandidate(
            provider="ollama",
            base_url="http://127.0.0.1:11434",
            models=["llama3.1:8b"],
            probe_results={},
            source="port_probe",
        )
        with self.assertRaises(NotImplementedError) as cm:
            generate_route_config(candidate, context=8192, max_tokens=512)
        self.assertIn("ollama", str(cm.exception).lower())
```

- [ ] **Step 2: Run tests, verify fail**

```bash
python3 -m unittest tests.test_providers_ollama -v
```

Expected: `ModuleNotFoundError: No module named 'openclaw_bench.providers.ollama'`.

- [ ] **Step 3: Implement the Ollama detect-only stub**

```python
# openclaw_bench/providers/ollama.py
from __future__ import annotations

import json

from .detect import ProviderCandidate


def detect(body: str) -> list[str]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return []
    models = payload.get("models") or []
    return [
        entry.get("name")
        for entry in models
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    ]


def generate_route_config(
    candidate: ProviderCandidate,
    *,
    context: int,
    max_tokens: int,
) -> dict:
    raise NotImplementedError(
        "ollama provider config generation is not yet implemented; "
        "this slice ships detect-only. See "
        "https://docs.openclaw.ai/providers/ollama for the target shape."
    )
```

Update `openclaw_bench/providers/__init__.py` to expose the submodule:

```python
from . import ollama  # noqa: F401
```

And add `"ollama"` to `__all__`.

- [ ] **Step 4: Run tests, verify pass**

```bash
python3 -m unittest tests.test_providers_ollama -v
```

Expected: `OK` with 2 tests.

- [ ] **Step 5: Commit**

```bash
git add openclaw_bench/providers/ollama.py openclaw_bench/providers/__init__.py tests/test_providers_ollama.py
git commit -m "$(cat <<'EOF'
Add Ollama provider detect-only stub

Parses /api/tags responses; generate_route_config raises a clear
NotImplementedError pointing at the OpenClaw Ollama provider docs.
Generator implementation follows once we have a live Ollama instance
on the RTX Pro 5000 to validate against.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: llama.cpp detect-only stub

**Files:**
- Create: `openclaw_bench/providers/llamacpp.py`
- Modify: `openclaw_bench/providers/__init__.py`
- Test: `tests/test_providers_llamacpp.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_providers_llamacpp.py
import unittest
from pathlib import Path

from openclaw_bench.providers import ProviderCandidate
from openclaw_bench.providers.llamacpp import detect, generate_route_config


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "provider_responses"


class LlamaCppDetectTests(unittest.TestCase):
    def test_detect_parses_model_ids_from_v1_models(self):
        body = (FIXTURES / "llamacpp_v1_models.json").read_text()
        models = detect(body)
        self.assertEqual(models, ["default"])


class LlamaCppGenerateRouteConfigTests(unittest.TestCase):
    def test_generate_route_config_raises_not_implemented(self):
        candidate = ProviderCandidate(
            provider="llamacpp",
            base_url="http://127.0.0.1:8080/v1",
            models=["default"],
            probe_results={},
            source="port_probe",
        )
        with self.assertRaises(NotImplementedError) as cm:
            generate_route_config(candidate, context=8192, max_tokens=512)
        self.assertIn("llama", str(cm.exception).lower())
```

- [ ] **Step 2: Run tests, verify fail**

```bash
python3 -m unittest tests.test_providers_llamacpp -v
```

Expected: `ModuleNotFoundError: No module named 'openclaw_bench.providers.llamacpp'`.

- [ ] **Step 3: Implement the llama.cpp detect-only stub**

```python
# openclaw_bench/providers/llamacpp.py
from __future__ import annotations

import json

from .detect import ProviderCandidate


def detect(body: str) -> list[str]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return []
    data = payload.get("data") or []
    return [
        entry.get("id")
        for entry in data
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    ]


def generate_route_config(
    candidate: ProviderCandidate,
    *,
    context: int,
    max_tokens: int,
) -> dict:
    raise NotImplementedError(
        "llama.cpp provider config generation is not yet implemented; "
        "this slice ships detect-only. The generator should emit a custom "
        "openai-completions provider matching llama-server's /v1 surface."
    )
```

Update `openclaw_bench/providers/__init__.py`:

```python
from . import llamacpp  # noqa: F401
```

Add `"llamacpp"` to `__all__`.

- [ ] **Step 4: Run tests, verify pass**

```bash
python3 -m unittest tests.test_providers_llamacpp -v
```

Expected: `OK` with 2 tests.

- [ ] **Step 5: Commit**

```bash
git add openclaw_bench/providers/llamacpp.py openclaw_bench/providers/__init__.py tests/test_providers_llamacpp.py
git commit -m "$(cat <<'EOF'
Add llama.cpp provider detect-only stub

Parses /v1/models from llama-server; generator raises until we have a
running llama.cpp instance to validate the OpenAI-compatible config
shape against.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: LM Studio detect-only stub

**Files:**
- Create: `openclaw_bench/providers/lmstudio.py`
- Modify: `openclaw_bench/providers/__init__.py`
- Test: `tests/test_providers_lmstudio.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_providers_lmstudio.py
import unittest
from pathlib import Path

from openclaw_bench.providers import ProviderCandidate
from openclaw_bench.providers.lmstudio import detect, generate_route_config


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "provider_responses"


class LmStudioDetectTests(unittest.TestCase):
    def test_detect_parses_model_ids_from_v1_models(self):
        body = (FIXTURES / "lmstudio_v1_models.json").read_text()
        models = detect(body)
        self.assertEqual(models, ["qwen2.5-7b-instruct"])


class LmStudioGenerateRouteConfigTests(unittest.TestCase):
    def test_generate_route_config_raises_not_implemented(self):
        candidate = ProviderCandidate(
            provider="lmstudio",
            base_url="http://127.0.0.1:1234/v1",
            models=["qwen2.5-7b-instruct"],
            probe_results={},
            source="port_probe",
        )
        with self.assertRaises(NotImplementedError) as cm:
            generate_route_config(candidate, context=8192, max_tokens=512)
        self.assertIn("lmstudio", str(cm.exception).lower())
```

- [ ] **Step 2: Run tests, verify fail**

```bash
python3 -m unittest tests.test_providers_lmstudio -v
```

Expected: `ModuleNotFoundError: No module named 'openclaw_bench.providers.lmstudio'`.

- [ ] **Step 3: Implement the LM Studio detect-only stub**

```python
# openclaw_bench/providers/lmstudio.py
from __future__ import annotations

import json

from .detect import ProviderCandidate


def detect(body: str) -> list[str]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return []
    data = payload.get("data") or []
    return [
        entry.get("id")
        for entry in data
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    ]


def generate_route_config(
    candidate: ProviderCandidate,
    *,
    context: int,
    max_tokens: int,
) -> dict:
    raise NotImplementedError(
        "lmstudio provider config generation is not yet implemented; "
        "this slice ships detect-only. See "
        "https://docs.openclaw.ai/providers/lmstudio for the target shape."
    )
```

Update `openclaw_bench/providers/__init__.py`:

```python
from . import lmstudio  # noqa: F401
```

Add `"lmstudio"` to `__all__`.

- [ ] **Step 4: Run tests, verify pass**

```bash
python3 -m unittest tests.test_providers_lmstudio -v
```

Expected: `OK` with 2 tests.

- [ ] **Step 5: Commit**

```bash
git add openclaw_bench/providers/lmstudio.py openclaw_bench/providers/__init__.py tests/test_providers_lmstudio.py
git commit -m "$(cat <<'EOF'
Add LM Studio provider detect-only stub

Parses /v1/models from LM Studio's local server; generator raises until
we have a running LM Studio instance to validate against the OpenClaw
lmstudio provider config shape.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Public verification gates + `oc-bench preflight` CLI

**Files:**
- Modify: `openclaw_bench/preflight.py` (add `run_verification_gates` + `VerificationReport`)
- Modify: `openclaw_bench/cli.py` (add `preflight` subcommand)
- Test: `tests/test_providers_preflight.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_providers_preflight.py
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from openclaw_bench.cli import main as cli_main
from openclaw_bench.preflight import PreflightCheck, VerificationReport, run_verification_gates


class RunVerificationGatesTests(unittest.TestCase):
    def test_returns_pass_when_all_four_gates_pass(self):
        passes = [
            PreflightCheck("openclaw_profile_config", "pass", "ok"),
            PreflightCheck("openclaw_models_list", "pass", "ok"),
            PreflightCheck("provider_health", "pass", "ok"),
            PreflightCheck("openclaw_route_smoke", "pass", "ok"),
        ]
        with patch(
            "openclaw_bench.preflight._run_gate", side_effect=passes
        ):
            report = run_verification_gates(
                profile="benchclaw",
                provider="vllm",
                base_url="http://10.68.198.1:8000/v1",
                route_model="vllm/gpt-oss-20b",
            )
        self.assertIsInstance(report, VerificationReport)
        self.assertTrue(report.ok)
        self.assertEqual(len(report.checks), 4)

    def test_fails_loud_when_any_gate_fails(self):
        results = [
            PreflightCheck("openclaw_profile_config", "pass", "ok"),
            PreflightCheck("openclaw_models_list", "fail", "model not registered"),
            PreflightCheck("provider_health", "pass", "ok"),
            PreflightCheck("openclaw_route_smoke", "pass", "ok"),
        ]
        with patch("openclaw_bench.preflight._run_gate", side_effect=results):
            report = run_verification_gates(
                profile="benchclaw",
                provider="vllm",
                base_url="http://10.68.198.1:8000/v1",
                route_model="vllm/gpt-oss-20b",
            )
        self.assertFalse(report.ok)


class PreflightCliTests(unittest.TestCase):
    def test_preflight_command_exits_zero_on_pass(self):
        passes = VerificationReport(
            ok=True,
            checks=(
                PreflightCheck("openclaw_profile_config", "pass", "ok"),
                PreflightCheck("openclaw_models_list", "pass", "ok"),
                PreflightCheck("provider_health", "pass", "ok"),
                PreflightCheck("openclaw_route_smoke", "pass", "ok"),
            ),
        )
        buf = io.StringIO()
        with patch("openclaw_bench.cli.run_verification_gates", return_value=passes), redirect_stdout(buf):
            exit_code = cli_main(
                [
                    "preflight",
                    "--profile",
                    "benchclaw",
                    "--provider",
                    "vllm",
                    "--base-url",
                    "http://10.68.198.1:8000/v1",
                    "--route-model",
                    "vllm/gpt-oss-20b",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertIn("openclaw_route_smoke", buf.getvalue())

    def test_preflight_command_exits_nonzero_on_failure(self):
        failure = VerificationReport(
            ok=False,
            checks=(
                PreflightCheck("openclaw_profile_config", "pass", "ok"),
                PreflightCheck("openclaw_models_list", "fail", "model missing"),
                PreflightCheck("provider_health", "pass", "ok"),
                PreflightCheck("openclaw_route_smoke", "fail", "route timed out"),
            ),
        )
        buf = io.StringIO()
        with patch("openclaw_bench.cli.run_verification_gates", return_value=failure), redirect_stdout(buf):
            exit_code = cli_main(
                [
                    "preflight",
                    "--profile",
                    "benchclaw",
                    "--provider",
                    "vllm",
                    "--base-url",
                    "http://10.68.198.1:8000/v1",
                    "--route-model",
                    "vllm/gpt-oss-20b",
                ]
            )
        self.assertEqual(exit_code, 1)
        out = buf.getvalue()
        self.assertIn("FAIL", out.upper())
        self.assertIn("model missing", out)
```

- [ ] **Step 2: Run tests, verify fail**

```bash
python3 -m unittest tests.test_providers_preflight -v
```

Expected: `ImportError: cannot import name 'VerificationReport'` (and the CLI subcommand will not be registered yet).

- [ ] **Step 3: Add `VerificationReport` + `run_verification_gates` to `preflight.py`**

Append to `openclaw_bench/preflight.py`:

```python
@dataclass(frozen=True)
class VerificationReport:
    ok: bool
    checks: tuple[PreflightCheck, ...]


def _run_gate(name: str, runner, *args, **kwargs) -> PreflightCheck:
    try:
        return runner(*args, **kwargs)
    except Exception as exc:  # surface unexpected failures as gate fails, not crashes
        return PreflightCheck(name, "fail", f"unexpected error: {exc!r}")


def run_verification_gates(
    *,
    profile: str,
    provider: str,
    base_url: str,
    route_model: str,
    container: str | None = None,
    home: Path | None = None,
    timeout_s: int = 60,
) -> VerificationReport:
    config_check = _run_gate(
        "openclaw_profile_config", _check_profile_config, profile, container, container is None
    )
    models_check = _run_gate(
        "openclaw_models_list",
        _check_openclaw_model_routes,
        profile,
        container,
        provider,
    )
    health_check = _run_gate(
        "provider_health",
        _check_provider_health,
        base_url,
        container,
        timeout_s,
    )
    smoke_check = _run_gate(
        "openclaw_route_smoke",
        _run_openclaw_route_smoke,
        profile,
        container,
        route_model,
        timeout_s,
    )
    checks = (config_check, models_check, health_check, smoke_check)
    return VerificationReport(ok=all(c.status == "pass" for c in checks), checks=checks)


def _check_provider_health(base_url: str, container: str | None, timeout_s: int) -> PreflightCheck:
    url = base_url.rstrip("/") + "/models"
    if container:
        cmd = ["incus", "exec", container, "--", "curl", "--silent", "--max-time", str(timeout_s), url]
    else:
        cmd = ["curl", "--silent", "--max-time", str(timeout_s), url]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s + 5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return PreflightCheck("provider_health", "fail", str(exc))
    if proc.returncode == 0 and proc.stdout.strip():
        return PreflightCheck("provider_health", "pass", _trim_output(proc.stdout))
    return PreflightCheck("provider_health", "fail", _trim_output(proc.stderr or proc.stdout or "no response"))
```

(`_check_openclaw_model_routes` and `_run_openclaw_route_smoke` already exist as private helpers in `preflight.py`. The new `run_verification_gates` function wraps them via the `_run_gate` indirection so tests can patch `_run_gate` to inject canned `PreflightCheck` results.)

- [ ] **Step 4: Add the `preflight` subcommand to `openclaw_bench/cli.py`**

In `openclaw_bench/cli.py`, near the existing subparsers, add:

```python
def _add_preflight_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "preflight",
        help="Run the four verification gates against an OpenClaw profile + provider route.",
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--provider", required=True, choices=["vllm", "ollama", "llamacpp", "lmstudio"])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--route-model", required=True)
    parser.add_argument("--container", default=None)
    parser.add_argument("--timeout-s", type=int, default=60)
    parser.set_defaults(handler=preflight_command)


def preflight_command(args) -> int:
    from .preflight import run_verification_gates  # noqa: WPS433 — lazy import keeps cold start fast

    report = run_verification_gates(
        profile=args.profile,
        provider=args.provider,
        base_url=args.base_url,
        route_model=args.route_model,
        container=args.container,
        timeout_s=args.timeout_s,
    )
    for check in report.checks:
        status = "PASS" if check.status == "pass" else "FAIL"
        print(f"{status}\t{check.name}\t{check.notes}")
    return 0 if report.ok else 1
```

Wire `_add_preflight_parser(subparsers)` into the existing `main()`'s parser construction. Add `from .preflight import run_verification_gates` at the top of `cli.py` so the test patch target `openclaw_bench.cli.run_verification_gates` resolves.

- [ ] **Step 5: Run tests, verify pass**

```bash
python3 -m unittest tests.test_providers_preflight -v
```

Expected: `OK` with 4 tests.

- [ ] **Step 6: Run full regression**

```bash
python3 -m unittest discover -s tests
```

Expected: all existing tests still pass plus the new ones (count grows by ~32 from prior 244 baseline; exact number depends on intermediate task counts, but the only failures should be zero).

- [ ] **Step 7: Commit**

```bash
git add openclaw_bench/preflight.py openclaw_bench/cli.py tests/test_providers_preflight.py
git commit -m "$(cat <<'EOF'
Add oc-bench preflight command and run_verification_gates

Wraps four gates - openclaw config validate, models list, provider
health from the configured runtime, and OpenClaw route smoke - into a
single VerificationReport and exposes it as `oc-bench preflight`.
Exit code is zero only if all four gates pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Extend `oc-bench init` to use detection

**Files:**
- Modify: `openclaw_bench/quickstart.py` (accept optional `ProviderCandidate`)
- Modify: `openclaw_bench/cli.py` (run detection before `init_quickstart`)
- Test: `tests/test_providers_init_integration.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_providers_init_integration.py
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openclaw_bench.cli import main as cli_main
from openclaw_bench.providers import DetectionReport, ProviderCandidate
from openclaw_bench.providers.probes import ProbeResult


def _ok(body: str) -> ProbeResult:
    return ProbeResult(ok=True, status_code=200, body=body, probe_name="host", error=None)


class InitWithDetectionTests(unittest.TestCase):
    def test_init_uses_detected_vllm_candidate_for_route_config(self):
        candidate = ProviderCandidate(
            provider="vllm",
            base_url="http://10.68.198.1:8000/v1",
            models=["gpt-oss-20b"],
            probe_results={"host": _ok("{}")},
            source="port_probe",
        )
        report = DetectionReport(candidates=(candidate,), findings=())
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            bench = Path(tmp) / "bench"
            with patch("openclaw_bench.cli.run_detection", return_value=report):
                exit_code = cli_main(
                    [
                        "init",
                        "--providers",
                        "local",
                        "--bench-root",
                        str(bench),
                        "--home",
                        str(home),
                        "--port",
                        "19222",
                        "--no-validate",
                    ]
                )
            self.assertEqual(exit_code, 0)
            config = json.loads((home / ".openclaw-benchclaw" / "openclaw.json").read_text())
            vllm = config["models"]["providers"]["vllm"]
            self.assertEqual(vllm["baseUrl"], "http://10.68.198.1:8000/v1")
            self.assertEqual(vllm["models"][0]["id"], "gpt-oss-20b")

    def test_no_detect_falls_back_to_env_var_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            bench = Path(tmp) / "bench"
            exit_code = cli_main(
                [
                    "init",
                    "--providers",
                    "local",
                    "--no-detect",
                    "--vllm-base-url",
                    "http://127.0.0.1:9999/v1",
                    "--vllm-model",
                    "fallback-model",
                    "--bench-root",
                    str(bench),
                    "--home",
                    str(home),
                    "--port",
                    "19223",
                    "--no-validate",
                ]
            )
        self.assertEqual(exit_code, 0)
        config = json.loads((home / ".openclaw-benchclaw" / "openclaw.json").read_text())
        self.assertEqual(config["models"]["providers"]["vllm"]["baseUrl"], "http://127.0.0.1:9999/v1")
        self.assertEqual(
            config["models"]["providers"]["vllm"]["models"][0]["id"], "fallback-model"
        )

    def test_zero_candidates_with_detect_aborts_with_clear_error(self):
        report = DetectionReport(candidates=(), findings=())
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            bench = Path(tmp) / "bench"
            with patch("openclaw_bench.cli.run_detection", return_value=report):
                exit_code = cli_main(
                    [
                        "init",
                        "--providers",
                        "local",
                        "--bench-root",
                        str(bench),
                        "--home",
                        str(home),
                        "--port",
                        "19224",
                        "--no-validate",
                    ]
                )
        self.assertEqual(exit_code, 2)
```

- [ ] **Step 2: Run tests, verify fail**

```bash
python3 -m unittest tests.test_providers_init_integration -v
```

Expected: failure on the detection patch target (`openclaw_bench.cli.run_detection` does not exist) and on missing `--no-detect`/`--bench-root` flags.

- [ ] **Step 3: Extend `init_quickstart` to accept a candidate**

In `openclaw_bench/quickstart.py`:

```python
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
    vllm_base_url: str | None = None,
    vllm_model: str | None = None,
    vllm_context: int = DEFAULT_VLLM_CONTEXT,
    vllm_max_tokens: int = DEFAULT_VLLM_MAX_TOKENS,
    detected_candidate: "ProviderCandidate | None" = None,  # NEW
) -> QuickstartInitResult:
    provider_mode = normalize_provider_selection(providers)
    if detected_candidate is not None and detected_candidate.provider == "vllm":
        vllm = VllmEndpoint(
            base_url=detected_candidate.base_url,
            model=detected_candidate.models[0],
            context=vllm_context,
            max_tokens=vllm_max_tokens,
        )
    else:
        vllm = _vllm_endpoint(vllm_base_url, vllm_model, vllm_context, vllm_max_tokens)
    # ... existing body unchanged ...
```

The forward reference `"ProviderCandidate | None"` avoids a circular import — only used as a type hint, never instantiated here.

- [ ] **Step 4: Wire detection into `cli.py`'s `init_command`**

In `openclaw_bench/cli.py`, at the top:

```python
from .providers import ProviderCandidate, derive_probes_for_profile, run_detection
```

Modify the `init` argparse setup to add:

```python
parser.add_argument("--no-detect", action="store_true", help="Skip provider auto-detection; use --vllm-* flags or env vars.")
parser.add_argument("--oc-runtime", default=None, help="Override OpenClaw runtime probe target (e.g., ssh:user@host).")
parser.add_argument("--no-validate", dest="validate", action="store_false", default=True)
parser.add_argument("--bench-root", default=None)
parser.add_argument("--home", default=None)
```

In `init_command`:

```python
def init_command(args) -> int:
    bench_root = Path(args.bench_root) if args.bench_root else None
    home = Path(args.home) if args.home else None

    detected: ProviderCandidate | None = None
    if not args.no_detect and args.providers in {"local", "both"}:
        probes = derive_probes_for_profile(
            args.profile if hasattr(args, "profile") and args.profile else "benchclaw",
            home=home or Path.home(),
            oc_runtime_override=args.oc_runtime,
        )
        report = run_detection(
            providers=["vllm", "ollama", "llamacpp", "lmstudio"],
            probes=probes,
            home=home or Path.home(),
        )
        for finding in report.findings:
            print(f"finding: {finding}")
        if not report.candidates:
            print(
                "no local provider detected; pass --no-detect with --vllm-base-url/--vllm-model "
                "to specify one explicitly, or start a model server first"
            )
            return 2
        detected = next((c for c in report.candidates if c.provider == "vllm"), report.candidates[0])

    init_quickstart(
        providers=args.providers,
        project_root=PROJECT_ROOT,
        bench_root=bench_root,
        home=home,
        port=args.port,
        validate=args.validate,
        vllm_base_url=args.vllm_base_url,
        vllm_model=args.vllm_model,
        detected_candidate=detected,
    )
    return 0
```

(Existing `init_command` may have a different signature; preserve other arguments and behaviors. The detection block is additive.)

- [ ] **Step 5: Run tests, verify pass**

```bash
python3 -m unittest tests.test_providers_init_integration -v
```

Expected: `OK` with 3 tests.

- [ ] **Step 6: Run full regression**

```bash
python3 -m unittest discover -s tests
```

Expected: all tests pass; no regressions in `test_quickstart.py`.

- [ ] **Step 7: Commit**

```bash
git add openclaw_bench/cli.py openclaw_bench/quickstart.py tests/test_providers_init_integration.py
git commit -m "$(cat <<'EOF'
Wire provider detection into oc-bench init

Init now runs the detection cascade before generating the OpenClaw
config and uses the detected vLLM candidate for baseUrl / model id.
--no-detect falls back to the prior env-var path; --oc-runtime forces
a probe target for SSH/remote OC runtimes. Empty detection aborts
with a clear message instead of writing a default-pointed config.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Live integration test, full regression, STATUS.md update

**Files:**
- Create: `tests/test_providers_live.py`
- Modify: `STATUS.md`

- [ ] **Step 1: Write the live integration test**

```python
# tests/test_providers_live.py
import os
import unittest

from openclaw_bench.providers import (
    IncusExecProbe,
    LocalProbe,
    run_detection,
)


@unittest.skipUnless(os.environ.get("OC_BENCH_LIVE") == "1", "OC_BENCH_LIVE!=1; skipping live test")
class LiveGptOssDetectionTests(unittest.TestCase):
    def test_detection_finds_gpt_oss_via_oc_stack(self):
        from pathlib import Path

        report = run_detection(
            providers=["vllm"],
            probes=[LocalProbe(), IncusExecProbe("oc-stack")],
            home=Path.home(),
        )
        vllm_candidates = [c for c in report.candidates if c.provider == "vllm"]
        self.assertTrue(vllm_candidates, msg=f"no vllm candidate; report={report}")
        candidate = vllm_candidates[0]
        self.assertIn("gpt-oss-20b", candidate.models)
        host_result = candidate.probe_results.get("host")
        runtime_result = candidate.probe_results.get("incus:oc-stack")
        self.assertTrue(host_result and host_result.ok)
        self.assertTrue(runtime_result and runtime_result.ok)
        self.assertFalse(
            any(f.startswith("reachable_from_host_not_runtime") for f in report.findings),
            msg=f"unexpected mismatch finding: {report.findings}",
        )
```

- [ ] **Step 2: Run the live test**

```bash
OC_BENCH_LIVE=1 python3 -m unittest tests.test_providers_live -v
```

Expected: `OK` with 1 test, with `host` and `incus:oc-stack` both reaching `gpt-oss-20b` (per STATUS.md UFW fix).

If the test fails:
- If host fails: vLLM is not running on `:8000` — start it.
- If runtime fails: UFW rule for `10.68.198.10` -> `10.68.198.1:8000` is missing — restore it per STATUS.md line 31.
- If `report.findings` includes a mismatch: same UFW issue, do not proceed until resolved.

- [ ] **Step 3: Run full unit regression**

```bash
python3 -m unittest discover -s tests
```

Expected: all tests pass. Test count grows from prior baseline by the new tests added across Tasks 1-14.

- [ ] **Step 4: Run simulator certification regression**

```bash
python3 -m openclaw_bench run \
  --backend simulator \
  --suite manifests/openclaw-certification-full.example.json \
  --models simulated-model \
  --kv fp8 \
  --concurrency 1 \
  --contexts 4096,8192,16384,32768,65536 \
  --out /tmp/openclaw-bench-m3-providers-verify \
  --run-id cert-full
```

Expected: `40` attempts, `0` failures.

- [ ] **Step 5: Update STATUS.md**

In `STATUS.md`, under the "Verified" section, append a new bullet block:

```markdown
- M3 provider-detection deployment surface slice committed:
  - `openclaw_bench/providers/` package with `probes.py`, `detect.py`, `vllm.py` (full), and detect-only stubs for `ollama.py`, `llamacpp.py`, `lmstudio.py`
  - `oc-bench preflight` command wraps four verification gates (openclaw config validate, models list, provider health, OpenClaw route smoke)
  - `oc-bench init` runs detection automatically; `--no-detect` keeps the env-var path; `--oc-runtime <kind:target>` for SSH/remote runtimes
  - Live integration test passes against GPT-OSS 20B via `oc-stack` (no host-vs-runtime mismatch finding)
  - `python3 -m unittest discover -s tests` all green; simulator certification full run produced 40 attempts, 0 failures
```

Under "Resume Point" / "Open Items", add:

```markdown
- M3 deployment surface for vLLM is shipped; Ollama, llama.cpp, and LM Studio remain detect-only stubs. Generators are follow-up commits gated by spinning up each runtime on the RTX Pro 5000 (or equivalent) and validating against a real instance.
- M2 tier-small live calibration record is still open. Pick a 4-8B candidate that the new detection surface discovers (e.g., Llama 3.2 3B or Qwen3-8B via Ollama) and rerun the small-tier slice.
```

- [ ] **Step 6: Commit STATUS.md update + live test**

```bash
git add STATUS.md tests/test_providers_live.py
git commit -m "$(cat <<'EOF'
Add live GPT-OSS detection test and record M3 surface slice

Live test gated by OC_BENCH_LIVE=1 verifies the cascade picks up
gpt-oss-20b through both the host probe and IncusExecProbe('oc-stack')
without raising the host-vs-runtime mismatch finding. STATUS.md now
records the deployment-surface slice as shipped and explicitly notes
that Ollama/llama.cpp/LM Studio generators and the M2 tier-small
calibration are still open.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage check:**

| Spec section | Tasks |
|---|---|
| Probe location model (default host, auto-derive split, SSH explicit flag) | Tasks 1, 2, 8 |
| Detection cascade (already-known → port probe → 30s budget → mismatch findings) | Tasks 5, 6, 7 |
| `oc-bench init` extended (auto-detect, `--no-detect`, `--oc-runtime`) | Task 14 |
| `oc-bench preflight` (four gates, exit code) | Task 13 |
| vLLM module: detect + generate + parameter shaping (Qwen + GPT-OSS) | Task 9 |
| Ollama / llama.cpp / LM Studio detect-only stubs (NotImplementedError on generate) | Tasks 10, 11, 12 |
| Reuse of `quickstart._vllm_provider_config` (no re-introduction of solved bugs) | Task 9 |
| Captured/realistic provider response fixtures | Task 3 |
| Live integration test gated by `OC_BENCH_LIVE=1` | Task 15 |
| STATUS.md updated with M3 slice + open items | Task 15 |

All spec sections have task coverage. No gaps.

**Placeholder scan:** none — every step contains the actual code or command an engineer needs.

**Type consistency check:**
- `ProbeResult` fields (`ok`, `status_code`, `body`, `probe_name`, `error`) are consistent across Tasks 1, 2, 6, 7.
- `ProviderCandidate` fields (`provider`, `base_url`, `models`, `probe_results`, `source`) are consistent across Tasks 4, 5, 6, 7, 9, 10, 11, 12, 14, 15.
- `DetectionReport` fields (`candidates`, `findings`) consistent across Tasks 4, 7, 14, 15.
- Probe class names (`LocalProbe`, `IncusExecProbe`, `DockerExecProbe`, `SSHProbe`) match across Tasks 1, 2, 8, 14, 15.
- `VerificationReport` and `run_verification_gates` shape stays consistent across Task 13's tests and the import in Task 14.
- Function names: `run_detection` (not `detect_providers`), `port_probe_provider` (not `probe_provider`), `derive_probes_for_profile` (not `get_probes`).

No drift between tasks.

**Scope:** Single subsystem — provider detection surface. Tractable for one execution session.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-02-provider-detection-surface-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints for review.

Which approach?
