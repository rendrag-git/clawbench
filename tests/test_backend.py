import sys
import time
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openclaw_bench.backend import OpenClawBackend, _run_openclaw_command
from openclaw_bench.models import ModelSpec


class BackendTests(unittest.TestCase):
    def test_openclaw_smoke_timeout_raw_output_is_json_safe_text(self):
        backend = OpenClawBackend(local=True)
        model = ModelSpec(model_id="model", served_model_name="provider/model")
        timeout = subprocess.TimeoutExpired(
            cmd=["openclaw"],
            timeout=1,
            output=b"partial stdout",
            stderr=b"partial stderr",
        )
        with patch("subprocess.Popen", return_value=_FakeProcess(timeout=timeout)):
            response = backend.smoke(model, timeout_s=1)
        self.assertEqual(response.text, "partial stdout")
        self.assertEqual(response.raw["stderr"], "partial stderr")
        self.assertEqual(response.error, "openclaw_timeout")
        self.assertEqual(response.request_errors, 1)

    def test_openclaw_smoke_timeout_with_unknown_model_is_route_failure(self):
        backend = OpenClawBackend(local=True)
        model = ModelSpec(model_id="model", served_model_name="provider/missing")
        timeout = subprocess.TimeoutExpired(
            cmd=["openclaw"],
            timeout=1,
            output=b"",
            stderr=b"FailoverError: Unknown model: openai/provider/missing",
        )
        with patch("subprocess.Popen", return_value=_FakeProcess(timeout=timeout)):
            response = backend.smoke(model, timeout_s=1)
        self.assertEqual(response.error, "model_route_failed")
        self.assertEqual(response.timed_out, False)

    def test_openclaw_commands_use_explicit_route_model_name(self):
        backend = OpenClawBackend(local=True)
        model = ModelSpec(
            model_id="vllm-model",
            served_model_name="vllm-served-name",
            openclaw_model_name="openclaw/local-alias",
        )
        with patch("subprocess.Popen", return_value=_FakeProcess(returncode=0, stdout='{"text":"ok"}', stderr="")) as popen_mock:
            response = backend.smoke(model, timeout_s=1)
        self.assertIsNone(response.error)
        self.assertEqual(response.request_errors, 0)
        cmd = popen_mock.call_args.args[0]
        self.assertIn("openclaw/local-alias", cmd)
        self.assertNotIn("vllm-served-name", cmd)

    def test_openclaw_extracts_extended_efficiency_metrics(self):
        stdout = (
            '{"text":"ok","metrics":{"tool_calls":4,"files_read":4,'
            '"file_reads":["app/a.py","app/b.py","app/a.py"],'
            '"time_to_first_relevant_file_s":1.25}}'
        )
        with patch("subprocess.Popen", return_value=_FakeProcess(returncode=0, stdout=stdout, stderr="")):
            response = _run_openclaw_command(["openclaw"], None, timeout_s=1, default_error="unknown")
        self.assertIsNone(response.error)
        self.assertEqual(response.tool_calls, 4)
        self.assertEqual(response.files_read, 4)
        self.assertEqual(response.duplicate_file_reads, 1)
        self.assertEqual(response.time_to_first_relevant_file_s, 1.25)

    def test_openclaw_container_smoke_uses_docker_exec(self):
        backend = OpenClawBackend(profile="bench", container="oc-bench-gateway")
        model = ModelSpec(model_id="model", served_model_name="provider/model")
        with patch("subprocess.Popen", return_value=_FakeProcess(returncode=0, stdout='{"text":"ok"}', stderr="")) as popen_mock:
            response = backend.smoke(model, timeout_s=1)
        self.assertIsNone(response.error)
        cmd = popen_mock.call_args.args[0]
        self.assertEqual(cmd[:4], ["docker", "exec", "oc-bench-gateway", "openclaw"])
        self.assertIn("--profile", cmd)

    def test_openclaw_error_counts_as_request_error(self):
        backend = OpenClawBackend(local=True)
        model = ModelSpec(model_id="model", served_model_name="provider/model")
        with patch("subprocess.Popen", return_value=_FakeProcess(returncode=1, stdout="", stderr="server error")):
            response = backend.smoke(model, timeout_s=1)
        self.assertEqual(response.error, "model_route_failed")
        self.assertEqual(response.request_errors, 1)

    def test_openclaw_context_window_error_is_classified(self):
        backend = OpenClawBackend(local=True)
        model = ModelSpec(model_id="model", served_model_name="provider/model")
        stderr = "This model's maximum context length is 4096 tokens. However, you requested 1024 output tokens and your prompt contains at least 3073 input tokens."
        with patch("subprocess.Popen", return_value=_FakeProcess(returncode=1, stdout="", stderr=stderr)):
            response = backend.smoke(model, timeout_s=1)
        self.assertEqual(response.error, "context_window_exceeded")

    def test_openclaw_tool_parser_error_is_classified(self):
        backend = OpenClawBackend(local=True)
        model = ModelSpec(model_id="model", served_model_name="provider/model")
        stderr = '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'
        with patch("subprocess.Popen", return_value=_FakeProcess(returncode=1, stdout="", stderr=stderr)):
            response = backend.smoke(model, timeout_s=1)
        self.assertEqual(response.error, "tool_parser_missing")

    def test_openclaw_gateway_model_override_rejection_is_classified_even_with_cli_fallback(self):
        stdout = '{"payloads":[{"text":"fallback answer"}],"meta":{"fallbackFrom":"gateway"}}'
        stderr = "EMBEDDED FALLBACK: Gateway agent failed; running embedded agent: provider/model overrides are not authorized for this caller."
        with patch("subprocess.Popen", return_value=_FakeProcess(returncode=0, stdout=stdout, stderr=stderr)):
            response = _run_openclaw_command(["openclaw"], None, timeout_s=1, default_error="unknown")
        self.assertEqual(response.error, "model_override_unauthorized")
        self.assertEqual(response.request_errors, 1)

    def test_openclaw_incomplete_terminal_response_is_classified_before_embedded_fallback(self):
        stderr = (
            "EMBEDDED FALLBACK: Gateway agent failed; running embedded agent: "
            "GatewayClientRequestError: FailoverError: vllm/qwen ended with an incomplete terminal response"
        )
        with patch("subprocess.Popen", return_value=_FakeProcess(returncode=0, stdout="", stderr=stderr)):
            response = _run_openclaw_command(["openclaw"], None, timeout_s=1, default_error="unknown")
        self.assertEqual(response.error, "incomplete_result")
        self.assertEqual(response.request_errors, 1)

    def test_openclaw_context_error_takes_priority_over_incomplete_terminal_response(self):
        stderr = (
            "FailoverError: vllm/qwen ended with an incomplete terminal response. "
            "This model's maximum context length is 8192 tokens. However, you requested 128 output tokens "
            "and your prompt contains at least 8065 input tokens."
        )
        with patch("subprocess.Popen", return_value=_FakeProcess(returncode=0, stdout="", stderr=stderr)):
            response = _run_openclaw_command(["openclaw"], None, timeout_s=1, default_error="unknown")
        self.assertEqual(response.error, "context_window_exceeded")

    def test_openclaw_success_json_error_payload_is_classified(self):
        stdout = (
            '{"payloads":[{"text":"Context overflow: prompt too large for the model."}],'
            '"meta":{"error":{"kind":"context_overflow","message":"maximum context length is 8192 tokens"}}}'
        )
        with patch("subprocess.Popen", return_value=_FakeProcess(returncode=0, stdout=stdout, stderr="")):
            response = _run_openclaw_command(["openclaw"], None, timeout_s=1, default_error="unknown")
        self.assertEqual(response.error, "context_window_exceeded")
        self.assertEqual(response.request_errors, 1)

    def test_workspace_agents_register_attempt_agent_and_omit_per_call_model_override(self):
        backend = OpenClawBackend(profile="bench", workspace_agents=True)
        model = ModelSpec(model_id="model", served_model_name="served", openclaw_model_name="vllm/served")
        task = _Task()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            completed = subprocess.CompletedProcess(args=[], returncode=0, stdout='{"ok":true}', stderr="")
            with patch("subprocess.run", return_value=completed) as run_mock:
                with patch("subprocess.Popen", return_value=_FakeProcess(returncode=0, stdout='{"text":"ok"}', stderr="")) as popen_mock:
                    response = backend.run(model, task, workspace, "run-w000-task-abcdef", timeout_s=30)
        self.assertIsNone(response.error)
        add_cmd = run_mock.call_args.args[0]
        self.assertEqual(add_cmd[:5], ["openclaw", "--profile", "bench", "agents", "add"])
        self.assertIn("--workspace", add_cmd)
        self.assertIn(str(workspace), add_cmd)
        self.assertIn("--model", add_cmd)
        self.assertIn("vllm/served", add_cmd)
        agent_cmd = popen_mock.call_args.args[0]
        self.assertIn("--agent", agent_cmd)
        self.assertIn("bench-run-w000-task-abcdef", agent_cmd)
        self.assertNotIn("--model", agent_cmd)

    def test_openclaw_container_agent_run_sets_container_workdir(self):
        backend = OpenClawBackend(profile="bench", container="oc-bench-gateway")
        model = ModelSpec(model_id="model", served_model_name="served", openclaw_model_name="vllm/served")
        task = _Task()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            with patch("subprocess.Popen", return_value=_FakeProcess(returncode=0, stdout='{"text":"ok"}', stderr="")) as popen_mock:
                response = backend.run(model, task, workspace, "run-w000-task-abcdef", timeout_s=30)
        self.assertIsNone(response.error)
        cmd = popen_mock.call_args.args[0]
        self.assertEqual(cmd[:4], ["docker", "exec", "-w", str(workspace)])
        self.assertEqual(cmd[4:6], ["oc-bench-gateway", "openclaw"])
        self.assertIsNone(popen_mock.call_args.kwargs["cwd"])

    def test_workspace_agent_setup_failure_is_reported_before_attempt(self):
        backend = OpenClawBackend(profile="bench", workspace_agents=True)
        model = ModelSpec(model_id="model", served_model_name="served", openclaw_model_name="vllm/served")
        task = _Task()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            completed = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="bad config")
            with patch("subprocess.run", return_value=completed):
                with patch("subprocess.Popen") as popen_mock:
                    response = backend.run(model, task, workspace, "run-w000-task-abcdef", timeout_s=30)
        self.assertEqual(response.error, "openclaw_agent_setup_failed")
        self.assertEqual(response.request_errors, 1)
        popen_mock.assert_not_called()

    def test_openclaw_timeout_kills_child_process_group_without_hanging_on_pipes(self):
        code = (
            "import subprocess, sys, time\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
            "time.sleep(30)\n"
        )
        started = time.monotonic()

        response = _run_openclaw_command([sys.executable, "-c", code], None, timeout_s=1, default_error="unknown")

        self.assertLess(time.monotonic() - started, 8)
        self.assertEqual(response.error, "openclaw_timeout")
        self.assertEqual(response.request_errors, 1)


class _FakeProcess:
    def __init__(
        self,
        returncode: int | None = 0,
        stdout: str = "",
        stderr: str = "",
        timeout: subprocess.TimeoutExpired | None = None,
    ) -> None:
        self.returncode = returncode
        self.pid = 999999
        self._stdout = stdout
        self._stderr = stderr
        self._timeout = timeout
        self._communicate_calls = 0

    def communicate(self, timeout: int | None = None):
        self._communicate_calls += 1
        if self._timeout is not None and self._communicate_calls == 1:
            raise self._timeout
        return self._stdout, self._stderr

    def wait(self, timeout: int | None = None):
        self.returncode = -15
        return self.returncode


class _Task:
    prompt = "Reply with ok"


if __name__ == "__main__":
    unittest.main()
