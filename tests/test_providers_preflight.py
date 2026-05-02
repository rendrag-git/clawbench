import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

from openclaw_bench.cli import main as cli_main
from openclaw_bench.preflight import PreflightCheck, VerificationReport, _check_provider_health, run_verification_gates
from openclaw_bench.providers.probes import ProbeResult


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


class ProviderPreflightCliTests(unittest.TestCase):
    def test_provider_preflight_command_exits_zero_on_pass(self):
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
                    "provider-preflight",
                    "--profile", "benchclaw",
                    "--provider", "vllm",
                    "--base-url", "http://10.68.198.1:8000/v1",
                    "--route-model", "vllm/gpt-oss-20b",
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertIn("openclaw_route_smoke", buf.getvalue())

    def test_provider_preflight_command_exits_nonzero_on_failure(self):
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
                    "provider-preflight",
                    "--profile", "benchclaw",
                    "--provider", "vllm",
                    "--base-url", "http://10.68.198.1:8000/v1",
                    "--route-model", "vllm/gpt-oss-20b",
                ]
            )
        self.assertEqual(exit_code, 1)
        out = buf.getvalue()
        self.assertIn("FAIL", out.upper())
        self.assertIn("model missing", out)


class CheckProviderHealthTests(unittest.TestCase):
    def test_unauthenticated_401_returns_fail_not_pass(self):
        # Simulate the silent-pass bug: 401 response with a JSON error body.
        # LocalProbe is lazily imported inside _check_provider_health to avoid
        # a circular-import cycle, so we patch it at its definition site.
        unauthorized = ProbeResult(
            ok=False,
            status_code=401,
            body='{"error":"Unauthorized"}',
            probe_name="host",
            error="http_401",
        )
        with patch("openclaw_bench.providers.probes.LocalProbe") as fake_probe_cls:
            fake_probe = MagicMock()
            fake_probe.http_get.return_value = unauthorized
            fake_probe_cls.return_value = fake_probe
            check = _check_provider_health(
                "http://10.68.198.1:8000/v1", None, 5, provider="vllm"
            )
        self.assertEqual(check.status, "fail")
        self.assertIn("401", check.notes)

    def test_authenticated_200_returns_pass(self):
        success = ProbeResult(
            ok=True, status_code=200, body='{"data":[]}', probe_name="host", error=None
        )
        with patch("openclaw_bench.providers.probes.LocalProbe") as fake_probe_cls:
            fake_probe = MagicMock()
            fake_probe.http_get.return_value = success
            fake_probe_cls.return_value = fake_probe
            check = _check_provider_health(
                "http://10.68.198.1:8000/v1", None, 5, provider="vllm"
            )
        self.assertEqual(check.status, "pass")
