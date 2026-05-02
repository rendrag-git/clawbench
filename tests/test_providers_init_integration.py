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
                        "--providers", "local",
                        "--bench-root", str(bench),
                        "--config-home", str(home),
                        "--gateway-port", "19222",
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
                    "--providers", "local",
                    "--no-detect",
                    "--vllm-base-url", "http://127.0.0.1:9999/v1",
                    "--vllm-model", "fallback-model",
                    "--bench-root", str(bench),
                    "--config-home", str(home),
                    "--gateway-port", "19223",
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
                        "--providers", "local",
                        "--bench-root", str(bench),
                        "--config-home", str(home),
                        "--gateway-port", "19224",
                        "--no-validate",
                    ]
                )
        self.assertEqual(exit_code, 2)
