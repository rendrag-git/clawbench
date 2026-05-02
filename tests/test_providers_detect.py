import json
import tempfile
import unittest
from pathlib import Path

from openclaw_bench.providers.detect import DetectionReport, ProviderCandidate, scan_existing_oc_profiles


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
