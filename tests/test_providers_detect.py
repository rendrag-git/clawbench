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
