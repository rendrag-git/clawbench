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
