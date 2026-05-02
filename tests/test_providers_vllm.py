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
