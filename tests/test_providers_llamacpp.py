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
