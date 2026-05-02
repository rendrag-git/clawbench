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
