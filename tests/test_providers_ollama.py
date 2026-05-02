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
