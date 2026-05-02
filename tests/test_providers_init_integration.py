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


class InitInheritsExistingProfileTests(unittest.TestCase):
    """The whole point: when detection finds an OC profile that already has
    providers configured, init clones it verbatim instead of regenerating.

    This avoids issues #1 (silent fallback for non-vLLM detection) and #7
    (parameter shaping dropped on the no-detect path) entirely — there is no
    generation step on this path.
    """

    def _seed_source_profile(self, home: Path, profile_name: str = "pmg") -> Path:
        """Drop a realistic source profile JSON in $HOME/.openclaw-<name>/."""
        profile_dir = home / f".openclaw-{profile_name}"
        profile_dir.mkdir(parents=True, exist_ok=True)
        config_path = profile_dir / "openclaw.json"
        config_path.write_text(json.dumps({
            "agents": {
                "defaults": {
                    "model": "vllm/qwen3.5-4b",
                    "models": {
                        "vllm/qwen3.5-4b": {
                            "params": {
                                "chatTemplateKwargs": {"enable_thinking": False},
                                "maxTokens": 256,
                            }
                        },
                        "ollama/qwen3:8b": {"params": {"maxTokens": 512}},
                    },
                    "params": {"maxTokens": 256},
                    "skipBootstrap": False,
                },
                "list": [{"id": "main", "model": "vllm/qwen3.5-4b", "tools": {"profile": "coding"}}],
            },
            "env": {"vars": {"VLLM_API_KEY": "test-api-key"}},
            "gateway": {
                "auth": {"mode": "token", "token": "source-token-XYZ"},
                "bind": "loopback",
                "mode": "local",
                "port": 19298,
                "tailscale": {"mode": "off"},
            },
            "models": {
                "providers": {
                    "vllm": {
                        "api": "openai-completions",
                        "baseUrl": "http://example.local:8003/v1",
                        "models": [{"id": "qwen3.5-4b", "name": "qwen3.5-4b", "contextWindow": 32768, "maxTokens": 256}],
                        "request": {
                            "auth": {
                                "mode": "authorization-bearer",
                                "token": {"id": "VLLM_API_KEY", "provider": "bench", "source": "env"},
                            },
                        },
                    },
                    "ollama": {
                        "api": "ollama",
                        "baseUrl": "http://example.local:11434",
                        "models": [{"id": "qwen3:8b", "name": "qwen3:8b", "contextWindow": 32768}],
                    },
                }
            },
            "secrets": {"providers": {"bench": {"allowlist": ["VLLM_API_KEY"], "source": "env"}}},
        }), encoding="utf-8")
        return config_path

    def test_init_clones_existing_profile_into_bench_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            bench = Path(tmp) / "bench"
            self._seed_source_profile(home, "pmg")

            # Detection runs against the seeded $HOME — no mocking needed; the
            # real scan_existing_oc_profiles will find pmg.
            exit_code = cli_main([
                "init",
                "--providers", "local",
                "--bench-root", str(bench),
                "--config-home", str(home),
                "--openclaw-profile", "bench-pmg",
                "--gateway-port", "19350",
                "--no-validate",
            ])
            self.assertEqual(exit_code, 0)

            # Bench profile written
            bench_config_path = home / ".openclaw-bench-pmg" / "openclaw.json"
            self.assertTrue(bench_config_path.is_file(), msg="bench profile should be written")
            bench_config = json.loads(bench_config_path.read_text(encoding="utf-8"))

            # Provider wiring preserved verbatim from source — both vLLM and Ollama
            self.assertIn("vllm", bench_config["models"]["providers"])
            self.assertIn("ollama", bench_config["models"]["providers"])
            self.assertEqual(
                bench_config["models"]["providers"]["vllm"]["baseUrl"],
                "http://example.local:8003/v1",
            )
            self.assertEqual(
                bench_config["models"]["providers"]["ollama"]["baseUrl"],
                "http://example.local:11434",
            )

            # Per-model parameter shaping preserved (the issue #7 regression scenario)
            qwen_params = bench_config["agents"]["defaults"]["models"]["vllm/qwen3.5-4b"]["params"]
            self.assertFalse(qwen_params["chatTemplateKwargs"]["enable_thinking"])

            # Bench overlay applied
            self.assertEqual(bench_config["gateway"]["port"], 19350)
            self.assertNotEqual(bench_config["gateway"]["auth"]["token"], "source-token-XYZ")
            self.assertTrue(bench_config["agents"]["defaults"]["skipBootstrap"])
            self.assertEqual(len(bench_config["agents"]["list"]), 1)
            self.assertEqual(bench_config["agents"]["list"][0]["id"], "bench")

            # Model manifest derived from source's providers
            manifest_path = bench / "manifests" / "starter-models.json"
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            routes = sorted(m["openclaw_model_name"] for m in manifest["models"])
            self.assertEqual(routes, ["ollama/qwen3:8b", "vllm/qwen3.5-4b"])
            self.assertEqual(manifest["manifest_scope"]["portability"], "inherited")

    def test_init_inherit_path_uses_detected_route_when_source_default_is_dict(self):
        """Real operator profiles have agents.defaults.model as a dict
        {primary: 'openai-codex/...', fallbacks: [...]} pointing at an external
        provider not in models.providers. The inherit path must fall back to the
        detected local provider instead of passing the dict to clone_profile."""
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            bench = Path(tmp) / "bench"
            # Source profile with dict-form agents.defaults.model (external provider)
            src_path = self._seed_source_profile(home, "pmg")
            config = json.loads(src_path.read_text())
            config["agents"]["defaults"]["model"] = {
                "primary": "openai-codex/gpt-5.5",
                "fallbacks": ["openai-codex/gpt-5.4"],
            }
            src_path.write_text(json.dumps(config), encoding="utf-8")

            exit_code = cli_main([
                "init",
                "--providers", "local",
                "--bench-root", str(bench),
                "--config-home", str(home),
                "--openclaw-profile", "bench-pmg",
                "--gateway-port", "19352",
                "--no-validate",
            ])
            self.assertEqual(exit_code, 0)

            bench_config = json.loads(
                (home / ".openclaw-bench-pmg" / "openclaw.json").read_text()
            )
            # Route must be a local provider, not the external openai-codex route
            route = bench_config["agents"]["defaults"]["model"]
            self.assertIsInstance(route, str)
            self.assertTrue(
                route.startswith("vllm/") or route.startswith("ollama/"),
                msg=f"expected a local provider route, got {route!r}",
            )

    def test_init_inherit_path_aborts_when_bench_dir_already_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            bench = Path(tmp) / "bench"
            self._seed_source_profile(home, "pmg")
            # Pre-create the bench profile dir to simulate a re-init
            (home / ".openclaw-bench-pmg").mkdir(parents=True)

            exit_code = cli_main([
                "init",
                "--providers", "local",
                "--bench-root", str(bench),
                "--config-home", str(home),
                "--openclaw-profile", "bench-pmg",
                "--gateway-port", "19351",
                "--no-validate",
            ])
            self.assertEqual(exit_code, 2)
