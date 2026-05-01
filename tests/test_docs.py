import re
import unittest
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class DocumentationTests(unittest.TestCase):
    def test_readme_documents_live_vllm_api_and_certification_workflow(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("manifests/vllm-gptoss-smoke.example.json", readme)
        self.assertIn("manifests/openclaw-agent-discovery-smoke.example.json", readme)
        self.assertIn("manifests/vllm-local.example.json", readme)
        self.assertIn("manifests/vllm-long-context.example.json", readme)
        self.assertIn("manifests/vllm-concurrency-sweep.example.json", readme)
        self.assertIn("manifests/vllm-hardware-setups.example.json", readme)
        self.assertIn("manifests/real-repo-readonly.example.json", readme)
        self.assertIn("--run-id local-vllm-real-repo", readme)
        self.assertIn("/home/ubuntu/openclaw-bench/results/local-vllm-real-repo", readme)
        self.assertIn("manifests/api-providers.example.json", readme)
        self.assertIn("python3 -m openclaw_bench certify", readme)
        self.assertIn("--failures-only", readme)
        self.assertIn("8k-only endpoint is useful for smoke and harness validation, but it cannot certify", readme)
        self.assertIn("OPENAI_API_KEY", readme)
        self.assertIn("ANTHROPIC_API_KEY", readme)
        self.assertIn("host-specific examples for this workstation", readme)
        self.assertIn('manifest_scope.portability = "host_specific"', readme)
        self.assertIn("preflight` can emit a warning", readme)
        self.assertIn("openclaw-config/vllm-provider-smoke.example.json", readme)
        self.assertIn("openclaw-config/qwen36-vllm-provider.merge.example.json", readme)
        self.assertIn("openclaw-config/qwen36-vllm-provider-lean8k.merge.example.json", readme)
        self.assertIn("openclaw-config/qwen36-vllm-provider-lean16k.merge.example.json", readme)
        self.assertIn("openclaw-config/qwen36-agent-default-params.example.json", readme)
        self.assertIn("openclaw-config/qwen36-agent-lean-8k-defaults.example.json", readme)
        self.assertIn("openclaw-config/qwen36-agent-lean-8k-params.example.json", readme)
        self.assertIn("manifests/vllm-qwen36-fp8-lean8k-live.example.json", readme)
        self.assertIn("manifests/vllm-qwen36-fp8-lean16k-live.example.json", readme)
        self.assertIn("config unset tools.profile", readme)
        self.assertIn("openclaw-config/openai-provider.example.json", readme)
        self.assertIn("openclaw-config/anthropic-provider.example.json", readme)
        self.assertIn("--openclaw-smoke-timeout", readme)
        self.assertIn("--openclaw-gateway-timeout", readme)
        self.assertIn("--no-ensure-openclaw-gateway", readme)
        self.assertIn("`oc-bench run`", readme)
        self.assertIn("`oc-bench init`", readme)
        self.assertIn("oc-bench quickstart --providers local --force --stop-after", readme)
        self.assertIn("OAuth-backed providers are bring-your-own-auth", readme)
        self.assertIn("Full certification matrices, long-context and local quant sweeps", readme)
        self.assertIn("gateway --dev --verbose start", readme)
        self.assertIn("--openclaw-container oc-bench-gateway", readme)
        self.assertIn("clawdaddy/openclaw:business-smoke", readme)
        self.assertIn("--no-ensure-openclaw-container", readme)
        self.assertIn("exact-path mounts for the repo, benchmark root, and any custom workspace root", readme)
        self.assertIn("Docker health is advisory", readme)
        self.assertIn("profile-aware gateway status probe", readme)
        self.assertIn("preflight uses `--smoke-timeout`", readme)
        self.assertIn("tools.profile=\"minimal\"", readme)
        self.assertIn("models.providers.vllm", readme)
        self.assertIn("models.providers.openai", readme)
        self.assertIn("models.providers.anthropic", readme)
        self.assertIn("hardware_profile", readme)
        self.assertIn("at least two local hardware/setup profiles", readme)

        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('oc-bench = "openclaw_bench.cli:main"', pyproject)
        self.assertIn('quickstart_assets/**/*.json', pyproject)

    def test_readme_openclaw_live_commands_do_not_use_simulator_manifest(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        bash_blocks = re.findall(r"```bash\n(.*?)\n```", readme, flags=re.DOTALL)
        openclaw_blocks = [block for block in bash_blocks if "--backend openclaw" in block]

        self.assertTrue(openclaw_blocks)
        for block in openclaw_blocks:
            self.assertNotIn("manifests/initial-models.json", block)

    def test_vllm_openclaw_provider_example_matches_documented_manifests(self):
        provider = json.loads((ROOT / "openclaw-config" / "vllm-provider-smoke.example.json").read_text(encoding="utf-8"))
        manifests = [
            "vllm-gptoss-smoke.example.json",
            "vllm-local.example.json",
            "vllm-long-context.example.json",
            "vllm-concurrency-sweep.example.json",
            "vllm-local-candidates.example.json",
            "vllm-hardware-setups.example.json",
        ]
        manifest_models = []
        for manifest_name in manifests:
            manifest = json.loads((ROOT / "manifests" / manifest_name).read_text(encoding="utf-8"))
            manifest_models.extend(manifest["models"])
        provider_models = {model["id"]: model for model in provider["models"]}

        self.assertEqual({model["api_base"] for model in manifest_models}, {provider["baseUrl"]})
        self.assertEqual(provider["api"], "openai-completions")
        self.assertEqual(provider["request"]["auth"]["mode"], "authorization-bearer")
        self.assertEqual({model["api_env"] for model in manifest_models}, {provider["request"]["auth"]["token"]["id"]})
        self.assertTrue(provider["request"]["allowPrivateNetwork"])
        for model in manifest_models:
            provider_model = provider_models[model["served_model_name"]]
            self.assertEqual(provider_model["name"], model["served_model_name"])
            self.assertEqual(provider_model["contextWindow"], max(model["contexts"]))
            self.assertEqual(provider_model["contextTokens"], max(model["contexts"]))
            self.assertEqual(provider_model["maxTokens"], 256)

    def test_hardware_setup_manifest_pairs_each_required_local_kv_mode(self):
        manifest = json.loads((ROOT / "manifests" / "vllm-hardware-setups.example.json").read_text(encoding="utf-8"))
        models = manifest["models"]
        required_kv_modes = {"fp8", "turboquant_k8v4", "turboquant_k3v4_nc"}
        profiles_by_kv = {}
        comparison_ids_by_kv = {}
        for model in models:
            self.assertEqual(model["contexts"], [4096])
            self.assertEqual(model["concurrency"], [1, 4])
            self.assertEqual(model["serve_command"][model["serve_command"].index("--kv-cache-dtype") + 1], model["kv_modes"][0])
            profiles_by_kv.setdefault(model["kv_modes"][0], set()).add(model["hardware_profile"])
            comparison_ids_by_kv.setdefault(model["kv_modes"][0], set()).add(model["comparison_id"])

        self.assertEqual(set(profiles_by_kv), required_kv_modes)
        for kv_mode in required_kv_modes:
            self.assertGreaterEqual(len(profiles_by_kv[kv_mode]), 2)
            self.assertEqual(comparison_ids_by_kv[kv_mode], {"qwen3-dense-hardware-kv"})

    def test_qwen36_bench_profile_patch_matches_live_manifest(self):
        manifest = json.loads((ROOT / "manifests" / "vllm-qwen36-fp8-live.example.json").read_text(encoding="utf-8"))
        provider = json.loads((ROOT / "openclaw-config" / "qwen36-vllm-provider.merge.example.json").read_text(encoding="utf-8"))
        params = json.loads((ROOT / "openclaw-config" / "qwen36-agent-default-params.example.json").read_text(encoding="utf-8"))
        model = manifest["models"][0]
        provider_model = provider["models"][0]

        self.assertEqual(provider["baseUrl"], model["api_base"])
        self.assertTrue(provider["request"]["allowPrivateNetwork"])
        self.assertEqual(provider_model["id"], model["served_model_name"])
        self.assertEqual(provider_model["name"], model["served_model_name"])
        self.assertEqual(provider_model["contextWindow"], max(model["contexts"]))
        self.assertEqual(provider_model["contextTokens"], max(model["contexts"]))
        self.assertEqual(provider_model["maxTokens"], 128)
        self.assertFalse(provider_model["reasoning"])
        self.assertEqual(params["chatTemplateKwargs"], {"enable_thinking": False})
        self.assertEqual(params["maxTokens"], 128)

    def test_qwen36_lean_8k_patch_documents_prompt_trim_variant(self):
        manifest = json.loads((ROOT / "manifests" / "vllm-qwen36-fp8-lean8k-live.example.json").read_text(encoding="utf-8"))
        provider = json.loads((ROOT / "openclaw-config" / "qwen36-vllm-provider-lean8k.merge.example.json").read_text(encoding="utf-8"))
        defaults = json.loads((ROOT / "openclaw-config" / "qwen36-agent-lean-8k-defaults.example.json").read_text(encoding="utf-8"))
        params = json.loads((ROOT / "openclaw-config" / "qwen36-agent-lean-8k-params.example.json").read_text(encoding="utf-8"))
        model = manifest["models"][0]
        provider_model = provider["models"][0]

        self.assertEqual(provider["baseUrl"], model["api_base"])
        self.assertEqual(provider_model["id"], model["served_model_name"])
        self.assertEqual(provider_model["contextWindow"], max(model["contexts"]))
        self.assertEqual(provider_model["contextTokens"], max(model["contexts"]))
        self.assertEqual(provider_model["maxTokens"], 32)
        self.assertIn("lean-max32", model["hardware_profile"])
        self.assertEqual(defaults["contextInjection"], "never")
        self.assertEqual(defaults["bootstrapMaxChars"], 1024)
        self.assertEqual(defaults["bootstrapTotalMaxChars"], 4096)
        self.assertEqual(defaults["experimental"], {"localModelLean": True})
        self.assertEqual(params["maxTokens"], 32)
        self.assertEqual(params["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(params["extra_body"]["max_completion_tokens"], 32)
        self.assertEqual(params["extra_body"]["chat_template_kwargs"], {"enable_thinking": False})

    def test_qwen36_lean_16k_patch_documents_larger_context_variant(self):
        manifest = json.loads((ROOT / "manifests" / "vllm-qwen36-fp8-lean16k-live.example.json").read_text(encoding="utf-8"))
        provider = json.loads((ROOT / "openclaw-config" / "qwen36-vllm-provider-lean16k.merge.example.json").read_text(encoding="utf-8"))
        model = manifest["models"][0]
        provider_model = provider["models"][0]

        self.assertEqual(provider["baseUrl"], model["api_base"])
        self.assertEqual(provider_model["id"], model["served_model_name"])
        self.assertEqual(provider_model["contextWindow"], max(model["contexts"]))
        self.assertEqual(provider_model["contextTokens"], max(model["contexts"]))
        self.assertEqual(provider_model["maxTokens"], 32)
        self.assertEqual(model["contexts"], [16384])
        self.assertIn("gmu95", model["hardware_profile"])
        self.assertIn("ctx16384-lean-max32", model["hardware_profile"])

    def test_api_openclaw_provider_examples_match_manifest(self):
        manifest = json.loads((ROOT / "manifests" / "api-providers.example.json").read_text(encoding="utf-8"))
        providers = {
            "openai": json.loads((ROOT / "openclaw-config" / "openai-provider.example.json").read_text(encoding="utf-8")),
            "anthropic": json.loads((ROOT / "openclaw-config" / "anthropic-provider.example.json").read_text(encoding="utf-8")),
        }
        models_by_route = {model["served_model_name"]: model for model in manifest["models"]}

        openai = providers["openai"]
        self.assertEqual(openai["api"], "openai-responses")
        self.assertEqual(openai["apiKey"]["id"], models_by_route["openai/gpt-4.1"]["api_env"])
        self.assertEqual(openai["models"][0]["id"], "gpt-4.1")
        self.assertEqual(openai["models"][0]["contextWindow"], max(models_by_route["openai/gpt-4.1"]["contexts"]))

        anthropic = providers["anthropic"]
        self.assertEqual(anthropic["api"], "anthropic-messages")
        self.assertEqual(anthropic["apiKey"]["id"], models_by_route["anthropic/claude-sonnet"]["api_env"])
        self.assertEqual(anthropic["models"][0]["id"], "claude-sonnet")
        self.assertEqual(anthropic["models"][0]["contextWindow"], max(models_by_route["anthropic/claude-sonnet"]["contexts"]))


if __name__ == "__main__":
    unittest.main()
