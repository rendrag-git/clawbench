import json
import unittest
from pathlib import Path

from openclaw_bench.manifest import load_model_specs, load_suite


ROOT = Path(__file__).resolve().parent.parent


class ManifestTests(unittest.TestCase):
    def test_load_core_suite(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        self.assertEqual(suite.suite_id, "openclaw-agent-core")
        self.assertEqual(len(suite.tasks), 9)
        self.assertEqual(suite.tasks[0].task_id, "workspace-discovery")
        context_by_task = {task.task_id: task.context_sizes for task in suite.tasks}
        self.assertEqual(context_by_task["workspace-needle-4k"], [4096])
        self.assertEqual(context_by_task["workspace-needle-8k"], [8192])
        self.assertEqual(context_by_task["workspace-needle-16k"], [16384])
        self.assertEqual(context_by_task["workspace-needle-32k"], [32768])
        self.assertEqual(context_by_task["workspace-needle-64k"], [65536])
        task_by_id = {task.task_id: task for task in suite.tasks}
        self.assertEqual(task_by_id["workspace-needle-64k"].expected["min_fixture_chars"], 65536)
        self.assertTrue(task_by_id["multi-file-bug-trace"].expected["behavior_checks"])
        self.assertTrue(task_by_id["patch-execution"].expected["behavior_checks"])

    def test_load_initial_models(self):
        models = load_model_specs(ROOT / "manifests" / "initial-models.json")
        self.assertEqual({model["model_id"] for model in models}, {"gpt-oss-20b-nvfp4", "qwen3-dense"})

    def test_load_api_provider_examples(self):
        models = load_model_specs(ROOT / "manifests" / "api-providers.example.json")
        self.assertEqual({model["provider_type"] for model in models}, {"api", "subscription"})
        self.assertEqual({model["kv_modes"][0] for model in models}, {"provider_default"})
        self.assertEqual({model["api_env"] for model in models}, {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"})

    def test_load_real_repo_readonly_suite(self):
        suite = load_suite(ROOT / "manifests" / "real-repo-readonly.example.json")
        self.assertEqual(suite.suite_id, "real-repo-readonly-example")
        self.assertEqual(len(suite.tasks), 3)
        self.assertEqual({task.task_type for task in suite.tasks}, {"repo_read_only", "repo_code_edit"})
        self.assertEqual({task.max_changed_files for task in suite.tasks}, {0, 1})
        self.assertEqual({task.fixture for task in suite.tasks}, {"real_repos/kingshot-ams-snapshot"})

    def test_load_full_certification_suite_includes_core_and_real_repo_tasks(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-certification-full.example.json")
        task_types = {task.task_type for task in suite.tasks}

        self.assertEqual(suite.suite_id, "openclaw-certification-full-example")
        self.assertEqual(len(suite.tasks), 12)
        self.assertEqual(
            {path.name for path in suite.source_paths},
            {"openclaw-certification-full.example.json", "openclaw-agent-core.json", "real-repo-readonly.example.json"},
        )
        self.assertIn("workspace_discovery", task_types)
        self.assertIn("repo_read_only", task_types)
        self.assertIn("repo_code_edit", task_types)

    def test_load_vllm_local_candidates(self):
        models = load_model_specs(ROOT / "manifests" / "vllm-local-candidates.example.json")
        self.assertEqual(len(models), 10)
        self.assertIn("RedHatAI/Qwen3-Coder-Next-NVFP4", {model["model_id"] for model in models})
        self.assertTrue(all(model["weight_quant"] == "nvfp4" for model in models))
        self.assertTrue(all(model["health_check_url"] == "http://127.0.0.1:8000/v1/models" for model in models))
        self.assertTrue(all(model["api_env"] == "VLLM_API_KEY" for model in models))
        self.assertTrue(all("--gpu-memory-utilization" in model["serve_command"] for model in models))
        self.assertTrue(all("--max-model-len" in model["serve_command"] for model in models))

    def test_load_vllm_long_context_manifest_matches_serve_max_model_len(self):
        models = load_model_specs(ROOT / "manifests" / "vllm-long-context.example.json")
        self.assertEqual(len(models), 4)
        kv_modes = {model["kv_modes"][0] for model in models}
        self.assertTrue({"fp8", "turboquant_k8v4", "turboquant_k3v4_nc"} <= kv_modes)
        for model in models:
            self.assertEqual(model["contexts"], [4096, 8192, 16384, 32768, 65536])
            index = model["serve_command"].index("--max-model-len")
            self.assertEqual(int(model["serve_command"][index + 1]), max(model["contexts"]))

    def test_load_vllm_concurrency_sweep_manifest(self):
        models = load_model_specs(ROOT / "manifests" / "vllm-concurrency-sweep.example.json")
        self.assertEqual(len(models), 4)
        kv_modes = {model["kv_modes"][0] for model in models}
        self.assertTrue({"fp8", "turboquant_k8v4", "turboquant_k3v4_nc"} <= kv_modes)
        for model in models:
            self.assertEqual(model["contexts"], [4096])
            self.assertEqual(model["concurrency"], [1, 2, 4, 8, 16, 32, 64])

    def test_load_vllm_hardware_setups_manifest(self):
        models = load_model_specs(ROOT / "manifests" / "vllm-hardware-setups.example.json")
        profiles = {model["hardware_profile"] for model in models}
        profiles_by_kv = {}
        memory_utilizations = {
            model["serve_command"][model["serve_command"].index("--gpu-memory-utilization") + 1]
            for model in models
        }
        for model in models:
            profiles_by_kv.setdefault(model["kv_modes"][0], set()).add(model["hardware_profile"])

        self.assertEqual(len(models), 6)
        self.assertEqual(len(profiles), 2)
        self.assertEqual(set(profiles_by_kv), {"fp8", "turboquant_k8v4", "turboquant_k3v4_nc"})
        self.assertTrue(all(len(kv_profiles) == 2 for kv_profiles in profiles_by_kv.values()))
        self.assertEqual(memory_utilizations, {"0.82", "0.9"})
        self.assertTrue(all(model["contexts"] == [4096] for model in models))

    def test_qwen36_live_manifests_keep_full_and_lean_rows_distinct(self):
        full = load_model_specs(ROOT / "manifests" / "vllm-qwen36-fp8-live.example.json")[0]
        lean = load_model_specs(ROOT / "manifests" / "vllm-qwen36-fp8-lean8k-live.example.json")[0]
        lean16k = load_model_specs(ROOT / "manifests" / "vllm-qwen36-fp8-lean16k-live.example.json")[0]

        self.assertEqual(full["served_model_name"], lean["served_model_name"])
        self.assertEqual(full["openclaw_model_name"], lean["openclaw_model_name"])
        self.assertEqual(lean16k["served_model_name"], lean["served_model_name"])
        self.assertEqual(lean16k["openclaw_model_name"], lean["openclaw_model_name"])
        self.assertEqual(full["contexts"], [8192])
        self.assertEqual(lean["contexts"], [8192])
        self.assertEqual(lean16k["contexts"], [16384])
        self.assertNotEqual(full["hardware_profile"], lean["hardware_profile"])
        self.assertNotEqual(lean["hardware_profile"], lean16k["hardware_profile"])
        self.assertIn("max128", full["hardware_profile"])
        self.assertIn("lean-max32", lean["hardware_profile"])
        self.assertIn("ctx16384-lean-max32", lean16k["hardware_profile"])

    def test_manifest_set_covers_certification_axes(self):
        manifest_names = [
            "api-providers.example.json",
            "vllm-gptoss-smoke.example.json",
            "vllm-local.example.json",
            "vllm-local-candidates.example.json",
            "vllm-long-context.example.json",
            "vllm-concurrency-sweep.example.json",
            "vllm-hardware-setups.example.json",
        ]
        models = [
            model
            for name in manifest_names
            for model in load_model_specs(ROOT / "manifests" / name)
        ]
        provider_types = {model["provider_type"] for model in models}
        local_setups = {
            (model["weight_quant"], kv)
            for model in models
            if model["provider_type"] == "local"
            for kv in model.get("kv_modes", [])
        }
        contexts = {
            int(context)
            for model in models
            for context in model.get("contexts", [])
        }
        concurrency = {
            int(level)
            for model in models
            for level in model.get("concurrency", [])
        }
        hardware_profiles = {
            model["hardware_profile"]
            for model in models
            if model["provider_type"] == "local" and "hardware_profile" in model
        }

        self.assertIn("local", provider_types)
        self.assertIn("api", provider_types)
        self.assertIn("subscription", provider_types)
        self.assertIn(("nvfp4", "fp8"), local_setups)
        self.assertIn(("nvfp4", "turboquant_k8v4"), local_setups)
        self.assertIn(("nvfp4", "turboquant_k3v4_nc"), local_setups)
        self.assertTrue({4096, 8192, 16384, 32768, 65536} <= contexts)
        self.assertTrue({1, 2, 4, 8, 16, 32, 64} <= concurrency)
        self.assertGreaterEqual(len(hardware_profiles), 2)

    def test_host_specific_vllm_manifests_are_marked(self):
        for path in (ROOT / "manifests").glob("vllm-*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            marker = payload.get("manifest_scope", {})
            serialized = path.read_text(encoding="utf-8")
            has_host_specific_value = any(
                value in serialized
                for value in ("/home/ubuntu", "CUDA_VISIBLE_DEVICES", "10.68.198.1", "127.0.0.1:8000")
            )

            if has_host_specific_value:
                self.assertEqual(marker.get("portability"), "host_specific", path.name)
                self.assertIn("edit", marker.get("notes", "").lower(), path.name)


if __name__ == "__main__":
    unittest.main()
