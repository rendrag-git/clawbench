import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from openclaw_bench.certification import REQUIRED_TASK_TYPES, certify_run_dirs, render_certification_text


ROOT = Path(__file__).resolve().parent.parent


class CertificationTests(unittest.TestCase):
    def test_complete_multirun_certification_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_rows = [
                _attempt(task_type, provider_type="local", kv_cache_dtype="fp8", context_limit=4096, concurrency=1)
                for task_type in sorted(REQUIRED_TASK_TYPES)
            ]
            for kv_mode in ("fp8", "turboquant_k8v4", "turboquant_k3v4_nc"):
                local_rows.append(_attempt("patch_execution", provider_type="local", kv_cache_dtype=kv_mode, context_limit=4096, concurrency=1))
                local_rows.append(_attempt("instruction_retention", provider_type="local", kv_cache_dtype=kv_mode, context_limit=4096, concurrency=1))
                for context in (4096, 8192, 16384, 32768, 65536):
                    local_rows.append(_attempt("workspace_needle", provider_type="local", kv_cache_dtype=kv_mode, context_limit=context, concurrency=1))
                for concurrency in (1, 2, 4, 8, 16, 32, 64):
                    local_rows.append(_attempt("workspace_discovery", provider_type="local", kv_cache_dtype=kv_mode, context_limit=4096, concurrency=concurrency))
            for concurrency in (1, 2, 4, 8, 16, 32, 64):
                local_rows.append(_attempt("patch_execution", provider_type="local", kv_cache_dtype="fp8", context_limit=4096, concurrency=concurrency))
            for kv_mode in ("fp8", "turboquant_k8v4", "turboquant_k3v4_nc"):
                local_rows.append(_attempt("workspace_discovery", provider_type="local", hardware_profile="rtx-pro-5000-gmu75", kv_cache_dtype=kv_mode, context_limit=4096, concurrency=1))
            api_rows = [
                _attempt(task_type, provider_type="api", weight_quant="provider_default", kv_cache_dtype="provider_default", concurrency=4, served_model_name="api-model")
                for task_type in sorted(REQUIRED_TASK_TYPES)
            ]
            api_rows.extend(
                _attempt("workspace_discovery", provider_type="api", weight_quant="provider_default", kv_cache_dtype="provider_default", context_limit=context, concurrency=4, served_model_name="api-model")
                for context in (8192, 32768)
            )
            api_rows.extend(
                _attempt("workspace_discovery", provider_type="api", weight_quant="provider_default", kv_cache_dtype="provider_default", context_limit=4096, concurrency=concurrency, served_model_name="api-model")
                for concurrency in (1, 16)
            )
            subscription_rows = [
                _attempt(task_type, provider_type="subscription", weight_quant="provider_default", kv_cache_dtype="provider_default", concurrency=4, served_model_name="subscription-model")
                for task_type in sorted(REQUIRED_TASK_TYPES)
            ]
            subscription_rows.extend(
                _attempt("workspace_discovery", provider_type="subscription", weight_quant="provider_default", kv_cache_dtype="provider_default", context_limit=context, concurrency=4, served_model_name="subscription-model")
                for context in (8192, 32768)
            )
            subscription_rows.extend(
                _attempt("workspace_discovery", provider_type="subscription", weight_quant="provider_default", kv_cache_dtype="provider_default", context_limit=4096, concurrency=concurrency, served_model_name="subscription-model")
                for concurrency in (1, 16)
            )
            local_dir = _write_run_dir(root / "local-live", local_rows, backend="openclaw")
            api_dir = _write_run_dir(root / "api-live", api_rows, backend="openclaw")
            subscription_dir = _write_run_dir(root / "subscription-live", subscription_rows, backend="openclaw")

            result = certify_run_dirs([local_dir, api_dir, subscription_dir])

            self.assertTrue(result.ok, result.to_json())
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["live_backend"].status, "pass")
            self.assertEqual(checks["local_setup_exploration"].status, "pass")
            self.assertEqual(checks["local_hardware_setup_exploration"].status, "pass")
            self.assertEqual(checks["local_hardware_setup_pairing"].status, "pass")
            self.assertEqual(checks["required_task_types_passed"].status, "pass")
            self.assertEqual(checks["local_required_task_types_passed"].status, "pass")
            self.assertEqual(checks["context_sweep"].status, "pass")
            self.assertEqual(checks["local_context_ceiling"].status, "pass")
            self.assertEqual(checks["local_setup_context_sweep"].status, "pass")
            self.assertEqual(checks["local_setup_representative_tasks"].status, "pass")
            self.assertEqual(checks["local_fp8_pairing"].status, "pass")
            self.assertEqual(checks["concurrency_sweep"].status, "pass")
            self.assertEqual(checks["local_setup_concurrency_sweep"].status, "pass")
            self.assertEqual(checks["local_concurrency_representative_tasks"].status, "pass")
            self.assertEqual(checks["api_or_subscription_task_success"].status, "pass")
            self.assertEqual(checks["api_or_subscription_required_task_types_passed"].status, "pass")
            self.assertEqual(checks["api_or_subscription_context_coverage"].status, "pass")
            self.assertEqual(checks["api_or_subscription_concurrency_coverage"].status, "pass")

    def test_simulator_only_run_fails_certification(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_attempt("workspace_discovery", backend="simulator", provider_type="local")]
            run_dir = _write_run_dir(Path(tmp) / "simulator", rows, backend="simulator")

            result = certify_run_dirs([run_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["live_backend"].status, "fail")
            self.assertEqual(checks["api_or_subscription_rows"].status, "fail")
            self.assertEqual(checks["required_task_types"].status, "fail")

    def test_simulator_rows_cannot_backfill_live_certification_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live_local_rows = [
                _attempt("workspace_discovery", provider_type="local", kv_cache_dtype="fp8", context_limit=4096, concurrency=1)
            ]
            simulator_backfill_rows = [
                _attempt(task_type, backend="simulator", provider_type="local", kv_cache_dtype="fp8", context_limit=4096, concurrency=1)
                for task_type in sorted(REQUIRED_TASK_TYPES)
            ]
            for kv_mode in ("fp8", "turboquant_k8v4", "turboquant_k3v4_nc"):
                for context in (4096, 8192, 16384, 32768, 65536):
                    simulator_backfill_rows.append(
                        _attempt("workspace_needle", backend="simulator", provider_type="local", kv_cache_dtype=kv_mode, context_limit=context, concurrency=1)
                    )
                for concurrency in (1, 2, 4, 8, 16, 32, 64):
                    simulator_backfill_rows.append(
                        _attempt("workspace_discovery", backend="simulator", provider_type="local", kv_cache_dtype=kv_mode, context_limit=4096, concurrency=concurrency)
                    )
            api_rows = [
                _attempt(task_type, provider_type="api", weight_quant="provider_default", kv_cache_dtype="provider_default", concurrency=4, served_model_name="api-model")
                for task_type in sorted(REQUIRED_TASK_TYPES)
            ]
            subscription_rows = [
                _attempt(task_type, provider_type="subscription", weight_quant="provider_default", kv_cache_dtype="provider_default", concurrency=4, served_model_name="subscription-model")
                for task_type in sorted(REQUIRED_TASK_TYPES)
            ]
            local_dir = _write_run_dir(root / "local-live", live_local_rows, backend="openclaw")
            simulator_dir = _write_run_dir(root / "simulator-backfill", simulator_backfill_rows, backend="simulator")
            api_dir = _write_run_dir(root / "api-live", api_rows, backend="openclaw")
            subscription_dir = _write_run_dir(root / "subscription-live", subscription_rows, backend="openclaw")

            result = certify_run_dirs([local_dir, simulator_dir, api_dir, subscription_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["live_backend"].status, "pass")
            self.assertEqual(checks["local_setup_exploration"].status, "fail")
            self.assertEqual(checks["local_required_task_types_passed"].status, "fail")
            self.assertEqual(checks["context_sweep"].status, "fail")
            self.assertEqual(checks["local_context_ceiling"].status, "fail")
            self.assertEqual(checks["concurrency_sweep"].status, "fail")

    def test_certification_surfaces_local_context_ceiling_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                _attempt("workspace_needle", provider_type="local", kv_cache_dtype="fp8", context_limit=8192),
            ]
            run_dir = _write_run_dir(Path(tmp) / "local-8k-only", rows, backend="openclaw")

            result = certify_run_dirs([run_dir])
            checks = {check.name: check for check in result.checks}

            self.assertEqual(checks["local_context_ceiling"].status, "fail")
            self.assertIn("maximum local context_limit=8192", checks["local_context_ceiling"].notes)
            self.assertIn("8k-only vLLM endpoint cannot certify", checks["local_context_ceiling"].notes)

    def test_certification_requires_hardware_and_throughput_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_rows = [
                _attempt(task_type, provider_type="local", kv_cache_dtype="fp8", context_limit=4096, concurrency=1)
                for task_type in sorted(REQUIRED_TASK_TYPES)
            ]
            for kv_mode in ("fp8", "turboquant_k8v4", "turboquant_k3v4_nc"):
                for context in (4096, 8192, 16384, 32768, 65536):
                    local_rows.append(_attempt("workspace_needle", provider_type="local", kv_cache_dtype=kv_mode, context_limit=context, concurrency=1))
                for concurrency in (1, 2, 4, 8, 16, 32, 64):
                    local_rows.append(_attempt("workspace_discovery", provider_type="local", kv_cache_dtype=kv_mode, context_limit=4096, concurrency=concurrency))
            for kv_mode in ("fp8", "turboquant_k8v4", "turboquant_k3v4_nc"):
                local_rows.append(_attempt("workspace_discovery", provider_type="local", hardware_profile="rtx-pro-5000-gmu75", kv_cache_dtype=kv_mode, context_limit=4096, concurrency=1))
            api_rows = [
                _attempt(task_type, provider_type="api", weight_quant="provider_default", kv_cache_dtype="provider_default", concurrency=4, served_model_name="api-model")
                for task_type in sorted(REQUIRED_TASK_TYPES)
            ]
            subscription_rows = [
                _attempt(task_type, provider_type="subscription", weight_quant="provider_default", kv_cache_dtype="provider_default", concurrency=4, served_model_name="subscription-model")
                for task_type in sorted(REQUIRED_TASK_TYPES)
            ]
            local_dir = _write_run_dir(root / "local-live", local_rows, backend="openclaw")
            api_dir = _write_run_dir(root / "api-live", api_rows, backend="openclaw")
            subscription_dir = _write_run_dir(root / "subscription-live", subscription_rows, backend="openclaw")
            server_path = local_dir / "server.json"
            server_payload = json.loads(server_path.read_text(encoding="utf-8"))
            server_payload.pop("hardware")
            server_payload["throughput_probes"] = []
            _write_json(server_path, server_payload)

            result = certify_run_dirs([local_dir, api_dir, subscription_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["hardware_inventory"].status, "fail")
            self.assertEqual(checks["throughput_probe_evidence"].status, "fail")

    def test_certification_rejects_thin_throughput_probe_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_attempt("workspace_discovery", provider_type="local", kv_cache_dtype="fp8", context_limit=4096)]
            run_dir = _write_run_dir(Path(tmp) / "thin-throughput", rows, backend="openclaw")
            server_path = run_dir / "server.json"
            server_payload = json.loads(server_path.read_text(encoding="utf-8"))
            server_payload["throughput_probes"] = [
                {
                    "model": "model-a",
                    "provider_type": "local",
                    "hardware_profile": "rtx-pro-5000-gmu90",
                    "weight_quant": "nvfp4",
                    "kv_cache_dtype": "fp8",
                    "context_limit": 4096,
                    "tokens_per_s": 42.0,
                }
            ]
            _write_json(server_path, server_payload)

            result = certify_run_dirs([run_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["throughput_probe_evidence"].status, "fail")
            self.assertIn("invalid", checks["throughput_probe_evidence"].notes)

    def test_certification_requires_route_probe_for_each_passing_model_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                _attempt("workspace_discovery", provider_type="local", kv_cache_dtype="fp8", context_limit=4096),
                _attempt("workspace_discovery", provider_type="local", kv_cache_dtype="turboquant_k8v4", context_limit=4096),
            ]
            run_dir = _write_run_dir(Path(tmp) / "missing-cell-route", rows, backend="openclaw")
            server_path = run_dir / "server.json"
            server_payload = json.loads(server_path.read_text(encoding="utf-8"))
            server_payload["serve_results"] = [
                result
                for result in server_payload["serve_results"]
                if result.get("kv_cache_dtype") != "turboquant_k8v4"
            ]
            server_payload["throughput_probes"] = [
                probe
                for probe in server_payload["throughput_probes"]
                if probe.get("kv_cache_dtype") != "turboquant_k8v4"
            ]
            _write_json(server_path, server_payload)

            result = certify_run_dirs([run_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["server_model_cell_evidence"].status, "pass")
            self.assertEqual(checks["route_probe_cell_evidence"].status, "fail")
            self.assertIn("turboquant_k8v4", checks["route_probe_cell_evidence"].notes)

    def test_route_probe_cell_evidence_does_not_require_direct_probe_for_external_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                _attempt("workspace_discovery", provider_type="local", kv_cache_dtype="fp8", context_limit=4096),
                _attempt(
                    "workspace_discovery",
                    provider_type="api",
                    weight_quant="provider_default",
                    kv_cache_dtype="provider_default",
                    served_model_name="api-model",
                ),
            ]
            run_dir = _write_run_dir(Path(tmp) / "external-openclaw-only-route", rows, backend="openclaw")
            server_path = run_dir / "server.json"
            server_payload = json.loads(server_path.read_text(encoding="utf-8"))
            for result in server_payload["serve_results"]:
                if result.get("provider_type") == "api":
                    result["route_probe"] = {"openclaw_route": {"success": True}}
            server_payload["throughput_probes"] = [
                probe
                for probe in server_payload["throughput_probes"]
                if probe.get("provider_type") != "api"
            ]
            _write_json(server_path, server_payload)

            result = certify_run_dirs([run_dir])

            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["route_probe_cell_evidence"].status, "pass")
            self.assertEqual(checks["api_or_subscription_route_probes"].status, "pass")

    def test_certification_requires_server_model_artifact_for_each_live_model_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                _attempt("workspace_discovery", provider_type="local", kv_cache_dtype="fp8", context_limit=4096),
                _attempt("workspace_discovery", provider_type="local", kv_cache_dtype="turboquant_k8v4", context_limit=4096),
            ]
            run_dir = _write_run_dir(Path(tmp) / "missing-server-cell", rows, backend="openclaw")
            server_path = run_dir / "server.json"
            server_payload = json.loads(server_path.read_text(encoding="utf-8"))
            server_payload["models"] = [
                model
                for model in server_payload["models"]
                if model.get("kv_cache_dtype") != "turboquant_k8v4"
            ]
            _write_json(server_path, server_payload)

            result = certify_run_dirs([run_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["server_model_cell_evidence"].status, "fail")
            self.assertIn("turboquant_k8v4", checks["server_model_cell_evidence"].notes)

    def test_certification_requires_multiple_local_hardware_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_rows = [
                _attempt(task_type, provider_type="local", kv_cache_dtype="fp8", context_limit=4096, concurrency=1)
                for task_type in sorted(REQUIRED_TASK_TYPES)
            ]
            for kv_mode in ("fp8", "turboquant_k8v4", "turboquant_k3v4_nc"):
                for context in (4096, 8192, 16384, 32768, 65536):
                    local_rows.append(_attempt("workspace_needle", provider_type="local", kv_cache_dtype=kv_mode, context_limit=context, concurrency=1))
                for concurrency in (1, 2, 4, 8, 16, 32, 64):
                    local_rows.append(_attempt("workspace_discovery", provider_type="local", kv_cache_dtype=kv_mode, context_limit=4096, concurrency=concurrency))
            api_rows = [
                _attempt(task_type, provider_type="api", weight_quant="provider_default", kv_cache_dtype="provider_default", concurrency=4, served_model_name="api-model")
                for task_type in sorted(REQUIRED_TASK_TYPES)
            ]
            subscription_rows = [
                _attempt(task_type, provider_type="subscription", weight_quant="provider_default", kv_cache_dtype="provider_default", concurrency=4, served_model_name="subscription-model")
                for task_type in sorted(REQUIRED_TASK_TYPES)
            ]
            local_dir = _write_run_dir(root / "local-live", local_rows, backend="openclaw")
            api_dir = _write_run_dir(root / "api-live", api_rows, backend="openclaw")
            subscription_dir = _write_run_dir(root / "subscription-live", subscription_rows, backend="openclaw")

            result = certify_run_dirs([local_dir, api_dir, subscription_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["local_hardware_setup_exploration"].status, "fail")
            self.assertIn("need at least 2", checks["local_hardware_setup_exploration"].notes)

    def test_certification_requires_same_cell_hardware_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                _attempt("workspace_discovery", provider_type="local", hardware_profile="profile-a", comparison_id="model-a", kv_cache_dtype="fp8"),
                _attempt("workspace_discovery", provider_type="local", hardware_profile="profile-b", comparison_id="model-b", kv_cache_dtype="fp8"),
            ]
            run_dir = _write_run_dir(Path(tmp) / "unpaired-hardware", rows, backend="openclaw")

            result = certify_run_dirs([run_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["local_hardware_setup_exploration"].status, "pass")
            self.assertEqual(checks["local_hardware_setup_pairing"].status, "fail")
            self.assertIn("no same model", checks["local_hardware_setup_pairing"].notes)

    def test_certification_requires_hardware_pairing_for_each_required_local_kv_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                _attempt("workspace_discovery", provider_type="local", hardware_profile="profile-a", kv_cache_dtype="fp8"),
                _attempt("workspace_discovery", provider_type="local", hardware_profile="profile-b", kv_cache_dtype="fp8"),
                _attempt("workspace_discovery", provider_type="local", hardware_profile="profile-a", kv_cache_dtype="turboquant_k8v4"),
                _attempt("workspace_discovery", provider_type="local", hardware_profile="profile-a", kv_cache_dtype="turboquant_k3v4_nc"),
            ]
            run_dir = _write_run_dir(Path(tmp) / "partially-paired-hardware", rows, backend="openclaw")

            result = certify_run_dirs([run_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["local_hardware_setup_exploration"].status, "pass")
            self.assertEqual(checks["local_hardware_setup_pairing"].status, "fail")
            self.assertIn("turboquant_k8v4", checks["local_hardware_setup_pairing"].notes)
            self.assertIn("turboquant_k3v4_nc", checks["local_hardware_setup_pairing"].notes)

    def test_certification_requires_representative_tasks_for_each_local_kv_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                _attempt("patch_execution", provider_type="local", kv_cache_dtype="fp8"),
                _attempt("instruction_retention", provider_type="local", kv_cache_dtype="fp8"),
                _attempt("workspace_discovery", provider_type="local", kv_cache_dtype="turboquant_k8v4"),
                _attempt("workspace_needle", provider_type="local", kv_cache_dtype="turboquant_k3v4_nc"),
            ]
            run_dir = _write_run_dir(Path(tmp) / "nonrepresentative-kv", rows, backend="openclaw")

            result = certify_run_dirs([run_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["local_setup_representative_tasks"].status, "fail")
            self.assertIn("turboquant_k8v4:instruction_retention, patch_execution", checks["local_setup_representative_tasks"].notes)
            self.assertIn("turboquant_k3v4_nc:instruction_retention, patch_execution", checks["local_setup_representative_tasks"].notes)

    def test_certification_requires_representative_tasks_at_each_concurrency_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                _attempt("workspace_discovery", provider_type="local", kv_cache_dtype="fp8", concurrency=level)
                for level in (1, 2, 4, 8, 16, 32, 64)
            ]
            rows.append(_attempt("patch_execution", provider_type="local", kv_cache_dtype="fp8", concurrency=1))
            run_dir = _write_run_dir(Path(tmp) / "nonrepresentative-concurrency", rows, backend="openclaw")

            result = certify_run_dirs([run_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["concurrency_sweep"].status, "pass")
            self.assertEqual(checks["local_concurrency_representative_tasks"].status, "fail")
            self.assertIn("2, 4, 8, 16, 32, 64", checks["local_concurrency_representative_tasks"].notes)

    def test_certification_fails_without_both_api_and_subscription_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_rows = [
                _attempt(task_type, provider_type="local", kv_cache_dtype="fp8", context_limit=4096, concurrency=1)
                for task_type in sorted(REQUIRED_TASK_TYPES)
            ]
            local_rows.append(_attempt("workspace_discovery", provider_type="local", kv_cache_dtype="turboquant_k8v4", concurrency=4))
            local_rows.append(_attempt("workspace_needle", provider_type="local", context_limit=32768, concurrency=64))
            api_rows = [
                _attempt("workspace_discovery", provider_type="api", weight_quant="provider_default", kv_cache_dtype="provider_default", concurrency=4, served_model_name="api-model")
            ]
            local_dir = _write_run_dir(root / "local-live", local_rows, backend="openclaw")
            api_dir = _write_run_dir(root / "api-live", api_rows, backend="openclaw")

            result = certify_run_dirs([local_dir, api_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["api_or_subscription_rows"].status, "fail")
            self.assertEqual(checks["api_or_subscription_rows"].notes, "missing subscription")
            self.assertEqual(checks["api_or_subscription_task_success"].status, "fail")

    def test_certification_requires_external_context_and_concurrency_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                _attempt("workspace_discovery", provider_type="api", weight_quant="provider_default", kv_cache_dtype="provider_default", context_limit=4096, concurrency=4, served_model_name="api-model"),
                _attempt("workspace_discovery", provider_type="subscription", weight_quant="provider_default", kv_cache_dtype="provider_default", context_limit=4096, concurrency=4, served_model_name="subscription-model"),
            ]
            run_dir = _write_run_dir(Path(tmp) / "thin-external-coverage", rows, backend="openclaw")

            result = certify_run_dirs([run_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["api_or_subscription_context_coverage"].status, "fail")
            self.assertEqual(checks["api_or_subscription_concurrency_coverage"].status, "fail")
            self.assertIn("api:8192,32768", checks["api_or_subscription_context_coverage"].notes)
            self.assertIn("subscription:1,16", checks["api_or_subscription_concurrency_coverage"].notes)

    def test_certification_requires_local_rows_for_context_and_concurrency_sweeps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_rows = [
                _attempt(task_type, provider_type="local", kv_cache_dtype="fp8", context_limit=4096, concurrency=1)
                for task_type in sorted(REQUIRED_TASK_TYPES)
            ]
            local_rows.append(_attempt("workspace_discovery", provider_type="local", kv_cache_dtype="turboquant_k8v4", context_limit=4096, concurrency=4))
            local_rows.append(_attempt("workspace_discovery", provider_type="local", kv_cache_dtype="turboquant_k3v4_nc", context_limit=4096, concurrency=4))
            api_rows = [
                _attempt(
                    "workspace_discovery",
                    provider_type="api",
                    weight_quant="provider_default",
                    kv_cache_dtype="provider_default",
                    context_limit=context,
                    concurrency=concurrency,
                    served_model_name="api-model",
                )
                for context in (4096, 8192, 16384, 32768, 65536)
                for concurrency in (1, 2, 4, 8, 16, 32, 64)
            ]
            subscription_rows = [
                _attempt(
                    "workspace_discovery",
                    provider_type="subscription",
                    weight_quant="provider_default",
                    kv_cache_dtype="provider_default",
                    context_limit=context,
                    concurrency=concurrency,
                    served_model_name="subscription-model",
                )
                for context in (4096, 8192, 16384, 32768, 65536)
                for concurrency in (1, 2, 4, 8, 16, 32, 64)
            ]
            local_dir = _write_run_dir(root / "local-live", local_rows, backend="openclaw")
            api_dir = _write_run_dir(root / "api-live", api_rows, backend="openclaw")
            subscription_dir = _write_run_dir(root / "subscription-live", subscription_rows, backend="openclaw")

            result = certify_run_dirs([local_dir, api_dir, subscription_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["context_sweep"].status, "fail")
            self.assertEqual(checks["context_sweep"].notes, "missing 8192, 16384, 32768, 65536")
            self.assertEqual(checks["concurrency_sweep"].status, "fail")
            self.assertEqual(checks["concurrency_sweep"].notes, "missing 2, 8, 16, 32, 64")

    def test_certification_requires_context_sweep_to_use_passing_needle_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            local_rows = [
                _attempt(task_type, provider_type="local", kv_cache_dtype="fp8", context_limit=4096, concurrency=1)
                for task_type in sorted(REQUIRED_TASK_TYPES)
            ]
            for context in (8192, 16384, 32768, 65536):
                local_rows.append(_attempt("workspace_discovery", provider_type="local", kv_cache_dtype="fp8", context_limit=context, concurrency=1))
            run_dir = _write_run_dir(Path(tmp) / "discovery-context-backfill", local_rows, backend="openclaw")

            result = certify_run_dirs([run_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["baseline_context"].status, "pass")
            self.assertEqual(checks["context_sweep"].status, "fail")
            self.assertEqual(checks["context_sweep"].notes, "missing 8192, 16384, 32768, 65536")
            self.assertIn("fp8:8192,16384,32768,65536", checks["local_setup_context_sweep"].notes)

    def test_certification_requires_concurrency_sweep_to_use_passing_local_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            local_rows = [
                _attempt(task_type, provider_type="local", kv_cache_dtype="fp8", context_limit=4096, concurrency=1)
                for task_type in sorted(REQUIRED_TASK_TYPES)
            ]
            for kv_mode in ("fp8", "turboquant_k8v4", "turboquant_k3v4_nc"):
                for context in (4096, 8192, 16384, 32768, 65536):
                    local_rows.append(_attempt("workspace_needle", provider_type="local", kv_cache_dtype=kv_mode, context_limit=context, concurrency=1))
                for concurrency in (2, 4, 8, 16, 32, 64):
                    local_rows.append(
                        _attempt(
                            "workspace_discovery",
                            provider_type="local",
                            kv_cache_dtype=kv_mode,
                            context_limit=4096,
                            concurrency=concurrency,
                            status="fail",
                        )
                    )
            api_rows = [
                _attempt(task_type, provider_type="api", weight_quant="provider_default", kv_cache_dtype="provider_default", concurrency=4, served_model_name="api-model")
                for task_type in sorted(REQUIRED_TASK_TYPES)
            ]
            subscription_rows = [
                _attempt(
                    task_type,
                    provider_type="subscription",
                    weight_quant="provider_default",
                    kv_cache_dtype="provider_default",
                    concurrency=4,
                    served_model_name="subscription-model",
                )
                for task_type in sorted(REQUIRED_TASK_TYPES)
            ]
            local_dir = _write_run_dir(Path(tmp) / "failed-concurrency-backfill-local", local_rows, backend="openclaw")
            api_dir = _write_run_dir(Path(tmp) / "failed-concurrency-backfill-api", api_rows, backend="openclaw")
            subscription_dir = _write_run_dir(Path(tmp) / "failed-concurrency-backfill-subscription", subscription_rows, backend="openclaw")

            result = certify_run_dirs([local_dir, api_dir, subscription_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["context_sweep"].status, "pass")
            self.assertEqual(checks["single_and_pool_concurrency"].status, "fail")
            self.assertEqual(checks["concurrency_sweep"].status, "fail")
            self.assertEqual(checks["stress_concurrency"].status, "fail")
            self.assertIn("fp8:2,4,8,16,32,64", checks["local_setup_concurrency_sweep"].notes)

    def test_certification_fails_incomplete_context_or_concurrency_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_rows = [
                _attempt(task_type, provider_type="local", kv_cache_dtype="fp8", context_limit=4096, concurrency=1)
                for task_type in sorted(REQUIRED_TASK_TYPES)
            ]
            local_rows.append(_attempt("workspace_discovery", provider_type="local", kv_cache_dtype="turboquant_k8v4", context_limit=4096, concurrency=4))
            local_rows.append(_attempt("workspace_needle", provider_type="local", kv_cache_dtype="fp8", context_limit=32768, concurrency=64))
            api_rows = [
                _attempt("workspace_discovery", provider_type="api", weight_quant="provider_default", kv_cache_dtype="provider_default", concurrency=4, served_model_name="api-model")
            ]
            subscription_rows = [
                _attempt("workspace_discovery", provider_type="subscription", weight_quant="provider_default", kv_cache_dtype="provider_default", concurrency=4, served_model_name="subscription-model")
            ]
            local_dir = _write_run_dir(root / "local-live", local_rows, backend="openclaw")
            api_dir = _write_run_dir(root / "api-live", api_rows, backend="openclaw")
            subscription_dir = _write_run_dir(root / "subscription-live", subscription_rows, backend="openclaw")

            result = certify_run_dirs([local_dir, api_dir, subscription_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["context_sweep"].status, "fail")
            self.assertEqual(checks["context_sweep"].notes, "missing 8192, 16384, 65536")
            self.assertEqual(checks["concurrency_sweep"].status, "fail")
            self.assertEqual(checks["concurrency_sweep"].notes, "missing 2, 8, 16, 32")

    def test_certification_fails_when_local_kv_setup_lacks_full_sweeps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_rows = [
                _attempt(task_type, provider_type="local", kv_cache_dtype="fp8", context_limit=4096, concurrency=1)
                for task_type in sorted(REQUIRED_TASK_TYPES)
            ]
            for context in (8192, 16384, 32768, 65536):
                local_rows.append(_attempt("workspace_needle", provider_type="local", kv_cache_dtype="fp8", context_limit=context, concurrency=1))
            for concurrency in (2, 4, 8, 16, 32, 64):
                local_rows.append(_attempt("workspace_discovery", provider_type="local", kv_cache_dtype="fp8", context_limit=4096, concurrency=concurrency))
            local_rows.append(_attempt("workspace_discovery", provider_type="local", kv_cache_dtype="turboquant_k8v4", context_limit=4096, concurrency=4))
            local_rows.append(_attempt("workspace_discovery", provider_type="local", kv_cache_dtype="turboquant_k3v4_nc", context_limit=4096, concurrency=4))
            api_rows = [
                _attempt("workspace_discovery", provider_type="api", weight_quant="provider_default", kv_cache_dtype="provider_default", concurrency=4, served_model_name="api-model")
            ]
            subscription_rows = [
                _attempt("workspace_discovery", provider_type="subscription", weight_quant="provider_default", kv_cache_dtype="provider_default", concurrency=4, served_model_name="subscription-model")
            ]
            local_dir = _write_run_dir(root / "local-live", local_rows, backend="openclaw")
            api_dir = _write_run_dir(root / "api-live", api_rows, backend="openclaw")
            subscription_dir = _write_run_dir(root / "subscription-live", subscription_rows, backend="openclaw")

            result = certify_run_dirs([local_dir, api_dir, subscription_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["context_sweep"].status, "pass")
            self.assertEqual(checks["concurrency_sweep"].status, "pass")
            self.assertEqual(checks["local_setup_context_sweep"].status, "fail")
            self.assertIn("turboquant_k8v4:4096,8192,16384,32768,65536", checks["local_setup_context_sweep"].notes)
            self.assertEqual(checks["local_setup_concurrency_sweep"].status, "fail")
            self.assertIn("turboquant_k3v4_nc:1,2,8,16,32,64", checks["local_setup_concurrency_sweep"].notes)

    def test_certification_requires_local_passes_for_all_required_task_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_rows = [
                _attempt(task_type, provider_type="local", kv_cache_dtype="fp8", context_limit=4096, concurrency=1)
                for task_type in sorted(REQUIRED_TASK_TYPES - {"repo_read_only"})
            ]
            for kv_mode in ("fp8", "turboquant_k8v4", "turboquant_k3v4_nc"):
                for context in (4096, 8192, 16384, 32768, 65536):
                    local_rows.append(_attempt("workspace_needle", provider_type="local", kv_cache_dtype=kv_mode, context_limit=context, concurrency=1))
                for concurrency in (1, 2, 4, 8, 16, 32, 64):
                    local_rows.append(_attempt("workspace_discovery", provider_type="local", kv_cache_dtype=kv_mode, context_limit=4096, concurrency=concurrency))
            api_rows = [
                _attempt("repo_read_only", provider_type="api", weight_quant="provider_default", kv_cache_dtype="provider_default", concurrency=4, served_model_name="api-model")
            ]
            subscription_rows = [
                _attempt("workspace_discovery", provider_type="subscription", weight_quant="provider_default", kv_cache_dtype="provider_default", concurrency=4, served_model_name="subscription-model")
            ]
            local_dir = _write_run_dir(root / "local-live", local_rows, backend="openclaw")
            api_dir = _write_run_dir(root / "api-live", api_rows, backend="openclaw")
            subscription_dir = _write_run_dir(root / "subscription-live", subscription_rows, backend="openclaw")

            result = certify_run_dirs([local_dir, api_dir, subscription_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["required_task_types_passed"].status, "pass")
            self.assertEqual(checks["local_required_task_types_passed"].status, "fail")
            self.assertEqual(checks["local_required_task_types_passed"].notes, "missing repo_read_only")

    def test_certification_requires_external_passes_for_all_required_task_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_rows = [
                _attempt(task_type, provider_type="local", kv_cache_dtype="fp8", context_limit=4096, concurrency=1)
                for task_type in sorted(REQUIRED_TASK_TYPES)
            ]
            for kv_mode in ("fp8", "turboquant_k8v4", "turboquant_k3v4_nc"):
                for context in (4096, 8192, 16384, 32768, 65536):
                    local_rows.append(_attempt("workspace_needle", provider_type="local", kv_cache_dtype=kv_mode, context_limit=context, concurrency=1))
                for concurrency in (1, 2, 4, 8, 16, 32, 64):
                    local_rows.append(_attempt("workspace_discovery", provider_type="local", kv_cache_dtype=kv_mode, context_limit=4096, concurrency=concurrency))
            api_rows = [
                _attempt("workspace_discovery", provider_type="api", weight_quant="provider_default", kv_cache_dtype="provider_default", concurrency=4, served_model_name="api-model")
            ]
            subscription_rows = [
                _attempt("workspace_discovery", provider_type="subscription", weight_quant="provider_default", kv_cache_dtype="provider_default", concurrency=4, served_model_name="subscription-model")
            ]
            local_dir = _write_run_dir(root / "local-live", local_rows, backend="openclaw")
            api_dir = _write_run_dir(root / "api-live", api_rows, backend="openclaw")
            subscription_dir = _write_run_dir(root / "subscription-live", subscription_rows, backend="openclaw")

            result = certify_run_dirs([local_dir, api_dir, subscription_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["api_or_subscription_task_success"].status, "pass")
            self.assertEqual(checks["api_or_subscription_required_task_types_passed"].status, "fail")
            self.assertIn("api:instruction_retention", checks["api_or_subscription_required_task_types_passed"].notes)
            self.assertIn("subscription:instruction_retention", checks["api_or_subscription_required_task_types_passed"].notes)

    def test_certification_requires_fp8_baseline_for_non_fp8_local_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_rows = [
                _attempt(task_type, provider_type="local", kv_cache_dtype="fp8", context_limit=4096, concurrency=1, comparison_id="baseline")
                for task_type in sorted(REQUIRED_TASK_TYPES)
            ]
            for context in (4096, 8192, 16384, 32768, 65536):
                local_rows.append(_attempt("workspace_needle", provider_type="local", kv_cache_dtype="fp8", context_limit=context, concurrency=1, comparison_id="baseline"))
            for concurrency in (1, 2, 4, 8, 16, 32, 64):
                local_rows.append(_attempt("workspace_discovery", provider_type="local", kv_cache_dtype="fp8", context_limit=4096, concurrency=concurrency, comparison_id="baseline"))
            for kv_mode in ("turboquant_k8v4", "turboquant_k3v4_nc"):
                for context in (4096, 8192, 16384, 32768, 65536):
                    local_rows.append(_attempt("workspace_needle", provider_type="local", kv_cache_dtype=kv_mode, context_limit=context, concurrency=1, comparison_id="unpaired"))
                for concurrency in (1, 2, 4, 8, 16, 32, 64):
                    local_rows.append(_attempt("workspace_discovery", provider_type="local", kv_cache_dtype=kv_mode, context_limit=4096, concurrency=concurrency, comparison_id="unpaired"))
            api_rows = [
                _attempt("workspace_discovery", provider_type="api", weight_quant="provider_default", kv_cache_dtype="provider_default", concurrency=4, served_model_name="api-model")
            ]
            subscription_rows = [
                _attempt("workspace_discovery", provider_type="subscription", weight_quant="provider_default", kv_cache_dtype="provider_default", concurrency=4, served_model_name="subscription-model")
            ]
            local_dir = _write_run_dir(root / "local-live", local_rows, backend="openclaw")
            api_dir = _write_run_dir(root / "api-live", api_rows, backend="openclaw")
            subscription_dir = _write_run_dir(root / "subscription-live", subscription_rows, backend="openclaw")

            result = certify_run_dirs([local_dir, api_dir, subscription_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["local_setup_context_sweep"].status, "pass")
            self.assertEqual(checks["local_setup_concurrency_sweep"].status, "pass")
            self.assertEqual(checks["local_fp8_pairing"].status, "fail")
            self.assertIn("missing passing fp8 baseline", checks["local_fp8_pairing"].notes)

    def test_certification_requires_fp8_baseline_to_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_rows = [
                _attempt(task_type, provider_type="local", kv_cache_dtype="fp8", context_limit=4096, concurrency=1)
                for task_type in sorted(REQUIRED_TASK_TYPES)
            ]
            for context in (8192, 16384, 32768, 65536):
                local_rows.append(_attempt("workspace_needle", provider_type="local", kv_cache_dtype="fp8", context_limit=context, concurrency=1, status="fail"))
            for concurrency in (2, 4, 8, 16, 32, 64):
                local_rows.append(_attempt("workspace_discovery", provider_type="local", kv_cache_dtype="fp8", context_limit=4096, concurrency=concurrency, status="fail"))
            for kv_mode in ("turboquant_k8v4", "turboquant_k3v4_nc"):
                for context in (4096, 8192, 16384, 32768, 65536):
                    local_rows.append(_attempt("workspace_needle", provider_type="local", kv_cache_dtype=kv_mode, context_limit=context, concurrency=1))
                for concurrency in (1, 2, 4, 8, 16, 32, 64):
                    local_rows.append(_attempt("workspace_discovery", provider_type="local", kv_cache_dtype=kv_mode, context_limit=4096, concurrency=concurrency))
            api_rows = [
                _attempt(task_type, provider_type="api", weight_quant="provider_default", kv_cache_dtype="provider_default", concurrency=4, served_model_name="api-model")
                for task_type in sorted(REQUIRED_TASK_TYPES)
            ]
            subscription_rows = [
                _attempt(task_type, provider_type="subscription", weight_quant="provider_default", kv_cache_dtype="provider_default", concurrency=4, served_model_name="subscription-model")
                for task_type in sorted(REQUIRED_TASK_TYPES)
            ]
            local_dir = _write_run_dir(root / "local-live", local_rows, backend="openclaw")
            api_dir = _write_run_dir(root / "api-live", api_rows, backend="openclaw")
            subscription_dir = _write_run_dir(root / "subscription-live", subscription_rows, backend="openclaw")

            result = certify_run_dirs([local_dir, api_dir, subscription_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["local_setup_context_sweep"].status, "fail")
            self.assertEqual(checks["local_setup_concurrency_sweep"].status, "fail")
            self.assertEqual(checks["local_fp8_pairing"].status, "fail")
            self.assertIn("missing passing fp8 baseline", checks["local_fp8_pairing"].notes)

    def test_certification_fails_when_raw_or_patch_artifacts_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                _attempt(task_type, provider_type="local", kv_cache_dtype="fp8", context_limit=4096, concurrency=1)
                for task_type in sorted(REQUIRED_TASK_TYPES)
            ]
            rows.append(_attempt("workspace_discovery", provider_type="api", weight_quant="provider_default", kv_cache_dtype="provider_default", served_model_name="api-model"))
            rows.append(_attempt("workspace_discovery", provider_type="local", kv_cache_dtype="turboquant_k8v4", concurrency=4))
            rows.append(_attempt("workspace_needle", provider_type="local", context_limit=32768, concurrency=64))
            run_dir = _write_run_dir(root / "incomplete-artifacts", rows, backend="openclaw")
            next((run_dir / "raw").glob("*.json")).unlink()
            next((run_dir / "patches").glob("*.diff")).unlink()

            result = certify_run_dirs([run_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["artifact_count:incomplete-artifacts:raw"].status, "fail")
            self.assertEqual(checks["artifact_count:incomplete-artifacts:patches"].status, "fail")

    def test_certification_fails_when_artifacts_are_not_bound_to_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                _attempt(task_type, provider_type="local", kv_cache_dtype="fp8", context_limit=4096, concurrency=1)
                for task_type in sorted(REQUIRED_TASK_TYPES)
            ]
            for kv_mode in ("fp8", "turboquant_k8v4", "turboquant_k3v4_nc"):
                for context in (4096, 8192, 16384, 32768, 65536):
                    rows.append(_attempt("workspace_needle", provider_type="local", kv_cache_dtype=kv_mode, context_limit=context, concurrency=1))
                for concurrency in (1, 2, 4, 8, 16, 32, 64):
                    rows.append(_attempt("workspace_discovery", provider_type="local", kv_cache_dtype=kv_mode, context_limit=4096, concurrency=concurrency))
            rows.append(_attempt("workspace_discovery", provider_type="local", hardware_profile="rtx-pro-5000-gmu75", kv_cache_dtype="fp8", context_limit=4096, concurrency=1))
            rows.extend(
                _attempt(task_type, provider_type="api", weight_quant="provider_default", kv_cache_dtype="provider_default", concurrency=4, served_model_name="api-model")
                for task_type in sorted(REQUIRED_TASK_TYPES)
            )
            rows.extend(
                _attempt(
                    task_type,
                    provider_type="subscription",
                    weight_quant="provider_default",
                    kv_cache_dtype="provider_default",
                    concurrency=4,
                    served_model_name="subscription-model",
                )
                for task_type in sorted(REQUIRED_TASK_TYPES)
            )
            run_dir = _write_run_dir(root / "stale-artifacts", rows, backend="openclaw")
            first_row = json.loads((run_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()[0])
            (run_dir / "raw" / f"{first_row['workspace_id']}.json").rename(run_dir / "raw" / "stale-raw.json")
            (run_dir / "patches" / f"{first_row['workspace_id']}.diff").rename(run_dir / "patches" / "stale-patch.diff")

            result = certify_run_dirs([run_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["artifact_count:stale-artifacts:raw"].status, "pass")
            self.assertEqual(checks["artifact_binding:stale-artifacts:raw_names"].status, "fail")
            self.assertIn(first_row["workspace_id"], checks["artifact_binding:stale-artifacts:raw_names"].notes)
            self.assertEqual(checks["artifact_binding:stale-artifacts:patch_names"].status, "fail")

    def test_certification_fails_when_raw_artifact_metadata_does_not_match_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_attempt("workspace_discovery", provider_type="local")]
            run_dir = _write_run_dir(Path(tmp) / "swapped-artifact", rows, backend="openclaw")
            row = json.loads((run_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()[0])
            raw_path = run_dir / "raw" / f"{row['workspace_id']}.json"
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            payload["task_type"] = "patch_execution"
            _write_json(raw_path, payload)

            result = certify_run_dirs([run_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["artifact_binding:swapped-artifact:raw_metadata"].status, "fail")
            self.assertIn("task_type", checks["artifact_binding:swapped-artifact:raw_metadata"].notes)

    def test_certification_fails_without_config_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_attempt("workspace_discovery", provider_type="local")]
            run_dir = _write_run_dir(Path(tmp) / "missing-provenance", rows, backend="openclaw")
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            config.pop("provenance")
            _write_json(run_dir / "config.json", config)

            result = certify_run_dirs([run_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["config:missing-provenance:provenance"].status, "fail")
            self.assertIn("missing provenance", checks["config:missing-provenance:provenance"].notes)

    def test_certification_fails_without_input_file_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_attempt("workspace_discovery", provider_type="local")]
            run_dir = _write_run_dir(Path(tmp) / "missing-input-provenance", rows, backend="openclaw")
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            config["provenance"]["input_files"] = []
            _write_json(run_dir / "config.json", config)

            result = certify_run_dirs([run_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["config:missing-input-provenance:provenance"].status, "fail")
            self.assertIn("input_files", checks["config:missing-input-provenance:provenance"].notes)

    def test_certification_fails_when_input_file_roles_are_wrong(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_attempt("workspace_discovery", provider_type="local")]
            run_dir = _write_run_dir(Path(tmp) / "wrong-input-role", rows, backend="openclaw")
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            config["provenance"]["input_files"] = [{"role": "notes", "path": "notes.txt", "digest": _sha256("notes")}]
            _write_json(run_dir / "config.json", config)

            result = certify_run_dirs([run_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["config:wrong-input-role:provenance"].status, "fail")
            self.assertIn("required suite", checks["config:wrong-input-role:provenance"].notes)

    def test_certification_requires_model_config_role_when_declared(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_attempt("workspace_discovery", provider_type="local")]
            run_dir = _write_run_dir(Path(tmp) / "missing-model-config-role", rows, backend="openclaw")
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            config["provenance"]["model_source"] = "model_config"
            _write_json(run_dir / "config.json", config)

            result = certify_run_dirs([run_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["config:missing-model-config-role:provenance"].status, "fail")
            self.assertIn("model_config", checks["config:missing-model-config-role:provenance"].notes)

    def test_certification_fails_without_model_matrix_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_attempt("workspace_discovery", provider_type="local")]
            run_dir = _write_run_dir(Path(tmp) / "missing-model-matrix-digest", rows, backend="openclaw")
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            config["provenance"].pop("model_matrix_digest")
            _write_json(run_dir / "config.json", config)

            result = certify_run_dirs([run_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["config:missing-model-matrix-digest:provenance"].status, "fail")
            self.assertIn("model_matrix_digest", checks["config:missing-model-matrix-digest:provenance"].notes)

    def test_certification_fails_without_runtime_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_attempt("workspace_discovery", provider_type="local")]
            run_dir = _write_run_dir(Path(tmp) / "missing-runtime", rows, backend="openclaw")
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            config.pop("runtime")
            _write_json(run_dir / "config.json", config)

            result = certify_run_dirs([run_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["config:missing-runtime:runtime"].status, "fail")
            self.assertIn("missing runtime", checks["config:missing-runtime:runtime"].notes)

    def test_certification_fails_live_run_without_openclaw_runtime_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_attempt("workspace_discovery", provider_type="local")]
            run_dir = _write_run_dir(Path(tmp) / "missing-openclaw-runtime", rows, backend="openclaw")
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            config["runtime"]["openclaw"] = {"status": "fail", "version": ""}
            _write_json(run_dir / "config.json", config)

            result = certify_run_dirs([run_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["config:missing-openclaw-runtime:runtime"].status, "fail")
            self.assertIn("openclaw status", checks["config:missing-openclaw-runtime:runtime"].notes)

    def test_certification_fails_live_run_without_openclaw_command_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_attempt("workspace_discovery", provider_type="local")]
            run_dir = _write_run_dir(Path(tmp) / "missing-openclaw-command", rows, backend="openclaw")
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            config["runtime"]["openclaw"] = {"status": "pass", "version": "OpenClaw 2026.4.26", "returncode": 0}
            _write_json(run_dir / "config.json", config)

            result = certify_run_dirs([run_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["config:missing-openclaw-command:runtime"].status, "fail")
            self.assertIn("openclaw cmd", checks["config:missing-openclaw-command:runtime"].notes)

    def test_certification_fails_when_gateway_ensure_result_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_attempt("workspace_discovery", provider_type="local")]
            run_dir = _write_run_dir(Path(tmp) / "missing-gateway-ensure", rows, backend="openclaw")
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            config.pop("openclaw_gateway_ensure")
            _write_json(run_dir / "config.json", config)

            result = certify_run_dirs([run_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["config:missing-gateway-ensure:openclaw_gateway_ensure"].status, "fail")
            self.assertIn("missing gateway ensure", checks["config:missing-gateway-ensure:openclaw_gateway_ensure"].notes)

    def test_certification_warns_when_gateway_ensure_was_explicitly_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_attempt("workspace_discovery", provider_type="local")]
            run_dir = _write_run_dir(Path(tmp) / "disabled-gateway-ensure", rows, backend="openclaw")
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            config["ensure_openclaw_gateway"] = False
            config["openclaw_gateway_ensure"] = None
            _write_json(run_dir / "config.json", config)

            result = certify_run_dirs([run_dir])

            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["config:disabled-gateway-ensure:openclaw_gateway_ensure"].status, "warn")

    def test_certification_fails_when_live_raw_response_lacks_openclaw_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_attempt("workspace_discovery", provider_type="local")]
            run_dir = _write_run_dir(Path(tmp) / "missing-live-provenance", rows, backend="openclaw")
            row = json.loads((run_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()[0])
            raw_path = run_dir / "raw" / f"{row['workspace_id']}.json"
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            payload["response"] = {}
            _write_json(raw_path, payload)

            result = certify_run_dirs([run_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["artifact_binding:missing-live-provenance:raw_metadata"].status, "fail")
            self.assertIn("missing OpenClaw command provenance", checks["artifact_binding:missing-live-provenance:raw_metadata"].notes)

    def test_certification_does_not_require_task_artifacts_for_load_level_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                _attempt("workspace_discovery", provider_type="local", failure_type="openclaw_timeout", status="fail", files_read=0, tool_calls=0),
                _attempt("patch_execution", provider_type="local", failure_type="server_timeout", status="fail", files_read=0, tool_calls=0),
                _attempt("repo_read_only", provider_type="local", failure_type="context_window_exceeded", status="fail", files_read=0, tool_calls=0),
                _attempt("workspace_needle", provider_type="local", failure_type="incomplete_result", status="fail", files_read=0, tool_calls=0),
                _attempt("instruction_retention", provider_type="local", failure_type="tool_parser_missing", status="fail", files_read=0, tool_calls=0),
            ]
            run_dir = _write_run_dir(Path(tmp) / "load-failures", rows, backend="openclaw", write_task_artifacts=False)

            result = certify_run_dirs([run_dir])

            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["artifact_count:load-failures:raw"].status, "pass")
            self.assertEqual(checks["artifact_count:load-failures:raw"].notes, "0 artifact(s) for 0 attempt(s)")
            self.assertEqual(checks["artifact_count:load-failures:patches"].status, "pass")

    def test_certification_requires_artifacts_for_attempted_zero_tool_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                _attempt(
                    "workspace_discovery",
                    provider_type="local",
                    failure_type="openclaw_embedded_fallback",
                    status="fail",
                    files_read=0,
                    tool_calls=0,
                    wall_time_s=212.4,
                )
            ]
            run_dir = _write_run_dir(Path(tmp) / "attempted-fallback", rows, backend="openclaw")

            result = certify_run_dirs([run_dir])

            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["artifact_count:attempted-fallback:raw"].status, "pass")
            self.assertEqual(checks["artifact_count:attempted-fallback:raw"].notes, "1 artifact(s) for 1 attempt(s)")
            self.assertEqual(checks["artifact_count:attempted-fallback:patches"].status, "pass")

    def test_certification_fails_when_passing_rows_lack_efficiency_telemetry(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                _attempt(
                    "workspace_discovery",
                    provider_type="local",
                    tool_calls=1,
                    files_read=1,
                    duplicate_file_reads=None,
                    time_to_first_relevant_file_s=None,
                )
            ]
            run_dir = _write_run_dir(Path(tmp) / "missing-efficiency", rows, backend="openclaw")

            result = certify_run_dirs([run_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["tool_file_efficiency_telemetry"].status, "fail")
            self.assertIn("workspace_discovery-case", checks["tool_file_efficiency_telemetry"].notes)

    def test_certification_fails_when_tool_file_efficiency_budget_is_exceeded(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                _attempt("workspace_discovery", provider_type="local", tool_calls=12, files_read=10, duplicate_file_reads=5, time_to_first_relevant_file_s=20.0),
                _attempt("workspace_discovery", provider_type="local", tool_calls=120, files_read=10, duplicate_file_reads=30, time_to_first_relevant_file_s=180.0),
            ]
            run_dir = _write_run_dir(Path(tmp) / "runaway-efficiency", rows, backend="openclaw")

            result = certify_run_dirs([run_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["tool_file_efficiency_telemetry"].status, "pass")
            self.assertEqual(checks["tool_file_efficiency_budget"].status, "fail")
            self.assertIn("local/workspace_discovery", checks["tool_file_efficiency_budget"].notes)
            self.assertIn("dupes=30/20", checks["tool_file_efficiency_budget"].notes)
            self.assertIn("first_file=180/120", checks["tool_file_efficiency_budget"].notes)

    def test_certification_fails_when_configured_context_exceeds_vllm_max_model_len(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_attempt("workspace_needle", provider_type="local", context_limit=32768)]
            run_dir = _write_run_dir(Path(tmp) / "context-mismatch", rows, backend="openclaw")
            _write_json(
                run_dir / "config.json",
                {
                    "backend": "openclaw",
                    "models": [
                        {
                            "provider_type": "local",
                            "context_limit": 32768,
                            "serve_command": ["vllm", "serve", "model", "--max-model-len", "4096"],
                        }
                    ],
                },
            )

            result = certify_run_dirs([run_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["config:context-mismatch:model-0:max_model_len"].status, "fail")

    def test_certification_fails_when_external_rows_lack_external_route_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_rows = [
                _attempt(task_type, provider_type="local", kv_cache_dtype="fp8", context_limit=4096, concurrency=1)
                for task_type in sorted(REQUIRED_TASK_TYPES)
            ]
            local_rows.append(_attempt("workspace_discovery", provider_type="local", kv_cache_dtype="turboquant_k8v4", concurrency=4))
            local_rows.append(_attempt("workspace_needle", provider_type="local", context_limit=32768, concurrency=64))
            api_rows = [_attempt("workspace_discovery", provider_type="api", weight_quant="provider_default", kv_cache_dtype="provider_default", concurrency=4, served_model_name="api-model")]
            subscription_rows = [
                _attempt("workspace_discovery", provider_type="subscription", weight_quant="provider_default", kv_cache_dtype="provider_default", concurrency=4, served_model_name="subscription-model")
            ]
            local_dir = _write_run_dir(root / "local-live", local_rows, backend="openclaw")
            api_dir = _write_run_dir(root / "api-live", api_rows, backend="openclaw", serve_results=[])
            subscription_dir = _write_run_dir(root / "subscription-live", subscription_rows, backend="openclaw")

            result = certify_run_dirs([local_dir, api_dir, subscription_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["route_probes"].status, "pass")
            self.assertEqual(checks["api_or_subscription_route_probes"].status, "fail")
            self.assertEqual(checks["api_or_subscription_route_probes"].notes, "1/2 external model route probe(s)")

    def test_certification_fails_when_direct_probe_false_even_if_openclaw_probe_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_attempt(task_type, provider_type="local", kv_cache_dtype="fp8") for task_type in sorted(REQUIRED_TASK_TYPES)]
            rows.append(_attempt("workspace_discovery", provider_type="api", weight_quant="provider_default", kv_cache_dtype="provider_default", concurrency=4, served_model_name="api-model"))
            rows.append(_attempt("workspace_discovery", provider_type="local", kv_cache_dtype="turboquant_k8v4", concurrency=4))
            rows.append(_attempt("workspace_needle", provider_type="local", context_limit=32768, concurrency=64))
            run_dir = _write_run_dir(
                Path(tmp) / "bad-direct-probe",
                rows,
                backend="openclaw",
                serve_results=[
                    {
                        "model": "model-a",
                        "load_success": True,
                        "route_probe": {
                            "success": False,
                            "openclaw_route": {"success": True},
                        },
                    }
                ],
            )

            result = certify_run_dirs([run_dir])

            self.assertFalse(result.ok)
            checks = {check.name: check for check in result.checks}
            self.assertEqual(checks["route_probes"].status, "fail")
            self.assertEqual(checks["route_probes"].notes, "direct=0/1, openclaw=1/1")

    def test_certify_cli_returns_nonzero_for_incomplete_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_attempt("workspace_discovery", backend="simulator", provider_type="local")]
            run_dir = _write_run_dir(Path(tmp) / "simulator", rows, backend="simulator")

            proc = subprocess.run(
                [sys.executable, "-m", "openclaw_bench", "certify", str(run_dir), "--json"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertFalse(payload["ok"])
            self.assertIn("summary", payload)
            self.assertGreater(payload["summary"]["fail"], 0)
            self.assertIn("live_backend", payload["summary"]["failed_checks"])

    def test_certification_text_can_render_failures_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_attempt("workspace_discovery", backend="simulator", provider_type="local")]
            run_dir = _write_run_dir(Path(tmp) / "simulator", rows, backend="simulator")

            result = certify_run_dirs([run_dir])
            text = render_certification_text(result, failures_only=True)

            self.assertIn("summary=pass:", text)
            self.assertIn("FAIL live_backend", text)
            self.assertNotIn("PASS artifact:", text)

    def test_certification_empty_live_pass_failures_have_clear_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_attempt("workspace_discovery", provider_type="local", status="fail", failure_type="context_window_exceeded", files_read=0, tool_calls=0, wall_time_s=0.0)]
            run_dir = _write_run_dir(Path(tmp) / "no-live-passes", rows, backend="openclaw")

            result = certify_run_dirs([run_dir])
            checks = {check.name: check for check in result.checks}

            self.assertEqual(checks["tool_file_efficiency_telemetry"].notes, "no passing live attempts")
            self.assertEqual(checks["route_probe_cell_evidence"].notes, "no passing local model cells")
            self.assertEqual(checks["local_resource_telemetry"].notes, "no passing local attempts")

    def test_certify_cli_supports_failures_only_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_attempt("workspace_discovery", backend="simulator", provider_type="local")]
            run_dir = _write_run_dir(Path(tmp) / "simulator", rows, backend="simulator")

            proc = subprocess.run(
                [sys.executable, "-m", "openclaw_bench", "certify", str(run_dir), "--failures-only"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            self.assertIn("summary=pass:", proc.stdout)
            self.assertIn("FAIL live_backend", proc.stdout)
            self.assertNotIn("PASS artifact:", proc.stdout)


def _attempt(
    task_type: str,
    backend: str = "openclaw",
    provider_type: str = "local",
    weight_quant: str = "nvfp4",
    kv_cache_dtype: str = "fp8",
    context_limit: int = 4096,
    concurrency: int = 1,
    status: str = "pass",
    failure_type: str | None = None,
    files_read: int = 3,
    tool_calls: int = 1,
    duplicate_file_reads: int | None = 0,
    time_to_first_relevant_file_s: float | None = 0.5,
    served_model_name: str = "model-a",
    comparison_id: str = "model-a",
    hardware_profile: str = "rtx-pro-5000-gmu90",
    wall_time_s: float | None = None,
) -> dict:
    row = {
        "backend": backend,
        "changed_files": [],
        "comparison_id": comparison_id,
        "concurrency": concurrency,
        "context_limit": context_limit,
        "failure_type": failure_type,
        "files_changed": 0,
        "files_read": files_read,
        "duplicate_file_reads": duplicate_file_reads,
        "hallucinated_paths": 0,
        "json_valid": True,
        "kv_cache_dtype": kv_cache_dtype,
        "model": "model-a",
        "hardware_profile": hardware_profile,
        "provider_type": provider_type,
        "score": 1.0 if status == "pass" else 0.0,
        "served_model_name": served_model_name,
        "status": status,
        "task_id": f"{task_type}-case",
        "task_tags": [],
        "task_type": task_type,
        "tests_passed": status == "pass",
        "tool_calls": tool_calls,
        "time_to_first_relevant_file_s": time_to_first_relevant_file_s,
        "weight_quant": weight_quant,
        "peak_vram_mb": 12345.0,
        "gpu_utilization_pct": 55.0,
    }
    if wall_time_s is not None:
        row["wall_time_s"] = wall_time_s
    return row


def _write_run_dir(run_dir: Path, rows: list[dict], backend: str, write_task_artifacts: bool = True, serve_results: list[dict] | None = None) -> Path:
    run_dir.mkdir(parents=True)
    raw_dir = run_dir / "raw"
    patch_dir = run_dir / "patches"
    raw_dir.mkdir()
    patch_dir.mkdir()
    model_artifacts = [
        {
            "served_model_name": row["served_model_name"],
            "provider_type": row["provider_type"],
            "hardware_profile": row.get("hardware_profile", "default"),
            "weight_quant": row["weight_quant"],
            "kv_cache_dtype": row["kv_cache_dtype"],
            "context_limit": row["context_limit"],
        }
        for row in rows
    ]
    _write_json(
        run_dir / "config.json",
        {
            "backend": backend,
            "models": model_artifacts,
            "openclaw_local": False,
            "ensure_openclaw_gateway": backend == "openclaw",
            "openclaw_gateway_ensure": (
                {"name": "openclaw_gateway", "status": "pass", "notes": "synthetic gateway ensured"}
                if backend == "openclaw"
                else None
            ),
            "provenance": _test_provenance(rows),
            "runtime": _test_runtime(backend),
        },
    )
    _write_json(run_dir / "summary.json", {"attempts": len(rows), "pass_rate": 1.0})
    resolved_serve_results = serve_results if serve_results is not None else [
        {
            "model": artifact["served_model_name"],
            "provider_type": artifact["provider_type"],
            "hardware_profile": artifact.get("hardware_profile", "default"),
            "weight_quant": artifact["weight_quant"],
            "kv_cache_dtype": artifact["kv_cache_dtype"],
            "context_limit": artifact["context_limit"],
            "load_success": backend != "simulator",
            "load_time_s": 1.0,
            "peak_vram_mb": 12345.0,
            "gpu_utilization_pct": 55.0,
            "route_probe": {
                "success": backend != "simulator",
                "prompt_chars": 1024,
                "wall_time_s": 0.071,
                "wall_time_p50_s": 0.023,
                "wall_time_p95_s": 0.024,
                "completion_tokens": 3,
                "total_tokens": 128,
                "tokens_per_s": 42.0,
                "tokens_per_s_p50": 42.0,
                "tokens_per_s_p95": 43.0,
                "sample_count": 3,
                "openclaw_route": {"success": backend != "simulator"},
            },
        }
        for artifact in _unique_model_artifacts(model_artifacts)
    ] or [
        {
            "model": "model-a",
            "provider_type": "local",
            "hardware_profile": "default",
            "weight_quant": "unknown",
            "kv_cache_dtype": "fp8",
            "context_limit": 4096,
            "load_success": backend != "simulator",
            "load_time_s": 1.0,
            "route_probe": {"success": backend != "simulator", "openclaw_route": {"success": backend != "simulator"}},
        }
    ]
    _write_json(
        run_dir / "server.json",
        {
            "hardware": {
                "available": True,
                "peak_vram_mb": 12345.0,
                "max_gpu_utilization_pct": 55.0,
                "devices": [{"index": 0, "name": "test-gpu", "memory_total_mb": 49152, "memory_used_mb": 12345, "utilization_pct": 55}],
            },
            "models": model_artifacts,
            "serve_results": resolved_serve_results,
            "throughput_probes": [
                {
                    "model": result.get("model"),
                    "provider_type": result.get("provider_type"),
                    "hardware_profile": result.get("hardware_profile", "default"),
                    "weight_quant": result.get("weight_quant"),
                    "kv_cache_dtype": result.get("kv_cache_dtype"),
                    "context_limit": result.get("context_limit"),
                    "prompt_chars": result.get("route_probe", {}).get("prompt_chars", 1024),
                    "wall_time_s": result.get("route_probe", {}).get("wall_time_s", 0.071),
                    "wall_time_p50_s": result.get("route_probe", {}).get("wall_time_p50_s", 0.023),
                    "wall_time_p95_s": result.get("route_probe", {}).get("wall_time_p95_s", 0.024),
                    "completion_tokens": result.get("route_probe", {}).get("completion_tokens", 3),
                    "total_tokens": result.get("route_probe", {}).get("total_tokens", 128),
                    "tokens_per_s": result.get("route_probe", {}).get("tokens_per_s", 42.0),
                    "tokens_per_s_p50": result.get("route_probe", {}).get("tokens_per_s_p50", 42.0),
                    "tokens_per_s_p95": result.get("route_probe", {}).get("tokens_per_s_p95", 43.0),
                    "sample_count": result.get("route_probe", {}).get("sample_count", 3),
                }
                for result in resolved_serve_results
                if isinstance(result.get("route_probe"), dict) and result["route_probe"].get("success") is True
            ],
        },
    )
    (run_dir / "failures.jsonl").write_text("", encoding="utf-8")
    with (run_dir / "attempts.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            row = {**row, "backend": backend}
            row.setdefault("workspace_id", f"attempt-{index:03d}-{row['task_id']}")
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            if write_task_artifacts:
                _write_json(
                    raw_dir / f"{row['workspace_id']}.json",
                    {
                        "task": row["task_id"],
                        "task_type": row["task_type"],
                        "workspace_id": row["workspace_id"],
                        "model": {
                            "served_model_name": row["served_model_name"],
                            "backend": backend,
                            "provider_type": row["provider_type"],
                            "hardware_profile": row.get("hardware_profile", "default"),
                            "weight_quant": row["weight_quant"],
                            "kv_cache_dtype": row["kv_cache_dtype"],
                            "context_limit": row["context_limit"],
                            "concurrency": row["concurrency"],
                        },
                        "response": _raw_response_for_backend(backend),
                    },
                )
                (patch_dir / f"{row['workspace_id']}.diff").write_text("", encoding="utf-8")
    return run_dir


def _unique_model_artifacts(model_artifacts: list[dict]) -> list[dict]:
    unique = {}
    for artifact in model_artifacts:
        key = (
            artifact["served_model_name"],
            artifact["provider_type"],
            artifact.get("hardware_profile", "default"),
            artifact["kv_cache_dtype"],
            artifact["context_limit"],
        )
        unique[key] = artifact
    return list(unique.values())


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _test_provenance(rows: list[dict]) -> dict:
    task_types = sorted({row["task_type"] for row in rows})
    fixtures = {f"fixture-{task_type}": _sha256(task_type) for task_type in task_types}
    return {
        "schema_version": 1,
        "suite_id": "synthetic-certification-test-suite",
        "suite_digest": _sha256(json.dumps(task_types, sort_keys=True)),
        "model_source": "aliases",
        "model_matrix_digest": _sha256("synthetic-model-matrix"),
        "task_count": max(1, len(task_types)),
        "input_files": [
            {
                "role": "suite",
                "path": "synthetic-suite.json",
                "digest": _sha256("synthetic-suite"),
            }
        ],
        "fixture_digests": fixtures or {"fixture-empty": _sha256("empty")},
    }


def _test_runtime(backend: str) -> dict:
    runtime = {
        "schema_version": 1,
        "python_version": "3.12.0",
        "harness_version": "0.1.0",
    }
    if backend == "openclaw":
        runtime["openclaw"] = {
            "cmd": ["openclaw", "--version"],
            "status": "pass",
            "version": "OpenClaw 2026.4.26",
            "returncode": 0,
        }
    return runtime


def _raw_response_for_backend(backend: str) -> dict:
    if backend == "simulator":
        return {"simulated": True, "session_id": "synthetic"}
    return {"cmd": ["openclaw", "--profile", "bench", "agent"], "returncode": 0, "stdout": "{}", "stderr": ""}


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    unittest.main()
