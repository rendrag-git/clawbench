import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openclaw_bench.manifest import load_suite
from openclaw_bench.models import BackendResponse, ModelSpec
from openclaw_bench.preflight import (
    OPENCLAW_PINNED_VERSION,
    _check_gateway,
    check_openclaw_version,
    ensure_openclaw_gateway,
    run_preflight,
)

ROOT = Path(__file__).resolve().parent.parent


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.version_patcher = patch(
            "openclaw_bench.preflight.check_openclaw_version",
            return_value=_version_check(),
        )
        self.version_patcher.start()

    def tearDown(self):
        self.version_patcher.stop()

    def test_simulator_preflight_passes_for_core_suite(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [
                sys.executable,
                "-m",
                "openclaw_bench",
                "preflight",
                "--backend",
                "simulator",
                "--suite",
                str(ROOT / "manifests" / "openclaw-agent-core.json"),
                "--model-config",
                str(ROOT / "manifests" / "initial-models.json"),
                "--out",
                str(Path(tmp) / "results"),
                "--json",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertTrue(payload["ok"])
            self.assertTrue(any(check["name"] == "fixture_size:workspace-needle-64k" and check["status"] == "pass" for check in payload["checks"]))

    def test_preflight_warns_for_host_specific_model_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [
                sys.executable,
                "-m",
                "openclaw_bench",
                "preflight",
                "--backend",
                "simulator",
                "--suite",
                str(ROOT / "manifests" / "openclaw-agent-discovery-smoke.example.json"),
                "--model-config",
                str(ROOT / "manifests" / "vllm-local.example.json"),
                "--out",
                str(Path(tmp) / "results"),
                "--json",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertTrue(payload["ok"])
            self.assertTrue(
                any(
                    check["name"] == "model_config_portability"
                    and check["status"] == "warn"
                    and "workstation" in check["notes"]
                    for check in payload["checks"]
                )
            )

    def test_preflight_fails_missing_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp) / "suite.json"
            suite.write_text(
                json.dumps(
                    {
                        "suite_id": "bad-suite",
                        "tasks": [
                            {
                                "task_id": "missing",
                                "task_type": "workspace_discovery",
                                "fixture": "does-not-exist",
                                "prompt": "x",
                                "expected": {}
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            cmd = [
                sys.executable,
                "-m",
                "openclaw_bench",
                "preflight",
                "--backend",
                "simulator",
                "--suite",
                str(suite),
                "--out",
                str(Path(tmp) / "results"),
                "--json",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertFalse(payload["ok"])
            self.assertTrue(any(check["name"] == "fixture:missing" for check in payload["checks"]))

    def test_preflight_fails_bad_discovery_expected_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp) / "fixtures" / "repo"
            fixture.mkdir(parents=True)
            suite_path = Path(tmp) / "suite.json"
            suite_path.write_text(
                json.dumps(
                    {
                        "suite_id": "bad-discovery",
                        "tasks": [
                            {
                                "task_id": "discovery",
                                "task_type": "workspace_discovery",
                                "fixture": "repo",
                                "prompt": "x",
                                "expected": {
                                    "test_command": "python -m unittest",
                                    "routes_file": "api/routes.py",
                                    "schema_file": "db/schema.py",
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            cmd = [
                sys.executable,
                "-m",
                "openclaw_bench",
                "preflight",
                "--backend",
                "simulator",
                "--suite",
                str(suite_path),
                "--fixtures-root",
                str(fixture.parent),
                "--out",
                str(Path(tmp) / "results"),
                "--json",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertTrue(any(check["name"] == "expected_paths:discovery:routes_file,schema_file" and check["status"] == "fail" for check in payload["checks"]))

    def test_preflight_passes_needle_expected_paths(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        with tempfile.TemporaryDirectory() as tmp:
            result = run_preflight(
                suite=suite,
                models=[ModelSpec.from_alias("simulated-model", "fp8", 4096)],
                backend_name="simulator",
                out_dir=Path(tmp) / "results",
                workspace_root=Path(tmp) / "workspaces",
                fixtures_root=ROOT / "fixtures",
                openclaw_profile="bench",
                openclaw_local=True,
            )
        self.assertTrue(result.ok, result.to_json())
        self.assertTrue(
            any(
                check.name == "expected_paths:workspace-needle-4k:source_file,target_file" and check.status == "pass"
                for check in result.checks
            )
        )

    def test_preflight_passes_action_gate_expected_paths(self):
        suite = load_suite(ROOT / "manifests" / "tier-medium.json")
        with tempfile.TemporaryDirectory() as tmp:
            result = run_preflight(
                suite=suite,
                models=[ModelSpec.from_alias("simulated-model", "fp8", 16384)],
                backend_name="simulator",
                out_dir=Path(tmp) / "results",
                workspace_root=Path(tmp) / "workspaces",
                fixtures_root=ROOT / "fixtures",
                openclaw_profile="bench",
                openclaw_local=True,
            )
        self.assertTrue(result.ok, result.to_json())
        self.assertTrue(
            any(
                check.name == "expected_paths:medium-ambiguous-spec-triage:evidence_files" and check.status == "pass"
                for check in result.checks
            )
        )
        self.assertTrue(
            any(
                check.name == "expected_paths:medium-ambiguous-spec-triage:preserved_files" and check.status == "pass"
                for check in result.checks
            )
        )

    def test_preflight_passes_format_drift_expected_paths(self):
        suite = load_suite(ROOT / "manifests" / "tier-medium.json")
        with tempfile.TemporaryDirectory() as tmp:
            result = run_preflight(
                suite=suite,
                models=[ModelSpec.from_alias("simulated-model", "fp8", 16384)],
                backend_name="simulator",
                out_dir=Path(tmp) / "results",
                workspace_root=Path(tmp) / "workspaces",
                fixtures_root=ROOT / "fixtures",
                openclaw_profile="bench",
                openclaw_local=True,
            )
        self.assertTrue(result.ok, result.to_json())
        self.assertTrue(
            any(
                check.name == "expected_paths:medium-format-drift-under-length:source_file,final_file"
                and check.status == "pass"
                for check in result.checks
            )
        )
        self.assertTrue(
            any(
                check.name == "expected_paths:medium-format-drift-under-length:trail_files" and check.status == "pass"
                for check in result.checks
            )
        )

    def test_openclaw_local_preflight_does_not_require_gateway(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [
                sys.executable,
                "-m",
                "openclaw_bench",
                "preflight",
                "--backend",
                "openclaw",
                "--openclaw-local",
                "--suite",
                str(ROOT / "manifests" / "openclaw-agent-core.json"),
                "--model-config",
                str(ROOT / "manifests" / "initial-models.json"),
                "--out",
                str(Path(tmp) / "results"),
                "--json",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
            payload = json.loads(proc.stdout)
            self.assertEqual(proc.returncode, 1)
            self.assertFalse(payload["ok"])
            self.assertTrue(any(check["name"] == "openclaw_cli" for check in payload["checks"]))
            self.assertTrue(any(check["name"] == "openclaw_gateway" and check["status"] == "warn" for check in payload["checks"]))
            self.assertTrue(any(check["name"].startswith("model:") and check["status"] == "fail" for check in payload["checks"]))

    def test_gateway_check_passes_foreground_gateway_with_disabled_service(self):
        output = (
            "Service: systemd (disabled)\n"
            "Config (cli): ~/.openclaw-bench/openclaw.json\n"
            "Probe target: ws://127.0.0.1:19091\n"
            "Runtime: stopped (state inactive, sub dead, last exit 0, reason 0)\n"
            "Connectivity probe: ok\n"
        )
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=output, stderr="")
        with patch("subprocess.run", return_value=completed):
            check = _check_gateway("bench")
        self.assertEqual(check.status, "pass")
        self.assertIn("Connectivity probe: ok", check.notes)

    def test_gateway_check_fails_when_connectivity_probe_fails(self):
        output = (
            "Probe target: ws://127.0.0.1:19091\n"
            "Runtime: stopped (state inactive, sub dead, last exit 0, reason 0)\n"
            "Connectivity probe: failed\n"
        )
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=output, stderr="")
        with patch("subprocess.run", return_value=completed):
            check = _check_gateway("bench")
        self.assertEqual(check.status, "fail")

    def test_ensure_gateway_starts_bench_profile_when_probe_fails(self):
        failed = subprocess.CompletedProcess(args=[], returncode=0, stdout="Connectivity probe: failed\n", stderr="")
        running = subprocess.CompletedProcess(args=[], returncode=0, stdout="Connectivity probe: ok\n", stderr="")
        with tempfile.TemporaryDirectory() as tmp:
            with patch("openclaw_bench.preflight._gateway_pid_path", return_value=Path(tmp) / "gateway.pid"):
                with patch("subprocess.Popen") as popen_mock:
                    popen_mock.return_value.pid = 1234
                    with patch("subprocess.run", side_effect=[failed, running]) as run_mock:
                        check = ensure_openclaw_gateway("bench")

        self.assertEqual(check.status, "pass")
        calls = [call.args[0] for call in run_mock.call_args_list]
        self.assertEqual(calls[0], ["openclaw", "--profile", "bench", "gateway", "status"])
        self.assertEqual(calls[1], ["openclaw", "--profile", "bench", "gateway", "status"])
        popen_mock.assert_called_once()
        self.assertEqual(popen_mock.call_args.args[0], ["openclaw", "--profile", "bench", "gateway", "--dev", "--verbose", "run"])
        self.assertIn("started bench gateway", check.notes)

    def test_ensure_gateway_polls_until_started(self):
        failed = subprocess.CompletedProcess(args=[], returncode=0, stdout="Connectivity probe: failed\n", stderr="")
        running = subprocess.CompletedProcess(args=[], returncode=0, stdout="Connectivity probe: ok\n", stderr="")
        with tempfile.TemporaryDirectory() as tmp:
            with patch("openclaw_bench.preflight._gateway_pid_path", return_value=Path(tmp) / "gateway.pid"):
                with patch("subprocess.Popen") as popen_mock:
                    popen_mock.return_value.pid = 1234
                    with patch("subprocess.run", side_effect=[failed, failed, running]) as run_mock:
                        with patch("time.sleep") as sleep_mock:
                            check = ensure_openclaw_gateway("bench", timeout_s=2)

        self.assertEqual(check.status, "pass")
        sleep_mock.assert_called_once_with(1)
        self.assertEqual(len(run_mock.call_args_list), 3)

    def test_ensure_gateway_respects_openclaw_container(self):
        failed = subprocess.CompletedProcess(args=[], returncode=0, stdout="Connectivity probe: failed\n", stderr="")
        started = subprocess.CompletedProcess(args=[], returncode=0, stdout="started\n", stderr="")
        running = subprocess.CompletedProcess(args=[], returncode=0, stdout="Connectivity probe: ok\n", stderr="")
        with patch("subprocess.run", side_effect=[failed, started, running]) as run_mock:
            check = ensure_openclaw_gateway("bench", "oc-bench-gateway")

        self.assertEqual(check.status, "pass")
        calls = [call.args[0] for call in run_mock.call_args_list]
        self.assertEqual(calls[0][:4], ["docker", "exec", "oc-bench-gateway", "openclaw"])
        self.assertEqual(calls[2][:4], ["docker", "exec", "oc-bench-gateway", "openclaw"])
        self.assertEqual(calls[1][:5], ["docker", "exec", "-d", "oc-bench-gateway", "sh"])
        self.assertIn("--profile bench gateway --port 19091", calls[1][-1])
        self.assertIn("--verbose", calls[1][-1])

    def test_container_preflight_checks_container_openclaw_and_route_smoke(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        models = [
            ModelSpec(
                model_id="local-vllm",
                served_model_name="qwen",
                openclaw_model_name="vllm/qwen",
                provider_type="api",
                api_env="OPENCLAW_TEST_KEY",
            )
        ]
        completed = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="config ok", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="Connectivity probe: ok\n", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout='{"text":"ok"}', stderr=""),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"OPENCLAW_TEST_KEY": "test-key"}):
                with patch("shutil.which", return_value="/usr/bin/docker"):
                    with patch("subprocess.run", side_effect=completed) as run_mock:
                        result = run_preflight(
                            suite=suite,
                            models=models,
                            backend_name="openclaw",
                            out_dir=Path(tmp) / "results",
                            workspace_root=Path(tmp) / "workspaces",
                            fixtures_root=ROOT / "fixtures",
                            openclaw_profile="bench",
                            openclaw_local=False,
                            openclaw_container="oc-bench-gateway",
                            smoke_turn=True,
                        )

        self.assertTrue(result.ok, result.to_json())
        calls = [call.args[0] for call in run_mock.call_args_list]
        self.assertEqual(calls[0][:4], ["docker", "exec", "oc-bench-gateway", "test"])
        self.assertIn(str(Path(tmp) / "workspaces"), calls[0])
        self.assertEqual(calls[1][:4], ["docker", "exec", "oc-bench-gateway", "openclaw"])
        self.assertIn("config", calls[1])
        self.assertEqual(calls[2][:4], ["docker", "exec", "oc-bench-gateway", "openclaw"])
        self.assertIn("gateway", calls[2])
        self.assertEqual(calls[3][:4], ["docker", "exec", "oc-bench-gateway", "openclaw"])
        self.assertIn("infer", calls[3])
        self.assertIn("vllm/qwen", calls[3])
        self.assertTrue(any(check.name == "container_workspace_root" and check.status == "pass" for check in result.checks))
        self.assertTrue(any(check.name == "openclaw_profile_config" and check.status == "pass" for check in result.checks))
        self.assertTrue(any(check.name == "openclaw_gateway" and check.status == "pass" for check in result.checks))

    def test_container_preflight_fails_when_workspace_root_is_not_mounted(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-discovery-smoke.example.json")
        models = [ModelSpec(model_id="local-vllm", served_model_name="qwen", openclaw_model_name="vllm/qwen")]
        completed = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as tmp:
            with patch("shutil.which", return_value="/usr/bin/docker"):
                with patch("subprocess.run", return_value=completed):
                    result = run_preflight(
                        suite=suite,
                        models=models,
                        backend_name="openclaw",
                        out_dir=Path(tmp) / "results",
                        workspace_root=Path(tmp) / "workspaces",
                        fixtures_root=ROOT / "fixtures",
                        openclaw_profile="bench",
                        openclaw_local=False,
                        openclaw_container="oc-bench-gateway",
                    )

        self.assertFalse(result.ok)
        self.assertTrue(
            any(
                check.name == "container_workspace_root"
                and check.status == "fail"
                and "missing or not writable" in check.notes
                for check in result.checks
            )
        )

    def test_agent_smoke_uses_configured_agent_path(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-discovery-smoke.example.json")
        models = [
            ModelSpec(
                model_id="local-vllm",
                served_model_name="qwen",
                openclaw_model_name="vllm/qwen",
                provider_type="api",
                api_env="OPENCLAW_TEST_KEY",
            )
        ]
        completed = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="config ok", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="Connectivity probe: ok\n", stderr=""),
        ]
        response = BackendResponse(
            text="ok",
            json_output={"text": "ok"},
            raw={"cmd": ["openclaw"]},
            tool_calls=3,
            files_read=2,
            duplicate_file_reads=0,
            time_to_first_relevant_file_s=0.2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"OPENCLAW_TEST_KEY": "test-key"}):
                with patch("shutil.which", return_value="/usr/bin/docker"):
                    with patch("subprocess.run", side_effect=completed):
                        with patch("openclaw_bench.preflight.OpenClawBackend") as backend_cls:
                            backend_cls.return_value.run.return_value = response
                            result = run_preflight(
                                suite=suite,
                                models=models,
                                backend_name="openclaw",
                                out_dir=Path(tmp) / "results",
                                workspace_root=Path(tmp) / "workspaces",
                                fixtures_root=ROOT / "fixtures",
                                openclaw_profile="bench",
                                openclaw_agent="dev",
                                openclaw_local=False,
                                openclaw_container="oc-bench-gateway",
                                openclaw_workspace_agents=True,
                                agent_smoke_turn=True,
                            )

        self.assertTrue(result.ok, result.to_json())
        backend_cls.assert_called_once_with(
            profile="bench",
            agent="dev",
            local=False,
            workspace_agents=True,
            container="oc-bench-gateway",
        )
        self.assertTrue(any(check.name.startswith("openclaw_agent:vllm/qwen") and check.status == "pass" for check in result.checks))

    def test_agent_smoke_requires_certification_telemetry(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-discovery-smoke.example.json")
        models = [
            ModelSpec(
                model_id="local-vllm",
                served_model_name="qwen",
                openclaw_model_name="vllm/qwen",
                provider_type="api",
                api_env="OPENCLAW_TEST_KEY",
            )
        ]
        completed = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="config ok", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="Connectivity probe: ok\n", stderr=""),
        ]
        response = BackendResponse(text="ok", json_output={"text": "ok"}, raw={"cmd": ["openclaw"]}, tool_calls=1, files_read=1)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"OPENCLAW_TEST_KEY": "test-key"}):
                with patch("shutil.which", return_value="/usr/bin/docker"):
                    with patch("subprocess.run", side_effect=completed):
                        with patch("openclaw_bench.preflight.OpenClawBackend") as backend_cls:
                            backend_cls.return_value.run.return_value = response
                            result = run_preflight(
                                suite=suite,
                                models=models,
                                backend_name="openclaw",
                                out_dir=Path(tmp) / "results",
                                workspace_root=Path(tmp) / "workspaces",
                                fixtures_root=ROOT / "fixtures",
                                openclaw_profile="bench",
                                openclaw_agent="dev",
                                openclaw_local=False,
                                openclaw_container="oc-bench-gateway",
                                openclaw_workspace_agents=True,
                                agent_smoke_turn=True,
                            )

        self.assertFalse(result.ok)
        self.assertTrue(
            any(
                check.name.startswith("openclaw_agent:vllm/qwen")
                and check.status == "fail"
                and "missing certification telemetry" in check.notes
                and "duplicate_file_reads" in check.notes
                for check in result.checks
            )
        )

    def test_agent_smoke_reports_openclaw_error(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-discovery-smoke.example.json")
        models = [
            ModelSpec(
                model_id="local-vllm",
                served_model_name="qwen",
                openclaw_model_name="vllm/qwen",
                provider_type="api",
                api_env="OPENCLAW_TEST_KEY",
            )
        ]
        completed = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="config ok", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="Connectivity probe: ok\n", stderr=""),
        ]
        response = BackendResponse(
            text="",
            json_output=None,
            raw={"stderr": "provider/model overrides are not authorized for this caller"},
            error="model_override_unauthorized",
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"OPENCLAW_TEST_KEY": "test-key"}):
                with patch("shutil.which", return_value="/usr/bin/docker"):
                    with patch("subprocess.run", side_effect=completed):
                        with patch("openclaw_bench.preflight.OpenClawBackend") as backend_cls:
                            backend_cls.return_value.run.return_value = response
                            result = run_preflight(
                                suite=suite,
                                models=models,
                                backend_name="openclaw",
                                out_dir=Path(tmp) / "results",
                                workspace_root=Path(tmp) / "workspaces",
                                fixtures_root=ROOT / "fixtures",
                                openclaw_profile="bench",
                                openclaw_agent="dev",
                                openclaw_local=False,
                                openclaw_container="oc-bench-gateway",
                                agent_smoke_turn=True,
                            )

        self.assertFalse(result.ok)
        self.assertTrue(
            any(
                check.name.startswith("openclaw_agent:vllm/qwen")
                and check.status == "fail"
                and "model_override_unauthorized" in check.notes
                for check in result.checks
            )
        )

    def test_preflight_fails_missing_serve_command_executable(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_config = Path(tmp) / "models.json"
            model_config.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "model_id": "local",
                                "served_model_name": "local",
                                "serve_command": ["definitely-not-vllm", "serve", "local"],
                                "health_check_url": "http://127.0.0.1:18000/health"
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            cmd = [
                sys.executable,
                "-m",
                "openclaw_bench",
                "preflight",
                "--backend",
                "openclaw",
                "--openclaw-local",
                "--suite",
                str(ROOT / "manifests" / "openclaw-agent-core.json"),
                "--model-config",
                str(model_config),
                "--out",
                str(Path(tmp) / "results"),
                "--json",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertTrue(any("serve command not found" in check["notes"] for check in payload["checks"]))

    def test_model_config_kv_override_rejects_local_serve_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_config = Path(tmp) / "models.json"
            model_config.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "model_id": "local",
                                "served_model_name": "local-fp8",
                                "serve_command": [sys.executable, "-c", "import time; time.sleep(5)"],
                                "health_check_url": "http://127.0.0.1:18000/v1/models",
                                "api_base": "http://127.0.0.1:18000/v1",
                                "kv_modes": ["fp8"]
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            cmd = [
                sys.executable,
                "-m",
                "openclaw_bench",
                "preflight",
                "--backend",
                "openclaw",
                "--suite",
                str(ROOT / "manifests" / "openclaw-agent-core.json"),
                "--model-config",
                str(model_config),
                "--kv",
                "turboquant_k8v4",
                "--out",
                str(Path(tmp) / "results"),
                "--json",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertEqual(proc.returncode, 2)
            self.assertIn("--kv/--contexts overrides are not safe", proc.stderr)

    def test_openclaw_local_preflight_fails_missing_declared_api_env(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        models = [
            ModelSpec(
                model_id="local-vllm",
                served_model_name="local-vllm",
                openclaw_model_name="vllm/local-vllm",
                serve_command=[sys.executable, "-c", "import time; time.sleep(5)"],
                health_check_url="http://127.0.0.1:8000/v1/models",
                api_base="http://127.0.0.1:8000/v1",
                api_env="VLLM_API_KEY",
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {}, clear=True):
                with patch("shutil.which", return_value="/usr/bin/openclaw"):
                    result = run_preflight(
                        suite=suite,
                        models=models,
                        backend_name="openclaw",
                        out_dir=Path(tmp) / "results",
                        workspace_root=Path(tmp) / "workspaces",
                        fixtures_root=ROOT / "fixtures",
                        openclaw_profile="bench",
                        openclaw_local=True,
                    )
        self.assertFalse(result.ok)
        self.assertTrue(any(check.status == "fail" and "VLLM_API_KEY" in check.notes for check in result.checks))

    def test_preflight_fails_openai_compatible_health_without_api_base(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        models = [
            ModelSpec(
                model_id="local-vllm",
                served_model_name="local-vllm",
                serve_command=[sys.executable, "-c", "import time; time.sleep(5)"],
                health_check_url="http://127.0.0.1:8000/v1/models",
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with patch("shutil.which", return_value="/usr/bin/openclaw"):
                result = run_preflight(
                    suite=suite,
                    models=models,
                    backend_name="openclaw",
                    out_dir=Path(tmp) / "results",
                    workspace_root=Path(tmp) / "workspaces",
                    fixtures_root=ROOT / "fixtures",
                    openclaw_profile="bench",
                    openclaw_local=True,
                )
        self.assertFalse(result.ok)
        self.assertTrue(any("requires api_base" in check.notes for check in result.checks))

    def test_smoke_turn_certifies_api_model_route(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        models = [
            ModelSpec(
                model_id="openai-gpt-4.1",
                served_model_name="openai/gpt-4.1",
                provider_type="api",
                kv_cache_dtype="provider_default",
                api_env="OPENAI_API_KEY",
            )
        ]
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout='{"text":"ok"}', stderr="")
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                with patch("shutil.which", return_value="/usr/bin/openclaw"):
                    with patch("subprocess.run", return_value=completed) as run_mock:
                        result = run_preflight(
                            suite=suite,
                            models=models,
                            backend_name="openclaw",
                            out_dir=Path(tmp) / "results",
                            workspace_root=Path(tmp) / "workspaces",
                            fixtures_root=ROOT / "fixtures",
                            openclaw_profile="bench",
                            openclaw_local=True,
                            smoke_turn=True,
                            smoke_timeout_s=7,
                        )
        self.assertTrue(result.ok)
        self.assertTrue(any(check.name.startswith("openclaw_route:") and check.status == "pass" for check in result.checks))
        cmd = run_mock.call_args.args[0]
        self.assertEqual(cmd[:5], ["openclaw", "--profile", "bench", "infer", "model"])
        self.assertIn("--local", cmd)
        self.assertIn("openai/gpt-4.1", cmd)
        self.assertEqual(run_mock.call_args.kwargs["timeout"], 7)

    def test_smoke_turn_skips_api_route_when_api_env_is_missing(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        models = [
            ModelSpec(
                model_id="openai-gpt-4.1",
                served_model_name="openai/gpt-4.1",
                provider_type="api",
                kv_cache_dtype="provider_default",
                api_env="OPENAI_API_KEY",
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {}, clear=True):
                with patch("shutil.which", return_value="/usr/bin/openclaw"):
                    with patch("subprocess.run") as run_mock:
                        result = run_preflight(
                            suite=suite,
                            models=models,
                            backend_name="openclaw",
                            out_dir=Path(tmp) / "results",
                            workspace_root=Path(tmp) / "workspaces",
                            fixtures_root=ROOT / "fixtures",
                            openclaw_profile="bench",
                            openclaw_local=True,
                            smoke_turn=True,
                        )

        self.assertFalse(result.ok)
        self.assertFalse(run_mock.called)
        self.assertTrue(
            any(
                check.name.startswith("openclaw_route:")
                and check.status == "fail"
                and "missing environment variable OPENAI_API_KEY" in check.notes
                for check in result.checks
            )
        )

    def test_smoke_turn_reuses_failed_route_status_for_duplicate_route(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        models = [
            ModelSpec(
                model_id="openai-gpt-4.1",
                served_model_name="openai/gpt-4.1",
                provider_type="api",
                kv_cache_dtype="provider_default",
                context_limit=4096,
                api_env="OPENAI_API_KEY",
            ),
            ModelSpec(
                model_id="openai-gpt-4.1",
                served_model_name="openai/gpt-4.1",
                provider_type="api",
                kv_cache_dtype="provider_default",
                context_limit=8192,
                api_env="OPENAI_API_KEY",
            ),
        ]
        completed = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="auth failed")
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                with patch("shutil.which", return_value="/usr/bin/openclaw"):
                    with patch("subprocess.run", return_value=completed) as run_mock:
                        result = run_preflight(
                            suite=suite,
                            models=models,
                            backend_name="openclaw",
                            out_dir=Path(tmp) / "results",
                            workspace_root=Path(tmp) / "workspaces",
                            fixtures_root=ROOT / "fixtures",
                            openclaw_profile="bench",
                            openclaw_local=True,
                            smoke_turn=True,
                        )

        route_checks = [check for check in result.checks if check.name.startswith("openclaw_route:openai/gpt-4.1")]
        self.assertFalse(result.ok)
        self.assertEqual(run_mock.call_count, 1)
        self.assertEqual([check.status for check in route_checks], ["fail", "fail"])
        self.assertIn("same OpenClaw route model already smoke-tested: fail", route_checks[1].notes)

    def test_smoke_turn_uses_openclaw_route_model_when_configured(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        models = [
            ModelSpec(
                model_id="local-vllm",
                served_model_name="vllm-served-name",
                openclaw_model_name="openclaw/local-alias",
                provider_type="api",
                api_env="OPENCLAW_TEST_KEY",
            )
        ]
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout='{"text":"ok"}', stderr="")
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"OPENCLAW_TEST_KEY": "test-key"}):
                with patch("shutil.which", return_value="/usr/bin/openclaw"):
                    with patch("subprocess.run", return_value=completed) as run_mock:
                        result = run_preflight(
                            suite=suite,
                            models=models,
                            backend_name="openclaw",
                            out_dir=Path(tmp) / "results",
                            workspace_root=Path(tmp) / "workspaces",
                            fixtures_root=ROOT / "fixtures",
                            openclaw_profile="bench",
                            openclaw_local=True,
                            smoke_turn=True,
                        )
        self.assertTrue(result.ok)
        cmd = run_mock.call_args.args[0]
        self.assertIn("openclaw/local-alias", cmd)
        self.assertNotIn("vllm-served-name", cmd)
        self.assertTrue(any("openclaw/local-alias" in check.name for check in result.checks))

    def test_smoke_turn_skips_harness_started_local_server(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        models = [
            ModelSpec(
                model_id="local",
                served_model_name="local",
                serve_command=[sys.executable, "-c", "import time; time.sleep(5)"],
                health_check_url="http://127.0.0.1:18000/v1/models",
                api_base="http://127.0.0.1:18000/v1",
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with patch("shutil.which", return_value="/usr/bin/openclaw"):
                with patch("subprocess.run") as run_mock:
                    result = run_preflight(
                        suite=suite,
                        models=models,
                        backend_name="openclaw",
                        out_dir=Path(tmp) / "results",
                        workspace_root=Path(tmp) / "workspaces",
                        fixtures_root=ROOT / "fixtures",
                        openclaw_profile="bench",
                        openclaw_local=True,
                        smoke_turn=True,
                    )
        self.assertTrue(result.ok)
        self.assertFalse(run_mock.called)
        self.assertTrue(any(check.name.startswith("openclaw_route:") and check.status == "warn" for check in result.checks))


class OpenClawVersionTests(unittest.TestCase):
    def test_pinned_openclaw_version_passes(self):
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=f"OpenClaw {OPENCLAW_PINNED_VERSION} (abc123)\n", stderr="")
        with patch("subprocess.run", return_value=completed):
            check = check_openclaw_version()
        self.assertEqual(check.status, "pass")

    def test_openclaw_429_is_blocked(self):
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="OpenClaw 2026.4.29 (abc123)\n", stderr="")
        with patch("subprocess.run", return_value=completed):
            check = check_openclaw_version()
        self.assertEqual(check.status, "fail")
        self.assertIn("2026.4.27", check.notes)
        self.assertIn("2026.4.29", check.notes)


def _version_check(status: str = "pass", notes: str = "OpenClaw 2026.4.27 matches pinned version 2026.4.27"):
    from openclaw_bench.preflight import PreflightCheck

    return PreflightCheck("openclaw_version", status, notes)


if __name__ == "__main__":
    unittest.main()
