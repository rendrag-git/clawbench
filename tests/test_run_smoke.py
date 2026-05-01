import json
import io
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from openclaw_bench.backend import SimulatorBackend
from openclaw_bench.cli import preflight_command, run_command
from openclaw_bench.manifest import load_suite
from openclaw_bench.models import BackendResponse, ModelSpec
from openclaw_bench.preflight import PreflightCheck, PreflightResult
from openclaw_bench.runner import BenchmarkRunner, RunConfig


ROOT = Path(__file__).resolve().parent.parent


class RunSmokeTests(unittest.TestCase):
    def setUp(self):
        self.version_patcher = patch(
            "openclaw_bench.cli.check_openclaw_version",
            return_value=PreflightCheck("openclaw_version", "pass", "OpenClaw 2026.4.27 matches pinned version 2026.4.27"),
        )
        self.version_patcher.start()

    def tearDown(self):
        self.version_patcher.stop()

    def test_openclaw_run_ensures_gateway_before_benchmarking(self):
        args = _run_args(
            backend="openclaw",
            openclaw_local=False,
            openclaw_workspace_agents=True,
            ensure_openclaw_gateway=True,
            run_id="ensure-gateway-run",
        )
        gateway_check = PreflightCheck("openclaw_gateway", "pass", "already running")
        container_check = _container_check("oc-bench-gateway already running")
        with patch("openclaw_bench.cli.ensure_openclaw_container", return_value=container_check) as container_mock:
            with patch("openclaw_bench.cli.ensure_openclaw_gateway", return_value=gateway_check) as ensure_mock:
                with patch("openclaw_bench.cli.BenchmarkRunner") as runner_cls:
                    runner_cls.return_value.run.return_value = []
                    with redirect_stdout(io.StringIO()):
                        code = run_command(args)

        self.assertEqual(code, 0)
        container_mock.assert_called_once()
        self.assertEqual(container_mock.call_args.kwargs["container"], "oc-bench-gateway")
        ensure_mock.assert_called_once_with("bench", "oc-bench-gateway", timeout_s=60)
        runner_cls.return_value.run.assert_called_once()
        config = runner_cls.return_value.run.call_args.args[0]
        self.assertTrue(config.ensure_openclaw_gateway)
        self.assertEqual(config.openclaw_container, "oc-bench-gateway")
        self.assertEqual(config.openclaw_gateway_ensure, gateway_check.to_row())
        self.assertEqual(config.openclaw_gateway_timeout_s, 60)

    def test_openclaw_gateway_run_defaults_to_workspace_agents(self):
        args = _run_args(
            backend="openclaw",
            openclaw_local=False,
            openclaw_workspace_agents=None,
            run_id="default-workspace-agents-run",
        )
        gateway_check = PreflightCheck("openclaw_gateway", "pass", "started bench gateway")
        with patch("openclaw_bench.cli.ensure_openclaw_container", return_value=_container_check()):
            with patch("openclaw_bench.cli.ensure_openclaw_gateway", return_value=gateway_check):
                with patch("openclaw_bench.cli.BenchmarkRunner") as runner_cls:
                    runner_cls.return_value.run.return_value = []
                    with redirect_stdout(io.StringIO()):
                        code = run_command(args)

        self.assertEqual(code, 0)
        config = runner_cls.return_value.run.call_args.args[0]
        self.assertTrue(config.openclaw_workspace_agents)

    def test_openclaw_run_passes_gateway_timeout(self):
        args = _run_args(
            backend="openclaw",
            openclaw_local=False,
            openclaw_workspace_agents=None,
            openclaw_gateway_timeout=120,
            run_id="gateway-timeout-run",
        )
        gateway_check = PreflightCheck("openclaw_gateway", "pass", "started bench gateway")
        with patch("openclaw_bench.cli.ensure_openclaw_container", return_value=_container_check()) as container_mock:
            with patch("openclaw_bench.cli.ensure_openclaw_gateway", return_value=gateway_check) as ensure_mock:
                with patch("openclaw_bench.cli.BenchmarkRunner") as runner_cls:
                    runner_cls.return_value.run.return_value = []
                    with redirect_stdout(io.StringIO()):
                        code = run_command(args)

        self.assertEqual(code, 0)
        self.assertEqual(container_mock.call_args.kwargs["timeout_s"], 120)
        ensure_mock.assert_called_once_with("bench", "oc-bench-gateway", timeout_s=120)
        config = runner_cls.return_value.run.call_args.args[0]
        self.assertEqual(config.openclaw_gateway_timeout_s, 120)

    def test_openclaw_run_can_skip_gateway_ensure_for_supervised_runtimes(self):
        args = _run_args(
            backend="openclaw",
            openclaw_local=False,
            openclaw_workspace_agents=True,
            ensure_openclaw_gateway=False,
            run_id="skip-gateway-run",
        )
        with patch("openclaw_bench.cli.ensure_openclaw_container", return_value=_container_check()) as container_mock:
            with patch("openclaw_bench.cli.ensure_openclaw_gateway") as ensure_mock:
                with patch("openclaw_bench.cli.BenchmarkRunner") as runner_cls:
                    runner_cls.return_value.run.return_value = []
                    with redirect_stdout(io.StringIO()):
                        code = run_command(args)

        self.assertEqual(code, 0)
        container_mock.assert_called_once()
        ensure_mock.assert_not_called()
        config = runner_cls.return_value.run.call_args.args[0]
        self.assertFalse(config.ensure_openclaw_gateway)
        self.assertIsNone(config.openclaw_gateway_ensure)

    def test_openclaw_run_can_skip_container_ensure_for_supervised_runtimes(self):
        args = _run_args(
            backend="openclaw",
            openclaw_local=False,
            openclaw_workspace_agents=True,
            ensure_openclaw_container=False,
            run_id="skip-container-run",
        )
        gateway_check = PreflightCheck("openclaw_gateway", "pass", "already running")
        with patch("openclaw_bench.cli.ensure_openclaw_container") as container_mock:
            with patch("openclaw_bench.cli.ensure_openclaw_gateway", return_value=gateway_check):
                with patch("openclaw_bench.cli.BenchmarkRunner") as runner_cls:
                    runner_cls.return_value.run.return_value = []
                    with redirect_stdout(io.StringIO()):
                        code = run_command(args)

        self.assertEqual(code, 0)
        container_mock.assert_not_called()

    def test_openclaw_container_ensure_receives_custom_workspace_root(self):
        args = _run_args(
            backend="openclaw",
            openclaw_local=False,
            openclaw_workspace_agents=True,
            workspace_root="/tmp/custom-oc-bench-workspaces",
            run_id="custom-workspace-run",
        )
        gateway_check = PreflightCheck("openclaw_gateway", "pass", "already running")
        with patch("openclaw_bench.cli.ensure_openclaw_container", return_value=_container_check()) as container_mock:
            with patch("openclaw_bench.cli.ensure_openclaw_gateway", return_value=gateway_check):
                with patch("openclaw_bench.cli.BenchmarkRunner") as runner_cls:
                    runner_cls.return_value.run.return_value = []
                    with redirect_stdout(io.StringIO()):
                        code = run_command(args)

        self.assertEqual(code, 0)
        self.assertEqual(container_mock.call_args.kwargs["workspace_root"], Path("/tmp/custom-oc-bench-workspaces"))

    def test_preflight_json_reports_container_ensure_without_prefix_output(self):
        args = _run_args(
            backend="openclaw",
            openclaw_local=False,
            openclaw_workspace_agents=True,
        )
        args.json = True
        args.smoke_turn = False
        args.agent_smoke_turn = False
        args.smoke_timeout = 60
        preflight_result = PreflightResult([PreflightCheck("suite_manifest", "pass", "ok")])
        with patch("openclaw_bench.cli.ensure_openclaw_container", return_value=_container_check()):
            with patch("openclaw_bench.cli.run_preflight", return_value=preflight_result):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    code = preflight_command(args)

        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["checks"][0]["name"], "openclaw_container")

    def test_simulator_end_to_end_run_writes_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results"
            cmd = [
                sys.executable,
                "-m",
                "openclaw_bench",
                "run",
                "--backend",
                "simulator",
                "--suite",
                str(ROOT / "manifests" / "openclaw-agent-core.json"),
                "--models",
                "simulated-model",
                "--kv",
                "fp8",
                "--concurrency",
                "1",
                "--contexts",
                "4096",
                "--out",
                str(out),
                "--run-id",
                "test-run",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            run_dir = out / "test-run"
            self.assertTrue((run_dir / "attempts.jsonl").exists())
            self.assertTrue((run_dir / "config.json").exists())
            self.assertTrue((run_dir / "failures.jsonl").exists())
            self.assertTrue((run_dir / "summary.md").exists())
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "server.json").exists())
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["attempts"], 5)
            self.assertEqual(summary["pass_rate"], 1.0)
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["backend"], "simulator")
            self.assertEqual(config["openclaw_smoke_timeout_s"], 60)
            self.assertEqual(config["workspace_root"], str(Path(tmp) / "workspaces" / "test-run"))
            self.assertEqual(config["models"][0]["hardware_profile"], "default")
            self.assertFalse(config["ensure_openclaw_gateway"])
            self.assertIsNone(config["openclaw_gateway_ensure"])
            self.assertEqual(config["provenance"]["suite_id"], "openclaw-agent-core")
            self.assertEqual(config["provenance"]["task_count"], 9)
            self.assertEqual(len(config["provenance"]["suite_digest"]), 64)
            self.assertEqual(config["provenance"]["model_source"], "aliases")
            self.assertEqual(len(config["provenance"]["model_matrix_digest"]), 64)
            self.assertEqual(config["provenance"]["input_files"][0]["role"], "suite")
            self.assertTrue(config["provenance"]["input_files"][0]["path"].endswith("openclaw-agent-core.json"))
            self.assertEqual(len(config["provenance"]["input_files"][0]["digest"]), 64)
            self.assertTrue(config["provenance"]["fixture_digests"])
            self.assertEqual(config["runtime"]["schema_version"], 1)
            self.assertTrue(config["runtime"]["python_version"])
            self.assertTrue(config["runtime"]["harness_version"])
            self.assertNotIn("openclaw", config["runtime"])
            server = json.loads((run_dir / "server.json").read_text(encoding="utf-8"))
            self.assertEqual(server["serve_results"][0]["load_success"], True)
            self.assertEqual(server["models"][0]["hardware_profile"], "default")
            self.assertEqual(server["support_probes"][0]["hardware_profile"], "default")
            self.assertIn("hardware", server)
            self.assertIn("throughput_probes", server)
            attempt = json.loads((run_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(attempt["hardware_profile"], "default")

    def test_real_repo_readonly_simulator_run_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results"
            cmd = [
                sys.executable,
                "-m",
                "openclaw_bench",
                "run",
                "--backend",
                "simulator",
                "--suite",
                str(ROOT / "manifests" / "real-repo-readonly.example.json"),
                "--models",
                "simulated-model",
                "--kv",
                "fp8",
                "--concurrency",
                "1",
                "--contexts",
                "4096",
                "--out",
                str(out),
                "--run-id",
                "real-repo-readonly",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            run_dir = out / "real-repo-readonly"
            attempts = [
                json.loads(line)
                for line in (run_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(attempts), 3)
            self.assertTrue(all(row["status"] == "pass" for row in attempts))
            changed_by_task = {row["task_id"]: row["files_changed"] for row in attempts}
            self.assertEqual(changed_by_task["real-repo-route-map"], 0)
            self.assertEqual(changed_by_task["real-repo-auth-guard"], 0)
            self.assertEqual(changed_by_task["real-repo-order-status-edit"], 1)
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["pass_rate"], 1.0)

    def test_tier_small_simulator_run_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results"
            cmd = [
                sys.executable,
                "-m",
                "openclaw_bench",
                "run",
                "--backend",
                "simulator",
                "--suite",
                str(ROOT / "manifests" / "tier-small.json"),
                "--models",
                "simulated-model",
                "--kv",
                "fp8",
                "--concurrency",
                "1",
                "--contexts",
                "4096",
                "--out",
                str(out),
                "--run-id",
                "tier-small",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            run_dir = out / "tier-small"
            attempts = [
                json.loads(line)
                for line in (run_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(attempts), 3)
            self.assertEqual({row["task_id"] for row in attempts}, {"small-workspace-discovery", "small-patch-execution", "small-workspace-needle-4k"})
            self.assertTrue(all(row["status"] == "pass" for row in attempts))

    def test_tier_medium_simulator_run_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results"
            cmd = [
                sys.executable,
                "-m",
                "openclaw_bench",
                "run",
                "--backend",
                "simulator",
                "--suite",
                str(ROOT / "manifests" / "tier-medium.json"),
                "--models",
                "simulated-model",
                "--kv",
                "fp8",
                "--concurrency",
                "1",
                "--contexts",
                "16384,32768",
                "--out",
                str(out),
                "--run-id",
                "tier-medium",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            run_dir = out / "tier-medium"
            attempts = [
                json.loads(line)
                for line in (run_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(attempts), 8)
            self.assertEqual(
                {row["task_id"] for row in attempts},
                {
                    "medium-multi-file-bug-trace",
                    "medium-instruction-retention",
                    "medium-workspace-needle-16k",
                    "medium-workspace-needle-32k",
                    "medium-tool-error-recovery-route-map",
                },
            )
            self.assertTrue(all(row["status"] == "pass" for row in attempts))

    def test_matrix_run_writes_one_raw_and_patch_artifact_per_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results"
            cmd = [
                sys.executable,
                "-m",
                "openclaw_bench",
                "run",
                "--backend",
                "simulator",
                "--suite",
                str(ROOT / "manifests" / "openclaw-agent-core.json"),
                "--model-config",
                str(ROOT / "manifests" / "initial-models.json"),
                "--kv",
                "fp8,turboquant_k8v4,turboquant_k3v4_nc",
                "--concurrency",
                "1",
                "--contexts",
                "4096",
                "--out",
                str(out),
                "--run-id",
                "matrix-run",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            run_dir = out / "matrix-run"
            attempts = (run_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(attempts), 30)
            self.assertEqual(len(list((run_dir / "raw").glob("*.json"))), 30)
            self.assertEqual(len(list((run_dir / "patches").glob("*.diff"))), 30)
            session_ids = []
            for path in (run_dir / "raw").glob("*.json"):
                payload = json.loads(path.read_text(encoding="utf-8"))
                session_ids.append(payload["response"]["session_id"])
            self.assertEqual(len(session_ids), len(set(session_ids)))

    def test_multi_concurrency_run_writes_unique_artifacts_per_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results"
            cmd = [
                sys.executable,
                "-m",
                "openclaw_bench",
                "run",
                "--backend",
                "simulator",
                "--suite",
                str(ROOT / "manifests" / "openclaw-agent-core.json"),
                "--models",
                "simulated-model",
                "--kv",
                "fp8",
                "--concurrency",
                "1,2",
                "--contexts",
                "4096",
                "--out",
                str(out),
                "--run-id",
                "multi-concurrency",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            run_dir = out / "multi-concurrency"
            attempts = (run_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(attempts), 15)
            self.assertEqual(len(list((run_dir / "raw").glob("*.json"))), 15)
            self.assertEqual(len(list((run_dir / "patches").glob("*.diff"))), 15)
            workspace_ids = [json.loads(line)["workspace_id"] for line in attempts]
            self.assertEqual(len(workspace_ids), len(set(workspace_ids)))
            self.assertTrue(any("-conc1-" in workspace_id for workspace_id in workspace_ids))
            self.assertTrue(any("-conc2-" in workspace_id for workspace_id in workspace_ids))

    def test_openclaw_session_ids_are_short_and_stable(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        long_model_name = "qwen3.6-35b-a3b-" + "very-long-hardware-label-" * 4
        model = ModelSpec(
            model_id="Qwen/Qwen3.6-35B-A3B-FP8",
            served_model_name=long_model_name,
            hardware_profile="gpu1-rtx-pro-5000-blackwell-gmu85-eager-ctx4096",
            kv_cache_dtype="auto",
            context_limit=4096,
            support_status="validated_external",
        )
        backend = _SessionIdCaptureBackend()
        with tempfile.TemporaryDirectory() as tmp:
            config = RunConfig(
                run_id="live-qwen36-fp8-4k-20260501T0725Z-with-a-long-suffix",
                suite=suite,
                models=[model],
                kv_modes=["auto"],
                contexts=[4096],
                concurrencies=[1],
                out_dir=Path(tmp) / "results",
                workspace_root=Path(tmp) / "workspaces",
                fixtures_root=ROOT / "fixtures",
                backend_name="simulator",
            )
            BenchmarkRunner(backend).run(config)
        self.assertEqual(len(backend.session_ids), 5)
        self.assertEqual(len(backend.session_ids), len(set(backend.session_ids)))
        for session_id in backend.session_ids:
            self.assertLessEqual(len(session_id), 80)
            self.assertRegex(session_id, r"^[A-Za-z0-9_-]+$")

    def test_runner_seeds_openclaw_context_before_backend_run(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-discovery-smoke.example.json")
        model = ModelSpec(
            model_id="qwen3",
            served_model_name="qwen3-1.7b",
            kv_cache_dtype="provider_default",
            context_limit=32768,
        )
        backend = _WorkspaceSeedCaptureBackend()
        with tempfile.TemporaryDirectory() as tmp:
            config = RunConfig(
                run_id="seeded-context",
                suite=suite,
                models=[model],
                kv_modes=["provider_default"],
                contexts=[32768],
                concurrencies=[1],
                out_dir=Path(tmp) / "results",
                workspace_root=Path(tmp) / "workspaces",
                fixtures_root=ROOT / "fixtures",
                backend_name="simulator",
                openclaw_agent="bench",
            )
            BenchmarkRunner(backend).run(config)

        self.assertEqual(backend.seed_files_seen["BOOTSTRAP.md"], False)
        self.assertEqual(backend.seed_files_seen["AGENTS.md"], True)
        self.assertEqual(backend.seed_files_seen["SOUL.md"], True)
        self.assertEqual(backend.seed_files_seen[".openclaw/workspace-state.json"], True)
        self.assertIn("qwen3-1.7b", backend.identity_text)

    def test_model_config_defaults_drive_matrix_without_cli_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results"
            cmd = [
                sys.executable,
                "-m",
                "openclaw_bench",
                "run",
                "--backend",
                "simulator",
                "--suite",
                str(ROOT / "manifests" / "openclaw-agent-core.json"),
                "--model-config",
                str(ROOT / "manifests" / "initial-models.json"),
                "--out",
                str(out),
                "--run-id",
                "default-matrix",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            run_dir = out / "default-matrix"
            attempts = (run_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(attempts), 20)
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(summary["kv_acceptance"]), 2)
            self.assertTrue(all(row["acceptance_status"] == "pending_live_benefit" for row in summary["kv_acceptance"]))

    def test_long_context_needle_tasks_are_context_gated(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results"
            cmd = [
                sys.executable,
                "-m",
                "openclaw_bench",
                "run",
                "--backend",
                "simulator",
                "--suite",
                str(ROOT / "manifests" / "openclaw-agent-core.json"),
                "--models",
                "simulated-model",
                "--kv",
                "fp8",
                "--concurrency",
                "1",
                "--contexts",
                "4096,8192,16384,32768,65536",
                "--out",
                str(out),
                "--run-id",
                "long-context",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            attempts = [
                json.loads(line)
                for line in (out / "long-context" / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(attempts), 25)
            needle_contexts = {
                row["task_id"]: row["context_limit"]
                for row in attempts
                if row["task_type"] == "workspace_needle"
            }
            self.assertEqual(
                needle_contexts,
                {
                    "workspace-needle-4k": 4096,
                    "workspace-needle-8k": 8192,
                    "workspace-needle-16k": 16384,
                    "workspace-needle-32k": 32768,
                    "workspace-needle-64k": 65536,
                },
            )

    def test_failed_serve_command_records_model_load_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_config = Path(tmp) / "models.json"
            model_config.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "model_id": "broken-local-model",
                                "served_model_name": "broken-local-model",
                                "serve_command": [sys.executable, "-c", "import sys, time; time.sleep(0.1); sys.exit(2)"],
                                "health_check_url": "http://127.0.0.1:9/health"
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            out = Path(tmp) / "results"
            cmd = [
                sys.executable,
                "-m",
                "openclaw_bench",
                "run",
                "--backend",
                "simulator",
                "--suite",
                str(ROOT / "manifests" / "openclaw-agent-core.json"),
                "--model-config",
                str(model_config),
                "--kv",
                "fp8",
                "--concurrency",
                "1",
                "--contexts",
                "4096",
                "--out",
                str(out),
                "--run-id",
                "failed-serve",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            run_dir = out / "failed-serve"
            failures = (run_dir / "failures.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(failures), 5)
            first = json.loads(failures[0])
            self.assertEqual(first["failure_type"], "model_load_failed")
            server = json.loads((run_dir / "server.json").read_text(encoding="utf-8"))
            self.assertEqual(server["serve_results"][0]["load_success"], False)

    def test_failed_serve_command_classifies_oom_on_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_config = Path(tmp) / "models.json"
            model_config.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "model_id": "oom-local-model",
                                "served_model_name": "oom-local-model",
                                "serve_command": [
                                    sys.executable,
                                    "-c",
                                    "import sys; print('CUDA out of memory while allocating KV cache', file=sys.stderr); sys.exit(2)",
                                ],
                                "health_check_url": "http://127.0.0.1:9/health"
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            out = Path(tmp) / "results"
            cmd = [
                sys.executable,
                "-m",
                "openclaw_bench",
                "run",
                "--backend",
                "simulator",
                "--suite",
                str(ROOT / "manifests" / "openclaw-agent-core.json"),
                "--model-config",
                str(model_config),
                "--kv",
                "fp8",
                "--concurrency",
                "1",
                "--contexts",
                "4096",
                "--out",
                str(out),
                "--run-id",
                "oom-serve",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            run_dir = out / "oom-serve"
            first = json.loads((run_dir / "failures.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(first["failure_type"], "oom_on_load")
            self.assertTrue(first["oom"])
            server = json.loads((run_dir / "server.json").read_text(encoding="utf-8"))
            self.assertEqual(server["serve_results"][0]["failure_type"], "oom_on_load")

    def test_unsupported_kv_records_taxonomy_without_attempting_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_config = Path(tmp) / "models.json"
            model_config.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "model_id": "qwen3-dense",
                                "served_model_name": "qwen3-dense",
                                "kv_modes": ["fp8", "turboquant_k8v4"],
                                "kv_support": {"turboquant_k8v4": "unsupported"},
                                "contexts": [4096]
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            out = Path(tmp) / "results"
            cmd = [
                sys.executable,
                "-m",
                "openclaw_bench",
                "run",
                "--backend",
                "simulator",
                "--suite",
                str(ROOT / "manifests" / "openclaw-agent-core.json"),
                "--model-config",
                str(model_config),
                "--out",
                str(out),
                "--run-id",
                "unsupported-kv",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            failures = (out / "unsupported-kv" / "failures.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(failures), 5)
            self.assertEqual(json.loads(failures[0])["failure_type"], "unsupported_kv_dtype")

    def test_openclaw_gateway_run_requires_workspace_agents(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results"
            cmd = [
                sys.executable,
                "-m",
                "openclaw_bench",
                "run",
                "--backend",
                "openclaw",
                "--no-openclaw-workspace-agents",
                "--suite",
                str(ROOT / "manifests" / "openclaw-agent-discovery-smoke.example.json"),
                "--models",
                "simulated-model",
                "--out",
                str(out),
                "--run-id",
                "bad-gateway-default",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("require --openclaw-workspace-agents", proc.stderr)

    def test_openclaw_run_refuses_unverified_local_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results"
            cmd = [
                sys.executable,
                "-m",
                "openclaw_bench",
                "run",
                "--backend",
                "openclaw",
                "--openclaw-local",
                "--suite",
                str(ROOT / "manifests" / "openclaw-agent-core.json"),
                "--model-config",
                str(ROOT / "manifests" / "initial-models.json"),
                "--out",
                str(out),
                "--run-id",
                "unverified-local",
            ]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            failures = (out / "unverified-local" / "failures.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(failures), 20)
            self.assertEqual(json.loads(failures[0])["failure_type"], "model_load_failed")

    def test_openclaw_run_refuses_model_with_missing_declared_api_env(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        model = ModelSpec(
            model_id="local-vllm",
            served_model_name="local-vllm",
            openclaw_model_name="vllm/local-vllm",
            serve_command=[sys.executable, "-c", "import time; time.sleep(5)"],
            health_check_url="http://127.0.0.1:8000/v1/models",
            api_base="http://127.0.0.1:8000/v1",
            api_env="VLLM_API_KEY",
            context_limit=4096,
        )
        with tempfile.TemporaryDirectory() as tmp:
            config = RunConfig(
                run_id="missing-local-api-env",
                suite=suite,
                models=[model],
                kv_modes=["fp8"],
                contexts=[4096],
                concurrencies=[1],
                out_dir=Path(tmp) / "results",
                workspace_root=Path(tmp) / "workspaces",
                fixtures_root=ROOT / "fixtures",
                backend_name="openclaw",
                openclaw_local=True,
            )
            with patch.dict("os.environ", {}, clear=True):
                results = BenchmarkRunner(_PassingSmokeSimulatorBackend()).run(config)
            self.assertEqual(len(results), 5)
            self.assertTrue(all(result.failure_type == "model_route_failed" for result in results))
            server = json.loads((config.out_dir / "server.json").read_text(encoding="utf-8"))
            self.assertEqual(server["serve_results"][0]["failure_type"], "model_route_failed")
            self.assertIn("VLLM_API_KEY", server["serve_results"][0]["notes"])

    def test_openclaw_run_route_smoke_failure_prevents_full_task_suite(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        model = ModelSpec(
            model_id="api-model",
            served_model_name="provider/api-model",
            provider_type="api",
            kv_cache_dtype="provider_default",
            context_limit=4096,
            api_env="API_MODEL_KEY",
        )
        with tempfile.TemporaryDirectory() as tmp:
            config = RunConfig(
                run_id="route-smoke-fail",
                suite=suite,
                models=[model],
                kv_modes=["provider_default"],
                contexts=[4096],
                concurrencies=[1],
                out_dir=Path(tmp) / "results",
                workspace_root=Path(tmp) / "workspaces",
                fixtures_root=ROOT / "fixtures",
                backend_name="openclaw",
            )
            backend = _FailingSmokeBackend()
            with patch.dict("os.environ", {"API_MODEL_KEY": "set"}):
                results = BenchmarkRunner(backend).run(config)
            self.assertEqual(backend.run_calls, 0)
            self.assertEqual(len(results), 5)
            self.assertTrue(all(result.failure_type == "model_route_failed" for result in results))
            self.assertTrue(all("OpenClaw route smoke failed" in result.notes for result in results))
            self.assertTrue(all(not result.timeout for result in results))
            self.assertTrue(all(result.request_errors == 1 for result in results))
            server = json.loads((config.out_dir / "server.json").read_text(encoding="utf-8"))
            serve_result = server["serve_results"][0]
            self.assertEqual(serve_result["load_success"], False)
            self.assertEqual(serve_result["failure_type"], "model_route_failed")
            self.assertEqual(serve_result["request_errors"], 1)
            self.assertEqual(serve_result["route_probe"]["openclaw_route"]["success"], False)

    def test_openclaw_run_route_smoke_timeout_marks_attempt_rows_timeout(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        model = ModelSpec(
            model_id="api-model",
            served_model_name="provider/api-model",
            provider_type="api",
            kv_cache_dtype="provider_default",
            context_limit=4096,
            api_env="API_MODEL_KEY",
        )
        with tempfile.TemporaryDirectory() as tmp:
            config = RunConfig(
                run_id="route-smoke-timeout",
                suite=suite,
                models=[model],
                kv_modes=["provider_default"],
                contexts=[4096],
                concurrencies=[1],
                out_dir=Path(tmp) / "results",
                workspace_root=Path(tmp) / "workspaces",
                fixtures_root=ROOT / "fixtures",
                backend_name="openclaw",
            )
            backend = _TimeoutSmokeBackend()
            with patch.dict("os.environ", {"API_MODEL_KEY": "set"}):
                results = BenchmarkRunner(backend).run(config)
            self.assertEqual(len(results), 5)
            self.assertTrue(all(result.failure_type == "openclaw_timeout" for result in results))
            self.assertTrue(all(result.timeout for result in results))
            self.assertTrue(all(result.request_errors == 1 for result in results))

    def test_openclaw_run_route_smoke_success_allows_task_suite(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        model = ModelSpec(
            model_id="api-model",
            served_model_name="provider/api-model",
            provider_type="api",
            kv_cache_dtype="provider_default",
            context_limit=4096,
            api_env="API_MODEL_KEY",
        )
        with tempfile.TemporaryDirectory() as tmp:
            config = RunConfig(
                run_id="route-smoke-pass",
                suite=suite,
                models=[model],
                kv_modes=["provider_default"],
                contexts=[4096],
                concurrencies=[1],
                out_dir=Path(tmp) / "results",
                workspace_root=Path(tmp) / "workspaces",
                fixtures_root=ROOT / "fixtures",
                backend_name="openclaw",
            )
            backend = _PassingSmokeSimulatorBackend()
            with patch.dict("os.environ", {"API_MODEL_KEY": "set"}):
                results = BenchmarkRunner(backend).run(config)
            self.assertEqual(backend.run_calls, 5)
            self.assertEqual(sum(1 for result in results if result.status == "pass"), 5)
            server = json.loads((config.out_dir / "server.json").read_text(encoding="utf-8"))
            self.assertEqual(server["serve_results"][0]["route_probe"]["openclaw_route"]["success"], True)

    def test_openclaw_run_uses_configured_route_smoke_timeout(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        model = ModelSpec(
            model_id="api-model",
            served_model_name="provider/api-model",
            provider_type="api",
            kv_cache_dtype="provider_default",
            context_limit=4096,
            api_env="API_MODEL_KEY",
        )
        with tempfile.TemporaryDirectory() as tmp:
            config = RunConfig(
                run_id="route-smoke-custom-timeout",
                suite=suite,
                models=[model],
                kv_modes=["provider_default"],
                contexts=[4096],
                concurrencies=[1],
                out_dir=Path(tmp) / "results",
                workspace_root=Path(tmp) / "workspaces",
                fixtures_root=ROOT / "fixtures",
                backend_name="openclaw",
                timeout_s=180,
                openclaw_smoke_timeout_s=123,
            )
            backend = _PassingSmokeSimulatorBackend()
            with patch.dict("os.environ", {"API_MODEL_KEY": "set"}):
                BenchmarkRunner(backend).run(config)
            self.assertEqual(backend.smoke_timeouts, [123])
            saved_config = json.loads((config.out_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_config["openclaw_smoke_timeout_s"], 123)


class _FailingSmokeBackend:
    def __init__(self):
        self.run_calls = 0

    def smoke(self, model, timeout_s):
        del model, timeout_s
        return BackendResponse(
            text="",
            json_output=None,
            raw={"returncode": 1},
            error="model_route_failed",
        )

    def run(self, model, task, workspace, session_id, timeout_s):
        del model, task, workspace, session_id, timeout_s
        self.run_calls += 1
        raise AssertionError("task suite should not run after route smoke failure")


class _TimeoutSmokeBackend:
    def smoke(self, model, timeout_s):
        del model, timeout_s
        return BackendResponse(
            text="",
            json_output=None,
            raw={"timeout": True},
            timed_out=True,
            error="openclaw_timeout",
        )

    def run(self, model, task, workspace, session_id, timeout_s):
        del model, task, workspace, session_id, timeout_s
        raise AssertionError("task suite should not run after route smoke timeout")


class _PassingSmokeSimulatorBackend:
    def __init__(self):
        self.simulator = SimulatorBackend()
        self.run_calls = 0
        self.smoke_timeouts = []

    def smoke(self, model, timeout_s):
        self.smoke_timeouts.append(timeout_s)
        return BackendResponse(
            text="ok",
            json_output={"text": "ok"},
            raw={"model": model.served_model_name},
        )

    def run(self, model, task, workspace, session_id, timeout_s):
        self.run_calls += 1
        return self.simulator.run(model, task, workspace, session_id, timeout_s)


class _SessionIdCaptureBackend:
    def __init__(self):
        self.simulator = SimulatorBackend()
        self.session_ids = []

    def smoke(self, model, timeout_s):
        del timeout_s
        return BackendResponse(text="ok", json_output={"text": "ok"}, raw={"model": model.served_model_name})

    def run(self, model, task, workspace, session_id, timeout_s):
        self.session_ids.append(session_id)
        return self.simulator.run(model, task, workspace, session_id, timeout_s)


class _WorkspaceSeedCaptureBackend:
    def __init__(self):
        self.simulator = SimulatorBackend()
        self.seed_files_seen = {}
        self.identity_text = ""

    def run(self, model, task, workspace, session_id, timeout_s):
        self.seed_files_seen = {
            "AGENTS.md": (workspace / "AGENTS.md").exists(),
            "SOUL.md": (workspace / "SOUL.md").exists(),
            ".openclaw/workspace-state.json": (workspace / ".openclaw" / "workspace-state.json").exists(),
            "BOOTSTRAP.md": (workspace / "BOOTSTRAP.md").exists(),
        }
        self.identity_text = (workspace / "IDENTITY.md").read_text(encoding="utf-8")
        return self.simulator.run(model, task, workspace, session_id, timeout_s)


def _run_args(**overrides):
    values = {
        "backend": "simulator",
        "suite": str(ROOT / "manifests" / "openclaw-agent-discovery-smoke.example.json"),
        "models": "simulated-model",
        "model_config": None,
        "kv": "fp8",
        "concurrency": "1",
        "contexts": "4096",
        "out": "/tmp/openclaw-bench-test-results",
        "workspace_root": None,
        "fixtures_root": str(ROOT / "fixtures"),
        "openclaw_profile": "bench",
        "openclaw_agent": "dev",
        "openclaw_local": False,
        "openclaw_container": "oc-bench-gateway",
        "ensure_openclaw_container": True,
        "openclaw_container_image": "clawdaddy/openclaw:business-smoke-2026.4.27",
        "openclaw_container_home": None,
        "openclaw_container_gateway_port": 19091,
        "openclaw_container_token": None,
        "openclaw_gateway_timeout": 60,
        "ensure_openclaw_gateway": True,
        "openclaw_workspace_agents": False,
        "thinking": None,
        "run_id": "test-run",
        "timeout": 300,
        "openclaw_smoke_timeout": 60,
    }
    values.update(overrides)
    return Namespace(**values)


def _container_check(notes: str = "oc-bench-gateway already running"):
    return PreflightCheck("openclaw_container", "pass", notes)


if __name__ == "__main__":
    unittest.main()
