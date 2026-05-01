import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openclaw_bench.container import _docker_run_command, ensure_openclaw_container
from openclaw_bench.models import ModelSpec


class ContainerEnsureTests(unittest.TestCase):
    def test_existing_running_container_is_reused(self):
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="true\n", stderr="")
        with tempfile.TemporaryDirectory() as tmp:
            with patch("subprocess.run", return_value=completed) as run_mock:
                result = ensure_openclaw_container(
                    container="oc-bench-gateway",
                    image="clawdaddy/openclaw:business-smoke",
                    profile="bench",
                    project_root=Path(tmp) / "repo",
                    bench_root=Path(tmp) / "bench",
                    workspace_root=Path(tmp) / "bench" / "workspaces" / "preflight",
                )

        self.assertEqual(result.status, "pass")
        self.assertIn("already running", result.notes)
        self.assertEqual(run_mock.call_args.args[0], ["docker", "inspect", "-f", "{{.State.Running}}", "oc-bench-gateway"])

    def test_stopped_container_is_started_not_recreated(self):
        inspect = subprocess.CompletedProcess(args=[], returncode=0, stdout="false\n", stderr="")
        started = subprocess.CompletedProcess(args=[], returncode=0, stdout="oc-bench-gateway\n", stderr="")
        with tempfile.TemporaryDirectory() as tmp:
            with patch("subprocess.run", side_effect=[inspect, started]) as run_mock:
                result = ensure_openclaw_container(
                    container="oc-bench-gateway",
                    image="clawdaddy/openclaw:business-smoke",
                    profile="bench",
                    project_root=Path(tmp) / "repo",
                    bench_root=Path(tmp) / "bench",
                    workspace_root=Path(tmp) / "bench" / "workspaces" / "preflight",
                )

        self.assertEqual(result.status, "pass")
        self.assertEqual(run_mock.call_args_list[1].args[0], ["docker", "start", "oc-bench-gateway"])

    def test_missing_container_is_created_with_bench_profile_gateway_and_healthcheck(self):
        inspect = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="not found")
        created = subprocess.CompletedProcess(args=[], returncode=0, stdout="container-id\n", stderr="")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai"}, clear=False):
                with patch("subprocess.run", side_effect=[inspect, created]) as run_mock:
                    result = ensure_openclaw_container(
                        container="oc-bench-gateway",
                        image="clawdaddy/openclaw:business-smoke",
                        profile="bench",
                        project_root=root / "repo",
                        bench_root=root / "bench",
                        workspace_root=root / "bench" / "workspaces" / "preflight",
                        models=[ModelSpec.from_mapping({"model_id": "api", "served_model_name": "api", "api_env": "OPENAI_API_KEY"})],
                    )

        self.assertEqual(result.status, "pass")
        cmd = run_mock.call_args_list[1].args[0]
        self.assertEqual(cmd[:5], ["docker", "run", "-d", "--name", "oc-bench-gateway"])
        self.assertIn("--network", cmd)
        self.assertIn("host", cmd)
        self.assertIn("--health-cmd", cmd)
        self.assertIn("openclaw --profile bench gateway status", cmd[cmd.index("--health-cmd") + 1])
        self.assertIn("Connectivity probe: ok", cmd[cmd.index("--health-cmd") + 1])
        self.assertIn("-e", cmd)
        self.assertIn("OPENAI_API_KEY=test-openai", cmd)
        self.assertIn("VLLM_API_KEY=vllm-local", cmd)
        self.assertIn("clawdaddy/openclaw:business-smoke", cmd)
        self.assertEqual(cmd[-3:], ["sh", "-lc", "sleep infinity"])

    def test_docker_run_mounts_workspace_outside_bench_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cmd = _docker_run_command(
                container="oc-bench-gateway",
                image="openclaw-image",
                profile="bench",
                project_root=root / "repo",
                bench_root=root / "bench",
                workspace_root=root / "custom-workspaces",
                container_home=root / "bench" / "container-home",
                gateway_port=19091,
                gateway_token="token",
                models=[],
            )

        volume_args = [cmd[index + 1] for index, value in enumerate(cmd) if value == "-v"]
        self.assertTrue(any("custom-workspaces" in volume for volume in volume_args))


if __name__ == "__main__":
    unittest.main()
