import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openclaw_bench.cli import init_command, quickstart_command
from openclaw_bench.preflight import PreflightCheck
from openclaw_bench.quickstart import (
    DEFAULT_PROFILE,
    choose_safe_port,
    detect_existing_profiles,
    init_quickstart,
    stop_benchclaw_gateway,
)


ROOT = Path(__file__).resolve().parent.parent


class QuickstartTests(unittest.TestCase):
    def test_init_generates_isolated_benchclaw_config_and_starter_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = init_quickstart(
                providers="both",
                project_root=ROOT,
                bench_root=root / "bench",
                home=root / "home",
                port=19222,
                validate=False,
            )

            self.assertEqual(result.profile, DEFAULT_PROFILE)
            self.assertEqual(result.providers, "both")
            self.assertEqual(result.port, 19222)
            self.assertEqual(result.paths.config_path, root / "home" / ".openclaw-benchclaw" / "openclaw.json")
            self.assertTrue(result.paths.suite_path.exists())
            self.assertTrue(result.paths.model_config_path.exists())
            self.assertTrue(result.paths.metadata_path.exists())
            self.assertTrue((result.paths.fixtures_root / "discovery_repo" / "api" / "routes.py").exists())

            config = json.loads(result.paths.config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["gateway"]["mode"], "local")
            self.assertEqual(config["gateway"]["bind"], "loopback")
            self.assertEqual(config["gateway"]["port"], 19222)
            self.assertEqual(config["gateway"]["auth"]["mode"], "none")
            self.assertEqual(set(config["models"]["providers"]), {"vllm", "openai", "anthropic"})
            self.assertEqual(config["agents"]["list"][0]["id"], "bench")

            model_config = json.loads(result.paths.model_config_path.read_text(encoding="utf-8"))
            provider_types = {model["provider_type"] for model in model_config["models"]}
            self.assertEqual(provider_types, {"local", "api", "subscription"})
            self.assertEqual(model_config["models"][0]["provider_type"], "local")
            self.assertIn("OAuth", model_config["manifest_scope"]["notes"])

            suite = json.loads(result.paths.suite_path.read_text(encoding="utf-8"))
            self.assertEqual(suite["suite_id"], "openclaw-agent-discovery-smoke")

    def test_init_refuses_to_overwrite_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kwargs = {
                "providers": "local",
                "project_root": ROOT,
                "bench_root": root / "bench",
                "home": root / "home",
                "port": 19223,
                "validate": False,
            }
            init_quickstart(**kwargs)
            with self.assertRaisesRegex(ValueError, "already exists"):
                init_quickstart(**kwargs)
            init_quickstart(**kwargs, force=True)

    def test_detect_existing_profiles_lists_only_profiles_with_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / ".openclaw").mkdir()
            (home / ".openclaw" / "openclaw.json").write_text("{}", encoding="utf-8")
            (home / ".openclaw-bench").mkdir()
            (home / ".openclaw-bench" / "openclaw.json").write_text("{}", encoding="utf-8")
            (home / ".openclaw-empty").mkdir()

            self.assertEqual(detect_existing_profiles(home), ["default", "bench"])

    def test_choose_safe_port_skips_bound_loopback_port(self):
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            occupied = sock.getsockname()[1]
            self.assertEqual(choose_safe_port(occupied, limit=2), occupied + 1)

    def test_stop_only_calls_benchclaw_gateway_stop(self):
        completed = _completed(returncode=0, stdout="stopped\n")
        with patch("subprocess.run", return_value=completed) as run_mock:
            check = stop_benchclaw_gateway("benchclaw")

        self.assertEqual(check.status, "pass")
        self.assertEqual(run_mock.call_args.args[0], ["openclaw", "--profile", "benchclaw", "gateway", "stop"])

    def test_cli_init_uses_noninteractive_provider_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _init_args(tmp, providers="api", no_validate=True)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = init_command(args)

            self.assertEqual(code, 0)
            self.assertIn("profile=benchclaw", stdout.getvalue())
            self.assertIn("providers=api", stdout.getvalue())
            config = json.loads((Path(tmp) / "home" / ".openclaw-benchclaw" / "openclaw.json").read_text(encoding="utf-8"))
            self.assertEqual(set(config["models"]["providers"]), {"openai", "anthropic"})

    def test_quickstart_orchestrates_init_start_preflight_run_and_optional_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _init_args(tmp, providers="api", no_validate=True)
            args.run_id = "starter"
            args.timeout = 300
            args.smoke_timeout = 60
            args.openclaw_gateway_timeout = 60
            args.stop_after = True
            args.backend = "openclaw"
            start = PreflightCheck("openclaw_gateway", "pass", "started")
            stop = PreflightCheck("openclaw_gateway_stop", "pass", "stopped")
            with patch("openclaw_bench.cli.start_benchclaw_gateway", return_value=start) as start_mock:
                with patch("openclaw_bench.cli.preflight_command", return_value=0) as preflight_mock:
                    with patch("openclaw_bench.cli.run_command", return_value=0) as run_mock:
                        with patch("openclaw_bench.cli.stop_benchclaw_gateway", return_value=stop) as stop_mock:
                            stdout = io.StringIO()
                            with redirect_stdout(stdout):
                                code = quickstart_command(args)

            self.assertEqual(code, 0)
            start_mock.assert_called_once_with("benchclaw", timeout_s=60)
            preflight_mock.assert_called_once()
            run_mock.assert_called_once()
            stop_mock.assert_called_once_with("benchclaw", timeout_s=30)
            self.assertEqual(run_mock.call_args.args[0].run_id, "starter")
            self.assertEqual(preflight_mock.call_args.args[0].fixtures_root, str(Path(tmp) / "bench" / "fixtures"))
            self.assertIn("result_path=", stdout.getvalue())

    def test_quickstart_reuses_existing_init_config_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_quickstart(
                providers="local",
                project_root=ROOT,
                bench_root=root / "bench",
                home=root / "home",
                port=19224,
                validate=False,
            )
            args = _init_args(tmp, providers="local", no_validate=True)
            args.run_id = "starter"
            args.timeout = 300
            args.smoke_timeout = 60
            args.openclaw_gateway_timeout = 60
            args.stop_after = False
            args.backend = "openclaw"

            with patch("openclaw_bench.cli.start_benchclaw_gateway", return_value=PreflightCheck("openclaw_gateway", "pass", "started")):
                with patch("openclaw_bench.cli.preflight_command", return_value=0):
                    with patch("openclaw_bench.cli.run_command", return_value=0):
                        with redirect_stdout(io.StringIO()):
                            code = quickstart_command(args)

            self.assertEqual(code, 0)


def _init_args(tmp: str, providers: str, no_validate: bool):
    args = SimpleNamespace()
    args.providers = providers
    args.bench_root = str(Path(tmp) / "bench")
    args.config_home = str(Path(tmp) / "home")
    args.openclaw_profile = "benchclaw"
    args.openclaw_agent = "bench"
    args.gateway_port = 19224
    args.force = False
    args.no_validate = no_validate
    return args


def _completed(returncode: int, stdout: str = "", stderr: str = ""):
    import subprocess

    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


if __name__ == "__main__":
    unittest.main()
