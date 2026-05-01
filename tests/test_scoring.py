import json
import tempfile
import unittest
from pathlib import Path

from openclaw_bench.backend import SimulatorBackend
from openclaw_bench.manifest import load_suite
from openclaw_bench.models import BackendResponse, ModelSpec, TaskSpec
from openclaw_bench.scoring import run_verify_command, score_task
from openclaw_bench.workspace import changed_files, copy_fixture, snapshot_files


ROOT = Path(__file__).resolve().parent.parent


class ScoringTests(unittest.TestCase):
    def test_simulated_bug_trace_scores_pass_after_tests(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        task = next(item for item in suite.tasks if item.task_id == "multi-file-bug-trace")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            copy_fixture(ROOT / "fixtures" / task.fixture, workspace)
            before = snapshot_files(workspace)
            response = SimulatorBackend().run(ModelSpec.from_alias("simulated-model", "fp8", 4096), task, workspace, "session", 60)
            changed = changed_files(before, snapshot_files(workspace))
            tests_passed, _ = run_verify_command(workspace, task.verify_command)
            score, failure, hallucinated, notes = score_task(task, workspace, response, changed, tests_passed)
            self.assertEqual(score, 1.0, notes)
            self.assertIsNone(failure)
            self.assertEqual(hallucinated, 0)

    def test_simulated_patch_scores_pass_after_tests(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        task = next(item for item in suite.tasks if item.task_id == "patch-execution")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            copy_fixture(ROOT / "fixtures" / task.fixture, workspace)
            before = snapshot_files(workspace)
            response = SimulatorBackend().run(ModelSpec.from_alias("simulated-model", "fp8", 4096), task, workspace, "session", 60)
            changed = changed_files(before, snapshot_files(workspace))
            tests_passed, _ = run_verify_command(workspace, task.verify_command)
            score, failure, hallucinated, notes = score_task(task, workspace, response, changed, tests_passed)
            self.assertEqual(score, 1.0, notes)
            self.assertIsNone(failure)
            self.assertEqual(hallucinated, 0)

    def test_hallucinated_json_path_fails_even_when_expected_fields_match(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        task = next(item for item in suite.tasks if item.task_id == "workspace-discovery")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            copy_fixture(ROOT / "fixtures" / task.fixture, workspace)
            payload = {
                "test_command": "python -m unittest discover -s tests",
                "routes_file": "api/routes.py",
                "schema_file": "db/schema.py",
                "extra_file": "ghost.py",
            }
            response = BackendResponse(text="", json_output=payload, raw={})
            score, failure, hallucinated, notes = score_task(task, workspace, response, [], True)
            self.assertEqual(score, 0.0, notes)
            self.assertEqual(failure, "hallucinated_file")
            self.assertEqual(hallucinated, 1)

    def test_discovery_requires_returned_test_command_to_run(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        task = next(item for item in suite.tasks if item.task_id == "workspace-discovery")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            copy_fixture(ROOT / "fixtures" / task.fixture, workspace)
            (workspace / "tests" / "test_api.py").write_text(
                "import unittest\n\n"
                "class BrokenDiscoveryCommand(unittest.TestCase):\n"
                "    def test_command_fails(self):\n"
                "        self.fail('discovery command is stale')\n",
                encoding="utf-8",
            )
            payload = {
                "test_command": "python -m unittest discover -s tests",
                "routes_file": "api/routes.py",
                "schema_file": "db/schema.py",
            }
            response = BackendResponse(text="", json_output=payload, raw={})
            score, failure, hallucinated, notes = score_task(task, workspace, response, [], True)
            self.assertLess(score, 1.0)
            self.assertEqual(failure, "wrong_file")
            self.assertEqual(hallucinated, 0)
            self.assertIn("test_command was not runnable", notes)

    def test_discovery_accepts_equivalent_runnable_test_command(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        task = next(item for item in suite.tasks if item.task_id == "workspace-discovery")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            copy_fixture(ROOT / "fixtures" / task.fixture, workspace)
            payload = {
                "test_command": "python -m unittest discover -s tests -p test_api.py",
                "routes_file": "api/routes.py",
                "schema_file": "db/schema.py",
            }
            response = BackendResponse(text="", json_output=payload, raw={})
            score, failure, hallucinated, notes = score_task(task, workspace, response, [], True)
            self.assertEqual(score, 1.0, notes)
            self.assertIsNone(failure)
            self.assertEqual(hallucinated, 0)

    def test_discovery_accepts_python_test_file_command(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        task = next(item for item in suite.tasks if item.task_id == "workspace-discovery")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            copy_fixture(ROOT / "fixtures" / task.fixture, workspace)
            payload = {
                "test_command": "python tests/test_api.py",
                "routes_file": "api/routes.py",
                "schema_file": "db/schema.py",
            }
            response = BackendResponse(text="", json_output=payload, raw={})
            score, failure, hallucinated, notes = score_task(task, workspace, response, [], True)
            self.assertEqual(score, 1.0, notes)
            self.assertIsNone(failure)
            self.assertEqual(hallucinated, 0)

    def test_discovery_accepts_unittest_module_command(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        task = next(item for item in suite.tasks if item.task_id == "workspace-discovery")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            copy_fixture(ROOT / "fixtures" / task.fixture, workspace)
            payload = {
                "test_command": "python -m unittest tests.test_api",
                "routes_file": "api/routes.py",
                "schema_file": "db/schema.py",
            }
            response = BackendResponse(text="", json_output=payload, raw={})
            score, failure, hallucinated, notes = score_task(task, workspace, response, [], True)
            self.assertEqual(score, 1.0, notes)
            self.assertIsNone(failure)
            self.assertEqual(hallucinated, 0)

    def test_discovery_does_not_execute_unexpected_test_command(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        task = next(item for item in suite.tasks if item.task_id == "workspace-discovery")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            copy_fixture(ROOT / "fixtures" / task.fixture, workspace)
            marker = workspace / "command-was-executed"
            payload = {
                "test_command": "python -c \"from pathlib import Path; Path('command-was-executed').write_text('bad')\"",
                "routes_file": "api/routes.py",
                "schema_file": "db/schema.py",
            }
            response = BackendResponse(text="", json_output=payload, raw={})
            score, failure, hallucinated, notes = score_task(task, workspace, response, [], True)
            self.assertLess(score, 1.0)
            self.assertEqual(failure, "wrong_file")
            self.assertEqual(hallucinated, 0)
            self.assertFalse(marker.exists())
            self.assertIn("test_command was not a safe test command", notes)

    def test_discovery_rejects_bare_test_file_as_command(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        task = next(item for item in suite.tasks if item.task_id == "workspace-discovery")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            copy_fixture(ROOT / "fixtures" / task.fixture, workspace)
            payload = {
                "test_command": "tests/test_api.py",
                "routes_file": "api/routes.py",
                "schema_file": "db/schema.py",
            }
            response = BackendResponse(text="", json_output=payload, raw={})
            score, failure, hallucinated, notes = score_task(task, workspace, response, [], True)
            self.assertLess(score, 1.0)
            self.assertEqual(failure, "wrong_file")
            self.assertEqual(hallucinated, 0)
            self.assertIn("test_command was not a safe test command", notes)

    def test_discovery_rejects_test_command_outside_workspace_tests(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        task = next(item for item in suite.tasks if item.task_id == "workspace-discovery")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            copy_fixture(ROOT / "fixtures" / task.fixture, workspace)
            payload = {
                "test_command": "python -m unittest discover -s /tmp",
                "routes_file": "api/routes.py",
                "schema_file": "db/schema.py",
            }
            response = BackendResponse(text="", json_output=payload, raw={})
            score, failure, hallucinated, notes = score_task(task, workspace, response, [], True)
            self.assertLess(score, 1.0)
            self.assertEqual(failure, "wrong_file")
            self.assertEqual(hallucinated, 0)
            self.assertIn("test_command was not a safe test command", notes)

    def test_repo_read_only_scores_expected_answer_and_evidence(self):
        suite = load_suite(ROOT / "manifests" / "real-repo-readonly.example.json")
        task = next(item for item in suite.tasks if item.task_id == "real-repo-auth-guard")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            copy_fixture(ROOT / "fixtures" / task.fixture, workspace)
            payload = {"answer": "api/auth.py", "evidence_files": ["api/auth.py", "api/routes.py"]}
            response = BackendResponse(text="", json_output=payload, raw={})
            score, failure, hallucinated, notes = score_task(task, workspace, response, [], True)
            self.assertEqual(score, 1.0, notes)
            self.assertIsNone(failure)
            self.assertEqual(hallucinated, 0)

    def test_repo_read_only_accepts_openclaw_wrapped_json_text(self):
        suite = load_suite(ROOT / "manifests" / "real-repo-readonly.example.json")
        task = next(item for item in suite.tasks if item.task_id == "real-repo-route-map")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            copy_fixture(ROOT / "fixtures" / task.fixture, workspace)
            payload = {"answer": "api/routes.py", "evidence_files": ["api/routes.py"]}
            response = BackendResponse(text=json.dumps(payload), json_output={"text": json.dumps(payload)}, raw={})
            score, failure, hallucinated, notes = score_task(task, workspace, response, [], True)
            self.assertEqual(score, 1.0, notes)
            self.assertIsNone(failure)
            self.assertEqual(hallucinated, 0)

    def test_repo_read_only_rejects_invalid_wrapped_json_as_bad_json(self):
        suite = load_suite(ROOT / "manifests" / "real-repo-readonly.example.json")
        task = next(item for item in suite.tasks if item.task_id == "real-repo-route-map")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            copy_fixture(ROOT / "fixtures" / task.fixture, workspace)
            response = BackendResponse(text="{not json", json_output={"text": "{not json"}, raw={})
            score, failure, hallucinated, notes = score_task(task, workspace, response, [], True)
            self.assertLess(score, 1.0)
            self.assertEqual(failure, "bad_json")
            self.assertEqual(hallucinated, 0)
            self.assertIn("response was not valid JSON", notes)

    def test_repo_read_only_rejects_file_changes(self):
        suite = load_suite(ROOT / "manifests" / "real-repo-readonly.example.json")
        task = next(item for item in suite.tasks if item.task_id == "real-repo-route-map")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            copy_fixture(ROOT / "fixtures" / task.fixture, workspace)
            payload = {"answer": "api/routes.py", "evidence_files": ["api/routes.py"]}
            response = BackendResponse(text="", json_output=payload, raw={})
            score, failure, hallucinated, notes = score_task(task, workspace, response, ["api/routes.py"], True)
            self.assertLess(score, 1.0)
            self.assertEqual(failure, "instruction_violation")
            self.assertEqual(hallucinated, 0)
            self.assertIn("read-only task changed files: api/routes.py", notes)

    def test_repo_read_only_rejects_wrong_evidence(self):
        suite = load_suite(ROOT / "manifests" / "real-repo-readonly.example.json")
        task = next(item for item in suite.tasks if item.task_id == "real-repo-auth-guard")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            copy_fixture(ROOT / "fixtures" / task.fixture, workspace)
            payload = {"answer": "api/routes.py", "evidence_files": ["api/routes.py"]}
            response = BackendResponse(text="", json_output=payload, raw={})
            score, failure, hallucinated, notes = score_task(task, workspace, response, [], True)
            self.assertLess(score, 1.0)
            self.assertEqual(failure, "wrong_file")
            self.assertEqual(hallucinated, 0)
            self.assertIn("answer did not match expected file", notes)
            self.assertIn("evidence_files missing expected file", notes)

    def test_repo_code_edit_scores_expected_real_repo_patch(self):
        suite = load_suite(ROOT / "manifests" / "real-repo-readonly.example.json")
        task = next(item for item in suite.tasks if item.task_id == "real-repo-order-status-edit")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            copy_fixture(ROOT / "fixtures" / task.fixture, workspace)
            before = snapshot_files(workspace)
            response = SimulatorBackend().run(ModelSpec.from_alias("simulated-model", "fp8", 4096), task, workspace, "session", 60)
            changed = changed_files(before, snapshot_files(workspace))
            tests_passed, _ = run_verify_command(workspace, task.verify_command)
            score, failure, hallucinated, notes = score_task(task, workspace, response, changed, tests_passed)
            self.assertEqual(score, 1.0, notes)
            self.assertIsNone(failure)
            self.assertEqual(hallucinated, 0)

    def test_repo_code_edit_rejects_route_file_change(self):
        suite = load_suite(ROOT / "manifests" / "real-repo-readonly.example.json")
        task = next(item for item in suite.tasks if item.task_id == "real-repo-order-status-edit")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            copy_fixture(ROOT / "fixtures" / task.fixture, workspace)
            (workspace / "services" / "orders.py").write_text(
                "def create_order(payload):\n"
                "    return {\"order_id\": payload[\"order_id\"], \"status\": \"created\"}\n\n\n"
                "def order_status(order_id):\n"
                "    status = \"shipped\" if str(order_id).startswith(\"SHIP-\") else \"processing\"\n"
                "    return {\"order_id\": order_id, \"status\": status}\n",
                encoding="utf-8",
            )
            (workspace / "api" / "routes.py").write_text("# unrelated route rewrite\n", encoding="utf-8")
            response = BackendResponse(text="Updated services/orders.py and api/routes.py.", json_output=None, raw={})
            tests_passed, _ = run_verify_command(workspace, task.verify_command)
            score, failure, hallucinated, notes = score_task(task, workspace, response, ["services/orders.py", "api/routes.py"], tests_passed)
            self.assertLess(score, 1.0)
            self.assertEqual(failure, "patch_unrelated")
            self.assertEqual(hallucinated, 0)
            self.assertIn("unexpected changed files: api/routes.py", notes)

    def test_workspace_needle_uses_manifest_source_and_target_paths(self):
        task = TaskSpec(
            task_id="custom-needle",
            task_type="workspace_needle",
            fixture="",
            prompt="",
            expected={
                "needle": "custom-token",
                "distractor": "wrong-token",
                "source_file": "docs/source.md",
                "target_file": "service/status.txt",
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            (workspace / "docs").mkdir(parents=True)
            (workspace / "service").mkdir()
            (workspace / "docs" / "source.md").write_text("BENCHMARK_NEEDLE_TOKEN=custom-token\n", encoding="utf-8")
            (workspace / "service" / "status.txt").write_text("custom-token\n", encoding="utf-8")
            response = BackendResponse(text="Found BENCHMARK_NEEDLE_TOKEN in docs/source.md.", json_output=None, raw={})

            score, failure, hallucinated, notes = score_task(task, workspace, response, ["service/status.txt"], True)

            self.assertEqual(score, 1.0, notes)
            self.assertIsNone(failure)
            self.assertEqual(hallucinated, 0)

    def test_workspace_needle_requires_manifest_source_citation(self):
        task = TaskSpec(
            task_id="custom-needle",
            task_type="workspace_needle",
            fixture="",
            prompt="",
            expected={
                "needle": "custom-token",
                "distractor": "wrong-token",
                "source_file": "docs/source.md",
                "target_file": "service/status.txt",
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            (workspace / "docs").mkdir(parents=True)
            (workspace / "service").mkdir()
            (workspace / "docs" / "source.md").write_text("BENCHMARK_NEEDLE_TOKEN=custom-token\n", encoding="utf-8")
            (workspace / "service" / "status.txt").write_text("custom-token\n", encoding="utf-8")
            response = BackendResponse(text="Found BENCHMARK_NEEDLE_TOKEN.", json_output=None, raw={})

            score, failure, hallucinated, notes = score_task(task, workspace, response, ["service/status.txt"], True)

            self.assertLess(score, 1.0)
            self.assertEqual(failure, "wrong_needle")
            self.assertEqual(hallucinated, 0)
            self.assertIn("response did not cite needle source", notes)

    def test_workspace_needle_requires_target_file_change(self):
        task = TaskSpec(
            task_id="custom-needle",
            task_type="workspace_needle",
            fixture="",
            prompt="",
            expected={
                "needle": "custom-token",
                "distractor": "wrong-token",
                "source_file": "docs/source.md",
                "target_file": "service/status.txt",
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            (workspace / "docs").mkdir(parents=True)
            (workspace / "service").mkdir()
            (workspace / "docs" / "source.md").write_text("BENCHMARK_NEEDLE_TOKEN=custom-token\n", encoding="utf-8")
            (workspace / "service" / "status.txt").write_text("custom-token\n", encoding="utf-8")
            response = BackendResponse(text="Found BENCHMARK_NEEDLE_TOKEN in docs/source.md.", json_output=None, raw={})

            score, failure, hallucinated, notes = score_task(task, workspace, response, [], True)

            self.assertLess(score, 1.0)
            self.assertEqual(failure, "wrong_needle")
            self.assertEqual(hallucinated, 0)
            self.assertIn("needle target was not changed", notes)

    def test_bug_trace_requires_structured_bug_path_files(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        task = next(item for item in suite.tasks if item.task_id == "multi-file-bug-trace")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            copy_fixture(ROOT / "fixtures" / task.fixture, workspace)
            target = workspace / "app" / "discounts.py"
            target.write_text(
                "def vip_discount_rate(customer):\n"
                "    if customer.get(\"tier\") == \"vip\":\n"
                "        return 0.10\n"
                "    return 0.0\n",
                encoding="utf-8",
            )
            response = BackendResponse(
                text="Fixed app/discounts.py after checking tests/test_discount.py.",
                json_output=None,
                raw={},
            )
            score, failure, hallucinated, notes = score_task(task, workspace, response, ["app/discounts.py"], True)
            self.assertLess(score, 1.0)
            self.assertEqual(failure, "wrong_file")
            self.assertEqual(hallucinated, 0)
            self.assertIn("bug path missing expected file: app/routes.py", notes)

    def test_patch_execution_rejects_unexpected_code_changes(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        task = next(item for item in suite.tasks if item.task_id == "patch-execution")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            copy_fixture(ROOT / "fixtures" / task.fixture, workspace)
            target = workspace / "app" / "slug.py"
            target.write_text(
                "import re\n\n\n"
                "def slugify(value):\n"
                "    value = value.strip().lower()\n"
                "    value = re.sub(r\"[^a-z0-9]+\", \"-\", value)\n"
                "    return value.strip(\"-\")\n",
                encoding="utf-8",
            )
            response = BackendResponse(text="Implemented app/slug.py.", json_output=None, raw={})
            score, failure, hallucinated, notes = score_task(
                task,
                workspace,
                response,
                ["app/slug.py", "app/__init__.py"],
                True,
            )
            self.assertLess(score, 1.0)
            self.assertEqual(failure, "patch_unrelated")
            self.assertEqual(hallucinated, 0)
            self.assertIn("unexpected changed files: app/__init__.py", notes)

    def test_multi_file_bug_trace_fails_hidden_behavior_regression(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        task = next(item for item in suite.tasks if item.task_id == "multi-file-bug-trace")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            copy_fixture(ROOT / "fixtures" / task.fixture, workspace)
            target = workspace / "app" / "discounts.py"
            target.write_text("def vip_discount_rate(customer):\n    return 0.10\n", encoding="utf-8")
            response = BackendResponse(
                text="Bug path: tests/test_discount.py -> app/routes.py -> app/discounts.py. Fixed app/discounts.py.",
                json_output=None,
                raw={},
            )
            tests_passed, _ = run_verify_command(workspace, task.verify_command)
            score, failure, hallucinated, notes = score_task(task, workspace, response, ["app/discounts.py"], tests_passed)
            self.assertLess(score, 1.0)
            self.assertEqual(failure, "test_failed")
            self.assertEqual(hallucinated, 0)
            self.assertIn("behavior check failed: app.routes.checkout_total", notes)

    def test_patch_execution_fails_visible_test_overfit(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        task = next(item for item in suite.tasks if item.task_id == "patch-execution")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            copy_fixture(ROOT / "fixtures" / task.fixture, workspace)
            target = workspace / "app" / "slug.py"
            target.write_text(
                "def slugify(value):\n"
                "    if value == ' Hello, OpenClaw Bench! ':\n"
                "        return 'hello-openclaw-bench'\n"
                "    if value == '---Already Here---':\n"
                "        return 'already-here'\n"
                "    return value\n",
                encoding="utf-8",
            )
            response = BackendResponse(text="Implemented app/slug.py.", json_output=None, raw={})
            tests_passed, _ = run_verify_command(workspace, task.verify_command)
            score, failure, hallucinated, notes = score_task(task, workspace, response, ["app/slug.py"], tests_passed)
            self.assertLess(score, 1.0)
            self.assertEqual(failure, "test_failed")
            self.assertEqual(hallucinated, 0)
            self.assertIn("behavior check failed: app.slug.slugify", notes)

    def test_instruction_retention_requires_existing_helper_import_and_call(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        task = next(item for item in suite.tasks if item.task_id == "instruction-retention")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            copy_fixture(ROOT / "fixtures" / task.fixture, workspace)
            (workspace / "app" / "service.py").write_text(
                "def normalize_status(value):\n"
                "    return str(value).strip().lower().replace(' ', '_')\n\n\n"
                "def render_status(value):\n"
                "    return {'status': normalize_status(value)}\n",
                encoding="utf-8",
            )
            response = BackendResponse(text=json.dumps({"changed": ["app/service.py"], "used_existing_helper": True}), json_output=None, raw={})
            tests_passed, _ = run_verify_command(workspace, task.verify_command)
            score, failure, hallucinated, notes = score_task(task, workspace, response, ["app/service.py"], tests_passed)
            self.assertLess(score, 1.0)
            self.assertEqual(failure, "instruction_violation")
            self.assertEqual(hallucinated, 0)
            self.assertIn("target does not import and call existing helper", notes)
            self.assertIn("target defines a new helper abstraction", notes)

    def test_instruction_retention_allows_module_import_of_existing_helper(self):
        suite = load_suite(ROOT / "manifests" / "openclaw-agent-core.json")
        task = next(item for item in suite.tasks if item.task_id == "instruction-retention")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            copy_fixture(ROOT / "fixtures" / task.fixture, workspace)
            (workspace / "app" / "service.py").write_text(
                "from app import helpers\n\n\n"
                "def render_status(value):\n"
                "    return {'status': helpers.normalize_status(value)}\n",
                encoding="utf-8",
            )
            response = BackendResponse(text=json.dumps({"changed": ["app/service.py"]}), json_output=None, raw={})
            tests_passed, _ = run_verify_command(workspace, task.verify_command)
            score, failure, hallucinated, notes = score_task(task, workspace, response, ["app/service.py"], tests_passed)
            self.assertEqual(score, 1.0, notes)
            self.assertIsNone(failure)
            self.assertEqual(hallucinated, 0)


if __name__ == "__main__":
    unittest.main()
