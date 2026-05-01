import tempfile
import unittest
from pathlib import Path

from openclaw_bench.workspace import changed_files, copy_fixture, read_text_files, seed_openclaw_workspace_files, snapshot_files


ROOT = Path(__file__).resolve().parent.parent


class WorkspaceTests(unittest.TestCase):
    def test_fixture_copies_are_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            copy_fixture(ROOT / "fixtures" / "patch_repo", first)
            copy_fixture(ROOT / "fixtures" / "patch_repo", second)
            (first / "app" / "slug.py").write_text("def slugify(value):\n    return 'changed'\n", encoding="utf-8")
            self.assertNotEqual((first / "app" / "slug.py").read_text(), (second / "app" / "slug.py").read_text())

    def test_changed_files_detects_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "file.txt"
            target.write_text("before", encoding="utf-8")
            before = snapshot_files(root)
            target.write_text("after", encoding="utf-8")
            after = snapshot_files(root)
            self.assertEqual(changed_files(before, after), ["file.txt"])

    def test_git_metadata_is_ignored_for_hashes_and_patch_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git" / "hooks").mkdir(parents=True)
            (root / ".git" / "HEAD").write_text("ref: refs/heads/master\n", encoding="utf-8")
            (root / ".git" / "hooks" / "pre-commit.sample").write_text("ignored\n", encoding="utf-8")
            (root / "app.py").write_text("print('tracked')\n", encoding="utf-8")

            self.assertEqual(snapshot_files(root), {"app.py": snapshot_files(root)["app.py"]})
            self.assertEqual(read_text_files(root), {"app.py": "print('tracked')\n"})

    def test_seed_openclaw_workspace_files_creates_real_agent_context_without_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed_openclaw_workspace_files(
                root,
                agent_id="bench-007",
                task_id="workspace-discovery",
                model_id="qwen3-1.7b",
            )

            self.assertTrue((root / "AGENTS.md").exists())
            self.assertTrue((root / "SOUL.md").exists())
            self.assertTrue((root / "TOOLS.md").exists())
            self.assertTrue((root / "IDENTITY.md").exists())
            self.assertTrue((root / "USER.md").exists())
            self.assertTrue((root / "HEARTBEAT.md").exists())
            self.assertFalse((root / "BOOTSTRAP.md").exists())
            self.assertIn("workspace-discovery", (root / "AGENTS.md").read_text(encoding="utf-8"))
            self.assertIn("qwen3-1.7b", (root / "IDENTITY.md").read_text(encoding="utf-8"))
            self.assertIn("setupCompletedAt", (root / ".openclaw" / "workspace-state.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
