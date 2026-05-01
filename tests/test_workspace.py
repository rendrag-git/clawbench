import tempfile
import unittest
from pathlib import Path

from openclaw_bench.workspace import changed_files, copy_fixture, read_text_files, snapshot_files


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


if __name__ == "__main__":
    unittest.main()
