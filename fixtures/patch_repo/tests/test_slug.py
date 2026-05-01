import unittest

from app.slug import slugify


class SlugTests(unittest.TestCase):
    def test_slugify_lowercases_and_collapses_separators(self):
        self.assertEqual(slugify(" Hello, OpenClaw Bench! "), "hello-openclaw-bench")

    def test_slugify_strips_edges(self):
        self.assertEqual(slugify("---Already Here---"), "already-here")


if __name__ == "__main__":
    unittest.main()

