import unittest

from app.service import render_status


class ServiceTests(unittest.TestCase):
    def test_status_is_normalized(self):
        self.assertEqual(render_status(" Needs Review "), {"status": "needs_review"})


if __name__ == "__main__":
    unittest.main()

