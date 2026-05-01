import unittest

from app.health import health


class HealthTests(unittest.TestCase):
    def test_health_uses_current_needle(self):
        payload = health()
        self.assertEqual(payload["token"], "oc_bench_real_token_8k_1842")


if __name__ == "__main__":
    unittest.main()
