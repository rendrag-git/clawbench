import unittest

from app.context import agent_traits, task_policy


class ContextTests(unittest.TestCase):
    def test_policy_is_onboarded_and_json_only(self):
        self.assertEqual(task_policy(), {"onboarded": True, "json_only": True})

    def test_traits_include_practical(self):
        self.assertIn("practical", agent_traits())


if __name__ == "__main__":
    unittest.main()
