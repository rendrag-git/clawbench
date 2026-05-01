import unittest

from api.routes import ROUTES
from db.schema import TABLES


class ApiDiscoveryTests(unittest.TestCase):
    def test_health_route_and_schema_exist(self):
        self.assertIn("/health", ROUTES)
        self.assertIn("users", TABLES)


if __name__ == "__main__":
    unittest.main()

