import unittest

from app.labels import sale_banner
from app.pricing import sale_total


class SaleConsistencyTests(unittest.TestCase):
    def test_pricing_and_label_use_new_rate(self):
        self.assertEqual(sale_total(100), 85.0)
        self.assertEqual(sale_banner(), "Holiday sale: 15% off")


if __name__ == "__main__":
    unittest.main()
