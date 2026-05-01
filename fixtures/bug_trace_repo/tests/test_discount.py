import unittest

from app.routes import checkout_total


class DiscountTests(unittest.TestCase):
    def test_vip_discount_is_ten_percent(self):
        customer = {"tier": "vip"}
        self.assertEqual(checkout_total(customer, 100.0), 90.0)


if __name__ == "__main__":
    unittest.main()

