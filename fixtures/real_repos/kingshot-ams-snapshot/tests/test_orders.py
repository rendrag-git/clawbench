import unittest

from services.orders import create_order, order_status


class OrderServiceTests(unittest.TestCase):
    def test_create_order_defaults_to_created(self):
        self.assertEqual(create_order({"order_id": "ORD-100"})["status"], "created")

    def test_regular_order_status_is_processing(self):
        self.assertEqual(order_status("ORD-100"), {"order_id": "ORD-100", "status": "processing"})

    def test_shipped_order_status_is_shipped(self):
        self.assertEqual(order_status("SHIP-200"), {"order_id": "SHIP-200", "status": "shipped"})


if __name__ == "__main__":
    unittest.main()
