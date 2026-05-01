import unittest

from app.audit import refund_audit_event
from app.messages import refund_policy_message
from app.refunds import refund_deadline_days


class RefundPolicyTests(unittest.TestCase):
    def test_refund_window_is_45_days(self):
        self.assertEqual(refund_deadline_days(), 45)
        self.assertEqual(refund_policy_message(), "Refunds are available for 45 days.")

    def test_audit_taxonomy_is_unchanged(self):
        self.assertEqual(refund_audit_event(), "refund.window.reviewed")


if __name__ == "__main__":
    unittest.main()
