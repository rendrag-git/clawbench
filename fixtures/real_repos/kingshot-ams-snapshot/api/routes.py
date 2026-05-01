from api.auth import require_staff_session
from services.orders import create_order, order_status


def register_order_routes(router):
    router.post("/orders", create_order)
    router.get("/orders/{order_id}/status", order_status)


def register_admin_routes(router):
    router.before_request(require_staff_session)
    router.get("/admin/orders/{order_id}/status", order_status)

