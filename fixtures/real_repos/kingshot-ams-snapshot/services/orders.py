def create_order(payload):
    return {
        "order_id": payload["order_id"],
        "status": "created",
    }


def order_status(order_id):
    return {
        "order_id": order_id,
        "status": "processing",
    }

