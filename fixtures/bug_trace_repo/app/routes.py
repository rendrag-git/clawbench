from app.discounts import vip_discount_rate


def checkout_total(customer, subtotal):
    return round(subtotal * (1 - vip_discount_rate(customer)), 2)

