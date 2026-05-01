def vip_discount_rate(customer):
    if customer.get("tier") == "vip":
        return 0.01
    return 0.0

