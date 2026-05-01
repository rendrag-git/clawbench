HOLIDAY_DISCOUNT_RATE = 0.10


def sale_total(subtotal):
    return round(float(subtotal) * (1 - HOLIDAY_DISCOUNT_RATE), 2)
