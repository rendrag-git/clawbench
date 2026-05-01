def health_route():
    return {"status": "ok"}


ROUTES = {
    "/health": health_route,
}

