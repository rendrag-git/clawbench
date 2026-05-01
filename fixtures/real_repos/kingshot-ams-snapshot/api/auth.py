def require_staff_session(request):
    token = request.headers.get("X-Staff-Session")
    if not token:
        raise PermissionError("missing staff session")
    return {"staff_id": token}

