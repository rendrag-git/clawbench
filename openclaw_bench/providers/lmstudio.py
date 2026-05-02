from __future__ import annotations

import json

from .detect import ProviderCandidate


def detect(body: str) -> list[str]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return []
    data = payload.get("data") or []
    return [
        entry.get("id")
        for entry in data
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    ]


def generate_route_config(
    candidate: ProviderCandidate,
    *,
    context: int,
    max_tokens: int,
) -> dict:
    raise NotImplementedError(
        "lmstudio provider config generation is not yet implemented; "
        "this slice ships detect-only. See "
        "https://docs.openclaw.ai/providers/lmstudio for the target shape."
    )
