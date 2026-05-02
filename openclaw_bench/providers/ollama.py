from __future__ import annotations

import json

from .detect import ProviderCandidate


def detect(body: str) -> list[str]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return []
    models = payload.get("models") or []
    return [
        entry.get("name")
        for entry in models
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    ]


def generate_route_config(
    candidate: ProviderCandidate,
    *,
    context: int,
    max_tokens: int,
) -> dict:
    raise NotImplementedError(
        "ollama provider config generation is not yet implemented; "
        "this slice ships detect-only. See "
        "https://docs.openclaw.ai/providers/ollama for the target shape."
    )
