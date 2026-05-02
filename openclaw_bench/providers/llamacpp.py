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
        "llama.cpp provider config generation is not yet implemented; "
        "this slice ships detect-only. The generator should emit a custom "
        "openai-completions provider matching llama-server's /v1 surface."
    )
