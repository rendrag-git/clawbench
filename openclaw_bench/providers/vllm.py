from __future__ import annotations

import json

from ..quickstart import VllmEndpoint, _vllm_provider_config
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
    if not candidate.models:
        raise ValueError("vllm candidate has no models; cannot generate route config")
    endpoint = VllmEndpoint(
        base_url=candidate.base_url,
        model=candidate.models[0],
        context=context,
        max_tokens=max_tokens,
    )
    return _vllm_provider_config(endpoint)


def parameter_shaping(candidate: ProviderCandidate) -> dict:
    if not candidate.models:
        raise ValueError("vllm candidate has no models; cannot shape parameters")
    model_id = candidate.models[0]
    params: dict = {"chatTemplateKwargs": {"enable_thinking": False}}
    if model_id.startswith("gpt-oss"):
        params["extra_body"] = {"reasoning_effort": "low"}
    return params
