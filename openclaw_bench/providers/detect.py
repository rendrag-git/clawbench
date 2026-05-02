from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .probes import ProbeResult


@dataclass(frozen=True)
class ProviderCandidate:
    provider: str
    base_url: str
    models: list[str]
    probe_results: dict[str, ProbeResult]
    source: str  # "already_known" | "port_probe"


@dataclass(frozen=True)
class DetectionReport:
    candidates: tuple[ProviderCandidate, ...] = field(default_factory=tuple)
    findings: tuple[str, ...] = field(default_factory=tuple)


KNOWN_PROVIDERS: tuple[str, ...] = ("vllm", "ollama", "llamacpp", "lmstudio")


PROVIDER_ENDPOINTS: dict[str, tuple[tuple[int, str], ...]] = {
    "vllm": (
        (8000, "/v1/models"),
        (8001, "/v1/models"),
        (8002, "/v1/models"),
        (8003, "/v1/models"),
        (8080, "/v1/models"),
    ),
    "llamacpp": (
        (8080, "/v1/models"),
        (8000, "/v1/models"),
    ),
    "ollama": (
        (11434, "/api/tags"),
    ),
    "lmstudio": (
        (1234, "/v1/models"),
    ),
}


def _parse_models_from_body(provider: str, body: str) -> list[str]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return []
    if provider == "ollama":
        models = payload.get("models") or []
        return [m.get("name") for m in models if isinstance(m, dict) and isinstance(m.get("name"), str)]
    data = payload.get("data") or []
    return [m.get("id") for m in data if isinstance(m, dict) and isinstance(m.get("id"), str)]


def port_probe_provider(
    provider: str,
    probes: list,
    *,
    total_timeout_s: float = 30.0,
    per_probe_timeout_s: float = 5.0,
) -> ProviderCandidate | None:
    endpoints = PROVIDER_ENDPOINTS.get(provider, ())
    if not endpoints:
        return None
    budget_s = total_timeout_s
    for port, path in endpoints:
        if budget_s < per_probe_timeout_s:
            break
        url = f"http://127.0.0.1:{port}{path}"
        primary = probes[0]
        t0 = time.monotonic()
        result = primary.http_get(url, timeout_s=per_probe_timeout_s)
        elapsed = time.monotonic() - t0
        # Charge at least per_probe_timeout_s so mocked/fast probes still
        # consume their allotted budget slice and the cap is enforced.
        budget_s -= max(elapsed, per_probe_timeout_s)
        if not result.ok:
            continue
        models = _parse_models_from_body(provider, result.body)
        probe_results = {primary.name: result}
        for extra in probes[1:]:
            extra_result = extra.http_get(url, timeout_s=per_probe_timeout_s)
            probe_results[extra.name] = extra_result
        base_url = url[: -len(path)] + ("/v1" if path.startswith("/v1") else "")
        return ProviderCandidate(
            provider=provider,
            base_url=base_url,
            models=models,
            probe_results=probe_results,
            source="port_probe",
        )
    return None


def scan_existing_oc_profiles(home: Path) -> list[ProviderCandidate]:
    home = Path(home).expanduser()
    candidates: list[ProviderCandidate] = []
    for profile_dir in sorted(home.glob(".openclaw*")):
        config_path = profile_dir / "openclaw.json"
        if not config_path.is_file():
            continue
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        providers = (((payload or {}).get("models") or {}).get("providers") or {})
        for key, block in providers.items():
            if key not in KNOWN_PROVIDERS:
                continue
            base_url = (block or {}).get("baseUrl")
            if not isinstance(base_url, str):
                continue
            model_ids: list[str] = []
            for entry in (block or {}).get("models") or []:
                model_id = (entry or {}).get("id") if isinstance(entry, dict) else None
                if isinstance(model_id, str):
                    model_ids.append(model_id)
            candidates.append(
                ProviderCandidate(
                    provider=key,
                    base_url=base_url,
                    models=model_ids,
                    probe_results={},
                    source="already_known",
                )
            )
    return candidates
