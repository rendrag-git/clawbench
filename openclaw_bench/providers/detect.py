from __future__ import annotations

import json
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
