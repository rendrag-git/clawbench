from __future__ import annotations

from dataclasses import dataclass, field

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
