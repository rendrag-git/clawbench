from __future__ import annotations

import json
import os
import subprocess
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
    source: str  # "already_known" | "port_probe" | "explicit"
    # Only set when source == "already_known": absolute path to the source
    # OC profile's openclaw.json. Lets the inherit path clone the full
    # profile config rather than regenerate it from scratch.
    source_profile_path: str | None = None


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


def _probe_headers_for_provider(provider: str) -> dict[str, str]:
    """Return auth headers to include when probing the given provider.

    vLLM may require an API key.  We read VLLM_API_KEY from the environment
    (defaulting to the well-known local sentinel "vllm-local") and include it
    as a Bearer token only when probing vLLM endpoints.  Other providers are
    left unauthenticated.
    """
    if provider == "vllm":
        key = os.environ.get("VLLM_API_KEY", "vllm-local")
        return {"Authorization": f"Bearer {key}"}
    return {}


def port_probe_provider(
    provider: str,
    probes: list,
    *,
    total_timeout_s: float = 30.0,
    per_probe_timeout_s: float = 5.0,
    probe_hosts: list[str] | None = None,
) -> ProviderCandidate | None:
    """Scan well-known ports for *provider* and return the first responding candidate.

    ``probe_hosts`` is the ordered list of host addresses to try for each
    port.  It defaults to ``["127.0.0.1"]`` for purely local detection, but
    callers may extend it with LAN/bridge addresses (e.g. ``"10.68.198.1"``)
    to reach services that do not bind to loopback.
    """
    endpoints = PROVIDER_ENDPOINTS.get(provider, ())
    if not endpoints:
        return None
    if probe_hosts is None:
        probe_hosts = ["127.0.0.1"]
    headers = _probe_headers_for_provider(provider)
    budget_s = total_timeout_s
    for port, path in endpoints:
        for host in probe_hosts:
            if budget_s < per_probe_timeout_s:
                break
            url = f"http://{host}:{port}{path}"
            primary = probes[0]
            t0 = time.monotonic()
            result = primary.http_get(url, timeout_s=per_probe_timeout_s, headers=headers)
            elapsed = time.monotonic() - t0
            # Charge at least per_probe_timeout_s so mocked/fast probes still
            # consume their allotted budget slice and the cap is enforced.
            budget_s -= max(elapsed, per_probe_timeout_s)
            if not result.ok:
                continue
            models = _parse_models_from_body(provider, result.body)
            probe_results = {primary.name: result}
            for extra in probes[1:]:
                extra_result = extra.http_get(url, timeout_s=per_probe_timeout_s, headers=headers)
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


def run_detection(
    *,
    providers: list[str],
    probes: list,
    home: Path,
    per_provider_timeout_s: float = 30.0,
    probe_hosts: list[str] | None = None,
) -> DetectionReport:
    candidates: list[ProviderCandidate] = []
    findings: list[str] = []

    already_known = scan_existing_oc_profiles(home, probes=probes)
    known_by_provider = {c.provider: c for c in already_known}

    for provider in providers:
        if provider in known_by_provider:
            candidates.append(known_by_provider[provider])
            continue
        candidate = port_probe_provider(
            provider,
            probes,
            total_timeout_s=per_provider_timeout_s,
            probe_hosts=probe_hosts,
        )
        if candidate is None:
            continue
        candidates.append(candidate)
        for probe_name, result in candidate.probe_results.items():
            if probe_name == probes[0].name:
                continue
            primary_ok = candidate.probe_results[probes[0].name].ok
            if primary_ok and not result.ok:
                findings.append(
                    f"reachable_from_host_not_runtime:{provider}@{candidate.base_url} "
                    f"(probe={probe_name})"
                )

    return DetectionReport(candidates=tuple(candidates), findings=tuple(findings))


from .probes import DockerExecProbe, IncusExecProbe, LocalProbe, Probe, SSHProbe


def derive_probes_for_profile(
    profile: str,
    *,
    home: Path,
    oc_runtime_override: str | None = None,
) -> list[Probe]:
    probes: list[Probe] = [LocalProbe()]
    if oc_runtime_override:
        probes.append(_probe_from_override(oc_runtime_override))
        return probes
    config_path = Path(home).expanduser() / f".openclaw-{profile}" / "openclaw.json"
    if config_path.is_file():
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        runtime = (((payload or {}).get("gateway") or {}).get("runtime") or {})
        kind = runtime.get("kind") if isinstance(runtime, dict) else None
        if kind == "incus":
            instance = runtime.get("instance")
            if isinstance(instance, str):
                probes.append(IncusExecProbe(instance))
        elif kind == "docker":
            container = runtime.get("container")
            if isinstance(container, str):
                probes.append(DockerExecProbe(container))
    return probes


def _probe_from_override(spec: str) -> Probe:
    if ":" not in spec:
        raise ValueError(
            f"--oc-runtime expects 'kind:target' (incus:<instance>, docker:<container>, ssh:<user@host>); got '{spec}'"
        )
    kind, _, target = spec.partition(":")
    kind = kind.strip().lower()
    target = target.strip()
    if kind == "incus":
        return IncusExecProbe(target)
    if kind == "docker":
        return DockerExecProbe(target)
    if kind == "ssh":
        return SSHProbe(target)
    raise ValueError(
        f"--oc-runtime kind '{kind}' not supported; use incus|docker|ssh"
    )


def scan_existing_oc_profiles(home: Path, probes: list | None = None) -> list[ProviderCandidate]:
    home = Path(home).expanduser()
    candidates: list[ProviderCandidate] = []
    for profile_dir in sorted(home.glob(".openclaw*")):
        config_path = profile_dir / "openclaw.json"
        if not config_path.is_file():
            continue
        try:
            candidates.extend(
                _candidates_from_config_text(
                    config_path.read_text(encoding="utf-8"),
                    source_profile_path=str(config_path),
                )
            )
        except (OSError, json.JSONDecodeError):
            continue
    for probe in probes or []:
        candidates.extend(_scan_runtime_oc_profiles(probe))
    return candidates


def _candidates_from_config_text(text: str, *, source_profile_path: str | None = None) -> list[ProviderCandidate]:
    payload = json.loads(text)
    candidates: list[ProviderCandidate] = []
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
                source_profile_path=source_profile_path,
            )
        )
    return candidates


def _scan_runtime_oc_profiles(probe) -> list[ProviderCandidate]:
    from .probes import DockerExecProbe, IncusExecProbe  # noqa: PLC0415

    if isinstance(probe, IncusExecProbe):
        cmd = ["incus", "exec", probe.instance, "--", "sh", "-lc", _profile_scan_script()]
    elif isinstance(probe, DockerExecProbe):
        cmd = ["docker", "exec", probe.container, "sh", "-lc", _profile_scan_script()]
    else:
        return []
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    candidates: list[ProviderCandidate] = []
    for text in _split_profile_scan_output(proc.stdout):
        try:
            candidates.extend(_candidates_from_config_text(text))
        except json.JSONDecodeError:
            continue
    return candidates


def _profile_scan_script() -> str:
    return r'''
for base in "$HOME" /home/* /root; do
  [ -d "$base" ] || continue
  for f in "$base"/.openclaw*/openclaw.json; do
    [ -f "$f" ] || continue
    printf '\n---OC_BENCH_PROFILE---\n'
    cat "$f"
  done
done
'''


def _split_profile_scan_output(output: str) -> list[str]:
    return [part.strip() for part in output.split("---OC_BENCH_PROFILE---") if part.strip()]
