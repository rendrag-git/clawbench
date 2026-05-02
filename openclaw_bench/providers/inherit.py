"""Clone an existing OpenClaw profile into a bench-isolated profile.

The bench's preferred deployment path: assume the user already has OC configured
with working local providers (vLLM, Ollama, llama.cpp, LM Studio) — base URLs,
auth, per-model parameter shaping, plugin entries, all already tuned. Rather
than regenerate any of that from probes (which has bitten us with silent
fallbacks and missing parameter shaping; see issues #1 and #7), we copy the
working profile verbatim and overlay a minimal set of bench-specific isolation
fields.

Pure function: in == source profile dict, out == bench profile dict. No I/O,
no env access, no clock dependency by default. The caller decides how to read
the source JSON and where to write the result.

The inheritance preserves verbatim:
  - models.providers.*        (every wiring detail: baseUrl, auth, request
                              shaping, model lists with context limits)
  - agents.defaults.models.*  (every per-model param tweak: enable_thinking,
                              reasoning_effort, extra_body, maxTokens)
  - secrets, env.vars, commands, plugins
  - gateway.bind, gateway.mode, gateway.tailscale

The overlay (the only things the bench actually changes):
  - gateway.port          -> caller-supplied unique loopback port
  - gateway.auth.token    -> fresh random token (or caller override for tests)
  - agents.list           -> single bench agent with caller-supplied id + route
  - agents.defaults.model -> bench route, if caller supplied one
  - agents.defaults.skipBootstrap -> always true (bench workspaces are
                                    pre-seeded; no bootstrap needed)
  - meta.lastTouchedVersion -> caller-supplied OC version
  - meta.lastTouchedAt    -> caller-supplied timestamp (or omitted; the OC
                            CLI will rewrite this on its first edit)
"""

from __future__ import annotations

import copy
import secrets as _secrets
from typing import Any


def clone_profile(
    source: dict,
    *,
    bench_profile: str,
    gateway_port: int,
    gateway_token: str | None = None,
    bench_agent_id: str = "bench",
    bench_route_model: str | None = None,
    openclaw_version: str = "2026.4.27",
    last_touched_at: str | None = None,
) -> dict:
    """Return a deep clone of `source` with bench-specific overlays applied.

    Args:
        source: parsed openclaw.json from the source profile (e.g. `pmg`).
        bench_profile: name of the bench profile being created. Used only for
            error messages — the profile name is encoded by the *file path*
            the caller writes to, not in the JSON itself.
        gateway_port: loopback port for the bench gateway. Must not collide
            with the source profile's gateway port.
        gateway_token: bench gateway auth token. If None, a fresh random
            URL-safe token is generated. Pass an explicit value in tests for
            determinism.
        bench_agent_id: id of the single bench agent the run will use.
        bench_route_model: which provider/model route the bench agent uses.
            If None, falls back to `source.agents.defaults.model`. Raises
            ValueError if neither is set.
        openclaw_version: target OC version for `meta.lastTouchedVersion`.
        last_touched_at: ISO 8601 timestamp for `meta.lastTouchedAt`. If None,
            the field is omitted (OC will populate it on first edit).

    Returns:
        A deep-cloned dict suitable for writing to
        `~/.openclaw-<bench_profile>/openclaw.json`.

    Raises:
        ValueError: if `source` is malformed, has no providers, or no model
            route is resolvable.
    """
    if not isinstance(source, dict):
        raise ValueError("source profile must be a dict (parsed JSON)")

    cloned: dict[str, Any] = copy.deepcopy(source)

    providers_block = (cloned.get("models") or {}).get("providers") or {}
    if not providers_block:
        raise ValueError(
            f"source profile has no models.providers configured; cannot inherit "
            f"into bench profile '{bench_profile}'"
        )

    # Resolve the bench agent's route. Prefer caller override; else inherit
    # from the source's agents.defaults.model.
    source_default_route = (cloned.get("agents") or {}).get("defaults", {}).get("model")
    route = bench_route_model or source_default_route
    if not isinstance(route, str) or not route:
        raise ValueError(
            f"no route_model resolvable for bench profile '{bench_profile}': "
            f"pass bench_route_model or set agents.defaults.model in the source"
        )

    # Verify the route's provider key actually exists in inherited providers.
    # Route format is "<provider>/<model_id>".
    provider_key = route.split("/", 1)[0] if "/" in route else None
    if provider_key not in providers_block:
        raise ValueError(
            f"route '{route}' references provider '{provider_key}' which is "
            f"not configured in the source profile (have: {sorted(providers_block)})"
        )

    # Overlay 1: gateway — bench-isolated port + fresh token, preserve bind/mode/tailscale.
    gateway = cloned.setdefault("gateway", {})
    gateway["port"] = int(gateway_port)
    auth = gateway.setdefault("auth", {})
    auth["mode"] = auth.get("mode", "token")
    auth["token"] = gateway_token if gateway_token is not None else _secrets.token_urlsafe(32)
    gateway.setdefault("bind", "loopback")
    gateway.setdefault("mode", "local")
    gateway.setdefault("tailscale", {"mode": "off"})

    # Overlay 2: agents.list — one bench agent. agents.defaults preserved verbatim
    # (per-model params come along for free), but agents.defaults.model is repointed
    # at the chosen bench route, and skipBootstrap is forced true (bench workspaces
    # are pre-seeded).
    agents = cloned.setdefault("agents", {})
    defaults = agents.setdefault("defaults", {})
    defaults["model"] = route
    defaults["skipBootstrap"] = True
    agents["list"] = [
        {
            "id": bench_agent_id,
            "model": route,
            "tools": {"profile": "coding"},
        }
    ]

    # Overlay 3: meta — pin OC version, optionally stamp timestamp.
    meta = cloned.setdefault("meta", {})
    meta["lastTouchedVersion"] = openclaw_version
    if last_touched_at is not None:
        meta["lastTouchedAt"] = last_touched_at
    elif "lastTouchedAt" in meta:
        # Drop a stale timestamp so OC rewrites it on first edit.
        meta.pop("lastTouchedAt", None)

    return cloned


def model_manifest_from_profile(
    source: dict,
    *,
    hardware_profile: str = "inherited",
    default_concurrency: tuple[int, ...] = (1,),
) -> dict:
    """Derive a benchmark model manifest from an inherited OC profile dict.

    Lists every model the source profile's `models.providers.*` block routes,
    one row per (provider, model_id). The runner consumes this manifest to know
    which `served_model_name`, `kv_modes`, `contexts`, and `api_base` to drive.

    Note: this does not pull `agents.defaults.models[<route>].params` into the
    manifest — those live on the bench profile itself (the runner reads them
    from there at run time). Only the provider-side routing data is materialized
    into the manifest.
    """
    providers = (source.get("models") or {}).get("providers") or {}
    models: list[dict] = []
    for prov_name, prov_block in sorted(providers.items()):
        if not isinstance(prov_block, dict):
            continue
        base_url = prov_block.get("baseUrl")
        # Auth env id (if configured): e.g. {"id": "VLLM_API_KEY", ...}
        api_env = (
            (prov_block.get("request", {}).get("auth", {}).get("token", {}) or {}).get("id")
        )
        for entry in prov_block.get("models") or []:
            if not isinstance(entry, dict):
                continue
            model_id = entry.get("id") or entry.get("name")
            if not isinstance(model_id, str):
                continue
            context = (
                entry.get("contextWindow")
                or entry.get("contextTokens")
                or 32768
            )
            row = {
                "model_id": model_id,
                "served_model_name": model_id,
                "openclaw_model_name": f"{prov_name}/{model_id}",
                "provider_type": "local",
                "hardware_profile": hardware_profile,
                "weight_quant": "inherited",
                "kv_modes": ["provider_default"],
                "contexts": [int(context)],
                "concurrency": list(default_concurrency),
                "api_base": base_url,
                "expected_support": f"Inherited from source profile (provider={prov_name}); bench will not start or restart this runtime.",
            }
            if api_env:
                row["api_env"] = api_env
            models.append(row)
    return {
        "manifest_scope": {
            "portability": "inherited",
            "notes": "Generated by oc-bench init from an inherited OC profile.",
            "source_manifests": ["inherited"],
        },
        "models": models,
    }
