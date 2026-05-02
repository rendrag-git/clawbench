# Provider Detection Surface — Design

Date: 2026-05-02
Owner: rendrag-git
Milestone: M3 — Provider breadth, inspect-first, four runtimes
Pinned OpenClaw version: `2026.4.27`

## Goal

Make `oc-bench init` work cold across the four runtimes the bench claims to support — vLLM, llama.cpp, Ollama, LM Studio — by inspecting what is already running on the user's machine, generating a correct OpenClaw provider config from that, and verifying it before the benchmark runs. The user should not need to know OpenClaw config shape, the 16k context floor, the meta-field requirement, or per-provider parameter shaping.

The motivating failure: getting a single provider/model combo working today requires ad-hoc UFW debugging, hand-tuned `extra_body`, manual meta-field repair, and per-runtime context-floor surprises. That work has been done once for vLLM/Qwen3.5-4B and once for vLLM/GPT-OSS-20B. Before doing it a third time for Ollama, encode the lessons.

## Scope

### In scope (this design)

- A new `openclaw_bench/providers/` package with a detection cascade, per-provider modules, and a runtime-aware probe layer.
- Extending `oc-bench init` to run detection automatically and generate the OpenClaw config from real probes instead of env-var defaults.
- A new `oc-bench preflight` command wrapping the four verification gates.
- vLLM module fully implemented (detect + generate + parameter shaping).
- Detect-only stubs for Ollama, llama.cpp, LM Studio (raise `NotImplementedError` on generate).
- Regression tests against fake provider servers; one live integration test gated by `OC_BENCH_LIVE=1`.

### Out of scope (deferred to follow-up commits)

- Ollama, llama.cpp, LM Studio config generators. Each is its own commit, gated by being able to validate against a real running instance (e.g., Ollama on the RTX Pro 5000).
- Apple Silicon-specific runtime hints (MLX detection, etc.). Treated as a future provider module.
- Replacing or rewriting `quickstart.py`. The new code calls `quickstart.py`'s existing `_vllm_provider_config`, `_external_vllm_model`, `_openclaw_route_context` helpers — those encode hard-won bug fixes (16k context floor, meta fields, plugin entries, `chatTemplateKwargs.enable_thinking=false`) that must not be re-introduced.
- Authoring new tier manifests or scoring rules.
- Closing M2's tier-small live calibration record. That follows from the deployment surface working — pick a 4–8B candidate after detection lands.

## Architecture

### Probe location model

Most users have one machine: Python and OpenClaw run in the same network namespace. A minority (this repo's author, anyone using Docker/Incus/SSH-to-OC) have a split where the bench's Python and OpenClaw live in different namespaces — that split is exactly what the GPT-OSS UFW failure exposed (host could reach `:8000`, oc-stack could not).

Rule:

- **Default:** probe from the host (where Python runs). One probe per candidate. Covers Mac/Linux/Windows native installs.
- **Auto-derive split:** while doing the cheap-first-pass scan of `~/.openclaw-*/openclaw.json`, read the gateway runtime field. If it indicates a containerized or remote runtime, the bench probes from both sides for every candidate and surfaces mismatches as a named finding (`reachable_from_host_not_runtime`). No flag required.
- **SSH / opaque remote:** the only case that needs explicit user input. If the OC profile names a remote runtime the bench cannot infer how to reach (e.g., a non-loopback gateway URL with no Incus/Docker hint), `oc-bench init` fails loud with: "OC profile `<name>` points at a remote runtime; pass `--oc-runtime ssh:user@host` to enable detection from there."

No auto-detection of "is OC native or containerized?" by guessing from the OS or installed binaries. That is brittle. The signal comes from the OC profile or an explicit flag.

### Probe abstraction

```python
class Probe(Protocol):
    name: str  # "host", "incus:oc-stack", "docker:oc", "ssh:user@host"
    def http_get(self, url: str, *, timeout_s: float) -> ProbeResult: ...
    def http_post(self, url: str, *, json: dict, timeout_s: float) -> ProbeResult: ...
```

Concrete implementations:

- `LocalProbe` — `httpx`/`urllib` from the bench's Python process.
- `IncusExecProbe` — `incus exec <name> -- curl …` shell-out.
- `DockerExecProbe` — `docker exec <name> -- curl …` shell-out.
- `SSHProbe` — `ssh user@host curl …` shell-out.

`detect()` accepts a list of probes (one or two, depending on whether a split was derived) and returns probe-tagged results so the cascade can report mismatches.

### Detection cascade

Per provider, executed in this order:

1. **Already-configured scan** — read each `~/.openclaw-*/openclaw.json`. If any has `models.providers.<provider>` configured *and* one of the configured probes can reach its `baseUrl`, surface as `already_known` with zero new probes.
2. **Port probe** — if not already known, probe the provider's well-known endpoint(s) using the configured probes. Endpoint table:
   - vLLM: `GET /v1/models` on common ports (`8000`, `8001`, `8002`, `8003`, `8080`).
   - llama.cpp: `GET /v1/models` (newer builds) or `GET /props` (older builds), same port set.
   - Ollama: `GET /api/tags` on `11434`.
   - LM Studio: `GET /v1/models` on `1234`.
3. **Hard timeout** — 30s per provider, total across all probes for that provider. If exhausted, the provider returns `not_found` with a `reason` of `"timeout"`.

Results aggregated into `DetectionReport`:

```python
@dataclass(frozen=True)
class ProviderCandidate:
    provider: str           # "vllm" | "ollama" | "llamacpp" | "lmstudio"
    base_url: str
    models: list[str]       # served model ids the endpoint reports
    probe_results: dict[str, ProbeResult]   # probe-name -> result
    source: str             # "already_known" | "port_probe"

@dataclass(frozen=True)
class DetectionReport:
    candidates: list[ProviderCandidate]
    findings: list[str]     # e.g., "reachable_from_host_not_runtime: vllm@:8000"
```

### Ambiguity handling

- Zero candidates and nothing already known → CLI prompts: "No local provider detected. Tell me what is running and where."
- Exactly one candidate → auto-select.
- Multiple candidates → CLI prompts: "Found N providers. Which one should the benchmark route to?"
- Multiple candidates *of the same provider type* (e.g., two vLLM endpoints) → CLI lists base URLs + served model ids and prompts.

### CLI surface

| Command | New / changed | Behavior |
|---|---|---|
| `oc-bench init` | changed | Runs detection cascade automatically. Falls back to env-var-driven defaults only with `--no-detect`. New flag `--oc-runtime <spec>` for SSH/remote runtimes. |
| `oc-bench preflight` | new | Runs the four verification gates: `openclaw config validate`, `openclaw models list --provider <p>`, direct health probe from the configured runtime, OpenClaw route smoke (`openclaw infer model run --gateway --model <route> --prompt 'Reply with exactly: ok'`). Idempotent. Exit code `0` only if all four pass. |
| `oc-bench run` | unchanged | — |

### Module layout

```
openclaw_bench/providers/
    __init__.py
    detect.py            # cascade + DetectionReport
    probes.py            # Probe protocol + Local/IncusExec/DockerExec/SSH implementations
    vllm.py              # detect() + generate_route_config() + parameter_shaping()
    ollama.py            # detect() only; generate raises NotImplementedError
    llamacpp.py          # detect() only; generate raises NotImplementedError
    lmstudio.py          # detect() only; generate raises NotImplementedError
```

`vllm.py.generate_route_config()` is a thin wrapper around `quickstart.py._vllm_provider_config()` plus the existing `_openclaw_route_context()` clamp. No copy-paste of the meta/plugin/16k logic.

`vllm.py.parameter_shaping()` returns the `agents.defaults.models[<route>].params` block, encoding:
- Always: `chatTemplateKwargs.enable_thinking=false` (Qwen-class models).
- If served model id starts with `gpt-oss`: `extra_body.reasoning_effort="low"` (per STATUS.md GPT-OSS finding).

## Data flow (init path)

1. User runs `oc-bench init --providers local`.
2. CLI loads bench config, derives the probe set (host-only, or host + runtime-side if profile shows a split).
3. CLI invokes `providers.detect.run(probes=…, providers=["vllm","ollama","llamacpp","lmstudio"])`.
4. Cascade returns a `DetectionReport`. CLI renders findings (including any `reachable_from_host_not_runtime` mismatches).
5. CLI selects a candidate (auto if one, prompt if many, prompt with empty result if zero).
6. CLI calls the selected provider module's `generate_route_config(candidate)` and `parameter_shaping(candidate)`.
7. CLI writes the OpenClaw config and model manifest using existing `quickstart.py` plumbing.
8. CLI runs `oc-bench preflight` automatically as the final init step. If it fails, init exits non-zero with the failed gate named.

## Testing

### Unit + integration (CI-suitable, no live model)

- Fake `/v1/models` server fixtures for vLLM, llama.cpp (both shapes), LM Studio.
- Fake `/api/tags` server fixture for Ollama.
- Cascade tests: zero candidates, one candidate, two candidates of one type, three candidates of three types, already-known via mocked OC profile config.
- Probe-mismatch test: host probe returns models, runtime probe returns connection refused → `reachable_from_host_not_runtime` finding emitted.
- Generator test: vLLM generator output matches `quickstart.py._vllm_provider_config` byte-for-byte for the same input.
- Parameter-shaping test: GPT-OSS served model id triggers `reasoning_effort="low"`; Qwen does not.
- 30s timeout test: hung-port fixture; cascade returns `not_found` with `reason="timeout"`.
- `oc-bench preflight` test: each gate stubbed pass/fail; exit code matches.

### Live integration (gated by `OC_BENCH_LIVE=1`)

- Probe `http://10.68.198.1:8000/v1` from `oc-stack`. Expect detection of `gpt-oss-20b`.
- Generate config, write to a throwaway profile, run preflight, expect all four gates pass.
- One smoke turn: `Reply with exactly: ok`. Expect visible `ok` and `< 5s` total wall time.

### Out of test scope

- Real Ollama / llama.cpp / LM Studio detection. Added when those provider modules graduate from stub.
- Apple Silicon-specific paths.
- Performance benchmarking of the cascade itself.

## Risks and known gaps

- **First-commit GPT-OSS smoke is a tier-medium candidate.** This validates the deployment surface, not a tier-small calibration record. M2's missing live floor record is *not* closed by this work.
- **Probe-runtime auto-derivation depends on OC profile shape.** If `2026.4.27` profile JSON does not name the runtime in a parseable field, the auto-derive falls back to host-only and the user must pass `--oc-runtime` explicitly. Mitigation: read a sample `oc-stack` profile during implementation; if the field is absent, treat that as a follow-up issue, not a blocker — explicit flag works for the author's own setup in the meantime.
- **Stub modules invite mistakes.** A future contributor might wire a stub provider into a benchmark run before its generator is implemented. Mitigation: stubs raise `NotImplementedError` on `generate_route_config`, and `oc-bench preflight` fails closed if no generator is wired.
- **Fake-server tests can drift from real server shapes.** Mitigation: each fake server is built from a captured response of the real one (vLLM `/v1/models` and Ollama `/api/tags` payloads stored under `tests/fixtures/provider_responses/`).

## Done definition

- `openclaw_bench/providers/` package exists with the modules listed in *Module layout*.
- `oc-bench init` detects vLLM/GPT-OSS-20B from the existing oc-stack setup, generates a working OpenClaw route config, and `oc-bench preflight` reports all four gates pass.
- All unit tests under `tests/test_providers_*.py` pass; `OC_BENCH_LIVE=1 python3 -m unittest tests.test_providers_live` passes against the live GPT-OSS endpoint.
- `python3 -m unittest discover -s tests` passes (current count `~244`, expect to grow by the new tests, no regressions).
- Simulator full-suite regression passes against the existing `manifests/openclaw-certification-full.example.json`.
- STATUS.md updated with: M3 deployment-surface slice complete, vLLM full / Ollama+llama.cpp+LM Studio detect-only, GPT-OSS 20B used as live smoke target, M2 tier-small calibration still open.

## Follow-ups (explicit, not stealth scope)

1. Bring up Ollama on the RTX Pro 5000, flesh out `ollama.py` generator, add real fake-server tests, gate by `OC_BENCH_LIVE=1` live test against the running Ollama.
2. Same for llama.cpp.
3. Same for LM Studio (will require a Mac or LM Studio Linux build; defer until a machine is available).
4. Pick a 4–8B floor candidate using the new detection surface and capture the M2 tier-small calibration record.
