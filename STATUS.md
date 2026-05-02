# Current Status

Last updated: 2026-05-02 04:00 UTC

## TL;DR

- **What this repo is for:** answering "is this local model good enough for OpenClaw agent work?" with evidence, not vibes. Full vision in [GOAL.md](GOAL.md).
- **Where we are:** harness mechanics + tier-suite tasks + provider-detection deployment surface for vLLM are all shipped. Not yet shipped: Ollama / llama.cpp / LM Studio config generators (currently detect-only stubs), and live floor/ceiling calibration records for any tier.
- **Active milestone:** M3 (provider breadth) is partially done — vLLM full, three other runtimes detect-only. M2 (tiered task suite) still owes a live tier-small calibration record on a model that actually clears the floor.
- **Pinned:** OpenClaw `2026.4.27`. `2026.4.29` blocked until observed regressions resolve.

## Runtime

- OpenClaw runtime under test: `oc-stack` (Incus container).
- Benchmark vLLM services on this host:
  - `openclaw-vllm-small-bench.service` — Qwen3.5-4B BF16 on GPU 0 (A4000 16GB), `http://10.68.198.1:8003/v1`, `--enforce-eager`, context 32768.
  - GPT-OSS 20B MXFP4 on GPU 1 (RTX PRO 5000 Blackwell), `http://10.68.198.1:8000/v1`, `quantization=gpt_oss_mxfp4`, `kv_cache_dtype=fp8`, `max_model_len=131072`, route auth `VLLM_API_KEY=vllm-local`. Always send `extra_body.reasoning_effort="low"` — without it the model burns the budget on reasoning.
- Bench profile: `benchclaw-m2` (use this single isolated profile for live calibrations; do not create timestamped per-run profiles).
- Larger-model context rule: do not start agent benchmark models with `--max-model-len 4096`. Use 32768 when feasible, 16384 only as fallback minimum. OpenClaw `2026.4.27` rejects `modelsConfig` context windows below 16000.

## What shipped

### M1 — Trust hygiene (complete)

- Scoring rules audited; the `workspace-discovery` exact-string-match bug, the slash-prose hallucinated-path false positive, and the discovery path-equivalence false positive are fixed with regression tests.
- Simulator full-suite produces 40 attempts / 0 failures and matches what the live backend would score.

### M2 — Tiered task suite (task design complete; live calibrations open)

Tier manifests committed: `manifests/tier-{small,medium,large,xlarge}.json`. Task-gap coverage across the suite:

- workspace discovery + patch execution + needle (small)
- tool-error recovery, ambiguous-spec triage, format-drift under length, AGENTS/SOUL adherence (medium)
- cross-file consistency, plan/action alignment (large)
- 128k needle, destructive-action refusal under social pressure (xlarge)

Calibration schema validation lives in `openclaw_bench/calibration.py` with a regression test. **Live floor/ceiling records are not yet in place** — Qwen3.5-4B times out on tier-small under the current 32k vLLM setup, so a different floor candidate is needed before this milestone closes.

### M3 — Provider breadth (vLLM shipped; 3 stubs)

`openclaw_bench/providers/` package with:

- Probe abstraction (`LocalProbe`, `IncusExecProbe`, `DockerExecProbe`, `SSHProbe`).
- Detection cascade — scans existing OC profiles first, port-probes second, with a 30s/provider hard cap.
- Runtime auto-derive — reads `gateway.runtime` from `~/.openclaw-<profile>/openclaw.json` to auto-pick the right probe; native installs get a single LocalProbe with no setup.
- Mismatch finding — `reachable_from_host_not_runtime` fires when a host probe succeeds but a runtime-side probe fails (the GPT-OSS UFW class).
- vLLM module (full): `detect`, `generate_route_config` (delegates to `quickstart._vllm_provider_config`, inheriting the 16k context floor + meta + plugin entry fixes), `parameter_shaping` (encodes `chatTemplateKwargs.enable_thinking=false` for Qwen, `extra_body.reasoning_effort="low"` for GPT-OSS).
- Ollama, llama.cpp, LM Studio modules: `detect` only; `generate_route_config` raises `NotImplementedError`.

CLI surface:

- `oc-bench init --providers local` runs the cascade automatically. `--no-detect` falls back to the env-var path. `--oc-runtime <kind:target>` (incus / docker / ssh) for split runtimes.
- `oc-bench provider-preflight` wraps four verification gates: `openclaw config validate`, `openclaw models list --provider X`, provider health probe (with auth header forwarding via `VLLM_API_KEY`), and OpenClaw route smoke. Exit 0 only if all four pass.

Live test gated by `OC_BENCH_LIVE=1` passes against GPT-OSS 20B via `oc-stack` with no host-vs-runtime mismatch finding. Simulator certification full run still produces 40 attempts / 0 failures. 293 unit tests pass (1 live test skipped under default env).

## Open items

1. **M3 — flesh out the three detect-only stubs.** Each is a self-contained follow-up: stand up the runtime, capture its real `/v1/models` (or `/api/tags`) shape, write the generator + tests, swap `NotImplementedError` for real config. Suggested order: Ollama → llama.cpp → LM Studio. Ollama on the RTX PRO 5000 is the obvious next step since the same hardware runs GPT-OSS.
2. **M2 — capture a tier-small calibration record.** Qwen3.5-4B is below the small floor under this OpenClaw/vLLM setup (small-tier tasks time out at 600+ seconds). Try a different 4–8B candidate the new detection surface can discover (Llama 3.2 3B or Qwen3-8B via Ollama once that's live). One live run with full task suite at 32768 context, scored, recorded in this file.
3. **M2 — capture floor/ceiling for medium / large / xlarge.** Same shape, larger models. GPT-OSS 20B is the medium-tier floor candidate per [GOAL.md](GOAL.md).
4. **OpenClaw `2026.4.29` regression**: blocked from upgrade until the observed issues are diagnosed. No work scheduled here.

## Durable operational notes

- The host UFW must allow `oc-stack` (`10.68.198.10`) to reach the GPT-OSS port (`10.68.198.1:8000/tcp`). Without that rule, container requests sit in `SYN-SENT` until OpenClaw times out — looks like a model bug from the gateway logs.
- Generated OpenClaw configs include pinned `meta.lastTouchedVersion = 2026.4.27` and the vLLM plugin entry. Without those, `openclaw config validate` silently restores the prior last-good config and the new context settings are discarded.
- `oc-bench init` writes the model manifest with both an `api_base` (the real vLLM URL) and a `route_model` (`vllm/<model>`); the OpenClaw provider config gets a clamped context window (max(`vllm_context`, 16000)) so OpenClaw can route even when the benchmark task asks for a 4k context.
- Per-run workspaces under `/tmp/oc-bench-root-*/workspaces/<run-id>/` are disposable. Per-run `results/<run-id>/` directories are not — they are the audit trail.
- Stop benchmark-owned root OpenClaw gateways before runs: `incus exec oc-stack -- bash -lc "ss -tlnp | grep -E '(19191|19193|192[89][0-9])'"` and kill stragglers.

## Latest live run

`live-m2-small-floor-qwen35-fixed-20260502002059`

- Code commit: `326024c` (pre-M3-slice)
- Result directory: `/tmp/oc-bench-root-m2-calib-20260502002059/results/live-m2-small-floor-qwen35-fixed-20260502002059`
- Suite: `manifests/tier-small.json`, model `qwen3.5-4b`, KV `provider_default`, context 32768
- Outcome: 2 attempts, 2 failures, both `openclaw_timeout` (666s and 624s wall time)
- Verdict: not a calibration record. Qwen3.5-4B is below the small floor under this setup.

History of the M2 task-design slice commits and the M3 deployment-surface slice commits is in `git log master`. This file no longer enumerates them — it is current state, not a running log.
