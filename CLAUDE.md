# CLAUDE.md — pointers for in-session agents

Companion to `AGENTS.md` (process), `LEARNINGS.md` (findings), `STATUS.md` (live state). This file orients an in-session Claude/Codex/Gemini agent to material that lives outside the repo or that requires operator-private context.

Read `AGENTS.md` first. Then `STATUS.md`. Then this file.

## Operator-private state lives in `LOCAL-NOTES.md`

`LOCAL-NOTES.md` is gitignored. It holds host-specific state that does not belong in a public repo: IP addresses, API key values, channel IDs, run IDs, paths under `/home/<user>/.openclaw/`, GPU model identifiers, exact systemd unit names tied to one host.

If you need a concrete value to reproduce or debug something, look there first. Public files in this repo must reference *shapes* — "the host's vLLM endpoint", "the configured local API key", "the agent's Discord group channel" — not the concrete values.

If you find concrete operator state in a committed file, scrub it and add the equivalent shape reference instead. The split is durable: public = portable findings, private = this-host-this-week.

## OpenClaw CLI surface — known gap

Some OpenClaw CLI command paths cannot resolve `gateway.auth.token` when it is a secret reference, surfacing as `GatewaySecretRefUnavailableError` from `openclaw health`, `openclaw models status --probe`, and `openclaw agent`. To verify a locally-served provider in this repo's flow, prefer:

- `openclaw infer model providers` — lists providers from local config
- `openclaw infer model run --model <id> --prompt ...` — direct provider call, runs `via local` and does not require gateway auth

Runtime agent flows through the gateway are unaffected; the gateway resolves its own secrets at startup. This gap blocks only CLI clients that re-dial the gateway from outside.

If you must use a `health`/`probe`/`agent` CLI path, see `LOCAL-NOTES.md` for the env-var workaround.

## Compose-managed vLLM (host-side, outside this repo)

The host's vLLM service is now compose-managed (env-driven profile files + a small `vllm-up`/`vllm-down`/`vllm-status` wrapper). This is operator-side state, not bench-state; the bench should treat the vLLM endpoint as a black-box `OpenAI-compatible URL with a configured API key` and discover details via `oc-bench init --providers local`.

If a bench profile needs a pinned vLLM image version (per the OpenClaw `2026.4.27` pin discipline), encode it in a bench-owned profile fixture under `openclaw-config/`; do not mutate operator profiles from the bench.

Concrete paths and image tags live in `LOCAL-NOTES.md`.

## STATUS.md drift discipline

`STATUS.md` "wins on current state" per `AGENTS.md`. When operator-side host state changes (model swapped, image upgraded, GPU pin moved, ports remapped), `STATUS.md` must be updated to match — but **without** restating values that belong in `LOCAL-NOTES.md`. Reference the endpoint by shape; capture the model class and the bench-relevant constraints (context window, parameter shaping, expected support); leave host specifics out.
