# Quickstart

`oc-bench init` creates an isolated `benchclaw` OpenClaw profile, chooses a loopback gateway port, writes a generated loopback gateway token, writes a local benchmark root, and generates starter suite/model manifests from a provider selection.

Omit `--providers` for the wizard, or pass `local`, `api`, or `both` for non-interactive setup.

```bash
oc-bench init --providers local
```

Local provider routes should connect to an existing local runtime such as vLLM, llama.cpp, Ollama, or an Apple Silicon local provider; the target setup flow is inspect-first, ask-second, so the CLI discovers what is already running before prompting the user for missing details. The quickstart manifest does not include a `serve_command` for discovered external providers, so it will not start, stop, restart, or containerize the user's model runtime. API-key routes for OpenAI and Anthropic are added next. OAuth-backed providers are bring-your-own-auth for this phase and should be configured directly in the `benchclaw` profile before running them.

Generated benchmark profiles set `agents.defaults.skipBootstrap=true`; each copied benchmark workspace is seeded with OpenClaw-style `AGENTS.md`, `SOUL.md`, `TOOLS.md`, `IDENTITY.md`, `USER.md`, `HEARTBEAT.md`, and completed workspace state, but no `BOOTSTRAP.md`.

## Pointing at an existing host vLLM

For an `oc-stack` profile that should use a small host vLLM service on a separate port, point the generated route at the vLLM host:

```bash
oc-bench init --providers local \
  --vllm-base-url http://<vllm-host>:<port>/v1 \
  --vllm-model qwen3.5-4b \
  --vllm-context 32768 \
  --vllm-max-tokens 128
```

The author's lab uses `http://10.68.198.1:8003/v1` (Incus host bridge); replace with your own host/port. See `deploy/openclaw-vllm-small-bench.service` for the matching systemd unit on this lab — copy and edit before reusing on another machine.

The repo's example service binds Qwen3.5 4B to GPU 0, uses served model name `qwen3.5-4b`, sets `--max-model-len 32768` for a 32k OpenClaw route context, enables vLLM auto tool choice with the Qwen3 coder parser for OpenClaw agent tool calls, and uses eager mode so the 16GB A4000 has enough memory for the 32k KV cache. Generated quickstart profiles mark local vLLM models as `reasoning=false` and set `chatTemplateKwargs.enable_thinking=false`, which prevents Qwen reasoning-only terminal turns from failing OpenClaw route smoke.

## One-command starter flow

```bash
oc-bench quickstart --providers local --force --stop-after
```

This initializes the same isolated profile, starts only the `benchclaw` gateway, runs preflight, executes the discovery smoke benchmark, prints the result path, and stops only that gateway afterward.

## Lifecycle helpers

Scoped to the benchmark profile only:

```bash
oc-bench start
oc-bench stop
```

The quickstart is not the full certification/upload flow. Full certification matrices, long-context and local quant sweeps, broad external-provider runs, and upload/database integration remain later phases — see [certification.md](certification.md).
