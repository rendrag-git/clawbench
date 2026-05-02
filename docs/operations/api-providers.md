# API and Subscription Providers

API/subscription provider examples live in `manifests/api-providers.example.json`. They intentionally reference env var names such as `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`; secrets stay in the shell or OpenClaw profile, not in benchmark manifests.

`<bench-root>` is the benchmark root from `oc-bench init`.

## Configure providers in OpenClaw

Repo-owned non-secret OpenClaw provider examples live alongside the vLLM route config:

```bash
openclaw --profile bench config set \
  models.providers.openai \
  "$(jq -c . openclaw-config/openai-provider.example.json)" \
  --strict-json \
  --dry-run

openclaw --profile bench config set \
  models.providers.anthropic \
  "$(jq -c . openclaw-config/anthropic-provider.example.json)" \
  --strict-json \
  --dry-run
```

Review dry-run output before applying either config. These examples reference env SecretRefs and do not store API keys in `openclaw.json`.

## Preflight

Run preflight with `--smoke-turn`; for API/subscription routes this is the readiness gate because the harness does not start a local model server:

```bash
python3 -m openclaw_bench preflight \
  --backend openclaw \
  --openclaw-local \
  --smoke-turn \
  --suite manifests/openclaw-agent-core.json \
  --model-config manifests/api-providers.example.json \
  --out <bench-root>/results
```

## Run the suites

```bash
python3 -m openclaw_bench run \
  --backend openclaw \
  --openclaw-local \
  --openclaw-workspace-agents \
  --suite manifests/openclaw-agent-core.json \
  --model-config manifests/api-providers.example.json \
  --out <bench-root>/results \
  --run-id api-core

python3 -m openclaw_bench run \
  --backend openclaw \
  --openclaw-local \
  --openclaw-workspace-agents \
  --suite manifests/real-repo-readonly.example.json \
  --model-config manifests/api-providers.example.json \
  --out <bench-root>/results \
  --run-id api-real-repo
```
