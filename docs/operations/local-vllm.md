# Local vLLM Operations

Operator cookbook for live local vLLM runs. The host paths and IPs in examples are from the author's lab — substitute your own.

`<bench-root>` is the benchmark root from `oc-bench init`. `<vllm-host>` is the address vLLM is reachable at from where OpenClaw will route from (the host, container, or LXC depending on your setup). The author's lab uses `10.68.198.1:8000` (Incus host bridge); see [STATUS.md](../../STATUS.md) for the full topology of this lab.

## Preflight a focused vLLM manifest

The focused vLLM manifests start and probe `vllm serve` on `http://127.0.0.1:8000/v1` by default. Set the same env var OpenClaw's vLLM route expects:

```bash
export VLLM_API_KEY=vllm-local

python3 -m openclaw_bench preflight \
  --backend openclaw \
  --openclaw-local \
  --suite manifests/openclaw-agent-core.json \
  --model-config manifests/vllm-gptoss-smoke.example.json \
  --out <bench-root>/results
```

## Configure the vLLM provider in OpenClaw

The local OpenClaw `bench` profile needs a vLLM provider route before `--openclaw-local` task runs can use names such as `vllm/gpt-oss-20b-nvfp4-smoke`. Keep the key in the environment and configure the provider to read it as a bearer token. The repo-owned, non-secret example is `openclaw-config/vllm-provider-smoke.example.json`; it caps provider output at 256 tokens so smoke turns leave room for OpenClaw's gateway prompt and tools. OpenClaw `2026.4.27` rejects route context windows below 16000 tokens, so benchmark manifests may still label a row as 4k/8k while the OpenClaw provider config uses a 16000-token route window.

```bash
openclaw --profile bench config set \
  models.providers.vllm \
  "$(jq -c . openclaw-config/vllm-provider-smoke.example.json)" \
  --strict-json \
  --dry-run
```

Review the dry-run output first. Remove `--dry-run` only when you are ready to create or update the `bench` profile config. If you change `served_model_name` in a benchmark manifest, add the same model id/name under `models.providers.vllm.models` or OpenClaw route smoke will fail even when direct vLLM probes pass.

## Qwen3.6 host endpoint (this lab)

For the isolated `oc-bench` container consuming the host Qwen3.6 vLLM endpoint, use the repo-owned merge examples. They declare the benchmark row as 8k in the manifest, use an OpenClaw-supported 16000-token provider route, cap output at 128 tokens, and disable Qwen thinking through `chatTemplateKwargs.enable_thinking=false` so OpenClaw agent turns have enough prompt/output budget for the gateway system prompt and repo tools.

The merge files reference `http://10.68.198.1:8000/v1` directly — edit `baseUrl` for your host before applying.

```bash
openclaw --profile bench config set \
  models.providers.vllm \
  "$(jq -c . openclaw-config/qwen36-vllm-provider.merge.example.json)" \
  --strict-json \
  --merge \
  --dry-run

openclaw --profile bench config set \
  'agents.defaults.models["vllm/qwen3.6-35b-a3b"].params' \
  "$(jq -c . openclaw-config/qwen36-agent-default-params.example.json)" \
  --strict-json \
  --dry-run
```

## Lean 8k variant

If the full OpenClaw gateway profile still overflows the 8k Qwen route, treat the next run as a separate lean-control-plane variant rather than a baseline continuation. The live failure mode is prompt budget, not model routing: verbose gateway logs showed the current 64-token setting reject `8129 input + 64 output = 8193 > 8192`. Keep normal repo-inspection tools available; the earlier `tools.profile="minimal"` variant removed `exec` and produced an unfair tool-availability failure. These dry-run-validated edits trim workspace bootstrap while preserving the standard tool profile:

```bash
openclaw --profile bench config set \
  models.providers.vllm \
  "$(jq -c . openclaw-config/qwen36-vllm-provider-lean8k.merge.example.json)" \
  --strict-json \
  --merge \
  --dry-run

openclaw --profile bench config set \
  agents.defaults.params \
  "$(jq -c . openclaw-config/qwen36-agent-lean-8k-params.example.json)" \
  --strict-json \
  --dry-run

openclaw --profile bench config set \
  agents.defaults \
  "$(jq -c . openclaw-config/qwen36-agent-lean-8k-defaults.example.json)" \
  --merge \
  --strict-json \
  --dry-run

openclaw --profile bench config unset tools.profile --dry-run
```

Remove `--dry-run` only when intentionally producing a lean 8k result row. Use `manifests/vllm-qwen36-fp8-lean8k-live.example.json` for that row; it labels the setup as `ctx8192-lean-max32` so it cannot be mixed with the full-profile 8k baseline.

A live agent-smoke preflight still overflowed this exact 8k setup at `8161 input + 32 output = 8193 > 8192`; move this Qwen setup to a larger served context window instead of further removing workspace tools.

## Lean 16k follow-up

The first larger-context follow-up is the lean 16k row. It keeps the same route/model name and agent maxTokens=32 but expects the vLLM server to be restarted with `--gpu-memory-utilization 0.95 --max-model-len 16384` on GPU 1, then the OpenClaw provider context window updated with the 16k merge file:

```bash
openclaw --profile bench config set \
  models.providers.vllm \
  "$(jq -c . openclaw-config/qwen36-vllm-provider-lean16k.merge.example.json)" \
  --strict-json \
  --merge \
  --dry-run
```

Use `manifests/vllm-qwen36-fp8-lean16k-live.example.json` for that result row. If the model does not fit at 16k, record that as an OOM/load failure for the `ctx16384` setup rather than reusing the 8k label.

## Running tasks

Preflight skips OpenClaw smoke turns for harness-started vLLM servers because the process is not running yet. The `run` command starts vLLM, checks `/v1/models`, sends a bounded `/v1/chat/completions` probe using `served_model_name`, then asks OpenClaw to smoke `openclaw_model_name`.

Use the discovery-only smoke suite before spending time on the full core suite when changing prompt-budget or gateway settings:

```bash
python3 -m openclaw_bench run \
  --backend openclaw \
  --openclaw-workspace-agents \
  --openclaw-smoke-timeout 120 \
  --suite manifests/openclaw-agent-discovery-smoke.example.json \
  --model-config manifests/vllm-qwen36-fp8-lean16k-live.example.json \
  --out <bench-root>/results \
  --run-id gateway-vllm-qwen36-lean16k-discovery-smoke
```

```bash
python3 -m openclaw_bench run \
  --backend openclaw \
  --openclaw-local \
  --openclaw-workspace-agents \
  --openclaw-smoke-timeout 120 \
  --suite manifests/openclaw-agent-core.json \
  --model-config manifests/vllm-gptoss-smoke.example.json \
  --out <bench-root>/results \
  --run-id local-vllm-smoke
```

Cold gateway starts can take longer than a single status check, especially with verbose logging and an 8k local route. Increase `--openclaw-gateway-timeout` for gateway startup and `--openclaw-smoke-timeout` for route smoke without changing the per-task `--timeout`.

Use `--openclaw-workspace-agents` for live agent task runs. OpenClaw agent turns use configured agent workspaces, not the subprocess `cwd`; this flag creates one configured benchmark agent per attempt, points it at the copied fixture workspace, and sets the model on that agent so gateway runs do not need a per-call `--model` override.

Gateway lifecycle defaults are intentionally conservative around the selected target: without `--openclaw-container`, the harness starts/checks the local OpenClaw `bench` profile; with `--openclaw-container oc-bench-gateway`, it first ensures that separate container exists, then runs the same OpenClaw commands through `docker exec oc-bench-gateway` and does not touch the host LXC `oc-stack`. Treat `openclaw --profile bench gateway status` as the readiness source of truth for the selected profile; Docker health is advisory and may still report stale image defaults on externally created containers. Containers created by `oc-bench` override the healthcheck to run the same profile-aware gateway status probe.

For readiness checks, preflight uses `--smoke-timeout`; full benchmark runs use `--openclaw-smoke-timeout` for the same OpenClaw route gate.

## Agent-smoke preflight

Use `--agent-smoke-turn` when preflight needs to prove the actual `openclaw agent` path, not just the model route. Pair it with `--openclaw-workspace-agents` for gateway benchmarks so preflight catches agent id mismatches, unauthorized per-call model overrides, missing container mounts, and prompt-budget failures before a full matrix run:

```bash
python3 -m openclaw_bench preflight \
  --backend openclaw \
  --openclaw-container oc-bench-gateway \
  --openclaw-profile bench \
  --openclaw-agent dev \
  --openclaw-workspace-agents \
  --agent-smoke-turn \
  --suite manifests/openclaw-agent-discovery-smoke.example.json \
  --model-config manifests/vllm-qwen36-fp8-live.example.json \
  --out <bench-root>/results \
  --smoke-timeout 120
```

## KV / context / concurrency / hardware sweeps

Use the focused local KV comparison once the smoke cell works:

```bash
python3 -m openclaw_bench run \
  --backend openclaw \
  --openclaw-local \
  --openclaw-workspace-agents \
  --suite manifests/openclaw-agent-core.json \
  --model-config manifests/vllm-local.example.json \
  --out <bench-root>/results \
  --run-id local-vllm-quality
```

Use a dedicated manifest for long-context runs so the reported context limit matches the vLLM server's `--max-model-len`:

```bash
python3 -m openclaw_bench run \
  --backend openclaw \
  --openclaw-local \
  --openclaw-workspace-agents \
  --suite manifests/openclaw-agent-core.json \
  --model-config manifests/vllm-long-context.example.json \
  --out <bench-root>/results \
  --run-id local-vllm-long-context
```

Use a separate manifest for concurrency stress so high worker counts are not accidentally applied to every quality candidate:

```bash
python3 -m openclaw_bench run \
  --backend openclaw \
  --openclaw-local \
  --openclaw-workspace-agents \
  --suite manifests/openclaw-agent-core.json \
  --model-config manifests/vllm-concurrency-sweep.example.json \
  --out <bench-root>/results \
  --run-id local-vllm-concurrency
```

Use `manifests/vllm-hardware-setups.example.json` to compare local serve setups that keep model/KV fixed while varying hardware-facing vLLM settings such as GPU memory utilization and eager mode:

```bash
python3 -m openclaw_bench run \
  --backend openclaw \
  --openclaw-local \
  --openclaw-workspace-agents \
  --suite manifests/openclaw-agent-core.json \
  --model-config manifests/vllm-hardware-setups.example.json \
  --out <bench-root>/results \
  --run-id local-vllm-hardware-setups
```

## Real-repo suite locally

Certification requires local-provider passes for every required task type, including `repo_read_only` and `repo_code_edit`; API/subscription real-repo passes do not substitute for local model behavior:

```bash
python3 -m openclaw_bench run \
  --backend openclaw \
  --openclaw-local \
  --openclaw-workspace-agents \
  --suite manifests/real-repo-readonly.example.json \
  --model-config manifests/vllm-local.example.json \
  --out <bench-root>/results \
  --run-id local-vllm-real-repo
```

Use `manifests/vllm-local-candidates.example.json` for the broader NVFP4 candidate sweep after the focused smoke and quality run are stable. The manifest examples pin port `8000`; if another service is already bound there, edit the manifest port, `health_check_url`, and `api_base` together before running.

Do not use `--kv` or `--contexts` to override a live local `--model-config` entry that has a `serve_command`. Local vLLM cells need the `served_model_name`, `openclaw_model_name`, `--kv-cache-dtype`, and `--max-model-len` to agree, so use a manifest that declares the exact cell instead of mutating only metadata at the CLI layer.
