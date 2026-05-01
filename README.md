# OpenClaw Local Model Benchmark Design

Current live testing state is tracked in [STATUS.md](STATUS.md).

## Goal

Measure whether a local model is useful for OpenClaw agent work, not just whether it serves tokens quickly.

The benchmark should answer:

- Can the model use OpenClaw's normal agent workflow to inspect workspace files?
- Can it keep instructions and repository facts straight across long contexts?
- Can it make correct code changes under realistic tool/file pressure?
- Does KV-cache quantization reduce quality, latency, or concurrency enough to matter?
- How many concurrent OpenClaw agent turns are usable on the local GPU server?

Raw vLLM throughput is a diagnostic. The primary score is OpenClaw task success.

## Non-Goals

- Do not benchmark by pasting all files directly into the prompt. The benchmark must let OpenClaw agents load and inspect files the way they normally do.
- Do not use production OpenClaw workspaces or sessions as mutable test fixtures.
- Do not force unsupported model/KV combinations. Record load/runtime failure as a benchmark result.
- Do not compare different weight quantizations unless that is the explicit test axis. For the KV study, keep model weights fixed and vary only KV cache mode.

## Benchmark Shape

Run two layers:

1. **Serve Layer**
   - Starts one model server per model/KV combination.
   - Records load success, load time, VRAM use, max context, OOMs, and vLLM serving metrics.
   - Uses the existing `bench_vllm_kv.py` style for baseline throughput.

2. **OpenClaw Layer**
   - Runs real `openclaw agent --json` turns against isolated benchmark workspaces.
   - Uses fixed task manifests and deterministic scoring.
   - Measures task success, file-use correctness, patch correctness, latency, tool count, and failure modes.

## Isolation

Use a dedicated profile and per-run workspace copies:

```text
/home/ubuntu/openclaw-bench/
  manifests/
  fixtures/
  workspaces/
    <run-id>/
      worker-000/
      worker-001/
  results/
    <run-id>/
      config.json
      raw/
      summaries/
      patches/
```

OpenClaw calls should use a benchmark profile/session id. For gateway runs, bind the model on the benchmark agent instead of passing a per-call model override:

```bash
openclaw --profile bench agent \
  --agent <bench-agent-id> \
  --session-id <run-id>-<task-id>-<worker-id> \
  --message "<task prompt>" \
  --timeout <seconds> \
  --json
```

Each concurrent worker gets a separate workspace copy so code-edit tasks do not conflict.

Fresh clones do not need a separate manual gateway start before the first benchmark. After installing the repo, `oc-bench` and `openclaw-bench` are equivalent entrypoints. For non-local OpenClaw runs, `oc-bench run` and `oc-bench preflight` ensure the selected `--openclaw-profile` gateway by default, using `openclaw --profile bench gateway --dev --verbose run` as a detached foreground gateway when the gateway is not already reachable, then polling readiness for up to `--openclaw-gateway-timeout` seconds. Gateway runs also default to configured benchmark workspace agents so model routing is bound on the agent instead of sent as an unauthorized per-call override. Use `--no-ensure-openclaw-gateway` only when a supervisor or container entrypoint already owns gateway lifecycle.

OpenClaw is pinned to `2026.4.27` for this harness. `2026.4.29` is blocked for benchmark runs until the regressions are resolved:

```bash
npm install -g openclaw@2026.4.27
openclaw --version
```

For isolated Docker runs, pass `--openclaw-container oc-bench-gateway`. On first use, `oc-bench` creates or starts that container with image `clawdaddy/openclaw:business-smoke-2026.4.27`, host networking, a separate home at `/home/ubuntu/openclaw-bench/container-home`, and exact-path mounts for the repo, benchmark root, and any custom workspace root. The container stays as a reusable OpenClaw runtime; the harness still starts/checks the selected `bench` gateway through `openclaw --profile bench gateway status` and a detached verbose foreground gateway when needed. This path is intentionally separate from any host or LXC OpenClaw install. Use `--no-ensure-openclaw-container` only when another supervisor already owns the container.

## Model Matrix

Each model entry should declare:

- `model_id`: Hugging Face id or local path.
- `served_model_name`: OpenAI-compatible name.
- `openclaw_model_name`: optional OpenClaw route name or alias when it differs from the vLLM/OpenAI-compatible served name.
- `comparison_id`: optional stable model family id for FP8-vs-TurboQuant comparisons when served names or provider routes differ.
- `hardware_profile`: stable label for the local serve setup, such as GPU target, memory-utilization cap, eager/graph mode, and context plan.
- `weight_quant`: `bf16`, `fp8`, `nvfp4`, `awq`, etc.
- `serve_args`: vLLM args needed to load it.
- `expected_support`: known constraints such as attention sinks, hybrid model, sliding window.

For the current local-model question, keep two tracks:

### Track A: Desired Production Candidates

These are the models we actually care about for OpenClaw use:

- GPT-OSS 20B NVFP4
- Qwen3.6 NVFP4
- Gemma 4 NVFP4
- Nemotron NVFP4
- Qwen3 Coder / Qwen3-Next NVFP4

KV modes:

- `fp8`
- `turboquant_k8v4`
- `turboquant_k3v4_nc`

Unsupported combinations are recorded, not retried endlessly.

### Track B: TurboQuant Sanity Candidate

Add one dense, normal-attention Qwen model that current vLLM can serve with TurboQuant KV. This separates "TurboQuant is useful on this machine" from "our desired model architecture is blocked."

KV modes:

- `fp8`
- `turboquant_k8v4`
- `turboquant_k3v4_nc`
- optionally `turboquant_4bit_nc`

## Task Suite

Use a mix of synthetic fixture repos and real local repos. Synthetic fixtures give exact scoring. Real repos catch usability problems synthetic tasks miss.

### 1. Workspace Discovery

Purpose: tests whether the agent can inspect a repo instead of hallucinating.

Prompt shape:

```text
In this workspace, identify the command that runs the API tests, the file that defines the API routes, and the file that defines the database schema. Return only JSON with keys test_command, routes_file, schema_file.
```

Score:

- Exact file paths match expected answers.
- Command is runnable.
- No nonexistent files.
- JSON is valid.

### 2. Multi-File Bug Trace

Purpose: tests repository reasoning across multiple files.

Fixture:

- A small app with a failing test.
- Bug requires reading at least 3 files: route, domain helper, test.

Prompt:

```text
Find why the named test fails. Explain the bug path with file references, then make the smallest code change to fix it. Do not change tests unless the test is wrong.
```

Score:

- Correct root cause.
- Patch touches expected file(s).
- Test passes.
- No broad unrelated rewrite.

### 3. Patch Execution

Purpose: tests if the model can produce useful code changes, not just analysis.

Fixture:

- TypeScript or Python repo with one narrow feature missing.
- Tests exist but fail until the feature is implemented.

Score:

- Tests pass.
- Patch size within limit.
- No new lint/type errors.
- No unrelated file churn.

### 4. Workspace Needle

Purpose: OpenClaw-flavored Needle-in-a-Haystack.

Fixture:

- A repo contains a buried fact in a realistic file, such as:
  - migration note
  - config comment
  - changelog entry
  - test fixture
- Distractor files contain similar but wrong facts.

Prompt:

```text
Find the current value of BENCHMARK_NEEDLE_TOKEN in the workspace and use it to update the health endpoint response. Do not guess; cite the file where you found it.
```

Context sizes:

- 4k
- 8k
- 16k
- 32k
- 64k if the model/server can support it

Score:

- Finds the correct needle.
- Does not use distractor value.
- Correctly applies it in code.
- Test passes.

### 5. Instruction Retention

Purpose: tests long-context obedience and constraint retention.

Prompt includes constraints at the beginning, then asks for a code task after repository exploration.

Example constraints:

- Do not edit tests.
- Do not add dependencies.
- Return final answer as JSON.
- Use the existing helper instead of adding a new abstraction.

Score:

- Task succeeds.
- All constraints preserved, verified from artifacts: tests/dependencies are unchanged, the target imports and calls the existing helper, and no replacement helper abstraction is added.
- No late-turn format drift.

### 6. Tool/File Efficiency

Purpose: catches models that eventually solve tasks but burn too many steps.

Score:

- Number of turns/tool calls.
- Duplicate file reads.
- Time to first relevant file.
- Time to final patch.

This should not dominate quality, but it matters for cost and latency.

### 7. Concurrent Agent Work

Purpose: tests real multi-agent usability.

Run the same task set across independent workspace copies at:

```text
1, 2, 4, 8, 16, 32, 64 concurrent agent turns
```

For large models, practical decision points are likely 4, 8, and 16. Higher levels are stress tests.

Score:

- Success rate by concurrency.
- P50/P95/P99 wall time.
- P50/P95/P99 time to first output.
- Server errors.
- OOMs.
- OpenClaw timeouts.

## Metrics

Collect one result row per task attempt:

```json
{
  "run_id": "20260501-qwen3-k8v4",
  "model": "qwen3-dense",
  "served_model_name": "qwen3-dense",
  "hardware_profile": "gpu1-vllm-4k",
  "weight_quant": "bf16",
  "kv_cache_dtype": "turboquant_k8v4",
  "context_limit": 32768,
  "concurrency": 8,
  "task_id": "workspace-needle-16k",
  "workspace_id": "worker-003",
  "status": "pass",
  "score": 1.0,
  "wall_time_s": 84.2,
  "ttft_s": 3.1,
  "tool_calls": 14,
  "files_read": 9,
  "duplicate_file_reads": 2,
  "time_to_first_relevant_file_s": 11.4,
  "files_changed": 1,
  "tests_passed": true,
  "json_valid": true,
  "hallucinated_paths": 0,
  "oom": false,
  "timeout": false,
  "notes": ""
}
```

Also collect server-level telemetry:

- model load time
- steady VRAM before run
- peak VRAM during run
- GPU utilization
- server crashes/restarts
- request errors
- tokens/sec
- TTFT and TPOT if available

## Scoring

Use quality gates before speed comparisons.

Suggested aggregate:

```text
OpenClaw Usability Score =
  45% task pass rate
  20% patch/test correctness
  15% workspace retrieval accuracy
  10% instruction retention / output format
  10% latency under target concurrency
```

Report both:

- absolute score
- relative score versus the same model with `fp8` KV

KV quantization should be considered acceptable only if:

- task quality is at least 95% of `fp8` for `turboquant_k8v4`
- task quality is at least 90% of `fp8` for `turboquant_k3v4_nc`
- no systematic workspace-needle failures appear
- latency or concurrency capacity improves enough to justify any quality loss

## Failure Taxonomy

Every failed attempt should be classified:

- `model_load_failed`
- `unsupported_kv_dtype`
- `oom_on_load`
- `oom_during_run`
- `openclaw_timeout`
- `server_timeout`
- `bad_json`
- `wrong_file`
- `hallucinated_file`
- `wrong_needle`
- `test_failed`
- `patch_unrelated`
- `instruction_violation`
- `tool_loop`
- `tool_parser_missing`
- `context_window_exceeded`
- `incomplete_result`
- `openclaw_embedded_fallback`
- `unknown`

This matters because "TurboQuant unsupported" and "TurboQuant makes the model dumb" are different outcomes. If an attempt reports `incomplete_result` or `openclaw_embedded_fallback`, inspect the verbose gateway log or raw stderr for the underlying cause; recent Qwen 8k runs used that wrapper for prompt-budget failures that should be interpreted as `context_window_exceeded` when the verbose log shows a context-length rejection.

## Run Phases

### Phase 0: Smoke

One model, one KV mode, one simple discovery task, concurrency 1.

Pass condition:

- OpenClaw can route to the local model.
- Agent can inspect the benchmark workspace.
- Result JSON is captured.

### Phase 1: Model/KV Support Probe

For each model/KV mode:

- start server
- send one tiny agent task
- record support or failure

No full benchmark until support is known.

### Phase 2: Quality Baseline

Run the full task suite at concurrency 1 and 4.

Purpose:

- identify quality regressions before spending time on high-concurrency tests

### Phase 3: Concurrency Sweep

Run selected tasks at:

```text
1, 2, 4, 8, 16, 32, 64
```

Purpose:

- identify usable concurrency range
- detect latency cliffs
- detect OOM thresholds

### Phase 4: Long-Context / Needle

Run workspace needle tasks at:

```text
4k, 8k, 16k, 32k, 64k
```

Purpose:

- detect KV quantization quality loss
- decide if memory savings are worth it

### Phase 5: Real Repo Tasks

Use local repos such as:

- `/home/ubuntu/projects/kingshot-ams`
- `/home/ubuntu/projects/chatterbox-tts-api`

Tasks should be read-only first, then patch tasks only on copied workspaces.

Purpose:

- catch behavior that synthetic fixtures miss

## Reporting

Each run should produce:

- `summary.md`: human-readable winner/loser table
- `summary.json`: machine-readable aggregate
- `config.json`: benchmark matrix, gateway ensure result, runtime identity, and suite/task and fixture provenance digests
- `attempts.jsonl`: one row per task attempt
- `failures.jsonl`: failed attempts with taxonomy
- `server.json`: load/runtime telemetry, host GPU inventory, route probes, and throughput probe evidence
- `raw/<workspace_id>.json`: per-attempt raw OpenClaw/model output, command provenance, session id, task id/type, workspace id, and model cell metadata
- `patches/<workspace_id>.diff`: per-attempt workspace diff

Summary tables:

```text
Model | Weight | KV | Ctx | Concurrency | Pass % | Needle % | Patch % | P95 Wall | Peak VRAM | Notes
```

Decision table:

```text
Use case | Best model/KV | Reason | Risk
single-agent coding | ... | ... | ...
4-agent background work | ... | ... | ...
long-context repo search | ... | ... | ...
64-concurrency stress | ... | ... | ...
```

## Initial Recommendation

Do not start with every desired model.

Start with:

1. GPT-OSS 20B NVFP4 + `fp8` KV as the current known working local candidate.
2. One dense Qwen model + `fp8`, `turboquant_k8v4`, `turboquant_k3v4_nc` as the TurboQuant sanity check.

Then decide:

- If dense Qwen TurboQuant fails quality, stop chasing TurboQuant for OpenClaw right now.
- If dense Qwen `k8v4` keeps quality and improves memory/concurrency, patch vLLM for Qwen3.6/Qwen3-Next hybrid TurboQuant next.
- GPT-OSS TurboQuant should wait until the attention-sink path is patched.
- Gemma 4 TurboQuant should wait until the Gemma/sliding-window path is patched.

## Target One-Line Command

The eventual local core-suite command should look like:

```bash
openclaw-bench run \
  --suite openclaw-agent-core \
  --models gpt-oss-20b-nvfp4,qwen3-dense \
  --kv fp8,turboquant_k8v4,turboquant_k3v4_nc \
  --concurrency 1,2,4,8,16,32,64 \
  --contexts 4096,8192,16384,32768,65536 \
  --out /home/ubuntu/openclaw-bench/results
```

That command should internally start the model server, run support probes, execute OpenClaw tasks, collect telemetry, and write the report. It is not by itself a certification-complete run; certification also requires local real-repo read-only and code-edit rows plus API and subscription provider rows with route-probe evidence.

## Current Harness Commands

The repository includes a runnable Python harness with:

- a simulator backend for proving the benchmark mechanics;
- an OpenClaw backend for live local, API, and subscription provider runs;
- a `certify` command that audits result directories against the full benchmark objective.

Use the simulator to validate harness changes. Do not treat simulator output as model certification.
The combined task manifest for certification-oriented live runs is `manifests/openclaw-certification-full.example.json`; it includes the core synthetic tasks and the real-repo read-only/code-edit suite.

### Quickstart

`oc-bench init` creates an isolated `benchclaw` OpenClaw profile, chooses a loopback gateway port, writes a generated loopback gateway token, writes a local benchmark root, and generates starter suite/model manifests from a provider selection. Omit `--providers` for the wizard, or pass `local`, `api`, or `both` for non-interactive setup. Local provider routes should connect to an existing local runtime such as vLLM, llama.cpp, Ollama, or an Apple Silicon local provider; the target setup flow is inspect-first, ask-second, so the CLI should discover what is already running before prompting the user for missing details. The quickstart manifest does not include a `serve_command` for discovered external providers, so it will not start, stop, restart, or containerize the user's model runtime. API-key routes for OpenAI and Anthropic are added next. OAuth-backed providers are bring-your-own-auth for this phase and should be configured directly in the `benchclaw` profile before running them. Generated benchmark profiles set `agents.defaults.skipBootstrap=true`; each copied benchmark workspace is seeded with OpenClaw-style `AGENTS.md`, `SOUL.md`, `TOOLS.md`, `IDENTITY.md`, `USER.md`, `HEARTBEAT.md`, and completed workspace state, but no `BOOTSTRAP.md`.

```bash
oc-bench init --providers local
```

For an `oc-stack` profile that should use a small host vLLM service on a separate port, point the generated route at the Incus host bridge address:

```bash
oc-bench init --providers local \
  --vllm-base-url http://10.68.198.1:8003/v1 \
  --vllm-model qwen3.5-4b \
  --vllm-context 32768 \
  --vllm-max-tokens 128
```

The repo includes `deploy/openclaw-vllm-small-bench.service` for a persistent Qwen3.5 4B API on GPU 0. It binds only to `10.68.198.1:8003`, uses served model name `qwen3.5-4b`, sets `--max-model-len 32768` for a 32k OpenClaw route context, enables vLLM auto tool choice with the Qwen3 coder parser for OpenClaw agent tool calls, and uses eager mode so the 16GB A4000 has enough memory for the 32k KV cache. Generated quickstart profiles mark local vLLM models as `reasoning=false` and set `chatTemplateKwargs.enable_thinking=false`, which prevents Qwen reasoning-only terminal turns from failing OpenClaw route smoke.

The one-command starter flow initializes the same isolated profile, starts only the `benchclaw` gateway, runs preflight, executes the discovery smoke benchmark, prints the result path, and can stop only that gateway afterward:

```bash
oc-bench quickstart --providers local --force --stop-after
```

Lifecycle helpers are intentionally scoped to the benchmark profile:

```bash
oc-bench start
oc-bench stop
```

The quickstart is not the full certification/upload flow. Full certification matrices, long-context and local quant sweeps, broad external-provider runs, and upload/database integration remain later phases.

### 1. Mechanics Smoke

```bash
python3 -m openclaw_bench run \
  --backend simulator \
  --suite manifests/openclaw-agent-core.json \
  --model-config manifests/initial-models.json \
  --out /home/ubuntu/openclaw-bench/results
```

Run the real-repo suite too when changing scoring, workspace isolation, or report generation. It includes read-only tasks plus a copied-workspace code-edit task:

```bash
python3 -m openclaw_bench run \
  --backend simulator \
  --suite manifests/real-repo-readonly.example.json \
  --models simulated-model \
  --kv fp8 \
  --concurrency 1 \
  --contexts 4096 \
  --out /home/ubuntu/openclaw-bench/results
```

### 2. Local vLLM Smoke

The focused vLLM manifests start and probe `vllm serve` on `http://127.0.0.1:8000/v1`. Set the same env var OpenClaw's vLLM route expects:

```bash
export VLLM_API_KEY=vllm-local

python3 -m openclaw_bench preflight \
  --backend openclaw \
  --openclaw-local \
  --suite manifests/openclaw-agent-core.json \
  --model-config manifests/vllm-gptoss-smoke.example.json \
  --out /home/ubuntu/openclaw-bench/results
```

The local OpenClaw `bench` profile also needs a vLLM provider route before `--openclaw-local` task runs can use names such as `vllm/gpt-oss-20b-nvfp4-smoke`. Keep the key in the environment and configure the provider to read it as a bearer token. The repo-owned, non-secret example is `openclaw-config/vllm-provider-smoke.example.json`; it caps provider output at 256 tokens so smoke turns leave room for OpenClaw's gateway prompt and tools.

```bash
openclaw --profile bench config set \
  models.providers.vllm \
  "$(jq -c . openclaw-config/vllm-provider-smoke.example.json)" \
  --strict-json \
  --dry-run
```

Review the dry-run output first. Remove `--dry-run` only when you are ready to create or update the `bench` profile config. If you change `served_model_name` in a benchmark manifest, add the same model id/name under `models.providers.vllm.models` or OpenClaw route smoke will fail even when direct vLLM probes pass.

For the isolated `oc-bench` container consuming the host Qwen3.6 vLLM endpoint at `10.68.198.1:8000`, use the repo-owned merge examples. They declare the live 8k route, cap output at 128 tokens, and disable Qwen thinking through `chatTemplateKwargs.enable_thinking=false` so OpenClaw agent turns have enough prompt/output budget for the gateway system prompt and repo tools.

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

Remove `--dry-run` only when intentionally producing a lean 8k result row. Use `manifests/vllm-qwen36-fp8-lean8k-live.example.json` for that row; it labels the setup as `ctx8192-lean-max32` so it cannot be mixed with the full-profile 8k baseline. A live agent-smoke preflight still overflowed this exact 8k setup at `8161 input + 32 output = 8193 > 8192`; move this Qwen setup to a larger served context window instead of further removing workspace tools.

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

Preflight skips OpenClaw smoke turns for harness-started vLLM servers because the process is not running yet. The `run` command starts vLLM, checks `/v1/models`, sends a bounded `/v1/chat/completions` probe using `served_model_name`, then asks OpenClaw to smoke `openclaw_model_name`.

Use the discovery-only smoke suite before spending time on the full core suite when changing prompt-budget or gateway settings:

```bash
python3 -m openclaw_bench run \
  --backend openclaw \
  --openclaw-workspace-agents \
  --openclaw-smoke-timeout 120 \
  --suite manifests/openclaw-agent-discovery-smoke.example.json \
  --model-config manifests/vllm-qwen36-fp8-lean16k-live.example.json \
  --out /home/ubuntu/openclaw-bench/results \
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
  --out /home/ubuntu/openclaw-bench/results \
  --run-id local-vllm-smoke
```

Cold gateway starts can take longer than a single status check, especially with verbose logging and an 8k local route. Increase `--openclaw-gateway-timeout` for gateway startup and `--openclaw-smoke-timeout` for route smoke without changing the per-task `--timeout`.

Use `--openclaw-workspace-agents` for live agent task runs. OpenClaw agent turns use configured agent workspaces, not the subprocess `cwd`; this flag creates one configured benchmark agent per attempt, points it at the copied fixture workspace, and sets the model on that agent so gateway runs do not need a per-call `--model` override.

Gateway lifecycle defaults are intentionally conservative around the selected target: without `--openclaw-container`, the harness starts/checks the local OpenClaw `bench` profile; with `--openclaw-container oc-bench-gateway`, it first ensures that separate container exists, then runs the same OpenClaw commands through `docker exec oc-bench-gateway` and does not touch the host LXC `oc-stack`. Treat `openclaw --profile bench gateway status` as the readiness source of truth for the selected profile; Docker health is advisory and may still report stale image defaults on externally created containers. Containers created by `oc-bench` override the healthcheck to run the same profile-aware gateway status probe.

For readiness checks, preflight uses `--smoke-timeout`; full benchmark runs use `--openclaw-smoke-timeout` for the same OpenClaw route gate.

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
  --out /home/ubuntu/openclaw-bench/results \
  --smoke-timeout 120
```

Use the focused local KV comparison once the smoke cell works:

```bash
python3 -m openclaw_bench run \
  --backend openclaw \
  --openclaw-local \
  --openclaw-workspace-agents \
  --suite manifests/openclaw-agent-core.json \
  --model-config manifests/vllm-local.example.json \
  --out /home/ubuntu/openclaw-bench/results \
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
  --out /home/ubuntu/openclaw-bench/results \
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
  --out /home/ubuntu/openclaw-bench/results \
  --run-id local-vllm-concurrency
```

Use `manifests/vllm-hardware-setups.example.json` to compare local serve setups that keep model/KV fixed while varying hardware-facing vLLM settings such as GPU memory utilization and eager mode.

```bash
python3 -m openclaw_bench run \
  --backend openclaw \
  --openclaw-local \
  --openclaw-workspace-agents \
  --suite manifests/openclaw-agent-core.json \
  --model-config manifests/vllm-hardware-setups.example.json \
  --out /home/ubuntu/openclaw-bench/results \
  --run-id local-vllm-hardware-setups
```

Run the real-repo suite locally as well. Certification requires local-provider passes for every required task type, including `repo_read_only` and `repo_code_edit`; API/subscription real-repo passes do not substitute for local model behavior:

```bash
python3 -m openclaw_bench run \
  --backend openclaw \
  --openclaw-local \
  --openclaw-workspace-agents \
  --suite manifests/real-repo-readonly.example.json \
  --model-config manifests/vllm-local.example.json \
  --out /home/ubuntu/openclaw-bench/results \
  --run-id local-vllm-real-repo
```

Use `manifests/vllm-local-candidates.example.json` for the broader NVFP4 candidate sweep after the focused smoke and quality run are stable. The manifest examples pin port `8000`; if another service is already bound there, edit the manifest port, `health_check_url`, and `api_base` together before running.

Do not use `--kv` or `--contexts` to override a live local `--model-config` entry that has a `serve_command`. Local vLLM cells need the `served_model_name`, `openclaw_model_name`, `--kv-cache-dtype`, and `--max-model-len` to agree, so use a manifest that declares the exact cell instead of mutating only metadata at the CLI layer.

### 3. API And Subscription Providers

API/subscription provider examples live in `manifests/api-providers.example.json`. They intentionally reference env var names such as `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`; secrets stay in the shell or OpenClaw profile, not in benchmark manifests.

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

Run preflight with `--smoke-turn`; for API/subscription routes this is the readiness gate because the harness does not start a local model server:

```bash
python3 -m openclaw_bench preflight \
  --backend openclaw \
  --openclaw-local \
  --smoke-turn \
  --suite manifests/openclaw-agent-core.json \
  --model-config manifests/api-providers.example.json \
  --out /home/ubuntu/openclaw-bench/results
```

Then run both the core suite and the real-repo read-only/code-edit suite:

```bash
python3 -m openclaw_bench run \
  --backend openclaw \
  --openclaw-local \
  --openclaw-workspace-agents \
  --suite manifests/openclaw-agent-core.json \
  --model-config manifests/api-providers.example.json \
  --out /home/ubuntu/openclaw-bench/results \
  --run-id api-core

python3 -m openclaw_bench run \
  --backend openclaw \
  --openclaw-local \
  --openclaw-workspace-agents \
  --suite manifests/real-repo-readonly.example.json \
  --model-config manifests/api-providers.example.json \
  --out /home/ubuntu/openclaw-bench/results \
  --run-id api-real-repo
```

### 4. Certification Audit

Certification has two hard live prerequisites that simulator runs cannot satisfy:

- Local vLLM evidence must include real local `workspace_needle` coverage through `65536` context tokens for each required KV mode. An 8k-only endpoint is useful for smoke and harness validation, but it cannot certify the full local sweep.
- External-provider evidence must include both `api` and `subscription` rows. If provider credentials such as `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` are unset, preflight can validate the manifests but the result set cannot certify external coverage.

Several checked-in live manifests are intentionally host-specific examples for this workstation. They carry `manifest_scope.portability = "host_specific"` metadata so `preflight` can emit a warning while loaders ignore the note. Paths such as `/home/ubuntu/.venvs/vllm/bin/vllm`, GPU targeting such as `CUDA_VISIBLE_DEVICES=1`, port `8000`, and host-reachable endpoints such as `10.68.198.1:8000` should be treated as examples to copy and adapt for another machine, not portable defaults.

Run certification over the live local and external-provider result directories. The command requires live non-simulator attempts, local rows, API and subscription rows, local `fp8`, `turboquant_k8v4`, and `turboquant_k3v4_nc` KV modes, the full local 4k/8k/16k/32k/64k context sweep as passing `workspace_needle` rows for each required local KV mode, the full local 1/2/4/8/16/32/64 concurrency sweep as passing local rows for each required local KV mode, baseline/8k/32k external-provider context rows, 1/4/16 external-provider concurrency rows, representative patch/instruction passes for each local KV mode and concurrency level, passing FP8 baseline pairing for non-FP8 local rows, successful route probes, and passing coverage for all required task types on local, API, and subscription providers:

```bash
python3 -m openclaw_bench certify \
  /home/ubuntu/openclaw-bench/results/local-vllm-quality \
  /home/ubuntu/openclaw-bench/results/local-vllm-hardware-setups \
  /home/ubuntu/openclaw-bench/results/local-vllm-long-context \
  /home/ubuntu/openclaw-bench/results/local-vllm-concurrency \
  /home/ubuntu/openclaw-bench/results/local-vllm-real-repo \
  /home/ubuntu/openclaw-bench/results/api-core \
  /home/ubuntu/openclaw-bench/results/api-real-repo
```

Use `--failures-only` while iterating over incomplete evidence so stale artifact and coverage failures stay readable:

```bash
python3 -m openclaw_bench certify \
  /home/ubuntu/openclaw-bench/results/local-vllm-quality \
  /home/ubuntu/openclaw-bench/results/api-core \
  /home/ubuntu/openclaw-bench/results/api-real-repo \
  --failures-only
```

`certification=ok` means the result set covers the objective well enough to compare candidates. A failure means the output is not a certified comparison yet. Simulator rows are ignored for proof of task, provider, KV, context, concurrency, and pass coverage, even when they appear beside live rows. For local vLLM cells, certification also checks that any declared `--max-model-len` is at least the reported `context_limit`.

Certification binds evidence artifacts to attempt rows by `workspace_id`: every attempted row must have matching `raw/<workspace_id>.json` and `patches/<workspace_id>.diff` files, and the raw artifact must repeat the task id/type, workspace id, model cell metadata from `attempts.jsonl`, and backend-appropriate response provenance. Stale, renamed, swapped, empty, or simulator-labeled live artifacts fail certification even when artifact counts match. `config.json` must also include source input file digests with a root `suite` role, `suite_include` roles for included suites, a `model_config` role when a model manifest was used, the normalized model matrix digest, suite/task and fixture provenance digests written by the runner, plus runtime identity fields; live OpenClaw runs must include a passing OpenClaw CLI version probe.

For non-local OpenClaw runs using the default gateway auto-ensure behavior, `config.json` must include a passing `openclaw_gateway_ensure` result. Runs launched with `--no-ensure-openclaw-gateway` are marked with a certification warning so supervised gateway lifecycle remains explicit.

Certification also requires `server.json` evidence that supports hardware-aware comparison: a host GPU inventory, at least two local hardware/setup profiles represented in live local attempts and server model artifacts, at least one same model/weight/KV/context/concurrency cell passing on multiple hardware profiles for each required local KV mode, successful route probes for each passing model cell, throughput probe rows for each successful direct model route, and GPU telemetry on passing local task rows. Throughput probe rows must include at least three samples plus positive `prompt_chars`, `wall_time_s`, `completion_tokens`, `total_tokens`, `tokens_per_s`, `tokens_per_s_p50`, and `tokens_per_s_p95` values; a cell label with no real timing/token evidence is not enough. Model-cell evidence is matched by served model, provider type, hardware profile, weight quantization, KV mode, and context limit so a route probe from one local setup cannot certify a different quant/context setup.

Passing live rows must include positive `tool_calls` and `files_read` telemetry, non-negative `duplicate_file_reads`, and `time_to_first_relevant_file_s` telemetry. Certification also enforces broad default efficiency budgets of p95 `tool_calls <= 80`, p95 `files_read <= 80`, p95 `duplicate_file_reads <= 20`, and p95 `time_to_first_relevant_file_s <= 120` per provider/task-type group, so a model that eventually succeeds by looping through the workspace does not certify as comparable.

For local live models, the model config must include a real `health_check_url` for an already-running endpoint or a `serve_command` plus `health_check_url` so the harness can prove the target is reachable before scoring attempts. `support_status: "assumed_supported"` is only documentary; it is not accepted as live readiness proof.

For vLLM local serving, start from `manifests/vllm-local.example.json` for the focused GPT-OSS/Qwen KV comparison or `manifests/vllm-local-candidates.example.json` for the broader local candidate sweep from the installed vLLM model suite. These manifests use `vllm serve ... --kv-cache-dtype ...` on OpenClaw's default vLLM endpoint, `http://127.0.0.1:8000/v1`, and declare `api_env: "VLLM_API_KEY"` because OpenClaw's vLLM provider uses that env var for auth/discovery. Preflight checks that the configured `vllm` executable exists and fails OpenAI-compatible `/v1/models` health checks that omit `api_base`, because those cells cannot prove the chat-completions route. During `run`, the harness checks the health endpoint and sends a tiny OpenAI-compatible `/v1/chat/completions` request using `served_model_name`. OpenClaw CLI smoke and task runs use `openclaw_model_name` when configured, otherwise they fall back to `served_model_name`; use this when the local vLLM server exposes one name but OpenClaw expects a configured provider alias. A health pass without a routable model name is treated as `model_route_failed`; a routable model that fails the bounded serve probe is treated as `serve_probe_failed`. Tool-parser setup errors are treated as `tool_parser_missing`, and prompt/output budget errors are treated as `context_window_exceeded`. Successful probe details are written into `server.json` under `route_probe` with prompt size, wall time, token counts, and tokens/sec.

When `nvidia-smi` is available, the harness samples GPU telemetry during model startup and each model/KV/context/concurrency cell. Attempt rows and summaries include `peak_vram_mb` and GPU utilization so local quant/KV choices can be judged against both OpenClaw task quality and hardware pressure.
