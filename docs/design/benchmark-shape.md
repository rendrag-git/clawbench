# Benchmark Shape

The durable design of the OpenClaw local-model benchmark: what gets measured, how attempts are scored, and how results are reported. For *what we're trying to accomplish* see [GOAL.md](../../GOAL.md). For *current operational state* see [STATUS.md](../../STATUS.md).

## Two layers

Run two layers:

1. **Serve Layer** (`openclaw_bench/serve.py`)
   - Treats each `(model, KV mode)` as one route to a model server. The bench attaches to whatever the user is already running by default; if the manifest provides a `serve_command`, the bench can start the process itself.
   - Works against any OpenAI-compatible runtime: vLLM, llama.cpp `llama-server`, Ollama (`/api/tags` + OpenAI bridge), LM Studio, hosted OpenAI/Anthropic.
   - Process topology depends on the runtime, not the bench:
     - **vLLM and llama.cpp** — one process per `(model, KV mode)`. KV cache dtype is fixed at startup.
     - **Ollama and LM Studio** — one process hosts many models; the bench discriminates by `served_model_name`. KV mode is not user-selectable in these runtimes.
   - Records load success, load time, VRAM use, GPU utilization, max context, OOMs, request errors, and route-probe metrics (TTFT, P50/P95/P99 latency).

2. **OpenClaw Layer** (`openclaw_bench/backend.py`, `openclaw_bench/runner.py`)
   - Runs real `openclaw agent --json` turns against isolated benchmark workspaces.
   - Uses fixed task manifests and deterministic scoring.
   - Measures task success, file-use correctness, patch correctness, latency, tool count, and failure modes.

## Isolation

Use a dedicated profile and per-run workspace copies:

```text
<bench-root>/
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

The harness is not pinned to a specific OpenClaw version. It runs against whatever version is installed (host, container, or runtime target). Check your version with:

```bash
openclaw --version
```

For isolated Docker runs, pass `--openclaw-container oc-bench-gateway`. On first use, `oc-bench` creates or starts that container with image `clawdaddy/openclaw:business-smoke-latest`, host networking, a separate home at `<bench-root>/container-home`, and exact-path mounts for the repo, benchmark root, and any custom workspace root. The container stays as a reusable OpenClaw runtime; the harness still starts/checks the selected `bench` gateway through `openclaw --profile bench gateway status` and a detached verbose foreground gateway when needed. This path is intentionally separate from any host or LXC OpenClaw install. Use `--no-ensure-openclaw-container` only when another supervisor already owns the container.

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
