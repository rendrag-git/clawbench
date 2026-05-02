# Certification

Certification has two hard live prerequisites that simulator runs cannot satisfy:

- Local vLLM evidence must include real local `workspace_needle` coverage through `65536` context tokens for each required KV mode. An 8k-only endpoint is useful for smoke and harness validation, but it cannot certify the full local sweep.
- External-provider evidence must include both `api` and `subscription` rows. If provider credentials such as `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` are unset, preflight can validate the manifests but the result set cannot certify external coverage.

`<bench-root>` is the benchmark root from `oc-bench init`.

## Host-specific manifests

Several checked-in live manifests are intentionally host-specific examples for the author's workstation. They carry `manifest_scope.portability = "host_specific"` metadata so `preflight` can emit a warning while loaders ignore the note. Paths such as `/home/ubuntu/.venvs/vllm/bin/vllm`, GPU targeting such as `CUDA_VISIBLE_DEVICES=1`, port `8000`, and host-reachable endpoints such as `10.68.198.1:8000` should be treated as examples to copy and adapt for another machine, not portable defaults.

## Run certification

Run certification over the live local and external-provider result directories. The command requires live non-simulator attempts, local rows, API and subscription rows, local `fp8`, `turboquant_k8v4`, and `turboquant_k3v4_nc` KV modes, the full local 4k/8k/16k/32k/64k context sweep as passing `workspace_needle` rows for each required local KV mode, the full local 1/2/4/8/16/32/64 concurrency sweep as passing local rows for each required local KV mode, baseline/8k/32k external-provider context rows, 1/4/16 external-provider concurrency rows, representative patch/instruction passes for each local KV mode and concurrency level, passing FP8 baseline pairing for non-FP8 local rows, successful route probes, and passing coverage for all required task types on local, API, and subscription providers:

```bash
python3 -m openclaw_bench certify \
  <bench-root>/results/local-vllm-quality \
  <bench-root>/results/local-vllm-hardware-setups \
  <bench-root>/results/local-vllm-long-context \
  <bench-root>/results/local-vllm-concurrency \
  <bench-root>/results/local-vllm-real-repo \
  <bench-root>/results/api-core \
  <bench-root>/results/api-real-repo
```

Use `--failures-only` while iterating over incomplete evidence so stale artifact and coverage failures stay readable:

```bash
python3 -m openclaw_bench certify \
  <bench-root>/results/local-vllm-quality \
  <bench-root>/results/api-core \
  <bench-root>/results/api-real-repo \
  --failures-only
```

`certification=ok` means the result set covers the objective well enough to compare candidates. A failure means the output is not a certified comparison yet.

## What certification checks

Simulator rows are ignored for proof of task, provider, KV, context, concurrency, and pass coverage, even when they appear beside live rows. For local vLLM cells, certification also checks that any declared `--max-model-len` is at least the reported `context_limit`.

Certification binds evidence artifacts to attempt rows by `workspace_id`: every attempted row must have matching `raw/<workspace_id>.json` and `patches/<workspace_id>.diff` files, and the raw artifact must repeat the task id/type, workspace id, model cell metadata from `attempts.jsonl`, and backend-appropriate response provenance. Stale, renamed, swapped, empty, or simulator-labeled live artifacts fail certification even when artifact counts match.

`config.json` must also include source input file digests with a root `suite` role, `suite_include` roles for included suites, a `model_config` role when a model manifest was used, the normalized model matrix digest, suite/task and fixture provenance digests written by the runner, plus runtime identity fields; live OpenClaw runs must include a passing OpenClaw CLI version probe.

For non-local OpenClaw runs using the default gateway auto-ensure behavior, `config.json` must include a passing `openclaw_gateway_ensure` result. Runs launched with `--no-ensure-openclaw-gateway` are marked with a certification warning so supervised gateway lifecycle remains explicit.

Certification also requires `server.json` evidence that supports hardware-aware comparison: a host GPU inventory, at least two local hardware/setup profiles represented in live local attempts and server model artifacts, at least one same model/weight/KV/context/concurrency cell passing on multiple hardware profiles for each required local KV mode, successful route probes for each passing model cell, throughput probe rows for each successful direct model route, and GPU telemetry on passing local task rows. Throughput probe rows must include at least three samples plus positive `prompt_chars`, `wall_time_s`, `completion_tokens`, `total_tokens`, `tokens_per_s`, `tokens_per_s_p50`, and `tokens_per_s_p95` values; a cell label with no real timing/token evidence is not enough. Model-cell evidence is matched by served model, provider type, hardware profile, weight quantization, KV mode, and context limit so a route probe from one local setup cannot certify a different quant/context setup.

Passing live rows must include positive `tool_calls` and `files_read` telemetry, non-negative `duplicate_file_reads`, and `time_to_first_relevant_file_s` telemetry. Certification also enforces broad default efficiency budgets of p95 `tool_calls <= 80`, p95 `files_read <= 80`, p95 `duplicate_file_reads <= 20`, and p95 `time_to_first_relevant_file_s <= 120` per provider/task-type group, so a model that eventually succeeds by looping through the workspace does not certify as comparable.

## Local vLLM readiness

For local live models, the model config must include a real `health_check_url` for an already-running endpoint or a `serve_command` plus `health_check_url` so the harness can prove the target is reachable before scoring attempts. `support_status: "assumed_supported"` is only documentary; it is not accepted as live readiness proof.

For vLLM local serving, start from `manifests/vllm-local.example.json` for the focused GPT-OSS/Qwen KV comparison or `manifests/vllm-local-candidates.example.json` for the broader local candidate sweep from the installed vLLM model suite. These manifests use `vllm serve ... --kv-cache-dtype ...` on OpenClaw's default vLLM endpoint, `http://127.0.0.1:8000/v1`, and declare `api_env: "VLLM_API_KEY"` because OpenClaw's vLLM provider uses that env var for auth/discovery. Preflight checks that the configured `vllm` executable exists and fails OpenAI-compatible `/v1/models` health checks that omit `api_base`, because those cells cannot prove the chat-completions route. During `run`, the harness checks the health endpoint and sends a tiny OpenAI-compatible `/v1/chat/completions` request using `served_model_name`. OpenClaw CLI smoke and task runs use `openclaw_model_name` when configured, otherwise they fall back to `served_model_name`; use this when the local vLLM server exposes one name but OpenClaw expects a configured provider alias. A health pass without a routable model name is treated as `model_route_failed`; a routable model that fails the bounded serve probe is treated as `serve_probe_failed`. Tool-parser setup errors are treated as `tool_parser_missing`, and prompt/output budget errors are treated as `context_window_exceeded`. Successful probe details are written into `server.json` under `route_probe` with prompt size, wall time, token counts, and tokens/sec.

When `nvidia-smi` is available, the harness samples GPU telemetry during model startup and each model/KV/context/concurrency cell. Attempt rows and summaries include `peak_vram_mb` and GPU utilization so local quant/KV choices can be judged against both OpenClaw task quality and hardware pressure.
