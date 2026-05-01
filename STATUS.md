# Current Status

Last updated: 2026-05-01 21:45 UTC

## Runtime

- OpenClaw runtime under test: `oc-stack`
- OpenClaw version: `2026.4.27`
- `2026.4.29` is blocked for this harness until the observed regressions are resolved.
- Benchmark vLLM service: `openclaw-vllm-small-bench.service`
- vLLM endpoint: `http://10.68.198.1:8003/v1`
- Served model: `qwen3.5-4b`
- Hugging Face model: `Qwen/Qwen3.5-4B`
- Weight format loaded by vLLM: BF16, not quantized
- Model checkpoint size reported by vLLM: `8.68 GiB`
- GPU model load reported by vLLM: `7.99 GiB`
- Context target: `32768`

The service runs on GPU 0, the 16GB A4000, and leaves GPU 1 alone. It uses `--enforce-eager`; without eager mode, vLLM OOMed during CUDA graph / KV-cache profiling at 32k context.

## Product Goal

The intended user workflow is not "let the benchmark own vLLM." The benchmark should work with any local model provider that OpenClaw can route to, including vLLM, llama.cpp, Ollama, and Apple Silicon local runtimes.

The target workflow is:

1. A user already has a local or LAN-reachable model server running, or has a local runtime installed that can be started outside the benchmark.
2. `oc-bench init --providers local` creates an isolated OpenClaw benchmark profile.
3. The setup flow inspects the machine first: common ports, process names, OpenAI-compatible `/v1/models` endpoints, Ollama endpoints, llama.cpp endpoints, and relevant local runtime hints.
4. If inspection finds one clear provider/model, the generated OpenClaw config points a local route at that existing API.
5. If inspection finds none or multiple ambiguous candidates, it asks the user what is running and which provider/model to benchmark.
6. The benchmark starts/checks only the isolated OpenClaw gateway/profile.
7. The benchmark creates per-attempt OpenClaw workspace agents so model routing is bound in config, not passed as a per-call override.
8. The benchmark runs real OpenClaw agent turns against seeded workspace copies and records whether the model can use tools and return scoreable task output.

For this flow, the model runtime is an external dependency. The quickstart-generated model manifest intentionally does not include a `serve_command` by default; it should not start, stop, restart, or containerize a user's existing model server. Containerization is still useful for isolating OpenClaw itself, but it should not be required for the model runtime.

The current `openclaw-vllm-small-bench.service` is just a repo-owned test fixture for this machine: a persistent host vLLM API that simulates what a normal user would already have running.

## Provider Scope

The local-provider setup should be provider-agnostic. Initial support should cover at least:

- vLLM via OpenAI-compatible `/v1` endpoints.
- llama.cpp server via OpenAI-compatible `/v1` endpoints.
- Ollama via detected local Ollama service and any available OpenAI-compatible route.
- Apple Silicon local runtimes, where the benchmark should still use the same inspect-first flow and generate an OpenClaw route for the discovered local provider.

The UX principle is inspect first, ask second. The tool should not force a user to know the exact config shape if the machine already exposes enough information to infer it.

## Test Suite Goal

The repo needs a full local-provider setup test suite before this is considered done. It should cover:

- Detection with one clear vLLM endpoint.
- Detection with one clear llama.cpp endpoint.
- Detection with one clear Ollama service.
- Detection on Apple Silicon-style local provider inputs without assuming NVIDIA or Linux-only hardware.
- Ambiguous detection where multiple providers are running and the CLI asks the user to choose.
- Empty detection where no provider is found and the CLI asks the user what is running.
- Generated OpenClaw config for each provider family.
- Generated model manifest for each provider family.
- No `serve_command` by default for discovered external providers.
- Route smoke behavior against a fake OpenAI-compatible server.
- Provider-specific parameter shaping, such as disabling Qwen thinking only when it applies.
- Resume behavior from `oc-bench run` without redoing init/preflight when the prior setup is already valid.

## Verified

- Host and `oc-stack` can reach `/v1/models`.
- Direct OpenAI-compatible chat completions work.
- Direct OpenAI-compatible tool calling works.
- OpenClaw route smoke works after disabling Qwen thinking in generated local vLLM config:
  - provider model: `reasoning=false`
  - agent params: `chatTemplateKwargs.enable_thinking=false`
- Unit tests pass locally: `python3 -m unittest discover -s tests` ran `202` tests.
- Unit tests pass inside `oc-stack` from the staged repo snapshot: `202` tests.

## Latest E2E

Latest staged repo:

```text
/tmp/openclaw-local-model-bench-qwen35-20260501212915
```

Latest result directory:

```text
/tmp/oc-bench-root-qwen35-20260501212915/results/e2e-qwen35-20260501212915
```

Result summary:

- Preflight: pass
- vLLM health: pass
- vLLM direct route probe: pass
- OpenClaw route smoke: pass
- Benchmark attempts: `1`
- Benchmark failures: `1`
- Failure type: `wrong_file`
- Score: `0.7143`
- Pass rate: `0%`
- Wall time: `200.306s`
- Tool calls: `10`

The model used OpenClaw successfully and found:

- `api/routes.py`
- `db/schema.py`

It failed because it returned the test command as `tests/test_api.py` instead of the expected runnable command.

## Resume Point

The init/preflight/service setup is no longer the blocker. Resume from `oc-bench run` against the existing suite/model config, with a new run id, instead of rerunning full quickstart unless profile/config generation is being tested.

Use the staged repo path above or restage the current tree into `oc-stack`, then run from the staged repo with:

```bash
HOME=/tmp/oc-bench-home-qwen35-20260501212915 \
python3 -m openclaw_bench run \
  --suite /tmp/oc-bench-root-qwen35-20260501212915/manifests/starter-suite.json \
  --model-config /tmp/oc-bench-root-qwen35-20260501212915/manifests/starter-models.json \
  --out /tmp/oc-bench-root-qwen35-20260501212915/results \
  --workspace-root /tmp/oc-bench-root-qwen35-20260501212915/workspaces/quickstart \
  --fixtures-root /tmp/oc-bench-root-qwen35-20260501212915/fixtures \
  --backend openclaw \
  --openclaw-profile benchclaw-e2e-qwen35-20260501212915 \
  --openclaw-agent bench \
  --openclaw-workspace-agents \
  --ensure-openclaw-gateway \
  --openclaw-gateway-timeout 120 \
  --openclaw-smoke-timeout 120 \
  --timeout 600 \
  --run-id e2e-qwen35-resume-$(date -u +%Y%m%d%H%M%S)
```

## Open Items

- Decide whether the discovery task should accept `tests/test_api.py` as a partial command answer or require the exact runnable command.
- If the exact runnable command is required, improve prompt/task scoring pressure rather than changing the model route again.
- Consider trying a quantized 4B or a larger better-instructed model; the current Qwen3.5-4B is BF16 and functional but did not pass the first benchmark task.
