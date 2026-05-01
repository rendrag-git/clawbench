# Current Status

Last updated: 2026-05-01 23:08 UTC

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
- Unit tests pass locally after the M1 scoring fix: `python3 -m unittest discover -s tests` ran `207` tests.
- Simulator full-suite regression passes after the M1 scoring fix:
  - `python3 -m openclaw_bench run --backend simulator --suite manifests/openclaw-certification-full.example.json --models simulated-model --kv fp8 --concurrency 1 --contexts 4096,8192,16384,32768,65536 --out /tmp/openclaw-bench-m1-trust-main2 --run-id cert-a`
  - `python3 -m openclaw_bench run --backend simulator --suite manifests/openclaw-certification-full.example.json --models simulated-model --kv fp8 --concurrency 1 --contexts 4096,8192,16384,32768,65536 --out /tmp/openclaw-bench-m1-trust-main2 --run-id cert-b`
  - Both runs produced `40` attempts, `0` failures, and identical stable score/status fields.
- Unit tests pass inside `oc-stack` from the staged repo snapshot `/tmp/openclaw-local-model-bench-m1-20260501223912`: `207` tests.
- M2 tier-manifest slice tests pass locally:
  - `python3 -m unittest discover -s tests` ran `211` tests.
  - `python3 -m openclaw_bench run --backend simulator --suite manifests/openclaw-certification-full.example.json --models simulated-model --kv fp8 --concurrency 1 --contexts 4096,8192,16384,32768,65536 --out /tmp/openclaw-bench-m2-tier-slice-verify --run-id cert-full` produced `40` attempts, `0` failures.
  - Targeted simulator smoke tests pass for `manifests/tier-small.json` and `manifests/tier-medium.json`.
- M2 tool-error recovery slice tests pass locally:
  - `python3 -m unittest discover -s tests` ran `211` tests.
  - `python3 -m openclaw_bench run --backend simulator --suite manifests/openclaw-certification-full.example.json --models simulated-model --kv fp8 --concurrency 1 --contexts 4096,8192,16384,32768,65536 --out /tmp/openclaw-bench-m2-tool-recovery-verify --run-id cert-full` produced `40` attempts, `0` failures.

## Latest E2E

Latest staged repo:

```text
/tmp/openclaw-local-model-bench-m1-20260501223912
```

Latest result directory:

```text
/tmp/oc-bench-root-m1-20260501223912/results/live-m1-qwen35-rerun-20260501225000
```

Result summary:

- Live anchor rerun complete.
- Log: `/tmp/live-m1-qwen35-rerun-20260501225000.log`
- Gateway startup: pass.
- vLLM health: pass.
- vLLM direct route probe: pass.
- OpenClaw route smoke: pass.
- Benchmark attempts: `1`
- Benchmark failures: `0`
- Failure type: none
- Score: `1.0`
- Pass rate: `100%`
- Wall time: `200.328s`
- Tool calls: `10`
- OpenClaw version for the staged run: `OpenClaw 2026.4.27 (cbc2ba0)`.
- Live anchor record: run id `live-m1-qwen35-rerun-20260501225000`, code commit `a9fd98b`, model `qwen3.5-4b`, KV mode `provider_default`, context `32768`, concurrency `1`, date `2026-05-01`.
- The model returned a runnable equivalent command:
  - `python tests/test_api.py`
- Prior post-fix live run: `/tmp/oc-bench-root-m1-20260501223143/results/live-m1-qwen35-20260501223143`
  - Preflight: pass
  - vLLM health: pass
  - vLLM direct route probe: pass
  - OpenClaw route smoke: pass
  - Benchmark attempts: `1`
  - Benchmark failures: `1`
  - Failure type: `wrong_file`
  - Score: `0.8333`
  - Wall time: `164.179s`
  - Tool calls: `8`
  - Diagnosis: scorer still rejected runnable unittest module command `python -m unittest tests.test_api`; this has been fixed and covered by `test_discovery_accepts_unittest_module_command`.

## Resume Point

M1 trust hygiene is complete. Active milestone: M2 tiered, discriminating task suite.

First M2 slice in progress:

- Added explicit additive tier manifests:
  - `manifests/tier-small.json`
  - `manifests/tier-medium.json`
- Added manifest and simulator smoke coverage for those suites.
- This is not full M2 completion. Missing M2 work remains:
  - `tier-xlarge.json`
  - floor/ceiling calibration records for every tier
  - task-gap coverage for destructive-action refusal, plan/action coherence, AGENTS/SOUL adherence, format drift after 10+ tool calls, and ambiguous-spec triage
  - per-task tool-loop / stop-condition scoring
- Added first task-gap slice:
  - `fixtures/tool_error_recovery_repo`
  - `medium-tool-error-recovery-route-map` in `manifests/tier-medium.json`
  - Targeted manifest + simulator tests pass.
- Added cross-file consistency slice:
  - `fixtures/cross_file_consistency_repo`
  - `manifests/tier-large.json`
  - `large-cross-file-sale-rate`
  - Targeted scorer + manifest + simulator tests pass.

The abandoned detached quickstart rerun `live-m1-qwen35-20260501223912` stuck during gateway probing before any attempt. Its benchmark-owned temp processes were stopped; it is not the active run.

Inspect the latest live anchor with:

```bash
incus exec oc-stack -- bash -lc "tail -n 120 /tmp/live-m1-qwen35-rerun-20260501225000.log"
incus exec oc-stack -- bash -lc "jq . /tmp/oc-bench-root-m1-20260501223912/results/live-m1-qwen35-rerun-20260501225000/summary.json"
incus exec oc-stack -- bash -lc "cat /tmp/oc-bench-root-m1-20260501223912/results/live-m1-qwen35-rerun-20260501225000/attempts.jsonl"
```

## Open Items

- Add the next M2 task-gap slice. Recommended next: calibration record schema planning before floor/ceiling runs, or AGENTS/SOUL adherence.
- The two-attempt cap was reached for the `workspace_discovery` command scorer in the M1 iteration; do not make another scoring change in that branch without a fresh diagnosis and explicit pivot.
