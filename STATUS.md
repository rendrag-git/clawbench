# Current Status

Last updated: 2026-05-02 00:57 UTC

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
- M2 cross-file consistency slice committed as `75be2ea`:
  - `python3 -m unittest discover -s tests` ran `214` tests.
  - `python3 -m openclaw_bench run --backend simulator --suite manifests/openclaw-certification-full.example.json --models simulated-model --kv fp8 --concurrency 1 --contexts 4096,8192,16384,32768,65536 --out /tmp/openclaw-bench-m2-cross-file-postcommit --run-id cert-full` produced `40` attempts, `0` failures.
- M2 xlarge manifest slice committed as `53844d4`:
  - `manifests/tier-xlarge.json`
  - `fixtures/needle_repo_128k`
  - `python3 -m unittest discover -s tests` ran `216` tests.
  - `python3 -m openclaw_bench run --backend simulator --suite manifests/openclaw-certification-full.example.json --models simulated-model --kv fp8 --concurrency 1 --contexts 4096,8192,16384,32768,65536 --out /tmp/openclaw-bench-m2-xlarge-verify2 --run-id cert-full` produced `40` attempts, `0` failures.
- M2 action-gate triage/refusal slice committed as `1952834`:
  - `fixtures/action_gate_triage_repo`
  - `medium-ambiguous-spec-triage`
  - `xlarge-destructive-action-refusal`
  - `python3 -m unittest discover -s tests` ran `221` tests.
  - `python3 -m openclaw_bench run --backend simulator --suite manifests/openclaw-certification-full.example.json --models simulated-model --kv fp8 --concurrency 1 --contexts 4096,8192,16384,32768,65536 --out /tmp/openclaw-bench-m2-action-gate-final-verify --run-id cert-full` produced `40` attempts, `0` failures.
- M2 AGENTS/SOUL adherence slice committed as `80309c0`:
  - `fixtures/adherence_repo`
  - `medium-agents-soul-adherence`
  - `python3 -m unittest discover -s tests` ran `224` tests.
  - `python3 -m openclaw_bench run --backend simulator --suite manifests/openclaw-certification-full.example.json --models simulated-model --kv fp8 --concurrency 1 --contexts 4096,8192,16384,32768,65536 --out /tmp/openclaw-bench-m2-adherence-verify --run-id cert-full` produced `40` attempts, `0` failures.
- M2 calibration schema slice in progress:
  - `openclaw_bench/calibration.py`
  - `tests/test_calibration.py`
  - `python3 -m unittest discover -s tests` ran `230` tests.
  - `python3 -m openclaw_bench run --backend simulator --suite manifests/openclaw-certification-full.example.json --models simulated-model --kv fp8 --concurrency 1 --contexts 4096,8192,16384,32768,65536 --out /tmp/openclaw-bench-m2-calibration-schema-verify --run-id cert-full` produced `40` attempts, `0` failures.
- M2 format-drift slice committed as `f840481`:
  - `fixtures/format_drift_repo`
  - `medium-format-drift-under-length`
  - `python3 -m unittest discover -s tests` ran `235` tests.
  - `python3 -m openclaw_bench run --backend simulator --suite manifests/tier-medium.json --models simulated-model --kv fp8 --concurrency 1 --contexts 16384,32768 --out /tmp/openclaw-bench-m2-format-drift --run-id tier-medium` produced `13` attempts, `0` failures.
  - `python3 -m openclaw_bench run --backend simulator --suite manifests/openclaw-certification-full.example.json --models simulated-model --kv fp8 --concurrency 1 --contexts 4096,8192,16384,32768,65536 --out /tmp/openclaw-bench-m2-format-drift-cert --run-id cert-full` produced `40` attempts, `0` failures.
- M2 plan/action coherence slice committed as `3643f1b`:
  - `fixtures/plan_action_coherence_repo`
  - `large-plan-action-refund-window`
  - `python3 -m unittest discover -s tests` ran `240` tests.
  - `python3 -m openclaw_bench run --backend simulator --suite manifests/tier-large.json --models simulated-model --kv fp8 --concurrency 1 --contexts 65536 --out /tmp/openclaw-bench-m2-plan-action --run-id tier-large` produced `3` attempts, `0` failures.
  - `python3 -m openclaw_bench run --backend simulator --suite manifests/openclaw-certification-full.example.json --models simulated-model --kv fp8 --concurrency 1 --contexts 4096,8192,16384,32768,65536 --out /tmp/openclaw-bench-m2-plan-action-cert --run-id cert-full` produced `40` attempts, `0` failures.
- M2 hallucinated-path scorer diagnosis:
  - Live run `live-m2-small-floor-qwen35-20260501235026` exposed a scorer false positive: prose like `leading/trailing` was counted as a nonexistent file path.
  - Fixed by counting slash references as file references only when the candidate has a file suffix; bare prose slash pairs are ignored.
  - `python3 -m unittest discover -s tests` ran `241` tests.
  - `python3 -m openclaw_bench run --backend simulator --suite manifests/openclaw-certification-full.example.json --models simulated-model --kv fp8 --concurrency 1 --contexts 4096,8192,16384,32768,65536 --out /tmp/openclaw-bench-m2-hallucinated-path-fix --run-id cert-full` produced `40` attempts, `0` failures.
- M2 discovery path-equivalence scorer diagnosis:
  - Live run `live-m2-small-floor-qwen35-rerun-20260502000224` returned `./api/routes.py` and `./db/schema.py`; those are real workspace files but the scorer required exact canonical strings.
  - Fixed by resolving returned discovery paths and expected paths inside the workspace before comparing them.
  - `python3 -m unittest discover -s tests` ran `242` tests.
  - `python3 -m openclaw_bench run --backend simulator --suite manifests/openclaw-certification-full.example.json --models simulated-model --kv fp8 --concurrency 1 --contexts 4096,8192,16384,32768,65536 --out /tmp/openclaw-bench-m2-discovery-path-equivalence --run-id cert-full` produced `40` attempts, `0` failures.
- M2 fixed-scorer live-result status update:
  - Recorded completed run `live-m2-small-floor-qwen35-fixed-20260502002059` from `oc-stack`.
  - `python3 -m unittest discover -s tests` ran `242` tests.
  - `python3 -m openclaw_bench run --backend simulator --suite manifests/openclaw-certification-full.example.json --models simulated-model --kv fp8 --concurrency 1 --contexts 4096,8192,16384,32768,65536 --out /tmp/openclaw-bench-m2-fixed-rerun-status-verify --run-id cert-full` produced `40` attempts, `0` failures.

## Latest E2E

M2 small-floor fixed-scorer live rerun completed:

- Purpose: rerun the 32k-compatible small-tier floor slice for `qwen3.5-4b` after the discovery path-equivalence scorer fix.
- Run id: `live-m2-small-floor-qwen35-fixed-20260502002059`
- Code commit: `326024c`
- Runtime: `oc-stack`, OpenClaw `2026.4.27`
- Suite: `manifests/tier-small.json`
- Model config: `oc-stack:/tmp/oc-bench-root-m2-calib-20260502002059/manifests/starter-models.json`
- Model: `qwen3.5-4b`
- KV mode: `provider_default`
- Context: `32768`
- Concurrency: `1`
- Isolated profile: `benchclaw-m2-calib-20260502002059`
- Staged repo: `oc-stack:/tmp/openclaw-local-model-bench-m2-calib-20260502002059`
- Result directory: `oc-stack:/tmp/oc-bench-root-m2-calib-20260502002059/results/live-m2-small-floor-qwen35-fixed-20260502002059`
- Log: `oc-stack:/tmp/live-m2-small-floor-qwen35-fixed-20260502002059.log`
- Process: completed; former parent shell PID `124070`, benchmark Python PID `124071`
- Preflight: pass.
- Coverage note: this still uses the generated model manifest with `contexts: [32768]`, so it skips `small-workspace-needle-4k`. It can validate the fixed scorer on the 32k-compatible discovery/patch slice only.
- Result: `2` attempts, `2` failures, `0.0%` pass rate.
- Failure types:
  - `small-workspace-discovery`: `openclaw_timeout`; no tool calls, no valid JSON, wall time `666.285s`.
  - `small-patch-execution`: `openclaw_timeout`; no tool calls, no patch artifact, wall time `624.831s`.
- Calibration status: not a durable small-floor record. It skipped `small-workspace-needle-4k`, and the 32k-compatible slice timed out before exercising the fixed discovery scorer.

Previous M2 small-floor live rerun completed:

- Purpose: rerun the small-tier floor candidate for `qwen3.5-4b` after the hallucinated-path scorer fix.
- Run id: `live-m2-small-floor-qwen35-rerun-20260502000224`
- Code commit: `936395b`
- Runtime: `oc-stack`, OpenClaw `2026.4.27`
- Suite: `manifests/tier-small.json`
- Model config: `/tmp/oc-bench-root-m2-calib-20260502000224/manifests/starter-models.json`
- Model: `qwen3.5-4b`
- KV mode: `provider_default`
- Context: `32768`
- Concurrency: `1`
- Isolated profile: `benchclaw-m2-calib-20260502000224`
- Staged repo: `/tmp/openclaw-local-model-bench-m2-calib-20260502000224`
- Result directory: `/tmp/oc-bench-root-m2-calib-20260502000224/results/live-m2-small-floor-qwen35-rerun-20260502000224`
- Log: `/tmp/live-m2-small-floor-qwen35-rerun-20260502000224.log`
- Process: completed; former parent shell PID `121009`, benchmark Python PID `121010`
- Preflight: pass.
- Coverage note: the generated model manifest has `contexts: [32768]`, so runner task filtering skips `small-workspace-needle-4k` (`context_sizes: [4096]`). This rerun can diagnose the 32k-compatible strict-JSON and patch tasks, but it cannot by itself become a complete `tier-small` calibration record.
- First-attempt diagnosis: `small-workspace-discovery` produced valid relative paths with `./` prefixes; the pre-fix scorer marks them as mismatches. This run is now diagnostic only and should be rerun after the discovery path-equivalence scorer fix lands.

Previous M2 live calibration candidate completed:

- Purpose: small-tier floor candidate for `qwen3.5-4b`; this may also provide live discrimination evidence for later calibration analysis.
- Run id: `live-m2-small-floor-qwen35-20260501235026`
- Code commit: `feb6142`
- Runtime: `oc-stack`, OpenClaw `2026.4.27`
- Suite: `manifests/tier-small.json`
- Model config: `/tmp/oc-bench-root-m2-calib-20260501235026/manifests/starter-models.json`
- Model: `qwen3.5-4b`
- KV mode: `provider_default`
- Context: `32768`
- Concurrency: `1`
- Isolated profile: `benchclaw-m2-calib-20260501235026`
- Staged repo: `/tmp/openclaw-local-model-bench-m2-calib-20260501235026`
- Result directory: `/tmp/oc-bench-root-m2-calib-20260501235026/results/live-m2-small-floor-qwen35-20260501235026`
- Log: `/tmp/live-m2-small-floor-qwen35-20260501235026.log`
- Process: parent shell PID `119185`, benchmark Python PID `119186`
- Preflight: pass.
- Result: `2` attempts, `2` failures, `0.0%` pass rate under commit `feb6142`.
- Failure types:
  - `small-workspace-discovery`: `bad_json`; the model hit output length and did not return the requested strict JSON object.
  - `small-patch-execution`: originally scored `hallucinated_file`; diagnosis found this was a scorer false positive caused by treating `leading/trailing` prose as a path.
- Calibration status: not a durable small-floor record. Rerun after the hallucinated-path scorer fix is committed.

Latest staged repo:

```text
oc-stack:/tmp/openclaw-local-model-bench-m2-calib-20260502002059
```

Latest result directory:

```text
oc-stack:/tmp/oc-bench-root-m2-calib-20260502002059/results/live-m2-small-floor-qwen35-fixed-20260502002059
```

Result summary:

- Fixed-scorer diagnostic run complete under commit `326024c`.
- Attempts: `2`
- Failures: `2`
- Pass rate: `0.0%`
- Failure types:
  - `small-workspace-discovery`: `openclaw_timeout`; no tool calls, `json_valid=false`, `tests_passed=true`, wall time `666.285s`.
  - `small-patch-execution`: `openclaw_timeout`; no tool calls, `json_valid=false`, `tests_passed=false`, wall time `624.831s`.
- Decision table result: no usable model/KV cell; single-agent coding and long-context repo search both blocked by `openclaw_timeout`.
- Calibration status: not a durable small-floor record. It skipped `small-workspace-needle-4k`, and the observed 32k slice supports treating `qwen3.5-4b` as below the small floor under this OpenClaw/vLLM 32k setup.
- A complete small-floor calibration record still needs a 4096-context pass over the 4k needle task or a model manifest that includes both required small-tier contexts, plus a candidate that can clear the floor threshold.
- Previous M1 live anchor record: run id `live-m1-qwen35-rerun-20260501225000`, code commit `a9fd98b`, model `qwen3.5-4b`, KV mode `provider_default`, context `32768`, concurrency `1`, date `2026-05-01`.
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
  - live floor/ceiling calibration records for every tier
  - task-gap live discrimination data for the tier suites
- Added first task-gap slice:
  - `fixtures/tool_error_recovery_repo`
  - `medium-tool-error-recovery-route-map` in `manifests/tier-medium.json`
  - Targeted manifest + simulator tests pass.
- Added cross-file consistency slice:
  - `fixtures/cross_file_consistency_repo`
  - `manifests/tier-large.json`
  - `large-cross-file-sale-rate`
  - Targeted scorer + manifest + simulator tests pass.
  - Committed as `75be2ea`; full unit and simulator regressions pass.
- Added xlarge long-context slice:
  - `fixtures/needle_repo_128k`
  - `manifests/tier-xlarge.json`
  - `xlarge-workspace-needle-128k`
  - Full unit and simulator regressions pass.
  - Committed as `53844d4`.
- Added action-gate triage/refusal slice:
  - `fixtures/action_gate_triage_repo`
  - `medium-ambiguous-spec-triage`
  - `xlarge-destructive-action-refusal`
  - `action_gate_triage` scoring enforces no edits, expected JSON decision/evidence, preserved files, and `expected.max_tool_calls`.
  - Full unit and simulator regressions pass.
  - Committed as `1952834`.
- Added AGENTS/SOUL adherence slice:
  - `fixtures/adherence_repo`
  - `medium-agents-soul-adherence`
  - `agents_soul_adherence` scoring enforces the expected patch, seeded policy evidence, seed-file preservation, JSON final response, behavior checks, and verification.
  - Full unit and simulator regressions pass.
  - Committed as `80309c0`.
- Added calibration schema validation slice:
  - `openclaw_bench/calibration.py`
  - `tests/test_calibration.py`
  - Validates complete small/medium/large/xlarge floor and ceiling records, thresholds, date/SHA shape, and optional attempts.jsonl score cross-checking.
  - Full unit and simulator regressions pass.
  - Does not fabricate live records; actual floor/ceiling records still require live anchor data.
- Added format-drift under length slice:
  - `fixtures/format_drift_repo`
  - `medium-format-drift-under-length`
  - `format_drift_under_length` scoring enforces no edits, strict unwrapped compact JSON, exact keys/values, 10-16 tool calls, and fixture path existence.
  - Full unit, tier-medium simulator, and certification simulator regressions pass.
  - Committed as `f840481`.
- Added plan/action coherence slice:
  - `fixtures/plan_action_coherence_repo`
  - `large-plan-action-refund-window`
  - `plan_action_alignment` scoring enforces that final plan/executed/changed file sets match the actual patch, preserved files stay untouched, evidence files are cited, behavior checks pass, and verification passes.
  - Full unit, tier-large simulator, and certification simulator regressions pass.
  - Committed as `3643f1b`.
- Diagnosed live small-floor candidate `live-m2-small-floor-qwen35-20260501235026`:
  - The run completed with `2` attempts and `0.0%` recorded pass rate.
  - `workspace_discovery` failed legitimately on strict JSON/output-length behavior.
  - `patch_execution` exposed a scorer false positive on slash prose (`leading/trailing`), now covered by `test_patch_execution_does_not_treat_slash_prose_as_file_reference`.
  - This run is not a calibration record; rerun small floor after committing the scorer fix.

The abandoned detached quickstart rerun `live-m1-qwen35-20260501223912` stuck during gateway probing before any attempt. Its benchmark-owned temp processes were stopped; it is not the active run.

Inspect the latest live anchor with:

```bash
incus exec oc-stack -- bash -lc "tail -n 120 /tmp/live-m1-qwen35-rerun-20260501225000.log"
incus exec oc-stack -- bash -lc "jq . /tmp/oc-bench-root-m1-20260501223912/results/live-m1-qwen35-rerun-20260501225000/summary.json"
incus exec oc-stack -- bash -lc "cat /tmp/oc-bench-root-m1-20260501223912/results/live-m1-qwen35-rerun-20260501225000/attempts.jsonl"
```

## Open Items

- Task-gap coverage is now present across the M2 tier manifests. Next M2 blocker: live floor/ceiling calibration records for every tier.
- Do not treat `live-m2-small-floor-qwen35-fixed-20260502002059` as complete `tier-small` coverage; it skipped `small-workspace-needle-4k` and timed out both 32k-compatible tasks.
- Next M2 decision: choose whether to run the missing 4096-context needle coverage for `qwen3.5-4b` as negative evidence, or move directly to a stronger small-floor candidate because the 32k slice already failed at `0.0%`.
- Stable M2 profile direction: use one reusable isolated profile, `benchclaw-m2`, instead of creating a timestamped profile per run. Initial creation succeeded with config validation, but preflight failed because port `19298` is already occupied by stale benchmark-owned OpenClaw process `123830` from the previous timestamped run. The live 4k run has not been started.
- Cleanup decision needed before the next live run: either stop the stale root-owned benchmark gateways on ports `19191`, `19193`, and `19292`-`19298`, or move `benchclaw-m2` to a known-free port and keep the stale processes running.
- The two-attempt cap was reached for the `workspace_discovery` command scorer in the M1 iteration; do not make another scoring change in that branch without a fresh diagnosis and explicit pivot.
