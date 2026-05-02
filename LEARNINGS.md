# LEARNINGS.md

Operational facts and hard-won findings that future agents should check before repeating work. Keep this file short, evidence-based, and linked to runs or commits where possible.

## OpenClaw Runtime

- OpenClaw `2026.4.27` rejects model routes whose configured `modelsConfig` context window is below `16000` tokens. Evidence: live run `live-m2-small-needle4k-qwen35-20260502010400` failed before task execution with `Model context window too small (4096 tokens; source=modelsConfig). Minimum is 16000.` Direct vLLM health and chat-completions probes passed, so this was an OpenClaw route/config constraint, not a model-server failure.
- Treat benchmark context tiers, OpenClaw route context, and model serve context as separate fields. A 4k/8k task can remain labeled and filtered as 4k/8k in the manifest, but generated OpenClaw provider config must advertise at least the runtime-supported route window so the agent reaches the task.
- Do not recommend serving an OpenClaw agent benchmark model with `--max-model-len 4096`. Even a "4k" task can exceed that once OpenClaw's gateway prompt, tool schema, workspace-agent setup, and answer budget are included. Prefer `32768` for live agent-serving experiments when VRAM allows; fall back to `16384` only when 32k OOMs.
- GPT-OSS 20B ModelOpt NVFP4 without an artificial serve context cap resolves to `max_seq_len=131072`, but failed to start on the RTX PRO 5000 Blackwell with the current vLLM/FlashInfer stack: FlashInfer FP4 GEMM raised `No supported CUDA architectures found for major versions [12]` for SM120. Do not keep retrying this exact launch shape; fix the FP4 kernel stack or use a different checkpoint/quantization path.
- The official `openai/gpt-oss-20b` MXFP4 checkpoint starts successfully on the RTX PRO 5000 Blackwell under the same no-artificial-cap policy. Evidence: vLLM selected `max_model_len=131072`, `quantization=gpt_oss_mxfp4`, loaded on GPU 1 / port `8000`, and direct `/v1/chat/completions` returned visible `ok` with `max_tokens=128`.
- GPT-OSS 20B can spend small `max_tokens` budgets entirely on reasoning. For short route/smoke prompts, pass `reasoning_effort: "low"` and enough output budget; otherwise the visible answer can be empty or truncated even though the model transport works.
- If direct GPT-OSS vLLM calls work but OpenClaw route smoke times out, verify the outbound payload before changing token budgets again. A proxy capture from `benchclaw-m2-gptoss` proved `agents.defaults.models["vllm/gpt-oss-20b"].params.extra_body.reasoning_effort="low"` becomes top-level `reasoning_effort: "low"` in the OpenAI-compatible chat-completions request. The gateway debug line `creating streamFn wrapper with params: {"maxTokens":512}` does not include `extra_body`, so it is not proof that request-body overrides were ignored. If the route still times out, compare vLLM timing for the exact captured payload under the gateway timeout budget.
- OpenClaw config validation can restore a prior `last-good` config when a generated profile file lacks OpenClaw `meta` fields. Keep generated benchmark configs stamped with pinned OpenClaw metadata, or `init --force` can appear to succeed while validation puts old provider values back.
- Do not create a fresh timestamped OpenClaw profile for every M2 run. Use one stable isolated profile, `benchclaw-m2`, and vary run IDs, result paths, workspace roots, suite manifests, and model manifests. If the profile config changes, record the migration in `STATUS.md`.
- Benchmark-started foreground OpenClaw gateways must be stopped after the run. Commit `e5ef3db` updates `run_command` to stop only non-container foreground gateways that the harness itself started and leave already-running gateways alone.
- Before assigning a gateway port, check listeners inside `oc-stack` with `ss -ltnp`. Stale benchmark gateways previously occupied ports `19191`, `19193`, and `19292`-`19298`, causing `benchclaw-m2` on `19298` to receive `401` from the wrong gateway.

## M2 Calibration

- `qwen3.5-4b` at 32k on the current vLLM/OpenClaw setup is below the intended small-floor behavior on the observed slice. Run `live-m2-small-floor-qwen35-fixed-20260502002059` produced `2` attempts, `2` `openclaw_timeout` failures, and `0.0%` pass rate. It skipped `small-workspace-needle-4k`, so it is diagnostic evidence, not a durable small-floor calibration record.
- A 4k small-tier live task cannot be represented by setting the OpenClaw route context to `4096` under OpenClaw `2026.4.27`. If a task is logically a 4k task, keep the route context at an OpenClaw-supported value and preserve the benchmark context metadata separately, or revise the tier calibration plan.

## Scoring

- Slash-separated prose is not necessarily a file path. The scorer now treats slash references as file references only when the candidate has a file suffix; this avoids false hallucinated-file failures on prose such as `leading/trailing`.
- Workspace discovery answers should be scored by resolving paths inside the workspace, not by exact string equality. `./api/routes.py` and `api/routes.py` are equivalent if they resolve to the expected file.
