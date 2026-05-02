# LEARNINGS.md

Operational facts and hard-won findings that future agents should check before repeating work. Keep this file short, evidence-based, and linked to runs or commits where possible.

## OpenClaw Runtime

- OpenClaw `2026.4.27` rejects model routes whose configured `modelsConfig` context window is below `16000` tokens. Evidence: live run `live-m2-small-needle4k-qwen35-20260502010400` failed before task execution with `Model context window too small (4096 tokens; source=modelsConfig). Minimum is 16000.` Direct vLLM health and chat-completions probes passed, so this was an OpenClaw route/config constraint, not a model-server failure.
- Treat benchmark context tiers and OpenClaw route context as separate fields. A 4k/8k task can remain labeled and filtered as 4k/8k in the manifest, but generated OpenClaw provider config must advertise at least the runtime-supported route window so the agent reaches the task.
- Do not create a fresh timestamped OpenClaw profile for every M2 run. Use one stable isolated profile, `benchclaw-m2`, and vary run IDs, result paths, workspace roots, suite manifests, and model manifests. If the profile config changes, record the migration in `STATUS.md`.
- Benchmark-started foreground OpenClaw gateways must be stopped after the run. Commit `e5ef3db` updates `run_command` to stop only non-container foreground gateways that the harness itself started and leave already-running gateways alone.
- Before assigning a gateway port, check listeners inside `oc-stack` with `ss -ltnp`. Stale benchmark gateways previously occupied ports `19191`, `19193`, and `19292`-`19298`, causing `benchclaw-m2` on `19298` to receive `401` from the wrong gateway.

## M2 Calibration

- `qwen3.5-4b` at 32k on the current vLLM/OpenClaw setup is below the intended small-floor behavior on the observed slice. Run `live-m2-small-floor-qwen35-fixed-20260502002059` produced `2` attempts, `2` `openclaw_timeout` failures, and `0.0%` pass rate. It skipped `small-workspace-needle-4k`, so it is diagnostic evidence, not a durable small-floor calibration record.
- A 4k small-tier live task cannot be represented by setting the OpenClaw route context to `4096` under OpenClaw `2026.4.27`. If a task is logically a 4k task, keep the route context at an OpenClaw-supported value and preserve the benchmark context metadata separately, or revise the tier calibration plan.

## Scoring

- Slash-separated prose is not necessarily a file path. The scorer now treats slash references as file references only when the candidate has a file suffix; this avoids false hallucinated-file failures on prose such as `leading/trailing`.
- Workspace discovery answers should be scored by resolving paths inside the workspace, not by exact string equality. `./api/routes.py` and `api/routes.py` are equivalent if they resolve to the expected file.
