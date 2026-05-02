# Goal

Make local-model selection for OpenClaw agent work a one-command, evidence-based decision — not guesswork, not vibes, not "I tried it for an hour and it seemed fine."

This is **not** a generic LLM benchmark. It does not produce chat-quality scores, leaderboards, or fluff. It answers exactly one question, with evidence: *is this local model usable for OpenClaw agent work, at what concurrency, at what context, and where does it fail?*

## Target Outcome

A user with a local or LAN-reachable model server (vLLM, llama.cpp, Ollama, LM Studio, hosted OpenAI/Anthropic) runs `oc-bench quickstart`. The bench inspects what's already running, generates an OpenClaw routing config from real probes, runs the tier suite, and produces a decision-quality summary that tells them "for your hardware and use case, this model wins, this one doesn't, and here's why" — backed by per-attempt evidence rows another person can audit or reproduce.

The answer holds across model classes (4B on a single consumer GPU up to hosted DeepSeek / Kimi-class frontier) and across the runtimes people actually use.

## Required Capabilities

Each capability has an Acceptance checklist. Items are marked complete only when verified evidence is recorded in the Progress section below (file:line, command output, run-id, artifact path, commit SHA). "Should work" claims are not evidence.

1. **Trust hygiene — scoring is honest**

   The scorer cannot be the source of truth if it has known false-positive or false-negative bugs. Every scoring branch has a one-line "intent" comment naming the failure mode it catches. The simulator backend produces a result for every task type that mirrors what the live backend would score.

   Acceptance:
   - [x] Workspace-discovery exact-string-match bug fixed; safelist runnability check covers `python -m unittest|pytest` + `pytest` against `tests/`.
   - [x] Discovery path-equivalence false positive fixed; equivalent relative paths resolve under workspace before comparing.
   - [x] Slash-prose hallucinated-path false positive fixed; only candidates with a file suffix count as references.
   - [x] Simulator full-suite produces 40 attempts / 0 failures and matches live backend output shape.

2. **Tiered, discriminating task suite**

   Four tier manifests (`tier-small`, `tier-medium`, `tier-large`, `tier-xlarge`) calibrated against a floor model (must pass) and a ceiling model (must fail). Target separation: floor ≥ 90 %, ceiling ≤ 30 %. Every behavior in the design — tool-loop discipline, tool-error recovery, destructive-action refusal, plan-then-execute coherence, cross-file consistency, AGENTS.md adherence, format-drift under length, ambiguous-spec triage, long-context needle, workspace discovery + patch — has at least one covering task in the appropriate tier.

   Acceptance:
   - [x] `manifests/tier-{small,medium,large,xlarge}.json` committed, with at least one task per tier.
   - [x] Every behavior listed above has a covering task; calibration schema validation in `openclaw_bench/calibration.py`.
   - [ ] Live floor calibration record for tier-small (run-id, commit, model id, KV mode, context, concurrency, score, date).
   - [ ] Live floor + ceiling records for tier-medium, tier-large, tier-xlarge.
   - [ ] Multi-seed reliability metrics (`pass^k`, worst-of-n, pass-rate) recorded for every calibration run with `--runs-per-task ≥ 3`. Adopted from `openclaw/clawbench` upstream after the audit found 47 % of their 40-task variance was seed noise; single-run scoring can't tell a 90 % model from a flaky 90 %/20 % one.

3. **Provider breadth — inspect-first, four runtimes**

   `oc-bench init --providers local` works cold across vLLM, llama.cpp, Ollama, and LM Studio. Detection runs from where OpenClaw will route from (host or container), surfaces host-vs-runtime mismatches as a named finding, and generates a correct OpenClaw provider config per runtime — no `serve_command` for discovered external providers. Provider-specific parameter shaping (Qwen `enable_thinking=false`, GPT-OSS `reasoning_effort="low"`) is encoded.

   Acceptance:
   - [x] Detection cascade with `LocalProbe`, `IncusExecProbe`, `DockerExecProbe`, `SSHProbe`; profile-config scan; 30 s/provider port-probe budget; host-vs-runtime mismatch finding.
   - [x] vLLM module: detect, generate (delegates to `quickstart._vllm_provider_config`, inheriting 16 k context floor + meta + plugin entries), parameter shaping. Live test against GPT-OSS via `oc-stack` passes.
   - [x] `oc-bench provider-preflight` wraps four gates (config validate, models list, provider health with auth-header forwarding, OpenClaw route smoke).
   - [ ] Ollama generator implemented and validated against a live Ollama instance.
   - [ ] llama.cpp generator implemented and validated against a live `llama-server`.
   - [ ] LM Studio generator implemented and validated against a live LM Studio instance.

4. **Concurrency and long-context sweeps produce real data**

   For at least one anchor model per tier, the bench produces P50/P95/P99 wall time, TTFT, server errors, OOMs, and OC timeouts at concurrency 1/2/4/8/16/32/64, plus needle pass-rate by context size at 4 k / 8 k / 16 k / 32 k / 64 k. Every failure is classified — no `unknown` entries without a stderr breadcrumb.

   Acceptance:
   - [ ] Concurrency sweep recorded for at least one tier-small anchor.
   - [ ] Concurrency sweep for at least one tier-medium anchor.
   - [ ] Long-context sweep recorded across the full context grid for at least one tier-medium anchor.
   - [ ] Failure taxonomy enforced: zero `unknown` failure types in any recorded run.

5. **KV-quant decision data — deferred, side investigation**

   KV-cache quantization (TurboQuant K8V4 / K3V4 vs FP8 vs `provider_default`) is an interesting axis that *might* affect OC agent quality on long contexts and high concurrency. It is **not** the umbrella motivating question — that lives in the Target Outcome above. M5 is a side investigation that surfaces *if* and *when* the question becomes practical to answer.

   Practical blockers today:
   - Closing M5 requires side-by-side anchor-model runs at fixed weights with KV mode as the only variable, on multiple KV-mode runtime variants (vLLM TurboQuant builds, FlashInfer KV cache backends). Most setups don't have that hardware access.
   - A defensible answer needs community-shared runs to corroborate single-machine results. The bench has no aggregation path, and no community of submitters exists.
   - For Ollama / LM Studio / hosted APIs, KV mode is not user-selectable, so the question is moot for them.

   Acceptance: M5 stays open as a recorded gap until both the hardware-access and corroboration problems are solvable. No checkboxes — there is no path to verifiable progress here today.

6. **Real-repo coverage beyond one snapshot**

   `fixtures/real_repos/` includes more than the kingshot-ams-snapshot. Read-only and patch tasks are folded into the tier suites for at least one TypeScript repo, one Python repo with a non-trivial test suite, and one repo with a real bug fixture pulled from history.

   Acceptance:
   - [ ] At least one new real-repo fixture beyond `kingshot-ams-snapshot`.
   - [ ] At least one tier manifest references real-repo read-only tasks.
   - [ ] At least one tier manifest references a real-repo patch task with a `verify_command`.

7. **Decision-quality reporting**

   `summary.md` produced by a benchmark run populates the decision table — not template, populated, with links back to the attempt rows that justify each row.

   Acceptance:
   - [ ] `summary.md` includes a decision table covering: single-agent coding, 4-agent background work, long-context repo search, high-concurrency stress.
   - [ ] Each row names a recommended model + reason + risk, with at least one attempt-row link backing the recommendation.
   - [ ] `summary.json` carries the same data in machine-readable form.

8. **Certification + shareable artifacts**

   `oc-bench certify` audits a result directory against the umbrella objective and produces a portable artifact bundle (`summary.md` + `summary.json` + `attempts.jsonl` + `server.json` + per-attempt raw + patches) that another person can reproduce or audit offline.

   Acceptance:
   - [x] Certification module exists with multi-run audit, sim-vs-live gating, hardware/throughput coverage. (50 tests in `test_certification.py`.)
   - [ ] At least one full certified result directory exists from a live run, audited end-to-end.
   - [ ] Bundle is self-contained (no external file references) and re-loadable on another machine.

## Build Principles

- **Provider-agnostic.** Anything that works only against vLLM must have a degraded path for llama.cpp / Ollama / LM Studio / Apple Silicon, even if "degraded" means "logs a clear unsupported message."
- **Inspect-first, ask-second UX.** The CLI never forces a user to know the exact config shape if the machine already exposes enough info to infer it.
- **Machine-checkable scoring only** in this phase. No LLM-judge, no fuzzy semantic match. Behavior checks, file-existence checks, and `verify_command` are the bar.
- **Simulator parity.** Every new task type produces a deterministic simulator result so bench-mechanics regressions are caught without burning live tokens.
- **Anchor calibration is durable.** Every tier ships with a calibration record (run-id, commit, model id, KV mode, context, concurrency, score, date). Recalibration is triggered by date drift, not vibes.
- **Pinned versions are pinned.** OpenClaw `2026.4.27`. `2026.4.29` is blocked until observed regressions resolve.
- **No production OC workspaces or sessions as test fixtures. Ever.**
- **Don't destroy things while debugging.** Touching shared config, killing services, or changing the host vLLM systemd unit needs explicit user confirmation.
- **Land Required Capabilities in order with one explicit exception:** Capability 3 (deployment surface) unblocks Capability 2 (live calibrations). Trying to capture floor/ceiling records before the bench can detect and configure non-vLLM runtimes traps you in per-environment debugging every time you swap candidates.
- **Verify before marking done.** Run the test, read the diff, confirm the output. "Should work" is not evidence.
- **If the same fix fails twice, stop.** Write the blocker into Progress instead of trying a third variant.
- **Challenge the goal itself if data forces it.** If a Capability turns out to be unsolvable or its question is already answered by data, record that and move on. Do not chase work the data has answered.

## Current Source Of Truth

- `GOAL.md` (this file) — what we're trying to accomplish, with verifiable Acceptance per Capability.
- `STATUS.md` — current operational state: services, endpoints, profiles, latest runs.
- `AGENTS.md` — process rules for any agent (Codex, Claude Code, human) working in the repo.
- `README.md` — design rationale and CLI surface. Useful background, stale-prone.
- `manifests/*.json` — the actual benchmark contract. Treat as API surface.
- `openclaw_bench/scoring.py` — every machine-checkable rule, with intent comments.

Anything not listed here is reference material only unless explicitly promoted.

## Progress

Maintained as the canonical view of what is done, in flight, and blocked. Update before taking action on each continuation turn. Every entry includes verified evidence — file:line, artifact path, command output, commit SHA, run-id. Keep entries terse: one line each, evidence by reference.

### Completed

- 2026-05-02, **Capability 1 / all four items.** Scoring fixes committed in main branch history (`a9fd98b` workspace-discovery, `4013993` discovery path-equivalence, `936395b` slash-prose hallucinated-path). Simulator certification full run produces 40 attempts / 0 failures consistently; latest verification at `/tmp/openclaw-bench-m3-providers-verify` (commit `cb951f7`).
- 2026-05-02, **Capability 2 / first two items.** Tier manifests `manifests/tier-{small,medium,large,xlarge}.json` committed; 14 tasks across 4 tiers covering all 9 design behaviors (verified against GOAL behavior list). Calibration schema validation in `openclaw_bench/calibration.py` with regression test (commit `80309c0` and follow-ups).
- 2026-05-02, **Capability 3 / first three items.** Provider-detection deployment surface shipped across 17 commits (`5074859..129f79a`). vLLM full module at `openclaw_bench/providers/vllm.py`, detect-only stubs for Ollama/llama.cpp/LM Studio, `oc-bench provider-preflight` wraps four gates, live test against GPT-OSS via `oc-stack` passes (`tests/test_providers_live.py`, `OC_BENCH_LIVE=1`). 293 unit tests pass.
- 2026-05-02, **Capability 8 / first item.** Certification module exists with 50 tests in `tests/test_certification.py`.

### In Progress

- **Bridge:** the multi-seed work feeds Capability 2 / Acceptance item "Multi-seed reliability metrics (`pass^k`, worst-of-n, pass-rate) recorded for every calibration run." Output enters product state when an `oc-bench run --runs-per-task 3 ...` produces a `summary.md` with a Reliability section that names per-task `pass^k` numbers backed by attempt rows in `attempts.jsonl`.
- 2026-05-02, branch `feature/multi-seed-runs` open. Items in flight: `--runs-per-task` CLI flag → runner loop with `run_index` per attempt → `openclaw_bench/aggregation.py` for pure metric computation → reporting wires it into `summary.md`/`summary.json` → tests (unit + simulator end-to-end). Single PR; user merges. Once merged, follow-up PR adds bootstrap CIs + Taguchi S/N (item 2 from the upstream audit) and a third PR adds the `tier-audit` SNR decomposition (item 3).
- Bridge for parallel track: Capability 3 / Ollama generator remains the next non-reliability priority. Stand up Ollama on the RTX Pro 5000 (GPU 1, alongside or replacing GPT-OSS), capture its `/api/tags` response shape, write `openclaw_bench/providers/ollama.py.generate_route_config` per the OpenClaw Ollama provider docs, validate end-to-end with `oc-bench provider-preflight --provider ollama`.

### Blockers / Open Questions

- **Capability 2 / live tier-small floor record blocked.** Qwen3.5-4B times out (>600 s wall time) on the 32 k tier-small slice under the current vLLM setup; not a calibration record. Need a different 4–8 B candidate. Suggested path: ship the Ollama generator (Capability 3) so Llama 3.2 3B or Qwen3-8B via Ollama becomes a discoverable candidate. See latest live run `live-m2-small-floor-qwen35-fixed-20260502002059` in STATUS.md for evidence.
- **Capability 3 / detection-driven init has silent-fallback bug.** [Issue #1](https://github.com/rendrag-git/clawbench/issues/1) — when detection finds a non-vLLM provider (e.g. Ollama), `init` silently falls back to default vLLM values rather than erroring. ~10-line fix in `openclaw_bench/cli.py` `init_command`.
- **Capability 3 / detection probes loopback only.** [Issue #2](https://github.com/rendrag-git/clawbench/issues/2) — services bound to a bridge address (e.g. `10.68.198.1` for both Qwen and GPT-OSS in this setup) are missed by default `init`. `--probe-hosts` flag not yet exposed.
- **OpenClaw `2026.4.29` upgrade blocked** until observed regressions diagnose. No work scheduled here.
- **Capability 5 deferred** with practical blockers spelled out in the Capability description above. Not active work.

### Iteration Log

Append-only. One line per continuation turn.

```
- 2026-05-02 04:30 UTC, restructured GOAL.md to capability-and-acceptance format per codex-goal-loop template; reframed M5 as deferred side investigation (commit cb951f7); seeded Progress section with state from M3 deployment-surface slice. next: pick up Capability 3 / Ollama generator — stand up Ollama on RTX Pro 5000 and validate `oc-bench provider-preflight --provider ollama`.
- 2026-05-02 05:00 UTC, audit of openclaw/clawbench upstream identified three adoptable patterns (multi-seed runs + pass^k aggregation, bootstrap CIs + Taguchi S/N, per-task SNR variance decomposition). Added new Capability 2 acceptance bullet for multi-seed reliability metrics. Started feature branch `feature/multi-seed-runs` for item 1. next: implement runner loop over `--runs-per-task N`, add aggregation module, wire reliability into reports, open PR.
```
