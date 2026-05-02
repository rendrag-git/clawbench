# GOAL

## THE Goal

Make local-model selection for OpenClaw agent work a **one-command, evidence-based decision** — not guesswork, not vibes, not "I tried it for an hour and it seemed fine."

A user with a local or LAN-reachable model server should be able to:

1. Run one command.
2. Get a decision-quality answer to "is this model good enough for OC agent work, at what concurrency, at what context budget, and where does it fail?"
3. Trust the answer enough to make a hardware, quantization, or model-family choice on the strength of it.

That answer must hold across the model classes people actually run — from a 4B local on a single consumer GPU up to a hosted DeepSeek/Kimi-class frontier model — and across the runtimes people actually use (vLLM, llama.cpp, Ollama, Apple Silicon).

Throughput is a diagnostic. **OpenClaw task success is the primary score.**

## Why this matters

Right now OC users pick local models the same way everyone picks local models: rumor, benchmark leaderboards designed for chat not agent work, and the occasional "it crashed at 16k context, oh well." That is not how a runtime that actually depends on tool-calling, multi-file reasoning, long-context retrieval, and instruction adherence should be evaluated. This repo exists so that "is this model usable for OC?" stops being a guess.

## Definition of done (umbrella)

The repo is "done enough to ship as the recommended way to evaluate local models for OC" when:

- **Anyone can run it cold.** Fresh clone → `oc-bench quickstart` → working result, with provider auto-discovery across vLLM, llama.cpp, Ollama, and Apple Silicon local runtimes.
- **The score is trustworthy.** Scoring is machine-checkable, deterministic on the simulator backend, and free of known false-positive/negative bugs. Anchor-model calibration is recorded with date, commit, and config.
- **Tiers discriminate.** Small/medium/large/xlarge suites each separate a documented floor model from a documented ceiling model on simulator + at least one live anchor.
- **The hard questions get answered.** The benchmark produces a populated decision table covering single-agent coding, multi-agent background work, long-context repo search, and stress-concurrency — not just per-task scores.
- **Concurrency and long-context sweeps produce defensible recommendations.** The decision table tells a user "at this hardware and use case, this model wins" with attempt rows behind every row. KV-quant comparison (M5) is a side investigation, not the umbrella; see M5 for why it stays deferred.
- **Real-repo coverage exists beyond one snapshot.** Multiple real local repos with read-only and patch tasks, so synthetic fixtures aren't the only signal.
- **Results are shareable.** A certified run produces an artifact (summary.md + summary.json + attempts.jsonl + server.json + per-attempt raw + patches) that another person can reproduce or audit.

## Milestones

In rough priority order. Each milestone has its own definition of done; the loop should land them sequentially unless a hard blocker forces parallel work.

### M1. Trust hygiene — scoring is honest

The bench cannot be the source of truth if its scorer has known bugs. This is the prerequisite for everything else.

- Fix the `workspace-discovery` `test_command` exact-string-match issue (STATUS.md: latest E2E failed because the model returned a real file path instead of the canonical command). Either tighten the prompt to demand a runnable shell command, or accept any answer that passes `_test_command_runnable`. Add a regression test either way.
- Audit `scoring.py` for other exact-match-vs-semantic-correctness traps. Document each scoring rule's intent next to the code so future task authors don't re-introduce the same class of bug.
- Confirm simulator backend produces a result for every task type that mirrors what the live backend would score, so simulator runs are a real regression signal.

### M2. Tiered, discriminating task suite

Four tier manifests — `tier-small`, `tier-medium`, `tier-large`, `tier-xlarge` — each calibrated against a floor model (must pass) and a ceiling model (must fail). Target separation: floor ≥ 90%, ceiling ≤ 30%.

Anchor candidates:

| Tier | Floor | Ceiling | Character |
|---|---|---|---|
| small (3–8B) | Qwen3.5-4B / Llama 3.2 3B | — | Single-file, strict JSON, ≤4k ctx, ≤6 tool calls. "Usable at all?" |
| medium (14–32B) | GPT-OSS 20B / Qwen3-32B | Qwen3.5-4B | Multi-file edits, 16–32k needle, instruction retention, tool-error recovery, AGENTS.md adherence. "Production candidate?" |
| large (70–200B) | Llama 3.3 70B / GPT-OSS 120B | GPT-OSS 20B | Plan-then-execute, cross-file consistency, real-repo refactor, 64k needle. "Worth the VRAM?" |
| xlarge (400B+) | DeepSeek V3.5 / Kimi K2.6 | Llama 3.3 70B | Long-session coherence, destructive-action refusal under social pressure, multi-agent delegation, 128k+ needle. "Frontier-class agent work?" |

Task-gap coverage required across the tier set (at least one task per behavior, in the appropriate tier):

- Tool-loop / stop-condition discipline (score `tool_calls ≤ N` after answer is in context).
- Tool-error recovery (file moved/renamed; recover via grep/ls, not hallucinate).
- Destructive-action refusal (`rm -rf`, force-push, schema drop; pass = refuse or ask).
- Plan-then-execute coherence (plan-vs-action alignment scored).
- Cross-file consistency (one logical change must land in two files that have to agree).
- AGENTS.md / SOUL.md adherence (correct answer requires honoring seeded agent docs).
- Format-drift under length (strict JSON after 10+ tool calls).
- Ambiguous-spec triage (underspecified prompt; correct behavior is to ask, not assume).

Existing `openclaw-agent-core.json` and `real-repo-readonly.example.json` stay as-is for backward compatibility; tiers are additive.

### M3. Provider breadth — inspect-first, four runtimes

The provider story before this milestone was "vLLM works, the rest are aspirational." Goal: vLLM, llama.cpp, Ollama, and LM Studio (covering Apple Silicon and other consumer setups) each have a working detection path, generated config, route smoke, and a passing tier-small run.

- Detection tests pass for each runtime (one clear endpoint, ambiguous endpoint, empty endpoint).
- Generated OpenClaw config is correct per runtime (no `serve_command` for discovered external providers).
- Provider-specific parameter shaping handled (Qwen thinking off when applicable, GPT-OSS `reasoning_effort="low"`, etc.).
- `oc-bench init --providers local` survives a cold-start install on a non-NVIDIA machine.
- Host-vs-runtime mismatches (e.g., a UFW rule that blocks the OC container from reaching a host vLLM) surface as a named finding, not a silent timeout.

**Current state:** the inspect-first deployment surface is shipped, with vLLM fully implemented (detect + generate + parameter shaping) and Ollama / llama.cpp / LM Studio as detect-only stubs. The cascade, runtime auto-derive, four-gate `oc-bench provider-preflight`, and host/runtime mismatch detection are all live. Each remaining stub graduates to a real generator one runtime at a time, gated by being able to validate against a real instance.

### M4. Concurrency + long-context sweeps produce real data

Phases 3 and 4 from the README design: 1/2/4/8/16/32/64 concurrency, 4k/8k/16k/32k/64k context. Today these are outlined; the goal is producing trustworthy result tables for at least one anchor model per tier.

- Concurrency sweep records P50/P95/P99 wall time, TTFT, server errors, OOMs, OC timeouts.
- Long-context sweep produces needle pass-rate by context size with KV-mode breakdown.
- Failure taxonomy is enforced: every failure has a classification, no `unknown` without a stderr breadcrumb.

### M5. KV-quant decision data — deferred, side investigation

KV-cache quantization (TurboQuant K8V4 / K3V4 vs FP8 vs `provider_default`) is an interesting axis that *might* affect OC agent quality on long contexts and high concurrency. It is **not** the umbrella motivating question — that lives in the umbrella objective above ("which local model is good enough to run for OC agent work"). M5 is a side investigation that surfaces *if* and *when* the question becomes practical to answer.

Practical blockers today:

- Closing this milestone requires side-by-side anchor-model runs at fixed weights with KV mode as the only variable. That requires access to multiple KV-mode runtime variants (vLLM TurboQuant builds, FlashInfer KV cache backends) on the same hardware. Most setups don't have that.
- Even with the hardware, a defensible answer needs community-shared runs to corroborate single-machine results — the bench currently has no upload/share path (M8 is portable artifacts, not aggregation), and no community of submitters exists.
- For non-vLLM runtimes (Ollama, LM Studio, hosted APIs) KV mode is not user-selectable at all, so the question is moot for them.

Until both the hardware-access and corroboration problems are solvable, M5 stays open as a recorded gap, not an active milestone. The primary "which model should I run" answer comes from M7 (Decision-quality reporting) fed by M2-M4 calibrations across tier suites, concurrency sweeps, and long-context sweeps — not from M5.

### M6. Real-repo coverage beyond one snapshot

`fixtures/real_repos/` includes more than the kingshot-ams snapshot. At minimum: one TypeScript repo, one Python repo with a non-trivial test suite, one repo with a real bug fixture from history. Read-only and patch tasks for each.

### M7. Decision-quality reporting

`summary.md` populates the decision table from the README:

```text
Use case | Best model/KV | Reason | Risk
single-agent coding | ... | ... | ...
4-agent background work | ... | ... | ...
long-context repo search | ... | ... | ...
high-concurrency stress | ... | ... | ...
```

Not template — populated, with links back to the attempt rows that justify each row.

### M8. Certification + shareable artifacts

The certification flow (`oc-bench certify`) audits a result directory against the umbrella objective and produces a portable artifact bundle. Upload/database integration is still scoped out as a separate goal, but the artifact must be self-contained and auditable offline.

## Constraints

- **Provider-agnostic.** No milestone may require vLLM specifically. Anything that works only against vLLM has to also have a path for llama.cpp / Ollama / Apple Silicon, even if degraded.
- **Inspect-first, ask-second UX.** The CLI never forces a user to know the exact config shape if the machine already exposes enough info to infer it.
- **Machine-checkable scoring only** in this phase. No LLM-judge, no fuzzy semantic match. Behavior checks, file-existence checks, and `verify_command` are the bar.
- **Simulator parity.** Every new task type must produce a deterministic simulator result so bench-mechanics regressions are caught without burning live tokens.
- **Anchor calibration is durable.** Every tier ships with a calibration record (run-id, commit, model id, KV mode, context, concurrency, score, date). Recalibration is triggered by date drift, not vibes.
- **OpenClaw `2026.4.27` is pinned.** Do not chase newer OC versions inside this goal; that is a separate decision.
- **No production OC workspaces or sessions as test fixtures.** Ever.
- **Don't destroy things while debugging.** Touching shared config, killing services, or changing the host vLLM systemd unit needs explicit user confirmation.

## Out of scope for THE goal

- Adding non-local provider families beyond OpenAI/Anthropic API + the four local runtimes (subscription/OAuth providers stay BYO-auth for this phase).
- Upload/database persistence of results (M8 stops at portable artifacts).
- Migrating off OC `2026.4.27`.
- Fine-tuning, training, or model conversion work.
- Building a UI on top of the bench.

## Notes for the loop

- Land milestones in order, with one explicit exception: **the M3 deployment surface unblocks the M2 live calibrations**. Trying to capture floor/ceiling records before the bench can detect and configure non-vLLM runtimes traps you in per-environment debugging (UFW, meta-field, context-floor) every time you swap candidates. Ship the surface first; floor-model discovery falls out the side.
- After every task, scoring change, or fixture change, run `python3 -m unittest discover -s tests` AND a simulator end-to-end. If either regresses, do not proceed.
- Keep `STATUS.md` as **current state**, not a running log. Milestone-by-milestone commit history belongs in `git log`. STATUS should answer "what shipped, what's running, what's open" in under 100 lines.
- When a task fails to discriminate between floor and ceiling, do not delete it — record what it actually measures and either reassign it to the right tier or repurpose it. Negative results are signal.
- If iteration on a single task's prompt exceeds 3 rounds, stop: the task shape is probably wrong, not the wording.
- "Done" for a milestone means the regression test exists, the simulator passes, at least one live anchor run is recorded (or, for M3, the deployment-surface live test is recorded), and `STATUS.md` reflects it. Anything less is in-progress, not done.
- Challenge the goal itself if data forces it. If, for example, KV-quant turns out not to matter for OC quality at all, that is a finding — record it, close M5, and move on. Do not chase work that the data has already answered.
