# AGENTS.md — Working rules for agents in this repo

This file tells any agent (Codex /goal loop, Claude Code, or human) how to work in this repo without making it worse. Read it before touching anything. The umbrella mission lives in [GOAL.md](GOAL.md). Live status lives in [STATUS.md](STATUS.md). The design narrative lives in [README.md](README.md).

If GOAL.md, STATUS.md, and this file disagree, GOAL.md wins on intent, STATUS.md wins on current state, and AGENTS.md wins on process.

## Source of truth

- **GOAL.md** — what we're trying to accomplish and why. Milestone order. Constraints. Out of scope.
- **STATUS.md** — what is currently true: active milestone, blockers, latest staged repo, latest live-anchor run.
- **README.md** — the design rationale and CLI surface. Useful background, but stale-prone.
- **`manifests/*.json`** — the actual benchmark contract. Treat these like API surface.
- **`openclaw_bench/`** — the harness. Scoring, runner, backends, reporting.
- **`tests/`** — the regression net. Treat as load-bearing, but don't pad it.

## Default working posture

1. **Diagnose before fixing.** When something fails, find the root cause before applying a fix. Propose the fix with reasoning. Don't patch-and-pray.
2. **Two-attempt cap.** If a fix isn't working after two real attempts, stop, write up what you learned, and either escalate or pivot. Do not brute-force.
3. **Milestone discipline.** Work GOAL.md milestones in order. Don't start the next one until the prior one's regression test, simulator end-to-end, and (where applicable) live-anchor record are landed.
4. **Update STATUS.md every iteration.** Active milestone, current blocker, latest live-anchor run id, any diagnoses-without-fixes that need human input.
5. **Provider-agnostic by default.** Anything that works only against vLLM must have a degraded path for llama.cpp / Ollama / Apple Silicon, even if "degraded" means "logs a clear unsupported message."
6. **Machine-checkable scoring only.** No LLM-judge, no fuzzy semantic match. Behavior checks, file-existence checks, and `verify_command` are the bar.
7. **Don't destroy things while debugging.** Touching shared config, killing services, restarting the host vLLM systemd unit, mutating a real OC profile — confirm with the human first.
8. **Pinned versions are pinned.** OpenClaw `2026.4.27`. Don't chase newer versions inside this goal.

## Subagent delegation

This repo benefits from parallel work. Main context is the conductor, not the digger.

### When to spawn subagents

Any time there are **2 or more independent threads** of investigation or work, spawn them in parallel. Examples that show up here constantly:

- Auditing scoring rules across multiple task types.
- Detecting providers across vLLM / llama.cpp / Ollama / Apple Silicon.
- Diffing simulator vs live behavior for several task types.
- Triaging anchor model results across tiers.
- Building fixture variants for needle-at-N-context.
- Sweeping concurrency/context combos.

Don't serialize independent work in the main context. Dispatch, synthesize, decide.

### Which model for which agent

Pick the cheapest model that can do the work correctly. The hierarchy:

| Model class | Use for | Examples |
|---|---|---|
| Frontier reasoning (Opus, GPT-5.3 high, etc.) | Diagnosis, root-cause analysis, ambiguous design calls, scoring policy decisions, milestone planning | "Why is this anchor model failing tier-medium?" "What's the right shape for the destructive-refusal task?" |
| Mid (Sonnet, GPT-5.3 mid) | Multi-file refactors with judgment calls, fixture authoring, well-specified-but-non-trivial implementation | "Implement the tool-error-recovery task per this spec." "Add tier-small manifest based on these prompts." |
| Lightweight (GPT-5.3 Spark, Haiku, etc.) | Unambiguous mechanical edits, file moves, fixture stamping, lint fixes, format normalization, ripgrep-style lookups | "Update every needle-task target_file to `app/health.py`." "Find all `manifests/*.example.json` that don't have `suite_id`." "Stamp this prompt across N fixtures." |

**Default to lightweight.** A frontier model running a `sed` is wasted reasoning. If the task can be specified as "do X to Y, no judgment required," route it to Spark/Haiku-class. Reserve frontier reasoning for the parts that actually need it.

If a subagent comes back with an ambiguous result or asks a clarifying question, that's a signal it should have been a higher tier. Note it, requeue at the right tier, and move on.

### Subagent hygiene

- **Brief them like a smart colleague who walked into the room.** They have no memory of this conversation. Include file paths, the goal, what's been ruled out, and what success looks like.
- **Run independent agents in parallel** — single message, multiple Agent tool calls. Don't serialize.
- **Trust but verify.** A subagent's summary describes what it intended to do, not what it did. Check the diff before reporting work as done.
- **Don't delegate understanding.** Don't write "based on the research, implement it." That's the main context's job. Delegate the digging, not the synthesis.

## Test hygiene — don't over-create tests

We have ~200 tests. We do not need 400. The goal is regression coverage of the **harness contract**, not exhaustive coverage of every code path.

### Add a test when

- A scoring rule changed and the change could plausibly regress.
- A new task type was added (one happy-path simulator test).
- A bug was fixed and the bug was reachable through normal use.
- A new manifest schema field was added and parsing logic depends on it.
- Provider detection logic changed and the new path is non-trivial.

### Don't add a test when

- It just exercises a Python stdlib pass-through.
- It tests a private helper that's already covered transitively by a public test.
- It tests an obvious type guard that mypy/typing already enforces.
- It duplicates an existing test with a trivial input variation.
- It exists to "increase coverage" rather than catch a real regression.
- The behavior is enforced by the simulator end-to-end run anyway.

### Prefer

- One end-to-end simulator test per task type, plus targeted unit tests for the scoring branches that have non-obvious logic.
- Fixture-driven tests where the fixture *is* the spec — not parametric tests with 12 string permutations.
- Integration coverage via simulator runs over deep mocking of the OpenClaw backend.

If you find yourself writing a test and can't articulate "this catches the following regression class," delete it.

## Manifests are API

`manifests/*.json` are the contract this repo offers to outside users. Treat them with the discipline you'd treat a public API:

- Don't break field names without a migration note in the README.
- Additive fields are fine. Renames are not.
- `*.example.json` files are templates — keep them runnable against the simulator.
- Generated quickstart manifests must stay deterministic given the same inputs.

## Fixtures are seed data, not throwaways

`fixtures/` repos are seeded into copied workspaces per task attempt. They are part of the scoring contract.

- Don't hand-edit a fixture to make a flaky task pass. Fix the task or the scorer.
- New fixtures should be the smallest realistic repo that exercises the behavior. Don't ship a 200-file fixture for a 1-file question.
- Needle fixtures must have at least one realistic distractor. The needle is not load-bearing if a model can stumble onto the right answer for the wrong reason.
- Real-repo snapshots under `fixtures/real_repos/` are point-in-time copies. Don't update them in place — add a new dated snapshot if the source repo changes.

## Scoring rules

- Every scoring branch in `openclaw_bench/scoring.py` should be explainable in one sentence: "this catches X failure mode."
- If you add a check, leave a one-line comment naming the failure mode (no prose paragraphs).
- Resist exact-string-match on free-form fields. The current `workspace-discovery` `test_command` bug exists because the scorer demanded the exact canonical command instead of any runnable equivalent. Don't repeat that pattern. Score on **whether the answer works**, not whether it matches a string.

## Live runs and external state

- Live OC runs cost real tokens and real GPU time. Don't run a live anchor calibration from a subagent — the main context decides when a live run is worth it.
- The host vLLM systemd unit (`openclaw-vllm-small-bench.service`) is a shared fixture. Don't restart it without the human's go-ahead.
- The `bench` OpenClaw profile is the benchmark's profile. Don't use any other profile for a benchmark run; don't mutate a non-`bench` profile.
- Per-run workspaces under `/home/ubuntu/openclaw-bench/workspaces/<run-id>/` are disposable. Per-run results under `.../results/<run-id>/` are not — they're the audit trail.

## Verify before claiming done

- After every change run `python3 -m unittest discover -s tests`.
- After scoring, runner, or fixture changes also run a simulator end-to-end suite.
- "Done" means the regression test exists, the simulator passes, and STATUS.md reflects the new state. Less than that is in-progress.
- Don't claim a milestone complete without the calibration record (run-id, commit, model id, KV mode, context, concurrency, score, date).

## When data contradicts the plan

If the data answers a milestone's question (e.g., KV-quant turns out not to matter for OC quality), that is a finding. Record it, close the milestone, and move on. Do not manufacture follow-up work to keep the milestone "active."

If the data contradicts an assumption baked into GOAL.md or AGENTS.md, surface it in STATUS.md and propose the doc edit. Don't silently work around the doc.

## Things to never do

- Run `git push --force` on `master`.
- Skip hooks (`--no-verify`) without explicit human approval.
- Delete fixtures, manifests, or result directories to make a problem go away.
- Embed secrets in manifests or fixtures.
- Use a production OC workspace or session as a test fixture.
- Bypass the simulator backend in the regression suite.
- Mark a milestone done because "the test passes" without the calibration record.
- Pivot to a new tool, framework, or approach mid-loop without writing down why in STATUS.md.
