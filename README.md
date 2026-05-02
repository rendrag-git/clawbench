# clawbench

Benchmark harness for OpenClaw agent quality across local and API LLM providers. Answers exactly one question, with evidence: *is this local model usable for OpenClaw agent work, at what concurrency, at what context, and where does it fail?*

This is **not** a generic LLM benchmark. No leaderboards, no chat-quality scores. Output is decision-grade — per-attempt rows another person can audit or reproduce.

## Where to read what

| File | Purpose | Read when |
|---|---|---|
| [GOAL.md](GOAL.md) | Capability + acceptance + progress log | You want to know what this project is trying to accomplish |
| [STATUS.md](STATUS.md) | Current operational state — services, endpoints, profiles, latest runs | You want to know what's running right now |
| [LEARNINGS.md](LEARNINGS.md) | Durable operational facts and incident root causes | You hit a problem that someone else likely already hit |
| [docs/design/benchmark-shape.md](docs/design/benchmark-shape.md) | Two layers, isolation model, task suite, metrics, scoring, failure taxonomy, reporting | You want the conceptual model |
| [docs/design/model-matrix.md](docs/design/model-matrix.md) | Original model matrix + phased rollout (predates the GOAL.md tier model) | Historical / schema reference |
| [docs/operations/quickstart.md](docs/operations/quickstart.md) | `oc-bench init` and `oc-bench quickstart` | First-time setup |
| [docs/operations/simulator.md](docs/operations/simulator.md) | Mechanics smoke (no live model) | Validating harness changes |
| [docs/operations/local-vllm.md](docs/operations/local-vllm.md) | Live local vLLM cookbook (host-specific examples) | Running against your own vLLM |
| [docs/operations/api-providers.md](docs/operations/api-providers.md) | OpenAI / Anthropic provider runs | Running API-key providers |
| [docs/operations/certification.md](docs/operations/certification.md) | `oc-bench certify` audit | Producing a certified result set |

## Install

```bash
pip install -e .
```

`oc-bench` and `openclaw-bench` are equivalent entrypoints (both → `openclaw_bench.cli:main`). All commands also work as `python3 -m openclaw_bench <subcommand>`.

OpenClaw is pinned to `2026.4.27`; `2026.4.29` is blocked until observed regressions resolve. See [STATUS.md](STATUS.md).

## Three-line quickstart

```bash
oc-bench init --providers local       # discover what's running, write a profile + manifest
oc-bench quickstart --providers local --force --stop-after  # run discovery smoke against it
oc-bench start                          # start the bench gateway when you want to run more
```

For anything beyond smoke, see [docs/operations/quickstart.md](docs/operations/quickstart.md) and the runtime-specific cookbooks above.

## Project layout

```text
openclaw_bench/        Python package: backend, runner, scoring, providers/, cli, ...
manifests/             Suite + model manifests (*.example.json are host-specific examples)
openclaw-config/       Repo-owned, non-secret OpenClaw provider config examples
fixtures/              Synthetic + real-repo fixtures used by tasks
tests/                 Unit and integration tests (live tests gated by OC_BENCH_LIVE=1)
deploy/                Sample systemd units (host-specific; copy and edit)
docs/                  Design + operations documentation (this README is the index)
```

## Contributing

- The simulator backend (`--backend simulator`) covers harness mechanics without live tokens. Run it before sending changes that touch scoring, workspace isolation, or report generation.
- Tests live under `tests/`. Live tests are skipped unless `OC_BENCH_LIVE=1`.
- `GOAL.md` is the source of truth for capabilities and acceptance. Update it before opening work that closes a gap.

## Acknowledgments

This project is distinct from but informed by **[openclaw/clawbench](https://github.com/openclaw/clawbench)** (MIT). That repo is the canonical OpenClaw agent benchmark — broader scope, signal-curated task selection, trace-based scoring with judge advisory, dynamical-systems diagnostics. We adopt specific methodology where it strengthens the local-runtime decision question this repo is built around:

- **Multi-seed reliability metrics** (`pass^k`, worst-of-n, pass-rate, `cell_status`) — `openclaw_bench/aggregation.py`. Pattern adopted from upstream after their v4 sweep audit decomposed 40-task variance and found 47 % was seed noise. See `CLAWBENCH_V0_4_SPEC.md` in that repo for the original spec.
- *(Planned)* Bootstrap CIs and Taguchi S/N for decision-table reporting; per-task SNR variance decomposition for tier-suite audits. Both also derive from the upstream methodology.

What this repo does **not** adopt from upstream and why is recorded in the design notes — chiefly the LLM judge sidecar (this phase is machine-checkable scoring only, per [GOAL.md](GOAL.md) Build Principles) and the dynamical-systems diagnostics suite (out of scope for "is this local model usable for OC agent work").
