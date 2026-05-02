# Model Matrix and Run Phases

> **Status: predates the current GOAL.md framework.**
>
> This document captures the original model-matrix research plan (Tracks A/B) and the original phased rollout (Phase 0–5). Both have been superseded:
>
> - GOAL.md restructured around four tier manifests (`tier-small`, `tier-medium`, `tier-large`, `tier-xlarge`) instead of explicit phases. See `manifests/tier-*.json`.
> - GOAL.md reframed KV-quant comparison (TurboQuant vs FP8) as a deferred side investigation under M5, not the motivating question.
>
> Kept here as a reference for the original design intent and because the model-matrix metadata (`model_id`, `served_model_name`, `weight_quant`, `serve_args`, etc.) is still the schema used by manifests today.

## Model Matrix schema

Each model entry should declare:

- `model_id`: Hugging Face id or local path.
- `served_model_name`: OpenAI-compatible name.
- `openclaw_model_name`: optional OpenClaw route name or alias when it differs from the vLLM/OpenAI-compatible served name.
- `comparison_id`: optional stable model family id for FP8-vs-TurboQuant comparisons when served names or provider routes differ.
- `hardware_profile`: stable label for the local serve setup, such as GPU target, memory-utilization cap, eager/graph mode, and context plan.
- `weight_quant`: `bf16`, `fp8`, `nvfp4`, `awq`, etc.
- `serve_args`: vLLM args needed to load it.
- `expected_support`: known constraints such as attention sinks, hybrid model, sliding window.

For the current local-model question, keep two tracks:

### Track A: Desired Production Candidates

These are the models we actually care about for OpenClaw use:

- GPT-OSS 20B NVFP4
- Qwen3.6 NVFP4
- Gemma 4 NVFP4
- Nemotron NVFP4
- Qwen3 Coder / Qwen3-Next NVFP4

KV modes:

- `fp8`
- `turboquant_k8v4`
- `turboquant_k3v4_nc`

Unsupported combinations are recorded, not retried endlessly.

### Track B: TurboQuant Sanity Candidate

Add one dense, normal-attention Qwen model that current vLLM can serve with TurboQuant KV. This separates "TurboQuant is useful on this machine" from "our desired model architecture is blocked."

KV modes:

- `fp8`
- `turboquant_k8v4`
- `turboquant_k3v4_nc`
- optionally `turboquant_4bit_nc`

## Run Phases (original plan)

### Phase 0: Smoke

One model, one KV mode, one simple discovery task, concurrency 1.

Pass condition:

- OpenClaw can route to the local model.
- Agent can inspect the benchmark workspace.
- Result JSON is captured.

### Phase 1: Model/KV Support Probe

For each model/KV mode:

- attach to (or start) a server for the route
- send one tiny agent task
- record support or failure

No full benchmark until support is known. KV-mode probing applies only to runtimes where KV cache dtype is a startup parameter (vLLM, llama.cpp). For Ollama/LM Studio/hosted APIs, KV mode is recorded as `provider_default`.

### Phase 2: Quality Baseline

Run the full task suite at concurrency 1 and 4.

Purpose:

- identify quality regressions before spending time on high-concurrency tests

### Phase 3: Concurrency Sweep

Run selected tasks at:

```text
1, 2, 4, 8, 16, 32, 64
```

Purpose:

- identify usable concurrency range
- detect latency cliffs
- detect OOM thresholds

### Phase 4: Long-Context / Needle

Run workspace needle tasks at:

```text
4k, 8k, 16k, 32k, 64k
```

Purpose:

- detect KV quantization quality loss
- decide if memory savings are worth it

### Phase 5: Real Repo Tasks

Use local repos. Tasks should be read-only first, then patch tasks only on copied workspaces.

Purpose:

- catch behavior that synthetic fixtures miss

## Initial Recommendation (original plan, superseded)

> The recommendation below was the starting plan when the matrix shape was the primary question. GOAL.md M5 has since reframed KV-quant as a deferred side investigation.

Do not start with every desired model.

Start with:

1. GPT-OSS 20B NVFP4 + `fp8` KV as the current known working local candidate.
2. One dense Qwen model + `fp8`, `turboquant_k8v4`, `turboquant_k3v4_nc` as the TurboQuant sanity check.

Then decide:

- If dense Qwen TurboQuant fails quality, stop chasing TurboQuant for OpenClaw right now.
- If dense Qwen `k8v4` keeps quality and improves memory/concurrency, patch vLLM for Qwen3.6/Qwen3-Next hybrid TurboQuant next.
- GPT-OSS TurboQuant should wait until the attention-sink path is patched.
- Gemma 4 TurboQuant should wait until the Gemma/sliding-window path is patched.

## Target One-Line Command (aspirational)

The eventual local core-suite command should look like:

```bash
openclaw-bench run \
  --suite openclaw-agent-core \
  --models gpt-oss-20b-nvfp4,qwen3-dense \
  --kv fp8,turboquant_k8v4,turboquant_k3v4_nc \
  --concurrency 1,2,4,8,16,32,64 \
  --contexts 4096,8192,16384,32768,65536 \
  --out <bench-root>/results
```

That command should internally start the model server, run support probes, execute OpenClaw tasks, collect telemetry, and write the report. It is not by itself a certification-complete run; certification also requires local real-repo read-only and code-edit rows plus API and subscription provider rows with route-probe evidence.
