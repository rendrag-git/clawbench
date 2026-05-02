# Simulator (mechanics smoke)

Use the simulator backend to validate harness changes — scoring, workspace isolation, report generation. Do not treat simulator output as model certification.

`<bench-root>` is the benchmark root you passed to `oc-bench init` (or wherever you want results written).

## Core suite

```bash
python3 -m openclaw_bench run \
  --backend simulator \
  --suite manifests/openclaw-agent-core.json \
  --model-config manifests/initial-models.json \
  --out <bench-root>/results
```

## Real-repo suite

Run this too when changing scoring, workspace isolation, or report generation. It includes read-only tasks plus a copied-workspace code-edit task:

```bash
python3 -m openclaw_bench run \
  --backend simulator \
  --suite manifests/real-repo-readonly.example.json \
  --models simulated-model \
  --kv fp8 \
  --concurrency 1 \
  --contexts 4096 \
  --out <bench-root>/results
```

The simulator full-suite should produce `40 attempts / 0 failures`.
