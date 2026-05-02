"""Reliability aggregation across multi-seed attempts.

Pure functions over `AttemptResult.to_row()` dicts. Used when `--runs-per-task > 1`
to compute pass^k (all-pass), worst-of-n, mean, pass-rate, and the failure-type
distribution per (served_model_name, comparison_id, backend, hardware_profile,
weight_quant, kv_cache_dtype, context_limit, concurrency, task_id) cell.

The cell key matches the 8-dimensional identity that `reporting.summarize` uses
across the rest of the summary, plus `task_id` so cells are per-task. Aggregating
on a narrower key (e.g. just `model`) would merge attempts from different
hardware profiles or backends into one cell, inflating `n` and misclassifying
stable configs as flaky.

Adopted pattern: openclaw/clawbench upstream (CLAWBENCH_V0_4_SPEC.md §reliability).
The principle: a model that scores 90 % on one seed and 20 % on the next is not a
55 % model — it is an unreliable model. Users experience the worst seed, not the
average. pass^k makes that explicit.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


# Failure types that mean the model never started this cell at all. When every
# attempt in a cell has one of these, reliability metrics are not meaningful and
# the cell is reported as `cell_status="load_failed"` instead.
LOAD_FAILURE_TYPES = frozenset(
    {
        "model_load_failed",
        "unsupported_kv_dtype",
        "oom_on_load",
        "model_route_failed",
        "tool_parser_missing",
        "serve_probe_failed",
    }
)


# Cell key dimensions, in order. Mirrors the identity reporting.summarize uses
# (served_model_name, comparison_id, backend, hardware_profile, weight_quant,
# kv_cache_dtype, context_limit, concurrency) plus task_id for per-task cells.
_CELL_KEY_FIELDS = (
    "served_model_name",
    "comparison_id",
    "backend",
    "hardware_profile",
    "weight_quant",
    "kv_cache_dtype",
    "context_limit",
    "concurrency",
    "task_id",
)


@dataclass(frozen=True)
class CellReliability:
    """Reliability metrics for one runtime+config+task cell.

    The full identity (`served_model_name, comparison_id, backend, hardware_profile,
    weight_quant, kv_cache_dtype, context_limit, concurrency, task_id`) is carried
    so cells from different hardware setups serving the same `served_model_name`
    do not merge.
    """

    served_model_name: str
    comparison_id: str
    backend: str
    hardware_profile: str
    weight_quant: str
    kv_cache_dtype: str
    context_limit: int
    concurrency: int
    task_id: str
    n: int  # number of attempts (seeds) in this cell
    pass_k: float  # 1.0 iff every attempt passed; 0.0 otherwise
    worst_of_n: float  # min(score across attempts)
    best_of_n: float  # max(score across attempts)
    mean_score: float  # arithmetic mean
    pass_rate: float  # fraction of attempts with status == "pass"
    cell_status: str  # "all_pass" | "all_fail" | "flaky" | "load_failed"
    failure_types: dict[str, int]  # Counter of non-null failure_type values

    def to_row(self) -> dict:
        return {
            "served_model_name": self.served_model_name,
            "comparison_id": self.comparison_id,
            "backend": self.backend,
            "hardware_profile": self.hardware_profile,
            "weight_quant": self.weight_quant,
            "kv_cache_dtype": self.kv_cache_dtype,
            "context_limit": self.context_limit,
            "concurrency": self.concurrency,
            "task_id": self.task_id,
            "n": self.n,
            "pass_k": round(self.pass_k, 4),
            "worst_of_n": round(self.worst_of_n, 4),
            "best_of_n": round(self.best_of_n, 4),
            "mean_score": round(self.mean_score, 4),
            "pass_rate": round(self.pass_rate, 4),
            "cell_status": self.cell_status,
            "failure_types": dict(self.failure_types),
        }


def _cell_key(row: dict) -> tuple:
    return (
        str(row.get("served_model_name", "")),
        # comparison_id falls back to model id then served_model_name, matching
        # AttemptResult.to_row's normalization.
        str(row.get("comparison_id") or row.get("model") or row.get("served_model_name") or ""),
        str(row.get("backend", "")),
        str(row.get("hardware_profile", "")),
        str(row.get("weight_quant", "")),
        str(row.get("kv_cache_dtype", "")),
        int(row.get("context_limit", 0)),
        int(row.get("concurrency", 0)),
        str(row.get("task_id", "")),
    )


def _classify_cell(statuses: list[str], failure_types: Counter) -> str:
    pass_count = statuses.count("pass")
    if pass_count == len(statuses):
        return "all_pass"
    if pass_count == 0:
        # Load-failed only when at least one failure_type is recorded and
        # every recorded failure_type is in the load-failure set. An empty
        # failure_types Counter (no failure_type fields populated) does not
        # qualify — that's "all_fail" with missing classification.
        if failure_types and all(ft in LOAD_FAILURE_TYPES for ft in failure_types):
            return "load_failed"
        return "all_fail"
    return "flaky"


def aggregate_attempts(rows: list[dict]) -> list[CellReliability]:
    """Group attempt rows into cells and compute reliability metrics per cell.

    Each input row is a dict produced by `AttemptResult.to_row()`. Cells are keyed
    on the full runtime+config identity (see `_CELL_KEY_FIELDS`) plus `task_id`.
    Output is one `CellReliability` per cell.
    """
    if not rows:
        return []

    cells: dict[tuple, list[dict]] = {}
    for row in rows:
        cells.setdefault(_cell_key(row), []).append(row)

    out: list[CellReliability] = []
    for key, cell_rows in sorted(cells.items()):
        (
            served_model_name,
            comparison_id,
            backend,
            hardware_profile,
            weight_quant,
            kv_cache_dtype,
            context_limit,
            concurrency,
            task_id,
        ) = key
        scores = [float(r.get("score", 0.0) or 0.0) for r in cell_rows]
        statuses = [str(r.get("status", "fail")) for r in cell_rows]
        failure_types = Counter(
            str(r["failure_type"])
            for r in cell_rows
            if r.get("failure_type")
        )
        n = len(cell_rows)
        pass_count = statuses.count("pass")
        cell = CellReliability(
            served_model_name=served_model_name,
            comparison_id=comparison_id,
            backend=backend,
            hardware_profile=hardware_profile,
            weight_quant=weight_quant,
            kv_cache_dtype=kv_cache_dtype,
            context_limit=context_limit,
            concurrency=concurrency,
            task_id=task_id,
            n=n,
            pass_k=1.0 if pass_count == n else 0.0,
            worst_of_n=min(scores),
            best_of_n=max(scores),
            mean_score=sum(scores) / n,
            pass_rate=pass_count / n,
            cell_status=_classify_cell(statuses, failure_types),
            failure_types=failure_types,
        )
        out.append(cell)
    return out


def summarize_reliability(cells: list[CellReliability]) -> dict:
    """Roll cell metrics up to a (served_model_name, comparison_id, backend,
    hardware_profile, weight_quant, kv_cache_dtype) level summary for the
    decision table.

    The rollup key matches the 6-dimensional identity that `reporting.summarize`
    uses for its main per-row groupings (the runtime+config identity, without
    context, concurrency, or task). Hardware-distinct configs serving the same
    `served_model_name` stay separate rows.

    Returns: {(served_model_name, comparison_id, backend, hardware_profile,
              weight_quant, kv_cache_dtype): {n_cells, n_all_pass, n_all_fail,
              n_flaky, n_load_failed, mean_pass_k, mean_worst_of_n, mean_pass_rate}}
    """
    if not cells:
        return {}
    by_model: dict[tuple, list[CellReliability]] = {}
    for cell in cells:
        rollup_key = (
            cell.served_model_name,
            cell.comparison_id,
            cell.backend,
            cell.hardware_profile,
            cell.weight_quant,
            cell.kv_cache_dtype,
        )
        by_model.setdefault(rollup_key, []).append(cell)
    summary: dict = {}
    for rollup_key, group in sorted(by_model.items()):
        n = len(group)
        summary[rollup_key] = {
            "n_cells": n,
            "n_all_pass": sum(1 for c in group if c.cell_status == "all_pass"),
            "n_all_fail": sum(1 for c in group if c.cell_status == "all_fail"),
            "n_flaky": sum(1 for c in group if c.cell_status == "flaky"),
            "n_load_failed": sum(1 for c in group if c.cell_status == "load_failed"),
            "mean_pass_k": round(sum(c.pass_k for c in group) / n, 4),
            "mean_worst_of_n": round(sum(c.worst_of_n for c in group) / n, 4),
            "mean_pass_rate": round(sum(c.pass_rate for c in group) / n, 4),
        }
    return summary
