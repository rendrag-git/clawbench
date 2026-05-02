from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .aggregation import aggregate_attempts, summarize_reliability
from .models import AttemptResult


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_reports(out_dir: Path, results: list[AttemptResult], server: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [result.to_row() for result in results]
    write_jsonl(out_dir / "attempts.jsonl", rows)
    write_jsonl(out_dir / "failures.jsonl", [row for row in rows if row["status"] != "pass"])
    (out_dir / "server.json").write_text(json.dumps(server, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = summarize(results)
    reliability_cells = aggregate_attempts(rows)
    if any(c.n > 1 for c in reliability_cells):
        # Only emit reliability metrics when at least one cell has multiple seeds.
        # Single-run runs keep the existing summary shape unchanged.
        cell_rows = [c.to_row() for c in reliability_cells]
        write_jsonl(out_dir / "reliability.jsonl", cell_rows)
        summary["reliability"] = {
            "cells": cell_rows,
            "by_model": [
                {
                    "served_model_name": served_model_name,
                    "comparison_id": comparison_id,
                    "backend": backend,
                    "hardware_profile": hardware_profile,
                    "weight_quant": weight_quant,
                    "kv_cache_dtype": kv_cache_dtype,
                    **stats,
                }
                for (
                    served_model_name,
                    comparison_id,
                    backend,
                    hardware_profile,
                    weight_quant,
                    kv_cache_dtype,
                ), stats in summarize_reliability(reliability_cells).items()
            ],
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(render_summary_md(summary), encoding="utf-8")


def summarize(results: list[AttemptResult]) -> dict:
    groups: dict[tuple, list[AttemptResult]] = defaultdict(list)
    for result in results:
        comparison_id = result.comparison_id or result.model or result.served_model_name
        key = (
            result.served_model_name,
            comparison_id,
            result.backend,
            result.hardware_profile,
            result.weight_quant,
            result.kv_cache_dtype,
            result.context_limit,
            result.concurrency,
        )
        groups[key].append(result)

    rows = []
    for (model, comparison_id, backend, hardware_profile, weight, kv, context, concurrency), items in sorted(groups.items()):
        pass_rate = sum(1 for item in items if item.status == "pass") / len(items)
        patch_items = [item for item in items if "patch" in item.task_tags or item.task_type in {"multi_file_bug_trace", "patch_execution"}]
        retrieval_items = [item for item in items if "retrieval" in item.task_tags or "discovery" in item.task_tags]
        instruction_items = [item for item in items if "instruction" in item.task_tags or item.task_type == "instruction_retention"]
        needle_items = [item for item in items if "needle" in item.task_tags]
        wall_times = [item.wall_time_s for item in items]
        ttft_values = [item.ttft_s for item in items if item.ttft_s is not None]
        tool_calls = [float(item.tool_calls) for item in items]
        files_read = [float(item.files_read) for item in items]
        duplicate_file_reads = [float(item.duplicate_file_reads) for item in items if item.duplicate_file_reads is not None]
        first_relevant_file = [item.time_to_first_relevant_file_s for item in items if item.time_to_first_relevant_file_s is not None]
        p50_wall = _percentile(wall_times, 0.50)
        p95_wall = _percentile([item.wall_time_s for item in items], 0.95)
        p99_wall = _percentile(wall_times, 0.99)
        quality_score = _mean_score(items)
        latency_score = 0.0 if quality_score == 0 else _latency_score(p95_wall)
        usability_score = (
            0.45 * pass_rate
            + 0.20 * _mean_score(patch_items)
            + 0.15 * _mean_score(retrieval_items)
            + 0.10 * _mean_score(instruction_items)
            + 0.10 * latency_score
        )
        rows.append(
            {
                "model": model,
                "comparison_id": comparison_id,
                "backend": backend,
                "hardware_profile": hardware_profile,
                "weight_quant": weight,
                "kv_cache_dtype": kv,
                "context_limit": context,
                "concurrency": concurrency,
                "attempts": len(items),
                "pass_rate": round(pass_rate, 4),
                "absolute_usability_score": round(usability_score, 4),
                "quality_score": round(quality_score, 4),
                "patch_correctness": _mean_score(patch_items),
                "retrieval_accuracy": _mean_score(retrieval_items),
                "instruction_retention": _mean_score(instruction_items),
                "latency_score": round(latency_score, 4),
                "needle_rate": _rate(needle_items),
                "patch_rate": _rate(patch_items),
                "p50_wall_time_s": round(p50_wall, 3),
                "p95_wall_time_s": round(p95_wall, 3),
                "p99_wall_time_s": round(p99_wall, 3),
                "p50_ttft_s": _rounded_percentile(ttft_values, 0.50),
                "p95_ttft_s": _rounded_percentile(ttft_values, 0.95),
                "p99_ttft_s": _rounded_percentile(ttft_values, 0.99),
                "p50_tool_calls": _rounded_percentile(tool_calls, 0.50),
                "p95_tool_calls": _rounded_percentile(tool_calls, 0.95),
                "p50_files_read": _rounded_percentile(files_read, 0.50),
                "p95_files_read": _rounded_percentile(files_read, 0.95),
                "p95_duplicate_file_reads": _rounded_percentile(duplicate_file_reads, 0.95),
                "p95_time_to_first_relevant_file_s": _rounded_percentile(first_relevant_file, 0.95),
                "peak_vram_mb": _max_optional([item.peak_vram_mb for item in items]),
                "max_gpu_utilization_pct": _max_optional([item.gpu_utilization_pct for item in items]),
                "request_errors": sum(item.request_errors for item in items),
                "failures": sorted({item.failure_type for item in items if item.failure_type}),
            }
        )
    return {
        "attempts": len(results),
        "pass_rate": round(sum(1 for item in results if item.status == "pass") / len(results), 4) if results else 0.0,
        "groups": rows,
        "kv_acceptance": _kv_acceptance(rows),
        "decision_table": _decision_table(rows),
    }


def render_summary_md(summary: dict) -> str:
    lines = [
        "# OpenClaw Benchmark Summary",
        "",
        f"Attempts: {summary['attempts']}",
        f"Overall pass rate: {summary['pass_rate']:.2%}",
        "",
        "| Model | Backend | Hardware Setup | Weight | KV | Ctx | Concurrency | Usability | Pass % | Needle % | Patch % | P50 Wall | P95 Wall | P99 Wall | P95 Tools | P95 Files | P95 Dup Reads | P95 First File | Peak VRAM | GPU % | Notes |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["groups"]:
        notes = ", ".join(row["failures"]) if row["failures"] else ""
        display = {
            **row,
            "p95_tool_calls": "" if row["p95_tool_calls"] is None else row["p95_tool_calls"],
            "p95_files_read": "" if row["p95_files_read"] is None else row["p95_files_read"],
            "p95_duplicate_file_reads": "" if row["p95_duplicate_file_reads"] is None else row["p95_duplicate_file_reads"],
            "p95_time_to_first_relevant_file_s": "" if row["p95_time_to_first_relevant_file_s"] is None else row["p95_time_to_first_relevant_file_s"],
            "peak_vram": "" if row["peak_vram_mb"] is None else row["peak_vram_mb"],
            "gpu_util": "" if row["max_gpu_utilization_pct"] is None else row["max_gpu_utilization_pct"],
            "notes": notes,
        }
        lines.append(
            "| {model} | {backend} | {hardware_profile} | {weight_quant} | {kv_cache_dtype} | {context_limit} | {concurrency} | {absolute_usability_score:.1%} | {pass_rate:.1%} | {needle_rate:.1%} | {patch_rate:.1%} | {p50_wall_time_s:.3f} | {p95_wall_time_s:.3f} | {p99_wall_time_s:.3f} | {p95_tool_calls} | {p95_files_read} | {p95_duplicate_file_reads} | {p95_time_to_first_relevant_file_s} | {peak_vram} | {gpu_util} | {notes} |".format(
                **display,
            )
        )
    lines.extend(["", "## KV Acceptance", "", "| Model | Weight | KV | Ctx | Concurrency | Relative Quality | Threshold | Needle OK | Status |", "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |"])
    for row in summary["kv_acceptance"]:
        lines.append(
            "| {model} | {weight_quant} | {kv_cache_dtype} | {context_limit} | {concurrency} | {relative_quality_vs_fp8:.1%} | {threshold:.1%} | {needle_ok} | {acceptance_status} |".format(**row)
        )
    lines.extend(["", "## Decision Table", "", "| Use case | Best model/KV | Reason | Risk |", "| --- | --- | --- | --- |"])
    for row in summary["decision_table"]:
        lines.append(f"| {row['use_case']} | {row['best_model_kv']} | {row['reason']} | {row['risk']} |")
    reliability = summary.get("reliability")
    if reliability:
        lines.extend([
            "",
            "## Reliability (multi-seed)",
            "",
            "Per (model, hardware, KV) cell roll-up. `pass^k` = 1.0 means every seed of every cell passed; lower means at least one cell had a flaky or all-fail seed pattern. `worst-of-n` is the floor a user would actually feel. Hardware-distinct configs serving the same model stay on separate rows.",
            "",
            "| Model | Backend | Hardware | Weight | KV | Cells | All-pass | Flaky | All-fail | Load failed | Mean pass^k | Mean worst-of-n | Mean pass-rate |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in reliability["by_model"]:
            lines.append(
                "| {served_model_name} | {backend} | {hardware_profile} | {weight_quant} | {kv_cache_dtype} | {n_cells} | {n_all_pass} | {n_flaky} | {n_all_fail} | {n_load_failed} | {mean_pass_k:.3f} | {mean_worst_of_n:.3f} | {mean_pass_rate:.3f} |".format(**row)
            )
    return "\n".join(lines) + "\n"


def _rate(items: list[AttemptResult]) -> float:
    if not items:
        return 0.0
    return round(sum(1 for item in items if item.status == "pass") / len(items), 4)


def _mean_score(items: list[AttemptResult]) -> float:
    if not items:
        return 0.0
    return round(sum(item.score for item in items) / len(items), 4)


def _max_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return max(present)


def _latency_score(p95_wall_time_s: float) -> float:
    if p95_wall_time_s <= 0:
        return 1.0
    return max(0.0, min(1.0, 300.0 / p95_wall_time_s))


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * pct)))
    return ordered[index]


def _rounded_percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    return round(_percentile(values, pct), 3)


def _kv_acceptance(rows: list[dict]) -> list[dict]:
    by_baseline = {
        _baseline_key(row): row
        for row in rows
        if row["kv_cache_dtype"] == "fp8"
    }
    acceptance = []
    for row in rows:
        if row["kv_cache_dtype"] in {"fp8", "provider_default"}:
            continue
        baseline = by_baseline.get(_baseline_key(row))
        threshold = _kv_threshold(row["kv_cache_dtype"])
        if baseline is None or baseline["absolute_usability_score"] == 0:
            relative_quality = 0.0
            latency_delta = None
            needle_ok = False
            status = "missing_fp8_baseline"
            reason = "no fp8 baseline row for same model/weight/context/concurrency"
        else:
            relative_quality = row["absolute_usability_score"] / baseline["absolute_usability_score"]
            latency_delta = row["p95_wall_time_s"] - baseline["p95_wall_time_s"]
            needle_ok = row["needle_rate"] >= baseline["needle_rate"] and row["needle_rate"] > 0
            quality_ok = relative_quality >= threshold
            if not quality_ok:
                status = "fail"
                reason = "quality below fp8-relative threshold"
            elif not needle_ok:
                status = "fail"
                reason = "workspace needle regression versus fp8"
            elif row["backend"] == "simulator":
                status = "pending_live_benefit"
                reason = "simulator cannot prove latency or concurrency benefit"
            elif row["p95_wall_time_s"] >= baseline["p95_wall_time_s"] and row["concurrency"] <= baseline["concurrency"]:
                status = "fail"
                reason = "no measurable latency or concurrency benefit versus fp8"
            else:
                status = "pass"
                reason = "quality threshold met with measured latency/concurrency benefit"
        acceptance.append(
            {
                "model": row["model"],
                "backend": row["backend"],
                "weight_quant": row["weight_quant"],
                "kv_cache_dtype": row["kv_cache_dtype"],
                "context_limit": row["context_limit"],
                "concurrency": row["concurrency"],
                "relative_quality_vs_fp8": round(relative_quality, 4),
                "latency_delta_vs_fp8_s": None if latency_delta is None else round(latency_delta, 3),
                "threshold": threshold,
                "needle_ok": needle_ok,
                "acceptance_status": status,
                "acceptance_reason": reason,
            }
        )
    return acceptance


def _baseline_key(row: dict) -> tuple:
    return (
        row["comparison_id"],
        row["backend"],
        row["hardware_profile"],
        row["weight_quant"],
        row["context_limit"],
        row["concurrency"],
    )


def _kv_threshold(kv_cache_dtype: str) -> float:
    if kv_cache_dtype == "turboquant_k8v4":
        return 0.95
    if kv_cache_dtype == "turboquant_k3v4_nc":
        return 0.90
    return 1.0


def _decision_table(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    single_agent_rows = _lowest_context_rows([row for row in rows if row["concurrency"] == 1])
    background_rows, background_risk = _target_concurrency_rows(rows, 4)
    long_context_rows = _lowest_concurrency_rows(_max_context_rows(rows))
    stress_rows = [row for row in rows if row["concurrency"] == 64] or [row for row in rows if row["concurrency"] > 64]
    return [
        _decision_row(
            "single-agent coding",
            single_agent_rows,
            lambda row: (row["patch_correctness"], row["patch_rate"], row["instruction_retention"], row["absolute_usability_score"], -row["request_errors"], -row["p95_wall_time_s"]),
            missing_risk="" if single_agent_rows else "No concurrency-1 row in this run",
        ),
        _decision_row(
            "background agent pool",
            background_rows,
            lambda row: (row["absolute_usability_score"], row["pass_rate"], -row["request_errors"], -row["p95_wall_time_s"]),
            missing_risk=background_risk,
        ),
        _decision_row(
            "long-context repo search",
            long_context_rows,
            lambda row: (row["retrieval_accuracy"], row["needle_rate"], row["instruction_retention"], row["absolute_usability_score"], -row["p95_wall_time_s"]),
        ),
        _decision_row(
            "64-concurrency stress",
            stress_rows,
            lambda row: (row["pass_rate"], row["absolute_usability_score"], -row["request_errors"], -row["p95_wall_time_s"]),
            missing_risk="" if stress_rows else "No 64-concurrency row in this run",
        ),
    ]


def _lowest_context_rows(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    lowest = min(row["context_limit"] for row in rows)
    return [row for row in rows if row["context_limit"] == lowest]


def _max_context_rows(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    highest = max(row["context_limit"] for row in rows)
    return [row for row in rows if row["context_limit"] == highest]


def _lowest_concurrency_rows(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    lowest = min(row["concurrency"] for row in rows)
    return [row for row in rows if row["concurrency"] == lowest]


def _target_concurrency_rows(rows: list[dict], target: int) -> tuple[list[dict], str]:
    exact = [row for row in rows if row["concurrency"] == target]
    if exact:
        return exact, ""
    higher_levels = sorted({row["concurrency"] for row in rows if row["concurrency"] > target})
    if higher_levels:
        level = higher_levels[0]
        return [row for row in rows if row["concurrency"] == level], f"No {target}-concurrency row in this run"
    return [], f"No {target}-concurrency row in this run"


def _decision_row(use_case: str, rows: list[dict], key_fn, missing_risk: str = "") -> dict:
    if not rows:
        return {
            "use_case": use_case,
            "best_model_kv": "none",
            "reason": "no model/KV cell matched this use case",
            "risk": missing_risk or "missing required benchmark row",
        }
    best = max(rows, key=key_fn)
    if best["absolute_usability_score"] <= 0:
        return {
            "use_case": use_case,
            "best_model_kv": "none",
            "reason": "no model/KV cell produced usable task results",
            "risk": ", ".join(best["failures"]) if best["failures"] else "all candidates failed",
        }
    risk_parts = []
    if best["backend"] == "simulator":
        risk_parts.append("Needs live OpenClaw/model validation")
    if missing_risk:
        risk_parts.append(missing_risk)
    return {
        "use_case": use_case,
        "best_model_kv": f"{best['model']} / {best['kv_cache_dtype']}",
        "reason": f"{best['absolute_usability_score']:.1%} usability at concurrency {best['concurrency']}",
        "risk": "; ".join(risk_parts),
    }
