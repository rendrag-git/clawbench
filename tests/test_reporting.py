import unittest

from openclaw_bench.models import AttemptResult
from openclaw_bench.reporting import summarize


class ReportingTests(unittest.TestCase):
    def test_kv_acceptance_pairs_same_model_against_fp8_baseline(self):
        results = []
        results.extend(_group("model-a", "fp8", 1.0, status="pass"))
        results.extend(_group("model-a", "turboquant_k8v4", 0.85, status="pass"))
        results.extend(_group("model-b", "fp8", 0.5, status="pass"))
        results.extend(_group("model-b", "turboquant_k8v4", 0.5, status="pass"))

        summary = summarize(results)
        acceptance = {
            (row["model"], row["kv_cache_dtype"]): row
            for row in summary["kv_acceptance"]
        }

        self.assertEqual(acceptance[("model-a", "turboquant_k8v4")]["acceptance_status"], "fail")
        self.assertLess(acceptance[("model-a", "turboquant_k8v4")]["relative_quality_vs_fp8"], 0.95)
        self.assertEqual(acceptance[("model-b", "turboquant_k8v4")]["acceptance_status"], "pending_live_benefit")

    def test_needle_regression_rejects_kv_mode(self):
        baseline = _group("model-a", "fp8", 1.0, status="pass")
        kv = _group("model-a", "turboquant_k3v4_nc", 1.0, status="pass")
        kv[-1].status = "fail"
        kv[-1].failure_type = "wrong_needle"

        summary = summarize(baseline + kv)
        acceptance = summary["kv_acceptance"][0]

        self.assertEqual(acceptance["acceptance_status"], "fail")
        self.assertEqual(acceptance["acceptance_reason"], "workspace needle regression versus fp8")

    def test_kv_acceptance_uses_comparison_id_when_served_names_differ(self):
        baseline = _group(
            "qwen3-dense-vllm-fp8",
            "fp8",
            1.0,
            status="pass",
            model_id="RedHatAI/Qwen3.6-35B-A3B-NVFP4",
        )
        kv = _group(
            "qwen3-dense-vllm-k8v4",
            "turboquant_k8v4",
            1.0,
            status="pass",
            model_id="RedHatAI/Qwen3.6-35B-A3B-NVFP4",
        )

        summary = summarize(baseline + kv)
        acceptance = summary["kv_acceptance"][0]

        self.assertEqual(acceptance["model"], "qwen3-dense-vllm-k8v4")
        self.assertEqual(acceptance["relative_quality_vs_fp8"], 1.0)
        self.assertEqual(acceptance["acceptance_status"], "pending_live_benefit")

    def test_provider_default_participates_in_decisions_without_kv_acceptance(self):
        results = _group("openai/gpt-4.1", "provider_default", 1.0, status="pass")
        for result in results:
            result.provider_type = "api"
            result.backend = "openclaw"

        summary = summarize(results)
        table = {row["use_case"]: row for row in summary["decision_table"]}

        self.assertEqual(summary["kv_acceptance"], [])
        self.assertEqual(table["single-agent coding"]["best_model_kv"], "openai/gpt-4.1 / provider_default")
        self.assertEqual(table["single-agent coding"]["risk"], "")

    def test_model_load_failures_have_zero_usability(self):
        results = _group("model-a", "fp8", 0.0, status="fail")
        for result in results:
            result.failure_type = "model_load_failed"
            result.wall_time_s = 0.0

        summary = summarize(results)
        group = summary["groups"][0]

        self.assertEqual(group["latency_score"], 0.0)
        self.assertEqual(group["absolute_usability_score"], 0.0)
        self.assertTrue(all(row["best_model_kv"] == "none" for row in summary["decision_table"]))
        self.assertTrue(any(row["risk"] == "model_load_failed" for row in summary["decision_table"]))

    def test_route_failures_have_zero_usability(self):
        results = _group("model-a", "fp8", 0.0, status="fail")
        for result in results:
            result.failure_type = "openclaw_timeout"
            result.wall_time_s = 0.0

        summary = summarize(results)
        group = summary["groups"][0]

        self.assertEqual(group["latency_score"], 0.0)
        self.assertEqual(group["absolute_usability_score"], 0.0)
        self.assertTrue(all(row["best_model_kv"] == "none" for row in summary["decision_table"]))
        self.assertTrue(any(row["risk"] == "openclaw_timeout" for row in summary["decision_table"]))

    def test_decision_table_selects_winners_per_use_case(self):
        results = []
        results.extend(
            _group(
                "single-coder",
                "fp8",
                0.4,
                status="pass",
                task_scores={
                    "multi-file-bug-trace": 1.0,
                    "patch-execution": 1.0,
                    "instruction-retention": 1.0,
                },
            )
        )
        results.extend(_group("pool-runner", "fp8", 0.95, status="pass", concurrency=8))
        results.extend(
            _group(
                "long-context",
                "fp8",
                0.4,
                status="pass",
                context_limit=32768,
                task_scores={
                    "workspace-discovery": 1.0,
                    "workspace-needle-4k": 1.0,
                },
            )
        )
        results.extend(_group("stress-runner", "fp8", 0.9, status="pass", concurrency=64))

        summary = summarize(results)
        table = {row["use_case"]: row for row in summary["decision_table"]}

        self.assertEqual(table["single-agent coding"]["best_model_kv"], "single-coder / fp8")
        self.assertEqual(table["background agent pool"]["best_model_kv"], "pool-runner / fp8")
        self.assertEqual(table["long-context repo search"]["best_model_kv"], "long-context / fp8")
        self.assertEqual(table["64-concurrency stress"]["best_model_kv"], "stress-runner / fp8")

    def test_background_pool_prefers_exact_four_concurrency(self):
        results = []
        results.extend(_group("four-agent", "fp8", 0.8, status="pass", concurrency=4))
        results.extend(_group("stress-runner", "fp8", 1.0, status="pass", concurrency=64))

        summary = summarize(results)
        table = {row["use_case"]: row for row in summary["decision_table"]}

        self.assertEqual(table["background agent pool"]["best_model_kv"], "four-agent / fp8")

    def test_stress_decision_requires_64_concurrency_row(self):
        results = []
        results.extend(_group("eight-agent", "fp8", 1.0, status="pass", concurrency=8))
        results.extend(_group("sixteen-agent", "fp8", 1.0, status="pass", concurrency=16))

        summary = summarize(results)
        table = {row["use_case"]: row for row in summary["decision_table"]}

        self.assertEqual(table["64-concurrency stress"]["best_model_kv"], "none")
        self.assertIn("No 64-concurrency row", table["64-concurrency stress"]["risk"])

    def test_summary_reports_latency_and_tool_efficiency_percentiles(self):
        results = _group("model-a", "fp8", 1.0, status="pass")
        for index, result in enumerate(results, start=1):
            result.wall_time_s = float(index)
            result.ttft_s = float(index) / 10
            result.tool_calls = index
            result.files_read = index + 10
            result.duplicate_file_reads = index - 1
            result.time_to_first_relevant_file_s = float(index) / 2

        summary = summarize(results)
        group = summary["groups"][0]

        self.assertEqual(group["p50_wall_time_s"], 3.0)
        self.assertEqual(group["p95_wall_time_s"], 5.0)
        self.assertEqual(group["p99_wall_time_s"], 5.0)
        self.assertEqual(group["p50_ttft_s"], 0.3)
        self.assertEqual(group["p95_tool_calls"], 5.0)
        self.assertEqual(group["p95_files_read"], 15.0)
        self.assertEqual(group["p95_duplicate_file_reads"], 4.0)
        self.assertEqual(group["p95_time_to_first_relevant_file_s"], 2.5)


def _group(
    model: str,
    kv: str,
    score: float,
    status: str,
    model_id: str | None = None,
    context_limit: int = 4096,
    concurrency: int = 1,
    task_scores: dict[str, float] | None = None,
) -> list[AttemptResult]:
    specs = [
        ("workspace-discovery", "workspace_discovery", ["discovery"]),
        ("multi-file-bug-trace", "multi_file_bug_trace", ["patch", "reasoning"]),
        ("patch-execution", "patch_execution", ["patch"]),
        ("instruction-retention", "instruction_retention", ["instruction", "patch"]),
        ("workspace-needle-4k", "workspace_needle", ["needle", "retrieval"]),
    ]
    task_scores = task_scores or {}
    results = []
    for task_id, task_type, tags in specs:
        task_score = task_scores.get(task_id, score)
        task_status = status if task_score > 0 else "fail"
        results.append(
            AttemptResult(
                run_id="test",
                model=model_id or model,
                served_model_name=model,
                comparison_id=model_id or model,
                backend="simulator",
                provider_type="local",
                hardware_profile="default",
                weight_quant="bf16",
                kv_cache_dtype=kv,
                context_limit=context_limit,
                concurrency=concurrency,
                task_id=task_id,
                task_type=task_type,
                task_tags=tags,
                workspace_id=f"{model}-{kv}-{task_id}",
                status=task_status,
                score=task_score,
                wall_time_s=1.0,
                ttft_s=None,
                tool_calls=1,
                files_read=1,
                duplicate_file_reads=0,
                time_to_first_relevant_file_s=0.1,
                files_changed=0,
                changed_files=[],
                tests_passed=task_status == "pass",
                json_valid=True,
                hallucinated_paths=0,
                oom=False,
                timeout=False,
                peak_vram_mb=123,
                gpu_utilization_pct=45,
            )
        )
    return results


if __name__ == "__main__":
    unittest.main()
