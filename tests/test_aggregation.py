import unittest

from openclaw_bench.aggregation import (
    aggregate_attempts,
    summarize_reliability,
)


def _attempt(
    *,
    model="m",
    kv="fp8",
    context=4096,
    concurrency=1,
    task_id="t1",
    score=1.0,
    status="pass",
    failure_type=None,
    run_index=0,
):
    return {
        "model": model,
        "kv_cache_dtype": kv,
        "context_limit": context,
        "concurrency": concurrency,
        "task_id": task_id,
        "score": score,
        "status": status,
        "failure_type": failure_type,
        "run_index": run_index,
    }


class AggregateAttemptsTests(unittest.TestCase):
    def test_empty_input_yields_empty_output(self):
        self.assertEqual(aggregate_attempts([]), [])

    def test_single_seed_per_cell_passes_through(self):
        rows = [_attempt(score=1.0, status="pass")]
        cells = aggregate_attempts(rows)
        self.assertEqual(len(cells), 1)
        cell = cells[0]
        self.assertEqual(cell.n, 1)
        self.assertEqual(cell.pass_k, 1.0)
        self.assertEqual(cell.worst_of_n, 1.0)
        self.assertEqual(cell.best_of_n, 1.0)
        self.assertEqual(cell.cell_status, "all_pass")

    def test_all_seeds_pass_yields_pass_k_1(self):
        rows = [
            _attempt(run_index=0, score=1.0, status="pass"),
            _attempt(run_index=1, score=1.0, status="pass"),
            _attempt(run_index=2, score=1.0, status="pass"),
        ]
        cells = aggregate_attempts(rows)
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0].pass_k, 1.0)
        self.assertEqual(cells[0].pass_rate, 1.0)
        self.assertEqual(cells[0].cell_status, "all_pass")
        self.assertEqual(cells[0].n, 3)

    def test_one_seed_fails_yields_pass_k_0_and_flaky(self):
        rows = [
            _attempt(run_index=0, score=1.0, status="pass"),
            _attempt(run_index=1, score=0.2, status="fail", failure_type="bad_json"),
            _attempt(run_index=2, score=1.0, status="pass"),
        ]
        cells = aggregate_attempts(rows)
        cell = cells[0]
        self.assertEqual(cell.pass_k, 0.0)  # not every seed passed
        self.assertAlmostEqual(cell.pass_rate, 2 / 3)
        self.assertEqual(cell.worst_of_n, 0.2)
        self.assertEqual(cell.best_of_n, 1.0)
        self.assertEqual(cell.cell_status, "flaky")
        self.assertEqual(cell.failure_types, {"bad_json": 1})

    def test_all_seeds_fail_yields_all_fail_status(self):
        rows = [
            _attempt(run_index=0, score=0.0, status="fail", failure_type="wrong_file"),
            _attempt(run_index=1, score=0.5, status="fail", failure_type="wrong_file"),
        ]
        cells = aggregate_attempts(rows)
        cell = cells[0]
        self.assertEqual(cell.pass_k, 0.0)
        self.assertEqual(cell.pass_rate, 0.0)
        self.assertEqual(cell.cell_status, "all_fail")
        self.assertEqual(cell.failure_types["wrong_file"], 2)

    def test_load_failure_only_yields_load_failed_status(self):
        rows = [
            _attempt(run_index=0, score=0.0, status="fail", failure_type="model_load_failed"),
            _attempt(run_index=1, score=0.0, status="fail", failure_type="model_load_failed"),
        ]
        cells = aggregate_attempts(rows)
        self.assertEqual(cells[0].cell_status, "load_failed")

    def test_separate_cells_grouped_by_full_key(self):
        rows = [
            _attempt(model="m1", task_id="ta", score=1.0, status="pass"),
            _attempt(model="m1", task_id="tb", score=0.0, status="fail"),
            _attempt(model="m2", task_id="ta", score=0.7, status="fail"),
        ]
        cells = aggregate_attempts(rows)
        self.assertEqual(len(cells), 3)
        keyed = {(c.model, c.task_id): c for c in cells}
        self.assertEqual(keyed[("m1", "ta")].cell_status, "all_pass")
        self.assertEqual(keyed[("m1", "tb")].cell_status, "all_fail")
        self.assertEqual(keyed[("m2", "ta")].cell_status, "all_fail")


class SummarizeReliabilityTests(unittest.TestCase):
    def test_empty_cells_returns_empty(self):
        self.assertEqual(summarize_reliability([]), {})

    def test_groups_by_model_kv_and_counts_statuses(self):
        rows = [
            _attempt(model="m1", kv="fp8", task_id="t1", run_index=0, score=1.0, status="pass"),
            _attempt(model="m1", kv="fp8", task_id="t1", run_index=1, score=1.0, status="pass"),
            _attempt(model="m1", kv="fp8", task_id="t2", run_index=0, score=0.5, status="fail", failure_type="wrong_file"),
            _attempt(model="m1", kv="fp8", task_id="t2", run_index=1, score=1.0, status="pass"),
            _attempt(model="m1", kv="fp8", task_id="t3", run_index=0, score=0.0, status="fail", failure_type="bad_json"),
            _attempt(model="m1", kv="fp8", task_id="t3", run_index=1, score=0.0, status="fail", failure_type="bad_json"),
        ]
        cells = aggregate_attempts(rows)
        summary = summarize_reliability(cells)
        key = ("m1", "fp8")
        self.assertIn(key, summary)
        self.assertEqual(summary[key]["n_cells"], 3)
        self.assertEqual(summary[key]["n_all_pass"], 1)
        self.assertEqual(summary[key]["n_flaky"], 1)
        self.assertEqual(summary[key]["n_all_fail"], 1)
        # mean_pass_k = (1.0 + 0.0 + 0.0) / 3 = 0.3333
        self.assertAlmostEqual(summary[key]["mean_pass_k"], 1 / 3, places=3)


if __name__ == "__main__":
    unittest.main()
