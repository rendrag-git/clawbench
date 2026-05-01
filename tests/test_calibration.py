import json
import tempfile
import unittest
from pathlib import Path

from openclaw_bench.calibration import load_calibration_records, validate_record_against_result_root


class CalibrationTests(unittest.TestCase):
    def test_load_complete_calibration_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            path.write_text(json.dumps(_payload()), encoding="utf-8")
            calibration = load_calibration_records(path)

        self.assertEqual(len(calibration.records), 8)
        self.assertEqual(calibration.floor_min_score, 0.9)
        self.assertEqual(calibration.ceiling_max_score, 0.3)

    def test_missing_tier_role_pair_fails_complete_validation(self):
        payload = _payload()
        payload["records"] = payload["records"][:-1]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing calibration records"):
                load_calibration_records(path)

    def test_threshold_violations_fail(self):
        payload = _payload()
        payload["records"][0]["score"] = 0.89
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "floor score"):
                load_calibration_records(path)

    def test_rejects_model_manifest_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            path.write_text(json.dumps({"schema_version": 1, "models": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not be embedded under models"):
                load_calibration_records(path)

    def test_recomputes_record_score_from_attempts(self):
        payload = _payload()
        record = payload["records"][0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration_path = root / "calibration.json"
            calibration_path.write_text(json.dumps(payload), encoding="utf-8")
            attempts_dir = root / record["run_id"]
            attempts_dir.mkdir()
            rows = [
                {**_attempt_row(record), "score": 1.0},
                {**_attempt_row(record), "score": 0.8},
            ]
            (attempts_dir / "attempts.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            calibration = load_calibration_records(calibration_path)
            validate_record_against_result_root(calibration.records[0], root)

    def test_recomputed_score_mismatch_fails(self):
        payload = _payload()
        record = payload["records"][0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration_path = root / "calibration.json"
            calibration_path.write_text(json.dumps(payload), encoding="utf-8")
            attempts_dir = root / record["run_id"]
            attempts_dir.mkdir()
            (attempts_dir / "attempts.jsonl").write_text(json.dumps({**_attempt_row(record), "score": 1.0}) + "\n", encoding="utf-8")

            calibration = load_calibration_records(calibration_path)
            with self.assertRaisesRegex(ValueError, "score mismatch"):
                validate_record_against_result_root(calibration.records[0], root)


def _payload():
    records = []
    for tier in ("small", "medium", "large", "xlarge"):
        records.append(_record(tier, "floor", 0.9))
        records.append(_record(tier, "ceiling", 0.3))
    return {
        "schema_version": 1,
        "score_metric": "quality_score",
        "thresholds": {"floor_min_score": 0.9, "ceiling_max_score": 0.3},
        "records": records,
    }


def _record(tier, role, score):
    return {
        "tier": tier,
        "role": role,
        "suite_id": f"tier-{tier}",
        "run_id": f"live-m2-{tier}-{role}",
        "commit": "80309c0",
        "backend": "openclaw",
        "model_id": f"model-{tier}-{role}",
        "served_model_name": f"served-{tier}-{role}",
        "provider_type": "local",
        "hardware_profile": "bench",
        "weight_quant": "bf16",
        "kv_cache_dtype": "provider_default",
        "context_limit": 32768,
        "concurrency": 1,
        "score": score,
        "date": "2026-05-01",
    }


def _attempt_row(record):
    return {
        "run_id": record["run_id"],
        "model": record["model_id"],
        "served_model_name": record["served_model_name"],
        "backend": record["backend"],
        "provider_type": record["provider_type"],
        "hardware_profile": record["hardware_profile"],
        "weight_quant": record["weight_quant"],
        "kv_cache_dtype": record["kv_cache_dtype"],
        "context_limit": record["context_limit"],
        "concurrency": record["concurrency"],
    }


if __name__ == "__main__":
    unittest.main()
