from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


TIERS = ("small", "medium", "large", "xlarge")
ROLES = ("floor", "ceiling")
REQUIRED_RECORD_FIELDS = {
    "tier",
    "role",
    "suite_id",
    "run_id",
    "commit",
    "backend",
    "model_id",
    "served_model_name",
    "provider_type",
    "hardware_profile",
    "weight_quant",
    "kv_cache_dtype",
    "context_limit",
    "concurrency",
    "score",
    "date",
}


@dataclass(frozen=True)
class CalibrationRecord:
    tier: str
    role: str
    suite_id: str
    run_id: str
    commit: str
    backend: str
    model_id: str
    served_model_name: str
    provider_type: str
    hardware_profile: str
    weight_quant: str
    kv_cache_dtype: str
    context_limit: int
    concurrency: int
    score: float
    date: str
    result_dir: str | None = None


@dataclass(frozen=True)
class CalibrationSet:
    schema_version: int
    score_metric: str
    floor_min_score: float
    ceiling_max_score: float
    records: list[CalibrationRecord]


def load_calibration_records(path: Path, *, require_complete: bool = True) -> CalibrationSet:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("calibration file must be a JSON object")
    if "models" in data:
        raise ValueError("calibration records must not be embedded under models")
    if data.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if data.get("score_metric") != "quality_score":
        raise ValueError("score_metric must be quality_score")

    thresholds = data.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("thresholds must be an object")
    floor_min_score = _required_score(thresholds, "floor_min_score")
    ceiling_max_score = _required_score(thresholds, "ceiling_max_score")

    raw_records = data.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("records must be a list")
    records = [_parse_record(item, floor_min_score, ceiling_max_score) for item in raw_records]
    _validate_pairs(records, require_complete=require_complete)
    return CalibrationSet(
        schema_version=1,
        score_metric="quality_score",
        floor_min_score=floor_min_score,
        ceiling_max_score=ceiling_max_score,
        records=records,
    )


def validate_record_against_result_root(record: CalibrationRecord, result_root: Path, *, tolerance: float = 0.0001) -> None:
    attempts_path = result_root / record.run_id / "attempts.jsonl"
    if not attempts_path.is_file():
        raise ValueError(f"missing attempts.jsonl for run_id {record.run_id}")
    matching_scores = []
    for line in attempts_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if _row_matches_record(row, record):
            matching_scores.append(float(row["score"]))
    if not matching_scores:
        raise ValueError(f"no attempts matched calibration record {record.tier}/{record.role}")
    observed = round(sum(matching_scores) / len(matching_scores), 4)
    if abs(observed - record.score) > tolerance:
        raise ValueError(f"calibration score mismatch: record {record.score} != attempts {observed}")


def _parse_record(item: object, floor_min_score: float, ceiling_max_score: float) -> CalibrationRecord:
    if not isinstance(item, dict):
        raise ValueError("calibration record must be an object")
    missing = sorted(REQUIRED_RECORD_FIELDS - set(item))
    if missing:
        raise ValueError("calibration record missing " + ", ".join(missing))

    tier = _required_string(item, "tier")
    role = _required_string(item, "role")
    if tier not in TIERS:
        raise ValueError(f"invalid tier: {tier}")
    if role not in ROLES:
        raise ValueError(f"invalid role: {role}")

    commit = _required_string(item, "commit")
    if re.fullmatch(r"[0-9a-fA-F]{7,40}", commit) is None:
        raise ValueError("commit must be a short or full hex SHA")
    record_date = _required_string(item, "date")
    try:
        date.fromisoformat(record_date)
    except ValueError as exc:
        raise ValueError("date must be YYYY-MM-DD") from exc

    score = _required_score(item, "score")
    if role == "floor" and score < floor_min_score:
        raise ValueError(f"{tier} floor score {score} is below {floor_min_score}")
    if role == "ceiling" and score > ceiling_max_score:
        raise ValueError(f"{tier} ceiling score {score} is above {ceiling_max_score}")

    return CalibrationRecord(
        tier=tier,
        role=role,
        suite_id=_required_string(item, "suite_id"),
        run_id=_required_string(item, "run_id"),
        commit=commit,
        backend=_required_string(item, "backend"),
        model_id=_required_string(item, "model_id"),
        served_model_name=_required_string(item, "served_model_name"),
        provider_type=_required_string(item, "provider_type"),
        hardware_profile=_required_string(item, "hardware_profile"),
        weight_quant=_required_string(item, "weight_quant"),
        kv_cache_dtype=_required_string(item, "kv_cache_dtype"),
        context_limit=_required_int(item, "context_limit"),
        concurrency=_required_int(item, "concurrency"),
        score=score,
        date=record_date,
        result_dir=item.get("result_dir") if isinstance(item.get("result_dir"), str) else None,
    )


def _validate_pairs(records: list[CalibrationRecord], *, require_complete: bool) -> None:
    seen = set()
    for record in records:
        pair = (record.tier, record.role)
        if pair in seen:
            raise ValueError(f"duplicate calibration record for {record.tier}/{record.role}")
        seen.add(pair)
    if require_complete:
        expected = {(tier, role) for tier in TIERS for role in ROLES}
        missing = sorted(expected - seen)
        if missing:
            formatted = ", ".join(f"{tier}/{role}" for tier, role in missing)
            raise ValueError("missing calibration records: " + formatted)


def _row_matches_record(row: dict[str, Any], record: CalibrationRecord) -> bool:
    return (
        row.get("run_id") == record.run_id
        and row.get("model") == record.model_id
        and row.get("served_model_name") == record.served_model_name
        and row.get("backend") == record.backend
        and row.get("provider_type") == record.provider_type
        and row.get("hardware_profile") == record.hardware_profile
        and row.get("weight_quant") == record.weight_quant
        and row.get("kv_cache_dtype") == record.kv_cache_dtype
        and row.get("context_limit") == record.context_limit
        and row.get("concurrency") == record.concurrency
    )


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _required_score(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a number")
    score = float(value)
    if score < 0.0 or score > 1.0:
        raise ValueError(f"{key} must be between 0 and 1")
    return score
