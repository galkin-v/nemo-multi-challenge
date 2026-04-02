from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nemo_evaluator.api.api_dataclasses import EvaluationResult

_STAT_KEYS = {
    "count",
    "sum",
    "sum_squared",
    "min",
    "max",
    "mean",
    "variance",
    "stddev",
    "stderr",
}


def _to_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_byob_results(payload: dict[str, Any]) -> EvaluationResult:
    tasks: dict[str, Any] = {}
    for task_name, task_payload in (payload.get("tasks") or {}).items():
        metrics: dict[str, Any] = {}
        for metric_name, metric_payload in (task_payload.get("metrics") or {}).items():
            scores: dict[str, Any] = {}
            for score_name, score_payload in (metric_payload.get("scores") or {}).items():
                if isinstance(score_payload, dict):
                    score_value = _to_float(score_payload.get("value"), default=0.0)
                    stats = {
                        key: _to_float(value, default=0.0)
                        for key, value in (score_payload.get("stats") or {}).items()
                        if key in _STAT_KEYS
                    }
                else:
                    score_value = _to_float(score_payload, default=0.0)
                    stats = {}
                scores[str(score_name)] = {"value": score_value, "stats": stats}
            metrics[str(metric_name)] = {"scores": scores}
        tasks[str(task_name)] = {"metrics": metrics}
    return EvaluationResult.model_validate({"tasks": tasks})


def parse_output(output_dir: str) -> EvaluationResult:
    path = Path(output_dir) / "byob_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing expected output file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _normalize_byob_results(payload)
