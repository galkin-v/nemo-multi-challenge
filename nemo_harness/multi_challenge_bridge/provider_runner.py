from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import requests


MULTI_CHALLENGE_ROOT = Path(
    os.environ.get("MULTI_CHALLENGE_ROOT", "/workspace/nemo-multi-challenge")
)


JUDGE_PROMPT = """You are evaluating whether a model response satisfies a criterion question.

<MODEL_RESPONSE>
{model_response}
</MODEL_RESPONSE>

<CRITERION_QUESTION>
{criterion_question}
</CRITERION_QUESTION>

Respond with JSON only:
{{"reasoning": "short reasoning", "verdict": "YES" or "NO"}}
"""

_TRACE_WRITE_LOCK = threading.Lock()


def _sanitize_for_trace(value: Any) -> Any:
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key_l = str(k).lower()
            if any(token in key_l for token in ("authorization", "api_key", "token", "secret", "password")):
                out[str(k)] = "***REDACTED***"
            else:
                out[str(k)] = _sanitize_for_trace(v)
        return out
    if isinstance(value, list):
        return [_sanitize_for_trace(v) for v in value]
    return value


def _append_trace(trace_path: Path | None, payload: Mapping[str, Any]) -> None:
    if trace_path is None:
        return
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": datetime.now(UTC).isoformat(), **_sanitize_for_trace(dict(payload))}
    line = json.dumps(row, ensure_ascii=False)
    with _TRACE_WRITE_LOCK:
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Provider contract runner for nemo-multi-challenge.")
    parser.add_argument("--benchmark-name", default="nemo_multi_challenge")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-base-url", required=True)
    parser.add_argument("--candidate-model-id", required=True)
    parser.add_argument("--candidate-api-key", default="")
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument("--limit-samples", type=int, default=-1)
    parser.add_argument("--request-params-json", default="{}")
    parser.add_argument("--resume", default="1")
    parser.add_argument("--show-live-stats", default="1")
    return parser.parse_args()


def _to_bool(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _safe_json_obj(raw: str) -> dict[str, Any]:
    payload = json.loads(raw) if raw.strip() else {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _normalize_yes_no(value: Any, *, default: str = "YES") -> str:
    text = str(value).strip().upper()
    if text in {"YES", "NO"}:
        return text
    return default


def _sanitize_metric_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_").lower() or "metric"


def _resolve_input_data(request_params: Mapping[str, Any]) -> Path:
    raw = request_params.get("multichallenge_input_data") or request_params.get("input_data")
    if raw is None or str(raw).strip() == "":
        return (MULTI_CHALLENGE_ROOT / "data" / "benchmark_questions.jsonl").resolve()
    candidate = Path(str(raw)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (MULTI_CHALLENGE_ROOT / candidate).resolve()


def _normalize_base_url(base_url: str) -> str:
    value = str(base_url).strip()
    if value.endswith("/chat/completions"):
        value = value[: -len("/chat/completions")]
    return value.rstrip("/")


def _resolve_secret(value: Any, default: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        return default
    if text.startswith("$"):
        return os.environ.get(text[1:], default)
    return text


def _extract_candidate_generation_params(request_params: Mapping[str, Any]) -> dict[str, Any]:
    reserved = {
        "multichallenge_input_data",
        "multichallenge_judge_base_url",
        "multichallenge_judge_model_id",
        "multichallenge_judge_api_key",
    }
    params: dict[str, Any] = {}
    for key, value in request_params.items():
        key_text = str(key)
        if key_text in reserved:
            continue
        if key_text.startswith("external_"):
            continue
        if key_text.startswith("multichallenge_judge_"):
            continue
        if key_text == "max_new_tokens" and "max_tokens" not in params:
            params["max_tokens"] = value
            continue
        params[key_text] = value
    return params


def _extract_judge_config(
    *,
    request_params: Mapping[str, Any],
    candidate_base_url: str,
    candidate_model_id: str,
    candidate_api_key: str,
) -> tuple[str, str, str, dict[str, Any]]:
    judge_base_url = _normalize_base_url(
        str(request_params.get("multichallenge_judge_base_url") or candidate_base_url)
    )
    judge_model_id = str(request_params.get("multichallenge_judge_model_id") or candidate_model_id)
    judge_api_key = _resolve_secret(
        request_params.get("multichallenge_judge_api_key"),
        default=candidate_api_key,
    )

    judge_params: dict[str, Any] = {}
    for key, value in request_params.items():
        key_text = str(key)
        if not key_text.startswith("multichallenge_judge_"):
            continue
        short_key = key_text[len("multichallenge_judge_") :]
        if short_key in {"base_url", "model_id", "api_key"}:
            continue
        if short_key == "max_new_tokens":
            judge_params["max_tokens"] = value
            continue
        judge_params[short_key] = value

    if "temperature" not in judge_params:
        judge_params["temperature"] = 0.0
    if "top_p" not in judge_params:
        judge_params["top_p"] = 1.0
    if "max_tokens" not in judge_params:
        judge_params["max_tokens"] = 512

    return judge_base_url, judge_model_id, judge_api_key, judge_params


def _first_text_content(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, Mapping):
        content = value.get("content")
        nested = _first_text_content(content)
        if nested:
            return nested
        text = value.get("text")
        nested = _first_text_content(text)
        if nested:
            return nested
        return None
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            nested = _first_text_content(item)
            if nested:
                parts.append(nested)
        if parts:
            return "\n".join(parts)
    return None


def _call_chat_completion(
    *,
    base_url: str,
    model_id: str,
    api_key: str,
    messages: list[dict[str, str]],
    request_timeout: int,
    request_params: Mapping[str, Any],
    trace_path: Path | None = None,
    trace_role: str = "target",
    trace_context: Mapping[str, Any] | None = None,
) -> tuple[str, str | None]:
    endpoint = f"{_normalize_base_url(base_url)}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
    }
    payload.update(request_params)

    started = time.perf_counter()
    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=max(1, request_timeout),
        )
        response.raise_for_status()
        body = response.json()

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            return "", "ExternalPredictionError: missing choices"

        choice = choices[0] if isinstance(choices[0], Mapping) else {}
        message = choice.get("message") if isinstance(choice, Mapping) else {}
        content = _first_text_content(message)
        if content:
            _append_trace(
                trace_path,
                {
                    "role": trace_role,
                    "url": endpoint,
                    "model_id": model_id,
                    "request": payload,
                    "response": {"content": content},
                    "status_code": response.status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                    "context": dict(trace_context or {}),
                },
            )
            return content, None

        _append_trace(
            trace_path,
            {
                "role": trace_role,
                "url": endpoint,
                "model_id": model_id,
                "request": payload,
                "error": "ExternalPredictionError: empty content",
                "status_code": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                "context": dict(trace_context or {}),
            },
        )
        return "", "ExternalPredictionError: empty content"
    except Exception as exc:
        _append_trace(
            trace_path,
            {
                "role": trace_role,
                "url": endpoint,
                "model_id": model_id,
                "request": payload,
                "error": f"{type(exc).__name__}: {exc}",
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                "context": dict(trace_context or {}),
            },
        )
        return "", f"{type(exc).__name__}: {exc}"


def _parse_judge_response(text: str) -> tuple[str | None, str]:
    if not text.strip():
        return None, ""

    parsed_json: dict[str, Any] | None = None
    try:
        candidate = json.loads(text)
        if isinstance(candidate, Mapping):
            parsed_json = dict(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                candidate = json.loads(match.group(0))
                if isinstance(candidate, Mapping):
                    parsed_json = dict(candidate)
            except json.JSONDecodeError:
                parsed_json = None

    if parsed_json is not None:
        verdict = _normalize_yes_no(parsed_json.get("verdict"), default="")
        reasoning = str(parsed_json.get("reasoning", "")).strip()
        if verdict in {"YES", "NO"}:
            return verdict, reasoning or text.strip()

    verdict_matches = re.findall(r"\b(YES|NO)\b", text.upper())
    verdict = verdict_matches[-1] if verdict_matches else None
    return verdict, text.strip()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _conversation_to_prompt(conversation: Any) -> str:
    if not isinstance(conversation, list):
        return ""
    lines: list[str] = []
    for message in conversation:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role", "user")).strip() or "user"
        content = str(message.get("content", ""))
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _make_prediction(
    *,
    sample_id: int,
    row: Mapping[str, Any],
    candidate_base_url: str,
    candidate_model_id: str,
    candidate_api_key: str,
    judge_base_url: str,
    judge_model_id: str,
    judge_api_key: str,
    request_timeout: int,
    candidate_generation_params: Mapping[str, Any],
    judge_generation_params: Mapping[str, Any],
    target_trace_path: Path | None = None,
    judge_trace_path: Path | None = None,
) -> dict[str, Any]:
    question_id = str(row.get("QUESTION_ID", sample_id))
    axis = str(row.get("AXIS", "UNKNOWN"))
    criterion_question = str(row.get("TARGET_QUESTION", "")).strip()
    pass_criteria = _normalize_yes_no(row.get("PASS_CRITERIA"), default="YES")

    raw_conversation = row.get("CONVERSATION")
    conversation = list(raw_conversation) if isinstance(raw_conversation, list) else []
    messages: list[dict[str, str]] = []
    for message in conversation:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role", "user")).strip() or "user"
        content = str(message.get("content", ""))
        messages.append({"role": role, "content": content})

    response_text, response_error = _call_chat_completion(
        base_url=candidate_base_url,
        model_id=candidate_model_id,
        api_key=candidate_api_key,
        messages=messages,
        request_timeout=request_timeout,
        request_params=candidate_generation_params,
        trace_path=target_trace_path,
        trace_role="target",
        trace_context={"sample_id": sample_id, "question_id": question_id},
    )

    judge_verdict: str | None = None
    judge_reasoning = ""
    judge_error: str | None = None

    if response_error is None:
        judge_prompt = JUDGE_PROMPT.format(
            model_response=response_text,
            criterion_question=criterion_question,
        )
        judge_text, judge_error = _call_chat_completion(
            base_url=judge_base_url,
            model_id=judge_model_id,
            api_key=judge_api_key,
            messages=[{"role": "user", "content": judge_prompt}],
            request_timeout=request_timeout,
            request_params=judge_generation_params,
            trace_path=judge_trace_path,
            trace_role="judge",
            trace_context={"sample_id": sample_id, "question_id": question_id},
        )
        if judge_error is None:
            judge_verdict, judge_reasoning = _parse_judge_response(judge_text)
            if judge_verdict not in {"YES", "NO"}:
                judge_error = "JudgeEvaluationError: verdict is missing"
    else:
        judge_error = "JudgeSkipped: candidate_response_error"

    passed = int(judge_verdict == pass_criteria and response_error is None and judge_error is None)
    axis_metric = f"multi_challenge_{_sanitize_metric_name(axis)}_pass"

    error: str | None = response_error or judge_error
    status = "scored" if error is None else "error"

    return {
        "sample_id": sample_id,
        "prompt": _conversation_to_prompt(conversation),
        "response": response_text,
        "target": criterion_question,
        "status": status,
        "error": error,
        "scores": {
            "multi_challenge_pass": passed,
            axis_metric: passed,
        },
        "metadata": {
            "model_id": candidate_model_id,
            "question_id": question_id,
            "axis": axis,
            "expected_pass_criteria": pass_criteria,
            "judge_verdict": judge_verdict,
            "judge_reasoning": judge_reasoning,
            "candidate_error": response_error,
            "judge_error": judge_error,
            "judge_model_id": judge_model_id,
        },
    }


def _load_resume_predictions(path: Path) -> dict[str, dict[str, Any]]:
    by_qid: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(path):
        metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
        question_id = str(metadata.get("question_id", "")).strip()
        if not question_id:
            continue
        by_qid[question_id] = row
    return by_qid


def main() -> int:
    args = _parse_args()
    started_at = datetime.now(UTC)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "traces"
    target_trace_path = trace_dir / "target.jsonl"
    judge_trace_path = trace_dir / "judge.jsonl"

    show_live_stats = _to_bool(args.show_live_stats)
    resume = _to_bool(args.resume)
    request_params = _safe_json_obj(args.request_params_json)

    input_data_path = _resolve_input_data(request_params)
    if not input_data_path.exists():
        raise FileNotFoundError(f"MultiChallenge input dataset not found: {input_data_path}")

    rows = _load_jsonl(input_data_path)
    if args.limit_samples >= 0:
        rows = rows[: args.limit_samples]

    candidate_base_url = _normalize_base_url(args.candidate_base_url)
    candidate_generation_params = _extract_candidate_generation_params(request_params)
    judge_base_url, judge_model_id, judge_api_key, judge_generation_params = _extract_judge_config(
        request_params=request_params,
        candidate_base_url=candidate_base_url,
        candidate_model_id=args.candidate_model_id,
        candidate_api_key=args.candidate_api_key,
    )

    if show_live_stats:
        print(
            f"[{args.benchmark_name}] samples={len(rows)} model={args.candidate_model_id} "
            f"judge_model={judge_model_id}",
            flush=True,
        )

    resume_path = output_dir / "byob_predictions.jsonl"
    resume_by_qid = _load_resume_predictions(resume_path) if resume and resume_path.exists() else {}

    predictions: list[dict[str, Any] | None] = [None] * len(rows)
    pending_indices: list[int] = []
    for idx, row in enumerate(rows):
        question_id = str(row.get("QUESTION_ID", idx))
        resumed = resume_by_qid.get(question_id)
        if resumed is not None:
            resumed["sample_id"] = idx
            predictions[idx] = resumed
            continue
        pending_indices.append(idx)

    with ThreadPoolExecutor(max_workers=max(1, args.parallelism)) as executor:
        future_to_index = {
            executor.submit(
                _make_prediction,
                sample_id=idx,
                row=rows[idx],
                candidate_base_url=candidate_base_url,
                candidate_model_id=args.candidate_model_id,
                candidate_api_key=args.candidate_api_key,
                judge_base_url=judge_base_url,
                judge_model_id=judge_model_id,
                judge_api_key=judge_api_key,
                request_timeout=max(1, args.request_timeout),
                candidate_generation_params=candidate_generation_params,
                judge_generation_params=judge_generation_params,
                target_trace_path=target_trace_path,
                judge_trace_path=judge_trace_path,
            ): idx
            for idx in pending_indices
        }

        completed = 0
        total_pending = len(future_to_index)
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            predictions[idx] = future.result()
            completed += 1
            if show_live_stats and (completed % 10 == 0 or completed == total_pending):
                print(f"[{args.benchmark_name}] processed {completed}/{total_pending}", flush=True)

    finalized_predictions: list[dict[str, Any]] = []
    for idx, row in enumerate(predictions):
        if row is None:
            question_id = str(rows[idx].get("QUESTION_ID", idx))
            axis = str(rows[idx].get("AXIS", "UNKNOWN"))
            fallback = {
                "sample_id": idx,
                "prompt": _conversation_to_prompt(rows[idx].get("CONVERSATION")),
                "response": "",
                "target": str(rows[idx].get("TARGET_QUESTION", "")),
                "status": "error",
                "error": "InternalError: prediction missing",
                "scores": {
                    "multi_challenge_pass": 0,
                    f"multi_challenge_{_sanitize_metric_name(axis)}_pass": 0,
                },
                "metadata": {
                    "model_id": args.candidate_model_id,
                    "question_id": question_id,
                    "axis": axis,
                    "expected_pass_criteria": _normalize_yes_no(rows[idx].get("PASS_CRITERIA")),
                    "judge_verdict": None,
                    "judge_reasoning": "",
                    "candidate_error": "InternalError: prediction missing",
                    "judge_error": "InternalError: prediction missing",
                    "judge_model_id": judge_model_id,
                },
            }
            finalized_predictions.append(fallback)
            continue
        row["sample_id"] = idx
        finalized_predictions.append(row)

    axis_total: dict[str, int] = {}
    axis_passed: dict[str, int] = {}
    for prediction in finalized_predictions:
        metadata = prediction.get("metadata") if isinstance(prediction.get("metadata"), Mapping) else {}
        axis = str(metadata.get("axis", "UNKNOWN"))
        axis_total[axis] = axis_total.get(axis, 0) + 1
        passed = int(prediction.get("scores", {}).get("multi_challenge_pass", 0))
        axis_passed[axis] = axis_passed.get(axis, 0) + passed

    sample_count = len(finalized_predictions)
    total_passed = sum(int(prediction.get("scores", {}).get("multi_challenge_pass", 0)) for prediction in finalized_predictions)
    weighted_accuracy = (total_passed / sample_count) if sample_count else 0.0

    axis_accuracies: list[float] = []
    metric_values: dict[str, tuple[float, int]] = {}
    for axis in sorted(axis_total):
        total = axis_total[axis]
        passed = axis_passed.get(axis, 0)
        accuracy = (passed / total) if total else 0.0
        axis_accuracies.append(accuracy)
        metric_values[f"multi_challenge_{_sanitize_metric_name(axis)}_accuracy"] = (accuracy, total)

    axis_mean_accuracy = sum(axis_accuracies) / len(axis_accuracies) if axis_accuracies else 0.0
    metric_values["multi_challenge_weighted_accuracy"] = (weighted_accuracy, sample_count)
    metric_values["multi_challenge_axis_mean_accuracy"] = (axis_mean_accuracy, len(axis_accuracies))

    successful_count = sum(1 for prediction in finalized_predictions if prediction.get("error") in (None, ""))
    judge_success_count = sum(
        1
        for prediction in finalized_predictions
        if not prediction.get("metadata", {}).get("judge_error")
    )
    if sample_count:
        metric_values["multi_challenge_response_success_rate"] = (successful_count / sample_count, sample_count)
        metric_values["multi_challenge_judge_success_rate"] = (judge_success_count / sample_count, sample_count)

    _write_jsonl(output_dir / "byob_predictions.jsonl", finalized_predictions)
    scores_payload = {
        metric_name: {
            "stats": {
                "count": count,
                "mean": round(value, 6),
                "stddev": 0.0,
                "stderr": 0.0,
            },
            "value": value,
        }
        for metric_name, (value, count) in metric_values.items()
    }
    _write_json(
        output_dir / "byob_results.json",
        {
            "tasks": {
                args.benchmark_name: {
                    "metrics": {
                        "pass@1": {
                            "scores": scores_payload,
                        }
                    }
                }
            }
        },
    )

    finished_at = datetime.now(UTC)
    inference_time = max(0.0, (finished_at - started_at).total_seconds())
    _write_json(
        output_dir / "eval_factory_metrics.json",
        {
            "response_stats": {
                "count": sample_count,
                "successful_count": successful_count,
                "avg_latency_ms": 0.0,
                "avg_total_tokens": 0.0,
                "avg_completion_tokens": 0.0,
            },
            "timing": {
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "inference_time_seconds": inference_time,
            },
        },
    )

    _write_json(
        output_dir / "params.json",
        {
            "parallelism": max(1, args.parallelism),
            "request_timeout": max(1, args.request_timeout),
            "limit_samples": args.limit_samples if args.limit_samples >= 0 else None,
            "resume": resume,
            "show_live_stats": show_live_stats,
            "request_params": request_params,
            "multichallenge_input_data": str(input_data_path),
            "judge": {
                "base_url": judge_base_url,
                "model_id": judge_model_id,
                "has_api_key": bool(judge_api_key),
                "request_params": judge_generation_params,
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
