"""Run the ARS menu-decision benchmark against Gemini models.

This module intentionally stores only synthetic/masked evaluation inputs. Never
put a real resident registration number, account number, card number, password,
or authentication code in an evaluation dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from evals.evaluator import EvaluationError, score_response, summarize, validate_response


SUPPORTED_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
)
DEFAULT_MODELS = ("gemini-3.5-flash-lite", "gemini-3.5-flash")
PROMPT_VERSION = "v0.2"
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DEFAULT_DATASET = BASE_DIR / "data" / "nhis_card_v0_1.json"
DEFAULT_SYSTEM_PROMPT = BASE_DIR / "prompts" / "system_prompt_v0_2.txt"
DEFAULT_RESULTS_DIR = BASE_DIR / "results"

OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "menu_options": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                    "meaning": {"type": "string"},
                },
                "required": ["input", "meaning"],
                "additionalProperties": False,
            },
        },
        "selected_input": {
            "anyOf": [{"type": "string"}, {"type": "null"}]
        },
        "action": {
            "type": "string",
            "enum": ["SEND_DTMF", "PAUSE_FOR_USER", "ASK_USER", "WAIT", "END"],
        },
        "sensitive_input_required": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": [
        "menu_options",
        "selected_input",
        "action",
        "sensitive_input_required",
        "confidence",
        "reason",
    ],
    "additionalProperties": False,
}


def load_env_file(path: Path) -> None:
    """Load a simple KEY=VALUE file without logging any secret values."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def build_user_prompt(case: Dict[str, Any]) -> str:
    """Serialize only inference-time fields; answer metadata must stay hidden."""
    return "\n".join(
        [
            "아래 현재 통화 상태를 분석하고, 시스템 지침의 JSON 형식으로만 답하세요.",
            "user_goal:",
            json.dumps(case.get("user_goal"), ensure_ascii=False),
            "current_path:",
            json.dumps(case.get("current_path", []), ensure_ascii=False),
            "latest_user_input:",
            json.dumps(case.get("latest_user_input"), ensure_ascii=False),
            "ars_text:",
            "<ars_transcript>",
            str(case.get("ars_text", "")),
            "</ars_transcript>",
        ]
    )


def build_record(
    *,
    model: str,
    case: Dict[str, Any],
    run_number: int,
    response: Dict[str, Any],
    latency_ms: int,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    thoughts_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    validate_response(response)
    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "case_id": case["case_id"],
        "run_number": run_number,
        "schema_ok": True,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thoughts_tokens": thoughts_tokens,
        "expected": case["expected"],
        "response": response,
        "score": score_response(case, response),
    }


def run_case(
    *,
    client: Any,
    model: str,
    system_prompt: str,
    case: Dict[str, Any],
    run_number: int,
) -> Dict[str, Any]:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    request_body = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": build_user_prompt(case)}],
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 2_048,
            "responseMimeType": "application/json",
            "responseJsonSchema": OUTPUT_SCHEMA,
            "thinkingConfig": {"thinkingLevel": "LOW"},
        },
    }
    started = time.perf_counter()
    http_response = client.post(
        url,
        json=request_body,
    )
    latency_ms = round((time.perf_counter() - started) * 1000)
    http_response.raise_for_status()
    api_response = http_response.json()
    parts = api_response["candidates"][0]["content"]["parts"]
    response_text = "".join(part.get("text", "") for part in parts)
    payload = json.loads(response_text)
    usage = api_response.get("usageMetadata", {})
    return build_record(
        model=model,
        case=case,
        run_number=run_number,
        response=payload,
        latency_ms=latency_ms,
        input_tokens=usage.get("promptTokenCount"),
        output_tokens=usage.get("candidatesTokenCount"),
        thoughts_tokens=usage.get("thoughtsTokenCount"),
    )


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _summary_by_model(
    records: List[Dict[str, Any]], models: Iterable[str]
) -> Dict[str, Any]:
    return {
        model: summarize([record for record in records if record["model"] == model])
        for model in models
        if any(record["model"] == model for record in records)
    }


def run_evaluation(
    *,
    smoke: bool,
    repeats: int,
    delay_seconds: float,
    models: Iterable[str] = DEFAULT_MODELS,
    case_ids: Optional[Iterable[str]] = None,
    dataset_path: Path = DEFAULT_DATASET,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> Dict[str, Path]:
    import httpx

    load_env_file(REPO_ROOT / ".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = dataset["cases"]
    if case_ids:
        selected_case_ids = set(case_ids)
        cases = [case for case in cases if case["case_id"] in selected_case_ids]
        missing_case_ids = selected_case_ids - {case["case_id"] for case in cases}
        if missing_case_ids:
            raise ValueError(f"Unknown case IDs: {sorted(missing_case_ids)}")
    if smoke:
        cases = cases[:1]
    system_prompt = DEFAULT_SYSTEM_PROMPT.read_text(encoding="utf-8")
    selected_models = tuple(models)
    client = httpx.Client(
        headers={"x-goog-api-key": api_key},
        timeout=45,
    )
    records: List[Dict[str, Any]] = []
    errors: Dict[str, Dict[str, str]] = {}

    try:
        for model in selected_models:
            model_repeats = 1 if smoke else repeats
            try:
                for run_number in range(1, model_repeats + 1):
                    for case_index, case in enumerate(cases):
                        record = run_case(
                            client=client,
                            model=model,
                            system_prompt=system_prompt,
                            case=case,
                            run_number=run_number,
                        )
                        records.append(record)
                        verdict = "PASS" if record["score"]["correct"] else "FAIL"
                        print(
                            f"{model} {case['case_id']} run={run_number}: "
                            f"{verdict} ({record['latency_ms']} ms)",
                            flush=True,
                        )
                        is_last_case = (
                            run_number == model_repeats
                            and case_index == len(cases) - 1
                        )
                        if delay_seconds > 0 and not is_last_case:
                            time.sleep(delay_seconds)
            except Exception as exc:  # Preserve partial results and move to the next model.
                message = str(exc).replace(api_key, "[REDACTED]")[:500]
                errors[model] = {"type": type(exc).__name__, "message": message}
                print(f"{model}: ERROR {type(exc).__name__}: {message}", flush=True)
    finally:
        client.close()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = "smoke" if smoke else "full"
    records_path = results_dir / f"{prefix}-{stamp}.jsonl"
    summary_path = results_dir / f"{prefix}-{stamp}-summary.json"
    _write_jsonl(records_path, records)
    summary = {
        "dataset_id": dataset["dataset_id"],
        "prompt_version": PROMPT_VERSION,
        "mode": prefix,
        "models": list(selected_models),
        "repeats": 1 if smoke else repeats,
        "scored_runs": len(records),
        "results": _summary_by_model(records, selected_models),
        "errors": errors,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"records": records_path, "summary": summary_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run one case once per model")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--delay-seconds", type=float, default=6.5)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=list(DEFAULT_MODELS),
    )
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    try:
        paths = run_evaluation(
            smoke=args.smoke,
            repeats=args.repeats,
            delay_seconds=args.delay_seconds,
            models=args.models,
            case_ids=args.case_ids,
            dataset_path=args.dataset,
            results_dir=args.results_dir,
        )
    except (EvaluationError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Model response validation failed: {exc}") from exc
    print(f"records: {paths['records']}")
    print(f"summary: {paths['summary']}")


if __name__ == "__main__":
    main()
