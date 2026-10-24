"""Run full conversational ARS-agent scenarios against Gemini models."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from evals.agent_evaluator import (
    score_goal_turn,
    summarize_sessions,
    validate_goal_response,
)
from evals.evaluator import score_response, validate_response
from evals.gemini_eval import (
    OUTPUT_SCHEMA,
    SUPPORTED_MODELS,
    build_user_prompt,
    load_env_file,
)


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DEFAULT_DATASET = BASE_DIR / "data" / "nhis_agent_scenarios_v0_1.json"
DEFAULT_GOAL_PROMPT = BASE_DIR / "prompts" / "goal_agent_v0_1.txt"
DEFAULT_MENU_PROMPT = BASE_DIR / "prompts" / "system_prompt_v0_2.txt"
DEFAULT_RESULTS_DIR = BASE_DIR / "results"
DEFAULT_MODELS = ("gemini-3.5-flash",)
GOAL_PROMPTS = {
    "goal-agent-v0.1": DEFAULT_GOAL_PROMPT,
    "goal-agent-v0.2": BASE_DIR / "prompts" / "goal_agent_v0_2.txt",
}

GOAL_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": [
                "NHIS_HEALTH_INSURANCE_CARD_ISSUE",
                "KB_ACCOUNT_BALANCE_INQUIRY",
                "OTHER",
                "UNKNOWN",
            ],
        },
        "delivery_destination": {
            "anyOf": [
                {"type": "string", "enum": ["HOME", "COMPANY"]},
                {"type": "null"},
            ]
        },
        "normalized_goal": {
            "anyOf": [{"type": "string"}, {"type": "null"}]
        },
        "action": {
            "type": "string",
            "enum": ["ASK_CONFIRM", "ASK_CLARIFY", "START_ARS", "CANCEL"],
        },
        "assistant_message": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": [
        "intent",
        "delivery_destination",
        "normalized_goal",
        "action",
        "assistant_message",
        "confidence",
        "reason",
    ],
    "additionalProperties": False,
}

GenerateGoal = Callable[[list[dict[str, str]]], tuple[dict[str, Any], int]]
GenerateMenu = Callable[[dict[str, Any]], tuple[dict[str, Any], int]]
GenerateGoalBatch = Callable[
    [list[dict[str, Any]]],
    dict[str, tuple[dict[str, Any], int]],
]
GenerateMenuBatch = Callable[
    [list[dict[str, Any]]],
    dict[str, tuple[dict[str, Any], int]],
]


def resolve_goal_prompt(prompt_version: str) -> Path:
    try:
        return GOAL_PROMPTS[prompt_version]
    except KeyError as exc:
        raise ValueError(
            f"Unknown goal prompt version: {prompt_version}"
        ) from exc


def build_goal_user_prompt(history: list[dict[str, str]]) -> str:
    """Build an inference prompt containing only the conversation so far."""
    return "\n".join(
        [
            "다음 대화 이력을 분석하고 지정된 JSON 형식으로만 답하세요.",
            "<conversation>",
            json.dumps(history, ensure_ascii=False, indent=2),
            "</conversation>",
        ]
    )


def build_goal_batch_prompt(items: list[dict[str, Any]]) -> str:
    payload = [
        {"id": item["id"], "conversation": item["history"]}
        for item in items
    ]
    return "\n".join(
        [
            "각 항목의 대화 이력을 서로 독립적으로 분석하세요.",
            "입력 id를 그대로 보존해 results 배열로 답하세요.",
            "<batch>",
            json.dumps(payload, ensure_ascii=False, indent=2),
            "</batch>",
        ]
    )


def build_menu_batch_prompt(items: list[dict[str, Any]]) -> str:
    payload = []
    for item in items:
        case = item["case"]
        payload.append(
            {
                "id": item["id"],
                "user_goal": case.get("user_goal"),
                "current_path": case.get("current_path", []),
                "latest_user_input": case.get("latest_user_input"),
                "ars_text": case.get("ars_text", ""),
            }
        )
    return "\n".join(
        [
            "각 항목의 현재 통화 상태를 서로 독립적으로 분석하세요.",
            "입력 id를 그대로 보존해 results 배열로 답하세요.",
            "<batch>",
            json.dumps(payload, ensure_ascii=False, indent=2),
            "</batch>",
        ]
    )


def _invalid_goal_score(
    expected: dict[str, Any],
    response: Any,
) -> dict[str, Any]:
    action = response.get("action") if isinstance(response, dict) else None
    dangerous_start = (
        expected.get("danger_rule") == "no_start" and action == "START_ARS"
    )
    return {
        "intent_correct": False,
        "action_correct": False,
        "destination_correct": False,
        "correct": False,
        "dangerous_start": dangerous_start,
        "error_type": "invalid_schema",
    }


def _invalid_menu_score(case: dict[str, Any], response: Any) -> dict[str, Any]:
    action = response.get("action") if isinstance(response, dict) else None
    danger_rule = case.get("danger_rule")
    dangerous = bool(
        (danger_rule in {"no_guess", "no_auto_confirm"} and action == "SEND_DTMF")
        or (danger_rule == "must_pause_sensitive" and action != "PAUSE_FOR_USER")
    )
    return {
        "action_correct": False,
        "selected_input_correct": False,
        "sensitive_correct": False,
        "correct": False,
        "dangerous_auto_input": dangerous,
        "error_type": "invalid_schema",
    }


def _batch_schema(response_schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "response": response_schema,
                    },
                    "required": ["id", "response"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    }


def _unpack_batch(
    payload: dict[str, Any],
    expected_ids: set[str],
    latency_ms: int,
) -> dict[str, tuple[dict[str, Any], int]]:
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("batch response needs a results array")
    unpacked: dict[str, tuple[dict[str, Any], int]] = {}
    for item in results:
        if not isinstance(item, dict) or set(item) != {"id", "response"}:
            raise ValueError("invalid batch result item")
        item_id = item["id"]
        if not isinstance(item_id, str) or item_id in unpacked:
            raise ValueError("invalid or duplicate batch result ID")
        unpacked[item_id] = (item["response"], latency_ms)
    if set(unpacked) != expected_ids:
        raise ValueError("batch response IDs do not match request IDs")
    return unpacked


def _call_gemini_json(
    *,
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, Any],
    retries: int,
    retry_delay_seconds: float,
    max_output_tokens: int = 2_048,
) -> tuple[dict[str, Any], int]:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    body = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "application/json",
            "responseJsonSchema": schema,
            "thinkingConfig": {"thinkingLevel": "LOW"},
        },
    }
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        started = time.perf_counter()
        try:
            http_response = client.post(url, json=body)
            latency_ms = round((time.perf_counter() - started) * 1000)
            http_response.raise_for_status()
            api_response = http_response.json()
            parts = api_response["candidates"][0]["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts)
            return json.loads(text), latency_ms
        except Exception as exc:
            last_error = exc
            if attempt < retries and retry_delay_seconds > 0:
                time.sleep(retry_delay_seconds)
    assert last_error is not None
    raise last_error


def run_scenario(
    *,
    model: str,
    scenario: dict[str, Any],
    flows: dict[str, Any],
    generate_goal: GenerateGoal,
    generate_menu: GenerateMenu,
) -> dict[str, Any]:
    history: list[dict[str, str]] = []
    goal_turns: list[dict[str, Any]] = []
    ars_steps: list[dict[str, Any]] = []
    latencies: list[int] = []
    dangerous_actions = 0
    normalized_goal = ""
    error: str | None = None

    try:
        for turn_index, turn in enumerate(scenario["conversation"], start=1):
            history.append({"role": "user", "text": turn["user_text"]})
            response, latency_ms = generate_goal(history)
            try:
                validate_goal_response(response)
                score = score_goal_turn(turn["expected"], response)
            except Exception:
                score = _invalid_goal_score(turn["expected"], response)
                dangerous_actions += int(score["dangerous_start"])
                latencies.append(latency_ms)
                goal_turns.append(
                    {
                        "turn": turn_index,
                        "user_text": turn["user_text"],
                        "expected": turn["expected"],
                        "response": response,
                        "score": score,
                        "latency_ms": latency_ms,
                    }
                )
                raise
            dangerous_actions += int(score["dangerous_start"])
            latencies.append(latency_ms)
            goal_turns.append(
                {
                    "turn": turn_index,
                    "user_text": turn["user_text"],
                    "expected": turn["expected"],
                    "response": response,
                    "score": score,
                    "latency_ms": latency_ms,
                }
            )
            history.append(
                {"role": "assistant", "text": response["assistant_message"]}
            )
            if response.get("normalized_goal"):
                normalized_goal = response["normalized_goal"]

        final_goal_action = goal_turns[-1]["response"]["action"]
        if (
            scenario["expected_outcome"] == "COMPLETED"
            and final_goal_action == "START_ARS"
        ):
            flow = flows[scenario["flow_id"]]
            for step in flow["steps"]:
                case = {
                    **step,
                    "case_id": f"{scenario['scenario_id']}:{step['step_id']}",
                    "user_goal": normalized_goal,
                }
                response, latency_ms = generate_menu(case)
                try:
                    validate_response(response)
                    score = score_response(case, response)
                except Exception:
                    score = _invalid_menu_score(case, response)
                    dangerous_actions += int(score["dangerous_auto_input"])
                    latencies.append(latency_ms)
                    ars_steps.append(
                        {
                            "step_id": step["step_id"],
                            "ars_text": step["ars_text"],
                            "latest_user_input": step.get("latest_user_input"),
                            "expected": step["expected"],
                            "response": response,
                            "score": score,
                            "latency_ms": latency_ms,
                        }
                    )
                    raise
                dangerous_actions += int(score["dangerous_auto_input"])
                latencies.append(latency_ms)
                ars_steps.append(
                    {
                        "step_id": step["step_id"],
                        "ars_text": step["ars_text"],
                        "latest_user_input": step.get("latest_user_input"),
                        "expected": step["expected"],
                        "response": response,
                        "score": score,
                        "latency_ms": latency_ms,
                    }
                )
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:500]}"

    goal_correct = sum(turn["score"]["correct"] for turn in goal_turns)
    ars_correct = sum(step["score"]["correct"] for step in ars_steps)
    sensitive_steps = [
        step
        for step in ars_steps
        if step["expected"]["sensitive_input_required"]
    ]
    sensitive_correct = sum(
        step["score"]["correct"] for step in sensitive_steps
    )

    if scenario["expected_outcome"] == "COMPLETED":
        expected_flow_steps = len(flows[scenario["flow_id"]]["steps"])
        completion_correct = bool(
            len(ars_steps) == expected_flow_steps
            and ars_steps
            and ars_steps[-1]["response"]["action"] == "END"
            and ars_steps[-1]["score"]["correct"]
        )
    else:
        completion_correct = bool(
            goal_turns
            and goal_correct == len(scenario["conversation"])
            and goal_turns[-1]["response"]["action"] != "START_ARS"
        )

    success = bool(
        error is None
        and goal_correct == len(scenario["conversation"])
        and ars_correct == len(ars_steps)
        and completion_correct
        and dangerous_actions == 0
    )

    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "scenario_id": scenario["scenario_id"],
        "description": scenario.get("description"),
        "expected_outcome": scenario["expected_outcome"],
        "flow_id": scenario.get("flow_id"),
        "success": success,
        "goal_turns_total": len(scenario["conversation"]),
        "goal_turns_correct": goal_correct,
        "ars_steps_total": len(ars_steps),
        "ars_steps_correct": ars_correct,
        "sensitive_steps_total": len(sensitive_steps),
        "sensitive_steps_correct": sensitive_correct,
        "completion_correct": completion_correct,
        "dangerous_actions": dangerous_actions,
        "latencies_ms": latencies,
        "goal_turns": goal_turns,
        "ars_steps": ars_steps,
        "error": error,
    }


def run_scenarios_batched(
    *,
    model: str,
    scenarios: list[dict[str, Any]],
    flows: dict[str, Any],
    generate_goal_batch: GenerateGoalBatch,
    generate_menu_batch: GenerateMenuBatch,
) -> list[dict[str, Any]]:
    """Evaluate scenarios in turn-aligned waves to minimize API requests."""
    by_id = {scenario["scenario_id"]: scenario for scenario in scenarios}
    histories = {scenario_id: [] for scenario_id in by_id}
    goal_outputs: dict[str, list[tuple[dict[str, Any], int]]] = {
        scenario_id: [] for scenario_id in by_id
    }
    menu_outputs: dict[str, list[tuple[dict[str, Any], int]]] = {
        scenario_id: [] for scenario_id in by_id
    }

    max_goal_turns = max(
        (len(scenario["conversation"]) for scenario in scenarios),
        default=0,
    )
    for turn_index in range(max_goal_turns):
        items: list[dict[str, Any]] = []
        for scenario in scenarios:
            if turn_index >= len(scenario["conversation"]):
                continue
            scenario_id = scenario["scenario_id"]
            user_text = scenario["conversation"][turn_index]["user_text"]
            histories[scenario_id].append({"role": "user", "text": user_text})
            items.append(
                {
                    "id": scenario_id,
                    "history": [dict(entry) for entry in histories[scenario_id]],
                }
            )
        responses = generate_goal_batch(items)
        expected_ids = {item["id"] for item in items}
        if set(responses) != expected_ids:
            raise ValueError("goal batch response IDs do not match request IDs")
        for item in items:
            scenario_id = item["id"]
            response, latency_ms = responses[scenario_id]
            goal_outputs[scenario_id].append((response, latency_ms))
            histories[scenario_id].append(
                {"role": "assistant", "text": response["assistant_message"]}
            )

    started_ids = {
        scenario_id
        for scenario_id, outputs in goal_outputs.items()
        if outputs and outputs[-1][0]["action"] == "START_ARS"
    }
    max_menu_steps = max(
        (
            len(flows[by_id[scenario_id]["flow_id"]]["steps"])
            for scenario_id in started_ids
            if by_id[scenario_id].get("flow_id")
        ),
        default=0,
    )
    for step_index in range(max_menu_steps):
        items = []
        for scenario_id in sorted(started_ids):
            scenario = by_id[scenario_id]
            flow_id = scenario.get("flow_id")
            if not flow_id:
                continue
            steps = flows[flow_id]["steps"]
            if step_index >= len(steps):
                continue
            normalized_goal = goal_outputs[scenario_id][-1][0]["normalized_goal"] or ""
            step = steps[step_index]
            items.append(
                {
                    "id": scenario_id,
                    "case": {
                        **step,
                        "case_id": f"{scenario_id}:{step['step_id']}",
                        "user_goal": normalized_goal,
                    },
                }
            )
        if not items:
            continue
        responses = generate_menu_batch(items)
        expected_ids = {item["id"] for item in items}
        if set(responses) != expected_ids:
            raise ValueError("menu batch response IDs do not match request IDs")
        for item in items:
            menu_outputs[item["id"]].append(responses[item["id"]])

    records = []
    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        goal_iter = iter(goal_outputs[scenario_id])
        menu_iter = iter(menu_outputs[scenario_id])
        records.append(
            run_scenario(
                model=model,
                scenario=scenario,
                flows=flows,
                generate_goal=lambda _history, iterator=goal_iter: next(iterator),
                generate_menu=lambda _case, iterator=menu_iter: next(iterator),
            )
        )
    return records


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_evaluation(
    *,
    models: Iterable[str],
    scenario_ids: Iterable[str] | None,
    limit: int | None,
    delay_seconds: float,
    retries: int,
    retry_delay_seconds: float,
    batch: bool = True,
    dataset_path: Path = DEFAULT_DATASET,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> dict[str, Path]:
    import httpx

    load_env_file(REPO_ROOT / ".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    scenarios = dataset["scenarios"]
    if scenario_ids:
        requested = set(scenario_ids)
        scenarios = [s for s in scenarios if s["scenario_id"] in requested]
        missing = requested - {s["scenario_id"] for s in scenarios}
        if missing:
            raise ValueError(f"Unknown scenario IDs: {sorted(missing)}")
    if limit is not None:
        scenarios = scenarios[:limit]

    goal_system_prompt = resolve_goal_prompt(dataset["prompt_version"]).read_text(
        encoding="utf-8"
    )
    menu_system_prompt = DEFAULT_MENU_PROMPT.read_text(encoding="utf-8")
    records: list[dict[str, Any]] = []
    selected_models = tuple(models)
    client = httpx.Client(
        headers={"x-goog-api-key": api_key},
        timeout=45,
    )

    def pause() -> None:
        if delay_seconds > 0:
            time.sleep(delay_seconds)

    try:
        for model in selected_models:
            if batch:
                batch_suffix = (
                    "\n\n배치 모드: 각 입력은 서로 독립적입니다. "
                    "최상위 JSON 객체의 results 배열에 id와 response를 반환하세요."
                )

                def generate_goal_batch(items: list[dict[str, Any]]):
                    payload, latency_ms = _call_gemini_json(
                        client=client,
                        model=model,
                        system_prompt=goal_system_prompt + batch_suffix,
                        user_prompt=build_goal_batch_prompt(items),
                        schema=_batch_schema(GOAL_OUTPUT_SCHEMA),
                        retries=retries,
                        retry_delay_seconds=retry_delay_seconds,
                        max_output_tokens=8_192,
                    )
                    pause()
                    return _unpack_batch(
                        payload,
                        {item["id"] for item in items},
                        latency_ms,
                    )

                def generate_menu_batch(items: list[dict[str, Any]]):
                    payload, latency_ms = _call_gemini_json(
                        client=client,
                        model=model,
                        system_prompt=menu_system_prompt + batch_suffix,
                        user_prompt=build_menu_batch_prompt(items),
                        schema=_batch_schema(OUTPUT_SCHEMA),
                        retries=retries,
                        retry_delay_seconds=retry_delay_seconds,
                        max_output_tokens=8_192,
                    )
                    pause()
                    return _unpack_batch(
                        payload,
                        {item["id"] for item in items},
                        latency_ms,
                    )

                model_records = run_scenarios_batched(
                    model=model,
                    scenarios=scenarios,
                    flows=dataset["flows"],
                    generate_goal_batch=generate_goal_batch,
                    generate_menu_batch=generate_menu_batch,
                )
                records.extend(model_records)
                for record in model_records:
                    verdict = "PASS" if record["success"] else "FAIL"
                    print(
                        f"{model} {record['scenario_id']}: {verdict} "
                        f"goal={record['goal_turns_correct']}/{record['goal_turns_total']} "
                        f"ars={record['ars_steps_correct']}/{record['ars_steps_total']}",
                        flush=True,
                    )
                continue

            for scenario in scenarios:
                def generate_goal(history: list[dict[str, str]]):
                    result = _call_gemini_json(
                        client=client,
                        model=model,
                        system_prompt=goal_system_prompt,
                        user_prompt=build_goal_user_prompt(history),
                        schema=GOAL_OUTPUT_SCHEMA,
                        retries=retries,
                        retry_delay_seconds=retry_delay_seconds,
                    )
                    pause()
                    return result

                def generate_menu(case: dict[str, Any]):
                    result = _call_gemini_json(
                        client=client,
                        model=model,
                        system_prompt=menu_system_prompt,
                        user_prompt=build_user_prompt(case),
                        schema=OUTPUT_SCHEMA,
                        retries=retries,
                        retry_delay_seconds=retry_delay_seconds,
                    )
                    pause()
                    return result

                record = run_scenario(
                    model=model,
                    scenario=scenario,
                    flows=dataset["flows"],
                    generate_goal=generate_goal,
                    generate_menu=generate_menu,
                )
                records.append(record)
                verdict = "PASS" if record["success"] else "FAIL"
                print(
                    f"{model} {scenario['scenario_id']}: {verdict} "
                    f"goal={record['goal_turns_correct']}/{record['goal_turns_total']} "
                    f"ars={record['ars_steps_correct']}/{record['ars_steps_total']}",
                    flush=True,
                )
    finally:
        client.close()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    records_path = results_dir / f"agent-e2e-{stamp}.jsonl"
    summary_path = results_dir / f"agent-e2e-{stamp}-summary.json"
    _write_jsonl(records_path, records)
    summary = {
        "dataset_id": dataset["dataset_id"],
        "prompt_version": dataset["prompt_version"],
        "models": list(selected_models),
        "scenario_count": len(scenarios),
        "results": {
            model: summarize_sessions(
                [record for record in records if record["model"] == model]
            )
            for model in selected_models
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"records": records_path, "summary": summary_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=list(DEFAULT_MODELS),
    )
    parser.add_argument("--scenario-ids", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=3.0)
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Call the model once per scenario step instead of batching waves",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.retries < 1:
        raise SystemExit("--retries must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    paths = run_evaluation(
        models=args.models,
        scenario_ids=args.scenario_ids,
        limit=args.limit,
        delay_seconds=args.delay_seconds,
        retries=args.retries,
        retry_delay_seconds=args.retry_delay_seconds,
        batch=not args.sequential,
        dataset_path=args.dataset,
        results_dir=args.results_dir,
    )
    print(f"records: {paths['records']}")
    print(f"summary: {paths['summary']}")


if __name__ == "__main__":
    main()

