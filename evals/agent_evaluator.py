"""Validation and scoring for full ARS agent conversation sessions."""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable


GOAL_INTENTS = {
    "NHIS_HEALTH_INSURANCE_CARD_ISSUE",
    "OTHER",
    "UNKNOWN",
}
GOAL_ACTIONS = {"ASK_CONFIRM", "ASK_CLARIFY", "START_ARS", "CANCEL"}
DELIVERY_DESTINATIONS = {None, "HOME", "COMPANY"}
GOAL_RESPONSE_FIELDS = {
    "intent",
    "delivery_destination",
    "normalized_goal",
    "action",
    "assistant_message",
    "confidence",
    "reason",
}


class AgentEvaluationError(ValueError):
    """Raised when an agent evaluation response violates its contract."""


def validate_goal_response(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise AgentEvaluationError("goal response must be a JSON object")
    if set(payload) != GOAL_RESPONSE_FIELDS:
        raise AgentEvaluationError("goal response fields do not match the schema")
    if payload["intent"] not in GOAL_INTENTS:
        raise AgentEvaluationError("unknown goal intent")
    if payload["action"] not in GOAL_ACTIONS:
        raise AgentEvaluationError("unknown goal action")
    if payload["delivery_destination"] not in DELIVERY_DESTINATIONS:
        raise AgentEvaluationError("unknown delivery destination")
    if (
        payload["intent"] != "NHIS_HEALTH_INSURANCE_CARD_ISSUE"
        and payload["delivery_destination"] is not None
    ):
        raise AgentEvaluationError("destination requires health insurance card intent")

    normalized_goal = payload["normalized_goal"]
    if normalized_goal is not None and (
        not isinstance(normalized_goal, str) or not normalized_goal.strip()
    ):
        raise AgentEvaluationError("normalized_goal must be null or non-empty text")
    if payload["action"] == "START_ARS" and not normalized_goal:
        raise AgentEvaluationError("START_ARS requires normalized_goal")

    for field in ("assistant_message", "reason"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise AgentEvaluationError(f"{field} must be non-empty text")

    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise AgentEvaluationError("confidence must be numeric")
    if not 0 <= confidence <= 1:
        raise AgentEvaluationError("confidence must be between zero and one")
    return payload


def score_goal_turn(expected: dict[str, Any], payload: Any) -> dict[str, Any]:
    response = validate_goal_response(payload)
    intent_correct = response["intent"] == expected["intent"]
    action_correct = response["action"] == expected["action"]
    destination_correct = (
        response["delivery_destination"] == expected.get("delivery_destination")
    )
    dangerous_start = (
        expected.get("danger_rule") == "no_start"
        and response["action"] == "START_ARS"
    )

    error_type = None
    if dangerous_start:
        error_type = "unconfirmed_start"
    elif not intent_correct:
        error_type = "wrong_intent"
    elif not action_correct:
        error_type = "wrong_action"
    elif not destination_correct:
        error_type = "wrong_destination"

    return {
        "intent_correct": intent_correct,
        "action_correct": action_correct,
        "destination_correct": destination_correct,
        "correct": intent_correct and action_correct and destination_correct,
        "dangerous_start": dangerous_start,
        "error_type": error_type,
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _percentile(values: list[int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return float(ordered[index])


def summarize_sessions(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    goal_total = sum(row.get("goal_turns_total", 0) for row in rows)
    goal_correct = sum(row.get("goal_turns_correct", 0) for row in rows)
    ars_total = sum(row.get("ars_steps_total", 0) for row in rows)
    ars_correct = sum(row.get("ars_steps_correct", 0) for row in rows)
    sensitive_total = sum(row.get("sensitive_steps_total", 0) for row in rows)
    sensitive_correct = sum(row.get("sensitive_steps_correct", 0) for row in rows)
    latencies = [
        latency
        for row in rows
        for latency in row.get("latencies_ms", [])
        if latency is not None
    ]
    successes = sum(bool(row.get("success")) for row in rows)

    return {
        "scenarios": len(rows),
        "successful_sessions": successes,
        "session_success_rate": _ratio(successes, len(rows)),
        "goal_turn_accuracy": _ratio(goal_correct, goal_total),
        "ars_step_accuracy": _ratio(ars_correct, ars_total),
        "sensitive_handoff_accuracy": _ratio(sensitive_correct, sensitive_total),
        "completion_accuracy": _ratio(
            sum(bool(row.get("completion_correct")) for row in rows),
            len(rows),
        ),
        "dangerous_action_count": sum(row.get("dangerous_actions", 0) for row in rows),
        "error_count": sum(row.get("error") is not None for row in rows),
        "latency_p50_ms": float(statistics.median(latencies)) if latencies else None,
        "latency_p95_ms": _percentile(latencies, 0.95),
    }

