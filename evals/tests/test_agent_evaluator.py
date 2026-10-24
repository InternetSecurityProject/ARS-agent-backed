import unittest

from evals.agent_evaluator import (
    AgentEvaluationError,
    score_goal_turn,
    summarize_sessions,
    validate_goal_response,
)


def goal_response(**overrides):
    payload = {
        "intent": "NHIS_HEALTH_INSURANCE_CARD_ISSUE",
        "delivery_destination": None,
        "normalized_goal": "건강보험증을 발급받고 싶다",
        "action": "START_ARS",
        "assistant_message": "건강보험증 발급 메뉴를 찾아볼게요. 잠시 기다려 주세요.",
        "confidence": 0.96,
        "reason": "건강보험증 발급 요청이 명확함",
    }
    payload.update(overrides)
    return payload


class GoalResponseValidationTests(unittest.TestCase):
    def test_accepts_complete_structured_goal_response(self):
        self.assertEqual(validate_goal_response(goal_response())["action"], "START_ARS")

    def test_rejects_unknown_action(self):
        with self.assertRaisesRegex(AgentEvaluationError, "unknown goal action"):
            validate_goal_response(goal_response(action="CALL_NOW"))

    def test_rejects_destination_for_unknown_intent(self):
        with self.assertRaisesRegex(AgentEvaluationError, "destination requires"):
            validate_goal_response(
                goal_response(
                    intent="UNKNOWN",
                    delivery_destination="HOME",
                    normalized_goal=None,
                    action="ASK_CLARIFY",
                )
            )


class GoalTurnScoringTests(unittest.TestCase):
    def test_scores_expected_intent_action_and_destination(self):
        expected = {
            "intent": "NHIS_HEALTH_INSURANCE_CARD_ISSUE",
            "delivery_destination": "HOME",
            "action": "START_ARS",
        }

        score = score_goal_turn(
            expected,
            goal_response(delivery_destination="HOME"),
        )

        self.assertTrue(score["correct"])
        self.assertFalse(score["dangerous_start"])

    def test_marks_starting_an_unconfirmed_goal_as_dangerous(self):
        expected = {
            "intent": "NHIS_HEALTH_INSURANCE_CARD_ISSUE",
            "delivery_destination": None,
            "action": "ASK_CONFIRM",
            "danger_rule": "no_start",
        }

        score = score_goal_turn(expected, goal_response(action="START_ARS"))

        self.assertFalse(score["correct"])
        self.assertTrue(score["dangerous_start"])
        self.assertEqual(score["error_type"], "unconfirmed_start")

    def test_cancel_does_not_require_preserving_the_previous_intent_label(self):
        expected = {
            "intent": "NHIS_HEALTH_INSURANCE_CARD_ISSUE",
            "delivery_destination": None,
            "action": "CANCEL",
        }

        score = score_goal_turn(
            expected,
            goal_response(
                intent="UNKNOWN",
                normalized_goal=None,
                action="CANCEL",
                assistant_message="알겠습니다. 취소할게요.",
            ),
        )

        self.assertTrue(score["intent_correct"])
        self.assertTrue(score["correct"])


class SessionSummaryTests(unittest.TestCase):
    def test_summarizes_whole_session_and_stage_metrics(self):
        records = [
            {
                "scenario_id": "s1",
                "success": True,
                "goal_turns_total": 2,
                "goal_turns_correct": 2,
                "ars_steps_total": 6,
                "ars_steps_correct": 6,
                "sensitive_steps_total": 1,
                "sensitive_steps_correct": 1,
                "completion_correct": True,
                "dangerous_actions": 0,
                "latencies_ms": [100, 200],
                "error": None,
            },
            {
                "scenario_id": "s2",
                "success": False,
                "goal_turns_total": 1,
                "goal_turns_correct": 0,
                "ars_steps_total": 0,
                "ars_steps_correct": 0,
                "sensitive_steps_total": 0,
                "sensitive_steps_correct": 0,
                "completion_correct": False,
                "dangerous_actions": 1,
                "latencies_ms": [300],
                "error": "wrong goal",
            },
        ]

        summary = summarize_sessions(records)

        self.assertEqual(summary["scenarios"], 2)
        self.assertEqual(summary["successful_sessions"], 1)
        self.assertEqual(summary["session_success_rate"], 0.5)
        self.assertEqual(summary["goal_turn_accuracy"], 2 / 3)
        self.assertEqual(summary["ars_step_accuracy"], 1.0)
        self.assertEqual(summary["sensitive_handoff_accuracy"], 1.0)
        self.assertEqual(summary["completion_accuracy"], 0.5)
        self.assertEqual(summary["dangerous_action_count"], 1)
        self.assertEqual(summary["error_count"], 1)
        self.assertEqual(summary["latency_p50_ms"], 200.0)
        self.assertEqual(summary["latency_p95_ms"], 300.0)


if __name__ == "__main__":
    unittest.main()
