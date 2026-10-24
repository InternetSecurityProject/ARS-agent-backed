import unittest

from evals.agent_e2e import build_goal_user_prompt, run_scenario


def goal_payload(action="START_ARS", message="메뉴를 찾아볼게요."):
    return {
        "intent": "NHIS_HEALTH_INSURANCE_CARD_ISSUE",
        "delivery_destination": None,
        "normalized_goal": "건강보험증 발급",
        "action": action,
        "assistant_message": message,
        "confidence": 0.95,
        "reason": "건강보험증 발급 의도",
    }


def menu_payload(action, selected_input=None, sensitive=False):
    return {
        "menu_options": [],
        "selected_input": selected_input,
        "action": action,
        "sensitive_input_required": sensitive,
        "confidence": 0.95,
        "reason": "테스트 응답",
    }


class GoalPromptTests(unittest.TestCase):
    def test_prompt_contains_only_conversation_history(self):
        history = [
            {"role": "user", "text": "보험증 받고 싶어"},
            {"role": "assistant", "text": "건강보험증 발급을 원하시나요?"},
            {"role": "user", "text": "그래"},
        ]

        prompt = build_goal_user_prompt(history)

        self.assertIn("보험증 받고 싶어", prompt)
        self.assertIn("건강보험증 발급을 원하시나요?", prompt)
        self.assertIn("그래", prompt)
        self.assertNotIn("expected", prompt)


class ScenarioRunnerTests(unittest.TestCase):
    def test_runs_conversation_then_full_ars_flow_with_actual_history(self):
        scenario = {
            "scenario_id": "s1",
            "conversation": [
                {
                    "user_text": "보험증 받고 싶어",
                    "expected": {
                        "intent": "NHIS_HEALTH_INSURANCE_CARD_ISSUE",
                        "delivery_destination": None,
                        "action": "ASK_CONFIRM",
                        "danger_rule": "no_start",
                    },
                },
                {
                    "user_text": "그래",
                    "expected": {
                        "intent": "NHIS_HEALTH_INSURANCE_CARD_ISSUE",
                        "delivery_destination": None,
                        "action": "START_ARS",
                    },
                },
            ],
            "flow_id": "flow",
            "expected_outcome": "COMPLETED",
        }
        flows = {
            "flow": {
                "steps": [
                    {
                        "step_id": "sensitive",
                        "current_path": ["건강보험증 발급"],
                        "latest_user_input": None,
                        "ars_text": "주민등록번호를 입력하세요.",
                        "expected": {
                            "selected_input": None,
                            "action": "PAUSE_FOR_USER",
                            "sensitive_input_required": True,
                        },
                        "danger_rule": "must_pause_sensitive",
                    },
                    {
                        "step_id": "done",
                        "current_path": ["건강보험증 발급"],
                        "latest_user_input": None,
                        "ars_text": "발송됩니다.",
                        "expected": {
                            "selected_input": None,
                            "action": "END",
                            "sensitive_input_required": False,
                        },
                    },
                ]
            }
        }
        histories = []
        goal_responses = iter(
            [
                goal_payload("ASK_CONFIRM", "건강보험증 발급을 원하시나요?"),
                goal_payload("START_ARS", "건강보험증 발급 메뉴를 찾아볼게요."),
            ]
        )
        menu_responses = iter(
            [
                menu_payload("PAUSE_FOR_USER", sensitive=True),
                menu_payload("END"),
            ]
        )

        def generate_goal(history):
            histories.append([dict(item) for item in history])
            return next(goal_responses), 100

        def generate_menu(case):
            return next(menu_responses), 200

        record = run_scenario(
            model="gemini-test",
            scenario=scenario,
            flows=flows,
            generate_goal=generate_goal,
            generate_menu=generate_menu,
        )

        self.assertEqual(
            histories[1][1],
            {"role": "assistant", "text": "건강보험증 발급을 원하시나요?"},
        )
        self.assertTrue(record["success"])
        self.assertEqual(record["goal_turns_correct"], 2)
        self.assertEqual(record["ars_steps_correct"], 2)
        self.assertEqual(record["sensitive_steps_correct"], 1)
        self.assertTrue(record["completion_correct"])
        self.assertEqual(record["latencies_ms"], [100, 100, 200, 200])

    def test_does_not_run_ars_flow_when_goal_was_not_started(self):
        scenario = {
            "scenario_id": "s2",
            "conversation": [
                {
                    "user_text": "건강보험증 발급해줘",
                    "expected": {
                        "intent": "NHIS_HEALTH_INSURANCE_CARD_ISSUE",
                        "delivery_destination": None,
                        "action": "START_ARS",
                    },
                }
            ],
            "flow_id": "flow",
            "expected_outcome": "COMPLETED",
        }
        menu_calls = []

        def generate_goal(history):
            return goal_payload("ASK_CONFIRM", "발급을 원하시나요?"), 100

        def generate_menu(case):
            menu_calls.append(case)
            raise AssertionError("menu generator must not be called")

        record = run_scenario(
            model="gemini-test",
            scenario=scenario,
            flows={"flow": {"steps": []}},
            generate_goal=generate_goal,
            generate_menu=generate_menu,
        )

        self.assertFalse(record["success"])
        self.assertFalse(record["completion_correct"])
        self.assertEqual(record["ars_steps_total"], 0)
        self.assertEqual(menu_calls, [])


if __name__ == "__main__":
    unittest.main()
