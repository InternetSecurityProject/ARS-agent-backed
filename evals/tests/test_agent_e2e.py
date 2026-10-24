import unittest

from evals.agent_e2e import (
    build_goal_batch_prompt,
    build_goal_user_prompt,
    build_menu_batch_prompt,
    run_scenario,
    run_scenarios_batched,
)


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

    def test_batch_prompts_keep_ids_but_hide_answer_metadata(self):
        goal_prompt = build_goal_batch_prompt(
            [{"id": "s1", "history": [{"role": "user", "text": "보험증"}]}]
        )
        menu_prompt = build_menu_batch_prompt(
            [
                {
                    "id": "s1",
                    "case": {
                        "user_goal": "건강보험증 발급",
                        "current_path": [],
                        "latest_user_input": None,
                        "ars_text": "건강보험증 발급은 2번",
                        "expected": {"action": "SEND_DTMF"},
                        "danger_rule": "secret_rule",
                    },
                }
            ]
        )

        self.assertIn('"id": "s1"', goal_prompt)
        self.assertIn('"id": "s1"', menu_prompt)
        self.assertNotIn("expected", menu_prompt)
        self.assertNotIn("secret_rule", menu_prompt)


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

    def test_batched_runner_processes_turns_in_waves_without_future_leakage(self):
        scenarios = [
            {
                "scenario_id": "s1",
                "conversation": [
                    {
                        "user_text": "보험증 필요해",
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
            },
            {
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
            },
        ]
        flows = {
            "flow": {
                "steps": [
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
                    }
                ]
            }
        }
        goal_batches = []
        menu_batches = []

        def generate_goal_batch(items):
            goal_batches.append(items)
            if len(goal_batches) == 1:
                return {
                    "s1": (goal_payload("ASK_CONFIRM", "건강보험증인가요?"), 100),
                    "s2": (goal_payload("START_ARS"), 100),
                }
            return {"s1": (goal_payload("START_ARS"), 120)}

        def generate_menu_batch(items):
            menu_batches.append(items)
            return {
                item["id"]: (menu_payload("END"), 200)
                for item in items
            }

        records = run_scenarios_batched(
            model="gemini-test",
            scenarios=scenarios,
            flows=flows,
            generate_goal_batch=generate_goal_batch,
            generate_menu_batch=generate_menu_batch,
        )

        self.assertEqual([len(batch) for batch in goal_batches], [2, 1])
        self.assertEqual(
            goal_batches[1][0]["history"],
            [
                {"role": "user", "text": "보험증 필요해"},
                {"role": "assistant", "text": "건강보험증인가요?"},
                {"role": "user", "text": "그래"},
            ],
        )
        self.assertEqual(len(menu_batches), 1)
        self.assertEqual({item["id"] for item in menu_batches[0]}, {"s1", "s2"})
        self.assertTrue(all(record["success"] for record in records))


if __name__ == "__main__":
    unittest.main()
