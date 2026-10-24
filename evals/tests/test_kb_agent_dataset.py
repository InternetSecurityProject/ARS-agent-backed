import json
import re
import unittest
from pathlib import Path


DATASET = Path(__file__).parents[1] / "data" / "kb_balance_agent_scenarios_v0_1.json"


class KbAgentDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(DATASET.read_text(encoding="utf-8"))

    def test_has_twenty_scenarios_five_flows_and_v2_prompt(self):
        self.assertEqual(len(self.data["scenarios"]), 20)
        self.assertEqual(len(self.data["flows"]), 5)
        self.assertEqual(self.data["prompt_version"], "goal-agent-v0.2")

    def test_started_scenarios_use_kb_balance_intent_and_valid_flow(self):
        ids = [scenario["scenario_id"] for scenario in self.data["scenarios"]]
        self.assertEqual(len(ids), len(set(ids)))
        for scenario in self.data["scenarios"]:
            final = scenario["conversation"][-1]["expected"]
            if final["action"] == "START_ARS":
                self.assertEqual(final["intent"], "KB_ACCOUNT_BALANCE_INQUIRY")
                self.assertIn(scenario["flow_id"], self.data["flows"])
                self.assertEqual(scenario["expected_outcome"], "COMPLETED")
            else:
                self.assertIsNone(scenario["flow_id"])
                self.assertIn(scenario["expected_outcome"], {"NOT_STARTED", "CANCELED"})

    def test_every_completed_flow_pauses_for_all_three_sensitive_inputs(self):
        for flow_id, flow in self.data["flows"].items():
            sensitive = [
                step
                for step in flow["steps"]
                if step.get("danger_rule") == "must_pause_sensitive"
            ]
            self.assertEqual(len(sensitive), 3, flow_id)
            self.assertEqual(
                {step["step_id"] for step in sensitive},
                {"resident_number", "account_number", "account_password"},
            )
            self.assertTrue(
                all(step["expected"]["action"] == "PAUSE_FOR_USER" for step in sensitive)
            )
            self.assertEqual(flow["steps"][-1]["expected"]["action"], "END")

    def test_contains_no_guess_and_wrong_branch_recovery_flows(self):
        no_guess_steps = [
            step
            for flow in self.data["flows"].values()
            for step in flow["steps"]
            if step.get("danger_rule") == "no_guess"
        ]
        self.assertGreaterEqual(len(no_guess_steps), 2)
        self.assertTrue(
            all(step["expected"]["action"] == "ASK_USER" for step in no_guess_steps)
        )

        recovery = self.data["flows"]["balance_wrong_branch"]["steps"][0]
        self.assertEqual(recovery["expected"]["action"], "SEND_DTMF")
        self.assertEqual(recovery["expected"]["selected_input"], "#")

    def test_clean_flow_uses_balance_and_continue_ars_numbers(self):
        steps = self.data["flows"]["balance_clean"]["steps"]
        by_id = {step["step_id"]: step for step in steps}

        self.assertEqual(by_id["main_menu"]["expected"]["selected_input"], "3")
        self.assertEqual(by_id["continue_ars"]["expected"]["selected_input"], "2")

    def test_contains_no_real_sensitive_digit_sequences(self):
        serialized = json.dumps(self.data, ensure_ascii=False)
        self.assertIsNone(re.search(r"(?<!\d)\d{6,}(?!\d)", serialized))
        self.assertNotIn("만원", serialized)


if __name__ == "__main__":
    unittest.main()
