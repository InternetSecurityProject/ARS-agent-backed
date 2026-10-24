import json
import re
import unittest
from pathlib import Path


DATASET = Path(__file__).parents[1] / "data" / "nhis_agent_scenarios_v0_1.json"
PROMPT = Path(__file__).parents[1] / "prompts" / "goal_agent_v0_1.txt"


class AgentDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(DATASET.read_text(encoding="utf-8"))

    def test_contains_at_least_twenty_scenarios_and_four_flows(self):
        self.assertGreaterEqual(len(self.data["scenarios"]), 20)
        self.assertGreaterEqual(len(self.data["flows"]), 4)

    def test_scenario_ids_are_unique_and_start_actions_have_a_flow(self):
        ids = [scenario["scenario_id"] for scenario in self.data["scenarios"]]
        self.assertEqual(len(ids), len(set(ids)))

        for scenario in self.data["scenarios"]:
            self.assertTrue(scenario["conversation"])
            final_action = scenario["conversation"][-1]["expected"]["action"]
            if final_action == "START_ARS":
                self.assertIn(scenario["flow_id"], self.data["flows"])
                self.assertEqual(scenario["expected_outcome"], "COMPLETED")
            else:
                self.assertIsNone(scenario["flow_id"])
                self.assertIn(scenario["expected_outcome"], {"NOT_STARTED", "CANCELED"})

    def test_each_flow_covers_sensitive_handoff_and_completion(self):
        for flow_id, flow in self.data["flows"].items():
            actions = [step["expected"]["action"] for step in flow["steps"]]
            self.assertIn("PAUSE_FOR_USER", actions, flow_id)
            self.assertEqual(actions[-1], "END", flow_id)
            self.assertTrue(any(step.get("danger_rule") == "must_pause_sensitive" for step in flow["steps"]))

    def test_expected_goal_fields_are_complete(self):
        required = {"intent", "delivery_destination", "action"}
        for scenario in self.data["scenarios"]:
            for turn in scenario["conversation"]:
                self.assertEqual(set(turn["expected"]) - {"danger_rule"}, required)

    def test_dataset_has_no_long_digit_sequences_or_phone_numbers(self):
        serialized = json.dumps(self.data, ensure_ascii=False)
        self.assertIsNone(re.search(r"(?<!\d)\d{6,}(?!\d)", serialized))
        self.assertIsNone(re.search(r"01[016789][- ]?\d{3,4}[- ]?\d{4}", serialized))

    def test_goal_prompt_exists(self):
        prompt = PROMPT.read_text(encoding="utf-8")
        self.assertGreater(len(prompt.strip()), 300)


if __name__ == "__main__":
    unittest.main()
