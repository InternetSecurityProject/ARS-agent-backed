import json
import os
import unittest

from evals.gemini_eval import build_record, build_user_prompt


class PromptTests(unittest.TestCase):
    def test_prompt_separates_ars_transcript_and_hides_answer(self):
        case = {
            "case_id": "c1",
            "user_goal": "건강보험증 발급",
            "current_path": ["메인"],
            "latest_user_input": None,
            "ars_text": "건강보험증 발급은 2번",
            "expected": {
                "selected_input": "2",
                "action": "SEND_DTMF",
                "sensitive_input_required": False,
            },
            "danger_rule": "secret_answer_metadata",
        }

        prompt = build_user_prompt(case)

        self.assertIn("<ars_transcript>\n건강보험증 발급은 2번\n</ars_transcript>", prompt)
        self.assertIn('user_goal:\n"건강보험증 발급"', prompt)
        self.assertNotIn("SEND_DTMF", prompt)
        self.assertNotIn("secret_answer_metadata", prompt)


class RecordTests(unittest.TestCase):
    def test_record_contains_evaluation_data_but_never_api_key(self):
        os.environ["GEMINI_API_KEY"] = "test-secret-key"
        case = {
            "case_id": "c1",
            "expected": {
                "selected_input": "2",
                "action": "SEND_DTMF",
                "sensitive_input_required": False,
            },
        }
        response = {
            "menu_options": [{"input": "2", "meaning": "건강보험증 발급"}],
            "selected_input": "2",
            "action": "SEND_DTMF",
            "sensitive_input_required": False,
            "confidence": 0.99,
            "reason": "목적과 메뉴가 일치함",
        }

        record = build_record(
            model="gemini-test",
            case=case,
            run_number=1,
            response=response,
            latency_ms=123,
            input_tokens=100,
            output_tokens=20,
        )

        serialized = json.dumps(record, ensure_ascii=False)
        self.assertNotIn("test-secret-key", serialized)
        self.assertEqual(record["case_id"], "c1")
        self.assertTrue(record["schema_ok"])
        self.assertTrue(record["score"]["correct"])


if __name__ == "__main__":
    unittest.main()
