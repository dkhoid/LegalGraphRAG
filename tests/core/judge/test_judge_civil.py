import unittest
from unittest.mock import MagicMock
import json

from core.judge.judge_civil import format_law, format_fact, judge_civil, judge_civil_all


class TestJudgeCivil(unittest.TestCase):
    def test_format_law_success(self):
        law_used = [
            {
                "entry": "123",
                "disputes": ["Contract breach", "Compensation\nClaim"],
                "description": "This is a description.",
            },
            {"entry": "124", "description": "Another law."},
        ]
        expected = (
            "Điều luật 123 — Vấn đề pháp lý liên quan: Contract breach, Compensation Claim. Nội dung: This is a description.\n---\n"
            "Điều luật 124 — Vấn đề pháp lý liên quan: . Nội dung: Another law.\n---\n"
        )
        result = format_law(law_used)
        self.assertEqual(result, expected)

    def test_format_law_empty(self):
        self.assertEqual(format_law([]), "")

    def test_format_fact_success(self):
        facts = [
            {
                "dispute": ["Property damage", "Insurance"],
                "description": "Car was damaged",
            },
            {"description": "No dispute"},
        ]
        expected = (
            "Vấn đề pháp lý: Property damage, Insurance. Tình tiết vụ án: Car was damaged.\n"
            "Vấn đề pháp lý: . Tình tiết vụ án: No dispute.\n"
        )
        result = format_fact(facts)
        self.assertEqual(result, expected)

    def test_format_fact_empty(self):
        self.assertEqual(format_fact([]), "")

    def test_judge_civil_success(self):
        chatbot = MagicMock()
        # Assume the language model returns a list representation within the text
        chatbot.generate_response.return_value = "Here is the result: ['Issue A', 'Issue B']"

        result = judge_civil(chatbot, [], [], "Some case")
        self.assertIn("Issue A", result)
        self.assertIn("Issue B", result)
        self.assertEqual(len(result), 2)

    def test_judge_civil_with_think_tags(self):
        chatbot = MagicMock()
        chatbot.generate_response.return_value = (
            "<think>Suy nghĩ về [hành vi trái pháp luật] và [lỗi]</think>\n"
            "['Tranh chấp bồi thường', 'Tranh chấp hợp đồng']"
        )
        result = judge_civil(chatbot, [], [], "Some case")
        self.assertIn("Tranh chấp bồi thường", result)
        self.assertIn("Tranh chấp hợp đồng", result)
        self.assertEqual(len(result), 2)

    def test_judge_civil_parsing_error(self):
        chatbot = MagicMock()
        # Malformed list format to trigger an exception during eval()
        chatbot.generate_response.return_value = "This does not have a proper list representation"

        result = judge_civil(chatbot, [], [], "Some case")
        self.assertEqual(result, ["No applicable issue identified"])

    def test_judge_civil_all_success(self):
        chatbot = MagicMock()
        model_output = {
            "dispute_type": ["Contract"],
            "law_article": ["123"],
            "resolution": {"liability": "Yes", "compensation": "1000"},
        }
        expected_dict = {
            "dispute_type": ["Contract"],
            "law_article": ["Điều 123"],
            "resolution": {"liability": "Yes", "compensation": "1000"},
        }
        # Convert dict to JSON string block
        chatbot.generate_response.return_value = "JSON output: " + json.dumps(model_output)

        result = judge_civil_all(chatbot, [], [], "Some case")
        self.assertEqual(result, expected_dict)

    def test_judge_civil_all_parsing_error(self):
        chatbot = MagicMock()
        # Malformed JSON to trigger an exception
        chatbot.generate_response.return_value = "No json here, just plain text"

        result = judge_civil_all(chatbot, [], [], "Some case")
        expected_fallback = {
            "dispute_type": ["No applicable issue identified"],
            "law_article": ["N/A"],
            "resolution": {"liability": "N/A", "compensation": "N/A"},
        }
        self.assertEqual(result, expected_fallback)


if __name__ == "__main__":
    unittest.main()
