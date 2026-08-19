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
            "Article 123 — Applicable issues: Contract breach, Compensation Claim. Content: This is a description.\n---\n"
            "Article 124 — Applicable issues: . Content: Another law.\n---\n"
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
            "Legal issue: Property damage, Insurance. Fact description: Car was damaged.\n"
            "Legal issue: . Fact description: No dispute.\n"
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

    def test_judge_civil_parsing_error(self):
        chatbot = MagicMock()
        # Malformed list format to trigger an exception during eval()
        chatbot.generate_response.return_value = "This does not have a proper list representation"

        result = judge_civil(chatbot, [], [], "Some case")
        self.assertEqual(result, ["No applicable issue identified"])

    def test_judge_civil_all_success(self):
        chatbot = MagicMock()
        expected_dict = {
            "dispute_type": ["Contract"],
            "law_article": ["123"],
            "resolution": {"liability": "Yes", "compensation": "1000"},
        }
        # Convert dict to JSON string block
        chatbot.generate_response.return_value = "JSON output: " + json.dumps(expected_dict)

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
