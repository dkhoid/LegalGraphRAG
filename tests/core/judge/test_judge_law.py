import unittest
from unittest.mock import MagicMock
from core.judge.judge_law import judge_law


class TestJudgeLaw(unittest.TestCase):

    def test_judge_law_str_true(self):
        chatbot = MagicMock()
        chatbot.generate_response.return_value = "This is True and applicable."

        result, msg = judge_law(chatbot, "A test case", "A law string")
        self.assertTrue(result)
        self.assertEqual(msg, "")

    def test_judge_law_str_false(self):
        chatbot = MagicMock()
        chatbot.generate_response.return_value = "This is false and not applicable."

        result, msg = judge_law(chatbot, "A test case", "A law string")
        self.assertFalse(result)
        self.assertEqual(msg, "")

    def test_judge_law_dict_true(self):
        chatbot = MagicMock()
        # The function generate_response is called multiple times.
        # First for each item in judge_dep, then for the final summary.
        # We can just return "true" for all to make it pass the final check.
        chatbot.generate_response.return_value = "Yes, it is true"

        law_dict = {
            "judge_dep": ["Element 1", "Element 2"],
            "description": "Law description\n",
            "related_laws": ["Law A"],
        }

        result, msg = judge_law(chatbot, "A test case", law_dict)
        self.assertTrue(result)
        self.assertEqual(msg, "Yes, it is true")
        # Should be called len(judge_dep) + 1 times
        self.assertEqual(chatbot.generate_response.call_count, 3)

    def test_judge_law_dict_false(self):
        chatbot = MagicMock()
        chatbot.generate_response.return_value = "No, it is false"

        law_dict = {
            "judge_dep": ["Element 1"],
            "description": "Law description\n",
            "related_laws": ["Law A"],
        }

        result, msg = judge_law(chatbot, "A test case", law_dict)
        self.assertFalse(result)
        self.assertEqual(msg, "No, it is false")


if __name__ == "__main__":
    unittest.main()
