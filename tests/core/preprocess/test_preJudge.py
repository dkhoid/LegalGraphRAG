import unittest
from unittest.mock import MagicMock
from core.preprocess.preJudge import pre_judge


class TestPreJudge(unittest.TestCase):

    def test_pre_judge_success(self):
        model = MagicMock()
        model.generate_response.return_value = (
            "Here is the list: ['Candidate 1', 'Candidate 2', 'Candidate 3', 'Candidate 4']"
        )

        result = pre_judge(model, "A case")
        # Should return only the first 3
        self.assertEqual(result, ["Candidate 1", "Candidate 2", "Candidate 3"])

    def test_pre_judge_not_a_list(self):
        model = MagicMock()
        # Valid python eval, but not a list of strings
        model.generate_response.return_value = "Here is something else: [1, 2, 3]"

        result = pre_judge(model, "A case")
        self.assertEqual(result, [])

    def test_pre_judge_parse_error(self):
        model = MagicMock()
        # Invalid python eval
        model.generate_response.return_value = "No bracket or list"

        result = pre_judge(model, "A case")
        self.assertEqual(result, [])

    def test_pre_judge_invalid_eval(self):
        model = MagicMock()
        model.generate_response.return_value = "List [this is not valid]"

        result = pre_judge(model, "A case")
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
