import unittest
from unittest.mock import MagicMock
from core.preprocess.case_seg import segment_case_text_withname


class TestCaseSeg(unittest.TestCase):

    def test_segment_case_text_withname_success(self):
        model = MagicMock()
        model.generate_response.side_effect = ["Segment 1", "Segment 2"]

        case_text = "Full case text"
        criminals = ["John", "Jane"]

        result = segment_case_text_withname(model, case_text, criminals)
        expected = [
            {"name": "John", "description": "Segment 1"},
            {"name": "Jane", "description": "Segment 2"},
        ]
        self.assertEqual(result, expected)

    def test_segment_case_text_withname_empty_response(self):
        model = MagicMock()
        # If response is empty, it should fallback to original case text
        model.generate_response.return_value = ""

        case_text = "Full case text"
        criminals = ["John"]

        result = segment_case_text_withname(model, case_text, criminals)
        expected = [{"name": "John", "description": "Full case text"}]
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
