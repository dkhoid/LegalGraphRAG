import unittest
from unittest.mock import MagicMock
from core.preprocess.get_features import get_features


class TestGetFeatures(unittest.TestCase):

    def test_get_features_success(self):
        model = MagicMock()
        model.generate_response.return_value = 'Some text {"feature1": "value1"} more text'

        cases = {"name": "Test Name", "description": "Test Fact"}
        result = get_features(model, cases)
        self.assertEqual(result, {"feature1": "value1"})

    def test_get_features_no_json(self):
        model = MagicMock()
        model.generate_response.return_value = "Some text with no json"

        cases = {"name": "Test Name", "description": "Test Fact"}
        result = get_features(model, cases)
        self.assertEqual(result, {})

    def test_get_features_invalid_json(self):
        model = MagicMock()
        model.generate_response.return_value = 'Here is invalid json { "feature": "value" '

        cases = {"name": "Test Name", "description": "Test Fact"}
        result = get_features(model, cases)
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
