import unittest
from unittest.mock import patch, MagicMock
from core.models.openai_base import OpenAIBaseModel


class TestOpenAIBaseModel(unittest.TestCase):
    @patch.dict("os.environ", {"OPENAI_API_KEY": "fake_key"})
    @patch("core.models.openai_base.OpenAI")
    def test_init_success(self, mock_openai):
        model = OpenAIBaseModel(model_name="test_model", base_url="http://fake")
        self.assertEqual(model.api_key, "fake_key")
        self.assertEqual(model.base_url, "http://fake")
        mock_openai.assert_called_once()

    @patch.dict("os.environ", clear=True)
    def test_init_no_api_key(self):
        # OpenAIBaseModel allows initialization without API key (to allow dynamic web input)
        model = OpenAIBaseModel(model_name="test_model", base_url="http://fake")
        self.assertEqual(model.api_key, "")
        self.assertIsNone(model.client)
        # But calling generate_response without key will raise ValueError
        with self.assertRaises(ValueError):
            model.generate_response("test prompt")

    @patch.dict("os.environ", {"OPENAI_API_KEY": "fake_key"})
    def test_init_no_base_url(self):
        with self.assertRaises(ValueError):
            OpenAIBaseModel(model_name="test_model")

    @patch.dict("os.environ", {"OPENAI_API_KEY": "fake_key"})
    @patch("core.models.openai_base.OpenAI")
    def test_generate_response_success(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Mocked response"))]
        mock_client.chat.completions.create.return_value = mock_response

        model = OpenAIBaseModel(model_name="test_model", base_url="http://fake")
        res = model.generate_response("Hello")
        self.assertEqual(res, "Mocked response")

    @patch.dict("os.environ", {"OPENAI_API_KEY": "fake_key"})
    @patch("core.models.openai_base.OpenAI")
    def test_generate_response_exception(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")

        model = OpenAIBaseModel(model_name="test_model", base_url="http://fake")
        res = model.generate_response("Hello")
        self.assertEqual(res, "API call failed")


if __name__ == "__main__":
    unittest.main()
