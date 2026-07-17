import unittest
from unittest.mock import patch
from core.models.openai.deepseek_v3 import DeepSeekChatbot
from core.models.openai.gpt4o_mini import GPT4OMiniChatbot


class TestOpenAIModels(unittest.TestCase):
    @patch.dict("os.environ", {"DEEPSEEK_API_KEY": "fake_key"})
    @patch("core.models.openai_base.OpenAI")
    def test_deepseek_init(self, mock_openai):
        chatbot = DeepSeekChatbot()
        self.assertEqual(chatbot.model_name, "deepseek-chat")
        self.assertEqual(chatbot.api_key, "fake_key")

    @patch.dict("os.environ", {"OPENAI_API_KEY": "fake_key"})
    @patch("core.models.openai_base.OpenAI")
    def test_gpt4o_mini_init(self, mock_openai):
        chatbot = GPT4OMiniChatbot()
        self.assertEqual(chatbot.model_name, "gpt-4o-mini")
        self.assertEqual(chatbot.api_key, "fake_key")


if __name__ == "__main__":
    unittest.main()
