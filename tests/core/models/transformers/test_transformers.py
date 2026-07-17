import unittest
from unittest.mock import patch, MagicMock
from core.models.transformers.gemma3_model import GemmaChatbot
from core.models.transformers.glm4 import GlmChatbot
from core.models.transformers.Internlm3 import InternlmChatbot
from core.models.transformers.qwen2_5_model import Qwen2Chatbot
from core.models.transformers.qwen3_model import QwenChatbot


class TestTransformersModels(unittest.TestCase):
    def _mock_setup_auto_model(self, mock_auto_model, mock_auto_tokenizer):
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        return mock_model, mock_tokenizer

    def _test_generation(self, chatbot, mock_model, mock_tokenizer, method="AutoModelForCausalLM"):
        if method == "AutoProcessor":
            mock_inputs = {"input_ids": MagicMock()}
            mock_inputs["input_ids"].shape = [1, 3]

            mock_apply_chat_template_ret = MagicMock()
            mock_apply_chat_template_ret.to.return_value = mock_inputs
            mock_tokenizer.apply_chat_template.return_value = mock_apply_chat_template_ret

            mock_model.generate.return_value = [[1, 2, 3, 4, 5]]
            mock_tokenizer.decode.return_value = "Response"
        else:
            from transformers import BatchEncoding
            import torch

            mock_tokenizer.apply_chat_template.return_value = "Prompt"
            mock_inputs = BatchEncoding({"input_ids": torch.tensor([[1, 2, 3]])})
            mock_ret = MagicMock()
            mock_ret.to.return_value = mock_inputs
            mock_tokenizer.return_value = mock_ret

            mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
            mock_tokenizer.decode.return_value = "Response"

        res = chatbot.generate_response("Hello")
        self.assertTrue(isinstance(res, str))

    @patch("core.models.transformers.Internlm3.AutoTokenizer")
    @patch("core.models.transformers.Internlm3.AutoModelForCausalLM")
    def test_internlm(self, mock_model, mock_tokenizer):
        m, t = self._mock_setup_auto_model(mock_model, mock_tokenizer)
        chatbot = InternlmChatbot(model_name="test", device="cpu")
        self._test_generation(chatbot, m, t)

    @patch("core.models.transformers.glm4.AutoTokenizer")
    @patch("core.models.transformers.glm4.AutoModelForCausalLM")
    def test_glm4(self, mock_model, mock_tokenizer):
        m, t = self._mock_setup_auto_model(mock_model, mock_tokenizer)
        chatbot = GlmChatbot(model_name="test", device="cpu")
        self._test_generation(chatbot, m, t)

    @patch("core.models.transformers.qwen2_5_model.AutoTokenizer")
    @patch("core.models.transformers.qwen2_5_model.AutoModelForCausalLM")
    def test_qwen2_5(self, mock_model, mock_tokenizer):
        m, t = self._mock_setup_auto_model(mock_model, mock_tokenizer)
        chatbot = Qwen2Chatbot(model_name="test", device="cpu")
        self._test_generation(chatbot, m, t)

    @patch("core.models.transformers.qwen3_model.AutoTokenizer")
    @patch("core.models.transformers.qwen3_model.AutoModelForCausalLM")
    def test_qwen3(self, mock_model, mock_tokenizer):
        m, t = self._mock_setup_auto_model(mock_model, mock_tokenizer)
        chatbot = QwenChatbot(model_name="test", device="cpu")
        self._test_generation(chatbot, m, t)

    @patch("core.models.transformers.gemma3_model.AutoProcessor")
    @patch("core.models.transformers.gemma3_model.Gemma3ForConditionalGeneration")
    def test_gemma(self, mock_model, mock_processor):
        m, p = self._mock_setup_auto_model(mock_model, mock_processor)
        chatbot = GemmaChatbot(model_name="test", device="cpu")
        self._test_generation(chatbot, m, p, method="AutoProcessor")


if __name__ == "__main__":
    unittest.main()
