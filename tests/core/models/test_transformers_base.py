import unittest
from unittest.mock import patch, MagicMock
from core.models.transformers_base import TransformersBaseModel


class TestTransformersBaseModel(unittest.TestCase):
    def test_init(self):
        model = TransformersBaseModel(model_name="test_model", device="cpu")
        self.assertEqual(model.model_name, "test_model")
        self.assertEqual(model.device, "cpu")

    def test_generate_response_not_implemented(self):
        model = TransformersBaseModel(model_name="test_model")
        with self.assertRaises(NotImplementedError):
            model.generate_response("Hello")

    @patch("core.models.transformers_base.gc")
    @patch("core.models.transformers_base.torch")
    def test_release_model(self, mock_torch, mock_gc):
        model = TransformersBaseModel(model_name="test_model")
        model.model = MagicMock()
        model.tokenizer = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        model.release_model()
        mock_gc.collect.assert_called_once()
        mock_torch.cuda.empty_cache.assert_called_once()


if __name__ == "__main__":
    unittest.main()
