import unittest
from core.models.base import BaseModel


class DummyModel(BaseModel):
    def generate_response(self, user_input, max_length=4096):
        return "response"


class TestBaseModel(unittest.TestCase):
    def test_init(self):
        model = DummyModel("test_model", "cpu")
        self.assertEqual(model.model_name, "test_model")
        self.assertEqual(model.device, "cpu")
        self.assertEqual(model.generate_response("test"), "response")
        model.release_model()


if __name__ == "__main__":
    unittest.main()
