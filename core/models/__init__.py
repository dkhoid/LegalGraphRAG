"""Unified import interface for model module (with lazy loading for heavy transformer models)"""

# Base classes
from .base import BaseModel
from .openai_base import OpenAIBaseModel

# OpenAI / Cloud API type models (lightweight, zero torch dependency)
from .openai import DeepSeekChatbot, GPT4OMiniChatbot, GeminiChatbot

__all__ = [
    # Base classes
    "BaseModel",
    "OpenAIBaseModel",
    "TransformersBaseModel",
    # OpenAI / Cloud API type
    "DeepSeekChatbot",
    "GPT4OMiniChatbot",
    "GeminiChatbot",
    # Transformers type (lazy loaded)
    "QwenChatbot",
    "Qwen2Chatbot",
    "GemmaChatbot",
    "GlmChatbot",
    "InternlmChatbot",
]


def __getattr__(name: str):
    """Lazy load heavy PyTorch/Transformers classes only when accessed."""
    _transformer_names = {
        "TransformersBaseModel",
        "QwenChatbot",
        "Qwen2Chatbot",
        "GemmaChatbot",
        "GlmChatbot",
        "InternlmChatbot",
    }
    if name in _transformer_names:
        from .transformers_base import TransformersBaseModel
        from .transformers import (
            QwenChatbot,
            Qwen2Chatbot,
            GemmaChatbot,
            GlmChatbot,
            InternlmChatbot,
        )

        _exports = {
            "TransformersBaseModel": TransformersBaseModel,
            "QwenChatbot": QwenChatbot,
            "Qwen2Chatbot": Qwen2Chatbot,
            "GemmaChatbot": GemmaChatbot,
            "GlmChatbot": GlmChatbot,
            "InternlmChatbot": InternlmChatbot,
        }
        globals().update(_exports)
        return _exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
