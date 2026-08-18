"""Gemini model (Google GenAI SDK)"""

import os
from typing import Optional
from ..base import BaseModel


class GeminiChatbot(BaseModel):
    """Google Gemini chatbot using the google-genai SDK.

    Designed as the low-cost judge model for self-consistency sampling.
    gemini-2.0-flash-lite is the cheapest option (free tier available).

    Args:
        model_name: Gemini model ID. Options:
            - "gemini-2.0-flash-lite" (cheapest, good for classification)
            - "gemini-2.0-flash"      (balanced)
            - "gemini-2.5-flash"      (best quality)
        api_key: Gemini API key. Falls back to GEMINI_API_KEY env var.
        device: Ignored (kept for BaseModel interface consistency).
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-lite",
        device: str = "cpu",
        api_key: Optional[str] = None,
    ):
        super().__init__(model_name, device)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set the environment variable or pass api_key."
            )
        try:
            import google.genai as genai

            self._client = genai.Client(api_key=self.api_key)
        except ImportError as exc:
            raise ImportError(
                "google-genai is required. Install with: pip install google-genai"
            ) from exc

    def generate_response(
        self,
        user_input: str,
        max_length: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Generate a response from Gemini.

        Args:
            user_input: Prompt text.
            max_length: Max output tokens.
            temperature: Sampling temperature. Higher = more diverse (used for
                         self-consistency sampling, default 0.7).

        Returns:
            Generated text string.
        """
        try:
            from google.genai import types

            response = self._client.models.generate_content(
                model=self.model_name,
                contents=user_input,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_length,
                    temperature=temperature,
                ),
            )
            return response.text or ""
        except Exception as e:
            print(f"Gemini API error: {e}")
            return ""
