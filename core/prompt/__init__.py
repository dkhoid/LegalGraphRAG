"""Unified import interface for prompt module — English only"""

# Preprocess prompts
from .preprocess import (
    GET_FEATURES_PROMPT,
    CASE_SEG_PROMPT,
    PRE_JUDGE_PROMPT,
)

# Judge prompts
from .judge import (
    JUDGE_LAW_PROMPT,
    JUDGE_LAW_PROMPT0,
    JUDGE_LAW_PROMPT1,
    JUDGE_CIVIL_PROMPT,
    JUDGE_CIVIL_ALL_PROMPT,
)

# Retrieval prompts
from .retrieval import (
    RETRIEVE_LAW_PROMPT,
)

# Graph prompts
from .graph import (
    SUMMARIZE_TEXTS_PROMPT,
    RERANK_CLUSTERS_PROMPT_TEMPLATE,
    RERANK_PROMPT_TEMPLATE,
)

_PROMPTS = {
    "GET_FEATURES_PROMPT": GET_FEATURES_PROMPT,
    "CASE_SEG_PROMPT": CASE_SEG_PROMPT,
    "PRE_JUDGE_PROMPT": PRE_JUDGE_PROMPT,
    "JUDGE_LAW_PROMPT": JUDGE_LAW_PROMPT,
    "JUDGE_LAW_PROMPT0": JUDGE_LAW_PROMPT0,
    "JUDGE_LAW_PROMPT1": JUDGE_LAW_PROMPT1,
    "JUDGE_CIVIL_PROMPT": JUDGE_CIVIL_PROMPT,
    "JUDGE_CIVIL_ALL_PROMPT": JUDGE_CIVIL_ALL_PROMPT,
    "RETRIEVE_LAW_PROMPT": RETRIEVE_LAW_PROMPT,
    "SUMMARIZE_TEXTS_PROMPT": SUMMARIZE_TEXTS_PROMPT,
    "SUMMARIZE_TEXTS_INPUT_PREFIX": "\n**Bây giờ hãy xử lý dữ liệu đầu vào sau**: \n",
    "RERANK_CLUSTERS_PROMPT_TEMPLATE": RERANK_CLUSTERS_PROMPT_TEMPLATE,
    "RERANK_PROMPT_TEMPLATE": RERANK_PROMPT_TEMPLATE,
    "GET_FEATURES_INPUT_TEMPLATE": "\nTên đương sự: {name}\nTình tiết vụ án: {fact}",
    "JUDGE_CIVIL_ALL_INPUT_TEMPLATE": (
        "Đầu vào:\nQuy định pháp luật:\n-----\n{law}\n-----\n"
        "Vụ án cần xét xử:\n-----\n{case}\n-----\nĐầu ra:"
    ),
}


def set_prompt_language(language: str) -> None:
    """Kept for backward compatibility. Language is now always English."""
    pass


def get_prompt(name: str, language: str = None) -> str:
    """Return a prompt by name. Always returns English prompts."""
    try:
        return _PROMPTS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown prompt: {name}") from exc


__all__ = [
    "set_prompt_language",
    "get_prompt",
    "GET_FEATURES_PROMPT",
    "CASE_SEG_PROMPT",
    "PRE_JUDGE_PROMPT",
    "JUDGE_LAW_PROMPT",
    "JUDGE_LAW_PROMPT0",
    "JUDGE_LAW_PROMPT1",
    "JUDGE_CIVIL_PROMPT",
    "JUDGE_CIVIL_ALL_PROMPT",
    "RETRIEVE_LAW_PROMPT",
    "SUMMARIZE_TEXTS_PROMPT",
    "RERANK_CLUSTERS_PROMPT_TEMPLATE",
    "RERANK_PROMPT_TEMPLATE",
]
