"""
Centralized constants for LegalGraphRAG.
"""

# Model Provider Configurations
PROVIDER_CONFIGS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model_name": "gemini-1.5-flash",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model_name": "deepseek-chat",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model_name": "gpt-4o-mini",
    },
}

# Generation Defaults
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.1
DEFAULT_CASE_FACT_LIMIT = 4096

# Retrieval Defaults
DEFAULT_MAX_JUDGE_LAWS = 8
DEFAULT_RRF_K = 60
