from typing import List, Dict, Any
from pydantic import BaseModel, Field


# Input Models
class CaseRequest(BaseModel):
    fact: str = Field(..., max_length=5000, description="Tình tiết vụ án cần phân tích")
    top_k: int = Field(5, ge=1, le=10, description="Số lượng kết quả liên quan tối đa")
    api_key: str | None = Field(None, description="API Key tùy chọn cho request hiện tại")
    provider: str | None = Field(None, description="Provider tùy chọn (openai, gemini, deepseek)")


class ChatRequest(BaseModel):
    query: str = Field(..., max_length=5000, description="Câu hỏi pháp lý")
    api_key: str | None = Field(None, description="API Key tùy chọn cho request hiện tại")
    provider: str | None = Field(None, description="Provider tùy chọn (openai, gemini, deepseek)")


class APIKeyRequest(BaseModel):
    api_key: str = Field(..., description="API Key mới cần cập nhật")


# Output Models
class GeneratePromptResponse(BaseModel):
    retrieved_laws: List[Dict[str, Any]]
    retrieved_facts: List[Dict[str, Any]]
    prompt: str


class AnalyzeCivilResponse(BaseModel):
    retrieved_laws: List[Dict[str, Any]]
    retrieved_facts: List[Dict[str, Any]]
    prompt: str
    analysis_result: Dict[str, Any]
