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


class PartyAnalysisResult(BaseModel):
    name: str = Field(description="Tên của bên (Nguyên đơn, Bị đơn)")
    description: str = Field(description="Mô tả về bên này")
    judge_result: Dict[str, Any] = Field(description="Kết quả phán quyết từ LLM")
    used_laws: List[Dict[str, Any]] = Field(description="Các luật thực sự được áp dụng")
    used_facts: List[Dict[str, Any]] = Field(description="Các tình tiết thực sự được dùng")
    confidence: Dict[str, Any] = Field(description="Chi tiết độ tin cậy của AI")
    reasoning_trace: Dict[str, Any] = Field(description="Logs hệ thống")


class AnalyzeCivilResponse(BaseModel):
    results: List[PartyAnalysisResult]


class GraphInspectRequest(BaseModel):
    query: str = Field(
        ..., max_length=5000, description="Câu hỏi hoặc tình tiết vụ án cần trực quan hóa đồ thị"
    )
    top_k: int = Field(5, ge=1, le=20, description="Số lượng nút tối đa cần lấy")


class GraphInspectResponse(BaseModel):
    query: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    stats: Dict[str, Any]
