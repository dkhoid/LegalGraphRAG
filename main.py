import json
import os
from contextlib import asynccontextmanager
from typing import List, Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from core.LegalGraphRAG import LegalGraphRAG, LegalGraphRAGConfig
from core.graph_construct.feature_graph import query_similar_laws_naive, query_similar_nodes_naive
from core.utils.logger import logger

# Global RAG system
rag_system = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_system
    try:
        logger.info("Loading LegalGraphRAG configuration and initializing...")
        config = LegalGraphRAGConfig.from_env_file(".env")

        # Override some properties based on env directly if needed (porting from app.py)
        model_name = os.getenv("model_name")
        if model_name:
            config.model.model_name = model_name
            config.model.api_key = os.getenv(f"{model_name}_api_key", os.getenv("api_key"))
            config.model.base_url = os.getenv(f"{model_name}_base_url", os.getenv("base_url"))

        graph_db_path = os.getenv("graph_db_path")
        if graph_db_path:
            config.graph.graph_db_path = graph_db_path

        rag_system = LegalGraphRAG(config=config)

        # Keep app.py's specific retrieve settings
        rag_system.config.retrieve.to_dict = lambda: {
            "method": "vector",
            "direct_retrieve_top_k": 3,
        }

        logger.info("LegalGraphRAG initialized successfully.")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        rag_system = None

    yield

    # Cleanup on shutdown
    logger.info("Shutting down LegalGraphRAG API...")


app = FastAPI(
    title="LegalGraphRAG API",
    description="API for Vietnamese Civil Law analysis using Graph RAG",
    lifespan=lifespan,
)

# Fix CORS - DO NOT USE ["*"] IN PRODUCTION
# Allow localhost for development. You can update this via env vars later.
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(
    ","
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Mount the web directories
os.makedirs("web", exist_ok=True)
os.makedirs("static", exist_ok=True)
app.mount("/web", StaticFiles(directory="web"), name="web")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    # Redirect to the better, more specific UI (web)
    return RedirectResponse(url="/web/index.html")


# Input Models with Security (max_length to prevent token exhaustion / DOS)
class CaseRequest(BaseModel):
    fact: str = Field(..., max_length=5000, description="Tình tiết vụ án cần phân tích")
    top_k: int = Field(5, ge=1, le=10, description="Số lượng kết quả liên quan tối đa")


class ChatRequest(BaseModel):
    query: str = Field(..., max_length=5000, description="Câu hỏi pháp lý")


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


# Helper functions
def format_law(laws: List[Dict[str, Any]]) -> str:
    res = ""
    for law in laws:
        law_id = law.get("entry", law.get("id", "Unknown"))
        description = law.get("description", law.get("text", ""))
        res += f"Article {law_id}: {description}\n---\n"
    return res


def format_fact(facts: List[Dict[str, Any]]) -> str:
    res = ""
    for fact in facts:
        res += f"Similar Case Fact: {fact.get('description', '')}\n---\n"
    return res


def build_civil_prompt(
    fact: str, laws: List[Dict[str, Any]], similar_facts: List[Dict[str, Any]]
) -> str:
    prompt_template = """
Bạn là một chuyên gia tư vấn pháp lý chuyên về Luật Dân sự Việt Nam.
Dựa trên các quy định pháp luật (điều luật) và các tiền lệ (án lệ/vụ án tương tự) được cung cấp, hãy phân tích tình tiết vụ án sau đây và đưa ra hướng giải quyết.

**Thông tin đầu vào:**
- Quy định pháp luật (Laws):
{formatted_laws}

- Án lệ/Vụ án tương tự (Similar Cases):
{formatted_facts}

- Tình tiết vụ án cần phân tích (Fact):
{fact}

**Hướng dẫn phân tích:**
1. Xác định bản chất của quan hệ pháp luật và loại tranh chấp (VD: Tranh chấp hợp đồng, Bồi thường thiệt hại, Ly hôn, v.v.).
2. Xác định các điều luật cụ thể áp dụng phù hợp nhất cho vụ án này từ danh sách luật được cung cấp.
3. Đưa ra hướng giải quyết rõ ràng (VD: Ai là người chịu trách nhiệm? Bồi thường như thế nào? Xử lý hợp đồng ra sao?).
4. Phân tích của bạn phải dựa trên các điều luật và tình tiết đã cho. Không được tự bịa ra luật.

YÊU CẦU QUAN TRỌNG: Toàn bộ câu trả lời (bao gồm nội dung phân tích) PHẢI ĐƯỢC VIẾT BẰNG TIẾNG VIỆT.

**Định dạng đầu ra:**
Bạn CHỈ ĐƯỢC phép trả về một đối tượng JSON hợp lệ. Không bao gồm các khối mã markdown (như ```json), chỉ trả về JSON thuần túy.
Đối tượng JSON phải tuân thủ chính xác cấu trúc sau:
{{
    "dispute_type": "Tên loại tranh chấp dân sự",
    "applicable_laws": ["Điều X", "Điều Y"],
    "resolution_direction": "Giải thích chi tiết về hướng giải quyết pháp lý và trách nhiệm (BẰNG TIẾNG VIỆT)"
}}
"""
    # Xử lý sanitize input đơn giản
    safe_fact = fact.replace("{", "{{").replace("}", "}}")
    return prompt_template.format(
        formatted_laws=format_law(laws).replace("{", "{{").replace("}", "}}"),
        formatted_facts=format_fact(similar_facts).replace("{", "{{").replace("}", "}}"),
        fact=safe_fact,
    )


# --- Endpoints from api_server.py ---


@app.post("/generate_prompt", response_model=GeneratePromptResponse)
async def generate_prompt(request: CaseRequest):
    if not rag_system:
        raise HTTPException(
            status_code=500, detail="RAG system is not initialized. Please check startup logs."
        )
    try:
        # DB calls can be slightly blocking, but typically fast. If slow, we'd thread them too.
        retrieved_facts = await run_in_threadpool(
            query_similar_nodes_naive, rag_system.model, request.fact, top_k=request.top_k
        )
        retrieved_laws = await run_in_threadpool(
            query_similar_laws_naive, request.fact, top_k=request.top_k
        )

        retrieved_facts = retrieved_facts or []
        retrieved_laws = retrieved_laws or []

        prompt = build_civil_prompt(request.fact, retrieved_laws, retrieved_facts)
        return GeneratePromptResponse(
            retrieved_laws=retrieved_laws, retrieved_facts=retrieved_facts, prompt=prompt
        )
    except Exception as e:
        logger.error(f"Generate prompt error: {e}")
        raise HTTPException(
            status_code=500, detail="Internal Server Error during context retrieval"
        )


@app.post("/analyze_civil", response_model=AnalyzeCivilResponse)
async def analyze_civil(request: CaseRequest):
    if not rag_system:
        raise HTTPException(
            status_code=500, detail="RAG system is not initialized. Please check startup logs."
        )
    try:
        prompt_response = await generate_prompt(request)

        # Fix Async Blocking: Run LLM generation in threadpool
        raw_response = await run_in_threadpool(
            rag_system.model.generate_response, prompt_response.prompt, 4096
        )

        # Better JSON Parsing with fallback handling
        try:
            cleaned_response = raw_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]

            analysis_result = json.loads(cleaned_response.strip())
        except json.JSONDecodeError:
            # Avoid crashing, but clearly state JSON parsing failed.
            analysis_result = {
                "dispute_type": "Lỗi phân tích JSON",
                "applicable_laws": [],
                "resolution_direction": f"Mô hình không trả về JSON hợp lệ. Nội dung gốc: {raw_response[:200]}...",
            }

        return AnalyzeCivilResponse(
            retrieved_laws=prompt_response.retrieved_laws,
            retrieved_facts=prompt_response.retrieved_facts,
            prompt=prompt_response.prompt,
            analysis_result=analysis_result,
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Analyze civil error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during LLM generation")


# --- Endpoints from app.py ---


@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system is not initialized.")

    case_input = {"fact": request.query, "name": ["Nguyên đơn", "Bị đơn"]}
    try:
        # Fix Async Blocking: Run analysis in threadpool
        results = await run_in_threadpool(rag_system.analyze_case, case_input)

        if not results:
            raise HTTPException(status_code=500, detail="No analysis result returned")

        result = results[0]
        used_laws = result.get("used_laws", [])

        reply = "Dựa trên hệ thống phân tích LegalGraphRAG, đây là tư vấn pháp lý cho trường hợp của bạn:\n\n"
        if used_laws:
            reply += "**Các điều luật áp dụng:**\n"
            for law in used_laws:
                reply += f"- **{law.get('law', 'Điều luật không xác định')}**: {law.get('content', '')}\n"
            reply += "\n**Phân tích sự việc:**\n"
            reply += "Hệ thống xác định vụ việc của bạn thuộc phạm vi điều chỉnh của các điều luật trên. (Lưu ý: Mô hình tạo sinh văn bản tự do hiện đang bị hạn chế, bạn có thể tham khảo trực tiếp các điều luật được trích xuất ở trên).\n"
        else:
            reply += "Hệ thống không tìm thấy điều luật nào phù hợp với trường hợp của bạn. Vui lòng thử mô tả chi tiết hơn hoặc kiểm tra lại.\n"

        return {
            "status": "success",
            "reply": reply,
            "retrieved_laws": result.get("retrieved_laws", []),
            "used_laws": used_laws,
            "similar_cases": result.get("retrieved_facts", []),
        }
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail="Internal error during chat processing")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
