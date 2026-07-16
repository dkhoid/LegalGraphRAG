import json
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import uvicorn
import os

from core.LegalGraphRAG import LegalGraphRAG, LegalGraphRAGConfig
from core.graph_construct.feature_graph import query_similar_laws_naive, query_similar_nodes_naive
from core.utils.logger import logger

app = FastAPI(
    title="LegalGraphRAG Civil API",
    description="API for Vietnamese Civil Law analysis using Graph RAG",
)

# Mount the web directory for static HTML/CSS/JS
os.makedirs("web", exist_ok=True)
app.mount("/web", StaticFiles(directory="web"), name="web")


@app.get("/")
async def root():
    return RedirectResponse(url="/web/index.html")


# Global variables for the RAG system
rag_system = None


class CaseRequest(BaseModel):
    fact: str
    top_k: int = 5


class GeneratePromptResponse(BaseModel):
    retrieved_laws: List[Dict[str, Any]]
    retrieved_facts: List[Dict[str, Any]]
    prompt: str


class AnalyzeCivilResponse(BaseModel):
    retrieved_laws: List[Dict[str, Any]]
    retrieved_facts: List[Dict[str, Any]]
    prompt: str
    analysis_result: Dict[str, Any]


@app.on_event("startup")
async def startup_event():
    global rag_system
    try:
        logger.info("Loading LegalGraphRAG configuration and initializing...")
        config = LegalGraphRAGConfig.from_env_file(".env")
        # Initialize LegalGraphRAG (this will also load the graph DB if it exists)
        rag_system = LegalGraphRAG(config=config)
        logger.info("LegalGraphRAG initialized successfully.")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        # We don't raise here so the server can still start, but endpoints will fail if rag_system is None


def format_law(laws: List[Dict[str, Any]]) -> str:
    res = ""
    for law in laws:
        # Check if 'entry' or 'id' is present
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
    return prompt_template.format(
        formatted_laws=format_law(laws), formatted_facts=format_fact(similar_facts), fact=fact
    )


@app.post("/generate_prompt", response_model=GeneratePromptResponse)
async def generate_prompt(request: CaseRequest):
    if not rag_system:
        raise HTTPException(
            status_code=500, detail="RAG system is not initialized. Please check startup logs."
        )

    try:
        # Retrieve similar facts from the graph
        retrieved_facts = query_similar_nodes_naive(
            rag_system.model, request.fact, top_k=request.top_k
        )

        # Retrieve similar laws from the graph
        retrieved_laws = query_similar_laws_naive(request.fact, top_k=request.top_k)

        # If retrieved lists are None, initialize as empty
        retrieved_facts = retrieved_facts or []
        retrieved_laws = retrieved_laws or []

        # Build prompt
        prompt = build_civil_prompt(request.fact, retrieved_laws, retrieved_facts)

        return GeneratePromptResponse(
            retrieved_laws=retrieved_laws, retrieved_facts=retrieved_facts, prompt=prompt
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze_civil", response_model=AnalyzeCivilResponse)
async def analyze_civil(request: CaseRequest):
    if not rag_system:
        raise HTTPException(
            status_code=500, detail="RAG system is not initialized. Please check startup logs."
        )

    try:
        # Step 1: Generate prompt and retrieve context
        prompt_response = await generate_prompt(request)

        # Step 2: Call the LLM with the prompt
        raw_response = rag_system.model.generate_response(prompt_response.prompt, max_length=4096)

        # Step 3: Parse JSON response
        try:
            # Attempt to clean potential markdown formatting
            cleaned_response = raw_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]

            analysis_result = json.loads(cleaned_response.strip())
        except json.JSONDecodeError:
            # Fallback if LLM failed to return valid JSON
            analysis_result = {
                "dispute_type": "Error parsing LLM output",
                "applicable_laws": [],
                "resolution_direction": f"Raw Output: {raw_response}",
            }

        return AnalyzeCivilResponse(
            retrieved_laws=prompt_response.retrieved_laws,
            retrieved_facts=prompt_response.retrieved_facts,
            prompt=prompt_response.prompt,
            analysis_result=analysis_result,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
