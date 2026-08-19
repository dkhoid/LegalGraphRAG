import json
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from api.models import (
    CaseRequest,
    ChatRequest,
    APIKeyRequest,
    GeneratePromptResponse,
    AnalyzeCivilResponse,
)
from api.prompts import build_civil_prompt
from core.graph_construct.graph_search import (
    query_similar_laws_naive,
    query_similar_nodes_naive,
)
from core.utils.logger import logger
from core.constants import PROVIDER_CONFIGS, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE

router = APIRouter()


@router.post("/generate_prompt", response_model=GeneratePromptResponse)
async def generate_prompt(request: CaseRequest, req: Request):
    rag_system = req.app.state.rag_system
    if not rag_system:
        raise HTTPException(
            status_code=500,
            detail="RAG system is not initialized. Please check startup logs.",
        )
    try:
        retrieved_facts = await run_in_threadpool(
            query_similar_nodes_naive,
            rag_system.model,
            request.fact,
            top_k=request.top_k,
        )
        retrieved_laws = await run_in_threadpool(
            query_similar_laws_naive, request.fact, top_k=request.top_k
        )

        retrieved_facts = retrieved_facts or []
        retrieved_laws = retrieved_laws or []

        prompt = build_civil_prompt(request.fact, retrieved_laws, retrieved_facts)
        return GeneratePromptResponse(
            retrieved_laws=retrieved_laws,
            retrieved_facts=retrieved_facts,
            prompt=prompt,
        )
    except Exception as e:
        logger.error(f"Generate prompt error: {e}")
        raise HTTPException(
            status_code=500, detail="Internal Server Error during context retrieval"
        )


@router.post("/analyze_civil", response_model=AnalyzeCivilResponse)
async def analyze_civil(request: CaseRequest, req: Request):
    rag_system = req.app.state.rag_system
    if not rag_system:
        raise HTTPException(
            status_code=500,
            detail="RAG system is not initialized. Please check startup logs.",
        )

    from core.utils.context import request_api_key, request_base_url, request_model_name

    request_api_key.set(request.api_key)

    if request.provider and request.provider in PROVIDER_CONFIGS:
        provider_conf = PROVIDER_CONFIGS[request.provider]
        request_base_url.set(provider_conf["base_url"])
        request_model_name.set(provider_conf["model_name"])

    try:
        # Re-use generate_prompt logic, but call the function directly
        # Wait, generate_prompt uses req.app.state, so we can just extract logic or pass req
        # Let's call the function
        prompt_response = await generate_prompt(request, req)

        raw_response = await run_in_threadpool(
            rag_system.model.generate_response,
            prompt_response.prompt,
            DEFAULT_MAX_TOKENS,
            DEFAULT_TEMPERATURE,
            request.api_key,
        )

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


@router.post("/api/chat")
async def chat(request: ChatRequest, req: Request):
    rag_system = req.app.state.rag_system
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system is not initialized.")

    from core.utils.context import request_api_key, request_base_url, request_model_name

    request_api_key.set(request.api_key)
    if request.provider and request.provider in PROVIDER_CONFIGS:
        provider_conf = PROVIDER_CONFIGS[request.provider]
        request_base_url.set(provider_conf["base_url"])
        request_model_name.set(provider_conf["model_name"])

    case_input = {"fact": request.query, "name": ["Nguyên đơn", "Bị đơn"]}
    try:
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


@router.post("/api/set_key")
async def set_api_key(request: APIKeyRequest, req: Request):
    rag_system = req.app.state.rag_system
    if not rag_system or not rag_system.model:
        raise HTTPException(status_code=500, detail="RAG system is not initialized.")
    try:
        if hasattr(rag_system.model, "update_api_key"):
            rag_system.model.update_api_key(request.api_key)
            return {
                "status": "success",
                "message": "Đã cập nhật API Key thành công trên server.",
            }
        else:
            raise HTTPException(
                status_code=400,
                detail="Mô hình hiện tại không hỗ trợ cập nhật API key lúc runtime.",
            )
    except Exception as e:
        logger.error(f"Error updating API key: {e}")
        raise HTTPException(status_code=500, detail="Không thể cập nhật API Key")
