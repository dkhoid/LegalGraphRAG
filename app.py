from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys

# Append current directory to path
sys.path.append(os.path.abspath("."))
from dotenv import load_dotenv

load_dotenv()

from core.LegalGraphRAG import LegalGraphRAG
from core.LegalGraphRAG import LegalGraphRAGConfig
from core.utils.logger import logger

app = FastAPI(title="LegalGraphRAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
config = LegalGraphRAGConfig()
model_name = os.getenv("model_name", "gpt4o_mini")
config.model.model_name = model_name
config.model.api_key = os.getenv(f"{model_name}_api_key", os.getenv("api_key"))
config.model.base_url = os.getenv(f"{model_name}_base_url", os.getenv("base_url"))
config.model.device = "cpu"
config.graph.graph_db_path = os.getenv("graph_db_path", "./data/processed/graph.pkl")
config.graph.auto_build = False

logger.info("Initializing LegalGraphRAG for API Server...")
rag = LegalGraphRAG(config)
rag.config.retrieve.to_dict = lambda: {
    "method": "vector",
    "direct_retrieve_top_k": 3,
}  # Use Vector Hybrid by default as it scored best


class ChatRequest(BaseModel):
    query: str


@app.post("/api/chat")
async def chat(request: ChatRequest):
    case_input = {"fact": request.query, "name": ["Nguyên đơn", "Bị đơn"]}
    try:
        results = rag.analyze_case(case_input)
        if not results:
            return {"status": "error", "message": "No analysis result returned"}

        # Compile response
        result = results[0]
        used_laws = result.get("used_laws", [])

        # Format the response similar to the evaluation prompt response
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
        return {"status": "error", "message": str(e)}


# Serve static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
