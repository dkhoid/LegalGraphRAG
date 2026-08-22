import json
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from api.models import (
    CaseRequest,
    ChatRequest,
    APIKeyRequest,
    GeneratePromptResponse,
    AnalyzeCivilResponse,
    GraphInspectRequest,
    GraphInspectResponse,
)
from api.prompts import build_civil_prompt
from core.graph_construct.graph_search import (
    query_similar_laws_naive,
    query_similar_nodes_naive,
)
from core.utils.logger import logger
from core.constants import PROVIDER_CONFIGS

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
    else:
        request_base_url.set(None)
        request_model_name.set(None)

    try:
        case_input = {"fact": request.fact, "name": ["Tổng quan vụ việc"]}
        results_list = await run_in_threadpool(rag_system.analyze_case, case_input)

        parsed_results = []
        for res in results_list:
            judge_res = res.get("judge_result", {})
            if isinstance(judge_res, str):
                try:
                    judge_res = json.loads(judge_res)
                except json.JSONDecodeError:
                    judge_res = {
                        "dispute_type": "Không xác định",
                        "applicable_laws": [],
                        "resolution_direction": judge_res,
                    }

            parsed_results.append(
                {
                    "name": res.get("name", "Unknown"),
                    "description": res.get("description", ""),
                    "judge_result": judge_res,
                    "used_laws": res.get("used_laws", []),
                    "used_facts": res.get("used_facts", []),
                    "confidence": res.get("confidence", {}),
                    "reasoning_trace": res.get("reasoning_trace", {}),
                }
            )

        return AnalyzeCivilResponse(results=parsed_results)
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
    else:
        request_base_url.set(None)
        request_model_name.set(None)

    case_input = {"fact": request.query, "name": ["Tổng quan vụ việc"]}
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
                law_title = law.get("entry") or law.get("id") or "Điều luật không xác định"
                law_content = law.get("description") or law.get("text") or ""
                reply += f"- **{law_title}**: {law_content}\n"
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


@router.post("/api/graph/inspect", response_model=GraphInspectResponse)
async def inspect_graph_subgraph(request: GraphInspectRequest, req: Request):
    """Trực quan hóa đồ thị con (Subgraph Visualization) cho một câu hỏi hoặc vụ án."""
    from api.models import GraphInspectResponse
    from core.graph_construct.neo4j_manager import neo4j_manager
    from core.graph_construct.llm_utils import get_embedding

    if not neo4j_manager.driver:
        raise HTTPException(status_code=500, detail="Neo4j driver not connected")

    try:
        emb = await run_in_threadpool(get_embedding, request.query)
        if not emb:
            raise HTTPException(status_code=400, detail="Could not generate embedding for query")

        nodes = []
        edges = []
        seen_node_ids = set()

        # 1. Query Node (Trung tâm)
        query_node_id = "query_center"
        nodes.append(
            {
                "id": query_node_id,
                "label": "Câu hỏi / Vụ án đang xét",
                "type": "Query",
                "color": "#e74c3c",
                "title": request.query[:200],
                "value": 25,
            }
        )
        seen_node_ids.add(query_node_id)

        # 2. Truy vấn các Cases tương đồng & Quan hệ từ Neo4j
        cypher = """
        CALL db.index.vector.queryNodes('case_embeddings', $top_k, $emb)
        YIELD node AS c, score
        OPTIONAL MATCH (c)-[r1:BELONGS_TO_CLUSTER]->(cluster:Cluster)
        OPTIONAL MATCH (c)-[r2:RELATES_TO_LAW]->(law:Laws)
        RETURN c, score, cluster, collect(DISTINCT law) AS laws
        ORDER BY score DESC
        """
        with neo4j_manager.driver.session() as s:
            results = s.run(cypher, top_k=request.top_k, emb=emb).data()

        case_count = 0
        law_count = 0
        cluster_count = 0

        for record in results:
            c = record["c"]
            score = record["score"]
            c_id = f"case_{c['id']}"
            if c_id not in seen_node_ids:
                seen_node_ids.add(c_id)
                case_count += 1
                nodes.append(
                    {
                        "id": c_id,
                        "label": f"Vụ án: {c.get('caseId', 'Án lệ')}",
                        "type": "Case",
                        "color": "#3498db",
                        "title": f"Tương đồng: {score:.3f}\nTình tiết: {c.get('description', '')[:200]}...",
                        "value": 15,
                    }
                )
                # Cạnh nối Query -> Case
                edges.append(
                    {
                        "from": query_node_id,
                        "to": c_id,
                        "label": f"Sim {score:.2f}",
                        "color": "#3498db",
                    }
                )

            # Cluster
            cluster = record.get("cluster")
            if cluster:
                cl_id = f"cluster_{cluster['id']}"
                if cl_id not in seen_node_ids:
                    seen_node_ids.add(cl_id)
                    cluster_count += 1
                    nodes.append(
                        {
                            "id": cl_id,
                            "label": f"Cụm: {cluster.get('summary', 'Cộng đồng pháp lý')[:30]}...",
                            "type": "Cluster",
                            "color": "#9b59b6",
                            "title": cluster.get("summary", ""),
                            "value": 20,
                        }
                    )
                edges.append(
                    {
                        "from": c_id,
                        "to": cl_id,
                        "label": "BELONGS_TO_CLUSTER",
                        "color": "#9b59b6",
                    }
                )

            # Laws
            for law in record.get("laws", []):
                if law and law.get("id"):
                    l_id = f"law_{law['id']}"
                    if l_id not in seen_node_ids:
                        seen_node_ids.add(l_id)
                        law_count += 1
                        nodes.append(
                            {
                                "id": l_id,
                                "label": f"Điều luật: {law.get('entry', 'Điều luật')}",
                                "type": "Law",
                                "color": "#2ecc71",
                                "title": law.get("description", "")[:200] + "...",
                                "value": 12,
                            }
                        )
                    edges.append(
                        {
                            "from": c_id,
                            "to": l_id,
                            "label": "RELATES_TO_LAW",
                            "color": "#2ecc71",
                        }
                    )

        stats = {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "cases_count": case_count,
            "laws_count": law_count,
            "clusters_count": cluster_count,
        }
        return GraphInspectResponse(query=request.query, nodes=nodes, edges=edges, stats=stats)
    except Exception as e:
        logger.error(f"Error inspecting graph: {e}")
        raise HTTPException(status_code=500, detail=f"Graph inspection failed: {e}")


@router.get("/api/scenarios")
async def get_preset_scenarios():
    """Lấy danh sách các tình huống pháp lý mẫu có sẵn để chọn nhanh trên giao diện."""
    from api.sample_data import PRESET_SCENARIOS

    return {"scenarios": PRESET_SCENARIOS}


@router.get("/api/graph/sample", response_model=GraphInspectResponse)
async def get_sample_graph(scenario: str = "overview"):
    """Lấy đồ thị tri thức mẫu (Knowledge Graph) theo kịch bản để xem trực quan ngay lập tức."""
    from api.sample_data import SAMPLE_GRAPHS

    sample = SAMPLE_GRAPHS.get(scenario, SAMPLE_GRAPHS.get("overview"))

    nodes = sample["nodes"]
    edges = sample["edges"]
    stats = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "cases_count": len([n for n in nodes if n.get("type") == "Case"]),
        "laws_count": len([n for n in nodes if n.get("type") == "Law"]),
        "clusters_count": len([n for n in nodes if n.get("type") == "Cluster"]),
    }
    return GraphInspectResponse(query=sample["query"], nodes=nodes, edges=edges, stats=stats)


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
