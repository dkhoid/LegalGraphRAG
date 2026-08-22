"""Script thực nghiệm đánh giá khoa học chất lượng Tạo sinh & Phán quyết Pháp lý (Generation Evaluation Benchmark).

So sánh 4 phương pháp:
1. Zero-Shot LLM (Không dùng RAG)
2. Naive Vector RAG + Generation
3. Hybrid RAG (No Graph) + Generation
4. LegalGraphRAG (Proposed) + Generation

Tiêu chuẩn đánh giá (RAG Triad & Legal Grounding via LLM-as-a-Judge):
- Faithfulness (Độ trung thực): 1.0 - 5.0
- Answer Relevance (Độ phù hợp câu hỏi): 1.0 - 5.0
- Citation Precision (%): Tỷ lệ trích dẫn đúng điều luật
- Citation Recall (%): Tỷ lệ bao phủ điều luật cốt lõi
- Hallucination Rate (%): Tỷ lệ ảo giác pháp lý
"""

import os
import sys
import time
import json
import re
from typing import List, Dict, Any
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from core.graph_construct.neo4j_manager import Neo4jManager  # noqa: E402
from core.graph_construct.llm_utils import get_embedding  # noqa: E402
from core.models.openai.deepseek_v3 import DeepSeekChatbot  # noqa: E402
from core.retriever.graph_retriever import GraphRetriever  # noqa: E402
from core.judge.judge_civil import judge_civil_all, normalize_law_article  # noqa: E402
from core.utils.rrf import reciprocal_rank_fusion  # noqa: E402
from core.utils.logger import logger  # noqa: E402
from scripts.evaluate_benchmarks import BENCHMARK_CASES  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# 1. CÁC PIPELINE TẠO SINH LỜI GIẢI (GENERATION PIPELINES)
# ─────────────────────────────────────────────────────────────────────────────
class GenerationBenchmarkEvaluator:
    def __init__(self):
        self.neo4j = Neo4jManager()
        self.generator_llm = DeepSeekChatbot(model_name="deepseek-chat")
        self.judge_evaluator_llm = DeepSeekChatbot(model_name="deepseek-chat")
        self.graph_retriever = GraphRetriever(model=self.generator_llm)

        with open("data/clean/law_to_dispute_clean.json", "r", encoding="utf-8") as f:
            self.law_to_dispute = json.load(f)
        with open("data/clean/cases_clean.json", "r", encoding="utf-8") as f:
            self.cases_db = json.load(f)

    # 1. Zero-Shot LLM (Không dùng RAG)
    def generate_zero_shot(self, query: str) -> Dict[str, Any]:
        prompt = f"""Bạn là một chuyên gia phân tích pháp lý. Dựa trên kiến thức của bạn, hãy giải quyết vụ việc sau:
Vụ án / Câu hỏi: {query}

Trả về DUY NHẤT một JSON object theo cấu trúc:
{{
    "dispute_type": ["loại tranh chấp"],
    "law_article": ["Điều luật 1", "Điều luật 2"],
    "resolution": {{
        "liability": "Xác định trách nhiệm pháp lý của các bên",
        "compensation": "Hướng giải quyết bồi thường hoặc tài sản cụ thể"
    }}
}}
"""
        raw_res = self.generator_llm.generate_response(prompt, max_length=2048)
        return self._parse_json_result(raw_res)

    # 2. Naive Vector RAG + Generation
    def generate_naive_vector_rag(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        emb = get_embedding(query)
        retrieved_laws = []
        if emb:
            with self.neo4j.driver.session() as s:
                records = s.run(
                    """
                    CALL db.index.vector.queryNodes('law_embeddings', $top_k, $emb)
                    YIELD node AS law, score
                    RETURN law.id AS id, law.entry AS entry, law.description AS description
                    ORDER BY score DESC
                    """,
                    top_k=top_k,
                    emb=emb,
                ).data()
                for r in records:
                    retrieved_laws.append(
                        {
                            "id": r["id"],
                            "entry": r.get("entry", r["id"]),
                            "description": r.get("description", ""),
                        }
                    )
        return judge_civil_all(self.generator_llm, retrieved_laws, [], query)

    # 3. Hybrid RAG (No Graph) + Generation
    def generate_hybrid_rag(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        emb = get_embedding(query)
        vec_laws = []
        if emb:
            with self.neo4j.driver.session() as s:
                v_res = s.run(
                    "CALL db.index.vector.queryNodes('law_embeddings', $top_k, $emb) YIELD node AS law RETURN law.id AS id, law.entry AS entry, law.description AS description",
                    top_k=top_k * 2,
                    emb=emb,
                ).data()
                vec_laws = [
                    {
                        "id": r["id"],
                        "entry": r.get("entry", r["id"]),
                        "description": r.get("description", ""),
                    }
                    for r in v_res
                ]

        clean_words = re.sub(r"[^\w\s]", "", query).split()[:10]
        lucene_query = " OR ".join(clean_words) if clean_words else query
        bm25_laws = []
        with self.neo4j.driver.session() as s:
            try:
                b_res = s.run(
                    "CALL db.index.fulltext.queryNodes('law_fulltext', $lucene_query, {limit: $top_k}) YIELD node AS law RETURN law.id AS id, law.entry AS entry, law.description AS description",
                    lucene_query=lucene_query,
                    top_k=top_k * 2,
                ).data()
                bm25_laws = [
                    {
                        "id": r["id"],
                        "entry": r.get("entry", r["id"]),
                        "description": r.get("description", ""),
                    }
                    for r in b_res
                ]
            except Exception:
                pass

        fused = reciprocal_rank_fusion([vec_laws, bm25_laws], k=60)
        return judge_civil_all(self.generator_llm, fused[:top_k], [], query)

    # 4. LegalGraphRAG (Proposed: Full Graph Traversal + Weighted RRF + Reranker)
    def generate_legal_graphrag(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        case_dict = {"description": query, "name": "Bị đơn", "features": []}
        config = {
            "top_retrieve": True,
            "direct_retrieve": True,
            "augment_retrieve": False,
            "use_reranker": True,
            "rerank_top_k": top_k,
            "top_retrieve_top_k": top_k,
            "direct_retrieve_top_k": top_k,
            "rrf_weights": [1.3, 1.0, 0.5, 1.0],
        }
        _, fused_laws, retrieved_facts = self.graph_retriever.retrieve(
            case_dict, self.law_to_dispute, self.cases_db, config
        )
        return judge_civil_all(self.generator_llm, fused_laws[:top_k], retrieved_facts, query)

    def _parse_json_result(self, text: str) -> Dict[str, Any]:
        try:
            cleaned = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
            first = cleaned.find("{")
            last = cleaned.rfind("}") + 1
            if first != -1 and last > first:
                data = json.loads(cleaned[first:last])
                raw_laws = data.get("law_article", [])
                if isinstance(raw_laws, list):
                    data["law_article"] = [normalize_law_article(x) for x in raw_laws]
                return data
        except Exception:
            pass
        return {
            "dispute_type": ["Dân sự"],
            "law_article": [],
            "resolution": {"liability": text[:300], "compensation": "Theo quy định pháp luật"},
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. LLM-AS-A-JUDGE EVALUATION ENGINE (RAG TRIAD & LEGAL GROUNDING)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_generation_quality(
    evaluator_llm: DeepSeekChatbot,
    query: str,
    ground_truth_laws: List[str],
    generation_result: Dict[str, Any],
) -> Dict[str, float]:
    """Sử dụng LLM-as-a-Judge để chấm điểm Faithfulness, Answer Relevance và Citation Accuracy."""
    law_articles = generation_result.get("law_article", [])
    if isinstance(law_articles, str):
        law_articles = [law_articles]
    resolution = generation_result.get("resolution", {})
    resolution_text = f"Trách nhiệm: {resolution.get('liability', '')}. Hướng xử lý: {resolution.get('compensation', '')}"

    # 1. Tính Citation Precision & Recall dựa trên số hiệu Điều luật
    gt_numbers = set()
    for gt in ground_truth_laws:
        # e.g., 'zalo_91/2015/qh13+101' -> '101'
        m = re.findall(r"\+(\d+)", gt) or re.findall(r"\b(\d+)\b", gt)
        if m:
            gt_numbers.add(m[-1])

    cited_numbers = set()
    for art in law_articles:
        m = re.findall(r"\b(\d+)\b", str(art))
        if m and int(m[0]) < 2000:
            cited_numbers.add(m[0])

    if cited_numbers:
        correct_citations = cited_numbers.intersection(gt_numbers)
        citation_precision = len(correct_citations) / len(cited_numbers)
    else:
        citation_precision = 0.0

    if gt_numbers:
        citation_recall = len(cited_numbers.intersection(gt_numbers)) / len(gt_numbers)
    else:
        citation_recall = 0.0

    # 2. Prompt LLM-as-a-Judge cho Faithfulness & Answer Relevance
    eval_prompt = f"""Bạn là một Giám khảo Thẩm định Pháp lý Khoa học. Nhiệm vụ của bạn là đánh giá câu trả lời tư vấn/phán quyết do AI sinh ra dựa trên 2 tiêu chí khắt khe:

Câu hỏi / Tình tiết vụ việc:
\"\"\"{query}\"\"\"

Các điều luật chuẩn cần áp dụng:
\"\"\"{', '.join(ground_truth_laws)}\"\"\"

Phán quyết / Lời giải do hệ thống AI sinh ra:
- Loại tranh chấp: {generation_result.get('dispute_type', [])}
- Các điều luật viện dẫn: {law_articles}
- Nội dung giải quyết: {resolution_text}

HƯỚNG DẪN CHẤM ĐIỂM (Thang điểm 1.0 đến 5.0):
1. **Faithfulness (Độ trung thực & Không ảo giác)**:
   - Điểm 5.0: Toàn bộ lập luận, xác định nghĩa vụ bồi thường và căn cứ viện dẫn đều chuẩn xác, bám sát các điều luật chuẩn, không bịa đặt.
   - Điểm 3.0: Lập luận có lý nhưng trích dẫn điều luật chưa chuẩn xác hoặc thiếu căn cứ cụ thể.
   - Điểm 1.0: Ảo giác nghiêm trọng, viện dẫn sai hoàn toàn quy định pháp luật hoặc đưa ra phán quyết trái luật.
2. **Answer Relevance (Độ phù hợp & Giải quyết tranh chấp)**:
   - Điểm 5.0: Trả lời trực diện câu hỏi, chỉ rõ ai có trách nhiệm và hướng xử lý bồi thường rõ ràng, thỏa đáng.
   - Điểm 3.0: Trả lời chung chung, né tránh kết luận cụ thể.
   - Điểm 1.0: Lạc đề, không giải quyết được câu hỏi pháp lý.

Trả về DUY NHẤT một JSON object theo mẫu sau, không thêm bất kỳ văn bản nào khác:
{{"faithfulness": 4.5, "answer_relevance": 4.8, "hallucination_detected": false}}
"""
    try:
        raw_eval = evaluator_llm.generate_response(eval_prompt, max_length=512)
        cleaned = re.sub(r"(?is)<think>.*?</think>", "", raw_eval).strip()
        f_idx = cleaned.find("{")
        l_idx = cleaned.rfind("}") + 1
        if f_idx != -1 and l_idx > f_idx:
            eval_data = json.loads(cleaned[f_idx:l_idx])
            faithfulness = float(eval_data.get("faithfulness", 3.0))
            relevance = float(eval_data.get("answer_relevance", 3.0))
            hallucination_detected = bool(eval_data.get("hallucination_detected", False))
        else:
            faithfulness = 3.5
            relevance = 3.5
            hallucination_detected = False
    except Exception as e:
        logger.warning(f"LLM-as-a-judge scoring failed: {e}")
        faithfulness = 3.5
        relevance = 3.5
        hallucination_detected = False

    hallucination_rate = (
        1.0
        if (hallucination_detected or (citation_precision == 0.0 and len(cited_numbers) > 0))
        else 0.0
    )

    return {
        "faithfulness": round(faithfulness, 2),
        "answer_relevance": round(relevance, 2),
        "citation_precision": round(citation_precision, 4),
        "citation_recall": round(citation_recall, 4),
        "hallucination_rate": round(hallucination_rate, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. CHƯƠNG TRÌNH THỰC THI CHÍNH (MAIN EXECUTION)
# ─────────────────────────────────────────────────────────────────────────────
def main():
    evaluator = GenerationBenchmarkEvaluator()
    methods = {
        "1. Zero-Shot LLM (No RAG)": evaluator.generate_zero_shot,
        "2. Naive Vector RAG": evaluator.generate_naive_vector_rag,
        "3. Hybrid RAG (No Graph)": evaluator.generate_hybrid_rag,
        "4. LegalGraphRAG (Proposed)": evaluator.generate_legal_graphrag,
    }

    print("=" * 85)
    print("🚀 BẮT ĐẦU ĐÁNH GIÁ THỰC NGHIỆM CHẤT LƯỢNG TẠO SINH & PHÁN QUYẾT PHÁP LÝ (CRITICAL 1)")
    print(f"Số lượng kịch bản kiểm thử: {len(BENCHMARK_CASES)} vụ việc thực tế")
    print("Mô hình LLM sinh lời giải & LLM Judge: DeepSeek-V3 (deepseek-chat)")
    print("=" * 85)

    benchmark_summary = {}

    for method_name, generate_func in methods.items():
        logger.info(f"Đang chạy đánh giá: {method_name}...")
        faithfulness_list = []
        relevance_list = []
        prec_list = []
        rec_list = []
        hal_list = []
        latencies = []

        for case in BENCHMARK_CASES:
            t0 = time.time()
            res = generate_func(case["query"])
            lat = time.time() - t0
            latencies.append(lat)

            scores = evaluate_generation_quality(
                evaluator.judge_evaluator_llm,
                case["query"],
                case["ground_truth_laws"],
                res,
            )
            faithfulness_list.append(scores["faithfulness"])
            relevance_list.append(scores["answer_relevance"])
            prec_list.append(scores["citation_precision"])
            rec_list.append(scores["citation_recall"])
            hal_list.append(scores["hallucination_rate"])

        n = len(BENCHMARK_CASES)
        benchmark_summary[method_name] = {
            "Faithfulness": round(sum(faithfulness_list) / n, 2),
            "Answer_Relevance": round(sum(relevance_list) / n, 2),
            "Citation_Precision": round(sum(prec_list) / n * 100, 1),
            "Citation_Recall": round(sum(rec_list) / n * 100, 1),
            "Hallucination_Rate": round(sum(hal_list) / n * 100, 1),
            "Avg_Latency_s": round(sum(latencies) / n, 2),
        }

    # In kết quả dưới dạng Markdown Table
    print("\n" + "=" * 90)
    print("📊 BẢNG TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ CHẤT LƯỢNG TẠO SINH (GENERATION BENCHMARK TABLE)")
    print("=" * 90)
    print(
        f"| {'Phương pháp':<28} | {'Faithful (1-5)':<14} | {'Relevance (1-5)':<15} | {'Cit. Prec':<10} | {'Cit. Rec':<10} | {'Hallucination':<13} | {'Latency':<8} |"
    )
    print(
        "|"
        + "-" * 30
        + "|"
        + "-" * 16
        + "|"
        + "-" * 17
        + "|"
        + "-" * 12
        + "|"
        + "-" * 12
        + "|"
        + "-" * 15
        + "|"
        + "-" * 10
        + "|"
    )

    for name, m in benchmark_summary.items():
        print(
            f"| {name:<28} | "
            f"{m['Faithfulness']:>10.2f}/5.0 | "
            f"{m['Answer_Relevance']:>11.2f}/5.0 | "
            f"{m['Citation_Precision']:>9.1f}% | "
            f"{m['Citation_Recall']:>9.1f}% | "
            f"{m['Hallucination_Rate']:>12.1f}% | "
            f"{m['Avg_Latency_s']:>6.2f}s |"
        )
    print("=" * 90)

    # In mã LaTeX
    print("\n📝 MÃ LATEX BẢNG KẾT QUẢ DÀNH CHO BÁO CÁO KHOA HỌC / LUẬN VĂN:\n")
    latex_code = r"""\begin{table}[htbp]
\centering
\caption{Đánh giá chất lượng tạo sinh và tính chuẩn xác pháp lý giữa các phương pháp}
\label{tab:generation_benchmark}
\begin{tabular}{lcccccc}
\hline
\textbf{Phương pháp} & \textbf{Faithfulness} & \textbf{Relevance} & \textbf{Cit. Prec} & \textbf{Cit. Rec} & \textbf{Hallucination} & \textbf{Latency (s)} \\
\hline"""
    for name, m in benchmark_summary.items():
        short_name = name.split(". ")[-1]
        latex_code += f"\n{short_name} & {m['Faithfulness']:.2f} & {m['Answer_Relevance']:.2f} & {m['Citation_Precision']:.1f}\\% & {m['Citation_Recall']:.1f}\\% & {m['Hallucination_Rate']:.1f}\\% & {m['Avg_Latency_s']:.2f} \\\\"
    latex_code += r"""
\hline
\end{tabular}
\end{table}"""
    print(latex_code)

    # Lưu kết quả vào file JSON
    output_path = "data/outputs/generation_benchmark_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Đã lưu kết quả chi tiết vào {output_path}")


if __name__ == "__main__":
    main()
