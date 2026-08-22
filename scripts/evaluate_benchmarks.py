import os
import sys
import time
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from core.graph_construct.neo4j_manager import Neo4jManager  # noqa: E402
from core.graph_construct.llm_utils import get_embedding  # noqa: E402
from core.retriever.graph_retriever import GraphRetriever  # noqa: E402
from core.utils.rrf import reciprocal_rank_fusion  # noqa: E402
from core.utils.logger import logger  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# 1. BỘ DỮ LIỆU ĐÁNH GIÁ CHUẨN (GOLDEN LEGAL BENCHMARK DATASET)
# ─────────────────────────────────────────────────────────────────────────────
BENCHMARK_CASES = [
    {
        "id": "eval_01",
        "topic": "Trách nhiệm dân sự hộ gia đình",
        "query": "Hộ gia đình vay tiền ngân hàng để kinh doanh nông nghiệp nhưng các thành viên không có tài sản riêng, trách nhiệm tài sản của các thành viên được xử lý như thế nào theo Điều 101 Bộ luật Dân sự?",
        "ground_truth_laws": [
            "zalo_91/2015/qh13+101",
            "zalo_91/2015/qh13+103",
            "zalo_91/2015/qh13+104",
            "zalo_91/2015/qh13+212",
        ],
    },
    {
        "id": "eval_02",
        "topic": "Đơn phương chấm dứt HĐLĐ",
        "query": "Công ty đột ngột sa thải người lao động có hợp đồng không xác định thời hạn mà không báo trước 45 ngày theo quy định tại Điều 36 Bộ luật Lao động 2019.",
        "ground_truth_laws": ["zalo_45/2019/qh14+36", "zalo_45/2019/qh14+39"],
    },
    {
        "id": "eval_03",
        "topic": "Bồi thường do nguồn nguy hiểm cao độ",
        "query": "Xe ô tô của công ty giao cho tài xế lái gây tai nạn giao thông, trách nhiệm bồi thường thiệt hại do nguồn nguy hiểm cao độ thuộc về ai theo Bộ luật Dân sự?",
        "ground_truth_laws": ["zalo_91/2015/qh13+601", "zalo_91/2015/qh13+597"],
    },
    {
        "id": "eval_04",
        "topic": "Chia tài sản thừa kế không di chúc",
        "query": "Bố mẹ mất không để lại di chúc, các con muốn chia thừa kế quyền sử dụng đất theo pháp luật thì nguyên tắc chia tài sản và hàng thừa kế thứ nhất gồm những ai?",
        "ground_truth_laws": [
            "zalo_91/2015/qh13+651",
            "zalo_91/2015/qh13+654",
            "zalo_91/2015/qh13+626",
        ],
    },
    {
        "id": "eval_05",
        "topic": "Hợp đồng đặt cọc mua bán nhà đất",
        "query": "Bên bán nhận tiền đặt cọc 200 triệu để bán đất nhưng sau đó không chịu ký hợp đồng chuyển nhượng thì phải phạt cọc và bồi thường thế nào theo Điều 328 BLDS?",
        "ground_truth_laws": ["zalo_91/2015/qh13+328"],
    },
    {
        "id": "eval_06",
        "topic": "Thời hiệu khởi kiện tranh chấp hợp đồng",
        "query": "Thời hiệu khởi kiện để yêu cầu Tòa án giải quyết tranh chấp hợp đồng dân sự là bao nhiêu năm kể từ ngày quyền lợi bị xâm phạm?",
        "ground_truth_laws": ["zalo_91/2015/qh13+429", "zalo_91/2015/qh13+154"],
    },
    {
        "id": "eval_07",
        "topic": "Bồi thường thiệt hại do súc vật gây ra",
        "query": "Chó thả rông cắn người đi đường bị thương nặng, chủ sở hữu súc vật có phải bồi thường toàn bộ chi phí cứu chữa và thiệt hại sức khỏe không?",
        "ground_truth_laws": ["zalo_91/2015/qh13+603", "zalo_91/2015/qh13+590"],
    },
    {
        "id": "eval_08",
        "topic": "Xử lý kỷ luật sa thải lao động",
        "query": "Người lao động tự ý bỏ việc 5 ngày cộng dồn trong 30 ngày mà không có lý do chính đáng thì người sử dụng lao động có quyền áp dụng hình thức kỷ luật sa thải không?",
        "ground_truth_laws": ["zalo_45/2019/qh14+125"],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. CÁC PHƯƠNG PHÁP TRUY XUẤT (BASELINES & LEGALGRAPHRAG)
# ─────────────────────────────────────────────────────────────────────────────
class RetrievalEvaluator:
    def __init__(self):
        self.neo4j = Neo4jManager()
        self.graph_retriever = GraphRetriever(model=None)

    def retrieve_naive_vector(self, query: str, top_k: int = 5) -> List[str]:
        """Baseline 1: Naive Dense Vector Search on Law Embeddings"""
        emb = get_embedding(query)
        if not emb:
            return []
        with self.neo4j.driver.session() as s:
            res = s.run(
                """
                CALL db.index.vector.queryNodes('law_embeddings', $top_k, $emb)
                YIELD node AS law, score
                RETURN law.entry AS entry
                ORDER BY score DESC
                """,
                top_k=top_k,
                emb=emb,
            ).data()
            return [r["entry"] for r in res if r.get("entry")]

    def retrieve_bm25_lexical(self, query: str, top_k: int = 5) -> List[str]:
        """Baseline 2: BM25 / Lucene Fulltext Search on Law Text"""
        import re

        clean = re.sub(r"[^\w\s]", "", query)
        words = clean.split()[:10]
        lucene_query = " OR ".join(words)
        with self.neo4j.driver.session() as s:
            try:
                res = s.run(
                    """
                    CALL db.index.fulltext.queryNodes('law_fulltext', $lucene_query, {limit: $top_k})
                    YIELD node AS law, score
                    RETURN law.entry AS entry
                    ORDER BY score DESC
                    """,
                    lucene_query=lucene_query,
                    top_k=top_k,
                ).data()
                return [r["entry"] for r in res if r.get("entry")]
            except Exception:
                return []

    def retrieve_hybrid_norag(self, query: str, top_k: int = 5) -> List[str]:
        """Baseline 3: Hybrid Search (Vector + BM25) fused with RRF, without Graph"""
        vec_entries = self.retrieve_naive_vector(query, top_k=top_k * 2)
        bm25_entries = self.retrieve_bm25_lexical(query, top_k=top_k * 2)

        vec_list = [{"entry": e, "id": e} for e in vec_entries]
        bm25_list = [{"entry": e, "id": e} for e in bm25_entries]

        fused = reciprocal_rank_fusion([vec_list, bm25_list], k=60)
        return [item.get("entry") for item in fused[:top_k] if item.get("entry")]

    def retrieve_legal_graphrag(self, query: str, top_k: int = 5) -> List[str]:
        """Proposed Method: LegalGraphRAG with Graph Traversal, Community Retrieval & MMR"""
        case_dict = {"description": query, "name": "Bị đơn", "features": []}
        config = {
            "use_hyde": False,  # fast benchmark
            "top_retrieve": True,
            "direct_retrieve": True,
            "augment_retrieve": False,
            "use_mmr": True,
            "mmr_top_k": top_k,
            "top_retrieve_top_k": top_k,
            "direct_retrieve_top_k": top_k,
        }
        with open("data/clean/law_to_dispute_clean.json") as f:
            law_to_dispute = json.load(f)
        with open("data/clean/cases_clean.json") as f:
            cases_db = json.load(f)

        _, fused_laws, _ = self.graph_retriever.retrieve(
            case_dict, law_to_dispute, cases_db, config
        )
        return [law_obj.get("entry") for law_obj in fused_laws[:top_k] if law_obj.get("entry")]


# ─────────────────────────────────────────────────────────────────────────────
# 3. CHỈ SỐ ĐÁNH GIÁ (METRICS: HIT RATE, RECALL, MRR, MAP)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_method(method_func, cases: List[Dict[str, Any]], top_k: int = 5) -> Dict[str, float]:
    hit_1_count = 0
    hit_3_count = 0
    hit_5_count = 0
    reciprocal_ranks = []
    recalls = []
    latencies = []

    for case in cases:
        gt = set([g.lower().strip() for g in case["ground_truth_laws"]])
        start = time.time()
        retrieved = method_func(case["query"], top_k=top_k)
        lat = time.time() - start
        latencies.append(lat)

        ret_norm = [r.lower().strip() for r in retrieved]

        # Hit Rate @ 1, 3, 5
        if any(r in gt for r in ret_norm[:1]):
            hit_1_count += 1
        if any(r in gt for r in ret_norm[:3]):
            hit_3_count += 1
        if any(r in gt for r in ret_norm[:5]):
            hit_5_count += 1

        # MRR
        rr = 0.0
        for rank, r in enumerate(ret_norm, 1):
            if r in gt:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        # Recall @ K
        hits = sum(1 for r in ret_norm if r in gt)
        recall = hits / len(gt) if gt else 0.0
        recalls.append(recall)

    n = len(cases)
    return {
        "Hit@1": hit_1_count / n,
        "Hit@3": hit_3_count / n,
        "Hit@5": hit_5_count / n,
        "Recall@5": sum(recalls) / n,
        "MRR": sum(reciprocal_ranks) / n,
        "Avg_Latency_s": sum(latencies) / n,
    }


def main():
    evaluator = RetrievalEvaluator()
    methods = {
        "1. Naive Vector RAG": evaluator.retrieve_naive_vector,
        "2. BM25 Lexical RAG": evaluator.retrieve_bm25_lexical,
        "3. Hybrid RAG (No Graph)": evaluator.retrieve_hybrid_norag,
        "4. LegalGraphRAG (Proposed)": evaluator.retrieve_legal_graphrag,
    }

    logger.info("=" * 80)
    logger.info("🚀 TIẾN HÀNH THỰC NGHIỆM ĐÁNH GIÁ KHOA HỌC SO SÁNH CÁC PHƯƠNG PHÁP RAG PHÁP LÝ")
    logger.info(f"Số lượng câu hỏi kiểm thử: {len(BENCHMARK_CASES)} kịch bản chuẩn")
    logger.info("=" * 80)

    # Warm-up models (avoid cold-start download / lazy loading skewing latencies)
    evaluator.retrieve_legal_graphrag("Tranh chấp hợp đồng dân sự", top_k=2)

    results = {}
    for name, func in methods.items():
        logger.info(f"Đang đánh giá: {name}...")
        res = evaluate_method(func, BENCHMARK_CASES, top_k=5)
        results[name] = res

    # In kết quả dưới dạng bảng Markdown
    print("\n" + "=" * 80)
    print("📊 BẢNG TỔNG HỢP KẾT QUẢ SO SÁNH THỰC NGHIỆM (EVALUATION BENCHMARK TABLE)")
    print("=" * 80)
    header = f"| {'Phương pháp':<28} | {'Hit@1':<7} | {'Hit@3':<7} | {'Hit@5':<7} | {'Recall@5':<8} | {'MRR':<7} | {'Latency':<9} |"
    print(header)
    print(
        "|"
        + "-" * 30
        + "|"
        + "-" * 9
        + "|"
        + "-" * 9
        + "|"
        + "-" * 9
        + "|"
        + "-" * 10
        + "|"
        + "-" * 9
        + "|"
        + "-" * 11
        + "|"
    )
    for name, m in results.items():
        row = f"| {name:<28} | {m['Hit@1']*100:>6.1f}% | {m['Hit@3']*100:>6.1f}% | {m['Hit@5']*100:>6.1f}% | {m['Recall@5']*100:>7.1f}% | {m['MRR']:>7.4f} | {m['Avg_Latency_s']:>7.2f}s |"
        print(row)
    print("=" * 80)

    # Xuất bảng LaTeX chuẩn học thuật
    print("\n📝 MÃ LATEX BẢNG KẾT QUẢ DÀNH CHO BÁO CÁO KHOA HỌC / LUẬN VĂN:")
    latex_code = """
\\begin{table}[htbp]
\\centering
\\caption{So sánh hiệu năng truy xuất giữa LegalGraphRAG và các baseline}
\\label{tab:rag_benchmark}
\\begin{tabular}{lcccccc}
\\hline
\\textbf{Phương pháp} & \\textbf{Hit@1} & \\textbf{Hit@3} & \\textbf{Hit@5} & \\textbf{Recall@5} & \\textbf{MRR} & \\textbf{Độ trễ (s)} \\\\
\\hline
"""
    for name, m in results.items():
        clean_name = name.split(". ")[-1]
        latex_code += f"{clean_name} & {m['Hit@1']*100:.1f}\\% & {m['Hit@3']*100:.1f}\\% & {m['Hit@5']*100:.1f}\\% & {m['Recall@5']*100:.1f}\\% & {m['MRR']:.4f} & {m['Avg_Latency_s']:.2f} \\\\\n"
    latex_code += """\\hline
\\end{tabular}
\\end{table}
"""
    print(latex_code)

    # Lưu kết quả JSON vào data/outputs/
    os.makedirs("data/outputs", exist_ok=True)
    with open("data/outputs/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Đã lưu kết quả chi tiết vào data/outputs/benchmark_results.json")


if __name__ == "__main__":
    main()
