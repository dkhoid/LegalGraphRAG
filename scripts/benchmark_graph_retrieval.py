import os
import sys
import time
from dotenv import load_dotenv

# Thêm root vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from core.LegalGraphRAG import LegalGraphRAG, LegalGraphRAGConfig  # noqa: E402
from core.retriever.graph_retriever import GraphRetriever  # noqa: E402


def run_benchmark():
    print("=" * 80)
    print("🚀 BẮT ĐẦU BENCHMARK TRUY XUẤT ĐỒ THỊ (GRAPH RETRIEVAL BENCHMARK)")
    print("=" * 80)

    config = LegalGraphRAGConfig.from_env_file(".env")
    rag = LegalGraphRAG(config=config)
    retriever = GraphRetriever(model=rag.model)
    retrieve_config = config.retrieve.to_dict()

    test_queries = [
        {
            "topic": "Trách nhiệm dân sự hộ gia đình & tài sản (Điều 101 BLDS)",
            "query": "Hộ gia đình tôi vay tiền ngân hàng để sản xuất nông nghiệp, các thành viên không có tài sản riêng thì trách nhiệm tài sản của hộ gia đình được xử lý như thế nào theo Điều 101?",
            "target_keywords": ["101", "hộ gia đình", "tài sản chung", "nghĩa vụ"],
        },
        {
            "topic": "Bồi thường thiệt hại do nguồn nguy hiểm cao độ",
            "query": "Xe ô tô của công ty gây tai nạn giao thông trên đường, người lái xe có lỗi nhưng xe là tài sản của công ty. Trách nhiệm bồi thường thiệt hại ngoài hợp đồng được xác định như thế nào?",
            "target_keywords": ["584", "601", "nguồn nguy hiểm", "bồi thường thiệt hại"],
        },
        {
            "topic": "Đơn phương chấm dứt hợp đồng lao động",
            "query": "Công ty đột ngột sa thải người lao động không báo trước 45 ngày với hợp đồng không xác định thời hạn. Quyền lợi và bồi thường theo Bộ luật Lao động?",
            "target_keywords": ["lao động", "sa thải", "chấm dứt hợp đồng", "bồi thường"],
        },
        {
            "topic": "Tranh chấp thừa kế quyền sử dụng đất",
            "query": "Bố mẹ mất không để lại di chúc, các con muốn phân chia di sản thừa kế là quyền sử dụng đất và nhà ở thì thời hiệu và nguyên tắc chia thừa kế ra sao?",
            "target_keywords": ["thừa kế", "di chúc", "di sản", "quyền sử dụng đất"],
        },
    ]

    results = []

    for i, t in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] 🔎 Đang kiểm thử: {t['topic']}")
        print(f"Câu hỏi: \"{t['query']}\"")

        start_time = time.time()
        # 1. Truy xuất qua GraphRetriever (kết hợp Neo4j Vector + Graph Traversal + Fulltext + RRF + MMR)
        case_dict = {"description": t["query"], "name": "Bị đơn", "features": []}
        orig_res, fused_laws, retrieved_cases = retriever.retrieve(
            case_dict, rag.law_to_dispute, rag.cases_db, retrieve_config
        )
        latency = time.time() - start_time

        top_clusters = orig_res.get("top", {}).get("clusters", [])

        print(f"   ⏱️ Độ trễ: {latency:.2f}s")
        print(f"   🏛️ Số cụm tìm được: {len(top_clusters)}")
        print(f"   📋 Số vụ án tương tự: {len(retrieved_cases)}")
        print(f"   📜 Số điều luật tìm thấy: {len(fused_laws)}")

        top_laws = []
        for idx, law in enumerate(fused_laws[:5], 1):
            entry = law.get("entry") or law.get("id")
            desc = (law.get("description") or law.get("text") or "")[:120].replace("\n", " ")
            rrf_score = law.get("_rrf_score", 0.0)
            top_laws.append(f"      {idx}. [Điều {entry}] (Score: {rrf_score:.4f}): {desc}...")

        for l_str in top_laws:
            print(l_str)

        results.append(
            {
                "topic": t["topic"],
                "latency": latency,
                "laws_count": len(fused_laws),
                "clusters_count": len(top_clusters),
                "top_entry": fused_laws[0].get("entry") if fused_laws else "None",
            }
        )

    print("\n" + "=" * 80)
    print("📊 BẢNG TỔNG KẾT HIỆU NĂNG TRUY XUẤT ĐỒ THỊ")
    print("=" * 80)
    print(f"{'Chủ đề':<50} | {'Độ trễ':<8} | {'Số luật':<8} | {'Điều luật Top 1'}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['topic']:<50} | {r['latency']:<7.2f}s | {r['laws_count']:<8} | Điều {r['top_entry']}"
        )
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
