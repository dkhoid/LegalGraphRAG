import sys
import os

sys.path.append(".")
from core.LegalGraphRAG import LegalGraphRAG, LegalGraphRAGConfig
from core.graph_construct.graph_db import GraphDBManager

# 1. Load the graph
GraphDBManager.load("data/clean/graph.pkl")

# 2. Init RAG system
config = LegalGraphRAGConfig.from_env_file(".env")
rag = LegalGraphRAG(config=config)
# Cố tình load case db vào bộ nhớ vì test script không gọi build_graph()
rag.cases_db = rag._load_cases_db()
rag.law_to_dispute = rag._load_law_to_dispute()

# 3. Create a mock case
sample_case = {
    "id": "test_1",
    "name": "Nguyễn Văn A",
    "fact": "Tôi kết hôn năm 2018. Mới đây tôi phát hiện vợ tôi ngoại tình và đã tự ý mang sổ đỏ là tài sản chung của hai vợ chồng đi thế chấp ngân hàng để vay 500 triệu đồng. Tôi muốn ly hôn và hỏi giao dịch thế chấp này có hiệu lực hay không?",
    "description": "Tôi kết hôn năm 2018. Mới đây tôi phát hiện vợ tôi ngoại tình và đã tự ý mang sổ đỏ là tài sản chung của hai vợ chồng đi thế chấp ngân hàng để vay 500 triệu đồng. Tôi muốn ly hôn và hỏi giao dịch thế chấp này có hiệu lực hay không?",
    "feature": {
        "parties_info": ["Vợ chồng Nguyễn Văn A"],
        "dispute_acts": ["Ngoại tình", "Tự ý thế chấp sổ đỏ tài sản chung"],
        "subject_matter": ["Ly hôn", "Hợp đồng thế chấp"],
        "fault_and_evidence": [],
    },
}

print("\n--- BẮT ĐẦU TRUY VẤN RAG ---")
try:
    result = rag.analyze_case(sample_case)

    print("\n--- KẾT QUẢ TỪ LLM ---")
    for r in result:
        print(f"Bên bị ảnh hưởng: {r['name']}")
        print(f"\n[PHÂN TÍCH LUẬT VÀ TƯ VẤN]\n{r['judge_result']}")
        print(f"\n[LUẬT ĐƯỢC ÁP DỤNG TRÍCH XUẤT TỪ GRAPH]")
        for law in r.get("used_laws", []):
            print(f" - ID: {law.get('entry')} | {law.get('description', '')[:200]}...")
except Exception as e:
    import traceback

    traceback.print_exc()
