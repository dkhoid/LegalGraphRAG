import os
import sys
import argparse
import time
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv()

from core.LegalGraphRAG import LegalGraphRAG, LegalGraphRAGConfig
from core.utils.logger import logger


def main():
    parser = argparse.ArgumentParser(description="Compare Vector vs Graph Retrieval")
    parser.add_argument("--query", type=str, required=True, help="Tình huống pháp lý (Fact)")
    parser.add_argument("--name", type=str, default="Nguyên đơn", help="Tên các bên liên quan")
    args = parser.parse_args()

    # Khởi tạo Config từ biến môi trường (nếu có)
    config = LegalGraphRAGConfig()

    # Ưu tiên dùng API để tránh lỗi không có GPU (CUDA)
    model_name = os.getenv("model_name", "gpt4o_mini")
    config.model.model_name = model_name
    config.model.api_key = os.getenv(f"{model_name}_api_key", os.getenv("api_key"))
    config.model.base_url = os.getenv(f"{model_name}_base_url", os.getenv("base_url"))
    config.model.device = "cpu"  # Fallback an toàn
    config.graph.graph_db_path = os.getenv("graph_db_path", "./data/processed/graph.pkl")

    logger.info(f"Initializing LegalGraphRAG with model {model_name}...")
    rag = LegalGraphRAG(config)

    case_input = {
        "fact": args.query,
        "name": [args.name],  # Phải là list để không bị parse từng chữ cái
    }

    print("\n" + "=" * 50)
    print("🚀 RUNNING VECTOR RETRIEVAL (BASELINE)")
    print("=" * 50)

    # Configure for vector
    rag.config.retrieve.to_dict = lambda: {"method": "vector", "direct_retrieve_top_k": 3}

    start_time = time.time()
    vector_results = rag.analyze_case(case_input)
    vector_time = time.time() - start_time

    for idx, item in enumerate(vector_results):
        print(f"\n[Defendant {idx+1}]: {item.get('name')}")
        print(f"Luật đã lấy (Retrieved Laws): {len(item.get('retrieved_laws', []))}")
        print(f"Vụ án tương tự (Retrieved Facts): {len(item.get('retrieved_facts', []))}")
        print(f"Thời gian: {vector_time:.2f}s")

    print("\n" + "=" * 50)
    print("🚀 RUNNING GRAPH RETRIEVAL (PROPOSED)")
    print("=" * 50)

    # Configure for graph
    rag.config.retrieve.to_dict = lambda: {
        "method": "graph",
        "top_retrieve": True,
        "top_retrieve_top_k": 3,
        "direct_retrieve": True,
        "direct_retrieve_top_k": 3,
        "augment_retrieve": False,
    }

    start_time = time.time()
    graph_results = rag.analyze_case(case_input)
    graph_time = time.time() - start_time

    for idx, item in enumerate(graph_results):
        print(f"\n[Defendant {idx+1}]: {item.get('name')}")
        print(f"Luật đã lấy (Retrieved Laws): {len(item.get('retrieved_laws', []))}")
        print(f"Vụ án tương tự (Retrieved Facts): {len(item.get('retrieved_facts', []))}")
        print(f"Thời gian: {graph_time:.2f}s")

    print("\n" + "=" * 50)
    print("🏁 COMPARISON COMPLETE")
    print(f"Vector Time: {vector_time:.2f}s | Graph Time: {graph_time:.2f}s")
    print("=" * 50)


if __name__ == "__main__":
    main()
