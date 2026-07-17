import os
import sys
import json
import time
import argparse
import random
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv()

from core.LegalGraphRAG import LegalGraphRAG, LegalGraphRAGConfig
from core.utils.logger import logger


def calculate_metrics(retrieved_law_ids, ground_truth_law_ids):
    retrieved_set = set(str(x) for x in retrieved_law_ids)
    truth_set = set(str(x) for x in ground_truth_law_ids)

    if not truth_set:
        return 0.0, 0.0, 0.0

    true_positives = len(retrieved_set.intersection(truth_set))

    precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
    recall = true_positives / len(truth_set) if truth_set else 0.0

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description="Run Benchmark Evaluation for Retrievers")
    parser.add_argument("--num_samples", type=int, default=10, help="Số lượng vụ án test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    config = LegalGraphRAGConfig()
    model_name = os.getenv("model_name", "gpt4o_mini")
    config.model.model_name = model_name
    config.model.api_key = os.getenv(f"{model_name}_api_key", os.getenv("api_key"))
    config.model.base_url = os.getenv(f"{model_name}_base_url", os.getenv("base_url"))
    config.model.device = "cpu"
    config.graph.graph_db_path = os.getenv("graph_db_path", "./data/processed/graph.pkl")

    logger.info("Khởi tạo RAG System cho Benchmark...")
    rag = LegalGraphRAG(config)

    cases_db = rag.cases_db
    if not cases_db:
        logger.error("Không tìm thấy cases_db.")
        return

    # Lọc các vụ án có chứa nhãn 'law' để làm ground truth
    valid_cases = [c for c in cases_db if c.get("law")]
    if len(valid_cases) < args.num_samples:
        test_cases = valid_cases
    else:
        test_cases = random.sample(valid_cases, args.num_samples)

    logger.info(f"Đã chọn {len(test_cases)} vụ án làm Ground-truth dataset.")

    results = {
        "vector": {"precision": [], "recall": [], "f1": [], "time": []},
        "graph": {"precision": [], "recall": [], "f1": [], "time": []},
    }

    for idx, case in enumerate(test_cases):
        print(f"\n[{idx+1}/{len(test_cases)}] Đang chấm điểm Case ID: {case['id']}...")
        case_input = {"fact": case["fact"], "name": ["Nguyên đơn", "Bị đơn"]}

        # 1. Test Vector
        rag.config.retrieve.to_dict = lambda: {"method": "vector", "direct_retrieve_top_k": 3}
        start_time = time.time()
        vector_res = rag.analyze_case(case_input)
        vec_time = time.time() - start_time

        vec_retrieved = sum(len(r.get("retrieved_laws", [])) for r in vector_res)
        vec_used = sum(len(r.get("used_laws", [])) for r in vector_res)
        vec_precision = vec_used / vec_retrieved if vec_retrieved > 0 else 0.0

        results["vector"]["precision"].append(vec_precision)
        results["vector"]["time"].append(vec_time)

        # 2. Test Graph
        rag.config.retrieve.to_dict = lambda: {
            "method": "graph",
            "top_retrieve": True,
            "top_retrieve_top_k": 3,
            "direct_retrieve": True,
            "direct_retrieve_top_k": 3,
            "augment_retrieve": False,
        }
        start_time = time.time()
        graph_res = rag.analyze_case(case_input)
        graph_time = time.time() - start_time

        graph_retrieved = sum(len(r.get("retrieved_laws", [])) for r in graph_res)
        graph_used = sum(len(r.get("used_laws", [])) for r in graph_res)
        graph_precision = graph_used / graph_retrieved if graph_retrieved > 0 else 0.0

        results["graph"]["precision"].append(graph_precision)
        results["graph"]["time"].append(graph_time)

    # Calculate averages
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    print("\n" + "=" * 60)
    print("📊 KẾT QUẢ ĐÁNH GIÁ BENCHMARK (Context Precision)")
    print("=" * 60)
    print(f"Số lượng mẫu thử: {len(test_cases)}")
    print("\n📌 1. VECTOR RETRIEVAL (CƠ BẢN)")
    print(f" - Context Precision: {avg(results['vector']['precision']):.4f}")
    print(f" - Avg Time:  {avg(results['vector']['time']):.2f}s")

    print("\n📌 2. GRAPH RETRIEVAL (NÂNG CAO)")
    print(f" - Context Precision: {avg(results['graph']['precision']):.4f}")
    print(f" - Avg Time:  {avg(results['graph']['time']):.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
