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


import math


def calculate_metrics(retrieved_law_ids, ground_truth_law_ids, k=3):
    retrieved_list = []
    # Deduplicate while preserving order
    for x in retrieved_law_ids:
        xs = str(x)
        if xs not in retrieved_list:
            retrieved_list.append(xs)

    truth_set = set(str(x) for x in ground_truth_law_ids)

    if not truth_set:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    retrieved_set = set(retrieved_list)
    true_positives = len(retrieved_set.intersection(truth_set))

    precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
    recall = true_positives / len(truth_set) if truth_set else 0.0

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    # Calculate MRR
    mrr = 0.0
    for i, law_id in enumerate(retrieved_list):
        if law_id in truth_set:
            mrr = 1.0 / (i + 1)
            break

    # Calculate NDCG@K
    dcg = 0.0
    idcg = 0.0
    for i in range(min(k, len(retrieved_list))):
        if retrieved_list[i] in truth_set:
            dcg += 1.0 / math.log2(i + 2)
    for i in range(min(k, len(truth_set))):
        idcg += 1.0 / math.log2(i + 2)

    ndcg = dcg / idcg if idcg > 0 else 0.0

    return precision, recall, f1, mrr, ndcg


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
        logger.warning("cases_db in rag is None, manually loading...")
        try:
            cases_db = rag._load_cases_db()
            rag.cases_db = cases_db
        except Exception as e:
            logger.error(f"Failed to load cases_db: {e}")
            return

    # Lọc các vụ án có chứa nhãn 'law' để làm ground truth
    valid_cases = [c for c in cases_db if c.get("law")]
    if len(valid_cases) < args.num_samples:
        test_cases = valid_cases
    else:
        test_cases = random.sample(valid_cases, args.num_samples)

    logger.info(f"Đã chọn {len(test_cases)} vụ án làm Ground-truth dataset.")

    results = {
        "vector": {
            "precision": [],
            "recall": [],
            "f1": [],
            "mrr": [],
            "ndcg": [],
            "time": [],
            "context_precision": [],
        },
        "graph": {
            "precision": [],
            "recall": [],
            "f1": [],
            "mrr": [],
            "ndcg": [],
            "time": [],
            "context_precision": [],
        },
    }

    for idx, case in enumerate(test_cases):
        print(f"\n[{idx+1}/{len(test_cases)}] Đang chấm điểm Case ID: {case['id']}...")
        case_input = {"fact": case["fact"], "name": ["Nguyên đơn", "Bị đơn"]}

        # 1. Test Vector
        rag.config.retrieve.to_dict = lambda: {"method": "vector", "direct_retrieve_top_k": 3}
        start_time = time.time()
        vector_res = rag.analyze_case(case_input)
        vec_time = time.time() - start_time

        vec_retrieved_laws = []
        for r in vector_res:
            laws = r.get("retrieved_laws", [])
            # Extract just the ID if it's a dict, otherwise convert to str
            for law in laws:
                if isinstance(law, dict):
                    entry = str(law.get("entry", ""))
                    law_id = entry.split("+")[-1] if "+" in entry else str(law.get("id", ""))
                else:
                    law_id = str(law).split("+")[-1] if "+" in str(law) else str(law)
                vec_retrieved_laws.append(law_id)

        vec_p, vec_r, vec_f1, vec_mrr, vec_ndcg = calculate_metrics(
            vec_retrieved_laws, case["law"], k=5
        )

        results["vector"]["precision"].append(vec_p)
        results["vector"]["recall"].append(vec_r)
        results["vector"]["f1"].append(vec_f1)
        results["vector"]["mrr"].append(vec_mrr)
        results["vector"]["ndcg"].append(vec_ndcg)

        vec_used = sum(len(r.get("used_laws", [])) for r in vector_res)
        vec_context_p = vec_used / len(vec_retrieved_laws) if vec_retrieved_laws else 0.0
        results["vector"]["context_precision"].append(vec_context_p)
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

        graph_retrieved_laws = []
        for r in graph_res:
            laws = r.get("retrieved_laws", [])
            for law in laws:
                if isinstance(law, dict):
                    entry = str(law.get("entry", ""))
                    law_id = entry.split("+")[-1] if "+" in entry else str(law.get("id", ""))
                else:
                    law_id = str(law).split("+")[-1] if "+" in str(law) else str(law)
                graph_retrieved_laws.append(law_id)

        graph_p, graph_r, graph_f1, graph_mrr, graph_ndcg = calculate_metrics(
            graph_retrieved_laws, case["law"], k=5
        )

        results["graph"]["precision"].append(graph_p)
        results["graph"]["recall"].append(graph_r)
        results["graph"]["f1"].append(graph_f1)
        results["graph"]["mrr"].append(graph_mrr)
        results["graph"]["ndcg"].append(graph_ndcg)

        graph_used = sum(len(r.get("used_laws", [])) for r in graph_res)
        graph_context_p = graph_used / len(graph_retrieved_laws) if graph_retrieved_laws else 0.0
        results["graph"]["context_precision"].append(graph_context_p)
        results["graph"]["time"].append(graph_time)

    # Calculate averages
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    print("\n" + "=" * 70)
    print("📊 KẾT QUẢ ĐÁNH GIÁ BENCHMARK TỔNG HỢP")
    print("=" * 70)
    print(f"Số lượng mẫu thử: {len(test_cases)}")
    print("\n📌 1. VECTOR RETRIEVAL (CƠ BẢN)")
    print(f" - Precision: {avg(results['vector']['precision']):.4f}")
    print(f" - Recall:    {avg(results['vector']['recall']):.4f}")
    print(f" - F1-Score:  {avg(results['vector']['f1']):.4f}")
    print(f" - MRR:       {avg(results['vector']['mrr']):.4f}")
    print(f" - NDCG@5:    {avg(results['vector']['ndcg']):.4f}")
    print(
        f" - Context P: {avg(results['vector']['context_precision']):.4f} (Used/Retrieved by LLM)"
    )
    print(f" - Avg Time:  {avg(results['vector']['time']):.2f}s")

    print("\n📌 2. GRAPH RETRIEVAL (NÂNG CAO)")
    print(f" - Precision: {avg(results['graph']['precision']):.4f}")
    print(f" - Recall:    {avg(results['graph']['recall']):.4f}")
    print(f" - F1-Score:  {avg(results['graph']['f1']):.4f}")
    print(f" - MRR:       {avg(results['graph']['mrr']):.4f}")
    print(f" - NDCG@5:    {avg(results['graph']['ndcg']):.4f}")
    print(f" - Context P: {avg(results['graph']['context_precision']):.4f} (Used/Retrieved by LLM)")
    print(f" - Avg Time:  {avg(results['graph']['time']):.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
