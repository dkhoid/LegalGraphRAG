import os
import sys
import json
import time
import argparse
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    # Disable graph building, we just use the local pkl
    from dotenv import load_dotenv

    load_dotenv()

    config = LegalGraphRAGConfig()
    config.model.model_name = "gpt4o_mini"
    config.model.device = "cpu"
    config.graph.graph_db_path = "./data/processed/graph.pkl"
    rag = LegalGraphRAG(config)

    cases_db = rag.cases_db
    valid_cases = [c for c in cases_db if c.get("law") and c.get("features")]

    test_cases = random.sample(valid_cases, min(args.num_samples, len(valid_cases)))
    logger.info(f"Đã chọn {len(test_cases)} vụ án để test Siêu Tốc (Fast Eval).")

    from core.retriever.graph_retriever import GraphRetriever

    retriever = GraphRetriever(
        model=rag.model
    )  # Pass the model (only used for embedding in GraphRetriever)

    results = {"precision": [], "recall": [], "f1": [], "time": []}

    retrieve_config = {
        "top_retrieve": True,
        "top_retrieve_top_k": 5,
        "direct_retrieve": True,
        "direct_retrieve_top_k": 5,
        "augment_retrieve": False,
    }

    for idx, case in enumerate(test_cases):
        print(f"\rĐang chấm điểm [{idx+1}/{len(test_cases)}]...", end="")

        # Prepare input for retriever by bypassing LLM generation
        item = {"feature": case["features"]}

        start_time = time.time()
        # Retrieve laws based on pre-extracted features
        _, retrieved_laws, _ = retriever.retrieve(
            item, rag.law_to_dispute, cases_db, retrieve_config
        )
        elapsed = time.time() - start_time

        ground_truth = case.get("law", [])
        retrieved_ids = [l.get("entry") for l in retrieved_laws]

        p, r, f1 = calculate_metrics(retrieved_ids, ground_truth)
        results["precision"].append(p)
        results["recall"].append(r)
        results["f1"].append(f1)
        results["time"].append(elapsed)

    print("\n\n" + "=" * 50)
    print("🚀 KẾT QUẢ ĐÁNH GIÁ TRUY VẤN (RETRIEVAL METRICS) 🚀")
    print("=" * 50)
    print(f"Số lượng vụ án test : {len(test_cases)}")
    print(f"Độ chính xác (Prec) : {sum(results['precision']) / len(results['precision']):.2f}")
    print(f"Độ phủ (Recall)     : {sum(results['recall']) / len(results['recall']):.2f}")
    print(f"Điểm F1 trung bình  : {sum(results['f1']) / len(results['f1']):.2f}")
    print(f"Tốc độ trung bình   : {sum(results['time']) / len(results['time']):.2f} giây/vụ")
    print("=" * 50)


if __name__ == "__main__":
    main()
