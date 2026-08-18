import os
import sys
import time
import argparse
import random
import pandas as pd
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv()

from core.LegalGraphRAG import LegalGraphRAG, LegalGraphRAGConfig  # noqa: E402
from core.utils.logger import logger  # noqa: E402


def calculate_precision(retrieved_res):
    retrieved = sum(len(r.get("retrieved_laws", [])) for r in retrieved_res)
    used = sum(len(r.get("used_laws", [])) for r in retrieved_res)
    return used / retrieved if retrieved > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Multi-Model & Multi-Parameter Benchmark")
    parser.add_argument("--num_samples", type=int, default=5, help="Số lượng vụ án test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--models",
        type=str,
        default="gpt4o_mini",
        help="Comma-separated list of models (e.g. gpt4o_mini,deepseek_v3)",
    )
    parser.add_argument(
        "--top_ks", type=str, default="3,5", help="Comma-separated list of top_k values (e.g. 3,5)"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    models_to_test = [m.strip() for m in args.models.split(",")]
    top_ks_to_test = [int(k.strip()) for k in args.top_ks.split(",")]

    # 1. Load Data with a Base Config to get cases_db
    base_config = LegalGraphRAGConfig()
    base_config.model.model_name = "gpt4o_mini"
    base_config.model.device = "cpu"
    base_config.graph.auto_build = False
    base_rag = LegalGraphRAG(base_config)

    cases_db = base_rag.cases_db
    if not cases_db:
        logger.error("Không tìm thấy cases_db.")
        return

    # Lọc các vụ án có chứa nhãn 'law' để làm ground truth
    valid_cases = [c for c in cases_db if c.get("law") or c.get("laws")]
    if len(valid_cases) < args.num_samples:
        test_cases = valid_cases
    else:
        test_cases = random.sample(valid_cases, args.num_samples)

    logger.info(f"Đã chọn {len(test_cases)} vụ án làm Ground-truth dataset.")

    results_data = []

    for model_name in models_to_test:
        logger.info(f"\n{'='*50}\nĐang khởi tạo model: {model_name}\n{'='*50}")

        config = LegalGraphRAGConfig()
        config.model.model_name = model_name
        config.model.api_key = os.getenv(f"{model_name}_api_key", os.getenv("api_key"))
        config.model.base_url = os.getenv(f"{model_name}_base_url", os.getenv("base_url"))
        config.model.device = "cpu"
        config.graph.graph_db_path = os.getenv("graph_db_path", "./data/processed/graph.pkl")

        try:
            rag = LegalGraphRAG(config)
        except Exception as e:
            logger.error(f"Lỗi khởi tạo model {model_name}: {e}")
            continue

        for top_k in top_ks_to_test:
            for method in ["vector", "graph"]:
                logger.info(
                    f"Đang chạy cấu hình: Model={model_name} | Method={method} | Top_K={top_k}"
                )

                if method == "vector":
                    rag.config.retrieve.to_dict = lambda: {
                        "method": "vector",
                        "direct_retrieve_top_k": top_k,
                    }
                else:
                    rag.config.retrieve.to_dict = lambda: {
                        "method": "graph",
                        "top_retrieve": True,
                        "top_retrieve_top_k": top_k,
                        "direct_retrieve": True,
                        "direct_retrieve_top_k": top_k,
                        "augment_retrieve": False,
                    }

                total_precision = 0.0
                total_time = 0.0

                for idx, case in enumerate(test_cases):
                    case_input = {"fact": case.get("fact", ""), "name": ["Nguyên đơn", "Bị đơn"]}

                    start_time = time.time()
                    try:
                        res = rag.analyze_case(case_input)
                        elapsed = time.time() - start_time
                        precision = calculate_precision(res)
                    except Exception as e:
                        logger.warning(f"Lỗi khi xử lý case {case.get('id')} với {model_name}: {e}")
                        precision = 0.0
                        elapsed = time.time() - start_time

                    total_precision += precision
                    total_time += elapsed

                avg_precision = total_precision / len(test_cases)
                avg_time = total_time / len(test_cases)

                results_data.append(
                    {
                        "Model": model_name,
                        "Method": method.upper(),
                        "Top_K": top_k,
                        "Avg_Context_Precision": round(avg_precision, 4),
                        "Avg_Time_Sec": round(avg_time, 2),
                    }
                )

    print("\n" + "=" * 80)
    print("📊 KẾT QUẢ ĐÁNH GIÁ ĐA MÔ HÌNH VÀ THÔNG SỐ")
    print("=" * 80)
    df = pd.DataFrame(results_data)
    print(df.to_string(index=False))

    # Save to CSV
    os.makedirs("data/outputs/eval", exist_ok=True)
    out_path = f"data/outputs/eval/benchmark_{int(time.time())}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✅ Đã lưu kết quả chi tiết tại: {out_path}")


if __name__ == "__main__":
    main()
