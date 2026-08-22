import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.utils.logger import logger
import argparse
import os
import json
from tqdm import tqdm
import multiprocessing
import time
from typing import List, Dict, Any, Optional

from core.LegalGraphRAG import LegalGraphRAG, LegalGraphRAGConfig


def load_test_cases(datasets: str, datasets_path: str = "./data/eval_tiny") -> List[Dict[str, Any]]:
    case_file = os.path.join(datasets_path, "cases_with_feature.json")

    if not os.path.exists(case_file):
        raise FileNotFoundError(f"Test dataset not found: {case_file}")

    with open(case_file, "r", encoding="utf-8") as f:
        cases = json.load(f)

    return cases


def process_cases_worker(
    cases: List[Dict[str, Any]],
    config_dict: Dict[str, Any],
    device: str,
    output_file: str,
    model_name: str,
):
    config = LegalGraphRAGConfig.from_dict(config_dict)

    config.model.device = device
    config.model.model_name = model_name

    rag = LegalGraphRAG(config=config)

    results = []
    correct_count = 0

    try:
        from scripts.evaluation.calculate_step_metrics import extract_law_numbers, dispute_hit

        for case in tqdm(cases, desc=f"Processing on {device} with {model_name}"):
            fact = case.get("fact", "")
            true_dispute = case.get("dispute", [])
            law_article = case.get("laws", case.get("law", []))

            case_res = rag.analyze_case(case)

            results.append(
                {
                    "id": case.get("id"),
                    "fact": fact,
                    "true_dispute": true_dispute,
                    "judge_res": case_res,
                    "law_article": law_article,
                }
            )

            # Evaluate correctness
            if case_res and len(case_res) > 0:
                res = case_res[0]
                judge_result = res.get("judge_result", {})

                # Check Dispute
                pred_disputes = []
                if isinstance(judge_result, dict):
                    charges = judge_result.get("dispute_type", [])
                    if isinstance(charges, list):
                        pred_disputes.extend(charges)
                    elif isinstance(charges, str):
                        pred_disputes.append(charges)
                elif isinstance(judge_result, list):
                    pred_disputes.extend(judge_result)

                true_disputes_list = (
                    true_dispute if isinstance(true_dispute, list) else [true_dispute]
                )
                is_dispute_correct = dispute_hit(pred_disputes, true_disputes_list)

                # Check Law
                true_laws = law_article if isinstance(law_article, list) else [law_article]
                true_law_nums = set()
                for tl in true_laws:
                    true_law_nums.update(extract_law_numbers(tl))

                pred_laws = []
                if isinstance(judge_result, dict):
                    final_law = judge_result.get("law_article", "")
                    if isinstance(final_law, list):
                        pred_laws.extend(final_law)
                    elif isinstance(final_law, str):
                        pred_laws.append(final_law)

                pred_law_nums = set()
                for pl in pred_laws:
                    pred_law_nums.update(extract_law_numbers(pl))

                is_law_correct = bool(true_law_nums and true_law_nums.intersection(pred_law_nums))

                # Case is correctly classified if both dispute and law article are correct
                if is_dispute_correct and is_law_correct:
                    correct_count += 1

            # Save progressively after every case to avoid losing data on Ctrl+C
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        return correct_count, len(cases)

    finally:
        if hasattr(rag, "model") and hasattr(rag.model, "release_model"):
            try:
                rag.model.release_model()
            except Exception:
                pass


def run_evaluation(
    model_name: str,
    datasets: str = "vn_civil",
    dotenv_path: str = ".env",
    devices: Optional[List[str]] = None,
    datasets_path: str = "./data/eval_tiny",
    build_graph: bool = True,
    force_rebuild: bool = False,
    limit: Optional[int] = None,
):
    config = LegalGraphRAGConfig.from_env_file(dotenv_path)

    output_dir = os.path.join(config.data.output_dir, datasets)
    os.makedirs(output_dir, exist_ok=True)

    # Build graph before starting parallel processes
    if build_graph:
        logger.info("=" * 60)
        logger.info("Building graph database...")
        logger.info("=" * 60)

        # Use first device for graph construction (if devices specified), otherwise use config device
        if devices and len(devices) > 0:
            build_device = devices[0]
        else:
            build_device = config.model.device

        # Create configuration for graph construction (using first device)
        build_config = LegalGraphRAGConfig.from_dict(config.to_dict())
        build_config.model.device = build_device
        build_config.model.model_name = model_name

        # Create LegalGraphRAG instance and build graph
        logger.info(f"Using device {build_device} for graph construction...")
        rag_builder = LegalGraphRAG(config=build_config)
        rag_builder.build_graph(force_rebuild=force_rebuild)

        # Release model resources used for graph construction
        if hasattr(rag_builder, "model") and hasattr(rag_builder.model, "release_model"):
            try:
                rag_builder.model.release_model()
            except Exception:
                pass

        logger.info("=" * 60)
        logger.info("Graph database ready!")
        logger.info("=" * 60)
        logger.info("")

    test_cases = load_test_cases(datasets, datasets_path)
    if limit is not None:
        test_cases = test_cases[:limit]
    logger.info(f"Loaded {len(test_cases)} test cases from {datasets} dataset")

    if devices is None:
        devices = ["cpu"]
    if not devices or len(devices) == 0:
        raise ValueError("At least one device must be specified")
    num_processes = len(devices)

    chunks = [[] for _ in range(num_processes)]
    for i, case in enumerate(test_cases):
        chunk_index = i % num_processes
        chunks[chunk_index].append(case)

    logger.info(f"Split {len(test_cases)} cases into {num_processes} processes")
    for i, chunk in enumerate(chunks):
        logger.info(f"  Process {i} ({devices[i]}): {len(chunk)} cases")

    config_dict = config.to_dict()

    pool = multiprocessing.Pool(processes=num_processes)
    async_results = []

    time_before = time.time()

    for i, chunk in enumerate(chunks):
        output_file = f"{model_name}_results_part_{i}.json"
        async_results.append(
            pool.apply_async(
                process_cases_worker,
                args=(
                    chunk,
                    config_dict,
                    devices[i],
                    output_file,
                    model_name,
                ),
            )
        )

    pool.close()
    pool.join()

    time_after = time.time()
    elapsed_time = time_after - time_before

    total_correct = 0
    total_cases = 0
    for res in async_results:
        correct, count = res.get()
        total_correct += correct
        total_cases += count

    logger.info(f"\n{'='*60}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Dataset: {datasets}")
    logger.info(f"Total cases processed: {total_cases}")
    logger.info(f"Correctly classified: {total_correct}/{total_cases}")
    logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
    logger.info(f"{'='*60}\n")

    combined_results = []
    for i in range(len(chunks)):
        part_file = f"{model_name}_results_part_{i}.json"
        if os.path.exists(part_file):
            with open(part_file, "r", encoding="utf-8") as f:
                part_data = json.load(f)
                combined_results.extend(part_data)
            os.remove(part_file)

    combined_file = os.path.join(output_dir, f"{model_name}_results_combined.json")
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, ensure_ascii=False, indent=2)

    logger.info(f"Combined results saved to {combined_file}")

    stats_file = os.path.join(output_dir, f"{model_name}_stats.json")
    stats = {
        "model_name": model_name,
        "dataset": datasets,
        "total_cases": total_cases,
        "correct_count": total_correct,
        "elapsed_time": elapsed_time,
        "output_file": combined_file,
    }
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"Statistics saved to {stats_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Legal Case Analysis with Different Models using LegalGraphRAG"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "qwen3",
            "qwen2_5",
            "gemma3",
            "internlm3",
            "glm4",
            "deepseek_v3",
            "gpt4o_mini",
        ],
        help="Model to use for analysis",
    )
    parser.add_argument(
        "--dotenv_path",
        type=str,
        default=".env",
        help="Path to the .env file",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="vn_civil",
        help="Dataset name (e.g., vn_civil)",
    )
    parser.add_argument(
        "--datasets_path",
        type=str,
        default=None,
        help="Path to datasets directory (default: ./data/eval_tiny)",
    )
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=None,
        help="GPU devices to use (e.g., cuda:2 cuda:3)",
    )
    parser.add_argument(
        "--no-build-graph",
        action="store_true",
        help="Skip graph construction (assume graph already exists)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild graph even if it already exists",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of cases to evaluate",
    )

    args = parser.parse_args()

    multiprocessing.set_start_method("spawn", force=True)

    run_evaluation(
        model_name=args.model,
        datasets=args.datasets,
        dotenv_path=args.dotenv_path,
        devices=args.devices,
        datasets_path=args.datasets_path if args.datasets_path else "./data/eval_tiny",
        build_graph=not args.no_build_graph,
        force_rebuild=args.force_rebuild,
        limit=args.limit,
    )
