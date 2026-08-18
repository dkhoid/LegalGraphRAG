#!/usr/bin/env python3
"""
LegalGraphRAG — Comprehensive Ablation Evaluation
Compares multiple retrieval configs on the tiny eval dataset.
Metrics: Dispute Accuracy, Law Precision@K, Recall@K, F1@K
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

# ── Load environment BEFORE importing anything from core ──────────────────────
from dotenv import load_dotenv

load_dotenv(".env")

# Add project root to sys.path to allow importing 'core' when running as script
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.LegalGraphRAG import LegalGraphRAG, LegalGraphRAGConfig  # noqa: E402
from core.utils.logger import logger  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Caching to save token costs during ablation
# ─────────────────────────────────────────────────────────────────────────────
import copy  # noqa: E402
import core.utils.util as util_module  # noqa: E402
from core.preprocess.case_seg import segment_case_text_withname as orig_segment  # noqa: E402
from core.preprocess.get_features import get_features as orig_get_features  # noqa: E402
from core.judge.judge_law import judge_law as orig_judge_law  # noqa: E402
from core.judge.judge_civil import judge_civil_all as orig_judge_civil  # noqa: E402

segment_cache = {}
features_cache = {}
judge_law_cache = {}
judge_civil_cache = {}


def cached_segment(chatbot, text, name):
    cache_key = f"{name}___{text}"
    if cache_key not in segment_cache:
        segment_cache[cache_key] = orig_segment(chatbot, text, name)
    return copy.deepcopy(segment_cache[cache_key])


def cached_get_features(chatbot, item):
    cache_key = f"{item.get('name')}___{item.get('description')}"
    if cache_key not in features_cache:
        features_cache[cache_key] = orig_get_features(chatbot, item)
    return copy.deepcopy(features_cache[cache_key])


def cached_judge_law(chatbot, text, law):
    cache_key = f"{text}___law_{law.get('id')}"
    if cache_key not in judge_law_cache:
        judge_law_cache[cache_key] = orig_judge_law(chatbot, text, law)
    return copy.deepcopy(judge_law_cache[cache_key])


def cached_judge_civil_all(chatbot, law_used, fact_used, text):
    law_ids = "_".join(str(law.get("id")) for law in law_used)
    fact_ids = "_".join(str(f.get("caseId")) for f in fact_used)
    cache_key = f"{text}___laws_{law_ids}___facts_{fact_ids}"
    if cache_key not in judge_civil_cache:
        judge_civil_cache[cache_key] = orig_judge_civil(chatbot, law_used, fact_used, text)
    return copy.deepcopy(judge_civil_cache[cache_key])


# Monkey patch core module so analyze_case uses our cached versions
util_module.segment_case_text_withname = cached_segment
util_module.get_features = cached_get_features
util_module.judge_law = cached_judge_law
util_module.judge_civil_all = cached_judge_civil_all

# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────


def normalize(text: str) -> str:
    return str(text).lower().strip()


import re  # noqa: E402


def extract_law_numbers(law_str: str) -> List[str]:
    """Extract standard law article numbers from strings like 'Điều 90', '90', or 'zalo_23/2018'."""
    s = str(law_str).strip()
    # If string contains "Điều X" or "điều X"
    match = re.search(r"(?:điều|article)\s*(\d+)", s, re.IGNORECASE)
    if match:
        return [match.group(1)]
    # Extract standalone numbers or numbers before slash
    nums = re.findall(r"\b\d+\b", s)
    return nums if nums else [s]


def law_precision_recall_f1(
    pred_laws: List[str], true_laws: List[str]
) -> Tuple[float, float, float]:
    """Set-based Precision, Recall, F1 for law articles using extracted article numbers."""
    pred_set = set()
    for pl in pred_laws:
        pred_set.update(extract_law_numbers(pl))

    true_set = set()
    for tl in true_laws:
        true_set.update(extract_law_numbers(tl))

    if not pred_set:
        return 0.0, 0.0, 0.0
    tp = len(pred_set & true_set)
    p = tp / len(pred_set)
    r = tp / len(true_set) if true_set else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def dispute_hit(pred_disputes: List[str], true_disputes: List[str]) -> bool:
    """True if any predicted dispute matches any true dispute (exact, substring, token overlap, or root match)."""
    if not pred_disputes or not true_disputes:
        return False

    # Common Vietnamese legal root terms
    root_synonyms = {
        "lương": ["tiền lương", "nợ lương", "trả lương"],
        "sa thải": ["đơn phương chấm dứt", "chấm dứt hợp đồng", "kỷ luật"],
        "tai nạn": ["tnlđ", "tai nạn lao động", "bồi thường thiệt hại"],
        "thai sản": ["lao động nữ", "mang thai", "chế độ thai sản"],
        "cạnh tranh": ["không cạnh tranh", "bảo mật"],
    }

    stop_words = {
        "do",
        "về",
        "và",
        "không",
        "các",
        "nhà",
        "người",
        "tranh",
        "chấp",
        "lao",
        "động",
        "vi",
        "phạm",
        "hợp",
        "đồng",
        "quyền",
        "lợi",
        "nghĩa",
        "vụ",
        "sự",
        "việc",
    }

    for pd in pred_disputes:
        pd_norm = normalize(pd)
        if not pd_norm:
            continue
        pd_words = set(pd_norm.split())

        for td in true_disputes:
            td_norm = normalize(td)
            if not td_norm:
                continue
            td_words = set(td_norm.split())

            # Exact or substring match
            if pd_norm in td_norm or td_norm in pd_norm:
                return True

            # Root synonym check
            for root, syns in root_synonyms.items():
                if (root in pd_norm or any(s in pd_norm for s in syns)) and (
                    root in td_norm or any(s in td_norm for s in syns)
                ):
                    return True

            # Token overlap
            overlap = pd_words & td_words
            meaningful_overlap = [w for w in overlap if len(w) > 1 and w not in stop_words]
            meaningful_td_words = [w for w in td_words if len(w) > 1 and w not in stop_words]

            # Require at least 2 meaningful overlapping words, OR at least 50% overlap of meaningful words
            if len(meaningful_overlap) >= 2 or (
                len(meaningful_td_words) > 0
                and len(meaningful_overlap) / len(meaningful_td_words) >= 0.5
            ):
                return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Extract predictions from analyze_case() output
# ─────────────────────────────────────────────────────────────────────────────


def extract_predictions(case_res: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    pred_disputes: List[str] = []
    pred_laws: List[str] = []

    for defendant_res in case_res:
        judge_result = defendant_res.get("judge_result", {})

        if isinstance(judge_result, dict):
            # Disputes from final verdict
            charges = judge_result.get("dispute_type", [])
            if isinstance(charges, list):
                pred_disputes.extend(charges)
            elif isinstance(charges, str):
                pred_disputes.append(charges)

            # Law articles cited in final verdict (most accurate source)
            articles = judge_result.get("law_article", [])
            if isinstance(articles, list):
                pred_laws.extend([str(a) for a in articles])
            elif articles:
                pred_laws.append(str(articles))

        # Also pull from used_laws (laws that passed judge_law filter — capped list)
        # Prefer used_laws over retrieved_laws to avoid 1000s of false positives
        used = defendant_res.get("used_laws", [])
        if used:
            for law in used:
                entry = law.get("entry")
                if entry:
                    pred_laws.append(str(entry))
        else:
            # Fallback: top retrieved laws (first 8 only)
            for law in defendant_res.get("retrieved_laws", [])[:8]:
                entry = law.get("entry")
                if entry:
                    pred_laws.append(str(entry))

    return list(dict.fromkeys(pred_disputes)), list(dict.fromkeys(pred_laws))


# ─────────────────────────────────────────────────────────────────────────────
# Experiment configs (ablation study)
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENT_CONFIGS = [
    {
        "name": "Graph_Full",
        "description": "Graph RAG: Community + Direct + Augment",
        "retrieve": {
            "method": "graph",
            "top_retrieve": True,
            "direct_retrieve": True,
            "augment_retrieve": True,
            "top_retrieve_top_k": 3,
            "direct_retrieve_top_k": 3,
            "max_judge_laws": 6,
        },
    },
    {
        "name": "Graph_No_Augment",
        "description": "Graph RAG: Community + Direct (no augment)",
        "retrieve": {
            "method": "graph",
            "top_retrieve": True,
            "direct_retrieve": True,
            "augment_retrieve": False,
            "top_retrieve_top_k": 3,
            "direct_retrieve_top_k": 3,
            "max_judge_laws": 6,
        },
    },
    {
        "name": "Graph_Community_Only",
        "description": "Graph RAG: Community retrieval only",
        "retrieve": {
            "method": "graph",
            "top_retrieve": True,
            "direct_retrieve": False,
            "augment_retrieve": False,
            "top_retrieve_top_k": 3,
            "direct_retrieve_top_k": 3,
            "max_judge_laws": 6,
        },
    },
    {
        "name": "Graph_Direct_Only",
        "description": "Graph RAG: Direct (semantic) retrieval only",
        "retrieve": {
            "method": "graph",
            "top_retrieve": False,
            "direct_retrieve": True,
            "augment_retrieve": False,
            "top_retrieve_top_k": 3,
            "direct_retrieve_top_k": 3,
            "max_judge_laws": 6,
        },
    },
    {
        "name": "Vector_Hybrid",
        "description": "Baseline: Hybrid BM25 + Vector (no graph)",
        "retrieve": {
            "method": "vector",
            "top_retrieve": False,
            "direct_retrieve": True,
            "augment_retrieve": False,
            "top_retrieve_top_k": 5,
            "direct_retrieve_top_k": 5,
            "max_judge_laws": 6,
        },
    },
    {
        "name": "Graph_Full_TopK5",
        "description": "Graph RAG Full (top_k=5)",
        "retrieve": {
            "method": "graph",
            "top_retrieve": True,
            "direct_retrieve": True,
            "augment_retrieve": True,
            "top_retrieve_top_k": 5,
            "direct_retrieve_top_k": 5,
            "max_judge_laws": 8,
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Run one experiment
# ─────────────────────────────────────────────────────────────────────────────


def run_experiment(
    rag: LegalGraphRAG,
    test_cases: List[Dict[str, Any]],
    exp_config: Dict[str, Any],
) -> Dict[str, Any]:
    from core.utils.util import analyze_case

    retrieve_cfg = exp_config["retrieve"]
    dispute_hits = []
    law_precisions, law_recalls, law_f1s = [], [], []
    per_case_results = []

    start = time.time()
    for idx, case in enumerate(test_cases, 1):
        true_disputes = case.get("dispute", [])
        true_laws = [str(law) for law in case.get("laws", case.get("law", []))]

        logger.info(f"  [{idx}/{len(test_cases)}] Bắt đầu xử lý Case: {case.get('id')}...")

        try:
            case_res = analyze_case(rag.model, case, rag.law_to_dispute, rag.cases_db, retrieve_cfg)
        except Exception as e:
            logger.error(f"  [!] Case {case.get('id')} bị lỗi: {e}")
            per_case_results.append(
                {
                    "id": case.get("id"),
                    "error": str(e),
                    "true_disputes": true_disputes,
                    "true_laws": true_laws,
                }
            )
            dispute_hits.append(False)
            law_precisions.append(0.0)
            law_recalls.append(0.0)
            law_f1s.append(0.0)
            continue

        pred_disputes, pred_laws = extract_predictions(case_res)
        hit = dispute_hit(pred_disputes, true_disputes)
        p, r, f1 = law_precision_recall_f1(pred_laws, true_laws)

        dispute_hits.append(hit)
        law_precisions.append(p)
        law_recalls.append(r)
        law_f1s.append(f1)

        per_case_results.append(
            {
                "id": case.get("id"),
                "true_disputes": true_disputes,
                "pred_disputes": pred_disputes,
                "true_laws": true_laws,
                "pred_laws": pred_laws,
                "dispute_hit": hit,
                "law_precision": round(p, 4),
                "law_recall": round(r, 4),
                "law_f1": round(f1, 4),
            }
        )
        logger.info(f"    - True Disputes: {true_disputes}")
        logger.info(f"    - Pred Disputes: {pred_disputes}")
        logger.info(f"    - True Laws: {true_laws}")
        logger.info(f"    - Pred Laws: {pred_laws}")
        logger.info(
            f"    -> Kết quả: Dispute={'✅' if hit else '❌'} | Precision={p:.2f} | Recall={r:.2f} | F1={f1:.2f}\n"
        )

    elapsed = time.time() - start
    n = len(test_cases)

    return {
        "experiment": exp_config["name"],
        "description": exp_config["description"],
        "n_cases": n,
        "dispute_accuracy": round(sum(dispute_hits) / n, 4) if n else 0.0,
        "law_precision_macro": round(sum(law_precisions) / n, 4) if n else 0.0,
        "law_recall_macro": round(sum(law_recalls) / n, 4) if n else 0.0,
        "law_f1_macro": round(sum(law_f1s) / n, 4) if n else 0.0,
        "elapsed_seconds": round(elapsed, 2),
        "avg_seconds_per_case": round(elapsed / n, 2) if n else 0.0,
        "per_case": per_case_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────


def build_markdown_report(all_results: List[Dict[str, Any]], output_path: str):
    lines = [
        "# LegalGraphRAG — Ablation Evaluation Report",
        "",
        "**Dataset**: tiny_eval (5 cases, Vietnamese Labour Law)",
        f"**Model**: {os.getenv('model_name', 'gpt4o_mini')}",
        f"**Embedding**: {os.getenv('embedding_model', 'text-embedding-3-small')}",
        "",
        "## 📊 Summary Table",
        "",
        "| Config | Description | Dispute Acc | Law P | Law R | Law F1 | Sec/case |",
        "|--------|-------------|:-----------:|:-----:|:-----:|:------:|:--------:|",
    ]

    for r in all_results:
        lines.append(
            f"| **{r['experiment']}** | {r['description']} "
            f"| {r['dispute_accuracy']:.2%} "
            f"| {r['law_precision_macro']:.2%} "
            f"| {r['law_recall_macro']:.2%} "
            f"| {r['law_f1_macro']:.2%} "
            f"| {r['avg_seconds_per_case']:.1f}s |"
        )

    lines += [
        "",
        "## 📈 Metric Definitions",
        "",
        "- **Dispute Acc**: % of cases where ≥1 predicted dispute matches a ground-truth dispute (case-insensitive)",
        "- **Law P/R/F1**: Set-based Precision/Recall/F1 on predicted vs. true law article numbers",
        "- All metrics are macro-averaged across cases",
        "",
        "## 🔍 Per-Case Breakdown",
        "",
    ]

    for r in all_results:
        lines += [
            f"### {r['experiment']} — {r['description']}",
            "",
            "| Case ID | Dispute Hit | Law P | Law R | Law F1 | Pred Disputes | True Laws | Pred Laws |",
            "|---------|:-----------:|:-----:|:-----:|:------:|---------------|-----------|-----------|",
        ]
        for c in r.get("per_case", []):
            if "error" in c:
                lines.append(f"| {c['id']} | ❌ ERROR | — | — | — | {c['error'][:60]} | — | — |")
            else:
                hit_icon = "✅" if c["dispute_hit"] else "❌"
                pred_d = "<br>".join(c["pred_disputes"][:2]) or "—"
                true_l = ", ".join(c["true_laws"][:5])
                pred_l = ", ".join(c["pred_laws"][:5]) or "—"
                lines.append(
                    f"| {c['id']} | {hit_icon} "
                    f"| {c['law_precision']:.2f} "
                    f"| {c['law_recall']:.2f} "
                    f"| {c['law_f1']:.2f} "
                    f"| {pred_d} | {true_l} | {pred_l} |"
                )
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Markdown report saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    eval_data_path = "./data/processed/tiny_eval.json"
    output_dir = "./outputs/ablation"
    os.makedirs(output_dir, exist_ok=True)

    with open(eval_data_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    logger.info(f"Loaded {len(test_cases)} eval cases from {eval_data_path}")

    # Shared RAG instance — loads graph from pkl, does NOT rebuild
    config = LegalGraphRAGConfig.from_env_file(".env")
    config.graph.auto_build = False

    logger.info("Initializing LegalGraphRAG (loading existing graph)...")
    rag = LegalGraphRAG(config=config)
    logger.info("RAG ready. Starting ablation experiments...\n")

    # Run all experiments (Ablation configs)
    experiments_to_run = EXPERIMENT_CONFIGS
    all_results = []

    for exp in experiments_to_run:
        logger.info("=" * 60)
        logger.info(f"[EXP] {exp['name']} — {exp['description']}")
        logger.info("=" * 60)

        result = run_experiment(rag, test_cases, exp)
        all_results.append(result)

        logger.info(
            f"  ✔ Dispute Acc={result['dispute_accuracy']:.2%} | "
            f"Law P={result['law_precision_macro']:.2%} | "
            f"Law R={result['law_recall_macro']:.2%} | "
            f"Law F1={result['law_f1_macro']:.2%} | "
            f"{result['avg_seconds_per_case']:.1f}s/case\n"
        )

    # Save raw JSON
    json_path = os.path.join(output_dir, "ablation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"Raw results → {json_path}")

    # Save markdown report
    md_path = os.path.join(output_dir, "ablation_report.md")
    build_markdown_report(all_results, md_path)

    # Print final table
    print("\n" + "=" * 70)
    print(f"{'Config':<26} {'Dispute Acc':>12} {'Law P':>8} {'Law R':>8} {'Law F1':>8}")
    print("-" * 70)
    for r in all_results:
        print(
            f"{r['experiment']:<26} "
            f"{r['dispute_accuracy']:>11.2%} "
            f"{r['law_precision_macro']:>7.2%} "
            f"{r['law_recall_macro']:>7.2%} "
            f"{r['law_f1_macro']:>7.2%}"
        )
    print("=" * 70)
    print(f"\n📄 Report: {md_path}")
    print(f"📊 JSON:   {json_path}")


if __name__ == "__main__":
    main()
