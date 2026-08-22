import logging
from typing import Dict, List, Any, Tuple

from core.models.base import BaseModel
from core.retriever.graph_retriever import GraphRetriever
from core.preprocess.get_features import get_features
from core.preprocess.case_seg import segment_case_text_withname
from core.utils.confidence import compute_confidence
from core.judge.judge_law import judge_law, judge_law_batch, judge_law_self_consistent
from core.judge.judge_civil import judge_civil_all
from core.utils.util import filter_facts, _get_judge_chatbot
from core.config import RetrieveConfig

logger = logging.getLogger("LegalGraphRAG")


def _extract_features_and_segment(chatbot: BaseModel, case: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Segment case by defendant and extract features.

    Uses intelligent text handling instead of hard truncation (V1 fix):
    - Short texts (<= 6000 chars): pass directly
    - Long texts (> 6000 chars): summarize first to preserve key information from all sections
    """
    criminals = case["name"] if isinstance(case["name"], list) else [case["name"]]
    fact_text = case.get("fact", "")

    MAX_DIRECT_LENGTH = 6000
    if len(fact_text) > MAX_DIRECT_LENGTH:
        try:
            from core.graph_construct.llm_utils import summarize_texts

            logger.info(f"Case text is {len(fact_text)} chars, summarizing before segmentation...")
            summarized = summarize_texts(chatbot, fact_text[:12000])
            if summarized and len(summarized) >= 100:
                fact_text = summarized
            else:
                head = fact_text[:3000]
                tail = fact_text[-3000:]
                fact_text = head + "\n...\n" + tail
        except Exception as e:
            logger.warning(f"Summarization failed: {e}. Using head+tail fallback.")
            head = fact_text[:3000]
            tail = fact_text[-3000:]
            fact_text = head + "\n...\n" + tail

    case_by_defendant = segment_case_text_withname(chatbot, fact_text, criminals)

    for item in case_by_defendant:
        item["feature"] = get_features(chatbot, item)
    return case_by_defendant


def _retrieve_and_rerank_laws(
    retriever: GraphRetriever,
    item: Dict[str, Any],
    law_to_dispute: List[Dict[str, Any]],
    cases_db: List[Dict[str, Any]],
    retrieve_config: RetrieveConfig,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Retrieve laws and facts from graph.

    Note: Cross-encoder reranking is already applied inside GraphRetriever.retrieve()
    (step 6: Cross-Encoder Reranker pass). A second pass here was redundant
    and could distort rankings via double scoring bias (V5 fix).
    """
    original_retrieved_res, retrieved_laws, retrieved_facts = retriever.retrieve(
        item, law_to_dispute, cases_db, retrieve_config.to_dict()
    )
    return original_retrieved_res, retrieved_laws, retrieved_facts


def _evaluate_laws(
    chatbot: BaseModel,
    item: Dict[str, Any],
    retrieved_laws: List[Dict[str, Any]],
    retrieve_config: RetrieveConfig,
) -> Tuple[List[Dict[str, Any]], List[float], int]:
    """Evaluate laws using LLM judge and return (used_laws, confidence_scores, parsing_failures)."""
    laws_to_judge = retrieved_laws[: retrieve_config.max_judge_laws]

    # Easy-first ordering
    laws_to_judge = sorted(
        laws_to_judge,
        key=lambda law_item: (
            len(law_item.get("judge_dep", []))
            if isinstance(law_item.get("judge_dep", []), list)
            else 0
        ),
    )

    judge_chatbot = None
    if retrieve_config.use_self_consistent and retrieve_config.judge_chatbot:
        try:
            judge_chatbot = _get_judge_chatbot(retrieve_config.judge_chatbot)
        except Exception as e:
            logger.warning(
                f"Failed to init judge_chatbot '{retrieve_config.judge_chatbot}': {e}. "
                "Falling back to primary chatbot for self-consistency."
            )

    law_used = []
    law_confidence_scores = []
    parsing_failures = 0
    case_desc = f"Party: {item['name']}, Description: {item['description']}"

    for law in laws_to_judge:
        if len(law_used) >= retrieve_config.max_applicable_laws:
            break

        law = dict(law)
        if law.get("description") and len(law["description"]) > retrieve_config.law_desc_cap:
            law["description"] = law["description"][: retrieve_config.law_desc_cap]

        if retrieve_config.use_self_consistent:
            # V4 Adaptive Self-Consistency:
            # 1. First run a single batch judge screening pass
            initial_used, initial_reasoning = judge_law_batch(chatbot, case_desc, law)

            if (
                "failed" in str(initial_reasoning).lower()
                or "defaulted" in str(initial_reasoning).lower()
            ):
                parsing_failures += 1
                continue

            if initial_used:
                # 2. Only invoke multi-sample self-consistency for candidate laws that passed screen
                used, sc_confidence, _ = judge_law_self_consistent(
                    chatbot,
                    case_desc,
                    law,
                    n_samples=retrieve_config.self_consistent_n,
                    judge_chatbot=judge_chatbot,
                )
                if used:
                    law_used.append(law)
                    law_confidence_scores.append(sc_confidence)
                elif sc_confidence < 0.3:
                    parsing_failures += 1
        else:
            judge_fn = judge_law_batch if retrieve_config.batch_judge else judge_law
            used, reasoning = judge_fn(
                chatbot,
                case_desc,
                law,
            )
            if "failed" in str(reasoning).lower() or "defaulted" in str(reasoning).lower():
                parsing_failures += 1
            if used:
                law_used.append(law)

    return law_used, law_confidence_scores, parsing_failures


def _calculate_confidence(
    law_confidence_scores: List[float],
    retrieved_laws: List[Dict[str, Any]],
    law_used: List[Dict[str, Any]],
    retrieved_facts: List[Dict[str, Any]],
    parsing_failures: int = 0,
) -> Dict[str, Any]:
    """Calculate the confidence score of the retrieval and judgement."""
    if law_confidence_scores:
        sc_mean = sum(law_confidence_scores) / len(law_confidence_scores)
        penalty = min(parsing_failures * 0.05, 0.25)
        adj_mean = max(0.0, min(1.0, sc_mean - penalty))
        return {
            "retrieval_quality": round(min(len(retrieved_laws) / 5.0, 1.0), 3),
            "law_applicability": round(adj_mean, 3),
            "overall": round(adj_mean, 3),
            "grade": "HIGH" if adj_mean > 0.75 else "MEDIUM" if adj_mean > 0.45 else "LOW",
            "review_required": adj_mean < 0.45,
            "source": "self_consistency",
        }
    return compute_confidence(
        retrieved_laws=retrieved_laws,
        used_laws=law_used,
        retrieved_facts=retrieved_facts,
        parsing_failures=parsing_failures,
    ).to_dict()


def analyze_case_pipeline(
    chatbot: BaseModel,
    case: Dict[str, Any],
    law_to_dispute: List[Dict[str, Any]],
    cases_db: List[Dict[str, Any]],
    retrieve_config: RetrieveConfig,
) -> List[Dict[str, Any]]:
    """Main pipeline for analyzing a case."""
    retriever = GraphRetriever(chatbot)
    case_by_defendant = _extract_features_and_segment(chatbot, case)

    for item in case_by_defendant:
        original_retrieved_res, retrieved_laws, retrieved_facts = _retrieve_and_rerank_laws(
            retriever, item, law_to_dispute, cases_db, retrieve_config
        )

        if not (retrieved_laws and retrieved_facts):
            continue

        eval_res = _evaluate_laws(chatbot, item, retrieved_laws, retrieve_config)
        if len(eval_res) == 3:
            law_used, law_confidence_scores, parsing_failures = eval_res
        else:
            law_used, law_confidence_scores = eval_res
            parsing_failures = 0

        fact_used = filter_facts(law_used, retrieved_facts)

        # V9 fix: When ALL laws are rejected by the judge, do NOT force-accept
        # a rejected law. Instead, keep law_used empty so confidence reflects LOW/review_required.
        if not law_used and retrieved_laws:
            logger.warning(
                f"All {len(retrieved_laws)} retrieved laws were rejected for "
                f"defendant '{item.get('name', 'unknown')}'. Marking for review."
            )
            fact_used = retrieved_facts[:3] if retrieved_facts else []

        judge_result = judge_civil_all(
            chatbot,
            law_used,
            fact_used,
            f"Party: {item['name']}, Description: {item['description']}",
        )

        item["judge_result"] = judge_result
        item["retrieved_laws"] = retrieved_laws
        item["retrieved_facts"] = retrieved_facts
        item["original_retrieved_res"] = original_retrieved_res
        item["used_laws"] = law_used
        item["used_facts"] = fact_used
        item["confidence"] = _calculate_confidence(
            law_confidence_scores,
            retrieved_laws,
            law_used,
            retrieved_facts,
            parsing_failures=parsing_failures,
        )

        item["reasoning_trace"] = {
            "retrieved_laws_count": len(retrieved_laws),
            "retrieved_facts_count": len(retrieved_facts),
            "used_laws_count": len(law_used),
            "used_facts_count": len(fact_used),
            "batch_judge": True,
            "use_reranker": retrieve_config.use_reranker,
            "use_self_consistent": retrieve_config.use_self_consistent,
            "self_consistent_n": (
                retrieve_config.self_consistent_n if retrieve_config.use_self_consistent else None
            ),
            "judge_chatbot": retrieve_config.judge_chatbot,
            "fusion_method": original_retrieved_res.get("fusion_method", "union"),
        }

    return case_by_defendant
