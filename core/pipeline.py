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
    """Segment case by defendant and extract features."""
    criminals = case["name"] if isinstance(case["name"], list) else [case["name"]]
    case_by_defendant = segment_case_text_withname(chatbot, case["fact"][:4096], criminals)

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
    """Retrieve laws and facts from graph, and optionally rerank them."""
    # We pass the RetrieveConfig dict for backward compatibility with GraphRetriever internals
    # But wait, GraphRetriever might expect a dict! Let's pass the dict to it.
    original_retrieved_res, retrieved_laws, retrieved_facts = retriever.retrieve(
        item, law_to_dispute, cases_db, retrieve_config.to_dict()
    )

    if not (retrieved_laws and retrieved_facts):
        return original_retrieved_res, retrieved_laws, retrieved_facts

    if retrieve_config.use_reranker and retrieved_laws:
        try:
            from core.retriever.reranker import get_reranker

            query_text = f"{item.get('name', '')} {item.get('description', '')}"
            reranker = get_reranker(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                top_k=retrieve_config.reranker_top_k,
            )
            retrieved_laws = reranker.rerank(query_text, retrieved_laws)
            original_retrieved_res["reranker"] = reranker.model_name
        except Exception as e:
            logger.warning(f"Reranker failed, using original order: {e}")

    return original_retrieved_res, retrieved_laws, retrieved_facts


def _evaluate_laws(
    chatbot: BaseModel,
    item: Dict[str, Any],
    retrieved_laws: List[Dict[str, Any]],
    retrieve_config: RetrieveConfig,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """Evaluate laws using LLM judge and return the used laws and their confidence scores."""
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

    for law in laws_to_judge:
        if len(law_used) >= retrieve_config.max_applicable_laws:
            break

        law = dict(law)
        if law.get("description") and len(law["description"]) > retrieve_config.law_desc_cap:
            law["description"] = law["description"][: retrieve_config.law_desc_cap]

        if retrieve_config.use_self_consistent:
            used, sc_confidence, _ = judge_law_self_consistent(
                chatbot,
                f"Party: {item['name']}, Description: {item['description']}",
                law,
                n_samples=retrieve_config.self_consistent_n,
                judge_chatbot=judge_chatbot,
            )
            if used:
                law_used.append(law)
                law_confidence_scores.append(sc_confidence)
        else:
            judge_fn = judge_law_batch if retrieve_config.batch_judge else judge_law
            used, _ = judge_fn(
                chatbot,
                f"Party: {item['name']}, Description: {item['description']}",
                law,
            )
            if used:
                law_used.append(law)

    return law_used, law_confidence_scores


def _calculate_confidence(
    law_confidence_scores: List[float],
    retrieved_laws: List[Dict[str, Any]],
    law_used: List[Dict[str, Any]],
    retrieved_facts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Calculate the confidence score of the retrieval and judgement."""
    if law_confidence_scores:
        sc_mean = sum(law_confidence_scores) / len(law_confidence_scores)
        return {
            "retrieval_quality": round(min(len(retrieved_laws) / 5.0, 1.0), 3),
            "law_applicability": round(sc_mean, 3),
            "overall": round(sc_mean, 3),
            "grade": "HIGH" if sc_mean > 0.75 else "MEDIUM" if sc_mean > 0.45 else "LOW",
            "review_required": sc_mean < 0.45,
            "source": "self_consistency",
        }
    return compute_confidence(
        retrieved_laws=retrieved_laws,
        used_laws=law_used,
        retrieved_facts=retrieved_facts,
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

        law_used, law_confidence_scores = _evaluate_laws(
            chatbot, item, retrieved_laws, retrieve_config
        )

        fact_used = filter_facts(law_used, retrieved_facts)

        # Fallback: keep top 1 if all rejected
        if not law_used and retrieved_laws:
            law_used = retrieved_laws[:1]
            fact_used = filter_facts(law_used, retrieved_facts)
            if not fact_used and retrieved_facts:
                fact_used = retrieved_facts[:1]

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
            law_confidence_scores, retrieved_laws, law_used, retrieved_facts
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
