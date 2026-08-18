from core.preprocess.get_features import get_features
from core.preprocess.case_seg import segment_case_text_withname

from core.graph_construct.feature_graph import (
    query_similar_nodes,
    query_similar_laws,
    query_similar_laws_naive,
    query_similar_nodes_naive,
)

from core.judge.judge_law import judge_law, judge_law_batch, judge_law_self_consistent
from core.judge.judge_civil import judge_civil_all
from core.utils.confidence import compute_confidence
from core.prompt import get_prompt


def filter_facts(retrieved_laws, retrieved_facts):
    """
    Filter retrieved_facts, keeping only those whose law field contains
    at least one entry from retrieved_laws.

    Args:
    retrieved_laws: List of law objects, each containing an id field
    retrieved_facts: List of fact objects, each containing a law field (list of law ids)

    Returns:
    Filtered list of facts
    """
    # Extract all ids from retrieved_laws into a set
    law_ids = {str(law["id"]) for law in retrieved_laws}

    # Filter facts: keep only those where at least one law id matches
    filtered_facts = [
        fact for fact in retrieved_facts if any(law_id in law_ids for law_id in fact.get("law", []))
    ]

    return filtered_facts


def concat_feature_descriptions(description):
    res = ""
    res += "Parties Info: " + ", ".join(description.get("parties_info", [])) + ". "
    res += "Dispute Acts: " + ", ".join(description.get("dispute_acts", [])) + ". "
    res += "Subject Matter: " + ", ".join(description.get("subject_matter", [])) + ". "
    res += "Fault and Evidence: " + ", ".join(description.get("fault_and_evidence", [])) + ". "
    return res


def retrieve_law(chatbot, case):
    fact = case["description"][:1024]
    name = case["name"]
    response = chatbot.generate_response(
        get_prompt("RETRIEVE_LAW_PROMPT").format(name=name, fact=fact), max_length=256
    )
    try:
        first = response.find("[")
        last = response.rfind("]") + 1
        disputes = eval(response[first:last])
    except (ValueError, SyntaxError):
        return []
    laws = query_similar_laws(disputes, top_k=1)
    return laws


def retrieve(chatbot, cases, law_to_dispute, cases_db, retrieve_config):
    features = cases["feature"]
    original_retrieved_res, retrieved_facts, retrieved_laws = query_similar_nodes(
        chatbot, concat_feature_descriptions(features), retrieve_config
    )

    if not retrieved_facts:
        return {}, [], []

    augmented_laws = []
    if retrieve_config["augment_retrieve"]:
        augmented_laws = retrieve_law(chatbot, cases)
        original_retrieved_res["augmented"] = augmented_laws
    else:
        augmented_laws = []
    retrieved_laws = retrieved_laws + augmented_laws
    for item in retrieved_facts:
        for case in cases_db:
            if case["id"] == item["caseId"]:
                item["dispute"] = case.get("dispute", [])
                item["law"] = case.get("law", [])
                break
    final_retrieved_laws = []
    seen_law_ids = set()
    for law in retrieved_laws:
        if law["id"] in seen_law_ids:
            continue
        seen_law_ids.add(law["id"])
        law["judge_dep"] = eval(str(law.get("judge_dep", "[]")))
        law["related_laws"] = eval(str(law.get("related_laws", "[]")))
        final_retrieved_laws.append(law)

    return original_retrieved_res, final_retrieved_laws, retrieved_facts


def naive_retrieve(chatbot, cases, law_to_dispute, cases_db):
    features = cases["feature"]
    retrieved_facts = query_similar_nodes_naive(
        chatbot, concat_feature_descriptions(features), top_k=7
    )

    if not retrieved_facts:
        return None, None

    retrieved_laws = query_similar_laws_naive(concat_feature_descriptions(features), top_k=7)
    retrieved_laws = [str(law["entry"]) for law in retrieved_laws]
    for item in retrieved_facts:
        for case in cases_db:
            if case["id"] == item["caseId"]:
                item["dispute"] = case.get("dispute", [])
                item["law"] = case.get("law", [])
                retrieved_laws.extend(case.get("law", []))
                break
    retrieved_laws = list(set(retrieved_laws))
    final_retrieved_laws = []
    for x in retrieved_laws:
        if int(x) < 102:
            continue
        try:
            for item in law_to_dispute:
                if item["id"] == int(x):
                    for entry in item.get("items", [item]):
                        final_retrieved_laws.append(
                            {
                                "id": item["id"],
                                "text": entry.get("text", ""),
                                "dispute": entry.get("dispute", []),
                                "judge_dep": entry.get("judge_dep", []),
                                "related_laws": entry.get("related_laws", []),
                            }
                        )
                    break
        except IndexError:
            continue

    return final_retrieved_laws, retrieved_facts


def locate_law(law, laws):
    for item in laws:
        if law["id"] == item["id"]:
            return item
    return law["text"]


def analyze_case(chatbot, case, law_to_dispute, cases_db, retrieve_config):
    from core.retriever.graph_retriever import GraphRetriever

    # Always use GraphRetriever (which now handles Neo4j vector + fulltext hybrid search natively)
    retriever = GraphRetriever(chatbot)

    # Cap on how many laws to evaluate in judge_law (prevents unbounded API calls)
    max_judge_laws = retrieve_config.get("max_judge_laws", 8)

    # Feature flag: use batch judge (single LLM call per law) or original per-condition loop
    use_batch_judge = retrieve_config.get("batch_judge", True)

    # K1: Cross-encoder reranking (runs between RRF retrieval and judge_law)
    use_reranker = retrieve_config.get("use_reranker", False)
    reranker_model = retrieve_config.get("reranker_model", None)  # None = default model
    reranker_top_k = retrieve_config.get("reranker_top_k", max_judge_laws)

    # K4: Self-consistency sampling
    use_self_consistent = retrieve_config.get("use_self_consistent", False)
    self_consistent_n = retrieve_config.get("self_consistent_n", 5)
    # judge_chatbot_name: if set, a cheap model (e.g. gemini_flash_lite) is used for sampling
    judge_chatbot_name = retrieve_config.get("judge_chatbot", None)

    # Lazily initialize judge chatbot (Gemini) only if self-consistency is on
    judge_chatbot = None
    if use_self_consistent and judge_chatbot_name:
        try:
            judge_chatbot = _get_judge_chatbot(judge_chatbot_name)
        except Exception as e:
            import logging

            logging.getLogger("LegalGraphRAG").warning(
                f"Failed to init judge_chatbot '{judge_chatbot_name}': {e}. "
                "Falling back to primary chatbot for self-consistency."
            )

    criminals = case["name"] if isinstance(case["name"], list) else [case["name"]]
    case_by_defendant = segment_case_text_withname(chatbot, case["fact"][:4096], criminals)
    for item in case_by_defendant:
        item["feature"] = get_features(chatbot, item)
        original_retrieved_res, retrieved_laws, retrieved_facts = retriever.retrieve(
            item, law_to_dispute, cases_db, retrieve_config
        )
        if not (retrieved_laws and retrieved_facts):
            continue

        # K1: Cross-encoder reranking (refine ordering before judge)
        if use_reranker and retrieved_laws:
            try:
                from core.retriever.reranker import get_reranker

                query_text = f"{item.get('name', '')} {item.get('description', '')}"
                reranker = get_reranker(
                    model_name=reranker_model or "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    top_k=reranker_top_k,
                )
                retrieved_laws = reranker.rerank(query_text, retrieved_laws)
                original_retrieved_res["reranker"] = reranker.model_name
            except Exception as e:
                import logging

                logging.getLogger("LegalGraphRAG").warning(
                    f"Reranker failed, using original order: {e}"
                )

        # Limit laws sent to judge_law to control API costs
        laws_to_judge = retrieved_laws[:max_judge_laws]

        # M5: Easy-first ordering – sort by number of judge_dep conditions (fewest first).
        # Laws with fewer conditions are faster to evaluate → early exit saves more calls.
        laws_to_judge = sorted(
            laws_to_judge,
            key=lambda law_item: (
                len(law_item.get("judge_dep", []))
                if isinstance(law_item.get("judge_dep", []), list)
                else 0
            ),
        )

        # M5: Early exit – stop judging once enough applicable laws are found
        max_applicable = retrieve_config.get("max_applicable_laws", max_judge_laws)

        law_used = []
        law_confidence_scores: list[float] = []  # from self-consistency

        # M4: Description length cap – long descriptions waste tokens without adding signal
        desc_cap = retrieve_config.get("law_desc_cap", 600)  # chars, not tokens

        for law in laws_to_judge:
            # M5: Early exit
            if len(law_used) >= max_applicable:
                break

            # M4: Truncate description to cap (copy to avoid mutating cached dict)
            law = dict(law)
            if law.get("description") and len(law["description"]) > desc_cap:
                law["description"] = law["description"][:desc_cap]

            if use_self_consistent:
                # K4: Self-consistency – use cheap Gemini sampler
                used, sc_confidence, _ = judge_law_self_consistent(
                    chatbot,
                    f"Party: {item['name']}, Description: {item['description']}",
                    law,
                    n_samples=self_consistent_n,
                    judge_chatbot=judge_chatbot,
                )
                if used:
                    law_used.append(law)
                    law_confidence_scores.append(sc_confidence)
            else:
                judge_fn = judge_law_batch if use_batch_judge else judge_law
                used, _ = judge_fn(
                    chatbot, f"Party: {item['name']}, Description: {item['description']}", law
                )
                if used:
                    law_used.append(law)

        fact_used = filter_facts(law_used, retrieved_facts)

        # Fallback: If the strict LLM judge rejected all laws, keep at least the top 1 law
        # so we have some context for the final resolution (prevents Ragas Faithfulness = 0)
        if not law_used and laws_to_judge:
            law_used = laws_to_judge[:1]
            fact_used = filter_facts(law_used, retrieved_facts)
            if not fact_used and retrieved_facts:
                fact_used = retrieved_facts[:1]

        judge_result = judge_civil_all(
            chatbot,
            law_used,
            fact_used,
            f"Party: {item['name']}, Description: {item['description']}",
        )

        # Existing output fields (unchanged for backward compatibility)
        item["judge_result"] = judge_result
        item["retrieved_laws"] = retrieved_laws
        item["retrieved_facts"] = retrieved_facts
        item["original_retrieved_res"] = original_retrieved_res
        item["used_laws"] = law_used
        item["used_facts"] = fact_used

        # Confidence: prefer self-consistency scores if available, else heuristic
        if law_confidence_scores:
            sc_mean = sum(law_confidence_scores) / len(law_confidence_scores)
            item["confidence"] = {
                "retrieval_quality": round(min(len(retrieved_laws) / 5.0, 1.0), 3),
                "law_applicability": round(sc_mean, 3),
                "overall": round(sc_mean, 3),
                "grade": "HIGH" if sc_mean > 0.75 else "MEDIUM" if sc_mean > 0.45 else "LOW",
                "review_required": sc_mean < 0.45,
                "source": "self_consistency",
            }
        else:
            item["confidence"] = compute_confidence(
                retrieved_laws=retrieved_laws,
                used_laws=law_used,
                retrieved_facts=retrieved_facts,
            ).to_dict()

        # Reasoning trace for debugging and audit
        item["reasoning_trace"] = {
            "retrieved_laws_count": len(retrieved_laws),
            "retrieved_facts_count": len(retrieved_facts),
            "used_laws_count": len(law_used),
            "used_facts_count": len(fact_used),
            "batch_judge": use_batch_judge,
            "use_reranker": use_reranker,
            "use_self_consistent": use_self_consistent,
            "self_consistent_n": self_consistent_n if use_self_consistent else None,
            "judge_chatbot": judge_chatbot_name,
            "fusion_method": original_retrieved_res.get("fusion_method", "union"),
        }

    return case_by_defendant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_judge_chatbot_cache: dict = {}


def _get_judge_chatbot(model_name: str):
    """Return a cached cheap judge chatbot instance.

    Supports:
      - "gemini_flash_lite" → GeminiChatbot(gemini-2.0-flash-lite)
      - "gemini_flash"      → GeminiChatbot(gemini-2.0-flash)

    Args:
        model_name: Logical model name as used in ModelConfig.

    Returns:
        BaseModel instance ready to call generate_response().
    """
    if model_name in _judge_chatbot_cache:
        return _judge_chatbot_cache[model_name]

    from core.models.openai.gemini import GeminiChatbot

    gemini_ids = {
        "gemini_flash_lite": "gemini-3.5-flash-lite",
        "gemini_flash": "gemini-3.5-flash",
    }
    if model_name not in gemini_ids:
        raise ValueError(
            f"Unknown judge_chatbot '{model_name}'. " f"Supported: {list(gemini_ids.keys())}"
        )

    bot = GeminiChatbot(model_name=gemini_ids[model_name])
    _judge_chatbot_cache[model_name] = bot
    return bot
