from core.preprocess.get_features import get_features
from core.preprocess.case_seg import segment_case_text_withname

from core.graph_construct.feature_graph import (
    query_similar_nodes,
    query_similar_laws,
    query_similar_laws_naive,
    query_similar_nodes_naive,
)

from core.judge.judge_law import judge_law
from core.judge.judge_civil import judge_civil_all

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
        chatbot, concat_feature_descriptions(features), top_k=5
    )

    if not retrieved_facts:
        return None, None

    retrieved_laws = query_similar_laws_naive(concat_feature_descriptions(features), top_k=5)
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
    case_by_defendant = segment_case_text_withname(chatbot, case["fact"][:1024], case["name"])
    for item in case_by_defendant:
        item["feature"] = get_features(chatbot, item)
        original_retrieved_res, retrieved_laws, retrieved_facts = retrieve(
            chatbot, item, law_to_dispute, cases_db, retrieve_config
        )
        if not (retrieved_laws and retrieved_facts):
            continue
        law_used = []
        for law in retrieved_laws:
            used, _ = judge_law(
                chatbot, f"Party: {item['name']}, Description: {item['description']}", law
            )
            if used:
                law_used.append(law)
        fact_used = filter_facts(law_used, retrieved_facts)

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

    return case_by_defendant
