from core.graph_construct.graph_search import (
    query_similar_nodes,
    query_similar_laws,
    query_similar_laws_naive,
    query_similar_nodes_naive,
)

from core.utils.formatting import concat_feature_descriptions
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
    # Extract all ids and entries from retrieved_laws into a set
    law_ids = set()
    for law in retrieved_laws:
        if "id" in law:
            law_ids.add(str(law["id"]))
        if "entry" in law and law["entry"]:
            law_ids.add(str(law["entry"]))

    # Filter facts: keep only those where at least one law id matches
    filtered_facts = [
        fact for fact in retrieved_facts if any(law_id in law_ids for law_id in fact.get("law", []))
    ]

    return filtered_facts


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
    from core.pipeline import analyze_case_pipeline
    from core.config import RetrieveConfig

    if isinstance(retrieve_config, dict):
        valid_keys = RetrieveConfig.__dataclass_fields__.keys()
        filtered = {k: v for k, v in retrieve_config.items() if k in valid_keys}
        retrieve_config = RetrieveConfig(**filtered)

    return analyze_case_pipeline(chatbot, case, law_to_dispute, cases_db, retrieve_config)


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

    # Support gemini and openai models
    if "gemini" in model_name:
        from core.models.openai.gemini import GeminiChatbot

        gemini_ids = {
            "gemini_flash_lite": "gemini-3.5-flash-lite",
            "gemini_flash": "gemini-3.5-flash",
        }
        actual_model_name = gemini_ids.get(model_name, model_name)
        bot = GeminiChatbot(model_name=actual_model_name)
    else:
        # Fallback to GPT4OMiniChatbot for gpt4o_mini, etc.
        from core.models.openai.gpt4o_mini import GPT4OMiniChatbot

        actual_model_name = "gpt-4o-mini" if "mini" in model_name else model_name
        bot = GPT4OMiniChatbot(model_name=actual_model_name)

    _judge_chatbot_cache[model_name] = bot
    return bot
