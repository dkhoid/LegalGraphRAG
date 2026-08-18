import json
import re
from core.prompt import get_prompt


def judge_law(chatbot, case_description, law):
    if isinstance(law, str):
        judge_res = chatbot.generate_response(
            get_prompt("JUDGE_LAW_PROMPT1").format(law=law, case=case_description), max_length=128
        )
        decision = judge_res.strip().split("\n")[-1].lower()
        if "true" in decision:
            return True, ""
        else:
            return False, ""
    true_list = []
    false_list = []
    import ast

    judge_deps = law.get("judge_dep", [])
    if isinstance(judge_deps, str):
        try:
            judge_deps = ast.literal_eval(judge_deps)
        except (ValueError, SyntaxError):
            judge_deps = []

    for judge in judge_deps:
        law_desc = law.get("description", law.get("text", ""))
        judge_res = chatbot.generate_response(
            get_prompt("JUDGE_LAW_PROMPT").format(
                law_item=law_desc.replace("\n", ""),
                related=law.get("related_laws", []),
                element=judge,
                case=case_description,
            ),
            max_length=128,
        )
        decision = judge_res.strip().split("\n")[-1].lower()
        if "true" in decision:
            true_list.append(judge)
        elif "false" in decision:
            false_list.append(judge)

    law_desc = law.get("description", law.get("text", ""))
    res = chatbot.generate_response(
        get_prompt("JUDGE_LAW_PROMPT0").format(
            case=case_description,
            law=law_desc,
            true_list=true_list,
            false_list=false_list,
        ),
        max_length=1024,
    )
    decision = res.strip().split("\n")[-1].lower()
    if "true" in decision:
        return True, res
    return False, res


def judge_law_batch(chatbot, case_description: str, law: dict) -> tuple:
    """Evaluate all judge_dep conditions of a law in a single structured LLM call.

    This is the batch-optimised replacement for judge_law(). Instead of making
    N+1 sequential calls (one per condition + one aggregation), it sends all
    conditions in one prompt and expects a JSON response.

    Falls back to the original judge_law() when:
    - law has no judge_dep conditions (uses JUDGE_LAW_PROMPT1 path)
    - the model returns un-parseable JSON

    Args:
        chatbot: Any model instance with a generate_response() method.
        case_description: String description of the case/defendant.
        law: Law dict with at least "description"/"text" and optionally "judge_dep".

    Returns:
        Tuple (applicable: bool, reasoning: str) – same contract as judge_law().
    """
    import ast

    judge_deps = law.get("judge_dep", [])
    if isinstance(judge_deps, str):
        try:
            judge_deps = ast.literal_eval(judge_deps)
        except (ValueError, SyntaxError):
            judge_deps = []

    # No structured conditions → use the simpler single-call path
    if not judge_deps:
        return judge_law(chatbot, case_description, law)

    law_desc = law.get("description", law.get("text", ""))
    conditions_numbered = "\n".join(f"[{i + 1}] {cond}" for i, cond in enumerate(judge_deps))

    prompt = get_prompt("JUDGE_LAW_BATCH_PROMPT").format(
        law_desc=law_desc.replace("\n", " "),
        conditions_numbered=conditions_numbered,
        case=case_description,
    )
    response = chatbot.generate_response(prompt, max_length=512)

    try:
        # Extract first JSON object from response (model may add preamble text)
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in response")
        parsed = json.loads(match.group())
        applicable = bool(parsed.get("applicable", False))
        reasoning = str(parsed.get("conditions", []))
        return applicable, reasoning
    except Exception:
        # Safe fallback: never crash, just use the original method
        return judge_law(chatbot, case_description, law)


def judge_law_self_consistent(
    chatbot,
    case_description: str,
    law: dict,
    n_samples: int = 5,
    judge_chatbot=None,
) -> tuple[bool, float, str]:
    """Self-consistency judge: sample N decisions and take majority vote.

    Instead of calling the judge once and trusting the output, we call it N
    times with temperature > 0 to get diverse reasoning paths, then aggregate.
    Paths that consistently agree are more likely to be correct.

    This technique is from "Self-Consistency Improves Chain of Thought Reasoning"
    (Wang et al., 2023, https://arxiv.org/abs/2203.11171).

    Args:
        chatbot: Primary model (used as fallback if judge_chatbot is None).
        case_description: Case text for the current defendant.
        law: Law dict with "description"/"text" and optional "judge_dep".
        n_samples: Number of independent samples to take. More = more reliable
                   but higher latency/cost. Recommended: 3 (fast) to 7 (accurate).
        judge_chatbot: Optional cheap model (e.g., Gemini Flash Lite) for sampling.
                       If None, the primary chatbot is used.

    Returns:
        Tuple of (decision: bool, confidence: float, reasoning: str) where
        - decision    = majority vote result (True if > 50% samples say applicable)
        - confidence  = fraction of samples agreeing (e.g., 0.8 = 4/5 agree)
        - reasoning   = summary of agreement ratio
    """
    sampler = judge_chatbot if judge_chatbot is not None else chatbot
    votes: list[bool] = []

    for i in range(n_samples):
        try:
            # Use batch judge for structured output; vary temperature implicitly
            # by asking the sampler model (Gemini uses default temp 0.7)
            result, _ = judge_law_batch(sampler, case_description, law)
            votes.append(result)
        except Exception:
            # On any error, skip this sample rather than crashing
            continue

    if not votes:
        # All samples failed – fall back to primary chatbot single call
        result, reasoning = judge_law_batch(chatbot, case_description, law)
        return result, 0.5, reasoning

    true_count = sum(votes)
    total = len(votes)
    confidence = true_count / total
    decision = confidence > 0.5

    reasoning = (
        f"Self-consistency: {true_count}/{total} samples say applicable "
        f"(confidence={confidence:.2f})"
    )
    return decision, confidence, reasoning
