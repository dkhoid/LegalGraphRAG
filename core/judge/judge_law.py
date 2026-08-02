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
        judge_res = chatbot.generate_response(
            get_prompt("JUDGE_LAW_PROMPT").format(
                law_item=law["description"].replace("\n", ""),
                related=law["related_laws"],
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

    res = chatbot.generate_response(
        get_prompt("JUDGE_LAW_PROMPT0").format(
            case=case_description,
            law=law["description"],
            true_list=true_list,
            false_list=false_list,
        ),
        max_length=1024,
    )
    decision = res.strip().split("\n")[-1].lower()
    if "true" in decision:
        return True, res
    return False, res
