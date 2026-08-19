import json
from core.prompt import get_prompt


def format_law(law_used):
    res = ""
    for law in law_used:
        law["disputes"] = law.get("disputes", [])
        issues = [c.replace("\n", " ") for c in law["disputes"] if c]
        res += f"Article {law['entry']} — Applicable issues: {', '.join(issues)}. Content: {law['description']}\n---\n"

    return res


def format_fact(facts):
    res = ""
    for fact in facts:
        issues = fact.get("dispute", [])
        res += f"Legal issue: {', '.join(issues)}. Fact description: {fact['description']}.\n"
    return res


def judge_civil(chatbot, law_used, retrieved_facts, case_description):
    response = chatbot.generate_response(
        get_prompt("JUDGE_CIVIL_PROMPT").format(law=format_law(law_used), case=case_description),
        max_length=4096,
    )
    try:
        first = response.rfind("[")
        last = response.rfind("]") + 1
        response = response.replace("，", ",")
        response = list(set(eval(response[first:last])))
    except Exception as e:
        print(f"Error parsing response: {e}")
        response = ["No applicable issue identified"]
    response = [str(x).strip() for x in response if str(x).strip()]
    return response


def judge_civil_all(chatbot, law_used, retrieved_facts, case_description):
    response = chatbot.generate_response(
        get_prompt("JUDGE_CIVIL_ALL_PROMPT")
        + get_prompt("JUDGE_CIVIL_ALL_INPUT_TEMPLATE").format(
            law=format_law(law_used),
            facts=format_fact(retrieved_facts),
            case=case_description,
        ),
        max_length=4096,
    )
    try:
        first = response.find("{")
        last = response.rfind("}") + 1
        response = response[first:last]
        response = json.loads(response)
    except Exception as e:
        print(f"Error parsing response: {e}")
        response = {
            "dispute_type": ["No applicable issue identified"],
            "law_article": ["N/A"],
            "resolution": {"liability": "N/A", "compensation": "N/A"},
        }
    return response
