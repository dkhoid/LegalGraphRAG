import json
import re
from core.prompt import get_prompt


def normalize_law_article(raw: str) -> str:
    """Normalise a law article string returned by the LLM.

    Strips common noise patterns so the output matches evaluation ground-truth:
    - Garbage prefixes: ``zalo_``, ``Article `` (English), leading whitespace
    - Legislation codes embedded in the string: ``/2019/QH14``, ``/2015/ND-CP`` …
    - Plus-sign suffixes used in graph IDs: ``45/2019/qh14+132`` → ``Điều 132``
    - Condenses to the canonical ``Điều N`` form when possible.

    Examples::

        "zalo_45/2019/qh14+132" → "Điều 132"
        "Article zalo_45/2019/qh14+11" → "Điều 11"
        "Điều 28/2016/tt-blđtbxh+11" → "Điều 11"
        "Điều 36" → "Điều 36"
        "36" → "Điều 36"
    """
    if not isinstance(raw, str):
        return str(raw)

    s = raw.strip()
    # Remove known garbage prefixes (case-insensitive)
    s = re.sub(r"(?i)^(zalo_|article\s*)", "", s).strip()
    # Remove Vietnamese "Điều" / "dieu" prefix before processing
    s_stripped = re.sub(r"(?i)^(\u0111i\u1ec1u|dieu)\s*", "", s).strip()

    # If there's a '+' take only the last segment (graph ID format)
    if "+" in s_stripped:
        s_stripped = s_stripped.split("+")[-1].strip()

    # Strip legislation-code suffix like /2019/QH14, /2015/ND-CP
    s_stripped = re.sub(r"/\d{4}/[a-z0-9\-]+", "", s_stripped, flags=re.IGNORECASE).strip()

    # Extract the first number that is NOT a year (< 2000)
    nums = [n for n in re.findall(r"\b(\d+)\b", s_stripped) if int(n) < 2000]
    if nums:
        return f"Điều {nums[0]}"

    # Fallback: return the cleaned string as-is
    return s.strip()


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
        result = json.loads(response)

        # Normalize law_article entries to remove noise from LLM output
        raw_laws = result.get("law_article", [])
        if isinstance(raw_laws, list):
            result["law_article"] = [normalize_law_article(l) for l in raw_laws]
        elif isinstance(raw_laws, str):
            result["law_article"] = [normalize_law_article(raw_laws)]

        return result
    except Exception as e:
        print(f"Error parsing response: {e}")
        return {
            "dispute_type": ["No applicable issue identified"],
            "law_article": ["N/A"],
            "resolution": {"liability": "N/A", "compensation": "N/A"},
        }
