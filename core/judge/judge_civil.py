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


def format_law_entry(entry_id: str) -> str:
    """Format law entry ID into human-readable legal citation."""
    if not entry_id:
        return "Điều luật"
    s = str(entry_id).replace("zalo_", "").strip()
    doc_mapping = {
        "91/2015/qh13": "Bộ luật Dân sự 2015",
        "45/2019/qh14": "Bộ luật Lao động 2019",
        "52/2014/qh13": "Luật Hôn nhân và Gia đình 2014",
        "92/2015/qh13": "Bộ luật Tố tụng Dân sự 2015",
        "45/2013/qh13": "Luật Đất đai 2013",
        "100/2015/qh13": "Bộ luật Hình sự 2015",
    }
    if "+" in s:
        doc_code, art_num = s.split("+", 1)
        doc_name = doc_mapping.get(doc_code.lower(), doc_code.upper())
        return f"Điều luật {art_num} ({doc_name})"
    return f"Điều luật {s}"


def format_law(law_used):
    res = ""
    for law in law_used:
        law_disputes = law.get("disputes", [])
        if isinstance(law_disputes, list):
            issues = [str(c).replace("\n", " ") for c in law_disputes if c]
        else:
            issues = [str(law_disputes)]
        raw_entry = law.get("entry") or law.get("id", "")
        entry_title = format_law_entry(raw_entry)
        desc = law.get("description", law.get("text", ""))
        res += f"{entry_title} — Vấn đề pháp lý liên quan: {', '.join(issues)}. Nội dung: {desc}\n---\n"

    return res


def format_fact(facts):
    res = ""
    for fact in facts:
        issues = fact.get("dispute", [])
        res += f"Vấn đề pháp lý: {', '.join(issues)}. Tình tiết vụ án: {fact['description']}.\n"
    return res


def judge_civil(chatbot, law_used, retrieved_facts, case_description):
    import ast
    from core.utils.logger import logger

    response = chatbot.generate_response(
        get_prompt("JUDGE_CIVIL_PROMPT").format(law=format_law(law_used), case=case_description),
        max_length=4096,
    )
    try:
        # Strip DeepSeek reasoning tags if present
        cleaned = re.sub(r"(?is)<think>.*?</think>", "", response).strip()
        first = cleaned.find("[")
        last = cleaned.rfind("]") + 1
        if first != -1 and last > first:
            json_str = cleaned[first:last].replace("，", ",")
            try:
                parsed = json.loads(json_str)
            except Exception:
                parsed = ast.literal_eval(json_str)
            response_list = (
                list(set(parsed))
                if isinstance(parsed, (list, set, tuple))
                else ["No applicable issue identified"]
            )
        else:
            response_list = ["No applicable issue identified"]
    except Exception as e:
        logger.warning(f"Error parsing judge_civil response: {e}")
        response_list = ["No applicable issue identified"]
    return [str(x).strip() for x in response_list if str(x).strip()]


def judge_civil_all(chatbot, law_used, retrieved_facts, case_description):
    from core.utils.logger import logger

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
        # Strip DeepSeek reasoning tags if present
        cleaned = re.sub(r"(?is)<think>.*?</think>", "", response).strip()
        first = cleaned.find("{")
        last = cleaned.rfind("}") + 1
        if first != -1 and last > first:
            json_str = cleaned[first:last]
            result = json.loads(json_str)
        else:
            raise ValueError("No JSON object found in LLM response")

        # Normalize law_article entries to remove noise from LLM output
        raw_laws = result.get("law_article", [])
        if isinstance(raw_laws, list):
            result["law_article"] = [normalize_law_article(law_item) for law_item in raw_laws]
        elif isinstance(raw_laws, str):
            result["law_article"] = [normalize_law_article(raw_laws)]

        return result
    except Exception as e:
        logger.warning(f"Error parsing judge_civil_all response: {e}")
        return {
            "dispute_type": ["No applicable issue identified"],
            "law_article": ["N/A"],
            "resolution": {"liability": "N/A", "compensation": "N/A"},
        }
