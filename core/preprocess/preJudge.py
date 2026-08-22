import ast
import json
from core.prompt import get_prompt


def pre_judge(model, case) -> list:
    prompt_with_case = get_prompt("PRE_JUDGE_PROMPT").format(case_text=case)
    response = model.generate_response(prompt_with_case)
    try:
        # Attempt to parse response as a Python list
        first_bracket = response.find("[")
        last_bracket = response.rfind("]")
        if first_bracket != -1 and last_bracket > first_bracket:
            list_str = response[first_bracket : last_bracket + 1]
            try:
                candidates = json.loads(list_str)
            except Exception:
                candidates = ast.literal_eval(list_str)
        else:
            candidates = []

        if isinstance(candidates, list) and all(isinstance(item, str) for item in candidates):
            return candidates[:3]
        else:
            return []
    except Exception:
        # Return empty list if parsing fails
        return []
