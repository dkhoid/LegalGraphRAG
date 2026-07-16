import json
from core.prompt import get_prompt


def get_features(model, cases: dict) -> dict:
    prompt = get_prompt("GET_FEATURES_PROMPT") + get_prompt("GET_FEATURES_INPUT_TEMPLATE").format(
        name=cases["name"], fact=cases["description"]
    )

    response = model.generate_response(prompt)

    start_idx = response.find("{")
    end_idx = response.rfind("}") + 1

    if start_idx == -1 or end_idx == -1:
        print("No valid JSON string found in model response")
        return {}

    try:
        return json.loads(response[start_idx:end_idx])
    except json.JSONDecodeError:
        print("JSON parse error: could not decode model response")
        return {}
