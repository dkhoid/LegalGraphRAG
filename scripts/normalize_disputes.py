import json
import os
import shutil
import re
from dotenv import load_dotenv

load_dotenv()


def main():
    cases_path = "data/processed/cases_with_feature.json"
    laws_path = "data/processed/law_to_dispute.json"
    backup_path = cases_path + ".bak"

    if not os.path.exists(backup_path):
        print(f"Creating backup at {backup_path}")
        shutil.copy2(cases_path, backup_path)

    print("Loading standard disputes...")
    with open(laws_path, "r", encoding="utf-8") as f:
        laws = json.load(f)

    standard_disputes = set()
    for law in laws:
        for item in law.get("items", []):
            d = item.get("dispute")
            if isinstance(d, list):
                standard_disputes.update(d)
            elif d:
                standard_disputes.add(d)
    standard_disputes = sorted(list(standard_disputes))
    print(f"Found {len(standard_disputes)} standard disputes.")

    print("Loading cases...")
    with open(cases_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    case_disputes = set()
    for c in cases:
        d = c.get("dispute", [])
        if isinstance(d, list):
            case_disputes.update(d)
        elif d:
            case_disputes.add(d)
    case_disputes = sorted(list(case_disputes))
    print(f"Found {len(case_disputes)} unique case disputes.")

    print("Generating mapping via LLM...")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("base_url", "https://api.openai.com/v1")
    if base_url.endswith("/"):
        base_url = base_url[:-1]

    prompt = f"""You are a legal expert. I have {len(case_disputes)} raw dispute strings from cases, and {len(standard_disputes)} standard dispute categories.

STANDARD CATEGORIES:
{json.dumps(standard_disputes, ensure_ascii=False, indent=2)}

RAW STRINGS:
{json.dumps(case_disputes, ensure_ascii=False, indent=2)}

Please map EVERY raw string to the MOST APPROPRIATE standard category.
Output ONLY a raw JSON object where keys are the raw strings and values are the standard categories. Do not include markdown code blocks.
"""
    try:
        import requests

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        response = requests.post(
            f"{base_url}/chat/completions", headers=headers, json=data, timeout=60
        )
        response.raise_for_status()
        response_text = response.json()["choices"][0]["message"]["content"]

        # Clean up markdown if llm still outputs it
        response_text = re.sub(r"^```json\n", "", response_text)
        response_text = re.sub(r"\n```$", "", response_text).strip()

        mapping = json.loads(response_text)
    except Exception as e:
        print(f"Failed to generate mapping: {e}")
        return

    print("Mapping generated successfully.")
    for raw, std in list(mapping.items())[:10]:
        print(f"  {raw} -> {std}")

    print("\nApplying mapping to cases...")
    updated_cases = 0
    for c in cases:
        old_disputes = c.get("dispute", [])
        if isinstance(old_disputes, str):
            old_disputes = [old_disputes]

        new_disputes = set()
        for d in old_disputes:
            if d in mapping and mapping[d] in standard_disputes:
                new_disputes.add(mapping[d])
            else:
                new_disputes.add(d)  # fallback
        c["dispute"] = list(new_disputes)
        updated_cases += 1

    print("Saving normalized cases...")
    with open(cases_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=4)

    print("Done! Taxonomy normalization complete.")


if __name__ == "__main__":
    main()
