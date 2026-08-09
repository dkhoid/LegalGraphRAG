import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.LegalGraphRAG import LegalGraphRAG, LegalGraphRAGConfig
import random
from dotenv import load_dotenv

load_dotenv()

config = LegalGraphRAGConfig.from_env_file(".env")
rag = LegalGraphRAG(config)

cases_db = rag._load_cases_db()
valid_cases = [c for c in cases_db if c.get("law")]
test_case = valid_cases[0]

case_input = {"fact": test_case["fact"], "name": ["Nguyên đơn", "Bị đơn"]}
results = rag.analyze_case(case_input)

for r in results:
    print(f"--- Party: {r['name']} ---")
    print("used_laws count:", len(r.get("used_laws", [])))
    for law in r.get("used_laws", []):
        print("- Law:", law.get("entry", "unknown"))

    print("judge_result:")
    print(r.get("judge_result", ""))
    print("---------------------------------")
