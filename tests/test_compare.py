import sys, os, json, random
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()
from core.LegalGraphRAG import LegalGraphRAG, LegalGraphRAGConfig

config = LegalGraphRAGConfig.from_env_file(".env")
config.model.model_name = "gpt4o_mini"
config.model.device = "cpu"
rag = LegalGraphRAG(config)
rag.cases_db = rag._load_cases_db()
valid_cases = [c for c in rag.cases_db if c.get("law")]
case = valid_cases[0]  # Lấy case đầu tiên
case_input = {"fact": case["fact"], "name": ["Nguyên đơn", "Bị đơn"]}
rag.config.retrieve.to_dict = lambda: {"method": "vector", "direct_retrieve_top_k": 3}
vector_res = rag.analyze_case(case_input)

vec_retrieved_laws = []
for r in vector_res:
    laws = r.get("retrieved_laws", [])
    for law in laws:
        law_id = law.get("id") if isinstance(law, dict) else str(law)
        vec_retrieved_laws.append(law_id)

print("Ground Truth Laws:", case["law"])
print("Retrieved Laws:", vec_retrieved_laws)
