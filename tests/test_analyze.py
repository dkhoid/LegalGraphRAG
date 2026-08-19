import sys
import os
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.LegalGraphRAG import LegalGraphRAG, LegalGraphRAGConfig

config = LegalGraphRAGConfig.from_env_file(".env")
rag = LegalGraphRAG(config)
case_input = {"fact": "Anh Minh bị sa thải", "name": ["Nguyên đơn", "Bị đơn"]}
res = rag.analyze_case(case_input)
print(json.dumps(res, ensure_ascii=False, indent=2))
