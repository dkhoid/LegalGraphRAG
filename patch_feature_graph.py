import re

with open("core/graph_construct/feature_graph.py", "r") as f:
    content = f.read()

# Add BM25 import at the top
if "from rank_bm25 import BM25Okapi" not in content:
    content = content.replace(
        "from core.utils.logger import logger",
        "from core.utils.logger import logger\nfrom rank_bm25 import BM25Okapi\nimport json",
    )

# We'll inject BM25 logic right before aggregating results in query_similar_nodes
# Find this exact spot:
target_code = """
    # Aggregate results
    result_cases = []
"""

bm25_injection = """
    # Hybrid BM25 logic to find missing laws directly
    bm25_laws = []
    try:
        with open('./data/processed/law_to_dispute.json', 'r', encoding='utf-8') as f:
            law_to_dispute = json.load(f)

        corpus = []
        law_mapping = []
        for law in law_to_dispute:
            for item in law.get("items", [law]):
                text = item.get("text", "")
                if text:
                    corpus.append(text)
                    law_mapping.append({
                        "id": law["id"],
                        "entry": str(law["id"]),
                        "text": text,
                        "description": text,
                        "dispute": item.get("dispute", []),
                        "judge_dep": item.get("judge_dep", []),
                        "related_laws": item.get("related_laws", []),
                    })
        if corpus:
            tokenized_corpus = [doc.lower().split() for doc in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = query_text.lower().split()
            bm25_scores = bm25.get_scores(tokenized_query)
            top_k_bm25 = 3
            top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k_bm25]
            bm25_laws = [law_mapping[i] for i in top_indices if bm25_scores[i] > 0]
    except Exception as e:
        print(f"BM25 Graph Error: {e}")

    # Aggregate results
    result_cases = []
"""

if "Hybrid BM25 logic" not in content:
    content = content.replace(target_code, bm25_injection)

# Now we need to append bm25_laws to the final result_laws
target_code_2 = """
    original_retrieved_res = {
"""

append_logic = """
    # Append BM25 laws to the results
    for law in bm25_laws:
        if law["id"] not in seen_ids_laws:
            result_laws.append(law)
            seen_ids_laws.add(law["id"])
            direct_result_laws.append(law)

    original_retrieved_res = {
"""

if "Append BM25 laws to the results" not in content:
    content = content.replace(target_code_2, append_logic)

with open("core/graph_construct/feature_graph.py", "w") as f:
    f.write(content)

print("feature_graph.py patched successfully!")
