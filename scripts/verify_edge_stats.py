import sys

sys.path.append(".")
import numpy as np
from core.graph_construct.graph_db import GraphDBManager

db = GraphDBManager.get_db()
db.load("data/processed/graph.pkl")
G = db.graph


def cos_sim(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    v1, v2 = np.array(v1), np.array(v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / norm) if norm > 0 else 0.0


case_law_sims = []
case_law_dispute_match = 0
case_law_total = 0
unique_cases_with_law = set()

law_dispute_sims = []
case_case_sims = []

for u, v, d in G.edges(data=True):
    rel = d.get("relation_type")

    # We need embeddings
    emb_u = db.nodes_data.get(u, {}).get("data", {}).get("embedding")
    emb_v = db.nodes_data.get(v, {}).get("data", {}).get("embedding")

    if rel == "RELATES_TO_LAW":
        # Ensure u is Case, v is Law
        type_u = db.nodes_data.get(u, {}).get("type")
        if type_u != "Cases":
            u, v = v, u
            emb_u, emb_v = emb_v, emb_u

        case_data = db.nodes_data.get(u, {}).get("data", {})
        law_data = db.nodes_data.get(v, {}).get("data", {})

        sim = cos_sim(emb_u, emb_v)
        if sim > 0:
            case_law_sims.append(sim)

        # Check dispute alignment
        case_disputes = case_data.get("dispute", [])
        if isinstance(case_disputes, str):
            case_disputes = [case_disputes]
        law_disputes = law_data.get("disputes", [])
        if isinstance(law_disputes, str):
            law_disputes = [law_disputes]

        if set(case_disputes).intersection(set(law_disputes)):
            case_law_dispute_match += 1
        case_law_total += 1
        unique_cases_with_law.add(u)

    elif rel == "SIMILAR_TO":
        sim = cos_sim(emb_u, emb_v)
        if sim > 0:
            case_case_sims.append(sim)

    elif rel == "RELATED_DISPUTE":
        sim = cos_sim(emb_u, emb_v)
        if sim > 0:
            law_dispute_sims.append(sim)

print("=== NEW RELATION RELEVANCE STATISTICS ===")
total_cases = len(db.get_nodes_by_type("Cases"))
print(f"Graph Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
print(f"Total Cases: {total_cases}")
print(
    f"Cases with at least one Law connection: {len(unique_cases_with_law)} ({len(unique_cases_with_law)/total_cases*100:.1f}%)"
)

print(f"\n1. RELATES_TO_LAW (Case -> Law) [Total: {case_law_total}]")
if case_law_sims:
    print(f"   - Average Semantic Similarity: {np.mean(case_law_sims):.4f}")
print(
    f"   - Dispute Alignment (Case & Law share same dispute category): {case_law_dispute_match} / {case_law_total} ({case_law_dispute_match/max(1,case_law_total)*100:.1f}%)"
)

print(f"\n2. SIMILAR_TO (Case -> Case via KNN) [Total: {len(case_case_sims)}]")
if case_case_sims:
    print(f"   - Average Semantic Similarity: {np.mean(case_case_sims):.4f}")

print(f"\n3. RELATED_DISPUTE (Law -> Dispute) [Total: {len(law_dispute_sims)}]")
if law_dispute_sims:
    print(f"   - Average Semantic Similarity: {np.mean(law_dispute_sims):.4f}")
