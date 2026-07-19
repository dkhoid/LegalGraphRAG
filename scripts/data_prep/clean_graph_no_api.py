import os
import sys
import json

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.graph_construct.graph_db import GraphDBManager


def clean_graph_pkl():
    graph_path = os.path.join(PROJECT_ROOT, "data", "processed", "graph.pkl")
    if not os.path.exists(graph_path):
        print("Graph not found at", graph_path)
        return

    GraphDBManager.load(graph_path)
    db = GraphDBManager.get_db()

    target_ids = ["law_999999", "law_1000000", "qa_bhxh_1"]
    removed_count = 0

    for tid in target_ids:
        if tid in db.nodes:
            del db.nodes[tid]
            removed_count += 1
            print(f"Removed node {tid} from graph.pkl")

            # Clean up edges pointing to or from this node
            edges_to_remove = []
            for edge in db.edges:
                if edge["source"] == tid or edge["target"] == tid:
                    edges_to_remove.append(edge)

            for edge in edges_to_remove:
                db.edges.remove(edge)
                print(f"Removed edge associated with {tid}")

    if removed_count > 0:
        GraphDBManager.save(graph_path)
        print("Graph saved successfully.")
    else:
        print("No target nodes found in graph.pkl.")


def clean_json_files():
    # Clean law_to_dispute.json
    law_file = os.path.join(PROJECT_ROOT, "data", "processed", "law_to_dispute.json")
    if os.path.exists(law_file):
        with open(law_file, "r", encoding="utf-8") as f:
            laws = json.load(f)

        original_len = len(laws)
        laws = [
            law
            for law in laws
            if law.get("id") not in [999999, 1000000, "law_999999", "law_1000000"]
        ]

        if len(laws) < original_len:
            with open(law_file, "w", encoding="utf-8") as f:
                json.dump(laws, f, ensure_ascii=False, indent=2)
            print(f"Removed {original_len - len(laws)} entries from law_to_dispute.json")

    # Clean cases_with_feature.json
    cases_file = os.path.join(PROJECT_ROOT, "data", "processed", "cases_with_feature.json")
    if os.path.exists(cases_file):
        with open(cases_file, "r", encoding="utf-8") as f:
            cases = json.load(f)

        original_len = len(cases)
        cases = [case for case in cases if case.get("id") != "qa_bhxh_1"]

        if len(cases) < original_len:
            with open(cases_file, "w", encoding="utf-8") as f:
                json.dump(cases, f, ensure_ascii=False, indent=2)
            print(f"Removed {original_len - len(cases)} entries from cases_with_feature.json")


if __name__ == "__main__":
    print("Starting zero-cost data cleaning...")
    # Skipping clean_graph_pkl() to avoid OOM
    clean_json_files()
    print("Done!")
