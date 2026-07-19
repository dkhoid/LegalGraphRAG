import pickle

with open("data/processed/graph.pkl", "rb") as f:
    data = pickle.load(f)

for node_id, node_info in data["nodes_data"].items():
    if node_info["type"] == "Laws":
        emb = node_info["data"].get("embedding")
        if emb is not None:
            print(f"Embedding length: {len(emb)}")
            break
