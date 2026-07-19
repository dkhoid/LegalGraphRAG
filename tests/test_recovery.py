import pickle

with open("./data/processed/graph.pkl", "rb") as f:
    data = pickle.load(f)
cached_nodes = data.get("nodes_data", {})
keys = list(cached_nodes.keys())
print("First 5 keys:", keys[:5])
print("Are they strings?", all(isinstance(k, str) for k in keys[:5]))
print("First 5 values types:", [cached_nodes[k].get("type") for k in keys[:5]])
