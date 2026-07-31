"""
Graph Quality Audit Script
Kiểm tra chất lượng quan hệ trong đồ thị tri thức LegalGraphRAG.
"""

import pickle
import sys
import os
import random
import numpy as np
from collections import Counter, defaultdict

# Add project root to path
PROJECT_ROOT = "/home/rokisaki/Documents/Coding/testing_code/LegalGraphRAG"
sys.path.insert(0, PROJECT_ROOT)

GRAPH_PATH = os.path.join(PROJECT_ROOT, "data/processed/graph.pkl")


def load_graph(path):
    """Load pickled graph data"""
    print(f"Loading graph from {path}...")
    print(f"File size: {os.path.getsize(path) / 1024 / 1024:.1f} MB")
    with open(path, "rb") as f:
        data = pickle.load(f)
    print("Graph loaded successfully.\n")
    return data


def overview(data):
    """Print graph overview statistics"""
    graph = data["graph"]
    nodes_data = data["nodes_data"]

    print("=" * 70)
    print("TONG QUAN DO THI")
    print("=" * 70)

    type_counts = Counter()
    for nid, ninfo in nodes_data.items():
        type_counts[ninfo["type"]] += 1

    print(f"\nTong nodes: {len(nodes_data)}")
    print(f"Tong edges: {graph.number_of_edges()}")
    print("\nPhan loai nodes:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t:15s}: {c:,}")

    edge_types = Counter()
    for u, v, d in graph.edges(data=True):
        edge_types[d.get("relation_type", "UNKNOWN")] += 1

    print("\nPhan loai edges:")
    for t, c in sorted(edge_types.items(), key=lambda x: -x[1]):
        print(f"  {t:25s}: {c:,}")

    return type_counts, edge_types


def truncate(s, max_len=120):
    s = str(s).replace("\n", " ").strip()
    return s[:max_len] + "..." if len(s) > max_len else s


def check_relates_to_law(data, sample_size=10):
    """Check RELATES_TO_LAW: Case -> Law relationships"""
    graph = data["graph"]
    nodes_data = data["nodes_data"]

    print("\n" + "=" * 70)
    print("KIEM TRA RELATES_TO_LAW (Case -> Law)")
    print("=" * 70)

    edges = []
    for u, v, d in graph.edges(data=True):
        if d.get("relation_type") == "RELATES_TO_LAW":
            edges.append((u, v, d))

    print(f"\nTong quan he RELATES_TO_LAW: {len(edges):,}")

    if not edges:
        print("  Khong tim thay quan he nao!")
        return

    wrong_type = 0
    for u, v, d in edges:
        u_type = nodes_data.get(u, {}).get("type", "?")
        v_type = nodes_data.get(v, {}).get("type", "?")
        if u_type != "Cases" or v_type != "Laws":
            wrong_type += 1

    print(f"Sai loai node (Case->Law): {wrong_type}/{len(edges)}")

    # Count laws per case
    laws_per_case = defaultdict(int)
    for u, v, d in edges:
        laws_per_case[u] += 1
    lpc = list(laws_per_case.values())
    print(f"So case co quan he law: {len(laws_per_case):,}")
    print(f"Trung binh laws/case: {np.mean(lpc):.1f}, Max: {max(lpc)}, Min: {min(lpc)}")

    samples = random.sample(edges, min(sample_size, len(edges)))
    print(f"\nMau ngau nhien ({len(samples)} edges):")
    print("-" * 70)

    for i, (u, v, d) in enumerate(samples):
        case_data = nodes_data.get(u, {}).get("data", {})
        law_data = nodes_data.get(v, {}).get("data", {})

        case_desc = case_data.get("description", case_data.get("fact", "N/A"))
        law_entry = law_data.get("entry", v)
        law_desc = law_data.get("description", "N/A")
        case_laws = case_data.get("law", case_data.get("laws", []))

        print(f"\n  [{i+1}] Case: {truncate(u, 50)}")
        print(f"       Case desc: {truncate(case_desc, 120)}")
        print(f"       -> Law: {law_entry}")
        print(f"       Law desc: {truncate(law_desc, 120)}")
        if case_laws:
            print(f"       Case laws field: {truncate(str(case_laws), 100)}")


def check_related_dispute(data, sample_size=10):
    """Check RELATED_DISPUTE: Law -> Dispute relationships"""
    graph = data["graph"]
    nodes_data = data["nodes_data"]

    print("\n" + "=" * 70)
    print("KIEM TRA RELATED_DISPUTE (Law -> Dispute)")
    print("=" * 70)

    edges = []
    for u, v, d in graph.edges(data=True):
        if d.get("relation_type") == "RELATED_DISPUTE":
            edges.append((u, v, d))

    print(f"\nTong quan he RELATED_DISPUTE: {len(edges):,}")

    if not edges:
        print("  Khong tim thay quan he nao!")
        return

    law_disputes = defaultdict(list)
    for u, v, d in edges:
        law_disputes[u].append(v)

    dispute_counts = [len(v) for v in law_disputes.values()]
    print(f"So law co quan he dispute: {len(law_disputes):,}")
    print(f"Trung binh dispute/law: {np.mean(dispute_counts):.1f}")
    print(f"Max dispute/law: {max(dispute_counts)}")

    # List all dispute nodes
    disputes = [nid for nid, ninfo in nodes_data.items() if ninfo["type"] == "Disputes"]
    print(f"\nDanh sach tat ca Dispute nodes ({len(disputes)}):")
    for did in sorted(disputes):
        dd = nodes_data[did]["data"]
        name = dd.get("name", dd.get("dispute_type", did))
        # Count how many laws connect to this dispute
        in_count = sum(1 for u, v, d in edges if v == did)
        print(f"  - {did}: {name} ({in_count} laws)")

    # Sample
    samples = random.sample(edges, min(sample_size, len(edges)))
    print(f"\nMau ngau nhien ({len(samples)} edges):")
    print("-" * 70)

    for i, (u, v, d) in enumerate(samples):
        law_data = nodes_data.get(u, {}).get("data", {})
        disp_data = nodes_data.get(v, {}).get("data", {})

        law_entry = law_data.get("entry", u)
        law_desc = law_data.get("description", "N/A")
        law_dispute_field = law_data.get("dispute", [])
        disp_name = disp_data.get("name", disp_data.get("dispute_type", v))

        print(f"\n  [{i+1}] Law: {law_entry}")
        print(f"       Law desc: {truncate(law_desc, 120)}")
        print(f"       -> Dispute node: {disp_name}")
        print(f"       Law's dispute field: {truncate(str(law_dispute_field), 100)}")


def check_similar_to(data, sample_size=10):
    """Check SIMILAR_TO: Case <-> Case with cosine similarity"""
    graph = data["graph"]
    nodes_data = data["nodes_data"]
    embeddings = data.get("embeddings", {})

    print("\n" + "=" * 70)
    print("KIEM TRA SIMILAR_TO (Case <-> Case)")
    print("=" * 70)

    edges = []
    for u, v, d in graph.edges(data=True):
        if d.get("relation_type") == "SIMILAR_TO":
            edges.append((u, v, d))

    print(f"\nTong quan he SIMILAR_TO: {len(edges):,}")

    if not edges:
        print("  Khong tim thay quan he nao!")
        return

    case_embeddings = embeddings.get("Cases", {})
    similarities = []
    low_sim_edges = []

    for u, v, d in edges:
        emb_u = case_embeddings.get(u)
        emb_v = case_embeddings.get(v)
        if emb_u is not None and emb_v is not None:
            emb_u_flat = np.array(emb_u).flatten()
            emb_v_flat = np.array(emb_v).flatten()
            if emb_u_flat.shape == emb_v_flat.shape and emb_u_flat.shape[0] > 0:
                from scipy.spatial.distance import cosine as cos_dist

                sim = 1 - cos_dist(emb_u_flat, emb_v_flat)
                similarities.append(sim)
                if sim < 0.5:
                    low_sim_edges.append((u, v, sim))

    if similarities:
        sims = np.array(similarities)
        print(f"\nPhan phoi cosine similarity cua SIMILAR_TO edges:")
        print(f"  Min:    {sims.min():.4f}")
        print(f"  Q25:    {np.percentile(sims, 25):.4f}")
        print(f"  Median: {np.median(sims):.4f}")
        print(f"  Q75:    {np.percentile(sims, 75):.4f}")
        print(f"  Max:    {sims.max():.4f}")
        print(f"  Mean:   {sims.mean():.4f}")
        print(
            f"\n  Edges co similarity < 0.5: {len(low_sim_edges)} / {len(similarities)} ({100*len(low_sim_edges)/len(similarities):.1f}%)"
        )
        print(
            f"  Edges co similarity < 0.3: {sum(1 for s in similarities if s < 0.3)} / {len(similarities)}"
        )

    if low_sim_edges:
        low_sim_edges.sort(key=lambda x: x[2])
        show = min(sample_size, len(low_sim_edges))
        print(f"\nTop {show} cap SIMILAR_TO co similarity THAP NHAT:")
        print("-" * 70)
        for i, (u, v, sim) in enumerate(low_sim_edges[:show]):
            case_u = nodes_data.get(u, {}).get("data", {})
            case_v = nodes_data.get(v, {}).get("data", {})
            desc_u = case_u.get("description", case_u.get("fact", "N/A"))
            desc_v = case_v.get("description", case_v.get("fact", "N/A"))
            print(f"\n  [{i+1}] Similarity = {sim:.4f}")
            print(f"       Case A: {truncate(desc_u, 150)}")
            print(f"       Case B: {truncate(desc_v, 150)}")


def check_belongs_to_cluster(data, sample_size=3):
    """Check BELONGS_TO_CLUSTER: Case -> Cluster"""
    graph = data["graph"]
    nodes_data = data["nodes_data"]

    print("\n" + "=" * 70)
    print("KIEM TRA BELONGS_TO_CLUSTER (Case -> Cluster)")
    print("=" * 70)

    edges = []
    for u, v, d in graph.edges(data=True):
        if d.get("relation_type") == "BELONGS_TO_CLUSTER":
            edges.append((u, v, d))

    print(f"\nTong quan he BELONGS_TO_CLUSTER: {len(edges):,}")

    clusters = defaultdict(list)
    for u, v, d in edges:
        clusters[v].append(u)

    print(f"So clusters: {len(clusters)}")
    for cluster_id, cases in sorted(clusters.items()):
        cluster_data = nodes_data.get(cluster_id, {}).get("data", {})
        summary = cluster_data.get("summary", "N/A")
        print(f"\n  Cluster: {cluster_id} ({len(cases)} cases)")
        print(f"     Summary: {truncate(summary, 200)}")

        sample_cases = random.sample(cases, min(sample_size, len(cases)))
        for case_id in sample_cases:
            case_data = nodes_data.get(case_id, {}).get("data", {})
            desc = case_data.get("description", case_data.get("fact", "N/A"))
            print(f"     -> Case: {truncate(desc, 150)}")


def check_orphan_nodes(data):
    """Check for orphan nodes"""
    graph = data["graph"]
    nodes_data = data["nodes_data"]

    print("\n" + "=" * 70)
    print("KIEM TRA NODE MO COI (Khong co quan he)")
    print("=" * 70)

    orphans = defaultdict(list)
    for node_id in nodes_data:
        in_deg = graph.in_degree(node_id) if node_id in graph else 0
        out_deg = graph.out_degree(node_id) if node_id in graph else 0
        if in_deg == 0 and out_deg == 0:
            node_type = nodes_data[node_id]["type"]
            orphans[node_type].append(node_id)

    total_orphans = sum(len(v) for v in orphans.values())
    print(f"\nTong node mo coi: {total_orphans}")
    for t, ids in orphans.items():
        print(f"  {t:15s}: {len(ids):,}")
        for oid in ids[:3]:
            od = nodes_data.get(oid, {}).get("data", {})
            desc = od.get("entry", od.get("description", od.get("name", oid)))
            print(f"    -> {truncate(str(desc), 80)}")


def main():
    random.seed(42)
    data = load_graph(GRAPH_PATH)
    overview(data)
    check_relates_to_law(data)
    check_related_dispute(data)
    check_similar_to(data)
    check_belongs_to_cluster(data)
    check_orphan_nodes(data)

    print("\n" + "=" * 70)
    print("KIEM TRA HOAN TAT")
    print("=" * 70)


if __name__ == "__main__":
    main()
