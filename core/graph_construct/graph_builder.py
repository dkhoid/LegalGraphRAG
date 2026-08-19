import numpy as np
from .graph_db import GraphDBManager
from tqdm import tqdm
from core.utils.logger import logger

from .llm_utils import get_embedding, summarize_texts


def store_nodes_with_embeddings(nodes_data):
    """
    nodes_data: Dict with keys 'case', 'law', 'dispute', each containing a list of node dicts
    """
    store_nodes(nodes_data)
    build_relationships()


def store_nodes(nodes_data):
    """
    Store nodes into the in-memory graph database.
    nodes_data: Dict with keys 'case', 'law', 'dispute', each containing a list of node dicts
    """
    db = GraphDBManager.get_db()
    case_nodes_data, law_nodes_data, dispute_nodes_data = (
        nodes_data["case"],
        nodes_data["law"],
        nodes_data["dispute"],
    )

    # Store Case nodes
    for node in tqdm(case_nodes_data, desc="Storing case nodes"):
        db.add_node(
            node["id"],
            "Cases",
            {
                "description": node.get("description"),
                "embedding": node.get("embedding"),
                "caseId": node.get("caseId"),
                "dispute": node.get("dispute"),
                "law": node.get("law"),
            },
        )

    # Store Law nodes
    for node in tqdm(law_nodes_data, desc="Storing law nodes"):
        db.add_node(
            node["id"],
            "Laws",
            {
                "entry": node.get("entry"),
                "description": node.get("description"),
                "embedding": node.get("embedding"),
                "disputes": node.get("disputes"),
                "judge_dep": node.get("judge_dep"),
                "related_laws": node.get("related_laws"),
                "insights": "",
            },
        )

    # Store Issue/Topic nodes
    for node in tqdm(dispute_nodes_data, desc="Storing crime nodes"):
        db.add_node(
            node["id"],
            "Disputes",
            {
                "description": node.get("description"),
                "embedding": node.get("embedding"),
            },
        )


def build_relationships():
    """
    Build relationships between nodes in the graph.
    """
    db = GraphDBManager.get_db()

    # Remove Law nodes with entry <= 101 (general law provisions not relevant to cases) BEFORE building relationships
    nodes_to_delete = []
    for node_id, node_info in db.nodes_data.items():
        if node_info["type"] == "Laws":
            entry = node_info["data"].get("entry")
            if entry is not None and str(entry).isdigit() and int(entry) <= 101:
                nodes_to_delete.append(node_id)

    if nodes_to_delete:
        print(
            f"Removing {len(nodes_to_delete)} Law nodes with entry <= 101 and their relationships"
        )
        for node_id in nodes_to_delete:
            db.graph.remove_node(node_id)
            del db.nodes_data[node_id]
            # Remove from embeddings index
            for node_type in db.embeddings:
                if node_id in db.embeddings[node_type]:
                    del db.embeddings[node_type][node_id]
        print(f"Successfully removed {len(nodes_to_delete)} Law nodes.")

    # Create Case-to-Law relationships (based on entry matching)
    case_nodes = db.get_nodes_by_type("Cases")

    # Build lookup for O(1) matching
    law_nodes_by_entry = {}
    for node_id, node_info in db.nodes_data.items():
        if node_info["type"] == "Laws":
            entry_val = node_info["data"].get("entry")
            if entry_val is not None:
                law_nodes_by_entry[str(entry_val)] = node_id

    for case_node in tqdm(case_nodes, desc="Linking cases to laws"):
        case_id = case_node["id"]
        law_entries = case_node.get("law")

        if not law_entries:
            continue

        for law_entry in law_entries:
            law_entry_str = str(law_entry).strip()
            if not law_entry_str:
                continue

            if law_entry_str in law_nodes_by_entry:
                db.add_edge(case_id, law_nodes_by_entry[law_entry_str], "RELATES_TO_LAW")
            else:
                # Fuzzy match: Gather all candidates instead of picking the first one randomly
                candidates = []
                for graph_law_entry, law_node_id in law_nodes_by_entry.items():
                    if (
                        graph_law_entry.endswith(f"+{law_entry_str}")
                        or graph_law_entry == law_entry_str
                    ):
                        candidates.append(law_node_id)

                if not candidates:
                    # Fallback fuzzy match
                    for graph_law_entry, law_node_id in law_nodes_by_entry.items():
                        if (
                            f"{law_entry_str}" in graph_law_entry.split("+")[-1:]
                            or f"_{law_entry_str}" in graph_law_entry
                        ):
                            candidates.append(law_node_id)

                if candidates:
                    case_emb = case_node.get("embedding")
                    case_disputes = case_node.get("dispute", [])
                    if isinstance(case_disputes, str):
                        case_disputes = [case_disputes]

                    best_law = None
                    best_score = -1

                    for law_node_id in candidates:
                        law_data = db.nodes_data[law_node_id]["data"]
                        law_emb = law_data.get("embedding")
                        law_disputes = law_data.get("disputes", [])
                        if isinstance(law_disputes, str):
                            law_disputes = [law_disputes]

                        score = 0.0
                        if case_emb is not None and law_emb is not None:
                            import numpy as np

                            v1, v2 = np.array(case_emb), np.array(law_emb)
                            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
                            score = float(np.dot(v1, v2) / norm) if norm > 0 else 0.0

                        # Massive boost if they share the same dispute category
                        if set(case_disputes).intersection(set(law_disputes)):
                            score += 1.0

                        if score > best_score:
                            best_score = score
                            best_law = law_node_id

                    if best_law:
                        db.add_edge(case_id, best_law, "RELATES_TO_LAW")

    # Create Law-to-Issue relationships (based on issue description matching)
    law_nodes = db.get_nodes_by_type("Laws")

    # Build lookup for O(1) exact matching
    dispute_nodes_by_desc = {}
    for node_id, node_info in db.nodes_data.items():
        if node_info["type"] == "Disputes":
            desc_val = node_info["data"].get("description")
            if desc_val:
                dispute_nodes_by_desc[desc_val] = node_id

    for law_node in tqdm(law_nodes, desc="Linking laws to crimes"):
        law_id = law_node["id"]
        dispute_descriptions = law_node.get("disputes")

        if not dispute_descriptions:
            continue

        for dispute_desc in dispute_descriptions:
            # Check if relationship already exists
            existing_neighbors = db.get_neighbors(law_id, "RELATED_DISPUTE")
            if existing_neighbors:
                found = False
                for dispute_id in existing_neighbors:
                    dispute_data = db.get_node(dispute_id)
                    if dispute_data and dispute_data.get("description") == dispute_desc:
                        found = True
                        break
                if found:
                    continue

            # Attempt exact match using O(1) lookup
            if dispute_desc in dispute_nodes_by_desc:
                db.add_edge(
                    law_id,
                    dispute_nodes_by_desc[dispute_desc],
                    "RELATED_DISPUTE",
                    {"match_type": "exact"},
                )
                continue

            # Attempt fuzzy match if exact match failed
            for node_id, node_info in db.nodes_data.items():
                if node_info["type"] == "Disputes":
                    desc = node_info["data"].get("description", "")
                    if desc and dispute_desc and (dispute_desc in desc or desc in dispute_desc):
                        db.add_edge(
                            law_id,
                            node_id,
                            "RELATED_DISPUTE",
                            {"match_type": "fuzzy"},
                        )
                        break

    # Remove Law nodes with entry <= 101 (general law provisions not relevant to cases)
    nodes_to_delete = []
    for node_id, node_info in db.nodes_data.items():
        if node_info["type"] == "Laws":
            entry = node_info["data"].get("entry")
            if entry is not None and str(entry).isdigit() and int(entry) <= 101:
                nodes_to_delete.append(node_id)

    node_count = len(nodes_to_delete)
    print(f"Removing {node_count} Law nodes with entry <= 101 and their relationships")

    for node_id in nodes_to_delete:
        db.graph.remove_node(node_id)
        del db.nodes_data[node_id]
        # Remove from embeddings index
        for node_type in db.embeddings:
            if node_id in db.embeddings[node_type]:
                del db.embeddings[node_type][node_id]

    deleted_count = len(nodes_to_delete)
    print(f"Successfully removed {deleted_count} Law nodes and their relationships")


def run_knn(top_k=3):
    db = GraphDBManager.get_db()

    # Get all Cases nodes
    case_nodes = db.get_nodes_by_type("Cases")
    if len(case_nodes) < 2:
        return

    # Get embeddings for all Case nodes
    case_embeddings = []
    case_ids = []
    for node in case_nodes:
        emb = node.get("embedding")
        if emb is not None:
            case_embeddings.append(emb)
            case_ids.append(node["id"])

    if len(case_embeddings) < 2:
        return

    # Clear case_nodes to free up memory early
    del case_nodes

    # Convert to float32 for 50% memory savings
    case_embeddings = np.array(case_embeddings, dtype=np.float32)

    # Normalize in-place to prevent allocating a new large matrix
    norms = np.linalg.norm(case_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10  # Avoid division by zero
    case_embeddings /= norms
    normalized_embeddings = case_embeddings

    num_cases = len(case_ids)
    batch_size = 1000  # Adjust if memory is still an issue

    for start_idx in tqdm(range(0, num_cases, batch_size), desc="Running Vectorized KNN"):
        end_idx = min(start_idx + batch_size, num_cases)
        batch_embeddings = normalized_embeddings[start_idx:end_idx]

        # Dot product yields cosine similarity because vectors are normalized
        # Result shape: (batch_size, num_cases)
        sim_matrix = np.dot(batch_embeddings, normalized_embeddings.T)

        for i, row_idx in enumerate(range(start_idx, end_idx)):
            # Ignore self-similarity by setting it to -1
            sim_matrix[i, row_idx] = -1.0

            row_sims = sim_matrix[i]
            # Use argpartition to get top_k indices quickly
            if num_cases > top_k:
                top_indices = np.argpartition(row_sims, -top_k)[-top_k:]
                # Sort the top_k indices by similarity descending
                top_indices = top_indices[np.argsort(-row_sims[top_indices])]
            else:
                top_indices = np.argsort(-row_sims)

            for j in top_indices:
                if row_sims[j] < -0.99:  # Avoid self-similarity if num_cases <= top_k
                    continue
                score = float(row_sims[j])
                db.add_edge(case_ids[row_idx], case_ids[j], "SIMILAR_TO", {"score": score})


def create_clusters(model):
    db = GraphDBManager.get_db()

    # Run community detection and centrality analysis
    communities = db.detect_communities()

    # Update communityId for each node
    for node_id, comm_id in communities.items():
        db.update_node(node_id, {"communityId": comm_id})

    # Compute PageRank and degree centrality
    pagerank = db.compute_pagerank()
    degrees = db.compute_degree_centrality()

    # Update pagerank and degree for each node
    for node_id, score in pagerank.items():
        db.update_node(node_id, {"pagerank": score})
    for node_id, degree in degrees.items():
        db.update_node(node_id, {"degree": degree})

    # Get all unique community IDs (Cases nodes only)
    community_ids = set()
    for node_id, node_info in db.nodes_data.items():
        if node_info["type"] == "Cases" and node_info["data"].get("communityId") is not None:
            community_ids.add(node_info["data"]["communityId"])

    community_ids = sorted(list(community_ids))
    print(f"Detected {len(community_ids)} communities.")

    # Auto-save before starting the long clustering process
    try:
        import os

        graph_path = os.getenv("graph_db_path", "./data/processed/graph.pkl")
        GraphDBManager.save(graph_path)
        print(f"Checkpoint saved before clustering to {graph_path}")
    except Exception as e:
        print(f"Failed to save checkpoint: {e}")

    for i, community_id in enumerate(tqdm(community_ids, desc="Creating clusters")):
        # Select key nodes based on composite score of PageRank and degree centrality
        important_nodes = []
        for node_id, node_info in db.nodes_data.items():
            if (
                node_info["type"] == "Cases"
                and node_info["data"].get("communityId") == community_id
            ):
                pagerank_score = node_info["data"].get("pagerank", 0)
                degree_score = node_info["data"].get("degree", 0)
                composite_score = pagerank_score * 0.7 + degree_score * 0.3
                important_nodes.append(
                    {
                        "description": node_info["data"].get("description", ""),
                        "composite_score": composite_score,
                    }
                )

        important_nodes.sort(key=lambda x: x["composite_score"], reverse=True)
        # Giảm thiểu lượng token: Lấy top 3 case thay vì 10, và cắt ngắn mỗi case còn tối đa 500 ký tự
        descriptions = [
            (
                node["description"][:500] + "..."
                if len(node["description"]) > 500
                else node["description"]
            )
            for node in important_nodes[:3]
        ]

        if not descriptions:
            continue
        descriptions = "\n".join(descriptions)
        print(descriptions)

        # Get top 5 legal issues/topics connected to the most cases in this community
        # Count case occurrences per issue in this community
        dispute_counts = {}
        for node_id, node_info in db.nodes_data.items():
            if (
                node_info["type"] == "Cases"
                and node_info["data"].get("communityId") == community_id
            ):
                # Find laws linked to this case, then find issues linked to those laws
                law_neighbors = db.get_neighbors(node_id, "RELATES_TO_LAW")
                for law_id in law_neighbors:
                    crime_neighbors = db.get_neighbors(law_id, "RELATED_DISPUTE")
                    for dispute_id in crime_neighbors:
                        dispute_data = db.get_node(dispute_id)
                        if dispute_data:
                            dispute_name = dispute_data.get("description", "")
                            dispute_counts[dispute_name] = dispute_counts.get(dispute_name, 0) + 1

        top_disputes = sorted(dispute_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_disputes = [{"dispute_name": name, "case_count": count} for name, count in top_disputes]

        # Build a focused context prompt for summarization
        dispute_context = ""
        if top_disputes:
            dispute_descriptions = [
                f"{dispute['dispute_name']} ({dispute['case_count']} cases)"
                for dispute in top_disputes
            ]
            dispute_context = "Key legal issues: " + ", ".join(dispute_descriptions)

        enhanced_prompt = f"""
Community Overview:
- {dispute_context}

Key Case Descriptions:
{descriptions}
"""

        summary_data = summarize_texts(model, enhanced_prompt)
        community_embedding = get_embedding(summary_data)

        # Store top issue names in the Cluster node
        top_dispute_names = [dispute["dispute_name"] for dispute in top_disputes]
        top_dispute_counts = [dispute["case_count"] for dispute in top_disputes]

        # Create Cluster node with metadata
        cluster_id = str(community_id)
        db.add_node(
            cluster_id,
            "Cluster",
            {
                "summary": summary_data,
                "embedding": community_embedding,
                "top_disputes": top_dispute_names,
                "top_dispute_counts": top_dispute_counts,
            },
        )

        # Link Cluster to its member Case nodes
        key_nodes = []
        for node_id, node_info in db.nodes_data.items():
            if (
                node_info["type"] == "Cases"
                and node_info["data"].get("communityId") == community_id
            ):
                key_nodes.append(node_id)
                db.add_edge(node_id, cluster_id, "BELONGS_TO_CLUSTER")

        # Auto-save every 50 clusters
        if (i + 1) % 50 == 0:
            try:
                GraphDBManager.save(graph_path)
            except Exception:
                pass


def update_insights_in_graph(law_id, insights):
    db = GraphDBManager.get_db()
    db.update_node(law_id, {"insights": insights})


def construct_feature_graph(model, nodes_data):
    import concurrent.futures

    GraphDBManager.initialize()

    db = GraphDBManager.get_db()

    print("Attempting to recover generated embeddings from memory cache...")
    try:
        cache_by_desc = {}
        for nid, c_node in db.nodes_data.items():
            c_data = c_node.get("data", {})
            if "description" in c_data and "embedding" in c_data:
                cache_by_desc[c_data["description"]] = c_data["embedding"]

        recovered = 0
        for k in ["case", "law", "dispute"]:
            for node in nodes_data.get(k, []):
                desc = node.get("description")
                if desc and desc in cache_by_desc:
                    node["embedding"] = cache_by_desc[desc]
                    recovered += 1
        print(f"Successfully recovered {recovered} embeddings!")

        del cache_by_desc
    except Exception as e:
        print("Could not recover embeddings:", e)

    # VERY IMPORTANT: Clear the old graph data to prevent infinitely duplicating nodes
    # because force_rebuild expects a fresh graph construction!
    print("Clearing old graph to prevent duplication and free RAM...")
    db.graph.clear()
    db.nodes_data.clear()
    db.embeddings = {"Cases": {}, "Laws": {}, "Disputes": {}, "Cluster": {}}
    db._vector_indexes = {
        "Cases": {"vectors": [], "ids": []},
        "Laws": {"vectors": [], "ids": []},
        "Disputes": {"vectors": [], "ids": []},
        "Cluster": {"vectors": [], "ids": []},
    }
    import gc

    gc.collect()

    case_nodes_data, law_nodes_data, dispute_nodes_data = (
        nodes_data["case"],
        nodes_data["law"],
        nodes_data["dispute"],
    )

    def process_nodes(nodes_data_list, desc):
        def _get_emb(item):
            i, node = item
            if node.get("embedding") is not None:
                return i, node["id"], node["embedding"]
            emb = get_embedding(node.get("description") or "")
            return i, node["id"], emb

        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            futures = [
                executor.submit(_get_emb, (i, node)) for i, node in enumerate(nodes_data_list)
            ]
            for future in tqdm(
                concurrent.futures.as_completed(futures), total=len(futures), desc=desc
            ):
                i, node_id, emb = future.result()
                if emb is not None:
                    nodes_data_list[i]["embedding"] = emb
                else:
                    logger.error(f"Failed to generate embedding for node {node_id}")

    # Generate embedding vectors for each node using multi-threading
    process_nodes(case_nodes_data, "Generating case embeddings")
    process_nodes(law_nodes_data, "Generating law embeddings")
    process_nodes(dispute_nodes_data, "Generating crime embeddings")

    # Store nodes and embeddings
    store_nodes_with_embeddings(nodes_data)
    build_relationships()

    # Run KNN and clustering
    run_knn(top_k=3)
    create_clusters(model)

    # Lưu file pkl trước khi sync sang Neo4j
    import os

    graph_path = os.getenv("graph_db_path", "./data/processed/graph.pkl")
    db = GraphDBManager.get_db()
    try:
        GraphDBManager.save(graph_path)
        logger.info(f"Đã lưu graph cục bộ tại {graph_path} trước khi sync Neo4j.")
    except Exception as e:
        logger.error(f"Lỗi khi lưu graph: {e}")

    # Sync to Neo4j
    try:
        from core.graph_construct.neo4j_manager import neo4j_manager

        neo4j_manager.sync_from_memory_graph(db)
    except Exception as e:
        logger.error(f"Failed to sync graph to Neo4j: {e}")
