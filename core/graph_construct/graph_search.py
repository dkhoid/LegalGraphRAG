import numpy as np
from .graph_db import GraphDBManager

from .llm_utils import get_embedding, rerank, _get_bm25_index, rerank_clusters


def search_similar_nodes_top(model, query_embedding, query_text, top_k=5):
    db = GraphDBManager.get_db()

    # Find the most similar Cluster first
    cluster_results = db.find_similar_nodes(query_embedding, "Cluster", top_k=5)

    if not cluster_results:
        return [], [], []

    clusters = []
    for ids, record in enumerate(cluster_results):
        clusters.append(
            {
                "code": ids,
                "cluster_id": record["id"],
                "summary": record.get("summary", ""),
            }
        )

    cluster_ids = rerank_clusters(model, clusters, query_text)
    cluster_ids = [c for c in cluster_ids if 0 <= c and c < len(clusters)]
    if not cluster_ids:
        cluster_ids = [0]

    neighbors = []
    for cluster_id in cluster_ids[:2]:
        cluster_node_id = clusters[cluster_id]["cluster_id"]

        # Find the most similar Node within this Cluster
        # Get all Cases nodes belonging to this cluster
        cluster_cases = []
        for node_id, node_info in db.nodes_data.items():
            if node_info["type"] == "Cases":
                # Check if there is a BELONGS_TO edge pointing to this cluster
                neighbors_list = db.get_neighbors(node_id, "BELONGS_TO")
                if cluster_node_id in neighbors_list:
                    node_data = node_info["data"].copy()
                    node_data["id"] = node_id
                    cluster_cases.append(node_data)

        # Compute similarity and sort results
        case_similarities = []
        for case_data in cluster_cases:
            emb = case_data.get("embedding")
            if emb is not None:
                sim = db.cosine_similarity(query_embedding, np.array(emb))
                case_similarities.append((case_data, sim))

        case_similarities.sort(key=lambda x: x[1], reverse=True)

        for case_data, similarity in case_similarities[:top_k]:
            if similarity < 0.55:
                continue
            neighbors.append(
                {
                    "id": case_data["id"],
                    "description": case_data.get("description", ""),
                    "caseId": case_data.get("caseId", ""),
                    "similarity": similarity,
                    "law": case_data.get("law", []),
                    "dispute": case_data.get("dispute", []),
                }
            )

    neighbors = sorted(neighbors, key=lambda x: x["similarity"], reverse=True)
    for ids, neighbor in enumerate(neighbors):
        neighbor["rank"] = ids + 1
    neighbors = rerank(model, query_text, neighbors)

    cases = []
    laws = []
    for neighbor in neighbors:
        # Get associated Law nodes
        law_neighbors = db.get_neighbors(neighbor["id"], "RELATES_TO_LAW")
        for law_id in law_neighbors:
            law_data = db.get_node(law_id)
            if law_data:
                laws.append(
                    {
                        "id": law_id,
                        "entry": law_data.get("entry"),
                        "description": law_data.get("description"),
                        "disputes": law_data.get("disputes"),
                        "judge_dep": law_data.get("judge_dep"),
                        "related_laws": law_data.get("related_laws"),
                        "insights": law_data.get("insights", ""),
                    }
                )
        cases.append(neighbor)

    return clusters, cases, laws


def search_similar_nodes_direct(model, query_embedding, query_text, top_k=5):
    db = GraphDBManager.get_db()

    # Directly search for the most similar nodes among all Cases nodes
    neighbor_results = db.find_similar_nodes(query_embedding, "Cases", top_k=top_k)

    if not neighbor_results:
        return [], []

    neighbors = []
    for record in neighbor_results:
        if record["similarity"] < 0.55:
            continue
        neighbors.append(
            {
                "id": record["id"],
                "description": record.get("description", ""),
                "caseId": record.get("caseId", ""),
                "similarity": record["similarity"],
                "law": record.get("law", []),
                "dispute": record.get("dispute", []),
            }
        )

    neighbors = sorted(neighbors, key=lambda x: x["similarity"], reverse=True)
    for ids, neighbor in enumerate(neighbors):
        neighbor["rank"] = ids + 1
    neighbors = rerank(model, query_text, neighbors)

    cases = []
    laws = []
    for neighbor in neighbors:
        # Get associated Law nodes
        law_neighbors = db.get_neighbors(neighbor["id"], "RELATES_TO_LAW")
        for law_id in law_neighbors:
            law_data = db.get_node(law_id)
            if law_data:
                laws.append(
                    {
                        "id": law_id,
                        "entry": law_data.get("entry"),
                        "description": law_data.get("description"),
                        "disputes": law_data.get("disputes"),
                        "judge_dep": law_data.get("judge_dep"),
                        "related_laws": law_data.get("related_laws"),
                        "insights": law_data.get("insights", ""),
                    }
                )
        cases.append(neighbor)

    return cases, laws


def query_similar_nodes_naive(model, query_text, top_k=3):
    query_embedding = get_embedding(query_text)
    if query_embedding is None:
        return []

    from core.graph_construct.neo4j_manager import neo4j_manager

    if not neo4j_manager.driver:
        return []

    neighbors = []
    with neo4j_manager.driver.session() as session:
        vector_query = """
        CALL db.index.vector.queryNodes('case_embeddings', $top_k, $query_embedding)
        YIELD node AS case, score
        RETURN case, score
        """
        try:
            results = session.run(vector_query, top_k=top_k, query_embedding=query_embedding)
            for i, record in enumerate(results):
                c = record["case"]
                neighbors.append(
                    {
                        "id": c.get("id"),
                        "description": c.get("description", ""),
                        "caseId": c.get("caseId", ""),
                        "similarity": record["score"],
                        "rank": i + 1,
                    }
                )
        except Exception as e:
            print(f"Neo4j case search error: {e}")

    return neighbors


def query_similar_nodes(model, query_text, retrieve_config):
    query_embedding = get_embedding(query_text)
    if query_embedding is None:
        return {}, [], []

    # Call the two retrieval strategies
    if retrieve_config["top_retrieve"]:
        top_result_clusters, top_result_cases, top_result_laws = search_similar_nodes_top(
            model,
            query_embedding,
            query_text,
            top_k=retrieve_config["top_retrieve_top_k"],
        )
    else:
        top_result_clusters, top_result_cases, top_result_laws = [], [], []
    if retrieve_config["direct_retrieve"]:
        direct_result_cases, direct_result_laws = search_similar_nodes_direct(
            model,
            query_embedding,
            query_text,
            top_k=retrieve_config["direct_retrieve_top_k"],
        )
    else:
        direct_result_cases, direct_result_laws = [], []

    # Hybrid BM25 logic to find missing laws directly
    bm25_laws = []
    bm25, law_mapping = _get_bm25_index()
    if bm25 and law_mapping:
        try:
            tokenized_query = query_text.lower().split()
            bm25_scores = bm25.get_scores(tokenized_query)
            top_k_bm25 = 5
            top_indices = sorted(
                range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
            )[:top_k_bm25]
            bm25_laws = [law_mapping[i] for i in top_indices if bm25_scores[i] > 0]
        except Exception as e:
            print(f"BM25 Graph Error: {e}")

    # Aggregate results
    result_cases = []
    seen_ids_cases = set()  # for deduplication
    result_laws = []
    seen_ids_laws = set()  # for deduplication

    # Process results from search_similar_nodes_top
    if top_result_cases:  # Ensure result is not None
        neighbors = top_result_cases
        for neighbor in neighbors:
            if neighbor["id"] and neighbor["id"] not in seen_ids_cases:
                result_cases.append(
                    {
                        "id": neighbor["id"],
                        "description": neighbor["description"],
                        "caseId": neighbor["caseId"],
                        "rank": neighbor["rank"],
                    }
                )
                seen_ids_cases.add(neighbor["id"])

    # Process law results from search_similar_nodes_top
    if top_result_laws:  # Ensure result is not None
        neighbors = top_result_laws
        for neighbor in neighbors:
            if neighbor["id"] and neighbor["id"] not in seen_ids_laws:
                result_laws.append(
                    {
                        "id": neighbor["id"],
                        "entry": neighbor["entry"],
                        "description": neighbor["description"],
                        "disputes": neighbor["disputes"],
                        "judge_dep": neighbor["judge_dep"],
                        "related_laws": neighbor["related_laws"],
                    }
                )
                seen_ids_laws.add(neighbor["id"])

    # Process results from search_similar_nodes_direct
    if direct_result_cases:  # Ensure result is not None
        neighbors = direct_result_cases
        # Add neighbor nodes
        for neighbor in neighbors:
            if neighbor["id"] and neighbor["id"] not in seen_ids_cases:
                result_cases.append(
                    {
                        "id": neighbor["id"],
                        "description": neighbor["description"],
                        "caseId": neighbor["caseId"],
                        "rank": neighbor["rank"],
                    }
                )
                seen_ids_cases.add(neighbor["id"])

    # Process law results from search_similar_nodes_direct
    if direct_result_laws:  # Ensure result is not None
        neighbors = direct_result_laws
        for neighbor in neighbors:
            if neighbor["id"] and neighbor["id"] not in seen_ids_laws:
                result_laws.append(
                    {
                        "id": neighbor["id"],
                        "entry": neighbor["entry"],
                        "description": neighbor["description"],
                        "disputes": neighbor["disputes"],
                        "judge_dep": neighbor["judge_dep"],
                        "related_laws": neighbor["related_laws"],
                    }
                )
                seen_ids_laws.add(neighbor["id"])
    # Reciprocal Rank Fusion (RRF) for Laws
    rrf_scores = {}
    law_dict = {}

    def add_to_rrf(law_list, k=60):
        for i, law in enumerate(law_list):
            lid = law["id"]
            if lid not in law_dict:
                law_dict[lid] = law
            rrf_scores[lid] = rrf_scores.get(lid, 0.0) + 1.0 / (k + i + 1)

    add_to_rrf(result_laws)
    add_to_rrf(bm25_laws)

    sorted_law_ids = sorted(rrf_scores.keys(), key=lambda lid: rrf_scores[lid], reverse=True)
    result_laws = [law_dict[lid] for lid in sorted_law_ids]

    original_retrieved_res = {
        "top": {
            "clusters": top_result_clusters,
            "cases": top_result_cases,
            "laws": top_result_laws,
        },
        "direct": {"cases": direct_result_cases, "laws": direct_result_laws},
        "augmented": [],
    }

    return original_retrieved_res, result_cases, result_laws


def query_similar_laws_naive(query_text, top_k=1):
    query_embedding = get_embedding(query_text)
    if query_embedding is None:
        return []

    from core.graph_construct.neo4j_manager import neo4j_manager

    if not neo4j_manager.driver:
        return []

    result_laws = []
    seen_law_ids = set()

    with neo4j_manager.driver.session() as session:
        vector_query = """
        CALL db.index.vector.queryNodes('law_embeddings', $top_k, $query_embedding)
        YIELD node AS law, score
        RETURN law, score
        """
        try:
            results = session.run(vector_query, top_k=top_k, query_embedding=query_embedding)
            for record in results:
                score = record["score"]
                if score < 0.55:
                    continue
                law = record["law"]
                entry = law.get("entry")
                if entry is not None and entry not in seen_law_ids:
                    seen_law_ids.add(entry)
                    result_laws.append(
                        {
                            "id": law.get("id"),
                            "entry": entry,
                            "description": law.get("description", ""),
                            "disputes": law.get("disputes", []),
                            "judge_dep": law.get("judge_dep", []),
                            "related_laws": law.get("related_laws", []),
                            "similarity": score,
                        }
                    )
        except Exception as e:
            print(f"Neo4j law search error: {e}")

    return result_laws


def query_similar_laws(dispute_list, top_k=1):
    """
    Query law nodes related to the most similar crime nodes based on a list of crime descriptions.

    Args:
        dispute_list (list[str]): List of crime descriptions as strings.
        top_k (int): Number of top similar crime nodes to retrieve per crime description.

    Returns:
        list[dict]: List of law nodes with their details, deduplicated.
    """
    db = GraphDBManager.get_db()
    result_laws = []
    seen_law_ids = set()  # For deduplication of law nodes

    for dispute in dispute_list:
        # Convert crime description to embedding
        dispute_embedding = get_embedding(dispute)
        if dispute_embedding is None:
            continue  # Skip if embedding generation fails

        # Query the most similar crime nodes
        dispute_results = db.find_similar_nodes(dispute_embedding, "Disputes", top_k=top_k)

        # Process each similar crime node
        for dispute_record in dispute_results:
            dispute_id = dispute_record["id"]
            dispute_similarity = dispute_record["similarity"]

            # Query law nodes related to this crime node
            # Find all Law nodes that point to this issue node
            for node_id, node_info in db.nodes_data.items():
                if node_info["type"] == "Laws":
                    neighbors = db.get_neighbors(node_id, "RELATED_DISPUTE")
                    if dispute_id in neighbors:
                        law_data = node_info["data"]
                        law_id = node_id
                        if law_id not in seen_law_ids:
                            result_laws.append(
                                {
                                    "id": law_id,
                                    "entry": law_data.get("entry"),
                                    "description": law_data.get("description"),
                                    "disputes": law_data.get("disputes"),
                                    "judge_dep": law_data.get("judge_dep"),
                                    "related_laws": law_data.get("related_laws"),
                                    "insights": law_data.get("insights", ""),
                                    "dispute_similarity": dispute_similarity,
                                }
                            )
                            seen_law_ids.add(law_id)

    # Sort results by crime similarity (descending) and assign ranks
    result_laws = sorted(result_laws, key=lambda x: x["dispute_similarity"], reverse=True)
    for rank, law in enumerate(result_laws, 1):
        law["rank"] = rank

    return result_laws
