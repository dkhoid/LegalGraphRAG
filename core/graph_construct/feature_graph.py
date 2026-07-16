import numpy as np
import requests
import re
from .graph_db import GraphDBManager
from tqdm import tqdm
from core.utils.logger import logger


def get_embedding(text):
    import os

    url = os.getenv("embedding_api_url", "http://localhost:11434/api/embed")
    model = os.getenv("embedding_model", "bge-m3")
    api_key = os.getenv("OPENAI_API_KEY", os.getenv("api_key", ""))

    headers = {"Content-Type": "application/json"}
    if "openai.com" in url or api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        # text-embedding-3-small max length is 8192 tokens.
        # Using ~8000 characters as a safe truncation limit to prevent 400 Bad Request.
        if len(text) > 8000:
            text = text[:8000]

    data = {"model": model, "input": text}

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        result = response.json()
        if "data" in result and len(result["data"]) > 0:
            return result["data"][0].get("embedding", [])
        return result.get("embeddings", [[]])[0]
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def summarize_texts(model, text):
    from core.prompt import get_prompt

    return model.generate_response(
        get_prompt("SUMMARIZE_TEXTS_PROMPT") + get_prompt("SUMMARIZE_TEXTS_INPUT_PREFIX") + text,
        max_length=512,
    ).strip()


def rerank_clusters(model, clusters, query_text):
    from core.prompt import get_prompt

    cluster_summaries = "\n".join([f"code{c['code']}：{c['summary']}\n" for c in clusters])
    prompt = get_prompt("RERANK_CLUSTERS_PROMPT_TEMPLATE").format(
        cluster_summaries=cluster_summaries, query_text=query_text
    )
    response = model.generate_response(prompt, max_length=512)
    match = re.search(r"rank: \[([\d,]+)\]", response)
    # print(f"Model response: {response}")
    if match:
        ranked_codes = [int(code) for code in match.group(1).split(",")]
        return ranked_codes
    return [0]


def rerank(model, query_text, neighbors):
    from core.prompt import get_prompt

    if not neighbors:
        return []

    neighbor_summaries = "\n".join([f"code{n['rank']}：{n['description']}\n" for n in neighbors])
    prompt = get_prompt("RERANK_PROMPT_TEMPLATE").format(
        neighbor_summaries=neighbor_summaries, query_text=query_text
    )
    response = model.generate_response(prompt, max_length=512)
    try:
        first_bracket = response.find("[")
        last_bracket = response.rfind("]")
        if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
            list_str = response[first_bracket : last_bracket + 1]
            ranked_indices = eval(list_str)
        else:
            ranked_indices = []
    except Exception as e:
        print(f"Error parsing response: {e}")
        ranked_indices = []
    if not ranked_indices:
        return neighbors[:3]
    neighbors = [n for n in neighbors if n["rank"] in ranked_indices]
    neighbors = sorted(neighbors, key=lambda x: ranked_indices.index(x["rank"]))
    return neighbors


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

    # Create Case-to-Law relationships (based on entry matching)
    # Retrieve all Case nodes and their law attributes from the graph
    case_nodes = db.get_nodes_by_type("Cases")

    for case_node in tqdm(case_nodes, desc="Linking cases to laws"):
        case_id = case_node["id"]
        law_entries = case_node.get("law")

        if not law_entries:
            continue

        for law_entry in law_entries:
            # Find the matching Law node
            law_found = False
            for node_id, node_info in db.nodes_data.items():
                if node_info["type"] == "Laws" and node_info["data"].get("entry") == int(law_entry):
                    db.add_edge(case_id, node_id, "RELATES_TO_LAW")
                    law_found = True
                    break

            if not law_found:
                print(f"Warning: Law node not found, entry={law_entry}")

    # Create Law-to-Issue relationships (based on issue description matching)
    # Retrieve all Law nodes and their crimes/issues attributes from the graph
    law_nodes = db.get_nodes_by_type("Laws")

    for law_node in tqdm(law_nodes, desc="Linking laws to crimes"):
        law_id = law_node["id"]
        dispute_descriptions = law_node.get("disputes")

        if not dispute_descriptions:
            continue

        for dispute_desc in dispute_descriptions:
            # Check if relationship already exists
            existing_neighbors = db.get_neighbors(law_id, "RELATED_DISPUTE")
            if existing_neighbors:
                # Check if a matching issue node already linked
                found = False
                for dispute_id in existing_neighbors:
                    dispute_data = db.get_node(dispute_id)
                    if dispute_data and dispute_data.get("description") == dispute_desc:
                        found = True
                        break
                if found:
                    continue

            # Attempt exact match
            dispute_found = False
            for node_id, node_info in db.nodes_data.items():
                if (
                    node_info["type"] == "Disputes"
                    and node_info["data"].get("description") == dispute_desc
                ):
                    db.add_edge(law_id, node_id, "RELATED_DISPUTE", {"match_type": "exact"})
                    dispute_found = True
                    break

            # Attempt fuzzy match if exact match failed
            if not dispute_found:
                for node_id, node_info in db.nodes_data.items():
                    if node_info["type"] == "Disputes":
                        desc = node_info["data"].get("description", "")
                        if dispute_desc in desc or desc in dispute_desc:
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
            if entry is not None and int(entry) <= 101:
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
                db._update_vector_index(node_type)

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
            case_embeddings.append(np.array(emb))
            case_ids.append(node["id"])

    if len(case_embeddings) < 2:
        return

    # Normalize embeddings for fast cosine similarity via dot product
    case_embeddings = np.array(case_embeddings)
    norms = np.linalg.norm(case_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10  # Avoid division by zero
    normalized_embeddings = case_embeddings / norms

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

    for community_id in tqdm(community_ids, desc="Creating clusters"):
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
        descriptions = [node["description"] for node in important_nodes[:10]]

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
        for node_id, node_info in db.nodes_data.items():
            if (
                node_info["type"] == "Cases"
                and node_info["data"].get("communityId") == community_id
            ):
                db.add_edge(node_id, cluster_id, "BELONGS_TO")


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
            neighbors.append(
                {
                    "id": case_data["id"],
                    "description": case_data.get("description", ""),
                    "caseId": case_data.get("caseId", ""),
                    "similarity": similarity,
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
        neighbors.append(
            {
                "id": record["id"],
                "description": record.get("description", ""),
                "caseId": record.get("caseId", ""),
                "similarity": record["similarity"],
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

    db = GraphDBManager.get_db()
    neighbor_results = db.find_similar_nodes(query_embedding, "Cases", top_k=top_k)

    if not neighbor_results:
        return []

    neighbors = []
    for record in neighbor_results:
        neighbors.append(
            {
                "id": record["id"],
                "description": record.get("description", ""),
                "caseId": record.get("caseId", ""),
                "similarity": record["similarity"],
            }
        )
    neighbors = sorted(neighbors, key=lambda x: x["similarity"], reverse=True)
    for ids, neighbor in enumerate(neighbors):
        neighbor["rank"] = ids + 1

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

    db = GraphDBManager.get_db()
    law_results = db.find_similar_nodes(query_embedding, "Laws", top_k=top_k)

    result_laws = []
    seen_law_ids = set()  # For deduplication of law nodes

    for law_record in law_results:
        entry = law_record.get("entry")
        if entry is not None and entry not in seen_law_ids:
            result_laws.append(
                {
                    "id": law_record.get("id"),
                    "entry": entry,
                    "description": law_record.get("description", ""),
                    "disputes": law_record.get("disputes", []),
                    "judge_dep": law_record.get("judge_dep", []),
                    "related_laws": law_record.get("related_laws", []),
                    "similarity": law_record["similarity"],
                }
            )
            seen_law_ids.add(entry)

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
        dispute_embedding = get_embedding(crime)
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


def update_insights_in_graph(law_id, insights):
    db = GraphDBManager.get_db()
    db.update_node(law_id, {"insights": insights})


def construct_feature_graph(model, nodes_data):
    import concurrent.futures

    GraphDBManager.initialize()

    case_nodes_data, law_nodes_data, dispute_nodes_data = (
        nodes_data["case"],
        nodes_data["law"],
        nodes_data["dispute"],
    )

    def process_nodes(nodes_data_list, desc):
        def _get_emb(item):
            i, node = item
            emb = get_embedding(node["description"])
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

    # Run KNN and clustering
    run_knn(top_k=3)
    create_clusters(model)

    # Sync to Neo4j
    try:
        from core.graph_construct.neo4j_manager import neo4j_manager

        db = GraphDBManager.get_db()
        neo4j_manager.sync_from_memory_graph(db)
    except Exception as e:
        logger.error(f"Failed to sync graph to Neo4j: {e}")
