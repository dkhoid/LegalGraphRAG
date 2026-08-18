from typing import Dict, Any, Tuple, List
from core.retriever.base_retriever import BaseRetriever
from core.prompt import get_prompt
from core.utils.rrf import reciprocal_rank_fusion
from core.utils.legal_text import preprocess_for_retrieval


def concat_feature_descriptions(description: Dict[str, Any]) -> str:
    res = ""
    res += "Parties Info: " + ", ".join(description.get("parties_info", [])) + ". "
    res += "Dispute Acts: " + ", ".join(description.get("dispute_acts", [])) + ". "
    res += "Subject Matter: " + ", ".join(description.get("subject_matter", [])) + ". "
    res += "Fault and Evidence: " + ", ".join(description.get("fault_and_evidence", [])) + ". "
    return res


class GraphRetriever(BaseRetriever):
    """
    Retriever that uses Graph Traversal (Community -> Cases -> Laws) and Direct similarity.
    This is the core LegalGraphRAG approach.
    """

    def _retrieve_law_augment(self, case: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Augment retrieval by using LLM to extract facts and directly query laws"""
        from core.graph_construct.feature_graph import query_similar_laws

        fact = case["description"][:1024]
        name = case["name"]

        prompt_text = get_prompt("RETRIEVE_LAW_PROMPT").format(name=name, fact=fact)
        response = self.model.generate_response(prompt_text, max_length=256)

        import json

        try:
            first = response.find("[")
            last = response.rfind("]") + 1
            array_str = response[first:last].replace("'", '"')  # Fix single quotes for JSON
            disputes = json.loads(array_str)
            if not isinstance(disputes, list):
                disputes = []
        except Exception:
            disputes = []

        laws = query_similar_laws(disputes, top_k=1)
        return laws

    def retrieve(
        self,
        case: Dict[str, Any],
        law_to_dispute: List[Dict[str, Any]],
        cases_db: List[Dict[str, Any]],
        retrieve_config: Dict[str, Any] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:

        if retrieve_config is None:
            # Default fallback config if none provided
            retrieve_config = {
                "top_retrieve": True,
                "top_retrieve_top_k": 5,
                "direct_retrieve": True,
                "direct_retrieve_top_k": 5,
                "augment_retrieve": False,
            }

        features = case.get("feature", {})
        query_text = concat_feature_descriptions(features)

        # Get query embedding
        from core.graph_construct.feature_graph import get_embedding

        query_embedding = get_embedding(query_text)

        original_retrieved_res = {}
        retrieved_facts = []
        seen_case_ids = set()

        # Separate ordered lists for RRF fusion (maintain retrieval order per source)
        vector_laws: List[Dict[str, Any]] = []
        bm25_laws: List[Dict[str, Any]] = []
        vector_law_ids: set = set()
        bm25_law_ids: set = set()

        from core.graph_construct.neo4j_manager import neo4j_manager

        if not neo4j_manager.driver:
            from core.utils.logger import logger

            logger.warning(
                "Neo4j driver not initialized. Falling back to local in-memory graph search."
            )
            from core.graph_construct.feature_graph import search_similar_nodes_direct

            top_k = retrieve_config.get("top_retrieve_top_k", 5)
            cases, laws = search_similar_nodes_direct(
                self.model, query_embedding, query_text, top_k=top_k
            )

            return {}, laws, cases

        with neo4j_manager.driver.session() as session:
            # 1. Vector Search for Cases
            if query_embedding and retrieve_config.get("top_retrieve", True):
                top_k = retrieve_config.get("top_retrieve_top_k", 5)
                vector_query = """
                CALL db.index.vector.queryNodes('case_embeddings', $top_k, $query_embedding)
                YIELD node AS case, score
                OPTIONAL MATCH (case)-[:RELATES_TO_LAW]->(l:Laws)
                RETURN case, collect(l) AS laws, score
                ORDER BY score DESC
                """
                results = session.run(vector_query, top_k=top_k, query_embedding=query_embedding)
                for record in results:
                    c = record["case"]
                    if c["id"] not in seen_case_ids:
                        seen_case_ids.add(c["id"])
                        retrieved_facts.append(
                            {
                                "id": c["id"],
                                "caseId": c.get("caseId"),
                                "description": c.get("description"),
                                "dispute": c.get("dispute", []),
                                "law": c.get("law", []),
                            }
                        )
                    for law_node in record["laws"]:
                        if law_node and law_node["id"] not in vector_law_ids:
                            vector_law_ids.add(law_node["id"])
                            vector_laws.append(
                                {
                                    "id": law_node["id"],
                                    "entry": law_node.get("entry"),
                                    "description": law_node.get("description"),
                                    "judge_dep": law_node.get("judge_dep", "[]"),
                                    "related_laws": law_node.get("related_laws", "[]"),
                                }
                            )

            # 2. Fulltext Search for Cases (BM25 replacement)
            # Neo4j fulltext requires lucene query syntax. We split words and use OR operator.
            if retrieve_config.get("direct_retrieve", True):
                import re

                # M7: Expand Vietnamese legal abbreviations before BM25 fulltext
                expand_abbrev = retrieve_config.get("expand_abbreviations", True)
                bm25_query_text = preprocess_for_retrieval(query_text, expand=expand_abbrev)

                clean_query = re.sub(r"[^\w\s]", "", bm25_query_text)
                words = clean_query.split()
                if words:
                    lucene_query = " OR ".join(
                        [f"*{w}*" for w in words[:10]]
                    )  # limit to first 10 words to avoid parsing errors
                    top_k_bm25 = retrieve_config.get("direct_retrieve_top_k", 5)
                    text_query = """
                    CALL db.index.fulltext.queryNodes('case_fulltext', $lucene_query, {limit: $top_k})
                    YIELD node AS case, score
                    OPTIONAL MATCH (case)-[:RELATES_TO_LAW]->(l:Laws)
                    RETURN case, collect(l) AS laws, score
                    ORDER BY score DESC
                    """
                    try:
                        results = session.run(
                            text_query, lucene_query=lucene_query, top_k=top_k_bm25
                        )
                        for record in results:
                            c = record["case"]
                            if c["id"] not in seen_case_ids:
                                seen_case_ids.add(c["id"])
                                retrieved_facts.append(
                                    {
                                        "id": c["id"],
                                        "caseId": c.get("caseId"),
                                        "description": c.get("description"),
                                        "dispute": c.get("dispute", []),
                                        "law": c.get("law", []),
                                    }
                                )
                            for law_node in record["laws"]:
                                if law_node and law_node["id"] not in bm25_law_ids:
                                    bm25_law_ids.add(law_node["id"])
                                    bm25_laws.append(
                                        {
                                            "id": law_node["id"],
                                            "entry": law_node.get("entry"),
                                            "description": law_node.get("description"),
                                            "judge_dep": law_node.get("judge_dep", "[]"),
                                            "related_laws": law_node.get("related_laws", "[]"),
                                        }
                                    )
                    except Exception as e:
                        print(f"Fulltext search error: {e}")

        if not retrieved_facts:
            return {}, [], []

        # 3. Augment Laws using LLM if configured
        augmented_laws = []
        if retrieve_config.get("augment_retrieve", False):
            augmented_laws = self._retrieve_law_augment(case)
            original_retrieved_res["augmented"] = augmented_laws

        # 4. RRF fusion: merge vector, BM25, and augmented law lists
        # M8: k is configurable (default 60, lower = sharper score spread for small datasets)
        rrf_k = retrieve_config.get("rrf_k", 60)
        fused_laws = reciprocal_rank_fusion(
            [vector_laws, bm25_laws, augmented_laws],
            k=rrf_k,
            id_key="id",
        )
        original_retrieved_res["fusion_method"] = "rrf"
        original_retrieved_res["rrf_k"] = rrf_k
        original_retrieved_res["vector_laws_count"] = len(vector_laws)
        original_retrieved_res["bm25_laws_count"] = len(bm25_laws)
        original_retrieved_res["augmented_laws_count"] = len(augmented_laws)

        # M2: Score threshold – drop laws below minimum RRF score
        min_score = retrieve_config.get("min_rrf_score", 0.0)  # 0.0 = disabled by default
        if min_score > 0:
            before = len(fused_laws)
            fused_laws = [
                law_item for law_item in fused_laws if law_item.get("_rrf_score", 1.0) >= min_score
            ]
            original_retrieved_res["threshold_filtered"] = before - len(fused_laws)

        # 5. Safe parsing of judge_dep and related_laws fields
        final_retrieved_laws = []
        import ast

        for law in fused_laws:
            try:
                law["judge_dep"] = ast.literal_eval(str(law.get("judge_dep", "[]")))
            except Exception:
                law["judge_dep"] = []
            try:
                law["related_laws"] = ast.literal_eval(str(law.get("related_laws", "[]")))
            except Exception:
                law["related_laws"] = []
            final_retrieved_laws.append(law)

        # M1: MMR diversity pass – reduce redundant laws before judge
        use_mmr = retrieve_config.get("use_mmr", False)
        if use_mmr and final_retrieved_laws:
            try:
                from core.utils.mmr import maximal_marginal_relevance

                mmr_k = retrieve_config.get("mmr_top_k", retrieve_config.get("max_judge_laws", 8))
                mmr_lambda = retrieve_config.get("mmr_lambda", 0.5)
                final_retrieved_laws = maximal_marginal_relevance(
                    query_vec=query_embedding,
                    laws=final_retrieved_laws,
                    k=mmr_k,
                    lambda_=mmr_lambda,
                )
                original_retrieved_res["mmr_applied"] = True
            except Exception as e:
                from core.utils.logger import logger

                logger.warning(f"MMR failed, using original order: {e}")

        return original_retrieved_res, final_retrieved_laws, retrieved_facts
