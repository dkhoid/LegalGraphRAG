from typing import Dict, Any, Tuple, List
from core.retriever.base_retriever import BaseRetriever
from core.prompt import get_prompt
from core.utils.rrf import reciprocal_rank_fusion
from core.utils.legal_text import preprocess_for_retrieval
from core.utils.formatting import concat_feature_descriptions


class GraphRetriever(BaseRetriever):
    """
    Retriever that uses Graph Traversal (Community -> Cases -> Laws) and Direct similarity.
    This is the core LegalGraphRAG approach.
    """

    def _retrieve_law_augment(self, case: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Augment retrieval by using LLM to extract facts and directly query laws"""
        from core.graph_construct.graph_search import query_similar_laws

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
        if isinstance(features, dict) and features:
            query_text = concat_feature_descriptions(features)
        else:
            query_text = case.get("description", "") or case.get("fact", "")

        # Query expansion: append dispute_acts and subject_matter from feature extraction
        # This dramatically improves BM25 and embedding recall for domain-specific terms
        # e.g. 'làm thêm giờ', 'tiền lương', 'sa thải' are critical for law matching
        _expansion_parts: List[str] = []
        if isinstance(features, dict):
            for field in ("dispute_acts", "subject_matter", "parties_info"):
                val = features.get(field, [])
                if isinstance(val, list):
                    _expansion_parts.extend([str(v) for v in val if v])
                elif isinstance(val, str) and val:
                    _expansion_parts.append(val)
        if _expansion_parts:
            query_text = query_text + " " + " ".join(_expansion_parts)

        # Get query embedding
        from core.graph_construct.llm_utils import get_embedding, generate_hyde_query

        use_hyde = retrieve_config.get("use_hyde", True)
        if use_hyde:
            try:
                hyde_query = generate_hyde_query(self.model, query_text)
                from core.utils.logger import logger

                logger.info("HyDE Query generated.")
                query_embedding = get_embedding(hyde_query)
            except Exception as e:
                from core.utils.logger import logger

                logger.warning(f"HyDE generation failed: {e}. Falling back to original query_text.")
                query_embedding = get_embedding(query_text)
        else:
            query_embedding = get_embedding(query_text)

        original_retrieved_res = {}
        retrieved_facts = []
        seen_case_ids = set()

        # Separate ordered lists for Weighted RRF fusion
        direct_laws: List[Dict[str, Any]] = []
        graph_laws: List[Dict[str, Any]] = []
        bm25_laws: List[Dict[str, Any]] = []
        direct_law_ids: set = set()
        graph_law_ids: set = set()
        bm25_law_ids: set = set()

        from core.utils.legal_text import get_hierarchy_boost
        from core.graph_construct.neo4j_manager import neo4j_manager

        if not neo4j_manager.driver:
            from core.utils.logger import logger

            logger.warning(
                "Neo4j driver not initialized. Falling back to local in-memory graph search."
            )
            from core.graph_construct.graph_search import (
                search_similar_nodes_direct,
                search_similar_nodes_top,
            )

            top_k = retrieve_config.get("top_retrieve_top_k", 5)
            top_retrieve = retrieve_config.get("top_retrieve", True)
            direct_retrieve = retrieve_config.get("direct_retrieve", True)

            if top_retrieve and direct_retrieve:
                _, c_top, l_top = search_similar_nodes_top(
                    self.model, query_embedding, query_text, top_k=top_k
                )
                c_dir, l_dir = search_similar_nodes_direct(
                    self.model, query_embedding, query_text, top_k=top_k
                )
                seen_ids = set()
                for c in c_top + c_dir:
                    if c["id"] not in seen_ids:
                        seen_ids.add(c["id"])
                        retrieved_facts.append(c)
                graph_laws = l_top
                direct_laws = l_dir
            elif top_retrieve:
                _, c_top, l_top = search_similar_nodes_top(
                    self.model, query_embedding, query_text, top_k=top_k
                )
                retrieved_facts = c_top
                graph_laws = l_top
            elif direct_retrieve:
                c_dir, l_dir = search_similar_nodes_direct(
                    self.model, query_embedding, query_text, top_k=top_k
                )
                retrieved_facts = c_dir
                direct_laws = l_dir
        else:
            import concurrent.futures

            top_k = retrieve_config.get("top_retrieve_top_k", 5)
            top_k_bm25 = retrieve_config.get("direct_retrieve_top_k", 5)
            min_traversal_sim = retrieve_config.get("min_traversal_similarity", 0.40)

            # 1a. Worker: Case vector search + Graph Traversal
            def _fetch_case_graph():
                local_cases = []
                local_laws = []
                if not (query_embedding and retrieve_config.get("top_retrieve", True)):
                    return local_cases, local_laws
                vector_query = """
                CALL db.index.vector.queryNodes('case_embeddings', $top_k, $query_embedding)
                YIELD node AS case, score
                OPTIONAL MATCH (case)-[:RELATES_TO_LAW]->(l:Laws)
                RETURN case, collect(l) AS laws, score
                ORDER BY score DESC
                """
                try:
                    with neo4j_manager.driver.session() as s:
                        res = s.run(vector_query, top_k=top_k, query_embedding=query_embedding)
                        for record in res:
                            c = record["case"]
                            local_cases.append(
                                {
                                    "id": c["id"],
                                    "caseId": c.get("caseId"),
                                    "description": c.get("description"),
                                    "dispute": c.get("dispute", []),
                                    "law": c.get("law", []),
                                }
                            )
                            for law_node in record["laws"]:
                                if law_node:
                                    law_emb = law_node.get("embedding")
                                    if law_emb and min_traversal_sim > 0 and query_embedding:
                                        import numpy as np

                                        emb_arr = np.array(law_emb, dtype=np.float32)
                                        q_arr = np.array(query_embedding, dtype=np.float32)
                                        sim = float(
                                            np.dot(q_arr, emb_arr)
                                            / (
                                                np.linalg.norm(q_arr) * np.linalg.norm(emb_arr)
                                                + 1e-9
                                            )
                                        )
                                        if sim < min_traversal_sim:
                                            continue
                                    local_laws.append(
                                        {
                                            "id": law_node["id"],
                                            "entry": law_node.get("entry"),
                                            "description": law_node.get("description"),
                                            "judge_dep": law_node.get("judge_dep", "[]"),
                                            "related_laws": law_node.get("related_laws", "[]"),
                                            "_embedding": law_node.get("embedding"),
                                        }
                                    )
                except Exception as e:
                    from core.utils.logger import logger

                    logger.warning(f"Case vector search error: {e}")
                return local_cases, local_laws

            # 1b. Worker: Direct Law Vector Search
            def _fetch_direct_laws():
                local_laws = []
                if not (query_embedding and retrieve_config.get("direct_retrieve", True)):
                    return local_laws
                law_vector_query = """
                CALL db.index.vector.queryNodes('law_embeddings', $top_k, $query_embedding)
                YIELD node AS law, score
                RETURN law, score
                ORDER BY score DESC
                """
                try:
                    with neo4j_manager.driver.session() as s:
                        res = s.run(law_vector_query, top_k=top_k, query_embedding=query_embedding)
                        for record in res:
                            law_node = record["law"]
                            if law_node:
                                local_laws.append(
                                    {
                                        "id": law_node["id"],
                                        "entry": law_node.get("entry"),
                                        "description": law_node.get("description"),
                                        "judge_dep": law_node.get("judge_dep", "[]"),
                                        "related_laws": law_node.get("related_laws", "[]"),
                                        "_embedding": law_node.get("embedding"),
                                    }
                                )
                except Exception as e:
                    from core.utils.logger import logger

                    logger.warning(f"Direct law vector search failed: {e}")
                return local_laws

            # 2. Worker: Fulltext BM25 Search
            def _fetch_fulltext():
                local_cases = []
                local_laws = []
                if not retrieve_config.get("direct_retrieve", True):
                    return local_cases, local_laws
                import re

                expand_abbrev = retrieve_config.get("expand_abbreviations", True)
                bm25_query_text = preprocess_for_retrieval(query_text, expand=expand_abbrev)
                clean_query = re.sub(r"[^\w\s]", "", bm25_query_text)
                words = clean_query.split()
                if not words:
                    return local_cases, local_laws
                lucene_query = " OR ".join([f"{w}" for w in words[:10]])

                try:
                    with neo4j_manager.driver.session() as s:
                        # 2a. Case fulltext
                        c_res = s.run(
                            "CALL db.index.fulltext.queryNodes('case_fulltext', $lucene_query, {limit: $top_k}) YIELD node AS case, score OPTIONAL MATCH (case)-[:RELATES_TO_LAW]->(l:Laws) RETURN case, score, collect(l) AS laws ORDER BY score DESC",
                            lucene_query=lucene_query,
                            top_k=top_k_bm25,
                        )
                        for record in c_res:
                            c = record["case"]
                            local_cases.append(
                                {
                                    "id": c["id"],
                                    "caseId": c.get("caseId"),
                                    "description": c.get("description"),
                                    "dispute": c.get("dispute", []),
                                    "law": c.get("law", []),
                                }
                            )
                            for law_node in record["laws"]:
                                if law_node:
                                    local_laws.append(
                                        {
                                            "id": law_node["id"],
                                            "entry": law_node.get("entry"),
                                            "description": law_node.get("description"),
                                            "judge_dep": law_node.get("judge_dep", "[]"),
                                            "related_laws": law_node.get("related_laws", "[]"),
                                            "_embedding": law_node.get("embedding"),
                                        }
                                    )
                        # 2b. Direct law fulltext
                        l_res = s.run(
                            "CALL db.index.fulltext.queryNodes('law_fulltext', $lucene_query, {limit: $top_k}) YIELD node AS law, score RETURN law, score ORDER BY score DESC",
                            lucene_query=lucene_query,
                            top_k=top_k_bm25,
                        )
                        for record in l_res:
                            law_node = record["law"]
                            if law_node:
                                local_laws.append(
                                    {
                                        "id": law_node["id"],
                                        "entry": law_node.get("entry"),
                                        "description": law_node.get("description"),
                                        "judge_dep": law_node.get("judge_dep", "[]"),
                                        "related_laws": law_node.get("related_laws", "[]"),
                                        "_embedding": law_node.get("embedding"),
                                    }
                                )
                except Exception as e:
                    from core.utils.logger import logger

                    logger.warning(f"Fulltext search error: {e}")
                return local_cases, local_laws

            # Execute concurrent queries in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                f_case_graph = executor.submit(_fetch_case_graph)
                f_direct_laws = executor.submit(_fetch_direct_laws)
                f_fulltext = executor.submit(_fetch_fulltext)

                case_graph_cases, case_graph_laws = f_case_graph.result()
                raw_direct_laws = f_direct_laws.result()
                fulltext_cases, fulltext_laws = f_fulltext.result()

            # Process Cases
            for c in case_graph_cases + fulltext_cases:
                if c["id"] not in seen_case_ids:
                    seen_case_ids.add(c["id"])
                    retrieved_facts.append(c)

            # Process Direct Laws
            for law_item in raw_direct_laws:
                if law_item["id"] not in direct_law_ids:
                    direct_law_ids.add(law_item["id"])
                    direct_laws.append(law_item)

            # Process Graph Laws (from case traversal)
            for law_item in case_graph_laws:
                if law_item["id"] not in graph_law_ids:
                    graph_law_ids.add(law_item["id"])
                    graph_laws.append(law_item)

            # Process BM25 Laws
            for law_item in fulltext_laws:
                if law_item["id"] not in bm25_law_ids:
                    bm25_law_ids.add(law_item["id"])
                    bm25_laws.append(law_item)

        if not retrieved_facts:
            return {}, [], []

        # 3. Augment Laws using LLM if configured with Smart Gating
        augmented_laws = []
        if retrieve_config.get("augment_retrieve", False):
            smart_gate = retrieve_config.get("smart_augment_gating", True)
            if smart_gate and (len(direct_laws) >= 3 or len(graph_laws) >= 3):
                pass  # Skip LLM call when high confidence direct/graph candidates already exist
            else:
                augmented_laws = self._retrieve_law_augment(case)
                original_retrieved_res["augmented"] = augmented_laws

        # 4. Weighted RRF fusion: merge direct, graph, BM25, and augmented law lists
        # Weights: [Direct Law Vector = 1.5, Graph Traversal = 1.2, BM25 = 0.8, Augmented = 0.5]
        rrf_k = retrieve_config.get("rrf_k", 60)
        rrf_weights = retrieve_config.get("rrf_weights", [1.5, 1.2, 0.8, 0.5])
        codex_boost = retrieve_config.get("codex_boost_factor", 1.35)

        fused_laws = reciprocal_rank_fusion(
            [direct_laws, graph_laws, bm25_laws, augmented_laws],
            k=rrf_k,
            id_key="id",
            weights=rrf_weights,
        )

        # Apply Legal Hierarchy Primacy Boost (Bộ luật Quốc hội > Nghị định > Thông tư)
        if codex_boost > 1.0:
            for law in fused_laws:
                entry = str(law.get("entry", ""))
                boost = get_hierarchy_boost(entry, default_boost=codex_boost)
                law["_rrf_score"] = law.get("_rrf_score", 0.0) * boost
            fused_laws.sort(key=lambda x: x.get("_rrf_score", 0.0), reverse=True)
        original_retrieved_res["fusion_method"] = "weighted_rrf"
        original_retrieved_res["rrf_k"] = rrf_k
        original_retrieved_res["direct_laws_count"] = len(direct_laws)
        original_retrieved_res["graph_laws_count"] = len(graph_laws)
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
        import json

        for law in fused_laws:
            jd = law.get("judge_dep")
            if isinstance(jd, list):
                law["judge_dep"] = jd
            elif isinstance(jd, str):
                try:
                    law["judge_dep"] = (
                        json.loads(jd) if jd.strip().startswith("[") else ast.literal_eval(jd)
                    )
                except Exception:
                    law["judge_dep"] = []
            else:
                law["judge_dep"] = []

            rl = law.get("related_laws")
            if isinstance(rl, list):
                law["related_laws"] = rl
            elif isinstance(rl, str):
                try:
                    law["related_laws"] = (
                        json.loads(rl) if rl.strip().startswith("[") else ast.literal_eval(rl)
                    )
                except Exception:
                    law["related_laws"] = []
            else:
                law["related_laws"] = []

            final_retrieved_laws.append(law)

        # 6. Cross-Encoder Reranker pass (contextual semantic scoring)
        use_reranker = retrieve_config.get("use_reranker", True)
        if use_reranker and final_retrieved_laws:
            try:
                from core.retriever.reranker import get_reranker

                reranker = get_reranker()
                rerank_top_k = retrieve_config.get("rerank_top_k", len(final_retrieved_laws))
                final_retrieved_laws = reranker.rerank(
                    query_text, final_retrieved_laws, top_k=rerank_top_k
                )
                original_retrieved_res["reranker_applied"] = True
            except Exception as e:
                from core.utils.logger import logger

                logger.warning(f"Cross-encoder reranking failed: {e}")

        # 7. MMR diversity pass – reduce redundant laws before judge
        use_mmr = retrieve_config.get("use_mmr", False)
        if use_mmr and final_retrieved_laws:
            try:
                from core.utils.mmr import maximal_marginal_relevance

                mmr_k = retrieve_config.get("mmr_top_k", retrieve_config.get("max_judge_laws", 8))
                mmr_lambda = retrieve_config.get("mmr_lambda", 0.85)
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
