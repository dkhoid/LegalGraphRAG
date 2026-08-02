from typing import Dict, Any, Tuple, List
from core.retriever.base_retriever import BaseRetriever
from core.graph_construct.feature_graph import query_similar_nodes
from core.prompt import get_prompt


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
        retrieved_laws = []
        seen_case_ids = set()
        seen_law_ids = set()

        from core.graph_construct.neo4j_manager import neo4j_manager

        if not neo4j_manager.driver:
            from core.utils.logger import logger

            logger.error("Neo4j driver not initialized. GraphRetriever failed.")
            return {}, [], []

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
                    for l in record["laws"]:
                        if l and l["id"] not in seen_law_ids:
                            seen_law_ids.add(l["id"])
                            retrieved_laws.append(
                                {
                                    "id": l["id"],
                                    "entry": l.get("entry"),
                                    "description": l.get("description"),
                                    "judge_dep": l.get("judge_dep", "[]"),
                                    "related_laws": l.get("related_laws", "[]"),
                                }
                            )

            # 2. Fulltext Search for Cases (BM25 replacement)
            # Neo4j fulltext requires lucene query syntax. We split words and use OR operator.
            if retrieve_config.get("direct_retrieve", True):
                import re

                clean_query = re.sub(r"[^\w\s]", "", query_text)
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
                            for l in record["laws"]:
                                if l and l["id"] not in seen_law_ids:
                                    seen_law_ids.add(l["id"])
                                    retrieved_laws.append(
                                        {
                                            "id": l["id"],
                                            "entry": l.get("entry"),
                                            "description": l.get("description"),
                                            "judge_dep": l.get("judge_dep", "[]"),
                                            "related_laws": l.get("related_laws", "[]"),
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

        for law in augmented_laws:
            if law["id"] not in seen_law_ids:
                seen_law_ids.add(law["id"])
                retrieved_laws.append(law)

        # 4. Safe parsing
        final_retrieved_laws = []
        import ast

        for law in retrieved_laws:
            try:
                law["judge_dep"] = ast.literal_eval(str(law.get("judge_dep", "[]")))
            except Exception:
                law["judge_dep"] = []
            try:
                law["related_laws"] = ast.literal_eval(str(law.get("related_laws", "[]")))
            except Exception:
                law["related_laws"] = []
            final_retrieved_laws.append(law)

        return original_retrieved_res, final_retrieved_laws, retrieved_facts
