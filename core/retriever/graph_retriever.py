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

        try:
            first = response.find("[")
            last = response.rfind("]") + 1
            disputes = eval(response[first:last])
        except (ValueError, SyntaxError, Exception):
            return []

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
                "top_retrieve_top_k": 3,
                "direct_retrieve": True,
                "direct_retrieve_top_k": 3,
                "augment_retrieve": False,
            }

        features = case.get("feature", {})
        query_text = concat_feature_descriptions(features)

        # 1. Retrieve using Graph traversal
        original_retrieved_res, retrieved_facts, retrieved_laws = query_similar_nodes(
            self.model, query_text, retrieve_config
        )

        if not retrieved_facts:
            return {}, [], []

        # 2. Augment Laws using LLM if configured
        augmented_laws = []
        if retrieve_config.get("augment_retrieve", False):
            augmented_laws = self._retrieve_law_augment(case)
            original_retrieved_res["augmented"] = augmented_laws

        retrieved_laws = retrieved_laws + augmented_laws

        # 3. Associate facts with full db objects
        for item in retrieved_facts:
            for db_case in cases_db:
                if db_case["id"] == item["caseId"]:
                    item["dispute"] = db_case.get("dispute", [])
                    item["law"] = db_case.get("law", [])
                    break

        # 4. Deduplicate and reconstruct laws
        final_retrieved_laws = []
        seen_law_ids = set()
        for law in retrieved_laws:
            if law["id"] in seen_law_ids:
                continue
            seen_law_ids.add(law["id"])

            # Ensure eval parsing works safely
            try:
                law["judge_dep"] = eval(str(law.get("judge_dep", "[]")))
            except:
                law["judge_dep"] = []

            try:
                law["related_laws"] = eval(str(law.get("related_laws", "[]")))
            except:
                law["related_laws"] = []

            final_retrieved_laws.append(law)

        return original_retrieved_res, final_retrieved_laws, retrieved_facts
