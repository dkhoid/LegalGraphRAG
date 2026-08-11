from typing import Dict, Any, Tuple, List
from core.retriever.base_retriever import BaseRetriever
from core.graph_construct.feature_graph import query_similar_nodes_naive, query_similar_laws_naive
from rank_bm25 import BM25Okapi
from core.utils.rrf import reciprocal_rank_fusion


def concat_feature_descriptions(description: Dict[str, Any]) -> str:
    res = ""
    res += "Parties Info: " + ", ".join(description.get("parties_info", [])) + ". "
    res += "Dispute Acts: " + ", ".join(description.get("dispute_acts", [])) + ". "
    res += "Subject Matter: " + ", ".join(description.get("subject_matter", [])) + ". "
    res += "Fault and Evidence: " + ", ".join(description.get("fault_and_evidence", [])) + ". "
    return res


class VectorRetriever(BaseRetriever):
    """
    Retriever that uses Hybrid Search (BM25 + Vector Cosine Similarity).
    """

    def __init__(self, model):
        super().__init__(model)
        self._bm25 = None
        self._law_mapping = None

    def _init_bm25(self, law_to_dispute):
        if self._bm25 is not None:
            return

        corpus = []
        self._law_mapping = []
        for law in law_to_dispute:
            for item in law.get("items", [law]):
                text = item.get("text", "")
                if text:
                    corpus.append(text)
                    self._law_mapping.append(
                        {
                            "id": law["id"],
                            "entry": str(law["id"]),
                            "text": text,
                            "description": text,
                            "dispute": item.get("dispute", []),
                            "judge_dep": item.get("judge_dep", []),
                            "related_laws": item.get("related_laws", []),
                        }
                    )

        tokenized_corpus = [doc.lower().split(" ") for doc in corpus]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(
        self,
        case: Dict[str, Any],
        law_to_dispute: List[Dict[str, Any]],
        cases_db: List[Dict[str, Any]],
        retrieve_config: Dict[str, Any] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:

        features = case.get("feature", {})
        query_text = concat_feature_descriptions(features)

        # Determine top_k from config or use default 5
        top_k = 7
        if retrieve_config and "direct_retrieve_top_k" in retrieve_config:
            top_k = retrieve_config["direct_retrieve_top_k"]

        # 1. Retrieve similar cases (Vector Search)
        retrieved_facts = query_similar_nodes_naive(self.model, query_text, top_k=top_k)

        # 2. Retrieve similar laws (Hybrid Search: BM25 + Vector Search)
        # 2a. Vector Search – returns ordered list (best first)
        vector_laws_raw = query_similar_laws_naive(query_text, top_k=top_k * 2)

        # 2b. BM25 Search
        self._init_bm25(law_to_dispute)
        tokenized_query = query_text.lower().split(" ")
        bm25_scores = self._bm25.get_scores(tokenized_query)

        top_n = top_k * 2
        top_bm25_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:top_n]
        bm25_laws_raw = [self._law_mapping[i] for i in top_bm25_indices if bm25_scores[i] > 0]

        # 2c. Collect laws attached to retrieved cases
        case_laws_raw = []
        for item in retrieved_facts:
            for db_case in cases_db:
                if db_case["id"] == item["caseId"]:
                    item["dispute"] = db_case.get("dispute", [])
                    item["law"] = db_case.get("law", [])
                    for law_entry in db_case.get("law", []):
                        case_laws_raw.append({"entry": str(law_entry), "id": str(law_entry)})
                    break

        # 2d. RRF fusion across all three sources
        fused_law_refs = reciprocal_rank_fusion(
            [vector_laws_raw, bm25_laws_raw, case_laws_raw],
            id_key="entry",
        )

        # Reconstruct full law node data from fused entry IDs
        seen_entries: set = set()
        final_retrieved_laws = []
        for ref in fused_law_refs:
            entry_id = str(ref.get("entry") or ref.get("id", ""))
            if not entry_id or entry_id in seen_entries:
                continue
            seen_entries.add(entry_id)

            try:
                if int(entry_id) < 102:  # Filter out generic introductory articles
                    continue
            except ValueError:
                pass

            try:
                for item in law_to_dispute:
                    if str(item["id"]) == entry_id:
                        for entry in item.get("items", [item]):
                            final_retrieved_laws.append(
                                {
                                    "id": item["id"],
                                    "entry": str(item["id"]),
                                    "text": entry.get("text", ""),
                                    "description": entry.get("text", ""),
                                    "dispute": entry.get("dispute", []),
                                    "judge_dep": entry.get("judge_dep", []),
                                    "related_laws": entry.get("related_laws", []),
                                }
                            )
                        break
            except IndexError:
                continue

        original_retrieved_res = {
            "method": "hybrid_bm25_vector",
            "fusion_method": "rrf",
            "top_k_used": top_k,
            "vector_laws_count": len(vector_laws_raw),
            "bm25_laws_count": len(bm25_laws_raw),
            "laws_found_count": len(final_retrieved_laws),
        }

        return original_retrieved_res, final_retrieved_laws, retrieved_facts
