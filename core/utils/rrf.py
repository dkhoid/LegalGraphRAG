"""RRF (Reciprocal Rank Fusion) utility for combining multiple ranked retrieval results.

Reference: Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion outperforms Condorcet
and individual Rank Learning Methods" (SIGIR 2009).
"""

from typing import Any, Dict, List


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    id_key: str = "id",
    weights: List[float] | None = None,
) -> List[Dict[str, Any]]:
    """Fuse multiple ranked lists using the RRF algorithm with optional list weights.

    Each document's score is the sum of weight * 1/(k + rank) across all lists it appears in.
    Documents appearing consistently near the top of high-weighted lists score highest.

    Args:
        ranked_lists: List of lists, each ordered best-first (index 0 = best).
        k: Smoothing constant. Default 60 per the original paper.
        id_key: Primary dict key used as unique identifier per item.
                Falls back to "entry", then "caseId", then object id().
        weights: Optional list of float weights matching ranked_lists. Default 1.0 each.

    Returns:
        Single merged list, ordered by RRF score descending (best first).
        Items preserve their original dict structure AND gain a ``_rrf_score``
        field that can be used for downstream threshold filtering (M2).
    """
    scores: Dict[str, Dict[str, Any]] = {}

    for list_idx, ranked_list in enumerate(ranked_lists):
        if not ranked_list:
            continue
        weight = weights[list_idx] if weights and list_idx < len(weights) else 1.0
        for rank, item in enumerate(ranked_list, start=1):
            item_id = str(item.get(id_key) or item.get("entry") or item.get("caseId") or id(item))
            if item_id not in scores:
                scores[item_id] = {"item": item, "rrf_score": 0.0}
            scores[item_id]["rrf_score"] += weight * (1.0 / (k + rank))

    result = []
    for entry in sorted(scores.values(), key=lambda x: -x["rrf_score"]):
        item = dict(entry["item"])  # copy to avoid mutating original
        item["_rrf_score"] = round(entry["rrf_score"], 6)  # M2: expose score
        result.append(item)
    return result
