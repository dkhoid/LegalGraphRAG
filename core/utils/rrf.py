"""RRF (Reciprocal Rank Fusion) utility for combining multiple ranked retrieval results.

Reference: Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion outperforms Condorcet
and individual Rank Learning Methods" (SIGIR 2009).
"""

from typing import Any


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    k: int = 60,
    id_key: str = "id",
) -> list[dict[str, Any]]:
    """Fuse multiple ranked lists using the RRF algorithm.

    Each document's score is the sum of 1/(k + rank) across all lists it appears in.
    Documents appearing consistently near the top of multiple lists score highest.

    Args:
        ranked_lists: List of lists, each ordered best-first (index 0 = best).
        k: Smoothing constant. Default 60 per the original paper. Higher k reduces
           the advantage of top-ranked items.
        id_key: Primary dict key used as unique identifier per item.
                Falls back to "entry", then "caseId", then object id().

    Returns:
        Single merged list, ordered by RRF score descending (best first).
        Items preserve their original dict structure.

    Examples:
        >>> l1 = [{"id": "A"}, {"id": "B"}, {"id": "C"}]
        >>> l2 = [{"id": "B"}, {"id": "A"}, {"id": "D"}]
        >>> result = reciprocal_rank_fusion([l1, l2])
        >>> result[0]["id"] in ("A", "B")
        True
    """
    scores: dict[str, dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        if not ranked_list:
            continue
        for rank, item in enumerate(ranked_list, start=1):
            # Resolve item identifier – law nodes may use "entry", case nodes "caseId"
            item_id = str(item.get(id_key) or item.get("entry") or item.get("caseId") or id(item))
            if item_id not in scores:
                scores[item_id] = {"item": item, "rrf_score": 0.0}
            scores[item_id]["rrf_score"] += 1.0 / (k + rank)

    return [entry["item"] for entry in sorted(scores.values(), key=lambda x: -x["rrf_score"])]
