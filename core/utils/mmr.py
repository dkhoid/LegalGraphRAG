"""MMR – Maximal Marginal Relevance for diverse law selection.

Prevents sending redundant laws to judge_law. Balances:
  - Relevance: how similar the law is to the query
  - Diversity: how different the law is from already-selected laws

Reference: Carbonell & Goldstein, "The Use of MMR, Diversity-Based Reranking for
Reordering Documents and Producing Summaries" (SIGIR 1998).
"""

from __future__ import annotations
import math
from typing import Any


def _cos_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def maximal_marginal_relevance(
    query_vec: list[float],
    laws: list[dict[str, Any]],
    k: int = 8,
    lambda_: float = 0.5,
    embedding_key: str = "_embedding",
) -> list[dict[str, Any]]:
    """Select top-K diverse and relevant laws using MMR.

    Laws must have a pre-computed embedding stored under ``embedding_key``.
    Laws that lack an embedding are appended at the end (fallback).

    Args:
        query_vec: Query embedding vector.
        laws: List of law dicts, each ideally with ``embedding_key`` field.
        k: Number of laws to select.
        lambda_: Trade-off weight.
            1.0 = pure relevance (same as cosine ranking),
            0.0 = pure diversity (greedy anti-clustering).
            0.5 is recommended default.
        embedding_key: Key in each law dict that holds its embedding vector.

    Returns:
        Top-K laws ordered by MMR score (best first).
    """
    if not laws or not query_vec:
        return laws[:k]

    # Split laws into those with embeddings vs without
    with_emb = [(i, law) for i, law in enumerate(laws) if law.get(embedding_key)]
    without_emb = [law for law in laws if not law.get(embedding_key)]

    if not with_emb:
        # No embeddings available – fall back to original order
        return laws[:k]

    selected_indices: list[int] = []
    selected_vecs: list[list[float]] = []
    remaining = list(range(len(with_emb)))

    while len(selected_indices) < k and remaining:
        best_r_idx, best_score = None, float("-inf")

        for r_idx in remaining:
            _, law = with_emb[r_idx]
            vec = law[embedding_key]

            relevance = _cos_sim(query_vec, vec)

            # Max similarity to already-selected laws (0 if none selected yet)
            max_sim_to_selected = max((_cos_sim(vec, sv) for sv in selected_vecs), default=0.0)

            mmr_score = lambda_ * relevance - (1.0 - lambda_) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_r_idx = r_idx

        _, best_law = with_emb[best_r_idx]
        selected_indices.append(best_r_idx)
        selected_vecs.append(best_law[embedding_key])
        remaining.remove(best_r_idx)

    result = [with_emb[i][1] for i in selected_indices]

    # Pad with no-embedding laws if we still need more
    remaining_needed = k - len(result)
    if remaining_needed > 0:
        result.extend(without_emb[:remaining_needed])

    return result
