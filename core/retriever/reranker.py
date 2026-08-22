"""Cross-Encoder Reranker for post-retrieval law ranking.

Implements Stage-2 of the two-stage retrieval pipeline:
  Stage 1: RRF fusion (bi-encoder, fast) → top-50 candidates
  Stage 2: Cross-encoder (full attention over query+law pair) → top-K precise

The cross-encoder reads the query and each law description together, producing
a relevance score that captures fine-grained interaction between them – something
bi-encoders cannot do since they encode independently.

Reference: Nogueira & Cho, "Passage Re-ranking with BERT" (2019)
"""

from __future__ import annotations

from typing import Optional, Dict, List, Any
import logging

logger = logging.getLogger("LegalGraphRAG")


class CrossEncoderReranker:
    """Lightweight wrapper around a sentence-transformers CrossEncoder.

    Designed to be lazy-loaded so the model is not downloaded unless reranking
    is actually enabled via retrieve_config.

    Args:
        model_name: HuggingFace model ID for the cross-encoder.
            Defaults to a multilingual MiniLM model that works well on
            Vietnamese text without requiring task-specific fine-tuning.
        batch_size: Number of (query, law) pairs to score in one forward pass.
        top_k: Number of laws to keep after reranking.
    """

    # Good defaults for Vietnamese legal text:
    # - ms-marco-MiniLM-L-6-v2 : fast, English-centric but often ok for vi
    # - ms-marco-MiniLM-L-12-v2: slower, slightly better
    # - nreimers/mmarco-mMiniLMv2-L12-H384-v1: multilingual, recommended
    DEFAULT_MODEL = "nreimers/mmarco-mMiniLMv2-L12-H384-v1"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
        top_k: int = 8,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.top_k = top_k
        self._model = None  # lazy load

    def _load(self):
        """Lazy-load cross-encoder model (only on first call)."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            logger.info(f"CrossEncoder loaded: {self.model_name}")
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for cross-encoder reranking. "
                "Install with: pip install sentence-transformers"
            ) from exc

    def rerank(
        self,
        query: str,
        laws: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Score and reorder a list of law dicts by relevance to query.

        Args:
            query: The case description / defendant feature text.
            laws: List of law dicts, each must have a "description" or "text" key.
            top_k: Optional override for the number of laws to return.

        Returns:
            Top-K laws ordered by cross-encoder score (best first).
            If fewer than top_k laws are provided, all are returned.
        """
        if not laws:
            return laws

        self._load()

        pairs = [(query, law.get("description") or law.get("text") or "") for law in laws]

        scores = self._model.predict(pairs, batch_size=self.batch_size)

        # Attach score for debugging, then sort
        scored = sorted(zip(scores, laws), key=lambda x: float(x[0]), reverse=True)

        k_limit = top_k if top_k is not None else self.top_k
        result = []
        for score, law in scored[:k_limit]:
            law = dict(law)  # avoid mutating original
            law["_rerank_score"] = round(float(score), 4)
            result.append(law)

        logger.debug(
            f"Reranked {len(laws)} laws → kept top {len(result)} "
            f"(scores: {[r['_rerank_score'] for r in result[:3]]}...)"
        )
        return result


# Module-level singleton – shared across requests to avoid repeated model loads
_default_reranker: Optional[CrossEncoderReranker] = None


def get_reranker(
    model_name: str = CrossEncoderReranker.DEFAULT_MODEL,
    top_k: int = 8,
) -> CrossEncoderReranker:
    """Return the shared reranker singleton, creating it on first call."""
    global _default_reranker
    if _default_reranker is None or _default_reranker.model_name != model_name:
        _default_reranker = CrossEncoderReranker(model_name=model_name, top_k=top_k)
    else:
        _default_reranker.top_k = top_k
    return _default_reranker
