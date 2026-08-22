"""Confidence scoring for LegalGraphRAG pipeline outputs.

Scores are derived purely from pipeline signals (heuristics) – no LLM call needed.
This keeps latency low while still providing useful calibration for callers.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ConfidenceResult:
    """Structured confidence output attached to each analyzed defendant item."""

    retrieval_quality: float
    """Proportion of laws retrieved vs target (capped at 1.0). Higher = better."""

    law_applicability: float
    """Ratio of retrieved laws that the judge accepted. Higher = stronger match."""

    has_evidence: float
    """1.0 if both used_laws and used_facts are non-empty, else 0.0."""

    overall: float
    """Weighted average of the three signals above (weights: 30/40/30)."""

    grade: Literal["HIGH", "MEDIUM", "LOW"]
    """Qualitative tier: HIGH > 0.75, MEDIUM > 0.45, LOW otherwise."""

    review_required: bool
    """True when overall < 0.45 – signals the case may need human review."""

    def to_dict(self) -> dict:
        """Serialize to a plain dict for inclusion in analyze_case() output."""
        return {
            "retrieval_quality": round(self.retrieval_quality, 3),
            "law_applicability": round(self.law_applicability, 3),
            "overall": round(self.overall, 3),
            "grade": self.grade,
            "review_required": self.review_required,
        }


def compute_confidence(
    retrieved_laws: list,
    used_laws: list,
    retrieved_facts: list,
    parsing_failures: int = 0,
) -> ConfidenceResult:
    """Compute confidence from pipeline signals without calling an LLM.

    Weighting rationale:
    - retrieval_quality (30%): Foundation. Without good retrieval nothing works.
    - law_applicability (40%): Most diagnostic – if the judge rejects all laws,
      something is likely wrong with the query or data.
    - has_evidence (30%): Binary signal that we have concrete grounding for output.
    - parsing_failures penalty: Deducts up to 0.25 when batch judge parsing failed.
    - all_rejected penalty: Deducts 0.15 when retrieved laws were found but all rejected.

    Args:
        retrieved_laws: All laws returned by the retriever.
        used_laws: Laws accepted by judge_law (subset of retrieved_laws).
        retrieved_facts: All case facts returned by the retriever.
        parsing_failures: Number of LLM parsing failures encountered during judging.

    Returns:
        ConfidenceResult with all fields populated.

    Examples:
        >>> c = compute_confidence([1, 2, 3, 4, 5], [1, 2], [1])
        >>> c.grade
        'HIGH'
        >>> c = compute_confidence([], [], [])
        >>> c.review_required
        True
    """
    retrieval_quality = min(len(retrieved_laws) / 5.0, 1.0)
    law_applicability = len(used_laws) / len(retrieved_laws) if retrieved_laws else 0.0
    has_evidence = 1.0 if (used_laws and retrieved_facts) else 0.0

    raw_overall = retrieval_quality * 0.3 + law_applicability * 0.4 + has_evidence * 0.3

    # Penalties
    parse_penalty = min(parsing_failures * 0.05, 0.25)
    all_rejected_penalty = 0.15 if (retrieved_laws and not used_laws) else 0.0

    overall = max(0.0, min(1.0, raw_overall - parse_penalty - all_rejected_penalty))

    grade: Literal["HIGH", "MEDIUM", "LOW"] = (
        "HIGH" if overall > 0.75 else "MEDIUM" if overall > 0.45 else "LOW"
    )

    return ConfidenceResult(
        retrieval_quality=retrieval_quality,
        law_applicability=law_applicability,
        has_evidence=has_evidence,
        overall=overall,
        grade=grade,
        review_required=overall < 0.45,
    )
