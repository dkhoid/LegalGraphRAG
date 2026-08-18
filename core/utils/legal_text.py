"""Vietnamese legal text utilities.

Handles domain-specific text normalization before retrieval:
- Abbreviation expansion: improves BM25/fulltext recall
- Legal stop-word removal: reduces noise in matching
"""

from __future__ import annotations
import re

# ─────────────────────────────────────────────────────────────────────────────
# Abbreviation table – Vietnamese legal text
# Keep both directions so expanding and reverting are cheap.
# ─────────────────────────────────────────────────────────────────────────────

_ABBREV_MAP: dict[str, str] = {
    # Code-level statutes
    "BLDS": "Bộ luật dân sự",
    "BLHS": "Bộ luật hình sự",
    "BLTTDS": "Bộ luật tố tụng dân sự",
    "BLTTHS": "Bộ luật tố tụng hình sự",
    "BLĐ": "Bộ luật lao động",
    # Specific laws
    "LHN": "Luật hôn nhân và gia đình",
    "LHNGĐ": "Luật hôn nhân và gia đình",
    "LTM": "Luật thương mại",
    "LDN": "Luật doanh nghiệp",
    "LKDBĐS": "Luật kinh doanh bất động sản",
    "LBHXH": "Luật bảo hiểm xã hội",
    "LSHTT": "Luật sở hữu trí tuệ",
    "LĐT": "Luật đầu tư",
    # Sub-statutory
    "NĐ": "Nghị định",
    "TT": "Thông tư",
    "QĐ": "Quyết định",
    "TTLT": "Thông tư liên tịch",
    "CT": "Chỉ thị",
    "NQ": "Nghị quyết",
    # Court / procedure
    "HĐXX": "Hội đồng xét xử",
    "TAND": "Tòa án nhân dân",
    "VKSND": "Viện kiểm sát nhân dân",
    "CQĐT": "Cơ quan điều tra",
    # Parties
    "NĐS": "nguyên đơn sự",
    "BĐS": "bị đơn sự",
}

# Compiled once at import time for performance
_ABBREV_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_ABBREV_MAP, key=len, reverse=True)) + r")\b"
)

# Legal stop-phrases – high-frequency but semantically empty for retrieval
_STOP_PHRASES = [
    "theo quy định tại",
    "căn cứ vào",
    "căn cứ theo",
    "theo quy định của",
    "theo quy định",
    "theo điều",
    "quy định tại khoản",
    "quy định tại điều",
]


def expand_abbreviations(text: str) -> str:
    """Expand Vietnamese legal abbreviations to improve BM25/fulltext recall.

    The expansion keeps the original abbreviation AND appends the full form in
    parentheses, so exact-match queries on abbreviations still work.

    Example:
        "Vi phạm BLDS 2015" → "Vi phạm BLDS (Bộ luật dân sự) 2015"

    Args:
        text: Raw Vietnamese legal text.

    Returns:
        Text with abbreviations expanded.
    """

    def replace_abbrev(m: re.Match) -> str:
        abbrev = m.group(1)
        full = _ABBREV_MAP.get(abbrev, abbrev)
        return f"{abbrev} ({full})"

    return _ABBREV_RE.sub(replace_abbrev, text)


def remove_legal_stopwords(text: str) -> str:
    """Remove high-frequency, low-signal legal stop-phrases.

    Useful before BM25 queries to reduce noise from boilerplate legal language.

    Args:
        text: Vietnamese legal text.

    Returns:
        Text with stop-phrases removed (preserving case).
    """
    for phrase in _STOP_PHRASES:
        text = re.sub(re.escape(phrase), " ", text, flags=re.IGNORECASE)
    return re.sub(r" {2,}", " ", text).strip()


def preprocess_for_retrieval(text: str, expand: bool = True, remove_stops: bool = False) -> str:
    """One-shot preprocessing pipeline for retrieval queries.

    Args:
        text: Input text (case description or feature string).
        expand: Whether to expand abbreviations (recommended True for BM25).
        remove_stops: Whether to remove legal stop-phrases (optional, can hurt recall).

    Returns:
        Preprocessed text ready for BM25/fulltext search.
    """
    if expand:
        text = expand_abbreviations(text)
    if remove_stops:
        text = remove_legal_stopwords(text)
    return text
