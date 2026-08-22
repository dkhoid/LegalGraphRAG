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


def parse_legal_identifier(entry: str) -> dict:
    """Parse Vietnamese legal identifier string into structured components.

    Format: [prefix]<doc_num>/<year>/<type>-<issuer>+<article>
    Examples:
        'zalo_45/2019/qh14+41' -> doc_type='CODE', doc_num='45', year=2019, article='41'
        'zalo_21/2014/tt-bkhcn+5' -> doc_type='CIRCULAR', doc_num='21', year=2014, article='5'
        'zalo_112/2020/nđ-cp+14' -> doc_type='DECREE', doc_num='112', year=2020, article='14'
    """
    if not isinstance(entry, str):
        return {"entry": str(entry), "doc_type": "OTHER", "article": "", "is_primary_code": False}

    s = entry.strip().lower()
    # Strip prefixes like zalo_
    s_clean = re.sub(r"^zalo_", "", s)

    article = ""
    doc_part = s_clean
    if "+" in s_clean:
        parts = s_clean.split("+", 1)
        doc_part = parts[0]
        article = parts[1].strip()

    is_primary_code = False
    doc_type = "OTHER"

    if "qh" in doc_part or "blds" in doc_part or "blld" in doc_part or "blhs" in doc_part:
        doc_type = "CODE"
        is_primary_code = True
    elif "ubtvqh" in doc_part:
        doc_type = "ORDINANCE"
        is_primary_code = True
    elif "nđ-cp" in doc_part or "nd-cp" in doc_part:
        doc_type = "DECREE"
    elif "tt-" in doc_part or "ttlt-" in doc_part:
        doc_type = "CIRCULAR"
    elif "qđ-" in doc_part or "qd-" in doc_part:
        doc_type = "DECISION"

    return {
        "raw": entry,
        "doc_part": doc_part,
        "doc_type": doc_type,
        "article": article,
        "is_primary_code": is_primary_code,
    }


def get_hierarchy_boost(entry: str, default_boost: float = 1.35) -> float:
    """Calculate hierarchical rank weighting based on Vietnamese legal hierarchy.

    Primary Codes (Quốc hội) > Ordinances > Decrees (Chính phủ) > Circulars (Bộ ngành).
    """
    parsed = parse_legal_identifier(entry)
    if parsed["is_primary_code"]:
        return default_boost
    elif parsed["doc_type"] == "DECREE":
        return 1.0
    elif parsed["doc_type"] in ("CIRCULAR", "DECISION"):
        return 0.8
    return 1.0
