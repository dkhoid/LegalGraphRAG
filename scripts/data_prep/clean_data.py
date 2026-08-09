"""
Data Cleaning Pipeline cho LegalGraphRAG.

Input:
  data/raw/law_corpus.jsonl
  data/raw/criminal_law_processed.json
  data/processed/cases_with_feature.json

Output (data/clean/):
  law_corpus_clean.jsonl     — filtered, deduped, source-tagged
  cases_clean.json           — normalized domain, law refs resolved
  law_to_dispute_clean.json  — related_laws populated
  cleaning_report.json       — stats

Chạy: python scripts/data_prep/clean_data.py
"""

import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.utils.logger import logger

# ─────────────────────────── CONSTANTS ───────────────────────────

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "clean")
CORPUS_RAW = os.path.join(PROJECT_ROOT, "data", "raw", "law_corpus.jsonl")
LAW_TO_DISPUTE_RAW = os.path.join(PROJECT_ROOT, "data", "raw", "criminal_law_processed.json")
CASES_RAW = os.path.join(PROJECT_ROOT, "data", "processed", "cases_with_feature.json")

# Domain canonical mapping
DOMAIN_MAP = {
    "dan_su": "dan_su",
    "dân sự": "dan_su",
    "Dân sự": "dan_su",
    "dân_sự": "dan_su",
    "lao_dong": "lao_dong",
    "lao động": "lao_dong",
    "Lao động": "lao_dong",
    "lao_động": "lao_dong",
    "bhxh": "bhxh",
    "bảo hiểm": "bhxh",
    "bao_hiem": "bhxh",
    "Bảo hiểm": "bhxh",
    "bảo_hiểm": "bhxh",
}

DOMAIN_CANONICAL = {"dan_su", "lao_dong", "bhxh"}

# Priority law keywords per domain (ordered: most specific first)
DOMAIN_PRIORITY_KEYWORDS: dict[str, list[str]] = {
    "dan_su": [
        "91/2015/qh13",  # BLDS 2015 Zalo ID
        "bộ luật dân sự 2015",
        "bộ luật dân sự",
        "dân sự",
    ],
    "lao_dong": [
        "45/2019/qh14",  # BLLĐ 2019 Zalo ID
        "bộ luật lao động 2019",
        "bộ luật lao động",
        "lao động",
    ],
    "bhxh": [
        "58/2014/qh13",  # Luật BHXH 2014 Zalo ID
        "luật bảo hiểm xã hội 2014",
        "luật bảo hiểm xã hội",
        "bảo hiểm xã hội",
    ],
}

# Zalo ministry filter: keep (True) or remove (False)
# Only entries that match a KEEP pattern are kept
ZALO_KEEP_PATTERNS = [
    "/qh",  # Luật Quốc hội
    "/nđ-cp",  # Nghị định CP
    "/nd-cp",
    "/tt-bldtbxh",  # Bộ Lao động TBXH
    "/tt-btp",  # Bộ Tư pháp
    "/tt-nhnn",  # Ngân hàng Nhà nước
    "/tt-btc",  # Bộ Tài chính
    "/tt-blđtbxh",  # alternate encoding
]

ZALO_FILTER_PATTERNS = [
    "/tt-byt",  # Y tế
    "/tt-bgtvt",  # Giao thông
    "/tt-btnmt",  # Môi trường
    "/tt-bca",  # Công an
    "/tt-bnnptnt",  # Nông nghiệp
    "/tt-bgddt",  # Giáo dục
    "/tt-bkhcn",  # Khoa học CN
    "/tt-btttt",  # Thông tin TT
    "/tt-bvhttdl",  # VH Thể thao
    "/tt-bkhdt",  # Kế hoạch ĐT
]

# Numeric ID filter: remove if name matches these patterns
NUMERIC_FILTER_KEYWORDS = [
    "quyết định",
    "chỉ thị số",
    "kế hoạch của chính phủ",
    "kế hoạch phổ biến",
    "kế hoạch thực hiện",
    "kế hoạch tổ chức",
    "chương trình",
]

# ─────────────────────────── HELPERS ───────────────────────────


def normalize_unicode(text: str) -> str:
    """Normalize to NFC and strip excessive whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def text_hash(text: str) -> str:
    """Hash first 500 chars of text for dedup."""
    return hashlib.md5(text[:500].encode("utf-8")).hexdigest()


def classify_zalo_entry(text_id: str) -> str:
    """Return 'keep', 'filter', or 'review' for a Zalo entry."""
    zid = text_id.replace("zalo_", "").lower()
    for pat in ZALO_FILTER_PATTERNS:
        if pat in zid:
            return "filter"
    for pat in ZALO_KEEP_PATTERNS:
        if pat in zid:
            return "keep"
    return "review"  # Default: keep but mark as review


def classify_numeric_entry(name: str) -> str:
    """Return 'keep' or 'filter' for a numeric (th1nhng0) entry."""
    name_lower = name.lower()
    for kw in NUMERIC_FILTER_KEYWORDS:
        if kw in name_lower:
            return "filter"
    return "keep"


def infer_doc_type(text_id: str, name: str) -> str:
    """Infer document type label for the source field."""
    name_lower = name.lower()
    zid = text_id.lower()
    if "/qh" in zid:
        return "Luật/Bộ luật"
    if "/nđ-cp" in zid or "/nd-cp" in zid:
        return "Nghị định"
    if "/tt-" in zid:
        return "Thông tư"
    if "/qđ-" in zid or "/qd-" in zid:
        return "Quyết định"
    if "bộ luật" in name_lower:
        return "Bộ luật"
    if "nghị định" in name_lower:
        return "Nghị định"
    if "thông tư" in name_lower:
        return "Thông tư"
    if "quyết định" in name_lower:
        return "Quyết định"
    return "Văn bản pháp luật"


# ─────────────────────────── MODULE A: CORPUS CLEAN ───────────────────────────


def clean_corpus(corpus_raw: str) -> tuple[list[dict], dict]:
    """
    Filter, dedupe, normalize, and add source fields to corpus.
    Returns (clean_entries, stats).
    """
    logger.info("=== Module A: Cleaning corpus ===")

    with open(corpus_raw, encoding="utf-8") as f:
        raw = [json.loads(l) for l in f if l.strip()]

    stats = {
        "total_raw": len(raw),
        "filtered_out": 0,
        "deduped": 0,
        "kept": 0,
        "filter_reasons": Counter(),
    }

    seen_hashes: set[str] = set()
    clean: list[dict] = []

    for entry in raw:
        text_id: str = entry["text_id"]
        text: str = entry.get("text", "")
        name: str = entry.get("name", "")

        # ── Filter step ──
        if text_id.startswith("zalo_"):
            decision = classify_zalo_entry(text_id)
            if decision == "filter":
                stats["filtered_out"] += 1
                stats["filter_reasons"][f"zalo_{decision}"] += 1
                continue
        else:
            # We enforce Plan A: ONLY keep Zalo-formatted IDs.
            # All legacy numeric IDs from th1nhng0 are dropped.
            stats["filtered_out"] += 1
            stats["filter_reasons"]["not_zalo_format"] += 1
            continue

        # ── Dedup step ──
        h = text_hash(text)
        if h in seen_hashes:
            stats["deduped"] += 1
            continue
        seen_hashes.add(h)

        # ── Normalize ──
        text = normalize_unicode(text)
        name = normalize_unicode(name)

        # ── Add metadata ──
        source = "zalo" if text_id.startswith("zalo_") else "th1nhng0"
        doc_type = infer_doc_type(text_id, name)

        clean.append(
            {
                "text_id": text_id,
                "text": text,
                "name": name,
                "source": source,
                "doc_type": doc_type,
            }
        )

    stats["kept"] = len(clean)
    logger.info(
        f"  Raw: {stats['total_raw']} → Kept: {stats['kept']} "
        f"(filtered: {stats['filtered_out']}, deduped: {stats['deduped']})"
    )
    return clean, stats


# ─────────────────────────── MODULE B: CASES CLEAN ───────────────────────────


def build_corpus_index(clean_corpus: list[dict]) -> dict[str, dict]:
    """Build lookup: text_id → entry."""
    return {e["text_id"]: e for e in clean_corpus}


def build_article_index(clean_corpus: list[dict]) -> dict[str, list[dict]]:
    """
    Build lookup: article_number → [entries containing that article].
    Used for Tier 2 disambiguation.
    """
    idx: dict[str, list[dict]] = defaultdict(list)
    for entry in clean_corpus:
        match = re.search(r"[Đđ]iều\s+(\d+)", entry["name"])
        if match:
            idx[match.group(1)].append(entry)
    return idx


def resolve_law_ref(
    law_str: str,
    domain: str,
    corpus_index: dict[str, dict],
    article_index: dict[str, list[dict]],
) -> dict:
    """
    Attempt to resolve a single law reference string.

    Returns:
        {
            "corpus_id": str | None,
            "name": str | None,
            "confidence": "exact" | "domain_match" | "ambiguous" | "not_found",
        }
    """
    law_str = law_str.strip()

    # Tier 1: exact key match
    if law_str in corpus_index:
        entry = corpus_index[law_str]
        return {"corpus_id": law_str, "name": entry["name"], "confidence": "exact"}

    # Tier 2: bare number — try domain-based disambiguation
    if re.fullmatch(r"\d+[a-z]?", law_str):
        candidates = article_index.get(law_str, [])
        if not candidates:
            return {"corpus_id": None, "name": None, "confidence": "not_found"}

        norm_domain = DOMAIN_MAP.get(domain, domain)
        priority_kws = DOMAIN_PRIORITY_KEYWORDS.get(norm_domain, [])

        # Score candidates by priority keyword match
        scored: list[tuple[int, dict]] = []
        for c in candidates:
            name_lower = c["name"].lower()
            text_id_lower = c["text_id"].lower()
            score = 0
            for i, kw in enumerate(priority_kws):
                if kw in name_lower or kw in text_id_lower:
                    score = len(priority_kws) - i  # higher score = more specific
                    break
            if score > 0:
                scored.append((score, c))

        if not scored:
            return {"corpus_id": None, "name": None, "confidence": "ambiguous"}

        # Sort descending by score, take best
        scored.sort(key=lambda x: -x[0])
        best_score, best = scored[0]

        # If multiple with same top score → still ambiguous but pick first
        confidence = "domain_match" if best_score > 0 else "ambiguous"
        return {"corpus_id": best["text_id"], "name": best["name"], "confidence": confidence}

    # Not a known pattern
    return {"corpus_id": None, "name": law_str, "confidence": "not_found"}


def clean_cases(
    cases_raw: str,
    corpus_index: dict[str, dict],
    article_index: dict[str, list[dict]],
) -> tuple[list[dict], dict]:
    """
    Normalize domain, resolve law refs, dedupe facts.
    Returns (clean_cases, stats).
    """
    logger.info("=== Module B: Cleaning cases ===")

    with open(cases_raw, encoding="utf-8") as f:
        raw: list[dict] = json.load(f)

    stats = {
        "total_raw": len(raw),
        "deduped": 0,
        "domain_normalized": 0,
        "law_refs_total": 0,
        "law_refs_exact": 0,
        "law_refs_domain_match": 0,
        "law_refs_ambiguous": 0,
        "law_refs_not_found": 0,
        "cases_fully_resolved": 0,
        "cases_partially_resolved": 0,
        "cases_unresolved": 0,
    }

    seen_facts: set[str] = set()
    clean: list[dict] = []

    for case in raw:
        # ── Dedup by fact hash ──
        fact = case.get("fact", "")
        fhash = text_hash(fact)
        if fhash in seen_facts:
            stats["deduped"] += 1
            continue
        seen_facts.add(fhash)

        # ── Normalize domain ──
        raw_domain = case.get("domain", "")
        canonical_domain = DOMAIN_MAP.get(raw_domain, raw_domain)
        if canonical_domain != raw_domain:
            stats["domain_normalized"] += 1

        # ── Resolve law refs ──
        raw_laws: list[str] = [str(l) for l in (case.get("law") or case.get("laws") or [])]
        resolved_refs: list[dict] = []
        confidence_counts: Counter = Counter()

        for law_str in raw_laws:
            stats["law_refs_total"] += 1
            result = resolve_law_ref(law_str, canonical_domain, corpus_index, article_index)
            resolved_refs.append(
                {
                    "raw": law_str,
                    **result,
                }
            )
            confidence_counts[result["confidence"]] += 1
            stats[f"law_refs_{result['confidence']}"] += 1

        # Case-level resolve status
        if not resolved_refs:
            resolve_status = "no_laws"
        elif all(r["confidence"] in ("exact", "domain_match") for r in resolved_refs):
            resolve_status = "resolved"
            stats["cases_fully_resolved"] += 1
        elif any(r["confidence"] in ("exact", "domain_match") for r in resolved_refs):
            resolve_status = "partial"
            stats["cases_partially_resolved"] += 1
        else:
            resolve_status = "unresolved"
            stats["cases_unresolved"] += 1

        clean_case = {
            **case,
            "domain": canonical_domain,
            "law_resolved": resolved_refs,
            "law_resolve_status": resolve_status,
        }
        clean.append(clean_case)

    logger.info(f"  Raw: {stats['total_raw']} → Kept: {len(clean)} (deduped: {stats['deduped']})")
    logger.info(f"  Domain normalized: {stats['domain_normalized']}")
    logger.info(
        f"  Law refs: {stats['law_refs_total']} total | "
        f"exact={stats['law_refs_exact']} "
        f"domain_match={stats['law_refs_domain_match']} "
        f"ambiguous={stats['law_refs_ambiguous']} "
        f"not_found={stats['law_refs_not_found']}"
    )
    logger.info(
        f"  Cases: fully_resolved={stats['cases_fully_resolved']} "
        f"partial={stats['cases_partially_resolved']} "
        f"unresolved={stats['cases_unresolved']}"
    )
    return clean, stats


# ─────────────────────────── MODULE C: related_laws ───────────────────────────


def build_related_laws(
    clean_cases: list[dict],
    clean_corpus: list[dict],
    min_co_citation: int = 1,
) -> list[dict]:
    """
    Populate related_laws in law_to_dispute via co-citation analysis.
    Two corpus entries are 'related' if they co-appear in >= min_co_citation cases.
    We build the law_to_dispute nodes dynamically from the clean_corpus.
    """
    logger.info("=== Module C: Building related_laws ===")

    # Initialize LTD nodes directly from corpus
    ltd_entries = []
    for corp_entry in clean_corpus:
        ltd_entries.append(
            {
                "id": corp_entry["text_id"],
                "items": [
                    {
                        "text": corp_entry["text"],
                        "dispute": [corp_entry["doc_type"]],
                        "judge_dep": [],
                        "related_laws": [],
                    }
                ],
            }
        )

    # Build co-citation matrix
    co_citation: Counter = Counter()
    for case in clean_cases:
        resolved_ids = [r["corpus_id"] for r in case.get("law_resolved", []) if r.get("corpus_id")]
        # All pairs in the same case
        for i in range(len(resolved_ids)):
            for j in range(i + 1, len(resolved_ids)):
                pair = tuple(sorted([resolved_ids[i], resolved_ids[j]]))
                co_citation[pair] += 1

    # Build per-ID related list
    related_map: dict[str, list[str]] = defaultdict(list)
    for (a, b), count in co_citation.items():
        if count >= min_co_citation:
            related_map[a].append(b)
            related_map[b].append(a)

    # Apply to law_to_dispute
    updated = 0
    clean_ltd: list[dict] = []
    for entry in ltd_entries:
        entry_id = str(entry["id"])
        related = related_map.get(entry_id, [])
        new_entry = {**entry}
        for item in new_entry.get("items", []):
            item["related_laws"] = related
        if related:
            updated += 1
        clean_ltd.append(new_entry)

    logger.info(f"  law_to_dispute entries with related_laws populated: {updated}/{len(clean_ltd)}")
    return clean_ltd


# ─────────────────────────── MAIN ───────────────────────────


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    report = {
        "generated_at": datetime.now().isoformat(),
        "corpus": {},
        "cases": {},
    }

    # ── A: Clean corpus ──
    clean_corp, corp_stats = clean_corpus(CORPUS_RAW)
    report["corpus"] = corp_stats
    report["corpus"]["filter_reasons"] = dict(corp_stats["filter_reasons"])

    # Write corpus
    corpus_out = os.path.join(OUTPUT_DIR, "law_corpus_clean.jsonl")
    with open(corpus_out, "w", encoding="utf-8") as f:
        for entry in clean_corp:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(clean_corp)} entries → {corpus_out}")

    # ── Build indices for law resolution ──
    corpus_idx = build_corpus_index(clean_corp)
    article_idx = build_article_index(clean_corp)

    # ── B: Clean cases ──
    clean_cses, case_stats = clean_cases(CASES_RAW, corpus_idx, article_idx)
    report["cases"] = case_stats

    cases_out = os.path.join(OUTPUT_DIR, "cases_clean.json")
    with open(cases_out, "w", encoding="utf-8") as f:
        json.dump(clean_cses, f, ensure_ascii=False, indent=2)
    logger.info(f"Wrote {len(clean_cses)} cases → {cases_out}")

    # ── C: related_laws ──
    clean_ltd = build_related_laws(clean_cses, clean_corp)

    ltd_out = os.path.join(OUTPUT_DIR, "law_to_dispute_clean.json")
    with open(ltd_out, "w", encoding="utf-8") as f:
        json.dump(clean_ltd, f, ensure_ascii=False, indent=2)
    logger.info(f"Wrote {len(clean_ltd)} entries → {ltd_out}")

    # ── Report ──
    report_out = os.path.join(OUTPUT_DIR, "cleaning_report.json")
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── Summary ──
    logger.info("\n" + "=" * 60)
    logger.info("CLEANING COMPLETE")
    logger.info("=" * 60)
    logger.info(
        f"Corpus:  {corp_stats['total_raw']} → {corp_stats['kept']} "
        f"(-{corp_stats['filtered_out'] + corp_stats['deduped']})"
    )
    logger.info(
        f"Cases:   {case_stats['total_raw']} → {len(clean_cses)} " f"(-{case_stats['deduped']})"
    )
    total = case_stats["law_refs_total"]
    matched = case_stats["law_refs_exact"] + case_stats["law_refs_domain_match"]
    logger.info(
        f"Law ref resolution: {matched}/{total} = {matched/total*100:.1f}%"
        if total
        else "Law ref resolution: N/A"
    )
    logger.info(f"Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
