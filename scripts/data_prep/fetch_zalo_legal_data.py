import json
import os
import sys
from datasets import load_dataset
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.utils.logger import logger
from scripts.fetch_vn_legal_data import classify_article_issue, generate_judge_dep


def guess_domain(text):
    text_lower = text.lower()

    civil_keywords = [
        "dân sự",
        "hợp đồng",
        "bồi thường",
        "sở hữu",
        "thừa kế",
        "tài sản",
        "giao dịch",
    ]
    labor_keywords = [
        "lao động",
        "người sử dụng lao động",
        "tiền lương",
        "sa thải",
        "thời giờ làm việc",
        "kỷ luật",
        "đình công",
    ]
    insurance_keywords = [
        "bảo hiểm xã hội",
        "bảo hiểm y tế",
        "bhxh",
        "bhyt",
        "trợ cấp",
        "hưu trí",
        "thai sản",
        "ốm đau",
    ]

    civil_score = sum(text_lower.count(k) for k in civil_keywords)
    labor_score = sum(text_lower.count(k) for k in labor_keywords)
    insurance_score = sum(text_lower.count(k) for k in insurance_keywords)

    max_score = max(civil_score, labor_score, insurance_score)
    if max_score < 2:  # Threshold to filter out irrelevant docs
        return None

    if max_score == civil_score:
        return "dân sự"
    elif max_score == labor_score:
        return "lao động"
    else:
        return "bảo hiểm"


def main():
    output_dir = os.path.join(PROJECT_ROOT, "datas")
    raw_data_dir = os.path.join(PROJECT_ROOT, "raw_data")

    ltd_path = os.path.join(output_dir, "law_to_dispute.json")
    corpus_path = os.path.join(raw_data_dir, "law_corpus.jsonl")
    raw_ltd_path = os.path.join(raw_data_dir, "law_to_dispute.json")

    logger.info("Loading existing datasets...")
    with open(ltd_path, "r", encoding="utf-8") as f:
        law_to_dispute_entries = json.load(f)

    law_corpus_entries = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                law_corpus_entries.append(json.loads(line))

    with open(raw_ltd_path, "r", encoding="utf-8") as f:
        raw_law_to_dispute = json.load(f)

    existing_ids = {str(entry["id"]) for entry in law_to_dispute_entries}

    logger.info("Loading GreenNode/zalo-ai-legal-text-retrieval-vn dataset (corpus)...")
    dataset = load_dataset(
        "GreenNode/zalo-ai-legal-text-retrieval-vn", "corpus", split="test", streaming=False
    )

    added_count = 0
    skipped_count = 0

    for row in tqdm(dataset, desc="Processing Zalo dataset"):
        zalo_id = f"zalo_{row['id']}"
        if zalo_id in existing_ids:
            continue

        text = row.get("text", "")
        title = row.get("title", "")
        if not text:
            skipped_count += 1
            continue

        domain = guess_domain(text)
        if not domain:
            skipped_count += 1
            continue

        issues = classify_article_issue(text, 0, domain)
        judge_deps = generate_judge_dep(text)

        # Add to law_to_dispute
        entry = {
            "id": zalo_id,
            "items": [
                {"text": text, "dispute": issues, "judge_dep": judge_deps, "related_laws": []}
            ],
        }
        law_to_dispute_entries.append(entry)

        # Add to raw_law_to_dispute
        raw_law_to_dispute.append({"id": zalo_id, "dispute": issues})

        # Add to law_corpus
        law_corpus_entries.append({"text_id": zalo_id, "text": text, "name": title})

        added_count += 1

    logger.info(f"Added {added_count} related articles. Skipped {skipped_count} irrelevant ones.")

    if added_count > 0:
        logger.info("Saving updated datasets...")
        with open(ltd_path, "w", encoding="utf-8") as f:
            json.dump(law_to_dispute_entries, f, ensure_ascii=False, indent=2)

        with open(corpus_path, "w", encoding="utf-8") as f:
            for entry in law_corpus_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        with open(raw_ltd_path, "w", encoding="utf-8") as f:
            json.dump(raw_law_to_dispute, f, ensure_ascii=False, indent=2)

        logger.info("Datasets updated successfully!")
    else:
        logger.info("No new entries added.")


if __name__ == "__main__":
    main()
