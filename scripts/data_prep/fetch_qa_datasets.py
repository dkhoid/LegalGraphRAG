import json
import os
import sys
from datasets import load_dataset
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.utils.logger import logger
from scripts.fetch_zalo_legal_data import guess_domain
from scripts.fetch_vn_legal_data import classify_article_issue


def main():
    cases_db_path = os.path.join(PROJECT_ROOT, "datas", "cases_with_feature.json")

    logger.info("Loading existing cases_with_feature.json...")
    with open(cases_db_path, "r", encoding="utf-8") as f:
        cases_db = json.load(f)

    existing_ids = {str(case.get("id")) for case in cases_db}
    initial_count = len(cases_db)

    added_count = 0
    skipped_count = 0

    # ---------------------------------------------------------
    # 1. Process adamwhite625/vietnam-legal-qa
    # ---------------------------------------------------------
    logger.info("Loading adamwhite625/vietnam-legal-qa...")
    try:
        ds1 = load_dataset("adamwhite625/vietnam-legal-qa", split="train", streaming=False)
        for row in tqdm(ds1, desc="Processing vietnam-legal-qa"):
            qa_id = f"qa1_{row['id']}"
            if qa_id in existing_ids:
                continue

            question = row.get("question", "")
            if not question:
                continue

            domain = guess_domain(question)
            if not domain:
                skipped_count += 1
                continue

            law_name = row.get("law_name", "")
            law_id = row.get("law_id", "")
            referenced_law = f"{law_name} {law_id}".strip()

            dispute = classify_article_issue(question, 0, domain)

            case_entry = {
                "id": qa_id,
                "fact": question,
                "dispute": dispute,
                "law": [referenced_law] if referenced_law else [],
                "laws": [referenced_law] if referenced_law else [],
                "domain": domain,
            }
            cases_db.append(case_entry)
            existing_ids.add(qa_id)
            added_count += 1
    except Exception as e:
        logger.error(f"Error processing vietnam-legal-qa: {e}")

    # ---------------------------------------------------------
    # 2. Process namphan1999/data-luat
    # ---------------------------------------------------------
    logger.info("Loading namphan1999/data-luat...")
    try:
        ds2 = load_dataset("namphan1999/data-luat", split="train", streaming=False)
        # IDs are not provided in this dataset, we will generate them
        for idx, row in enumerate(tqdm(ds2, desc="Processing data-luat")):
            qa_id = f"qa2_{idx}"
            if qa_id in existing_ids:
                continue

            question = row.get("question", "")
            if not question:
                continue

            domain = guess_domain(question)
            if not domain:
                skipped_count += 1
                continue

            referenced_law = row.get("terms", "").strip()

            dispute = classify_article_issue(question, 0, domain)

            case_entry = {
                "id": qa_id,
                "fact": question,
                "dispute": dispute,
                "law": [referenced_law] if referenced_law else [],
                "laws": [referenced_law] if referenced_law else [],
                "domain": domain,
            }
            cases_db.append(case_entry)
            existing_ids.add(qa_id)
            added_count += 1
    except Exception as e:
        logger.error(f"Error processing data-luat: {e}")

    # ---------------------------------------------------------
    # Save the updated cases DB
    # ---------------------------------------------------------
    logger.info(
        f"Summary: Added {added_count} related cases. Skipped {skipped_count} irrelevant ones."
    )
    logger.info(f"Total cases in DB before: {initial_count}")
    logger.info(f"Total cases in DB now: {len(cases_db)}")

    if added_count > 0:
        logger.info("Saving updated cases_with_feature.json...")
        with open(cases_db_path, "w", encoding="utf-8") as f:
            json.dump(cases_db, f, ensure_ascii=False, indent=2)
        logger.info("Cases database updated successfully!")
    else:
        logger.info("No new cases added.")


if __name__ == "__main__":
    main()
