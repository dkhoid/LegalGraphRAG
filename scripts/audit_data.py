from core.utils.logger import logger
import json
import os
import pickle


def load_json(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.info(f"Error loading {filepath}: {e}")
        return None


def load_jsonl(filepath):
    data = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    except Exception as e:
        logger.info(f"Error loading {filepath}: {e}")
        return None


def main():
    logger.info("=== DATA AUDIT REPORT ===")

    # 1. Check cases
    cases = load_json("datas/cases_with_feature.json")
    case_laws = set()
    missing_facts = 0
    missing_laws = 0
    if cases:
        logger.info(f"\n[1] cases_with_feature.json")
        logger.info(f"Total cases: {len(cases)}")
        for c in cases:
            if not c.get("fact"):
                missing_facts += 1
            laws = c.get("law", [])
            if not laws:
                missing_laws += 1
            for l in laws:
                case_laws.add(str(l))
        logger.info(f"- Cases missing 'fact': {missing_facts}")
        logger.info(f"- Cases missing 'law': {missing_laws}")
        logger.info(f"- Unique law articles referenced in cases: {len(case_laws)}")

    # 2. Check Civil Law to Dispute
    law_to_dispute = load_json("datas/law_to_dispute.json")
    civil_law_ids = set()
    laws_missing_dispute = 0
    if law_to_dispute:
        logger.info(f"\n[2] law_to_dispute.json")
        logger.info(f"Total civil laws: {len(law_to_dispute)}")
        for l in law_to_dispute:
            civil_law_ids.add(str(l.get("id")))
            items = l.get("items", [])
            has_dispute = False
            for item in items:
                if item.get("dispute"):
                    has_dispute = True
                    break
            if not has_dispute:
                laws_missing_dispute += 1
        logger.info(f"- Civil Laws missing any 'dispute' label: {laws_missing_dispute}")

    # 3. Check Criminal Law Processed
    criminal_law = load_json("raw_data/criminal_law_processed.json")
    criminal_law_ids = set()
    if criminal_law:
        logger.info(f"\n[3] criminal_law_processed.json")
        logger.info(f"Total criminal laws: {len(criminal_law)}")
        for l in criminal_law:
            criminal_law_ids.add(str(l.get("id")))

    # 4. Check Law Corpus JSONL
    law_corpus = load_jsonl("raw_data/law_corpus.jsonl")
    if law_corpus:
        logger.info(f"\n[4] law_corpus.jsonl")
        logger.info(f"Total lines in corpus: {len(law_corpus)}")

    # 5. Referential Integrity Check
    logger.info(f"\n[5] Referential Integrity (Cases -> Laws)")
    if cases and (civil_law_ids or criminal_law_ids):
        all_known_laws = civil_law_ids.union(criminal_law_ids)
        unknown_laws = case_laws - all_known_laws
        logger.info(f"- Total referenced laws in cases: {len(case_laws)}")
        logger.info(f"- Referenced laws NOT FOUND in civil/criminal law DB: {len(unknown_laws)}")
        if len(unknown_laws) > 0:
            logger.info(f"  Sample of missing laws: {list(unknown_laws)[:10]}")

    # 6. Graph file check
    graph_path = "datas/graph.pkl"
    logger.info(f"\n[6] Graph file check ({graph_path})")
    if os.path.exists(graph_path):
        size = os.path.getsize(graph_path)
        logger.info(f"File size: {size} bytes")
        if size < 1000:
            logger.info("- Warning: Graph file seems very small/empty.")
    else:
        logger.info("File does not exist.")


if __name__ == "__main__":
    main()
