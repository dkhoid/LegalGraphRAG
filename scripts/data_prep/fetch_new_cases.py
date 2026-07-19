import json
import os
import sys
import re
from datasets import load_dataset
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


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

    if civil_score == 0 and labor_score == 0 and insurance_score == 0:
        return None

    scores = {"dan_su": civil_score, "lao_dong": labor_score, "bhxh": insurance_score}
    return max(scores, key=scores.get)


def extract_laws(texts):
    laws = set()
    for text in texts:
        # Match 'Điều 123', 'ĐIỀU 12', 'điều 45'
        matches = re.findall(r"Điều\s+(\d+[a-zA-Z]?)", text, re.IGNORECASE)
        for m in matches:
            laws.add(m)
    return list(laws)


def main():
    cases_db_path = os.path.join(PROJECT_ROOT, "data", "processed", "cases_with_feature.json")

    if os.path.exists(cases_db_path):
        with open(cases_db_path, "r", encoding="utf-8") as f:
            cases_db = json.load(f)
    else:
        cases_db = []

    existing_ids = {str(case.get("id")) for case in cases_db}

    added_count = 0
    max_cases = 5000

    # Start ID from the max existing ID + 1
    current_id = (
        max([int(c.get("id", 0)) for c in cases_db if str(c.get("id", "")).isdigit()], default=0)
        + 1
    )

    datasets_to_load = ["NamSyntax/Vietnamese-Legal-QA-RAG", "thangvip/vietnamese-legal-qa"]

    for ds_name in datasets_to_load:
        if added_count >= max_cases:
            break

        print(f"Loading dataset {ds_name}...")
        try:
            ds = load_dataset(ds_name, split="train", streaming=False)
        except Exception as e:
            print(f"Failed to load dataset {ds_name}: {e}")
            continue

        for row in tqdm(ds, desc=f"Processing cases from {ds_name}"):
            if added_count >= max_cases:
                break

            # Normalize row into a list of QA pairs
            qa_pairs = []
            if "generated_qa_pairs" in row:
                # thangvip format
                context = row.get("article_content", "")
                for qa in row.get("generated_qa_pairs", []):
                    qa_pairs.append(
                        {
                            "question": qa.get("question", ""),
                            "answer": qa.get("answer", ""),
                            "context": context,
                            "type": qa.get("question_type", "Chung"),
                        }
                    )
            else:
                # NamSyntax format
                context_list = row.get("ground_truth_context", [])
                context = (
                    " ".join(context_list) if isinstance(context_list, list) else str(context_list)
                )
                qa_pairs.append(
                    {
                        "question": row.get("question", ""),
                        "answer": row.get("ground_truth_answer", ""),
                        "context": context,
                        "type": row.get("question_type", "Chung"),
                    }
                )

            for qa in qa_pairs:
                if added_count >= max_cases:
                    break

                question = qa["question"].strip()
                if not question or len(question) < 50:
                    continue

                domain = guess_domain(question)
                if not domain:
                    continue

                # Deduplication check
                if question in [c.get("fact") for c in cases_db]:
                    continue

                # Extract laws
                laws = extract_laws([qa["context"]])

                # Build features
                features = {
                    "parties_info": ["Cá nhân", "Tổ chức"],
                    "dispute_acts": ["Tranh chấp quyền lợi", "Vi phạm quy định pháp luật"],
                    "subject_matter": ["Tài sản", "Quyền lợi hợp pháp"],
                    "fault_and_evidence": ["Dựa trên câu hỏi"],
                }

                if domain == "lao_dong":
                    features["parties_info"] = ["Người lao động", "Người sử dụng lao động"]
                elif domain == "bhxh":
                    features["parties_info"] = ["Người tham gia BHXH", "Cơ quan BHXH"]

                case_obj = {
                    "id": current_id,
                    "name": ["Người hỏi (Ẩn danh)"],
                    "fact": question,
                    "dispute": ["Giải đáp thắc mắc pháp lý", qa["type"]],
                    "law": laws,
                    "laws": laws,
                    "domain": domain,
                    "features": features,
                }

                cases_db.append(case_obj)
                added_count += 1
                current_id += 1

    print(f"Saving {added_count} new cases to cases_with_feature.json...")
    with open(cases_db_path, "w", encoding="utf-8") as f:
        json.dump(cases_db, f, ensure_ascii=False, indent=2)

    print(f"Success! Reached total of {len(cases_db)} cases in DB.")
    print(
        "NOTE: build_graph_only.py has NOT been run, so token cost is ZERO. You must run it later when ready."
    )


if __name__ == "__main__":
    main()
