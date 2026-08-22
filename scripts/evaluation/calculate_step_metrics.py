import json
import os
import argparse
import re
from typing import List

# Synonym groups for Vietnamese legal dispute types.
# Each entry is a list of equivalent/overlapping terms.
# A hit is recorded if pred and true share any term from the same group.
_DISPUTE_SYNONYM_GROUPS: List[List[str]] = [
    [
        "sa thải",
        "chấm dứt hđlđ",
        "chấm dứt hợp đồng lao động",
        "đơn phương chấm dứt",
        "sa thải trái pháp luật",
    ],
    ["tiền lương", "nợ lương", "trả lương", "thanh toán lương", "lương"],
    ["làm thêm giờ", "tăng ca", "overtime"],
    ["tai nạn lao động", "an toàn lao động", "bồi thường tai nạn", "bảo hiểm tai nạn"],
    ["bồi thường thiệt hại", "bồi thường", "thiệt hại"],
    ["hợp đồng", "hợp đồng lao động", "hđlđ"],
    ["kỷ luật lao động", "xử lý kỷ luật", "giảm thời hạn kỷ luật"],
    ["thừa kế", "phân chia tài sản thừa kế", "di sản"],
    ["thế chấp", "cầm cố", "tài sản bảo đảm"],
    ["vay tài sản", "hợp đồng vay", "tranh chấp hợp đồng vay"],
    ["đại diện", "người giám hộ", "giám hộ"],
    ["giao dịch dân sự", "hợp đồng dân sự"],
]

# Pre-build lookup: term → group index
_SYNONYM_LOOKUP: dict = {}
for _gidx, _group in enumerate(_DISPUTE_SYNONYM_GROUPS):
    for _term in _group:
        _SYNONYM_LOOKUP[_term.lower()] = _gidx


def _get_synonym_group(text: str) -> int:
    """Return the synonym group index for any term contained in text, or -1."""
    text_lower = text.lower()
    for term, gidx in _SYNONYM_LOOKUP.items():
        if term in text_lower:
            return gidx
    return -1


def extract_law_numbers(s: str) -> List[str]:
    """Extract article numbers from a law reference string.

    Handles formats like:
      - "Điều 36", "Dieu 36", "Article 36"
      - "45/2019/QH14" → extracts "45" (the article, not the year)
      - "zalo_45/2019/qh14+132" → extracts "132" after the '+'
      - bare numbers like "36", "38"

    The heuristic: numbers >= 2000 are treated as years and skipped.
    Known noise prefixes (zalo_, Article , Điều ) are stripped first.
    """
    if not isinstance(s, str):
        return []

    # Normalise
    s = s.strip()
    # Strip known garbage prefixes (case-insensitive)
    s = re.sub(r"(?i)^(zalo_|article\s*)", "", s).strip()
    s_lower = s.lower().replace("điều", "").replace("dieu", "")

    # If there's a '+' suffix (e.g. "45/2019/qh14+132"), prefer the part after '+'
    if "+" in s_lower:
        s_lower = s_lower.split("+")[-1]

    # Extract all digit sequences
    raw_nums = re.findall(r"\b(\d+)\b", s_lower)

    # Filter out years (>= 2000) and law-code segments that look like QH/NĐ numbers
    article_nums = [n for n in raw_nums if int(n) < 2000]

    return article_nums if article_nums else [s_lower.strip()]


def normalize(s: str) -> str:
    """Normalize text for comparison."""
    if not isinstance(s, str):
        return ""
    return s.lower().strip()


def dispute_hit(pred_disputes: List[str], true_disputes: List[str]) -> bool:
    """Check if predicted disputes semantically match true disputes.

    Matching strategy (any one is sufficient):
    1. Exact substring containment after normalisation.
    2. Synonym-group overlap: both pred and true share a synonym group term.
    3. Token overlap: >= 1 meaningful shared word (length > 2).
    """
    if not pred_disputes or not true_disputes:
        return False

    for pd in pred_disputes:
        pd_norm = normalize(pd)
        pd_group = _get_synonym_group(pd_norm)

        for td in true_disputes:
            td_norm = normalize(td)

            # Strategy 1: substring containment
            if pd_norm in td_norm or td_norm in pd_norm:
                return True

            # Strategy 2: synonym group match
            if pd_group >= 0 and pd_group == _get_synonym_group(td_norm):
                return True

            # Strategy 3: meaningful token overlap (words longer than 2 chars)
            pd_words = {w for w in pd_norm.split() if len(w) > 2}
            td_words = {w for w in td_norm.split() if len(w) > 2}
            if pd_words and td_words and len(pd_words & td_words) >= 1:
                return True

    return False


def calculate_step_metrics(results_file: str):
    if not os.path.exists(results_file):
        print(f"❌ Không tìm thấy file kết quả: {results_file}")
        return

    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    if not results:
        print("⚠️ File kết quả trống.")
        return

    total_cases = len(results)

    # Metrics
    step1_success = 0

    step2_law_retrieval_hits = 0
    step2_avg_retrieved_laws = 0
    step2_precision_sum = 0

    step3_filtered_laws_hits = 0
    step3_avg_used_laws = 0
    step3_precision_sum = 0

    step4_dispute_correct = 0
    step4_law_hits = 0

    # Confidence Metrics
    avg_confidence = 0
    review_required_count = 0
    avg_retrieval_quality = 0

    print("\n📊 BÁO CÁO HIỆU SUẤT TỪNG BƯỚC (STEP-BY-STEP METRICS)")
    print(f"File: {os.path.basename(results_file)}")
    print(f"Tổng số ca kiểm thử: {total_cases}")
    print("=" * 60)

    for case in results:
        true_disputes = case.get("true_dispute", [])
        if isinstance(true_disputes, str):
            true_disputes = [true_disputes]

        true_laws = case.get("law_article", [])
        if isinstance(true_laws, str):
            true_laws = [true_laws]

        # Extract all true law numbers for evaluation
        true_law_nums = set()
        for tl in true_laws:
            true_law_nums.update(extract_law_numbers(tl))

        judge_res_list = case.get("judge_res", [])
        if not judge_res_list:
            continue

        # We evaluate the first defendant's result for simplicity
        res = judge_res_list[0]

        # Confidence processing
        conf = res.get("confidence", {})
        avg_confidence += conf.get("overall", 0)
        avg_retrieval_quality += conf.get("retrieval_quality", 0)
        if conf.get("review_required", False):
            review_required_count += 1

        # STEP 1: Feature Extraction
        feature = res.get("feature", {})
        if feature:
            step1_success += 1

        # STEP 2: Retrieval
        retrieved_laws = res.get("retrieved_laws", [])
        step2_avg_retrieved_laws += len(retrieved_laws)

        retrieved_law_nums = set()
        correct_retrieved = 0
        for law in retrieved_laws:
            # Check entry or id
            entry = law.get("entry", str(law.get("id", "")))
            nums = extract_law_numbers(str(entry))
            retrieved_law_nums.update(nums)
            if true_law_nums and true_law_nums.intersection(set(nums)):
                correct_retrieved += 1

        if true_law_nums and true_law_nums.intersection(retrieved_law_nums):
            step2_law_retrieval_hits += 1

        if retrieved_laws:
            step2_precision_sum += correct_retrieved / len(retrieved_laws)

        # STEP 3: LLM Judge (Filtering)
        used_laws = res.get("used_laws", [])
        step3_avg_used_laws += len(used_laws)

        used_law_nums = set()
        correct_used = 0
        for law in used_laws:
            entry = law.get("entry", str(law.get("id", "")))
            nums = extract_law_numbers(str(entry))
            used_law_nums.update(nums)
            if true_law_nums and true_law_nums.intersection(set(nums)):
                correct_used += 1

        if true_law_nums and true_law_nums.intersection(used_law_nums):
            step3_filtered_laws_hits += 1

        if used_laws:
            step3_precision_sum += correct_used / len(used_laws)

        # STEP 4: Final Output
        judge_result = res.get("judge_result", {})

        # Check Dispute Type
        pred_disputes = []
        if isinstance(judge_result, dict):
            charges = judge_result.get("dispute_type", [])
            if isinstance(charges, list):
                pred_disputes.extend(charges)
            elif isinstance(charges, str):
                pred_disputes.append(charges)
        elif isinstance(judge_result, list):
            pred_disputes.extend(judge_result)

        if dispute_hit(pred_disputes, true_disputes):
            step4_dispute_correct += 1

        # Check Final Law Article
        pred_laws = []
        if isinstance(judge_result, dict):
            final_law = judge_result.get("law_article", "")
            if isinstance(final_law, list):
                pred_laws.extend(final_law)
            elif isinstance(final_law, str):
                pred_laws.append(final_law)

        pred_law_nums = set()
        for pl in pred_laws:
            pred_law_nums.update(extract_law_numbers(pl))

        if true_law_nums and true_law_nums.intersection(pred_law_nums):
            step4_law_hits += 1

    # Print Metrics
    print("\n[STEP 1] Trích xuất đặc trưng (Feature Extraction)")
    print(
        f"  - Tỷ lệ trích xuất thành công: {step1_success}/{total_cases} ({(step1_success/max(1, total_cases))*100:.1f}%)"
    )

    print("\n[STEP 2] Lấy dữ liệu (Retrieval - Graph Search/Reranker)")
    print(
        f"  - Recall (Có chứa ít nhất 1 điều luật đúng): {step2_law_retrieval_hits}/{total_cases} ({(step2_law_retrieval_hits/max(1, total_cases))*100:.1f}%)"
    )
    print(
        f"  - Precision (Tỷ lệ điều luật đúng / tổng lấy ra): {(step2_precision_sum/max(1, total_cases))*100:.1f}%"
    )
    print(
        f"  - Trung bình số điều luật lấy ra: {step2_avg_retrieved_laws/max(1, total_cases):.1f} điều luật/vụ"
    )

    print("\n[STEP 3] LLM Judge (Lọc điều luật)")
    print(
        f"  - Recall (Giữ lại được ít nhất 1 điều luật đúng): {step3_filtered_laws_hits}/{total_cases} ({(step3_filtered_laws_hits/max(1, total_cases))*100:.1f}%)"
    )
    print(
        f"  - Precision (Tỷ lệ điều luật đúng / tổng giữ lại): {(step3_precision_sum/max(1, total_cases))*100:.1f}%"
    )
    print(
        f"  - Trung bình số điều luật giữ lại để suy luận: {step3_avg_used_laws/max(1, total_cases):.1f} điều luật/vụ"
    )

    filtered_out = step2_avg_retrieved_laws - step3_avg_used_laws
    print(
        f"  - Đã lọc bỏ trung bình: {filtered_out/max(1, total_cases):.1f} điều luật không liên quan mỗi vụ"
    )

    print("\n[STEP 4] Kết quả cuối cùng (Final Output)")
    print(
        f"  - Độ chính xác loại tranh chấp (Dispute Type): {step4_dispute_correct}/{total_cases} ({(step4_dispute_correct/max(1, total_cases))*100:.1f}%)"
    )
    print(
        f"  - Độ chính xác điều luật áp dụng (Law Article): {step4_law_hits}/{total_cases} ({(step4_law_hits/max(1, total_cases))*100:.1f}%)"
    )

    print("\n[STEP 5] Độ tin cậy của mô hình (Model Confidence)")
    print(
        f"  - Trung bình điểm tin cậy tổng thể (Overall): {avg_confidence/max(1, total_cases):.2f}"
    )
    print(
        f"  - Trung bình điểm chất lượng Retrieval: {avg_retrieval_quality/max(1, total_cases):.2f}"
    )
    print(
        f"  - Tỷ lệ ca cần người duyệt (Review Required): {review_required_count}/{total_cases} ({(review_required_count/max(1, total_cases))*100:.1f}%)"
    )
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate detailed step-by-step metrics from evaluate.py output"
    )
    parser.add_argument("--file", type=str, required=True, help="Path to the JSON results file")
    args = parser.parse_args()

    calculate_step_metrics(args.file)
