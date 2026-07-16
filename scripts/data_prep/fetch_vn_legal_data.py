from core.utils.logger import logger

"""
Fetch Vietnamese legal documents from HuggingFace dataset th1nhng0/vietnamese-legal-documents
and convert them into LegalGraphRAG format.

Targets: Civil Code, Labor Code, Social Insurance Law
Output: data/processed/law_to_dispute.json, data/raw/law_corpus.jsonl
"""

import json
import os
import re
import sys
from html.parser import HTMLParser

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class HTMLTextExtractor(HTMLParser):
    """Simple HTML to text converter."""

    def __init__(self):
        super().__init__()
        self.result = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True
        if tag in ("br", "p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"):
            self.result.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False
        if tag in ("p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"):
            self.result.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.result.append(data)

    def get_text(self):
        return "".join(self.result)


def html_to_text(html_content):
    """Convert HTML content to plain text."""
    if not html_content:
        return ""
    parser = HTMLTextExtractor()
    try:
        parser.feed(html_content)
        return parser.get_text().strip()
    except Exception:
        # Fallback: strip tags with regex
        text = re.sub(r"<[^>]+>", "\n", html_content)
        return text.strip()


def parse_articles_from_text(text, law_name=""):
    """
    Parse individual articles (Điều) from a Vietnamese legal document text.
    Returns a list of dicts: [{"article_number": int, "text": str}, ...]
    """
    articles = []

    # Pattern to match "Điều X." or "Điều X:" or "Điều X " followed by title
    # Vietnamese law format: "Điều 1. Phạm vi điều chỉnh"
    pattern = r"(?:^|\n)\s*(Điều\s+(\d+)[a-z]?)\s*[\.:\s]+"
    matches = list(re.finditer(pattern, text, re.MULTILINE))

    if not matches:
        # Try alternative patterns
        pattern = r"(?:^|\n)\s*(Điều\s+(\d+))\b"
        matches = list(re.finditer(pattern, text, re.MULTILINE))

    for i, match in enumerate(matches):
        article_num = int(match.group(2))
        start = match.start()

        # Get text until next article or end of document
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        article_text = text[start:end].strip()

        # Clean up the text
        article_text = re.sub(r"\n\s*\n+", "\n", article_text)
        article_text = article_text.strip()

        if len(article_text) > 20:  # Skip very short/empty articles
            articles.append(
                {
                    "article_number": article_num,
                    "text": article_text,
                    "law_name": law_name,
                }
            )

    return articles


def classify_article_issue(article_text, article_num, law_name):
    """
    Classify a legal article into issue categories based on keywords.
    Returns a list of issue names (Vietnamese).
    """
    text_lower = article_text.lower()
    issues = []

    if "dân sự" in law_name.lower() or "civil" in law_name.lower():
        # Civil Code classifications
        keyword_map = {
            "Hợp đồng": ["hợp đồng", "giao kết", "thỏa thuận", "ký kết"],
            "Bồi thường thiệt hại": ["bồi thường", "thiệt hại", "tổn thất"],
            "Quyền sở hữu": ["sở hữu", "tài sản", "chiếm hữu", "định đoạt"],
            "Thừa kế": ["thừa kế", "di chúc", "di sản", "người thừa kế"],
            "Nghĩa vụ dân sự": ["nghĩa vụ", "trách nhiệm dân sự", "thực hiện nghĩa vụ"],
            "Giao dịch dân sự": ["giao dịch", "hành vi pháp lý", "vô hiệu"],
            "Quyền nhân thân": ["nhân thân", "danh dự", "nhân phẩm", "uy tín", "bí mật đời tư"],
            "Đại diện": ["đại diện", "ủy quyền", "giám hộ"],
            "Thế chấp, cầm cố": ["thế chấp", "cầm cố", "bảo đảm", "đặt cọc", "ký cược"],
            "Hợp đồng vay tài sản": ["vay", "cho vay", "lãi suất", "trả nợ"],
            "Hợp đồng mua bán": ["mua bán", "chuyển nhượng", "giá bán"],
            "Hợp đồng thuê": ["thuê", "cho thuê", "tiền thuê"],
            "Hợp đồng lao vụ": ["dịch vụ", "gia công", "vận chuyển"],
        }
        for issue, keywords in keyword_map.items():
            if any(kw in text_lower for kw in keywords):
                issues.append(issue)

    elif "lao động" in law_name.lower() or "labor" in law_name.lower():
        keyword_map = {
            "Hợp đồng lao động": ["hợp đồng lao động", "giao kết", "ký hợp đồng"],
            "Tiền lương": ["tiền lương", "lương", "trả lương", "tiền công"],
            "Thời giờ làm việc": ["thời giờ", "giờ làm", "làm thêm", "tăng ca", "nghỉ phép"],
            "Sa thải, chấm dứt HĐLĐ": ["sa thải", "chấm dứt", "đơn phương", "kỷ luật"],
            "Bảo hiểm xã hội (LĐ)": ["bảo hiểm", "bhxh", "thai sản", "ốm đau"],
            "An toàn lao động": ["an toàn", "vệ sinh lao động", "tai nạn lao động"],
            "Tranh chấp lao động": ["tranh chấp", "đình công", "hòa giải"],
            "Lao động nữ": ["lao động nữ", "mang thai", "nuôi con nhỏ"],
            "Lao động chưa thành niên": ["chưa thành niên", "lao động trẻ em"],
        }
        for issue, keywords in keyword_map.items():
            if any(kw in text_lower for kw in keywords):
                issues.append(issue)

    elif "bảo hiểm" in law_name.lower() or "insurance" in law_name.lower():
        keyword_map = {
            "BHXH bắt buộc": ["bắt buộc", "đóng bhxh", "mức đóng"],
            "BHXH tự nguyện": ["tự nguyện"],
            "Chế độ hưu trí": ["hưu trí", "lương hưu", "tuổi nghỉ hưu"],
            "Chế độ ốm đau": ["ốm đau", "nghỉ ốm"],
            "Chế độ thai sản": ["thai sản", "sinh con", "nuôi con"],
            "Chế độ tai nạn LĐ": ["tai nạn lao động", "bệnh nghề nghiệp"],
            "Chế độ tử tuất": ["tử tuất", "chết", "mai táng"],
            "Bảo hiểm thất nghiệp": ["thất nghiệp", "mất việc"],
            "Bảo hiểm y tế": ["bảo hiểm y tế", "bhyt", "khám chữa bệnh"],
        }
        for issue, keywords in keyword_map.items():
            if any(kw in text_lower for kw in keywords):
                issues.append(issue)

    if not issues:
        issues = ["Quy định chung"]

    return issues


def generate_judge_dep(article_text):
    """
    Generate simple judgment dependencies (key conditions) from article text.
    """
    deps = []
    # Look for conditional patterns in Vietnamese legal text
    patterns = [
        r"(?:nếu|trường hợp|khi|trong trường hợp)\s+(.{10,80}?)(?:\.|,|;|thì)",
        r"(?:phải|được|không được|có quyền|có nghĩa vụ)\s+(.{10,60}?)(?:\.|,|;)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, article_text.lower())
        for m in matches[:3]:  # Max 3 deps per pattern
            dep = m.strip()
            if len(dep) > 10:
                deps.append(dep)

    return deps[:5]  # Max 5 deps total


def main():
    from datasets import load_dataset
    import pyarrow.parquet as pq
    import glob

    output_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    raw_data_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(raw_data_dir, exist_ok=True)

    # Backup old data
    backup_dir = os.path.join(PROJECT_ROOT, "data", "raw_backup")
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir, exist_ok=True)
        for fname in os.listdir(raw_data_dir):
            src = os.path.join(raw_data_dir, fname)
            dst = os.path.join(backup_dir, fname)
            if os.path.isfile(src) and not os.path.exists(dst):
                import shutil

                shutil.copy2(src, dst)
        logger.info(f"Backed up old data to {backup_dir}")

    logger.info("Loading dataset from HuggingFace...")
    logger.info("Loading metadata config...")
    ds_metadata = load_dataset("th1nhng0/vietnamese-legal-documents", "metadata", split="data")

    logger.info(f"Total documents in metadata: {len(ds_metadata)}")
    logger.info(f"Metadata columns: {ds_metadata.column_names}")

    # Print a sample to understand the structure
    sample = ds_metadata[0]
    logger.info(f"\nSample metadata entry keys: {list(sample.keys())}")
    for key, val in sample.items():
        val_str = str(val)[:200]
        logger.info(f"  {key}: {val_str}")

    # Search for our target laws
    target_keywords = [
        # Civil Code
        "bộ luật dân sự",
        "luật dân sự",
        # Labor Code
        "bộ luật lao động",
        "luật lao động",
        # Insurance
        "luật bảo hiểm xã hội",
        "luật bảo hiểm y tế",
        "bảo hiểm xã hội",
    ]

    # Find matching documents
    matched_docs = []
    title_key = None
    # Try to find the title column
    for candidate in ["title", "ten_van_ban", "name", "ten", "document_title"]:
        if candidate in ds_metadata.column_names:
            title_key = candidate
            break

    if title_key is None:
        # Fallback: print all columns and try first string column
        logger.info("\nNo obvious title column found. Columns available:")
        logger.info(ds_metadata.column_names)
        # Try to use the first column that looks like a title
        for col in ds_metadata.column_names:
            sample_val = str(ds_metadata[0][col]).lower()
            if any(kw in sample_val for kw in ["luật", "bộ luật", "nghị định"]):
                title_key = col
                logger.info(f"Using column '{col}' as title column based on content")
                break

    if title_key is None:
        logger.info("\nERROR: Could not identify title column. Available columns:")
        for col in ds_metadata.column_names:
            logger.info(f"  {col}: {str(ds_metadata[0][col])[:100]}")
        sys.exit(1)

    logger.info(f"\nUsing '{title_key}' as title column")
    logger.info(f"Searching for documents matching: {target_keywords}")

    for i in range(len(ds_metadata)):
        row = ds_metadata[i]
        title = str(row.get(title_key, "")).lower()
        if any(kw in title for kw in target_keywords):
            matched_docs.append(
                {
                    "index": i,
                    "title": row.get(title_key, ""),
                    "row": row,
                }
            )

    logger.info(f"\nFound {len(matched_docs)} matching documents:")
    for doc in matched_docs:
        logger.info(f"  [{doc['index']}] {doc['title']}")

    if not matched_docs:
        logger.info("\nNo matching documents found. Listing all document titles for debugging:")
        for i in range(min(50, len(ds_metadata))):
            logger.info(f"  [{i}] {ds_metadata[i].get(title_key, 'N/A')}")
        sys.exit(1)

    # --- Load content via pyarrow chunked read (memory efficient) ---
    logger.info("\nLoading content parquet via chunked pyarrow read...")
    hf_cache = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--th1nhng0--vietnamese-legal-documents"
    )
    content_parquet_files = glob.glob(
        os.path.join(hf_cache, "**", "content.parquet"), recursive=True
    )
    if not content_parquet_files:
        content_parquet_files = glob.glob(os.path.join(hf_cache, "**", "*.parquet"), recursive=True)
        content_parquet_files = [
            f for f in content_parquet_files if "content" in os.path.basename(f)
        ]

    if not content_parquet_files:
        logger.info("ERROR: Could not find content.parquet in HuggingFace cache.")
        sys.exit(1)

    content_parquet_path = content_parquet_files[0]
    logger.info(f"Reading (chunked): {content_parquet_path}")

    # Sniff schema first — zero memory cost
    pf = pq.ParquetFile(content_parquet_path)
    schema = pf.schema_arrow
    content_columns = schema.names
    logger.info(f"Content columns: {content_columns}")

    # Determine ID and content column names
    id_key = None
    for candidate in ["id", "document_id", "doc_id", "ma_van_ban"]:
        if candidate in ds_metadata.column_names and candidate in content_columns:
            id_key = candidate
            break

    content_col_name = None
    for col in ["content", "content_html", "noi_dung", "text", "html", "body"]:
        if col in content_columns:
            content_col_name = col
            break
    if content_col_name is None:
        logger.info(f"ERROR: No content column found. Available: {content_columns}")
        sys.exit(1)

    # Build the set of IDs or indices we actually need
    if id_key:
        needed_ids = {str(doc["row"].get(id_key, "")) for doc in matched_docs}
        logger.info(f"Matching by ID '{id_key}' — need {len(needed_ids)} docs out of 153k+")
        read_cols = [id_key, content_col_name]
    else:
        needed_indices = {doc["index"] for doc in matched_docs}
        logger.info(f"Matching by row index — need {len(needed_indices)} docs")
        read_cols = [content_col_name]

    # Stream through parquet in batches — only keep matched rows
    content_by_id = {}
    content_by_index = {}
    global_row = 0
    CHUNK = 2000  # rows per batch

    for batch in pf.iter_batches(batch_size=CHUNK, columns=read_cols):
        batch_len = batch.num_rows
        if id_key:
            ids = batch.column(id_key).to_pylist()
            texts = batch.column(content_col_name).to_pylist()
            for i in range(batch_len):
                sid = str(ids[i])
                if sid in needed_ids:
                    content_by_id[sid] = texts[i]
                    if len(content_by_id) == len(needed_ids):
                        break  # found all, stop early
        else:
            texts = batch.column(content_col_name).to_pylist()
            for i in range(batch_len):
                idx = global_row + i
                if idx in needed_indices:
                    content_by_index[idx] = texts[i]
            global_row += batch_len

        # Early exit if we found everything
        if id_key and len(content_by_id) == len(needed_ids):
            logger.info(f"  Found all {len(needed_ids)} docs early — stopping scan")
            break
        elif not id_key and len(content_by_index) == len(needed_indices):
            logger.info(f"  Found all {len(needed_indices)} docs early — stopping scan")
            break

    logger.info(
        f"  Retrieved {len(content_by_id) if id_key else len(content_by_index)} content entries"
    )

    # Process each matched document
    all_articles = []
    law_to_dispute_entries = []
    law_corpus_entries = []
    article_id_counter = 1

    for doc in matched_docs:
        title = doc["title"]
        idx = doc["index"]
        content_text = None

        if id_key:
            doc_id = str(doc["row"].get(id_key, ""))
            content_text = content_by_id.get(doc_id)
        else:
            content_text = content_by_index.get(idx)

        if not content_text:
            logger.info(f"  WARNING: No content found for '{title}'")
            continue

        # Convert HTML to text
        plain_text = html_to_text(str(content_text))
        logger.info(f"\n  Processing '{title}' ({len(plain_text)} chars)...")

        # Parse articles
        articles = parse_articles_from_text(plain_text, law_name=title)
        logger.info(f"    Found {len(articles)} articles (Điều)")

        # Create law_to_dispute entries
        for article in articles:
            article_num = article["article_number"]
            article_text = article["text"]
            issues = classify_article_issue(article_text, article_num, title)
            judge_deps = generate_judge_dep(article_text)

            entry = {
                "id": article_id_counter,
                "items": [
                    {
                        "text": article_text,
                        "dispute": issues,
                        "judge_dep": judge_deps,
                        "related_laws": [],
                    }
                ],
            }
            law_to_dispute_entries.append(entry)

            # Law corpus entry
            short_title = title.split(" - ")[0] if " - " in title else title
            law_corpus_entries.append(
                {
                    "text_id": str(article_id_counter),
                    "text": article_text,
                    "name": f"{short_title} Điều {article_num}",
                }
            )

            article_id_counter += 1
            all_articles.append(article)

    # Save outputs
    law_to_dispute_path = os.path.join(output_dir, "law_to_dispute.json")
    with open(law_to_dispute_path, "w", encoding="utf-8") as f:
        json.dump(law_to_dispute_entries, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSaved {len(law_to_dispute_entries)} law entries to {law_to_dispute_path}")

    law_corpus_path = os.path.join(raw_data_dir, "law_corpus.jsonl")
    with open(law_corpus_path, "w", encoding="utf-8") as f:
        for entry in law_corpus_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(law_corpus_entries)} corpus entries to {law_corpus_path}")

    # Also save the raw criminal_law_processed.json equivalent
    criminal_law_path = os.path.join(raw_data_dir, "criminal_law_processed.json")
    with open(criminal_law_path, "w", encoding="utf-8") as f:
        json.dump(law_to_dispute_entries, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved criminal_law_processed.json to {criminal_law_path}")

    # Save raw law_to_dispute mapping (simplified)
    raw_law_to_dispute = []
    for entry in law_to_dispute_entries:
        disputes = entry["items"][0].get("dispute", []) if entry["items"] else []
        raw_law_to_dispute.append({"id": entry["id"], "dispute": disputes})

    raw_ltd_path = os.path.join(raw_data_dir, "law_to_dispute.json")
    with open(raw_ltd_path, "w", encoding="utf-8") as f:
        json.dump(raw_law_to_dispute, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved raw law_to_dispute.json to {raw_ltd_path}")

    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY:")
    logger.info(f"  Documents processed: {len(matched_docs)}")
    logger.info(f"  Articles extracted: {len(all_articles)}")
    logger.info(f"  Law entries created: {len(law_to_dispute_entries)}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
