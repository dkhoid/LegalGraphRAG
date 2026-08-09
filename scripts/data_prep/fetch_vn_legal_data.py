"""
Fetch Vietnamese legal documents from HuggingFace dataset th1nhng0/vietnamese-legal-documents.
Specifically targets missing laws from the Zalo corpus (BLLĐ 2019, Luật BHXH 2014)
and formats them with Zalo IDs to merge perfectly into the knowledge graph.
"""

import json
import os
import re
import sys
from html.parser import HTMLParser

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.utils.logger import logger


class HTMLTextExtractor(HTMLParser):
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


def parse_articles_from_html(html_content, law_name=""):
    parser = HTMLTextExtractor()
    parser.feed(html_content)
    text_content = parser.get_text()

    pattern = re.compile(r"(Điều\s+\d+[a-zA-Z]*\.)")
    parts = pattern.split(text_content)

    articles = []
    if len(parts) > 1:
        for i in range(1, len(parts), 2):
            art_title = parts[i].strip()
            art_body = parts[i + 1].strip()

            num_match = re.search(r"\d+[a-zA-Z]*", art_title)
            art_num = num_match.group(0) if num_match else "unknown"

            body_lines = [line.strip() for line in art_body.split("\n") if line.strip()]
            name_ext = body_lines[0] if body_lines else ""
            full_name = f"{art_title} {name_ext}"
            text_body = "\n".join(body_lines[1:])

            # Clean up
            text_body = re.sub(r"\n\s*\n+", "\n", text_body).strip()

            if len(text_body) > 20:
                articles.append({"num": art_num, "name": full_name, "text": text_body})
    return articles


def main():
    from datasets import load_dataset
    import pyarrow.parquet as pq
    import glob

    raw_data_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    os.makedirs(raw_data_dir, exist_ok=True)
    corpus_path = os.path.join(raw_data_dir, "law_corpus.jsonl")

    logger.info("Loading th1nhng0 metadata...")
    ds_metadata = load_dataset("th1nhng0/vietnamese-legal-documents", "metadata", split="data")

    # Target specific laws missing from Zalo
    targets = {"45/2019/qh14": "Bộ luật Lao động 2019", "58/2014/qh13": "Luật Bảo hiểm xã hội 2014"}

    matched_docs = []

    # Identify title and id cols
    title_key = "title" if "title" in ds_metadata.column_names else "ten_van_ban"
    id_key = "id" if "id" in ds_metadata.column_names else "document_id"
    skh_key = "so_ky_hieu"

    for i in range(len(ds_metadata)):
        row = ds_metadata[i]
        skh = str(row.get(skh_key, "")).lower()
        title = str(row.get(title_key, "")).lower()

        for t_skh, t_name in targets.items():
            if t_skh in skh:
                matched_docs.append(
                    {
                        "index": i,
                        "id": row.get(id_key),
                        "title": row.get(title_key, ""),
                        "zalo_prefix": f"zalo_{t_skh}",
                    }
                )

    logger.info(f"Found {len(matched_docs)} matching documents.")
    for doc in matched_docs:
        logger.info(f"  {doc['title']} -> {doc['zalo_prefix']}")

    logger.info("Loading th1nhng0 content via streaming...")
    ds_content = load_dataset(
        "th1nhng0/vietnamese-legal-documents", "content", split="data", streaming=True
    )

    needed_ids = {str(doc["id"]) for doc in matched_docs}
    content_by_id = {}

    for row in ds_content:
        row_id = str(row.get(id_key, ""))
        if row_id in needed_ids:
            content_col = row.get("content_html") or row.get("content") or ""
            content_by_id[row_id] = content_col
            if len(content_by_id) == len(needed_ids):
                break

    # Process and append to corpus
    new_entries = []
    for doc in matched_docs:
        html_content = content_by_id.get(str(doc["id"]))
        if not html_content:
            continue

        articles = parse_articles_from_html(html_content, doc["title"])
        logger.info(f"Parsed {len(articles)} articles from {doc['title']}")

        for art in articles:
            new_entries.append(
                {
                    "text_id": f"{doc['zalo_prefix']}+{art['num']}",
                    "name": art["name"],
                    "text": art["text"],
                }
            )

    # Read existing corpus to avoid duplicates (if rerun)
    existing_ids = set()
    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_ids.add(json.loads(line)["text_id"])

    added_count = 0
    with open(corpus_path, "a", encoding="utf-8") as f:
        for entry in new_entries:
            if entry["text_id"] not in existing_ids:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                added_count += 1
                existing_ids.add(entry["text_id"])

    logger.info(f"Successfully appended {added_count} new formatted articles to {corpus_path}")


if __name__ == "__main__":
    main()
