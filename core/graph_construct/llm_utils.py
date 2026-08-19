import requests
import re
from rank_bm25 import BM25Okapi
import json

_bm25_instance = None
_bm25_law_mapping = None


def _get_bm25_index():
    global _bm25_instance, _bm25_law_mapping
    if _bm25_instance is not None:
        return _bm25_instance, _bm25_law_mapping

    try:
        with open("./data/processed/law_to_dispute.json", "r", encoding="utf-8") as f:
            law_to_dispute = json.load(f)

        corpus = []
        law_mapping = []
        for law in law_to_dispute:
            for item in law.get("items", [law]):
                text = item.get("text", "")
                if text:
                    corpus.append(text)
                    law_mapping.append(
                        {
                            "id": law["id"],
                            "entry": str(law["id"]),
                            "text": text,
                            "description": text,
                            "dispute": item.get("dispute", []),
                            "judge_dep": item.get("judge_dep", []),
                            "related_laws": item.get("related_laws", []),
                        }
                    )
        if corpus:
            tokenized_corpus = [doc.lower().split() for doc in corpus]
            _bm25_instance = BM25Okapi(tokenized_corpus)
            _bm25_law_mapping = law_mapping
    except Exception as e:
        print(f"BM25 Graph Init Error: {e}")

    return _bm25_instance, _bm25_law_mapping


def get_embedding(text):
    import os
    import time

    url = os.getenv("embedding_api_url", "http://localhost:11434/api/embed")
    model = os.getenv("embedding_model", "bge-m3")
    api_key = os.getenv("OPENAI_API_KEY", os.getenv("api_key", ""))

    headers = {"Content-Type": "application/json"}
    if not text:
        return None

    if "openai.com" in url or api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        # text-embedding-3-small max length is 8192 tokens.
        # Using ~8000 characters as a safe truncation limit to prevent 400 Bad Request.
        if len(text) > 8000:
            text = text[:8000]

    data = {"model": model, "input": text}

    max_retries = 10
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=data, headers=headers)

            if response.status_code == 200:
                result = response.json()
                if "data" in result and len(result["data"]) > 0:
                    return result["data"][0].get("embedding", [])
                return result.get("embeddings", [[]])[0]
            elif response.status_code == 429:
                if attempt < max_retries - 1:
                    sleep_time = base_delay * (1.5**attempt)  # Exponential backoff
                    # print(f"Rate limit (429). Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                    continue
                else:
                    print(f"Error 429: Rate limit exceeded after {max_retries} retries.")
                    return None
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                return None
        except Exception as e:
            if attempt < max_retries - 1:
                sleep_time = base_delay * (1.5**attempt)
                time.sleep(sleep_time)
            else:
                print(f"Failed to get embedding due to exception: {e}")
                return None


def summarize_texts(model, text):
    from core.prompt import get_prompt

    return model.generate_response(
        get_prompt("SUMMARIZE_TEXTS_PROMPT") + get_prompt("SUMMARIZE_TEXTS_INPUT_PREFIX") + text,
        max_length=512,
    ).strip()


def rerank_clusters(model, clusters, query_text):
    from core.prompt import get_prompt

    cluster_summaries = "\n".join([f"code{c['code']}：{c['summary']}\n" for c in clusters])
    prompt = get_prompt("RERANK_CLUSTERS_PROMPT_TEMPLATE").format(
        cluster_summaries=cluster_summaries, query_text=query_text
    )
    response = model.generate_response(prompt, max_length=512)
    match = re.search(r"rank: \[([\d,]+)\]", response)
    # print(f"Model response: {response}")
    if match:
        ranked_codes = [int(code) for code in match.group(1).split(",")]
        return ranked_codes
    return [0]


def rerank(model, query_text, neighbors):
    from core.prompt import get_prompt

    if not neighbors:
        return []

    neighbor_summaries = "\n".join(
        [
            (
                f"code{n['rank']}：{n['description'][:500]}..."
                if len(n["description"]) > 500
                else f"code{n['rank']}：{n['description']}"
            )
            for n in neighbors
        ]
    )
    prompt = get_prompt("RERANK_PROMPT_TEMPLATE").format(
        neighbor_summaries=neighbor_summaries, query_text=query_text
    )
    response = model.generate_response(prompt, max_length=512)
    try:
        first_bracket = response.find("[")
        last_bracket = response.rfind("]")
        if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
            list_str = response[first_bracket : last_bracket + 1]
            ranked_indices = eval(list_str)
        else:
            ranked_indices = []
    except Exception as e:
        print(f"Error parsing response: {e}")
        ranked_indices = []
    if not ranked_indices:
        return neighbors[:3]
    neighbors = [n for n in neighbors if n["rank"] in ranked_indices]
    neighbors = sorted(neighbors, key=lambda x: ranked_indices.index(x["rank"]))
    return neighbors
