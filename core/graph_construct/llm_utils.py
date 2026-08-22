import requests
import re
from rank_bm25 import BM25Okapi
import json
import ast
import os
import time
from core.utils.logger import logger

import hashlib
import pickle as _pickle

_bm25_instance = None
_bm25_law_mapping = None
_BM25_CACHE_PATH = "./data/clean/.bm25_cache.pkl"

# V7: Simple in-memory embedding cache (LRU, max 2000 entries)
_embedding_cache: dict = {}
_EMBEDDING_CACHE_MAX = 2000


def _text_hash(text: str) -> str:
    """Create a short hash key for caching."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def _tokenize_vi(text: str) -> list:
    """Tokenize Vietnamese text for BM25.

    Priority:
    1. ``underthesea.word_tokenize`` – best quality for Vietnamese
    2. ``pyvi.ViTokenizer.tokenize`` – lightweight alternative
    3. Whitespace split – original fallback (always works)

    The result is always a list of lowercase strings.
    """
    text = text.lower()
    try:
        from underthesea import word_tokenize  # type: ignore

        return word_tokenize(text, format="text").split()
    except Exception:
        pass
    try:
        from pyvi import ViTokenizer  # type: ignore

        return ViTokenizer.tokenize(text).split()
    except Exception:
        pass
    return text.split()


def _get_bm25_index():
    global _bm25_instance, _bm25_law_mapping
    if _bm25_instance is not None:
        return _bm25_instance, _bm25_law_mapping

    # V10: Try loading from cache first
    if os.path.exists(_BM25_CACHE_PATH):
        try:
            with open(_BM25_CACHE_PATH, "rb") as f:
                cached = _pickle.load(f)
            _bm25_instance = cached["bm25"]
            _bm25_law_mapping = cached["law_mapping"]
            logger.info("BM25 index loaded from cache.")
            return _bm25_instance, _bm25_law_mapping
        except Exception as e:
            logger.warning(f"BM25 cache load failed: {e}. Rebuilding from source.")

    candidate_paths = [
        os.getenv("law_to_dispute_path", ""),
        "./data/clean/law_to_dispute_clean.json",
        "./data/processed/law_to_dispute.json",
    ]
    file_path = None
    for p in candidate_paths:
        if p and os.path.exists(p):
            file_path = p
            break

    if not file_path:
        logger.warning("BM25 Graph Init: No law_to_dispute dataset file found.")
        return None, None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
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
            tokenized_corpus = [_tokenize_vi(doc) for doc in corpus]
            _bm25_instance = BM25Okapi(tokenized_corpus)
            _bm25_law_mapping = law_mapping

            # V10: Save to disk cache
            try:
                os.makedirs(os.path.dirname(_BM25_CACHE_PATH), exist_ok=True)
                with open(_BM25_CACHE_PATH, "wb") as f:
                    _pickle.dump({"bm25": _bm25_instance, "law_mapping": _bm25_law_mapping}, f)
                logger.info("BM25 index cached to disk.")
            except Exception as e:
                logger.warning(f"BM25 cache save failed: {e}")
    except Exception as e:
        logger.error(f"BM25 Graph Init Error: {e}")

    return _bm25_instance, _bm25_law_mapping


def get_embedding(text):
    if not text:
        return None

    # V7: Check memory cache first
    cache_key = _text_hash(text)
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    url = os.getenv("embedding_api_url", "http://localhost:11434/api/embed")
    model = os.getenv("embedding_model", "bge-m3")
    api_key = os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", os.getenv("api_key", "")))

    headers = {"Content-Type": "application/json"}

    if "openai.com" in url or api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        # text-embedding-3-small max length is 8192 tokens.
        # Using ~8000 characters as a safe truncation limit to prevent 400 Bad Request.
        if len(text) > 8000:
            text = text[:8000]

    data = {"model": model, "input": text}

    max_retries = 3
    base_delay = 0.5

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=data, headers=headers, timeout=10.0)

            if response.status_code == 200:
                result = response.json()
                emb = None
                if "data" in result and len(result["data"]) > 0:
                    emb = result["data"][0].get("embedding", [])
                else:
                    emb = result.get("embeddings", [[]])[0]

                if emb:
                    # Store in LRU cache
                    if len(_embedding_cache) >= _EMBEDDING_CACHE_MAX:
                        keys_to_remove = list(_embedding_cache.keys())[: _EMBEDDING_CACHE_MAX // 10]
                        for k in keys_to_remove:
                            _embedding_cache.pop(k, None)
                    _embedding_cache[cache_key] = emb
                return emb
            elif response.status_code == 429:
                if attempt < max_retries - 1:
                    sleep_time = base_delay * (1.5**attempt)
                    time.sleep(sleep_time)
                    continue
                else:
                    logger.error(
                        f"Embedding API Error 429: Rate limit exceeded after {max_retries} retries."
                    )
                    return None
            else:
                logger.error(f"Embedding API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            if attempt < max_retries - 1:
                sleep_time = base_delay * (1.5**attempt)
                time.sleep(sleep_time)
            else:
                logger.warning(f"Failed to get embedding due to exception: {e}")
                return None


def get_embeddings_batch(texts: list) -> list:
    """Batch embedding: send multiple texts in one API call with cache support (V6 fix).

    Falls back to sequential get_embedding() if batch is not supported.
    """
    if not texts:
        return []

    results = [None] * len(texts)
    uncached_indices = []
    for i, text in enumerate(texts):
        if not text:
            continue
        cache_key = _text_hash(text)
        if cache_key in _embedding_cache:
            results[i] = _embedding_cache[cache_key]
        else:
            uncached_indices.append(i)

    if not uncached_indices:
        return results

    url = os.getenv("embedding_api_url", "http://localhost:11434/api/embed")
    api_key = os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", os.getenv("api_key", "")))

    if ("openai.com" in url or api_key) and len(uncached_indices) > 1:
        try:
            model = os.getenv("embedding_model", "bge-m3")
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
            batch_texts = [texts[i][:8000] for i in uncached_indices]
            data = {"model": model, "input": batch_texts}
            response = requests.post(url, json=data, headers=headers, timeout=30.0)
            if response.status_code == 200:
                result = response.json()
                data_items = result.get("data", [])
                if data_items and len(data_items) == len(uncached_indices):
                    for j, idx in enumerate(uncached_indices):
                        emb = data_items[j].get("embedding", [])
                        results[idx] = emb
                        if emb:
                            _embedding_cache[_text_hash(texts[idx])] = emb
                    return results
        except Exception as e:
            logger.warning(f"Batch embedding failed: {e}. Falling back to sequential.")

    # Fallback: sequential calls
    for idx in uncached_indices:
        results[idx] = get_embedding(texts[idx])

    return results


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
            try:
                ranked_indices = json.loads(list_str)
            except Exception:
                ranked_indices = ast.literal_eval(list_str)
        else:
            ranked_indices = []
    except Exception as e:
        logger.warning(f"Error parsing rerank response: {e}")
        ranked_indices = []
    if not ranked_indices:
        return neighbors[:3]
    neighbors = [n for n in neighbors if n["rank"] in ranked_indices]
    neighbors = sorted(neighbors, key=lambda x: ranked_indices.index(x["rank"]))
    return neighbors


def generate_hyde_query(model, query_text):
    from core.prompt import get_prompt

    prompt = get_prompt("HYDE_PROMPT").format(query_text=query_text)
    response = model.generate_response(prompt, max_length=512)
    return response.strip()
