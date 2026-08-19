import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Ensure core can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.retriever.reranker import CrossEncoderReranker, get_reranker


class MockCrossEncoder:
    def __init__(self, model_name):
        self.model_name = model_name

    def predict(self, pairs, batch_size=32):
        # We mock predicting scores: just return the index or something simple
        # Let's say if the pair has "match" in the description, we score high.
        # Otherwise, score low.
        scores = []
        for q, doc in pairs:
            if "match" in doc.lower():
                scores.append(0.9)
            elif "partial" in doc.lower():
                scores.append(0.5)
            else:
                scores.append(0.1)
        return scores


@pytest.fixture
def mock_sentence_transformers():
    """Mock the sentence_transformers module so we don't download models during tests."""
    with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}):
        import sentence_transformers

        sentence_transformers.CrossEncoder = MockCrossEncoder
        yield sentence_transformers


@pytest.fixture
def sample_laws():
    return [
        {"id": "1", "description": "This is a random law"},
        {"id": "2", "description": "This law has a perfect match for the query"},
        {"id": "3", "text": "This is a partial law text"},
        {"id": "4", "description": "Another random unrelated law"},
    ]


def test_cross_encoder_lazy_load(mock_sentence_transformers):
    reranker = CrossEncoderReranker(model_name="test-model")
    assert reranker._model is None

    reranker._load()
    assert reranker._model is not None
    assert reranker._model.model_name == "test-model"


def test_reranker_sorting_and_top_k(mock_sentence_transformers, sample_laws):
    reranker = CrossEncoderReranker(model_name="test-model", top_k=2)

    query = "Find the matching law"
    results = reranker.rerank(query, sample_laws)

    assert len(results) == 2
    # The one with "match" should be first (score 0.9)
    assert results[0]["id"] == "2"
    assert results[0]["_rerank_score"] == 0.9

    # The one with "partial" should be second (score 0.5)
    assert results[1]["id"] == "3"
    assert results[1]["_rerank_score"] == 0.5


def test_reranker_empty_input(mock_sentence_transformers):
    reranker = CrossEncoderReranker()
    assert reranker.rerank("query", []) == []


def test_reranker_missing_text_keys(mock_sentence_transformers):
    reranker = CrossEncoderReranker(top_k=5)
    bad_laws = [{"id": "1"}, {"id": "2", "description": "match law"}]  # missing description/text

    results = reranker.rerank("query", bad_laws)
    assert len(results) == 2
    assert results[0]["id"] == "2"
    assert results[1]["id"] == "1"


def test_get_reranker_singleton(mock_sentence_transformers):
    import core.retriever.reranker as r_mod

    # reset singleton
    r_mod._default_reranker = None

    r1 = get_reranker("model-1")
    r2 = get_reranker("model-1")
    assert r1 is r2

    # Changing model name should create a new instance
    r3 = get_reranker("model-2")
    assert r1 is not r3
    assert r3.model_name == "model-2"
