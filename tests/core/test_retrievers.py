import pytest
import os
import sys

# Ensure core can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.retriever.vector_retriever import VectorRetriever
from core.retriever.graph_retriever import GraphRetriever


class MockModel:
    def generate_response(self, prompt, max_length=256):
        # Mock a valid JSON array for graph retriever
        return '["Cố ý gây thương tích", "Trộm cắp tài sản"]'


@pytest.fixture
def mock_data():
    case = {
        "feature": {
            "parties_info": ["Nguyễn Văn A", "Trần Thị B"],
            "dispute_acts": ["Đánh nhau"],
            "subject_matter": ["Thương tích"],
            "fault_and_evidence": ["Camera ghi hình"],
        },
        "name": "Bản án số 1",
    }
    law_to_dispute = [
        {
            "id": "1",
            "items": [
                {
                    "text": "Tội cố ý gây thương tích",
                    "dispute": ["Cố ý gây thương tích"],
                }
            ],
        }
    ]
    cases_db = [{"id": "case_1", "dispute": ["Cố ý gây thương tích"], "law": ["1"]}]
    return case, law_to_dispute, cases_db


def test_vector_retriever_bm25_caching(mock_data):
    case, law_to_dispute, cases_db = mock_data
    model = MockModel()
    retriever = VectorRetriever(model)

    # BM25 should be None initially
    assert retriever._bm25 is None

    # First retrieve - should build index
    # Note: query_similar_nodes_naive will fail if no DB, so we mock it
    import core.graph_construct.graph_search as fg

    original_func = fg.query_similar_nodes_naive
    original_laws_func = fg.query_similar_laws_naive

    fg.query_similar_nodes_naive = lambda m, q, top_k: [{"caseId": "case_1", "id": "node_1"}]
    fg.query_similar_laws_naive = lambda q, top_k: [{"entry": "1"}]

    try:
        retriever.retrieve(case, law_to_dispute, cases_db)
        # BM25 should now be built
        assert retriever._bm25 is not None

        # Second retrieve - should use cached
        old_bm25 = retriever._bm25
        retriever.retrieve(case, law_to_dispute, cases_db)
        assert retriever._bm25 is old_bm25  # Same instance
    finally:
        fg.query_similar_nodes_naive = original_func
        fg.query_similar_laws_naive = original_laws_func


def test_graph_retriever_safe_parsing(mock_data):
    case, law_to_dispute, cases_db = mock_data
    model = MockModel()
    retriever = GraphRetriever(model)

    # We only test the safe parsing via _retrieve_law_augment
    import core.graph_construct.graph_search as fg

    original_laws_func = fg.query_similar_laws

    # Track what is passed to query_similar_laws
    parsed_disputes = []
    fg.query_similar_laws = lambda d, top_k: (parsed_disputes.append(d) or [{"entry": "1"}])

    try:
        laws = retriever._retrieve_law_augment({"name": "Test", "description": "Fact"})
        # Should parse properly without eval
        assert parsed_disputes[0] == ["Cố ý gây thương tích", "Trộm cắp tài sản"]
        assert len(laws) > 0
    finally:
        fg.query_similar_laws = original_laws_func
