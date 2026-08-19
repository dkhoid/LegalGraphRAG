from unittest.mock import MagicMock, patch
from core.graph_construct.graph_search import search_similar_nodes_direct


@patch("core.graph_construct.graph_search.GraphDBManager")
@patch("core.graph_construct.graph_search.rerank")
def test_search_similar_nodes_direct(mock_rerank, mock_db_manager):
    # Setup mock DB methods
    mock_db = mock_db_manager.get_db.return_value
    mock_db.find_similar_nodes.return_value = [
        {"id": "node_1", "similarity": 0.9, "caseId": "case_1"}
    ]
    mock_db.get_neighbors.return_value = ["law_1"]
    mock_db.get_node.return_value = {"entry": "test_law_entry"}

    mock_model = MagicMock()
    mock_rerank.return_value = [{"id": "node_1", "similarity": 0.9, "caseId": "case_1"}]

    cases, laws = search_similar_nodes_direct(
        mock_model, query_embedding=[0.1] * 1536, query_text="Test", top_k=2
    )

    assert len(cases) == 1
    assert cases[0]["caseId"] == "case_1"
    assert len(laws) == 1
    assert laws[0]["entry"] == "test_law_entry"
