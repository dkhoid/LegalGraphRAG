from unittest.mock import MagicMock, patch
from core.graph_construct.graph_builder import construct_feature_graph


@patch("core.graph_construct.graph_builder.get_embedding")
@patch("core.graph_construct.graph_builder.store_nodes_with_embeddings")
@patch("core.graph_construct.graph_builder.build_relationships")
@patch("core.graph_construct.graph_builder.run_knn")
@patch("core.graph_construct.graph_builder.create_clusters")
def test_construct_feature_graph(
    mock_clusters, mock_knn, mock_rels, mock_store, mock_get_embedding
):
    mock_get_embedding.return_value = [0.1] * 1536
    # Setup test data
    nodes_data = {
        "case": [
            {"id": "1", "caseId": "1", "description": "Desc 1"},
            {"id": "2", "caseId": "2", "description": "Desc 2"},
        ],
        "law": [],
        "dispute": [],
    }

    mock_model = MagicMock()
    # Mock embedding function returning matching length vector
    mock_model.embedding.side_effect = lambda texts: [[0.1] * 1536 for _ in texts]

    # Mock GraphDBManager to bypass neo4j check
    with patch("core.graph_construct.graph_builder.GraphDBManager") as mock_db:
        mock_db.return_value.query.return_value = []  # Neo4j is empty, should rebuild

        construct_feature_graph(mock_model, nodes_data)

        # Verify the model was called
        assert mock_get_embedding.call_count > 0

        # Verify the pipeline stages were called
        mock_store.assert_called_once()
        mock_rels.assert_called_once()
        mock_knn.assert_called_once()
        mock_clusters.assert_called_once()
