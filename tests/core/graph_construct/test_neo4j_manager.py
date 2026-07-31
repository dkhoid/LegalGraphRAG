import unittest
from unittest.mock import patch, MagicMock
from core.graph_construct.neo4j_manager import Neo4jManager


class TestNeo4jManager(unittest.TestCase):
    @patch.dict(
        "os.environ", {"NEO4J_URI": "bolt://fake", "NEO4J_USER": "u", "NEO4J_PASSWORD": "p"}
    )
    @patch("core.graph_construct.neo4j_manager.GraphDatabase")
    def test_init_success(self, mock_graphdb):
        manager = Neo4jManager()
        self.assertEqual(manager.uri, "bolt://fake")
        mock_graphdb.driver.assert_called_once_with("bolt://fake", auth=("u", "p"))

    @patch("core.graph_construct.neo4j_manager.GraphDatabase")
    def test_init_failure(self, mock_graphdb):
        mock_graphdb.driver.side_effect = Exception("Connection failed")
        manager = Neo4jManager()
        self.assertIsNone(manager.driver)

    @patch("core.graph_construct.neo4j_manager.GraphDatabase")
    def test_close(self, mock_graphdb):
        manager = Neo4jManager()
        manager.close()
        manager.driver.close.assert_called_once()

    @patch("core.graph_construct.neo4j_manager.GraphDatabase")
    def test_sync_from_memory_graph(self, mock_graphdb):
        manager = Neo4jManager()
        mock_session = MagicMock()
        manager.driver.session.return_value.__enter__.return_value = mock_session

        mock_memory_db = MagicMock()
        mock_memory_db.nodes_data = {
            "node1": {
                "type": "Case",
                "data": {"name": "c1", "embedding": [1, 2, 3], "ignore_dict": {}},
            },
            "node2": {"type": "Case", "data": {"name": "c2"}},
        }

        # mock edges generator
        def edge_gen(data=False):
            yield ("node1", "node2", {"relation_type": "RELATED TO"})

        mock_memory_db.graph.edges = edge_gen

        manager.sync_from_memory_graph(mock_memory_db)

        # Verify run was called with clear
        mock_session.run.assert_any_call("MATCH (n) DETACH DELETE n")

        # Get all calls to session.run
        calls = mock_session.run.call_args_list

        # Verify node insert using UNWIND
        node_batch = [
            {"name": "c1", "embedding": [1, 2, 3], "id": "node1"},
            {"name": "c2", "id": "node2"},
        ]
        node_query_found = any(
            "CREATE (n:Case)" in call[0][0]
            and "UNWIND $batch AS props" in call[0][0]
            and call[1].get("batch") == node_batch
            for call in calls
        )
        self.assertTrue(node_query_found, "Node insert query not found or batch mismatched")

        # Verify edge insert using UNWIND
        edge_batch = [{"source": "node1", "target": "node2"}]
        edge_query_found = any(
            "MERGE (a)-[r:RELATED_TO]->(b)" in call[0][0]
            and "UNWIND $batch AS edge" in call[0][0]
            and call[1].get("batch") == edge_batch
            for call in calls
        )
        self.assertTrue(edge_query_found, "Edge insert query not found or batch mismatched")

    @patch("core.graph_construct.neo4j_manager.GraphDatabase")
    def test_sync_no_driver(self, mock_graphdb):
        mock_graphdb.driver.side_effect = Exception("Fail")
        manager = Neo4jManager()
        manager.sync_from_memory_graph(MagicMock())
        # Should just return and log error, no crash


if __name__ == "__main__":
    unittest.main()
