import os
from neo4j import GraphDatabase
from core.utils.logger import logger


class Neo4jManager:
    """Manages connection to Neo4j Database"""

    def __init__(self, uri=None, user=None, password=None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.driver = None

        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")

    def close(self):
        if self.driver:
            self.driver.close()

    def sync_from_memory_graph(self, memory_db):
        """Sync nodes and edges from InMemoryGraphDB to Neo4j"""
        if not self.driver:
            logger.error("Neo4j driver not initialized. Skipping sync.")
            return

        logger.info("Syncing graph data to Neo4j...")
        with self.driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")

            # Sync nodes
            for node_id, node_info in memory_db.nodes_data.items():
                node_type = node_info["type"]
                props = node_info["data"]

                # Exclude embedding from Neo4j props for simplicity
                safe_props = {
                    k: v
                    for k, v in props.items()
                    if k != "embedding" and isinstance(v, (str, int, float, bool))
                }

                query = f"CREATE (n:{node_type} {{id: $id}}) SET n += $props"
                session.run(query, id=str(node_id), props=safe_props)

            # Sync edges
            for source, target, data in memory_db.graph.edges(data=True):
                rel_type = data.get("relation_type", "RELATED_TO").replace(" ", "_").upper()
                query = f"""
                MATCH (a {{id: $source}}), (b {{id: $target}})
                MERGE (a)-[r:{rel_type}]->(b)
                """
                session.run(query, source=str(source), target=str(target))

        logger.info("Successfully synced graph to Neo4j.")


# Singleton instance
neo4j_manager = Neo4jManager()
