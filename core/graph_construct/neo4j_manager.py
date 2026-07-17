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

            # Organize nodes by type for bulk inserting
            nodes_by_type = {}
            for node_id, node_info in memory_db.nodes_data.items():
                node_type = node_info["type"]
                props = node_info["data"]

                safe_props = {
                    k: v
                    for k, v in props.items()
                    if k != "embedding" and isinstance(v, (str, int, float, bool))
                }
                # Thêm ID trực tiếp vào thuộc tính để sử dụng trong UNWIND
                safe_props["id"] = str(node_id)

                if node_type not in nodes_by_type:
                    nodes_by_type[node_type] = []
                nodes_by_type[node_type].append(safe_props)

            # Bulk insert nodes by type using UNWIND
            for node_type, nodes_list in nodes_by_type.items():
                batch_size = 2000
                for i in range(0, len(nodes_list), batch_size):
                    batch = nodes_list[i : i + batch_size]
                    query = f"""
                    UNWIND $batch AS props
                    CREATE (n:{node_type})
                    SET n = props
                    """
                    session.run(query, batch=batch)

            logger.info("Nodes synced successfully. Creating indexes and syncing edges...")

            # Tạo index trên 'id' để lệnh MATCH khi chèn edge được tối ưu hóa cực nhanh
            for node_type in nodes_by_type.keys():
                try:
                    # Cú pháp Neo4j 4.x+
                    session.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{node_type}) ON (n.id)")
                except Exception as e:
                    logger.warning(f"Could not create index for {node_type}: {e}")

            # Organize edges by relation_type and batch insert to save memory
            edges_buffers = {}
            for source, target, data in memory_db.graph.edges(data=True):
                rel_type = data.get("relation_type", "RELATED_TO").replace(" ", "_").upper()

                src_type = memory_db.nodes_data.get(source, {}).get("type")
                tgt_type = memory_db.nodes_data.get(target, {}).get("type")
                if not src_type or not tgt_type:
                    continue

                key = (rel_type, src_type, tgt_type)
                if key not in edges_buffers:
                    edges_buffers[key] = []
                edges_buffers[key].append({"source": str(source), "target": str(target)})

                # Flush to Neo4j if batch size reached
                if len(edges_buffers[key]) >= 5000:
                    batch = edges_buffers[key]
                    query = f"""
                    UNWIND $batch AS edge
                    MATCH (a:{src_type} {{id: edge.source}}), (b:{tgt_type} {{id: edge.target}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    """
                    session.run(query, batch=batch)
                    edges_buffers[key] = []

            # Flush remaining edges
            for (rel_type, src_type, tgt_type), batch in edges_buffers.items():
                if batch:
                    query = f"""
                    UNWIND $batch AS edge
                    MATCH (a:{src_type} {{id: edge.source}}), (b:{tgt_type} {{id: edge.target}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    """
                    session.run(query, batch=batch)

        logger.info("Successfully synced graph to Neo4j.")


# Singleton instance
neo4j_manager = Neo4jManager()
