from core.graph_construct.neo4j_manager import Neo4jManager
import os


def test_neo4j():
    print("Testing Neo4j connection...")
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    print(f"Connecting to {uri}")
    manager = Neo4jManager()

    if manager.driver is None:
        print("Failed to initialize Neo4j driver.")
        return

    try:
        with manager.driver.session() as session:
            result = session.run("RETURN 1 AS num")
            for record in result:
                print(f"Success! Neo4j is active and returned: {record['num']}")
    except Exception as e:
        print(f"Error executing query: {e}")
    finally:
        manager.close()


if __name__ == "__main__":
    test_neo4j()
