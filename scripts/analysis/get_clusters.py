from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
user = "neo4j"
password = "password"

driver = GraphDatabase.driver(uri, auth=(user, password))

with driver.session() as session:
    result = session.run(
        "MATCH (c:Cluster) RETURN c.id AS id, c.summary AS summary ORDER BY toInteger(c.id)"
    )
    print("Danh sách 9 Cụm chủ đề (Clusters):")
    print("-" * 50)
    for record in result:
        id = record["id"]
        summary = record["summary"]
        print(f"Cluster ID: {id}")
        print(f"Summary: {summary}")
        print("-" * 50)
driver.close()
