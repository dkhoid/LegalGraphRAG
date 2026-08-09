import sys
import os

sys.path.append(".")
from core.graph_construct.graph_db import GraphDBManager
from core.graph_construct.neo4j_manager import Neo4jManager
from dotenv import load_dotenv

load_dotenv(".env")

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

print(f"Connecting to Neo4j at {uri}...")
manager = Neo4jManager(uri=uri, user=user, password=password)

if manager.driver is None:
    print("FAILED TO CONNECT TO NEO4J AURA! Please check your internet connection or credentials.")
else:
    print("Connection successful! Loading local graph...")
    GraphDBManager.load("data/clean/graph.pkl")
    db = GraphDBManager.get_db()

    print("Syncing to Neo4j Aura...")
    manager.sync_from_memory_graph(db)
    print("Done!")
