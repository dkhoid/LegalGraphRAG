import sys
import os
from core.graph_construct.graph_db import InMemoryGraphDB
from core.graph_construct.neo4j_manager import neo4j_manager

db = InMemoryGraphDB()
try:
    db.load("./data/processed/graph.pkl")
    print("Graph loaded from pickle.")
    neo4j_manager.sync_from_memory_graph(db)
    print("Synced to Neo4j successfully!")
except Exception as e:
    print(f"Error: {e}")
