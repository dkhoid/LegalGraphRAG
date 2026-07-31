import os
import sys

PROJECT_ROOT = "/home/rokisaki/Documents/Coding/testing_code/LegalGraphRAG"
sys.path.insert(0, PROJECT_ROOT)

from core.LegalGraphRAG import LegalGraphRAG, LegalGraphRAGConfig


def main():
    print("Initializing LegalGraphRAG...")
    config = LegalGraphRAGConfig.from_env_file()

    # Enable auto build
    config.graph.auto_build = True

    # We pass config
    rag = LegalGraphRAG(config)

    print("Building graph (force_rebuild=True)...")
    rag.build_graph(force_rebuild=True)

    print("Graph built and saved.")


if __name__ == "__main__":
    main()
