# LegalGraphRAG: Vietnamese Civil Law Analysis

> A Vietnamese Civil Law analysis system that integrates multi-agent graph retrieval and supports legal resolution based on reference cases and articles.

---

## Highlights
- **Vietnamese Legal Knowledge Graph**: Integrates Vietnamese Civil Code, Labor Code, and Social Insurance Law into a knowledge graph.
- **API Server**: Includes a FastAPI server (`main.py`) with a web interface (`web/index.html`) for interactive legal analysis.
- **Automated Retrieval**: Retrieves similar cases and relevant laws from the graph based on user facts.
- **Generative Analysis**: Uses LLMs to analyze civil disputes, determine applicable laws, and propose resolution directions.

<p align="center">
  <img src="images/method.png" width="95%" alt="Framework Overview">
</p>

---

## Project Structure

```text
LegalGraphRAG/
├── core/                      # Core modules for Graph RAG and DB connection
├── scripts/                   # Scripts for analysis, data prep, and evaluation
├── tests/                     # Unit tests and connection tests
├── web/                       # Frontend web interface (HTML/JS/CSS)
├── data/                      # Generated knowledge graph data and databases
├── main.py                    # FastAPI application entry point
├── docker-compose.yml         # Docker Compose configuration for quick setup
└── README.md                  # Project documentation
```

---

## Getting Started

### Method 1: Using Docker (Recommended)
The easiest way to run the project is using Docker Compose, which automatically sets up the Neo4j database and the FastAPI backend.

1. Create your environment file:
```bash
cp env.example .env
# Edit .env with your LLM API keys
```

2. Start the services:
```bash
docker-compose up -d --build
```
The FastAPI application will be available at `http://localhost:8000/`.

### Method 2: Manual Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
pip install fastapi uvicorn pydantic pyvis
```

2. Configure environment variables:
```bash
cp env.example .env
# Edit .env with your model API keys and Neo4j connection string
```

3. Start the FastAPI backend:
```bash
python main.py
```
Open your browser and navigate to: `http://localhost:8000/`

---

## Configuration
Configuration is managed via `.env`. See `env.example` for the full configuration list, specifically Model APIs and database credentials used by `main.py`.
