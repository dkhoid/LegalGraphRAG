import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
import main

# Mock dependencies before importing the app
client = TestClient(main.app)


@pytest.fixture
def mock_rag_system():
    mock_rag = MagicMock()
    # Mock model
    mock_rag.model = MagicMock()
    mock_rag.model.generate_response = MagicMock(
        return_value='{"dispute_type": "Dân sự", "applicable_laws": [], "resolution_direction": "Hướng giải quyết"}'
    )
    return mock_rag


def test_root_endpoint():
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/web/index.html"


@patch("api.routes.run_in_threadpool", new_callable=AsyncMock)
def test_generate_prompt(mock_threadpool, mock_rag_system):
    # Setup mock returns for threadpool
    main.app.state.rag_system = mock_rag_system
    mock_threadpool.side_effect = [
        [{"id": "case_1", "description": "Tình tiết vụ án 1"}],  # returned by query_similar_nodes
        [{"entry": "1", "description": "Điều 1"}],  # returned by query_similar_laws
    ]

    response = client.post(
        "/generate_prompt",
        json={"fact": "Xin chào, tôi muốn hỏi về bồi thường", "top_k": 3},
    )

    assert response.status_code == 200
    data = response.json()
    assert "prompt" in data
    assert "retrieved_laws" in data
    assert "retrieved_facts" in data


@patch("api.routes.generate_prompt", new_callable=AsyncMock)
@patch("api.routes.run_in_threadpool", new_callable=AsyncMock)
def test_analyze_civil(mock_threadpool, mock_generate_prompt, mock_rag_system):
    main.app.state.rag_system = mock_rag_system
    # Setup prompt response
    mock_prompt_response = MagicMock()
    mock_prompt_response.prompt = "Test prompt"
    mock_prompt_response.retrieved_laws = []
    mock_prompt_response.retrieved_facts = []
    mock_generate_prompt.return_value = mock_prompt_response

    # Setup LLM response
    mock_threadpool.return_value = '```json\n{"dispute_type": "Tranh chấp hợp đồng", "applicable_laws": ["Điều 1"], "resolution_direction": "Giải quyết"}\n```'

    response = client.post(
        "/analyze_civil",
        json={"fact": "Xin chào, tôi muốn hỏi về bồi thường", "top_k": 3},
    )

    assert response.status_code == 200
    data = response.json()
    assert "analysis_result" in data
    assert data["analysis_result"]["dispute_type"] == "Tranh chấp hợp đồng"
