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


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


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


@patch("api.routes.run_in_threadpool", new_callable=AsyncMock)
def test_analyze_civil(mock_threadpool, mock_rag_system):
    main.app.state.rag_system = mock_rag_system

    # Setup analyze_case return structure
    mock_threadpool.return_value = [
        {
            "name": "Bị đơn",
            "description": "Mô tả bên bị đơn",
            "judge_result": {
                "dispute_type": "Tranh chấp hợp đồng",
                "applicable_laws": ["Điều 1"],
                "resolution_direction": "Giải quyết",
            },
            "used_laws": [{"id": "Điều 1", "text": "Nội dung điều 1"}],
            "used_facts": [{"fact": "Tình tiết 1", "similarity": 0.9}],
            "confidence": {"overall": 0.85, "grade": "HIGH"},
            "reasoning_trace": {"used_laws_count": 1},
        }
    ]

    response = client.post(
        "/analyze_civil",
        json={"fact": "Xin chào, tôi muốn hỏi về bồi thường", "top_k": 3},
    )

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 1
    assert data["results"][0]["name"] == "Bị đơn"
    assert data["results"][0]["judge_result"]["dispute_type"] == "Tranh chấp hợp đồng"


@patch("api.routes.run_in_threadpool", new_callable=AsyncMock)
def test_chat_endpoint_with_deepseek_provider(mock_threadpool, mock_rag_system):
    main.app.state.rag_system = mock_rag_system
    mock_threadpool.return_value = [
        {
            "name": "Bị đơn",
            "used_laws": [{"entry": "Điều 100", "description": "Quyền sở hữu"}],
            "retrieved_facts": [],
            "retrieved_laws": [{"entry": "Điều 100"}],
        }
    ]

    response = client.post(
        "/api/chat",
        json={
            "query": "Tư vấn tranh chấp đất đai",
            "provider": "deepseek",
            "api_key": "sk-deepseek-test",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "Điều 100" in data["reply"]
    assert "Quyền sở hữu" in data["reply"]


@patch("core.graph_construct.neo4j_manager.neo4j_manager.driver")
@patch("api.routes.run_in_threadpool", new_callable=AsyncMock)
def test_inspect_graph_endpoint(mock_threadpool, mock_driver):
    mock_threadpool.return_value = [0.1] * 1536
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    mock_session.run.return_value.data.return_value = [
        {
            "c": {"id": "c1", "caseId": "qa1_syn_1", "description": "Vụ án tranh chấp đất đai"},
            "score": 0.88,
            "cluster": {"id": "cl1", "summary": "Cụm tranh chấp đất đai"},
            "laws": [
                {"id": "l1", "entry": "zalo_91/2015/qh13+101", "description": "Điều 101 BLDS"}
            ],
        }
    ]

    response = client.post(
        "/api/graph/inspect",
        json={"query": "Tranh chấp đất đai", "top_k": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) >= 3
    assert any(n["type"] == "Query" for n in data["nodes"])
    assert any(n["type"] == "Case" for n in data["nodes"])
    assert any(n["type"] == "Law" for n in data["nodes"])
