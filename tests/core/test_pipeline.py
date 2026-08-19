import pytest
from core.pipeline import analyze_case_pipeline, _calculate_confidence
from core.config import RetrieveConfig
from unittest.mock import MagicMock, patch


@pytest.fixture
def base_config():
    return RetrieveConfig()


def test_calculate_confidence():
    # Test unanimous true
    res = _calculate_confidence([1.0, 1.0, 1.0], [{}, {}, {}], [{}, {}, {}], [{}, {}, {}])
    assert res["overall"] == 1.0

    # Test mixed
    res = _calculate_confidence([1.0, 0.0, 1.0], [{}, {}, {}], [{}, {}, {}], [{}, {}, {}])
    assert res["overall"] == 0.667

    # Test empty
    res = _calculate_confidence([], [], [], [])
    assert res["overall"] == 0.0


@patch("core.pipeline._extract_features_and_segment")
@patch("core.pipeline._retrieve_and_rerank_laws")
@patch("core.pipeline._evaluate_laws")
@patch("core.pipeline.judge_civil_all")
def test_analyze_case_pipeline_full_flow(
    mock_judge_civil, mock_evaluate_laws, mock_retrieve, mock_extract, base_config
):
    mock_model = MagicMock()

    # Mock extract
    mock_extract.return_value = [
        {
            "name": "test_party",
            "description": "test_desc",
            "is_civil": True,
            "feature": {"action": "steal"},
            "segment_dict": {"scene": "night"},
        }
    ]

    # Mock retrieval
    mock_retrieve.return_value = (
        {"fusion_method": "rrf"},
        [{"id": "1", "entry": "1", "fact": "law 1"}],
        [{"id": "1", "description": "fact 1"}],
    )

    # Mock evaluate
    mock_evaluate_laws.return_value = ([{"id": "1", "entry": "1"}], [1.0])

    # Mock judge civil
    mock_judge_civil.return_value = ("Success", 1.0)

    result = analyze_case_pipeline(
        mock_model,
        {"id": "test_case", "name": "test", "fact": "test"},
        [{"id": "1"}],
        [],
        base_config,
    )

    assert result is not None
    assert len(result) == 1
    assert "judge_result" in result[0]
    assert result[0]["judge_result"][0] == "Success"


@patch("core.pipeline._extract_features_and_segment")
def test_analyze_case_pipeline_extract_failure(mock_extract, base_config):
    mock_model = MagicMock()

    # Mock extract failing (returning None)
    mock_extract.return_value = []

    result = analyze_case_pipeline(
        mock_model,
        {"id": "test_case", "name": "test", "fact": "test"},
        [{"id": "1"}],
        [],
        base_config,
    )

    # It should return empty list if extraction fails
    assert result == []
