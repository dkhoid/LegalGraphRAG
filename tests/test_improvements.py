"""Unit tests for LegalGraphRAG core improvements.

Tests cover:
- Option B: judge_law_batch (batch judgment)
- Option D: reciprocal_rank_fusion (RRF retrieval)
- Option E: compute_confidence (confidence scoring)

All tests use mock objects and require no LLM or database connection.
Run with: pytest tests/test_improvements.py -v
"""

import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures & Helpers
# ─────────────────────────────────────────────────────────────────────────────


class MockChatbot:
    """Minimal chatbot mock for testing judge functions without a real LLM."""

    def __init__(
        self,
        response: str = '{"conditions": [{"id": 1, "met": true}], "applicable": true}',
    ):
        self._response = response

    def generate_response(self, prompt: str, **kwargs) -> str:
        return self._response


# ─────────────────────────────────────────────────────────────────────────────
# Option D: RRF Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestReciprocalRankFusion:
    """Tests for core/utils/rrf.py"""

    def setup_method(self):
        from core.utils.rrf import reciprocal_rank_fusion

        self.rrf = reciprocal_rank_fusion

    def test_basic_fusion_returns_all_items(self):
        """Items from all lists should appear in result."""
        l1 = [{"id": "A"}, {"id": "B"}]
        l2 = [{"id": "C"}, {"id": "D"}]
        result = self.rrf([l1, l2])
        result_ids = {r["id"] for r in result}
        assert result_ids == {"A", "B", "C", "D"}

    def test_top_item_is_consistent_across_sources(self):
        """Item consistently at top of multiple lists should rank highest."""
        l1 = [{"id": "A"}, {"id": "B"}, {"id": "C"}]
        l2 = [{"id": "A"}, {"id": "C"}, {"id": "D"}]
        result = self.rrf([l1, l2])
        assert result[0]["id"] == "A", "Item top in both lists should win"

    def test_deduplication(self):
        """Same item in multiple lists should appear only once in result."""
        l1 = [{"id": "A"}, {"id": "B"}]
        l2 = [{"id": "B"}, {"id": "A"}]
        result = self.rrf([l1, l2])
        ids = [r["id"] for r in result]
        assert len(ids) == len(set(ids)), "Duplicate IDs found in RRF output"

    def test_single_list_preserves_order(self):
        """With a single list, original order should be approximately maintained."""
        l1 = [{"id": "A"}, {"id": "B"}, {"id": "C"}]
        result = self.rrf([l1])
        assert result[0]["id"] == "A"

    def test_empty_list_in_input_is_skipped(self):
        """Empty sub-lists should not cause errors."""
        l1 = [{"id": "A"}, {"id": "B"}]
        result = self.rrf([l1, [], []])
        assert len(result) == 2

    def test_all_empty_returns_empty(self):
        """All empty input should return empty list."""
        result = self.rrf([[], [], []])
        assert result == []

    def test_entry_key_fallback(self):
        """Should resolve IDs via 'entry' key when 'id' is absent."""
        l1 = [{"entry": "584"}, {"entry": "585"}]
        l2 = [{"entry": "584"}, {"entry": "600"}]
        result = self.rrf([l1, l2], id_key="entry")
        entries = [r["entry"] for r in result]
        assert entries[0] == "584", "entry '584' appears in both lists – should be first"

    def test_custom_k_value(self):
        """Custom k should affect scores but not break correctness."""
        l1 = [{"id": "X"}, {"id": "Y"}]
        l2 = [{"id": "X"}, {"id": "Z"}]
        result_k60 = self.rrf([l1, l2], k=60)
        result_k1 = self.rrf([l1, l2], k=1)
        # Both should agree on the top item
        assert result_k60[0]["id"] == result_k1[0]["id"] == "X"

    def test_item_data_preserved(self):
        """Original dict data should be preserved in output."""
        l1 = [{"id": "A", "description": "law about contracts", "entry": "123"}]
        result = self.rrf([l1])
        assert result[0]["description"] == "law about contracts"
        assert result[0]["entry"] == "123"


# ─────────────────────────────────────────────────────────────────────────────
# Option E: Confidence Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeConfidence:
    """Tests for core/utils/confidence.py"""

    def setup_method(self):
        from core.utils.confidence import compute_confidence, ConfidenceResult

        self.compute = compute_confidence
        self.ConfidenceResult = ConfidenceResult

    def test_returns_confidence_result_type(self):
        result = self.compute([1, 2, 3], [1], [1])
        assert isinstance(result, self.ConfidenceResult)

    def test_high_grade_with_good_signals(self):
        """5 retrieved laws, 4 used, 2 facts → should be HIGH."""
        result = self.compute(
            retrieved_laws=[1, 2, 3, 4, 5],
            used_laws=[1, 2, 3, 4],
            retrieved_facts=[1, 2],
        )
        assert result.grade == "HIGH"
        assert result.review_required is False

    def test_low_grade_triggers_review(self):
        """No retrieved laws → overall = 0 → LOW + review required."""
        result = self.compute(
            retrieved_laws=[],
            used_laws=[],
            retrieved_facts=[],
        )
        assert result.grade == "LOW"
        assert result.review_required is True
        assert result.overall == 0.0

    def test_medium_grade_boundary(self):
        """1 retrieved law, 1 used, 0 facts → moderate score."""
        result = self.compute(
            retrieved_laws=[1],
            used_laws=[1],
            retrieved_facts=[],
        )
        # retrieval_quality = 0.2, law_applicability = 1.0, has_evidence = 0.0
        # overall = 0.2*0.3 + 1.0*0.4 + 0.0*0.3 = 0.06 + 0.40 = 0.46 → MEDIUM
        assert result.grade == "MEDIUM"

    def test_retrieval_quality_capped_at_one(self):
        """More than 5 laws should cap retrieval_quality at 1.0."""
        result = self.compute(
            retrieved_laws=list(range(20)),
            used_laws=list(range(10)),
            retrieved_facts=[1],
        )
        assert result.retrieval_quality == 1.0

    def test_to_dict_contains_required_keys(self):
        """to_dict() output must have all expected keys."""
        result = self.compute([1, 2], [1], [1]).to_dict()
        required_keys = {
            "retrieval_quality",
            "law_applicability",
            "overall",
            "grade",
            "review_required",
        }
        assert required_keys.issubset(result.keys())

    def test_to_dict_values_are_rounded(self):
        """Float values in to_dict should be rounded to 3 decimal places."""
        result = self.compute([1, 2, 3], [1], [1]).to_dict()
        for key in ("retrieval_quality", "law_applicability", "overall"):
            val = result[key]
            assert val == round(val, 3), f"{key} is not rounded to 3 decimals"

    def test_overall_bounded_between_zero_and_one(self):
        """Overall score should always be in [0, 1]."""
        for n_retrieved in [0, 1, 3, 5, 10]:
            for n_used in range(min(n_retrieved + 1, 5)):
                result = self.compute(
                    retrieved_laws=list(range(n_retrieved)),
                    used_laws=list(range(n_used)),
                    retrieved_facts=[1] if n_used > 0 else [],
                )
                assert 0.0 <= result.overall <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Option B: Batch Judge Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestJudgeLawBatch:
    """Tests for judge_law_batch() in core/judge/judge_law.py"""

    def setup_method(self):
        from core.judge.judge_law import judge_law_batch

        self.judge = judge_law_batch

    def _law(self, judge_dep=None, description="Điều luật về hợp đồng"):
        return {
            "description": description,
            "judge_dep": judge_dep or ["Có hành vi vi phạm", "Có thiệt hại xảy ra"],
            "related_laws": [],
        }

    def test_returns_tuple_of_bool_and_str(self):
        """Return type must match judge_law() contract: (bool, str)."""
        bot = MockChatbot('{"conditions": [{"id": 1, "met": true}], "applicable": true}')
        result = self.judge(bot, "test case", self._law())
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_applicable_true_when_model_says_true(self):
        bot = MockChatbot('{"conditions": [], "applicable": true}')
        applicable, _ = self.judge(bot, "test case", self._law())
        assert applicable is True

    def test_applicable_false_when_model_says_false(self):
        bot = MockChatbot('{"conditions": [], "applicable": false}')
        applicable, _ = self.judge(bot, "test case", self._law())
        assert applicable is False

    def test_fallback_on_invalid_json(self):
        """Invalid JSON from model should not crash – fallback to original judge_law."""
        bot = MockChatbot("This is not JSON at all.")
        # judge_law() will be called; it returns (False, "") for a simple law
        law = {"description": "simple law", "judge_dep": [], "related_laws": []}
        result = self.judge(bot, "test case", law)
        assert isinstance(result, tuple)
        assert isinstance(result[0], bool)

    def test_no_judge_deps_uses_fallback(self):
        """Law with empty judge_dep should fall back to original judge_law()."""
        # Mock returns plain "true" (as judge_law expects for JUDGE_LAW_PROMPT1)
        bot = MockChatbot("true")
        law = {"description": "simple law", "judge_dep": []}
        result = self.judge(bot, "case", law)
        assert isinstance(result[0], bool)

    def test_reasoning_is_string(self):
        """The reasoning string should always be a string, even if empty."""
        bot = MockChatbot(
            '{"conditions": [{"id": 1, "met": true, "reason": "test"}], "applicable": true}'
        )
        _, reasoning = self.judge(bot, "case", self._law())
        assert isinstance(reasoning, str)

    def test_json_with_preamble_text_is_parsed(self):
        """Model sometimes adds text before the JSON. Should still extract correctly."""
        preamble_response = 'Sure, here is my analysis:\n{"conditions": [], "applicable": true}'
        bot = MockChatbot(preamble_response)
        applicable, _ = self.judge(bot, "case", self._law())
        assert applicable is True

    def test_judge_dep_as_string_is_handled(self):
        """judge_dep stored as a string literal should be parsed correctly."""
        law = {
            "description": "test",
            "judge_dep": "['điều kiện 1', 'điều kiện 2']",
            "related_laws": [],
        }
        bot = MockChatbot('{"conditions": [], "applicable": true}')
        result = self.judge(bot, "case", law)
        assert isinstance(result[0], bool)


# ─────────────────────────────────────────────────────────────────────────────
# Option K1: Cross-Encoder Reranker Tests (interface only, no model download)
# ─────────────────────────────────────────────────────────────────────────────


class TestCrossEncoderReranker:
    """Tests for core/retriever/reranker.py – mock the CrossEncoder to avoid downloads."""

    def _make_reranker(self, top_k=3):
        from core.retriever.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(top_k=top_k)

        # Inject mock model that scores laws by position (first = highest)
        class MockCE:
            def predict(self, pairs, **kw):
                # Return descending scores so first pair ranks highest
                return [10.0 - i for i in range(len(pairs))]

        reranker._model = MockCE()
        return reranker

    def test_rerank_returns_top_k(self):
        """Should return at most top_k items."""
        reranker = self._make_reranker(top_k=2)
        laws = [{"id": str(i), "description": f"law {i}"} for i in range(5)]
        result = reranker.rerank("query", laws)
        assert len(result) == 2

    def test_rerank_preserves_law_data(self):
        """Law dicts should survive reranking intact."""
        reranker = self._make_reranker(top_k=5)
        laws = [{"id": "A", "description": "about contract", "judge_dep": ["c1"]}]
        result = reranker.rerank("query", laws)
        assert result[0]["id"] == "A"
        assert result[0]["judge_dep"] == ["c1"]

    def test_rerank_adds_score_field(self):
        """Reranked items should have _rerank_score for debugging."""
        reranker = self._make_reranker(top_k=3)
        laws = [{"id": "X", "description": "test"}]
        result = reranker.rerank("query", laws)
        assert "_rerank_score" in result[0]
        assert isinstance(result[0]["_rerank_score"], float)

    def test_rerank_empty_input(self):
        """Empty law list should return empty list without error."""
        reranker = self._make_reranker()
        result = reranker.rerank("query", [])
        assert result == []

    def test_rerank_fewer_than_top_k(self):
        """When fewer laws than top_k, all should be returned."""
        reranker = self._make_reranker(top_k=10)
        laws = [{"id": str(i), "description": f"law {i}"} for i in range(3)]
        result = reranker.rerank("query", laws)
        assert len(result) == 3

    def test_get_reranker_singleton(self):
        """get_reranker() should return same instance on repeated calls."""
        from core.retriever.reranker import get_reranker

        r1 = get_reranker()
        r2 = get_reranker()
        assert r1 is r2


# ─────────────────────────────────────────────────────────────────────────────
# Option K4: Self-Consistent Judge Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestJudgeLawSelfConsistent:
    """Tests for judge_law_self_consistent() in core/judge/judge_law.py"""

    def setup_method(self):
        from core.judge.judge_law import judge_law_self_consistent

        self.judge = judge_law_self_consistent

    def _law(self):
        return {"description": "test law", "judge_dep": ["cond1", "cond2"]}

    def test_returns_three_tuple(self):
        """Should return (bool, float, str)."""
        bot = MockChatbot('{"conditions": [], "applicable": true}')
        result = self.judge(bot, "case", self._law(), n_samples=3)
        assert isinstance(result, tuple) and len(result) == 3
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)
        assert isinstance(result[2], str)

    def test_unanimous_true_gives_high_confidence(self):
        """All samples agree on True → confidence should be 1.0."""
        bot = MockChatbot('{"conditions": [], "applicable": true}')
        decision, confidence, _ = self.judge(bot, "case", self._law(), n_samples=5)
        assert decision is True
        assert confidence == 1.0

    def test_unanimous_false_gives_high_confidence(self):
        """All samples agree on False → confidence should be 0.0, decision False."""
        bot = MockChatbot('{"conditions": [], "applicable": false}')
        decision, confidence, _ = self.judge(bot, "case", self._law(), n_samples=5)
        assert decision is False
        assert confidence == 0.0

    def test_confidence_bounded_zero_to_one(self):
        """Confidence should always be in [0, 1]."""
        bot = MockChatbot('{"conditions": [], "applicable": true}')
        _, confidence, _ = self.judge(bot, "case", self._law(), n_samples=3)
        assert 0.0 <= confidence <= 1.0

    def test_separate_judge_chatbot_used(self):
        """judge_chatbot parameter should override the sampler model."""
        primary = MockChatbot('{"conditions": [], "applicable": false}')
        cheap = MockChatbot('{"conditions": [], "applicable": true}')
        decision, confidence, _ = self.judge(
            primary, "case", self._law(), n_samples=3, judge_chatbot=cheap
        )
        # cheap bot always says true → should win
        assert decision is True

    def test_n_samples_respected(self):
        """Confidence denominator should match n_samples."""
        bot = MockChatbot('{"conditions": [], "applicable": true}')
        _, confidence, reasoning = self.judge(bot, "case", self._law(), n_samples=4)
        # 4/4 = 1.0
        assert confidence == 1.0
        assert "4/4" in reasoning

    def test_fallback_on_all_errors(self):
        """If all samples fail, should fall back to primary chatbot."""

        class ErrorBot:
            def generate_response(self, p, **kw):
                raise RuntimeError("always fails")

        primary = MockChatbot('{"conditions": [], "applicable": true}')
        result = self.judge(primary, "case", self._law(), n_samples=3, judge_chatbot=ErrorBot())
        # Falls back to primary → should succeed
        assert isinstance(result[0], bool)
