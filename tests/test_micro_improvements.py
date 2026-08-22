"""Unit tests for M1 (MMR), M2 (score threshold via RRF), and M7 (abbreviation expansion).

Run with: pytest tests/test_micro_improvements.py -v
"""

import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# M2 + M8: RRF _rrf_score attachment
# ─────────────────────────────────────────────────────────────────────────────


class TestRRFScoreAttachment:
    """_rrf_score is now attached to every output item."""

    def setup_method(self):
        from core.utils.rrf import reciprocal_rank_fusion

        self.rrf = reciprocal_rank_fusion

    def test_score_field_present(self):
        result = self.rrf([[{"id": "A"}, {"id": "B"}]])
        assert "_rrf_score" in result[0]

    def test_score_is_float(self):
        result = self.rrf([[{"id": "A"}]])
        assert isinstance(result[0]["_rrf_score"], float)

    def test_score_decreasing(self):
        """Top item should have higher score than second item."""
        l1 = [{"id": "A"}, {"id": "B"}, {"id": "C"}]
        l2 = [{"id": "A"}, {"id": "B"}, {"id": "D"}]
        result = self.rrf([l1, l2])
        scores = [r["_rrf_score"] for r in result]
        assert scores == sorted(scores, reverse=True), "Scores should be non-increasing"

    def test_original_dict_not_mutated(self):
        """The original law dicts should not be mutated by RRF."""
        original = {"id": "A", "description": "test"}
        self.rrf([[original]])
        assert "_rrf_score" not in original, "Original dict should not be mutated"

    def test_custom_k_changes_scores(self):
        """Different k values should produce different scores."""
        laws = [{"id": "A"}, {"id": "B"}]
        r_k60 = self.rrf([laws], k=60)
        r_k15 = self.rrf([laws], k=15)
        # k=15 → scores are larger (1/(15+1) > 1/(60+1))
        assert r_k15[0]["_rrf_score"] > r_k60[0]["_rrf_score"]

    def test_all_items_have_score(self):
        """Every item in output should have _rrf_score."""
        l1 = [{"id": str(i)} for i in range(5)]
        l2 = [{"id": str(i)} for i in range(3, 8)]
        result = self.rrf([l1, l2])
        for item in result:
            assert "_rrf_score" in item


# ─────────────────────────────────────────────────────────────────────────────
# M7: Vietnamese Legal Abbreviation Expansion
# ─────────────────────────────────────────────────────────────────────────────


class TestLegalTextExpansion:
    """Tests for core/utils/legal_text.py"""

    def setup_method(self):
        from core.utils.legal_text import expand_abbreviations, preprocess_for_retrieval

        self.expand = expand_abbreviations
        self.preprocess = preprocess_for_retrieval

    def test_blds_expanded(self):
        result = self.expand("Vi phạm BLDS 2015")
        assert "Bộ luật dân sự" in result
        assert "BLDS" in result  # abbreviation preserved

    def test_lhn_expanded(self):
        result = self.expand("Theo quy định LHN 2014")
        assert "Luật hôn nhân và gia đình" in result

    def test_multiple_abbrevs_in_one_text(self):
        result = self.expand("BLDS và BLHS quy định")
        assert "Bộ luật dân sự" in result
        assert "Bộ luật hình sự" in result

    def test_no_abbrev_unchanged(self):
        text = "không có chữ tắt nào ở đây"
        result = self.expand(text)
        assert result == text

    def test_case_sensitive_abbrev_not_expanded(self):
        """Lowercase should NOT be expanded – legal abbrevs are uppercase."""
        result = self.expand("blds không nên được mở rộng")
        assert "Bộ luật dân sự" not in result

    def test_preprocess_returns_string(self):
        result = self.preprocess("BLDS 2015 điều 584")
        assert isinstance(result, str)

    def test_expand_disabled(self):
        """With expand=False, abbreviations should not be expanded."""
        result = self.preprocess("BLDS và BLHS", expand=False)
        assert "Bộ luật dân sự" not in result
        assert "BLDS" in result


# ─────────────────────────────────────────────────────────────────────────────
# M1: MMR Diversity
# ─────────────────────────────────────────────────────────────────────────────


class TestMMR:
    """Tests for core/utils/mmr.py"""

    def setup_method(self):
        from core.utils.mmr import maximal_marginal_relevance

        self.mmr = maximal_marginal_relevance

    def _vec(self, *vals):
        return list(vals)

    def _law(self, id_: str, vec: list, desc: str = ""):
        return {"id": id_, "_embedding": vec, "description": desc}

    def test_returns_k_items(self):
        laws = [self._law(str(i), self._vec(i * 0.1, 0, 0)) for i in range(10)]
        result = self.mmr([1.0, 0, 0], laws, k=4)
        assert len(result) <= 4

    def test_empty_input(self):
        assert self.mmr([1.0, 0], [], k=3) == []

    def test_fewer_than_k_returns_all(self):
        laws = [self._law("A", [1, 0]), self._law("B", [0, 1])]
        result = self.mmr([1.0, 0], laws, k=10)
        assert len(result) == 2

    def test_lambda_1_prefers_relevance(self):
        """lambda=1.0 → pure relevance → most relevant law comes first."""
        query = [1.0, 0.0]
        laws = [
            self._law("relevant", [1.0, 0.0]),  # most similar to query
            self._law("diverse", [0.0, 1.0]),  # least similar
            self._law("medium", [0.7, 0.7]),
        ]
        result = self.mmr(query, laws, k=3, lambda_=1.0)
        assert result[0]["id"] == "relevant"

    def test_law_data_preserved(self):
        """Original law fields should be preserved in output."""
        law = self._law("X", [1.0, 0.0], desc="contract law")
        result = self.mmr([1.0, 0.0], [law], k=1)
        assert result[0]["description"] == "contract law"

    def test_no_embedding_fallback(self):
        """Laws without _embedding should be returned at the end as fallback."""
        law_with = self._law("A", [1.0, 0.0])
        law_without = {"id": "B", "description": "no embedding"}
        result = self.mmr([1.0, 0.0], [law_with, law_without], k=2)
        # Should return both (with embedding first, without at end)
        ids = {r["id"] for r in result}
        assert "A" in ids
        assert "B" in ids

    def test_selects_diverse_laws(self):
        """With lambda=0.0 (pure diversity), should avoid selecting similar laws."""
        query = [1.0, 0.0]
        # law_a and law_b are very similar, law_c is different
        law_a = self._law("a", [1.0, 0.01])
        law_b = self._law("b", [1.0, 0.02])  # very similar to law_a
        law_c = self._law("c", [0.0, 1.0])  # very different
        result = self.mmr(query, [law_a, law_b, law_c], k=2, lambda_=0.0)
        ids = {r["id"] for r in result}
        # Diversity mode → should pick one of {a,b} and c (not both a and b)
        assert "c" in ids

    def test_zero_vector_handled(self):
        """Zero vectors should not cause DivisionByZero."""
        query = [0.0, 0.0]
        laws = [self._law("A", [1.0, 0.0]), self._law("B", [0.0, 0.0])]
        result = self.mmr(query, laws, k=2)
        assert len(result) == 2


# ─────────────────────────────────────────────────────────────────────────────
# M4/K1: Reranker Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCrossEncoderReranker:
    """Tests for core/retriever/reranker.py"""

    def setup_method(self):
        from core.retriever.reranker import CrossEncoderReranker

        self.reranker_cls = CrossEncoderReranker
        # Use a dummy model name so it doesn't try to download anything
        self.reranker = self.reranker_cls(model_name="dummy", top_k=2)

    @patch("core.retriever.reranker.CrossEncoderReranker._load")
    def test_rerank_empty_list(self, mock_load):
        result = self.reranker.rerank("query", [])
        assert result == []
        mock_load.assert_not_called()

    @patch("core.retriever.reranker.CrossEncoderReranker._load")
    def test_rerank_top_k_applied(self, mock_load):
        # Mock predict to just return lengths of descriptions as dummy scores
        self.reranker._model = MagicMock()
        self.reranker._model.predict.return_value = [0.9, 0.1, 0.8]

        laws = [
            {"id": "A", "description": "High match"},
            {"id": "B", "description": "Low match"},
            {"id": "C", "description": "Medium match"},
        ]

        result = self.reranker.rerank("query", laws)

        # We set top_k = 2 in setup
        assert len(result) == 2
        # Highest scores are A (0.9) and C (0.8)
        assert result[0]["id"] == "A"
        assert result[1]["id"] == "C"

        # Verify score is attached
        assert "_rerank_score" in result[0]
        assert result[0]["_rerank_score"] == 0.9

    @patch("core.retriever.reranker.CrossEncoderReranker._load")
    def test_rerank_preserves_dict(self, mock_load):
        self.reranker._model = MagicMock()
        self.reranker._model.predict.return_value = [0.5]

        law = {"id": "1", "description": "test", "extra": "data"}
        result = self.reranker.rerank("query", [law])

        assert result[0]["extra"] == "data"
        assert "_rerank_score" in result[0]
        assert "_rerank_score" not in law, "Original dict should not be mutated"
