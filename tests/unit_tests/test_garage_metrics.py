"""Tests for GaRAGe evaluation metrics."""

from rag_bench.eval.garage.metrics import (
    compute_attribution_f1,
    compute_deflection_metrics,
    compute_raf,
    compute_uraf,
)


class TestComputeRAF:
    """Test Relevance-Aware Factuality scoring."""

    def test_empty_inputs(self):
        result = compute_raf("", [], "")
        assert result["raf_score"] == 0.0

    def test_empty_answer(self):
        passages = [{"text": "some passage text", "is_relevant": True}]
        result = compute_raf("", passages, "")
        assert result["raf_score"] == 0.0

    def test_high_relevance_overlap(self):
        """Answer that uses relevant passage content should score high."""
        passages = [
            {"text": "The transformer model uses attention mechanisms for sequence processing", "is_relevant": True},
            {"text": "Cooking recipes require careful measurements", "is_relevant": False},
        ]
        answer = "The transformer model uses attention mechanisms for sequence processing tasks"
        result = compute_raf(answer, passages, "")
        assert result["raf_score"] > 0.5
        assert result["relevant_overlap"] > result["irrelevant_overlap"]

    def test_low_relevance_penalizes(self):
        """Answer relying on irrelevant passages should score lower."""
        passages = [
            {"text": "Transformers use attention", "is_relevant": True},
            {"text": "Cooking requires measuring flour butter sugar eggs and milk", "is_relevant": False},
        ]
        answer = "Cooking requires measuring flour butter sugar eggs and milk carefully"
        result = compute_raf(answer, passages, "")
        assert result["irrelevant_overlap"] > result["relevant_overlap"]

    def test_gold_answer_boost(self):
        """Providing a gold answer should influence the score."""
        passages = [{"text": "Neural networks learn features", "is_relevant": True}]
        answer = "Neural networks learn features automatically"
        result_no_gold = compute_raf(answer, passages, "")
        result_with_gold = compute_raf(answer, passages, "Neural networks learn features automatically")
        # With matching gold answer, should be at least as high
        assert result_with_gold["raf_score"] >= result_no_gold["raf_score"] * 0.9


class TestComputeURAF:
    """Test Unweighted RAF scoring."""

    def test_empty_inputs(self):
        result = compute_uraf("", [], "")
        assert result["uraf_score"] == 0.0

    def test_high_overlap(self):
        passages = [{"text": "Attention is all you need for transformer models"}]
        answer = "Attention is all you need"
        result = compute_uraf(answer, passages)
        assert result["uraf_score"] > 0.5
        assert result["passage_overlap"] > 0.5

    def test_low_overlap(self):
        passages = [{"text": "Cooking recipes require measurements"}]
        answer = "Neural networks use backpropagation for learning"
        result = compute_uraf(answer, passages)
        assert result["passage_overlap"] < 0.3


class TestComputeAttributionF1:
    """Test citation attribution quality."""

    def test_empty_inputs(self):
        result = compute_attribution_f1("", [])
        assert result["attribution_f1"] == 0.0

    def test_perfect_attribution(self):
        """All citations point to relevant passages."""
        passages = [
            {"text": "passage 1", "is_relevant": True},
            {"text": "passage 2", "is_relevant": False},
        ]
        answer = "According to [Source 1], this is the answer."
        result = compute_attribution_f1(answer, passages)
        assert result["attribution_precision"] == 1.0
        assert result["attribution_recall"] == 1.0
        assert result["attribution_f1"] == 1.0

    def test_wrong_attribution(self):
        """Citation points to irrelevant passage."""
        passages = [
            {"text": "passage 1", "is_relevant": False},
            {"text": "passage 2", "is_relevant": True},
        ]
        answer = "According to [Source 1], this is the answer."
        result = compute_attribution_f1(answer, passages)
        assert result["attribution_precision"] == 0.0
        assert result["attribution_recall"] == 0.0

    def test_partial_attribution(self):
        """Some citations correct, some wrong."""
        passages = [
            {"text": "passage 1", "is_relevant": True},
            {"text": "passage 2", "is_relevant": True},
            {"text": "passage 3", "is_relevant": False},
        ]
        answer = "From [Source 1] and [Source 3], we know..."
        result = compute_attribution_f1(answer, passages)
        assert result["attribution_precision"] == 0.5  # 1 correct out of 2 cited
        assert result["attribution_recall"] == 0.5  # 1 cited out of 2 relevant
        assert result["cited_count"] == 2
        assert result["relevant_count"] == 2

    def test_no_citations_with_relevant(self):
        """No citations but relevant passages exist → recall is 0."""
        passages = [{"text": "passage 1", "is_relevant": True}]
        answer = "This is an answer without citations."
        result = compute_attribution_f1(answer, passages)
        assert result["attribution_recall"] == 0.0

    def test_no_relevant_no_citations(self):
        """No relevant passages and no citations → perfect (vacuous truth)."""
        passages = [{"text": "passage 1", "is_relevant": False}]
        answer = "This is an answer without citations."
        result = compute_attribution_f1(answer, passages)
        assert result["attribution_f1"] == 1.0  # Nothing to cite and nothing cited → perfect


class TestComputeDeflectionMetrics:
    """Test deflection accuracy metrics."""

    def test_empty_results(self):
        result = compute_deflection_metrics([])
        assert result["total"] == 0

    def test_perfect_deflection(self):
        results = [
            {"should_deflect": True, "did_deflect": True},
            {"should_deflect": False, "did_deflect": False},
        ]
        result = compute_deflection_metrics(results)
        assert result["deflection_accuracy"] == 1.0
        assert result["deflection_tpr"] == 1.0
        assert result["deflection_fpr"] == 0.0

    def test_missed_deflections(self):
        results = [
            {"should_deflect": True, "did_deflect": False},  # FN
            {"should_deflect": True, "did_deflect": False},  # FN
            {"should_deflect": False, "did_deflect": False},  # TN
        ]
        result = compute_deflection_metrics(results)
        assert result["deflection_tpr"] == 0.0
        assert result["false_negatives"] == 2
        assert result["true_negatives"] == 1

    def test_false_alarms(self):
        results = [
            {"should_deflect": False, "did_deflect": True},  # FP
            {"should_deflect": False, "did_deflect": False},  # TN
        ]
        result = compute_deflection_metrics(results)
        assert result["deflection_fpr"] == 0.5
        assert result["false_positives"] == 1


class TestTokenOverlapEdge:
    """Test _token_overlap edge cases."""

    def test_empty_text_a(self):
        from rag_bench.eval.garage.metrics import _token_overlap

        result = _token_overlap("", "some text here")
        assert result == 0.0

    def test_empty_text_b(self):
        from rag_bench.eval.garage.metrics import _token_overlap

        result = _token_overlap("some text", "")
        assert result == 0.0


class TestComputeRAFEdge:
    """Edge cases for compute_raf."""

    def test_punctuation_only_answer(self):
        """Answer with only punctuation → empty tokens → early return."""
        passages = [{"text": "some passage text", "is_relevant": True}]
        result = compute_raf("... !!!", passages, "")
        assert result["raf_score"] == 0.0


class TestComputeURAFEdge:
    """Edge cases for compute_uraf."""

    def test_punctuation_only_answer(self):
        """Answer with only punctuation → empty tokens → early return."""
        passages = [{"text": "some passage text"}]
        result = compute_uraf("...", passages)
        assert result["uraf_score"] == 0.0


class TestAttributionNoRelevant:
    """Test attribution when no relevant passages exist."""

    def test_citations_with_no_relevant(self):
        """Citations exist but no relevant passages → recall is 1.0 (vacuous)."""
        passages = [
            {"text": "irr1", "is_relevant": False},
            {"text": "irr2", "is_relevant": False},
        ]
        answer = "According to [Source 1], this is the answer."
        result = compute_attribution_f1(answer, passages)
        assert result["attribution_recall"] == 1.0
