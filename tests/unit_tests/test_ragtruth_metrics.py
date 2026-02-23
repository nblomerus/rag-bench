"""Tests for RAGTruth evaluation metrics."""

from rag_bench.eval.ragtruth.metrics import (
    _extract_spans_from_text,
    _normalize_type,
    _tokenize,
    case_level_accuracy,
    compute_hallucination_rate,
    hallucination_by_type,
    span_level_f1,
)


class TestComputeHallucinationRate:
    """Test hallucination rate computation."""

    def test_empty_results(self):
        result = compute_hallucination_rate([])
        assert result["total"] == 0
        assert result["predicted_rate"] == 0.0

    def test_no_hallucinations(self):
        results = [
            {"has_hallucination_predicted": False, "has_hallucination_gold": False},
            {"has_hallucination_predicted": False, "has_hallucination_gold": False},
        ]
        result = compute_hallucination_rate(results)
        assert result["predicted_rate"] == 0.0
        assert result["gold_rate"] == 0.0

    def test_all_hallucinations(self):
        results = [
            {"has_hallucination_predicted": True, "has_hallucination_gold": True},
            {"has_hallucination_predicted": True, "has_hallucination_gold": True},
        ]
        result = compute_hallucination_rate(results)
        assert result["predicted_rate"] == 1.0
        assert result["gold_rate"] == 1.0

    def test_partial_hallucinations(self):
        results = [
            {"has_hallucination_predicted": True, "has_hallucination_gold": True},
            {"has_hallucination_predicted": False, "has_hallucination_gold": False},
            {"has_hallucination_predicted": False, "has_hallucination_gold": True},
            {"has_hallucination_predicted": True, "has_hallucination_gold": False},
        ]
        result = compute_hallucination_rate(results)
        assert result["predicted_rate"] == 0.5
        assert result["gold_rate"] == 0.5
        assert result["total"] == 4


class TestSpanLevelF1:
    """Test span-level F1 computation."""

    def test_empty_both(self):
        result = span_level_f1([], [])
        assert result["span_f1"] == 1.0  # No hallucinations, perfect match

    def test_empty_predicted(self):
        result = span_level_f1([], ["some hallucinated text"])
        assert result["span_f1"] == 0.0
        assert result["span_recall"] == 0.0

    def test_empty_gold(self):
        result = span_level_f1(["predicted hallucination"], [])
        assert result["span_f1"] == 0.0
        assert result["span_precision"] == 0.0

    def test_exact_match(self):
        result = span_level_f1(
            ["the cat sat on the mat"],
            ["the cat sat on the mat"],
        )
        assert result["span_f1"] == 1.0
        assert result["span_precision"] == 1.0
        assert result["span_recall"] == 1.0

    def test_partial_overlap(self):
        result = span_level_f1(
            ["the cat sat"],
            ["the cat sat on the mat near the door"],
        )
        # Predicted tokens are subset of gold
        assert result["span_precision"] == 1.0
        assert result["span_recall"] < 1.0
        assert 0.0 < result["span_f1"] < 1.0

    def test_no_overlap(self):
        result = span_level_f1(
            ["completely different text here"],
            ["neural network architecture design"],
        )
        assert result["span_f1"] == 0.0


class TestCaseLevelAccuracy:
    """Test case-level binary classification."""

    def test_empty_results(self):
        result = case_level_accuracy([])
        assert result["accuracy"] == 0.0
        assert result["total"] == 0

    def test_perfect_classification(self):
        results = [
            {"has_hallucination_predicted": True, "has_hallucination_gold": True},
            {"has_hallucination_predicted": False, "has_hallucination_gold": False},
            {"has_hallucination_predicted": True, "has_hallucination_gold": True},
            {"has_hallucination_predicted": False, "has_hallucination_gold": False},
        ]
        result = case_level_accuracy(results)
        assert result["accuracy"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_all_wrong(self):
        results = [
            {"has_hallucination_predicted": True, "has_hallucination_gold": False},
            {"has_hallucination_predicted": False, "has_hallucination_gold": True},
        ]
        result = case_level_accuracy(results)
        assert result["accuracy"] == 0.0
        assert result["true_positives"] == 0
        assert result["false_positives"] == 1
        assert result["false_negatives"] == 1

    def test_confusion_matrix(self):
        results = [
            {"has_hallucination_predicted": True, "has_hallucination_gold": True},  # TP
            {"has_hallucination_predicted": True, "has_hallucination_gold": False},  # FP
            {"has_hallucination_predicted": False, "has_hallucination_gold": False},  # TN
            {"has_hallucination_predicted": False, "has_hallucination_gold": True},  # FN
        ]
        result = case_level_accuracy(results)
        assert result["true_positives"] == 1
        assert result["false_positives"] == 1
        assert result["true_negatives"] == 1
        assert result["false_negatives"] == 1
        assert result["accuracy"] == 0.5
        assert result["precision"] == 0.5
        assert result["recall"] == 0.5


class TestHallucinationByType:
    """Test hallucination type breakdown."""

    def test_empty_results(self):
        result = hallucination_by_type([])
        assert result["total_gold_spans"] == 0

    def test_type_counting(self):
        results = [
            {"gold_span_types": ["Evident Conflict", "Subtle Baseless"], "predicted_span_types": ["Evident Conflict"]},
            {"gold_span_types": ["Evident Conflict"], "predicted_span_types": []},
        ]
        result = hallucination_by_type(results)
        assert result["total_gold_spans"] == 3
        assert result["total_predicted_spans"] == 1
        assert result["breakdown"]["Evident Conflict"]["gold_count"] == 2

    def test_type_normalization(self):
        """Test that type strings are normalized correctly."""
        results = [
            {"gold_span_types": ["evident_conflict", "SUBTLE BASELESS"], "predicted_span_types": []},
        ]
        result = hallucination_by_type(results)
        assert "Evident Conflict" in result["breakdown"]
        assert "Subtle Baseless" in result["breakdown"]


class TestTokenize:
    """Test internal tokenizer."""

    def test_basic(self):
        tokens = _tokenize("Hello, world!")
        assert "hello" in tokens
        assert "world" in tokens

    def test_empty(self):
        assert _tokenize("") == []


class TestExtractSpansFromText:
    """Test span position extraction."""

    def test_found(self):
        text = "The model has a problem with hallucination."
        positions = _extract_spans_from_text(text, ["problem with hallucination"])
        assert len(positions) == 1
        assert positions[0][0] > 0

    def test_not_found(self):
        positions = _extract_spans_from_text("Hello world", ["missing text"])
        assert len(positions) == 0

    def test_empty_span(self):
        positions = _extract_spans_from_text("Hello world", ["", "  "])
        assert len(positions) == 0

    def test_case_insensitive(self):
        text = "The Model is great."
        positions = _extract_spans_from_text(text, ["the model"])
        assert len(positions) == 1


class TestNormalizeType:
    """Test hallucination type normalization."""

    def test_evident_conflict(self):
        assert _normalize_type("Evident Conflict") == "Evident Conflict"
        assert _normalize_type("evident_conflict") == "Evident Conflict"

    def test_subtle_conflict(self):
        assert _normalize_type("Subtle Conflict") == "Subtle Conflict"

    def test_evident_baseless(self):
        assert _normalize_type("Evident Baseless Info") == "Evident Baseless"

    def test_subtle_baseless(self):
        assert _normalize_type("SUBTLE BASELESS") == "Subtle Baseless"

    def test_conflict_only(self):
        assert _normalize_type("conflict") == "Evident Conflict"

    def test_baseless_only(self):
        assert _normalize_type("baseless") == "Evident Baseless"

    def test_unknown(self):
        assert _normalize_type("completely random") == "Other"
