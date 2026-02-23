"""Tests for the RAGTruth benchmark runner."""

from unittest.mock import MagicMock, patch

from rag_bench.eval.ragtruth.loader import HallucinationSpan, RAGTruthEntry
from rag_bench.eval.ragtruth.runner import RAGTruthReport, RAGTruthRunner, RAGTruthSingleResult


def _make_entry(
    id="test_1",
    prompt="What is X?",
    source_info="X is a concept in AI.",
    response="X is a concept in AI research.",
    has_hallucination=False,
    spans=None,
):
    return RAGTruthEntry(
        id=id,
        source_id=f"src_{id}",
        task_type="QA",
        source_info=source_info,
        prompt=prompt,
        reference_response=response,
        hallucination_spans=spans or [],
        has_hallucination=has_hallucination,
    )


class TestRAGTruthSingleResult:
    """Test the single result dataclass."""

    def test_defaults(self):
        r = RAGTruthSingleResult(id="t", source_id="s", task_type="QA", prompt="q")
        assert r.has_hallucination_gold is False
        assert r.gold_spans == []
        assert r.predicted_spans == []
        assert r.error is None

    def test_with_values(self):
        r = RAGTruthSingleResult(
            id="t",
            source_id="s",
            task_type="QA",
            prompt="q",
            has_hallucination_gold=True,
            gold_spans=[{"text": "bad", "label_type": "Evident Conflict"}],
            generated_answer="some answer",
            has_hallucination_predicted=True,
            predicted_spans=["bad"],
            latency_ms=100.5,
        )
        assert r.has_hallucination_gold is True
        assert r.latency_ms == 100.5


class TestRAGTruthReport:
    """Test the report dataclass."""

    def test_defaults(self):
        r = RAGTruthReport()
        assert r.summary == {}
        assert r.results == []
        assert r.metadata == {}


class TestDetectHeuristic:
    """Test heuristic hallucination detection."""

    def test_empty_answer(self):
        runner = RAGTruthRunner(generator=None)
        result = runner._detect_heuristic("", "Some context here")
        assert result["has_hallucination"] is False
        assert result["spans"] == []

    def test_empty_context(self):
        runner = RAGTruthRunner(generator=None)
        result = runner._detect_heuristic("Some answer here.", "")
        assert result["has_hallucination"] is False

    def test_faithful_answer(self):
        context = "Machine learning is a subset of artificial intelligence that enables systems to learn from data."
        answer = "Machine learning is a subset of artificial intelligence."
        runner = RAGTruthRunner(generator=None)
        result = runner._detect_heuristic(answer, context)
        assert result["has_hallucination"] is False

    def test_hallucinated_answer(self):
        context = "The cat sat on the mat."
        answer = "The quantum chromodynamics theoretical framework predicts spontaneous symmetry breaking in hadrons."
        runner = RAGTruthRunner(generator=None)
        result = runner._detect_heuristic(answer, context)
        assert result["has_hallucination"] is True
        assert len(result["spans"]) > 0
        assert result["span_types"][0] == "Evident Baseless"

    def test_short_sentences_skipped(self):
        context = "The sky is blue."
        answer = "Yes. It is."
        runner = RAGTruthRunner(generator=None)
        result = runner._detect_heuristic(answer, context)
        assert result["has_hallucination"] is False

    def test_mixed_sentences(self):
        context = "Neural networks use backpropagation for training. The learning rate controls step size."
        answer = (
            "Neural networks use backpropagation for training. "
            "Quantum entanglement enables faster-than-light communication between distant particles."
        )
        runner = RAGTruthRunner(generator=None)
        result = runner._detect_heuristic(answer, context)
        assert result["has_hallucination"] is True
        assert len(result["spans"]) >= 1


class TestExtractSpansFromReasoning:
    """Test span extraction from judge reasoning."""

    def test_empty_reasoning(self):
        spans = RAGTruthRunner._extract_spans_from_reasoning("", "Some answer text")
        assert spans == []

    def test_quoted_text_found(self):
        answer = "The model uses attention mechanism for inference."
        reasoning = 'The claim "attention mechanism for inference" is not supported by the context.'
        spans = RAGTruthRunner._extract_spans_from_reasoning(reasoning, answer)
        assert len(spans) == 1
        assert "attention mechanism for inference" in spans[0]

    def test_quoted_text_not_in_answer(self):
        answer = "The model is fast."
        reasoning = 'The claim "quantum computing enables better results" is unsupported.'
        spans = RAGTruthRunner._extract_spans_from_reasoning(reasoning, answer)
        assert spans == []


class TestRunSingle:
    """Test running a single RAGTruth entry."""

    def test_successful_run(self):
        mock_gen = MagicMock()
        mock_gen.answer.return_value = {
            "answer": "X is a concept in AI research.",
            "deflected": False,
        }
        runner = RAGTruthRunner(generator=mock_gen)
        entry = _make_entry()
        result = runner.run_single(entry)

        assert result.id == "test_1"
        assert result.generated_answer == "X is a concept in AI research."
        assert result.error is None
        assert result.latency_ms > 0

    def test_run_with_generator_error(self):
        mock_gen = MagicMock()
        mock_gen.answer.side_effect = RuntimeError("LLM failed")
        runner = RAGTruthRunner(generator=mock_gen)
        entry = _make_entry()
        result = runner.run_single(entry)

        assert result.error == "LLM failed"
        assert result.generated_answer == ""

    def test_run_with_judge(self):
        mock_gen = MagicMock()
        mock_gen.answer.return_value = {"answer": "X is Y.", "deflected": False}
        mock_judge = MagicMock()
        mock_judge.score_faithfulness.return_value = {
            "score": 4.5,
            "reasoning": "The answer is well supported.",
        }
        runner = RAGTruthRunner(generator=mock_gen, judge=mock_judge)
        entry = _make_entry()
        result = runner.run_single(entry)

        assert result.error is None
        assert result.has_hallucination_predicted is False

    def test_run_with_judge_detecting_hallucination(self):
        mock_gen = MagicMock()
        mock_gen.answer.return_value = {"answer": "X is completely wrong.", "deflected": False}
        mock_judge = MagicMock()
        mock_judge.score_faithfulness.return_value = {
            "score": 1.5,
            "reasoning": 'The claim "X is completely wrong" is not supported.',
        }
        runner = RAGTruthRunner(generator=mock_gen, judge=mock_judge)
        entry = _make_entry()
        result = runner.run_single(entry)

        assert result.has_hallucination_predicted is True

    def test_run_with_judge_failure_falls_back(self):
        mock_gen = MagicMock()
        mock_gen.answer.return_value = {"answer": "X is a concept.", "deflected": False}
        mock_judge = MagicMock()
        mock_judge.score_faithfulness.side_effect = RuntimeError("Judge failed")
        runner = RAGTruthRunner(generator=mock_gen, judge=mock_judge)
        entry = _make_entry(source_info="X is a concept in AI.")
        result = runner.run_single(entry)

        assert result.error is None  # Should fall back to heuristic


class TestRun:
    """Test full benchmark run."""

    @patch("rag_bench.eval.ragtruth.runner.load_ragtruth")
    def test_run_aggregates(self, mock_load):
        entries = [
            _make_entry(id="1", has_hallucination=False),
            _make_entry(
                id="2", has_hallucination=True, spans=[HallucinationSpan(text="bad claim", label_type="Evident Conflict")]
            ),
        ]
        mock_load.return_value = entries

        mock_gen = MagicMock()
        mock_gen.answer.return_value = {"answer": "X is a concept.", "deflected": False}

        runner = RAGTruthRunner(generator=mock_gen)
        report = runner.run(sample_size=2)

        assert len(report.results) == 2
        assert report.metadata["benchmark"] == "ragtruth"
        assert report.metadata["total_evaluated"] == 2
        assert "hallucination_rate" in report.summary
        assert "case_level_accuracy" in report.summary
        assert "avg_latency_ms" in report.summary

    @patch("rag_bench.eval.ragtruth.runner.load_ragtruth")
    def test_run_with_all_errors(self, mock_load):
        mock_load.return_value = [_make_entry()]
        mock_gen = MagicMock()
        mock_gen.answer.side_effect = RuntimeError("fail")

        runner = RAGTruthRunner(generator=mock_gen)
        report = runner.run(sample_size=1)

        assert len(report.results) == 1
        assert report.results[0].error is not None
        assert report.metadata["errors"] == 1
