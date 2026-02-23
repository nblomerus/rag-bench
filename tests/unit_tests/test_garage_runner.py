"""Tests for the GaRAGe benchmark runner."""

from unittest.mock import MagicMock, patch

from rag_bench.eval.garage.loader import GaRAGeEntry, GaRAGePassage
from rag_bench.eval.garage.runner import GaRAGeReport, GaRAGeRunner, GaRAGeSingleResult


def _make_entry(
    id="g1",
    question="What is X?",
    gold_answer="X is a concept.",
    should_deflect=False,
    passages=None,
):
    if passages is None:
        passages = [
            GaRAGePassage(passage_id="p1", text="X is a concept in machine learning.", is_relevant=True),
            GaRAGePassage(passage_id="p2", text="Y is unrelated noise.", is_relevant=False),
        ]
    return GaRAGeEntry(
        id=id,
        question=question,
        gold_answer=gold_answer,
        should_deflect=should_deflect,
        passages=passages,
        question_tag="factual",
        topic_tag="ml",
    )


class TestGaRAGeSingleResult:
    """Test the single result dataclass."""

    def test_defaults(self):
        r = GaRAGeSingleResult(id="g", question="q", gold_answer="a")
        assert r.generated_answer == ""
        assert r.should_deflect is False
        assert r.did_deflect is False
        assert r.raf == {}
        assert r.error is None

    def test_with_values(self):
        r = GaRAGeSingleResult(
            id="g",
            question="q",
            gold_answer="a",
            generated_answer="answer",
            latency_ms=50.0,
            raf={"raf_score": 0.8},
        )
        assert r.latency_ms == 50.0
        assert r.raf["raf_score"] == 0.8


class TestGaRAGeReport:
    """Test the report dataclass."""

    def test_defaults(self):
        r = GaRAGeReport()
        assert r.summary == {}
        assert r.results == []


class TestRunSingle:
    """Test running a single GaRAGe entry."""

    def test_successful_run(self):
        mock_gen = MagicMock()
        mock_gen.answer.return_value = {
            "answer": "X is a concept in machine learning [Source 1].",
            "deflected": False,
        }
        runner = GaRAGeRunner(generator=mock_gen)
        entry = _make_entry()
        result = runner.run_single(entry)

        assert result.id == "g1"
        assert result.error is None
        assert result.generated_answer != ""
        assert result.latency_ms > 0
        assert "raf_score" in result.raf

    def test_deflected_answer(self):
        mock_gen = MagicMock()
        mock_gen.answer.return_value = {
            "answer": "I cannot answer this.",
            "deflected": True,
        }
        runner = GaRAGeRunner(generator=mock_gen)
        entry = _make_entry()
        result = runner.run_single(entry)

        assert result.did_deflect is True
        # RAf/URAf not computed for deflected
        assert result.raf == {}

    def test_generator_error(self):
        mock_gen = MagicMock()
        mock_gen.answer.side_effect = RuntimeError("LLM failed")
        runner = GaRAGeRunner(generator=mock_gen)
        entry = _make_entry()
        result = runner.run_single(entry)

        assert result.error == "LLM failed"

    def test_entry_with_no_passages(self):
        mock_gen = MagicMock()
        mock_gen.answer.return_value = {"answer": "Dunno.", "deflected": False}
        runner = GaRAGeRunner(generator=mock_gen)
        entry = _make_entry(passages=[])
        result = runner.run_single(entry)
        assert result.error is None


class TestAggregate:
    """Test result aggregation."""

    def test_aggregate_answered(self):
        runner = GaRAGeRunner(generator=None)
        results = [
            GaRAGeSingleResult(
                id="1",
                question="q",
                gold_answer="a",
                generated_answer="ans",
                did_deflect=False,
                raf={"raf_score": 0.8},
                uraf={"uraf_score": 0.7},
                attribution={"attribution_f1": 0.6, "attribution_precision": 0.5, "attribution_recall": 0.7},
                latency_ms=100,
                question_tag="factual",
            ),
            GaRAGeSingleResult(
                id="2",
                question="q2",
                gold_answer="a2",
                generated_answer="ans2",
                did_deflect=False,
                raf={"raf_score": 0.6},
                uraf={"uraf_score": 0.5},
                attribution={"attribution_f1": 0.4, "attribution_precision": 0.3, "attribution_recall": 0.5},
                latency_ms=200,
                question_tag="factual",
            ),
        ]
        report = runner._aggregate(results)

        assert report.summary["raf_score"] == 0.7
        assert report.summary["total_answered"] == 2
        assert "factual" in report.by_category
        assert report.by_category["factual"]["count"] == 2

    def test_aggregate_all_deflected(self):
        runner = GaRAGeRunner(generator=None)
        results = [
            GaRAGeSingleResult(
                id="1",
                question="q",
                gold_answer="a",
                did_deflect=True,
                should_deflect=True,
            ),
        ]
        report = runner._aggregate(results)
        assert report.summary["total_answered"] == 0
        assert report.summary["raf_score"] == 0.0

    def test_aggregate_with_errors(self):
        runner = GaRAGeRunner(generator=None)
        results = [
            GaRAGeSingleResult(id="1", question="q", gold_answer="a", error="fail"),
        ]
        report = runner._aggregate(results)
        assert report.summary["total_answered"] == 0

    def test_mean_helper(self):
        assert GaRAGeRunner._mean([1, 2, 3]) == 2.0
        assert GaRAGeRunner._mean([]) == 0.0


class TestRun:
    """Test the full run method."""

    @patch("rag_bench.eval.garage.runner.load_garage")
    def test_run_end_to_end(self, mock_load):
        mock_load.return_value = [_make_entry()]
        mock_gen = MagicMock()
        mock_gen.answer.return_value = {"answer": "X is a concept.", "deflected": False}

        runner = GaRAGeRunner(generator=mock_gen)
        report = runner.run(sample_size=1)

        assert report.metadata["benchmark"] == "garage"
        assert report.metadata["total_evaluated"] == 1
        assert len(report.results) == 1

    @patch("rag_bench.eval.garage.runner.load_garage")
    def test_run_with_multiple_entries(self, mock_load):
        mock_load.return_value = [_make_entry(id="1"), _make_entry(id="2")]
        mock_gen = MagicMock()
        mock_gen.answer.return_value = {"answer": "ans", "deflected": False}

        runner = GaRAGeRunner(generator=mock_gen)
        report = runner.run(sample_size=2)

        assert report.metadata["total_evaluated"] == 2
        assert report.metadata["errors"] == 0
