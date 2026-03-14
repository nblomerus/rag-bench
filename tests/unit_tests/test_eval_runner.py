"""Tests for rag_bench.eval.judge, runner, and report modules."""

import json
import os
import sys
import tempfile
import unittest.mock
from unittest.mock import MagicMock, patch

import pytest

from rag_bench.core.types import ChunkData, GenerationResult, RetrievalResult
from rag_bench.eval.benchmark import (
    BenchmarkEntry,
    get_benchmark,
    get_benchmark_by_difficulty,
    get_benchmark_by_topic,
    get_benchmark_by_type,
)
from rag_bench.eval.judge import JudgeLLM
from rag_bench.eval.report import (
    generate_markdown_report,
    generate_terminal_summary,
    report_to_json,
    save_report,
)
from rag_bench.eval.runner import EvalReport, EvalRunner, SingleEvalResult, _clear_cuda_cache

# ═══════════════════════════════════════════════════════════════════════════
# JudgeLLM Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestJudgeLLMParsing:
    """Test score parsing without actual LLM calls."""

    @pytest.fixture
    def judge(self):
        j = JudgeLLM.__new__(JudgeLLM)
        return j

    def test_parse_score_standard(self, judge):
        score, reasoning = judge._parse_score("Score: 4\nReasoning: Good answer with sources")
        assert score == 4.0
        assert "Good answer" in reasoning

    def test_parse_score_with_slash(self, judge):
        score, reasoning = judge._parse_score("Score: 3/5\nReasoning: Mixed quality")
        assert score == 3.0

    def test_parse_score_number_first_line(self, judge):
        score, reasoning = judge._parse_score("4\nThe answer is well-grounded")
        assert score == 4.0

    def test_parse_score_json(self, judge):
        score, reasoning = judge._parse_score('{"score": 5, "reasoning": "Perfect"}')
        assert score == 5.0
        assert reasoning == "Perfect"

    def test_parse_score_unparseable(self, judge):
        score, reasoning = judge._parse_score("I don't understand the question")
        assert score == 3.0
        assert "Could not parse" in reasoning

    def test_parse_score_clamped(self, judge):
        score, _ = judge._parse_score("Score: 10")
        assert score == 5.0

        score, _ = judge._parse_score("Score: 0")
        assert score == 1.0


class TestJudgeLLMScoring:
    def test_score_faithfulness_with_mock(self):
        mock_backend = MagicMock()
        mock_backend.generate.return_value = "Score: 4\nReasoning: Well grounded"
        judge = JudgeLLM(mock_backend)

        result = judge.score_faithfulness(
            question="What is attention?",
            answer="Attention computes weights [Source 1].",
            source_passages=["The attention mechanism computes a weighted sum."],
        )
        assert result["score"] == 4.0
        assert "Well grounded" in result["reasoning"]
        mock_backend.generate.assert_called_once()

    def test_score_relevance_with_mock(self):
        mock_backend = MagicMock()
        mock_backend.generate.return_value = "Score: 5\nReasoning: Directly answers"
        judge = JudgeLLM(mock_backend)

        result = judge.score_relevance(
            question="What is attention?",
            answer="Attention is a mechanism for computing weights.",
        )
        assert result["score"] == 5.0

    def test_faithfulness_fallback_on_error(self):
        mock_backend = MagicMock()
        mock_backend.generate.side_effect = Exception("Connection error")
        judge = JudgeLLM(mock_backend)

        result = judge.score_faithfulness(
            question="What is attention?",
            answer="The attention mechanism computes weighted sums.",
            source_passages=["The attention mechanism computes a weighted sum of values."],
        )
        # Should fallback to heuristic
        assert 1.0 <= result["score"] <= 5.0
        assert "Heuristic" in result["reasoning"] or "heuristic" in result["reasoning"]

    def test_relevance_fallback_on_error(self):
        mock_backend = MagicMock()
        mock_backend.generate.side_effect = Exception("Timeout")
        judge = JudgeLLM(mock_backend)

        result = judge.score_relevance("question", "answer")
        assert result["score"] == 3.0

    def test_faithfulness_heuristic_no_sources(self):
        judge = JudgeLLM.__new__(JudgeLLM)
        result = judge._faithfulness_heuristic("some answer", [])
        assert result["score"] == 1.0

    def test_faithfulness_heuristic_empty_answer(self):
        """Empty answer → line 172: 'Empty answer' branch."""
        judge = JudgeLLM.__new__(JudgeLLM)
        result = judge._faithfulness_heuristic("", ["Some source passage."])
        assert result["score"] == 1.0
        assert result["reasoning"] == "Empty answer"

    def test_faithfulness_heuristic_sentence_with_no_words(self):
        """Sentence that has no words after split → line 178: 'if not words: continue'."""
        judge = JudgeLLM.__new__(JudgeLLM)
        # Answer with punctuation-only "sentence" and a real sentence
        result = judge._faithfulness_heuristic("Good answer. !!!", ["Good answer content."])
        assert "score" in result
        assert 1.0 <= result["score"] <= 5.0


# ═══════════════════════════════════════════════════════════════════════════
# EvalRunner Tests
# ═══════════════════════════════════════════════════════════════════════════


def _make_entry(**kwargs):
    defaults = {
        "id": "test-entry",
        "question": "What is attention?",
        "expected_sources": ["1706.03762"],
        "acceptable_sources": ["1706.03762"],
        "expected_answer_contains": ["attention"],
        "expected_answer_excludes": [],
        "query_type": "definition",
        "topic": "transformers",
        "difficulty": "easy",
        "should_deflect": False,
    }
    defaults.update(kwargs)
    return BenchmarkEntry(**defaults)


def _make_retrieval_results():
    """Build typed RetrievalResult objects for test mocks."""
    chunk = ChunkData(
        chunk_id="c1",
        doc_id="arxiv_1706_03762",
        text="Attention mechanism...",
        section="abstract",
        metadata={"paper_id": "arxiv_1706_03762", "title": "Attention Is All You Need", "section": "abstract"},
    )
    return [RetrievalResult(chunk=chunk, relevance_score=7.0, sources=["dense"])]


def _make_generation_result(**kwargs):
    """Build a typed GenerationResult for test mocks."""
    retrieval = _make_retrieval_results()
    defaults = dict(
        answer="The Transformer uses attention [Source 1].",
        deflected=False,
        sources=["[1] Vaswani et al."],
        deflection_reason=None,
        results=retrieval,
        scores=[7.0],
    )
    defaults.update(kwargs)
    return GenerationResult(**defaults)


class TestEvalRunner:
    @pytest.fixture
    def mock_retriever(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = _make_retrieval_results()
        return retriever

    @pytest.fixture
    def mock_generator(self):
        gen = MagicMock()
        gen.generate.return_value = _make_generation_result()
        return gen

    def test_run_single_basic(self, mock_retriever, mock_generator):
        runner = EvalRunner(
            retriever=mock_retriever,
            generator=mock_generator,
            benchmark=[_make_entry()],
        )
        result = runner.run_single(_make_entry())
        assert result.id == "test-entry"
        assert result.retrieval.get("mrr") == 1.0
        assert result.deflection["correct"] is True
        assert result.latency_ms > 0

    def test_run_single_with_judge(self, mock_retriever, mock_generator):
        mock_backend = MagicMock()
        mock_backend.generate.return_value = "Score: 4\nReasoning: Good"
        judge = JudgeLLM(mock_backend)

        runner = EvalRunner(
            retriever=mock_retriever,
            generator=mock_generator,
            judge=judge,
            benchmark=[_make_entry()],
        )
        result = runner.run_single(_make_entry())
        assert result.faithfulness.get("score") == 4.0
        assert result.relevance.get("score") == 4.0

    def test_run_single_deflected(self, mock_retriever, mock_generator):
        mock_generator.generate.return_value = _make_generation_result(
            answer="I can't answer that.",
            deflected=True,
            deflection_reason="Off topic",
        )
        entry = _make_entry(should_deflect=True)
        runner = EvalRunner(
            retriever=mock_retriever,
            generator=mock_generator,
            benchmark=[entry],
        )
        result = runner.run_single(entry)
        assert result.deflection["expected"] is True
        assert result.deflection["actual"] is True
        assert result.deflection["correct"] is True
        # Citation/completeness should be empty for deflections
        assert result.citation == {}

    def test_run_all_aggregation(self, mock_retriever, mock_generator):
        entries = [
            _make_entry(id="q1"),
            _make_entry(id="q2"),
        ]
        runner = EvalRunner(
            retriever=mock_retriever,
            generator=mock_generator,
            benchmark=entries,
        )
        report = runner.run_all()
        assert report.summary["total_queries"] == 2
        assert len(report.results) == 2
        assert "transformers" in report.by_topic

    def test_run_all_filter_topic(self, mock_retriever, mock_generator):
        entries = [
            _make_entry(id="q1", topic="transformers"),
            _make_entry(id="q2", topic="language_models"),
        ]
        runner = EvalRunner(
            retriever=mock_retriever,
            generator=mock_generator,
            benchmark=entries,
        )
        report = runner.run_all(filter_topic="transformers")
        assert report.summary["total_queries"] == 1

    def test_run_all_filter_type(self, mock_retriever, mock_generator):
        entries = [
            _make_entry(id="q1", query_type="definition"),
            _make_entry(id="q2", query_type="factual"),
        ]
        runner = EvalRunner(
            retriever=mock_retriever,
            generator=mock_generator,
            benchmark=entries,
        )
        report = runner.run_all(filter_type="factual")
        assert report.summary["total_queries"] == 1

    def test_retrieval_only_mode(self, mock_retriever, mock_generator):
        runner = EvalRunner(
            retriever=mock_retriever,
            generator=mock_generator,
            benchmark=[_make_entry()],
        )
        report = runner.run_all(retrieval_only=True)
        # Generator should NOT be called
        mock_generator.generate.assert_not_called()
        assert report.summary["total_queries"] == 1
        assert report.metadata["retrieval_only"] is True

    def test_error_handling(self, mock_retriever, mock_generator):
        mock_generator.generate.side_effect = Exception("Connection failed")
        runner = EvalRunner(
            retriever=mock_retriever,
            generator=mock_generator,
            benchmark=[_make_entry()],
        )
        result = runner.run_single(_make_entry())
        assert result.error == "Connection failed"

    def test_group_by(self, mock_retriever, mock_generator):
        entries = [
            _make_entry(id="q1", difficulty="easy"),
            _make_entry(id="q2", difficulty="hard"),
            _make_entry(id="q3", difficulty="easy"),
        ]
        runner = EvalRunner(
            retriever=mock_retriever,
            generator=mock_generator,
            benchmark=entries,
        )
        report = runner.run_all()
        assert "easy" in report.by_difficulty
        assert "hard" in report.by_difficulty
        assert report.by_difficulty["easy"]["total_queries"] == 2
        assert report.by_difficulty["hard"]["total_queries"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# Report Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestReport:
    @pytest.fixture
    def sample_report(self):
        return EvalReport(
            results=[
                SingleEvalResult(
                    id="test1",
                    question="What is attention?",
                    query_type="definition",
                    topic="transformers",
                    difficulty="easy",
                    retrieval={"mrr": 1.0, "precision_at_k": 0.4, "recall_at_k": 1.0, "ndcg_at_k": 0.9, "hit_rate": 1.0},
                    citation={"precision": 0.8, "recall": 1.0, "density": 1.5},
                    completeness={"score": 0.75},
                    faithfulness={"score": 4.0},
                    relevance={"score": 4.5},
                    deflection={"expected": False, "actual": False, "correct": True},
                    latency_ms=1500,
                ),
            ],
            summary={
                "total_queries": 1,
                "retrieval_mrr": 1.0,
                "retrieval_precision_at_5": 0.4,
                "retrieval_recall_at_5": 1.0,
                "retrieval_ndcg_at_5": 0.9,
                "retrieval_hit_rate": 1.0,
                "avg_faithfulness": 4.0,
                "avg_relevance": 4.5,
                "avg_citation_precision": 0.8,
                "avg_citation_recall": 1.0,
                "avg_citation_density": 1.5,
                "avg_completeness": 0.75,
                "deflection_accuracy": 1.0,
                "avg_latency_ms": 1500,
            },
            by_topic={"transformers": {"total_queries": 1, "retrieval_mrr": 1.0}},
            by_query_type={"definition": {"total_queries": 1}},
            by_difficulty={"easy": {"total_queries": 1}},
            metadata={"timestamp": "2026-02-16 14:30:00", "total_queries": 1},
        )

    def test_terminal_summary(self, sample_report):
        output = generate_terminal_summary(sample_report)
        assert "RAG-Bench Evaluation Report" in output
        assert "1.0000" in output  # MRR
        assert "4.0 / 5.0" in output  # Faithfulness

    def test_markdown_report(self, sample_report):
        output = generate_markdown_report(sample_report)
        assert "# RAG-Bench Evaluation Report" in output
        assert "| Retrieval MRR |" in output
        assert "transformers" in output

    def test_report_to_json(self, sample_report):
        data = report_to_json(sample_report)
        assert data["summary"]["total_queries"] == 1
        assert isinstance(data["results"], list)
        # Ensure it's JSON-serializable
        json.dumps(data)

    def test_markdown_failed_queries(self):
        """Low-scoring queries should appear in the failed section."""
        report = EvalReport(
            results=[
                SingleEvalResult(
                    id="bad1",
                    question="Bad query",
                    query_type="definition",
                    topic="test",
                    difficulty="easy",
                    retrieval={"mrr": 0.1},  # Low MRR → should be flagged
                    deflection={"expected": False, "actual": False, "correct": True},
                ),
            ],
            summary={"total_queries": 1},
            metadata={"timestamp": "test"},
        )
        output = generate_markdown_report(report)
        assert "Failed" in output or "Low-Scoring" in output
        assert "bad1" in output


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark Helper Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBenchmarkHelpers:
    def test_get_benchmark_returns_all(self):
        entries = get_benchmark()
        assert len(entries) > 0
        assert isinstance(entries[0], BenchmarkEntry)

    def test_get_benchmark_by_topic(self):
        entries = get_benchmark_by_topic("transformers")
        assert len(entries) > 0
        assert all(e.topic == "transformers" for e in entries)

    def test_get_benchmark_by_type(self):
        entries = get_benchmark_by_type("definition")
        assert len(entries) > 0
        assert all(e.query_type == "definition" for e in entries)

    def test_get_benchmark_by_difficulty(self):
        entries = get_benchmark_by_difficulty("easy")
        assert len(entries) > 0
        assert all(e.difficulty == "easy" for e in entries)

    def test_get_benchmark_by_topic_empty(self):
        entries = get_benchmark_by_topic("nonexistent_topic")
        assert entries == []

    def test_get_benchmark_by_type_empty(self):
        entries = get_benchmark_by_type("nonexistent_type")
        assert entries == []

    def test_get_benchmark_by_difficulty_empty(self):
        entries = get_benchmark_by_difficulty("nonexistent")
        assert entries == []


# ═══════════════════════════════════════════════════════════════════════════
# Save Report Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSaveReport:
    def test_save_report_creates_files(self):
        report = EvalReport(
            results=[
                SingleEvalResult(
                    id="t1",
                    question="Q?",
                    query_type="definition",
                    topic="test",
                    difficulty="easy",
                    retrieval={"mrr": 1.0},
                    deflection={"expected": False, "actual": False, "correct": True},
                )
            ],
            summary={"total_queries": 1, "retrieval_mrr": 1.0},
            metadata={"timestamp": "2026-02-17 10:00:00"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path, md_path = save_report(report, tmpdir)
            assert os.path.exists(json_path)
            assert os.path.exists(md_path)
            assert json_path.endswith(".json")
            assert md_path.endswith(".md")
            with open(json_path) as f:
                data = json.load(f)
            assert data["summary"]["total_queries"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# Report Edge Case Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMarkdownReportEdgeCases:
    def test_retrieval_only_mode_label(self):
        """Cover the retrieval_only flag in report metadata (line 71 of report.py)."""
        report = EvalReport(
            results=[],
            summary={"total_queries": 0},
            metadata={"timestamp": "test", "retrieval_only": True},
        )
        output = generate_markdown_report(report)
        assert "Retrieval Only" in output

    def test_failed_query_with_error_field(self):
        """Cover the error branch in failed queries section (line 155 of report.py)."""
        report = EvalReport(
            results=[
                SingleEvalResult(
                    id="err1",
                    question="Error query",
                    query_type="definition",
                    topic="test",
                    difficulty="easy",
                    retrieval={},
                    deflection={"expected": False, "actual": False, "correct": True},
                    error="Connection failed",
                ),
            ],
            summary={"total_queries": 1},
            metadata={"timestamp": "test"},
        )
        output = generate_markdown_report(report)
        assert "Connection failed" in output
        assert "err1" in output

    def test_failed_query_with_low_faithfulness(self):
        """Cover faithfulness check in failed queries section."""
        report = EvalReport(
            results=[
                SingleEvalResult(
                    id="faith1",
                    question="Faith query",
                    query_type="definition",
                    topic="test",
                    difficulty="easy",
                    retrieval={"mrr": 1.0},
                    faithfulness={"score": 2.0},
                    deflection={"expected": False, "actual": False, "correct": True},
                ),
            ],
            summary={"total_queries": 1},
            metadata={"timestamp": "test"},
        )
        output = generate_markdown_report(report)
        assert "faith1" in output
        assert "2.0" in output

    def test_failed_query_with_wrong_deflection(self):
        """Cover deflection mismatch in failed queries section (line 162 of report.py)."""
        report = EvalReport(
            results=[
                SingleEvalResult(
                    id="defl1",
                    question="Test query",
                    query_type="deflection",
                    topic="test",
                    difficulty="easy",
                    retrieval={},
                    deflection={"expected": True, "actual": False, "correct": False},
                ),
            ],
            summary={"total_queries": 1},
            metadata={"timestamp": "test"},
        )
        output = generate_markdown_report(report)
        assert "defl1" in output
        assert "expected=True" in output

    def test_failed_query_with_low_citation_recall(self):
        """Cover low citation recall in failed queries section."""
        report = EvalReport(
            results=[
                SingleEvalResult(
                    id="cit1",
                    question="Citation query",
                    query_type="definition",
                    topic="test",
                    difficulty="easy",
                    retrieval={"mrr": 1.0},
                    citation={"recall": 0.2, "precision": 0.5},
                    deflection={"expected": False, "actual": False, "correct": True},
                ),
            ],
            summary={"total_queries": 1},
            metadata={"timestamp": "test"},
        )
        output = generate_markdown_report(report)
        assert "cit1" in output


# ═══════════════════════════════════════════════════════════════════════════
# Runner Edge Case Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRunnerEdgeCases:
    @pytest.fixture
    def mock_retriever(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        return retriever

    @pytest.fixture
    def mock_generator(self):
        gen = MagicMock()
        gen.generate.return_value = _make_generation_result(
            answer="Deflected.",
            deflected=True,
            deflection_reason="Off topic",
            results=[],
            scores=[],
        )
        return gen

    def test_run_single_no_expected_sources(self, mock_retriever, mock_generator):
        """Cover branch: entry with no expected_sources (deflection)."""
        entry = BenchmarkEntry(
            id="defl-test",
            question="What is cake?",
            expected_sources=[],
            query_type="deflection",
            topic="off_topic",
            difficulty="easy",
            should_deflect=True,
        )
        runner = EvalRunner(retriever=mock_retriever, generator=mock_generator, benchmark=[entry])
        result = runner.run_single(entry)
        assert result.retrieval == {}

    def test_run_all_filter_difficulty(self, mock_retriever, mock_generator):
        """Cover filter_difficulty in run_all."""
        entries = [
            BenchmarkEntry(
                id="q1", question="Q1?", expected_sources=[], query_type="definition", topic="test", difficulty="easy"
            ),
            BenchmarkEntry(
                id="q2", question="Q2?", expected_sources=[], query_type="definition", topic="test", difficulty="hard"
            ),
        ]
        runner = EvalRunner(retriever=mock_retriever, generator=mock_generator, benchmark=entries)
        report = runner.run_all(filter_difficulty="hard")
        assert report.summary["total_queries"] == 1

    def test_run_all_empty_results_aggregate(self, mock_retriever, mock_generator):
        """Cover _aggregate_results with empty list."""
        runner = EvalRunner(retriever=mock_retriever, generator=mock_generator, benchmark=[])
        report = runner.run_all(filter_topic="nonexistent_topic")
        assert report.summary == {}

    def test_retrieval_only_no_expected_sources(self, mock_retriever):
        """Cover retrieval_only with no expected_sources."""
        entry = BenchmarkEntry(
            id="defl-test",
            question="Off topic?",
            expected_sources=[],
            query_type="deflection",
            topic="off_topic",
            difficulty="easy",
            should_deflect=True,
        )
        runner = EvalRunner(retriever=mock_retriever, generator=MagicMock(), benchmark=[entry])
        report = runner.run_all(retrieval_only=True)
        result = report.results[0]
        assert result.retrieval == {}

    def test_retrieval_only_error_handling(self, mock_retriever):
        """Cover exception in retrieval_only."""
        mock_retriever.retrieve.side_effect = Exception("Retrieval failed")
        entry = BenchmarkEntry(
            id="err-test",
            question="Test?",
            expected_sources=["1706.03762"],
            query_type="definition",
            topic="test",
            difficulty="easy",
        )
        runner = EvalRunner(retriever=mock_retriever, generator=MagicMock(), benchmark=[entry])
        report = runner.run_all(retrieval_only=True)
        assert report.results[0].error == "Retrieval failed"


# ═══════════════════════════════════════════════════════════════════════════
# Judge Error Isolation Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestJudgeErrorIsolation:
    """Test that faithfulness and relevance judge calls fail independently."""

    def test_faithfulness_failure_does_not_skip_relevance(self):
        """If faithfulness judge fails, relevance should still run."""
        mock_backend = MagicMock()
        call_count = [0]

        def generate_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:  # faithfulness calls (2 attempts with retry)
                raise Exception("Faithfulness OOM")
            return "Score: 4\nReasoning: Relevant answer"

        mock_backend.generate.side_effect = generate_side_effect
        judge = JudgeLLM(mock_backend)

        retriever = MagicMock()
        retriever.retrieve.return_value = _make_retrieval_results()
        gen = MagicMock()
        gen.generate.return_value = _make_generation_result()

        entry = _make_entry()
        runner = EvalRunner(retriever=retriever, generator=gen, judge=judge, benchmark=[entry])
        result = runner.run_single(entry)

        # Faithfulness should fall back to heuristic (non-empty)
        assert result.faithfulness.get("score", 0) > 0
        # Relevance should succeed
        assert result.relevance.get("score", 0) == 4.0

    def test_relevance_failure_preserves_faithfulness(self):
        """If relevance judge fails, faithfulness should still be recorded."""
        mock_backend = MagicMock()
        call_count = [0]

        def generate_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 1:  # first call is faithfulness
                return "Score: 5\nReasoning: Perfect"
            raise Exception("Relevance timeout")

        mock_backend.generate.side_effect = generate_side_effect
        judge = JudgeLLM(mock_backend)

        retriever = MagicMock()
        retriever.retrieve.return_value = _make_retrieval_results()
        gen = MagicMock()
        gen.generate.return_value = _make_generation_result()

        entry = _make_entry()
        runner = EvalRunner(retriever=retriever, generator=gen, judge=judge, benchmark=[entry])
        result = runner.run_single(entry)

        # Faithfulness should have the score from the LLM
        assert result.faithfulness.get("score") == 5.0
        # Relevance should be empty (both attempts failed)
        # The judge itself has retry logic, so it will try twice then return fallback
        assert result.error is None  # No top-level error


# ═══════════════════════════════════════════════════════════════════════════
# CUDA Memory Management Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCudaMemoryManagement:
    def test_clear_cuda_cache_without_torch(self):
        """_clear_cuda_cache should not crash if torch is not available."""
        with unittest.mock.patch.dict("sys.modules", {"torch": None}):
            _clear_cuda_cache()  # Should not raise

    def test_clear_cuda_cache_without_gpu(self):
        """_clear_cuda_cache should handle no CUDA gracefully."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with unittest.mock.patch.dict(sys.modules, {"torch": mock_torch}):
            _clear_cuda_cache()
        mock_torch.cuda.empty_cache.assert_not_called()

    def test_oom_retry_in_run_single(self):
        """CUDA OOM should trigger cache clear and retry."""
        retriever = MagicMock()
        retrieve_call_count = [0]

        def retrieve_side_effect(*args, **kwargs):
            retrieve_call_count[0] += 1
            if retrieve_call_count[0] == 1:
                raise RuntimeError("CUDA out of memory")
            return _make_retrieval_results()

        retriever.retrieve.side_effect = retrieve_side_effect

        gen = MagicMock()
        gen.generate.return_value = _make_generation_result()

        entry = _make_entry()
        runner = EvalRunner(retriever=retriever, generator=gen, benchmark=[entry])
        result = runner.run_single(entry)

        assert result.error is None
        assert retrieve_call_count[0] == 2  # First call OOM, second succeeds

    def test_non_oom_runtime_error_not_retried(self):
        """Non-OOM RuntimeError should propagate without retry."""
        retriever = MagicMock()
        retriever.retrieve.side_effect = RuntimeError("Some other error")

        gen = MagicMock()

        entry = _make_entry()
        runner = EvalRunner(retriever=retriever, generator=gen, benchmark=[entry])
        result = runner.run_single(entry)

        assert result.error == "Some other error"
        retriever.retrieve.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# Judge Retry Logic Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestJudgeRetryLogic:
    def test_faithfulness_retries_on_failure(self):
        """Faithfulness should retry once before falling back to heuristic."""
        mock_backend = MagicMock()
        mock_backend.generate.side_effect = [
            Exception("First failure"),
            "Score: 4\nReasoning: Good",
        ]
        judge = JudgeLLM(mock_backend)

        result = judge.score_faithfulness("Q?", "Answer text.", ["Source text."])
        assert result["score"] == 4.0
        assert mock_backend.generate.call_count == 2

    def test_faithfulness_falls_back_after_two_failures(self):
        """Faithfulness falls back to heuristic after 2 failures."""
        mock_backend = MagicMock()
        mock_backend.generate.side_effect = Exception("Always fails")
        judge = JudgeLLM(mock_backend)

        result = judge.score_faithfulness("Q?", "Source text answer.", ["Source text content."])
        assert result["score"] >= 1.0
        assert "euristic" in result.get("reasoning", "")  # "Heuristic"
        assert mock_backend.generate.call_count == 2

    def test_relevance_retries_on_failure(self):
        """Relevance should retry once before returning default."""
        mock_backend = MagicMock()
        mock_backend.generate.side_effect = [
            Exception("First failure"),
            "Score: 5\nReasoning: Perfect",
        ]
        judge = JudgeLLM(mock_backend)

        result = judge.score_relevance("Q?", "Answer text.")
        assert result["score"] == 5.0
        assert mock_backend.generate.call_count == 2

    def test_relevance_returns_default_after_two_failures(self):
        """Relevance returns 3.0 after 2 failures."""
        mock_backend = MagicMock()
        mock_backend.generate.side_effect = Exception("Always fails")
        judge = JudgeLLM(mock_backend)

        result = judge.score_relevance("Q?", "Answer.")
        assert result["score"] == 3.0
        assert mock_backend.generate.call_count == 2


# ═══════════════════════════════════════════════════════════════════════════
# Runner Judge Exception Isolation Tests (lines 148-153)
# ═══════════════════════════════════════════════════════════════════════════


class TestRunnerJudgeExceptionIsolation:
    """Cover lines 148-149 and 152-153: runner-level except blocks for judge calls."""

    def test_faithfulness_exception_caught_at_runner_level(self):
        """When judge.score_faithfulness raises, the runner catches it."""
        judge = MagicMock()
        judge.score_faithfulness.side_effect = Exception("OOM in judge")
        judge.score_relevance.return_value = {"score": 4.0, "reasoning": "Good"}

        retriever = MagicMock()
        retriever.retrieve.return_value = _make_retrieval_results()
        gen = MagicMock()
        gen.generate.return_value = _make_generation_result()

        entry = _make_entry()
        runner = EvalRunner(retriever=retriever, generator=gen, judge=judge, benchmark=[entry])
        result = runner.run_single(entry)

        # Faithfulness should be empty (exception caught), relevance should succeed
        assert result.faithfulness is None or result.faithfulness == {}
        assert result.relevance.get("score") == 4.0
        assert result.error is None

    def test_relevance_exception_caught_at_runner_level(self):
        """When judge.score_relevance raises, the runner catches it."""
        judge = MagicMock()
        judge.score_faithfulness.return_value = {"score": 5.0, "reasoning": "Perfect"}
        judge.score_relevance.side_effect = Exception("Timeout in judge")

        retriever = MagicMock()
        retriever.retrieve.return_value = _make_retrieval_results()
        gen = MagicMock()
        gen.generate.return_value = _make_generation_result()

        entry = _make_entry()
        runner = EvalRunner(retriever=retriever, generator=gen, judge=judge, benchmark=[entry])
        result = runner.run_single(entry)

        # Faithfulness should succeed, relevance should be empty (exception caught)
        assert result.faithfulness.get("score") == 5.0
        assert result.relevance is None or result.relevance == {}
        assert result.error is None


# ═══════════════════════════════════════════════════════════════════════════
# CUDA Cache Clear in run_all (line 188)
# ═══════════════════════════════════════════════════════════════════════════


class TestCudaCacheClearInRunAll:
    """Cover line 188: _clear_cuda_cache called every 10 entries in run_all."""

    def test_cuda_cache_cleared_every_10_entries(self):
        """run_all with 10+ entries triggers _clear_cuda_cache."""
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        gen = MagicMock()
        gen.generate.return_value = _make_generation_result(
            answer="Answer [Source 1].",
            deflected=True,
            deflection_reason="Off topic",
            results=[],
            scores=[],
        )
        entries = [_make_entry(id=f"q{i}", should_deflect=True) for i in range(11)]
        runner = EvalRunner(retriever=retriever, generator=gen, benchmark=entries)

        with patch("rag_bench.eval.runner._clear_cuda_cache") as mock_clear:
            runner.run_all()
            assert mock_clear.call_count == 1  # Called after entry 10
