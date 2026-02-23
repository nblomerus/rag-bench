"""Tests for benchmark API schemas."""

from rag_bench.api.schemas import (
    BenchmarkEvalRequest,
    BenchmarkEvalResponse,
    BenchmarkHistoryEntry,
    BenchmarkHistoryResponse,
    BenchmarkResultItem,
)


class TestBenchmarkEvalRequest:
    """Test benchmark evaluation request schema."""

    def test_valid_ragbench_request(self):
        req = BenchmarkEvalRequest(benchmark="ragbench", sample_size=50)
        assert req.benchmark == "ragbench"
        assert req.sample_size == 50

    def test_valid_ragtruth_request(self):
        req = BenchmarkEvalRequest(benchmark="ragtruth", sample_size=100)
        assert req.benchmark == "ragtruth"
        assert req.sample_size == 100

    def test_default_sample_size(self):
        req = BenchmarkEvalRequest(benchmark="ragbench")
        assert req.sample_size == 50

    def test_zero_sample_size(self):
        req = BenchmarkEvalRequest(benchmark="ragbench", sample_size=0)
        assert req.sample_size == 0


class TestBenchmarkEvalResponse:
    """Test benchmark evaluation response schema."""

    def test_minimal_response(self):
        resp = BenchmarkEvalResponse(benchmark="ragbench")
        assert resp.benchmark == "ragbench"
        assert resp.summary == {}
        assert resp.results == []
        assert resp.metadata == {}

    def test_full_response(self):
        results = [
            BenchmarkResultItem(
                id="test_1",
                question="What is X?",
                answer="X is Y",
                metrics={"ndcg_at_5": 0.85},
            ),
        ]
        resp = BenchmarkEvalResponse(
            benchmark="ragbench",
            summary={"retrieval_ndcg_at_5": 0.85, "total_queries": 1},
            results=results,
            metadata={"total_evaluated": 1, "total_time_ms": 1500},
        )
        assert resp.benchmark == "ragbench"
        assert len(resp.results) == 1
        assert resp.results[0].metrics["ndcg_at_5"] == 0.85


class TestBenchmarkHistory:
    """Test benchmark history schemas."""

    def test_history_entry(self):
        entry = BenchmarkHistoryEntry(
            timestamp="2025-01-15T10:30:00",
            benchmark="ragbench",
            total_evaluated=50,
            accuracy=0.85,
            summary={"retrieval_ndcg_at_5": 0.85},
        )
        assert entry.benchmark == "ragbench"
        assert entry.accuracy == 0.85

    def test_history_response(self):
        resp = BenchmarkHistoryResponse(
            runs=[
                BenchmarkHistoryEntry(
                    timestamp="2025-01-15T10:30:00",
                    benchmark="ragbench",
                    total_evaluated=50,
                ),
            ]
        )
        assert len(resp.runs) == 1

    def test_empty_history(self):
        resp = BenchmarkHistoryResponse()
        assert resp.runs == []
