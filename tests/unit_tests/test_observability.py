"""Tests for the observability package: logging, metrics, and request tracking."""

import os
import time

from rag_bench.observability.logging import get_logger, setup_logging
from rag_bench.observability.metrics import (
    ACTIVE_REQUESTS,
    BUILD_INFO,
    CITATION_COVERAGE,
    CORPUS_CHUNKS,
    CORPUS_PAPERS,
    GENERATION_DURATION,
    PIPELINE_READY,
    QUERIES_TOTAL,
    REQUEST_DURATION,
    RERANKING_DURATION,
    RETRIEVAL_DURATION,
    RETRIEVAL_TOP_SCORE,
    UNIQUE_USERS,
)
from rag_bench.observability.tracker import RequestTracker, _get_gpu_stats, _percentile

# ── Logging ──


class TestLogging:
    def test_setup_logging_console(self):
        """setup_logging configures structlog without errors."""
        setup_logging(log_level="DEBUG", json_logs=False)
        logger = get_logger("test")
        assert logger is not None

    def test_setup_logging_json(self):
        """setup_logging can configure JSON mode."""
        setup_logging(log_level="INFO", json_logs=True)
        logger = get_logger("test.json")
        assert logger is not None

    def test_setup_logging_auto_detect_json(self):
        """setup_logging auto-detects JSON mode from RAG_ENV."""
        old = os.environ.get("RAG_ENV")
        try:
            os.environ["RAG_ENV"] = "production"
            setup_logging(log_level="INFO")
            logger = get_logger("test.auto")
            assert logger is not None
        finally:
            if old is None:
                os.environ.pop("RAG_ENV", None)
            else:
                os.environ["RAG_ENV"] = old

    def test_get_logger_returns_structlog_logger(self):
        """get_logger returns a structlog logger proxy."""
        setup_logging(log_level="INFO", json_logs=False)
        logger = get_logger("test.bound")
        # structlog returns a lazy proxy that becomes a BoundLogger on first use
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")


# ── Metrics ──


class TestMetrics:
    def test_queries_total_counter(self):
        """QUERIES_TOTAL counter can be incremented."""
        before = QUERIES_TOTAL.labels(status="success")._value.get()
        QUERIES_TOTAL.labels(status="success").inc()
        after = QUERIES_TOTAL.labels(status="success")._value.get()
        assert after == before + 1

    def test_request_duration_histogram(self):
        """REQUEST_DURATION histogram can observe values."""
        REQUEST_DURATION.labels(endpoint="/api/query", method="POST").observe(1.5)

    def test_retrieval_duration_histogram(self):
        """RETRIEVAL_DURATION histogram can observe values."""
        RETRIEVAL_DURATION.observe(0.25)

    def test_generation_duration_histogram(self):
        """GENERATION_DURATION histogram can observe values."""
        GENERATION_DURATION.observe(3.5)

    def test_reranking_duration_histogram(self):
        """RERANKING_DURATION histogram can observe values."""
        RERANKING_DURATION.observe(0.1)

    def test_retrieval_top_score_histogram(self):
        """RETRIEVAL_TOP_SCORE histogram can observe values."""
        RETRIEVAL_TOP_SCORE.observe(2.5)

    def test_citation_coverage_histogram(self):
        """CITATION_COVERAGE histogram can observe values."""
        CITATION_COVERAGE.observe(0.75)

    def test_gauges(self):
        """All gauges can be set."""
        ACTIVE_REQUESTS.set(3)
        PIPELINE_READY.set(1)
        CORPUS_CHUNKS.set(100000)
        CORPUS_PAPERS.set(1500)
        UNIQUE_USERS.set(42)

    def test_build_info(self):
        """BUILD_INFO can be set with labels."""
        BUILD_INFO.info(
            {
                "version": "0.0.14",
                "llm_backend": "ollama",
                "llm_model": "test",
                "embedding_model": "test",
            }
        )


# ── Tracker ──


class TestRequestTracker:
    def test_empty_tracker_summary(self):
        """New tracker returns zero-valued summary."""
        tracker = RequestTracker()
        summary = tracker.summary()
        assert summary["total_queries"] == 0
        assert summary["unique_users"] == 0
        assert summary["latency"]["avg_ms"] == 0.0
        assert summary["latency"]["p50_ms"] == 0.0
        assert summary["recent_queries"] == []

    def test_record_query_updates_counts(self):
        """Recording a query increments counters."""
        tracker = RequestTracker()
        tracker.record_query(
            latency_ms=1500.0,
            status="success",
            question_preview="What is attention?",
            client_ip="127.0.0.1",
        )
        assert tracker.total_queries == 1
        assert tracker.status_counts["success"] == 1
        assert len(tracker.unique_ips) == 1

    def test_record_all_pipeline_timings(self):
        """Recording with all pipeline timings populates all deques."""
        tracker = RequestTracker()
        tracker.record_query(
            latency_ms=2000.0,
            status="success",
            question_preview="Full pipeline query",
            retrieval_ms=300.0,
            generation_ms=1500.0,
            reranking_ms=150.0,
            citation_coverage=0.85,
            top_score=3.2,
        )
        summary = tracker.summary()
        assert summary["pipeline"]["avg_retrieval_ms"] == 300.0
        assert summary["pipeline"]["avg_generation_ms"] == 1500.0
        assert summary["pipeline"]["avg_reranking_ms"] == 150.0
        assert summary["quality"]["avg_citation_coverage"] == 0.8
        assert summary["quality"]["avg_top_score"] == 3.2

    def test_record_multiple_queries(self):
        """Recording multiple queries computes correct stats."""
        tracker = RequestTracker()
        for i in range(10):
            tracker.record_query(
                latency_ms=1000.0 + i * 100,
                status="success" if i < 8 else "deflected",
                question_preview=f"Query {i}",
                retrieval_ms=200.0 + i * 10,
                generation_ms=500.0 + i * 20,
                citation_coverage=0.5 + i * 0.05,
                client_ip=f"192.168.1.{i}",
            )
        summary = tracker.summary()
        assert summary["total_queries"] == 10
        assert summary["unique_users"] == 10
        assert summary["queries_by_status"]["success"] == 8
        assert summary["queries_by_status"]["deflected"] == 2
        assert summary["latency"]["avg_ms"] > 0
        assert summary["latency"]["p50_ms"] > 0
        assert summary["latency"]["p90_ms"] >= summary["latency"]["p50_ms"]
        assert len(summary["recent_queries"]) == 10

    def test_recent_queries_ordered_newest_first(self):
        """Recent queries are returned newest-first."""
        tracker = RequestTracker()
        tracker.record_query(latency_ms=100, status="success", question_preview="First")
        tracker.record_query(latency_ms=200, status="success", question_preview="Second")
        summary = tracker.summary()
        assert summary["recent_queries"][0]["question"] == "Second"
        assert summary["recent_queries"][1]["question"] == "First"

    def test_question_preview_truncated(self):
        """Long question previews are truncated to 80 chars."""
        tracker = RequestTracker()
        long_q = "x" * 200
        tracker.record_query(latency_ms=100, status="success", question_preview=long_q)
        summary = tracker.summary()
        assert len(summary["recent_queries"][0]["question"]) == 80

    def test_hardware_stats_present(self):
        """Hardware stats include CPU and RAM fields."""
        tracker = RequestTracker()
        summary = tracker.summary()
        hw = summary["hardware"]
        assert "cpu_percent" in hw
        assert "ram_used_gb" in hw
        assert "ram_total_gb" in hw
        assert "ram_percent" in hw
        assert "gpus" in hw
        assert isinstance(hw["gpus"], list)

    def test_latency_history_in_summary(self):
        """Summary includes latency_history list."""
        tracker = RequestTracker()
        summary = tracker.summary()
        assert "latency_history" in summary
        assert isinstance(summary["latency_history"], list)

    def test_latency_history_per_query(self):
        """Each query appends a rolling percentile snapshot with pipeline timings."""
        tracker = RequestTracker()
        tracker.record_query(
            latency_ms=1000,
            status="success",
            question_preview="Q1",
            retrieval_ms=200,
            generation_ms=700,
            reranking_ms=50,
        )
        tracker.record_query(
            latency_ms=2000,
            status="success",
            question_preview="Q2",
            retrieval_ms=300,
            generation_ms=1500,
            reranking_ms=100,
        )
        tracker.record_query(latency_ms=3000, status="success", question_preview="Q3")
        assert len(tracker._latency_history) == 3
        point = tracker._latency_history[-1]
        assert "t" in point
        assert "p50" in point
        assert "p90" in point
        assert "p99" in point
        assert "n" in point
        assert point["n"] == 3
        # Pipeline timings included per-query
        assert "retrieval_ms" in point
        assert "generation_ms" in point
        assert "reranking_ms" in point
        # First point has pipeline timings
        p0 = tracker._latency_history[0]
        assert p0["retrieval_ms"] == 200.0
        assert p0["generation_ms"] == 700.0

    def test_uptime_increases(self):
        """Uptime reflects actual elapsed time."""
        tracker = RequestTracker()
        s1 = tracker.summary()["uptime_seconds"]
        time.sleep(0.1)
        s2 = tracker.summary()["uptime_seconds"]
        assert s2 >= s1


class TestPercentile:
    def test_empty_data(self):
        assert _percentile([], 0.5) == 0.0

    def test_single_value(self):
        assert _percentile([42.0], 0.5) == 42.0

    def test_p50(self):
        data = sorted([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _percentile(data, 0.5) == 3.0

    def test_p99_returns_near_max(self):
        data = sorted(float(i) for i in range(100))
        result = _percentile(data, 0.99)
        assert result >= 98.0


class TestGetGpuStats:
    def test_returns_list(self):
        """_get_gpu_stats returns a list (may be empty if no GPU)."""
        result = _get_gpu_stats()
        assert isinstance(result, list)
