"""Tests for benchmark-related API endpoints (latest, examples, ragtruth)."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from rag_bench.api.server import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_pipeline():
    """Mock global pipeline so server is considered loaded."""
    retriever = Mock()
    generator = Mock()
    retriever.collection = Mock()
    retriever.collection.count.return_value = 100
    with (
        patch("rag_bench.api.server._retriever", retriever),
        patch("rag_bench.api.server._generator", generator),
        patch("rag_bench.api.server._llm_backend_name", "ollama"),
        patch("rag_bench.api.server._llm_model_name", "test:7b"),
    ):
        yield retriever, generator


class TestBenchmarkLatest:
    """Test GET /api/eval/benchmark/latest/{benchmark}."""

    def test_ragbench_with_results(self, client, tmp_path):
        """Return latest eval file for ragbench."""
        eval_dir = tmp_path / "eval_results"
        eval_dir.mkdir()
        data = {"summary": {"total_queries": 10, "retrieval_mrr": 0.5}}
        (eval_dir / "eval_20260101_120000.json").write_text(json.dumps(data))

        with patch("rag_bench.api.server.EVAL_RESULTS_DIR", eval_dir):
            resp = client.get("/api/eval/benchmark/latest/ragbench")
        assert resp.status_code == 200
        body = resp.json()
        assert body["summary"]["total_queries"] == 10
        assert "_source_file" in body

    def test_ragbench_no_results(self, client, tmp_path):
        """404 when no eval files exist."""
        eval_dir = tmp_path / "eval_results"
        eval_dir.mkdir()

        with patch("rag_bench.api.server.EVAL_RESULTS_DIR", eval_dir):
            resp = client.get("/api/eval/benchmark/latest/ragbench")
        assert resp.status_code == 404

    def test_ragtruth_with_history(self, client):
        """Return latest ragtruth entry from history."""
        history = [
            {"benchmark": "ragtruth", "summary": {"case_level_accuracy": 0.9}},
            {"benchmark": "garage", "summary": {}},
        ]
        with patch("rag_bench.api.server._benchmark_history", history):
            resp = client.get("/api/eval/benchmark/latest/ragtruth")
        assert resp.status_code == 200
        assert resp.json()["summary"]["case_level_accuracy"] == 0.9

    def test_ragtruth_no_history(self, client):
        """404 when no ragtruth entries exist."""
        with patch("rag_bench.api.server._benchmark_history", []):
            resp = client.get("/api/eval/benchmark/latest/ragtruth")
        assert resp.status_code == 404

    def test_ragtruth_empty_summary_skipped(self, client):
        """Skip ragtruth entries with empty summary."""
        history = [
            {"benchmark": "ragtruth", "summary": {}},  # empty — should skip
        ]
        with patch("rag_bench.api.server._benchmark_history", history):
            resp = client.get("/api/eval/benchmark/latest/ragtruth")
        assert resp.status_code == 404

    def test_invalid_benchmark(self, client):
        """400 for unknown benchmark name."""
        resp = client.get("/api/eval/benchmark/latest/unknown")
        assert resp.status_code == 400


class TestBenchmarkExamples:
    """Test GET /api/eval/benchmark/examples."""

    def test_returns_examples(self, client):
        resp = client.get("/api/eval/benchmark/examples")
        assert resp.status_code == 200
        body = resp.json()
        assert "examples" in body
        assert len(body["examples"]) > 0
        ex = body["examples"][0]
        assert "question" in ex
        assert "expected_sources" in ex
        assert "topic" in ex
        assert "difficulty" in ex


class TestRagtruthDetect:
    """Test POST /api/eval/ragtruth/detect."""

    def test_detect_faithful(self, client):
        """Faithful answer returns no hallucination."""
        resp = client.post(
            "/api/eval/ragtruth/detect",
            json={
                "context": "Machine learning uses algorithms to learn from data and make predictions.",
                "response": "Machine learning uses algorithms to learn from data.",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["has_hallucination"] is False
        assert "latency_ms" in body

    def test_detect_hallucinated(self, client):
        """Unsupported answer is flagged."""
        resp = client.post(
            "/api/eval/ragtruth/detect",
            json={
                "context": "The cat sat on the mat.",
                "response": "Quantum field theory demonstrates the unification of electromagnetic"
                " and weak nuclear forces.",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["has_hallucination"] is True
        assert len(body["flagged_spans"]) > 0

    def test_detect_missing_context(self, client):
        """400 when context is missing."""
        resp = client.post(
            "/api/eval/ragtruth/detect",
            json={
                "context": "",
                "response": "Some answer.",
            },
        )
        assert resp.status_code == 400

    def test_detect_missing_response(self, client):
        """400 when response is missing."""
        resp = client.post(
            "/api/eval/ragtruth/detect",
            json={
                "context": "Some context.",
                "response": "",
            },
        )
        assert resp.status_code == 400


class TestRagtruthExamples:
    """Test GET /api/eval/ragtruth/examples."""

    def test_returns_examples(self, client):
        """Load examples from the cached RAGTruth dataset."""
        from rag_bench.eval.ragtruth.loader import RAGTruthEntry

        mock_entries = [
            RAGTruthEntry(
                id=str(i),
                source_id=f"src_{i}",
                task_type="QA",
                source_info=f"Context for question {i}.",
                prompt=f"Question {i}?",
                reference_response=f"Answer {i}.",
                has_hallucination=i % 3 == 0,
                metadata={"labels": [{"text": "bad", "label_type": "Baseless"}] if i % 3 == 0 else []},
            )
            for i in range(30)
        ]

        with patch("rag_bench.eval.ragtruth.loader.load_ragtruth", return_value=mock_entries):
            resp = client.get("/api/eval/ragtruth/examples?sample=10")

        assert resp.status_code == 200
        body = resp.json()
        assert "examples" in body
        assert len(body["examples"]) <= 10
        assert "total_with_hallucinations" in body
        assert "total_clean" in body

    def test_examples_loader_failure(self, client):
        """500 when RAGTruth dataset fails to load."""
        with patch("rag_bench.eval.ragtruth.loader.load_ragtruth", side_effect=RuntimeError("No cache")):
            resp = client.get("/api/eval/ragtruth/examples")
        assert resp.status_code == 500


class TestBenchmarkHistory:
    """Test GET /api/eval/benchmark/history."""

    def test_history_returns_entries(self, client):
        history = [
            {"benchmark": "ragbench", "timestamp": "2026-01-01", "total_evaluated": 50, "summary": {}},
        ]
        with patch("rag_bench.api.server._benchmark_history", history):
            resp = client.get("/api/eval/benchmark/history")
        assert resp.status_code == 200

    def test_history_empty(self, client):
        with patch("rag_bench.api.server._benchmark_history", []):
            resp = client.get("/api/eval/benchmark/history")
        assert resp.status_code == 200
        assert resp.json()["runs"] == []


class TestBenchmarkStatus:
    """Test GET /api/eval/benchmark/status."""

    def test_not_running(self, client):
        with patch("rag_bench.api.server._benchmark_running", False):
            resp = client.get("/api/eval/benchmark/status")
        assert resp.status_code == 200
        assert resp.json()["running"] is False

    def test_running(self, client):
        with patch("rag_bench.api.server._benchmark_running", True):
            resp = client.get("/api/eval/benchmark/status")
        assert resp.status_code == 200
        assert resp.json()["running"] is True


class TestRunBenchmarkValidation:
    """Test POST /api/eval/benchmark input validation."""

    def test_already_running(self, client, mock_pipeline):
        with patch("rag_bench.api.server._benchmark_running", True):
            resp = client.post(
                "/api/eval/benchmark",
                json={
                    "benchmark": "ragbench",
                    "sample_size": 10,
                },
            )
        assert resp.status_code == 409

    def test_pipeline_not_ready(self, client):
        with (
            patch("rag_bench.api.server._generator", None),
            patch("rag_bench.api.server._benchmark_running", False),
        ):
            resp = client.post(
                "/api/eval/benchmark",
                json={
                    "benchmark": "ragbench",
                    "sample_size": 10,
                },
            )
        assert resp.status_code == 503

    def test_invalid_benchmark_name(self, client, mock_pipeline):
        with patch("rag_bench.api.server._benchmark_running", False):
            resp = client.post(
                "/api/eval/benchmark",
                json={
                    "benchmark": "invalid",
                    "sample_size": 10,
                },
            )
        assert resp.status_code == 400


class TestServiceWorkerAndFrontend:
    """Test frontend serving endpoints."""

    def test_sw_not_found(self, client, tmp_path):
        dist = tmp_path / "dist"
        dist.mkdir()
        with patch("rag_bench.api.server.FRONTEND_DIST", dist):
            resp = client.get("/sw.js")
        assert resp.status_code == 404

    def test_frontend_not_built(self, client, tmp_path):
        html = tmp_path / "index.html"
        with patch("rag_bench.api.server.FRONTEND_HTML", html):
            resp = client.get("/")
        assert resp.status_code == 404


class TestSaveBenchmarkHistory:
    """Test _save_benchmark_history."""

    def test_save_success(self, tmp_path):
        from rag_bench.api.server import _save_benchmark_history

        history_file = tmp_path / "history.json"
        with (
            patch("rag_bench.api.server.BENCHMARK_HISTORY_FILE", history_file),
            patch("rag_bench.api.server._benchmark_history", [{"benchmark": "ragbench"}]),
        ):
            _save_benchmark_history()
        assert history_file.exists()
        data = json.loads(history_file.read_text())
        assert len(data) == 1

    def test_save_failure(self, tmp_path):
        """Gracefully handles write failure."""
        from rag_bench.api.server import _save_benchmark_history

        bad_path = tmp_path / "no" / "such" / "deep" / "dir" / "history.json"
        with (
            patch("rag_bench.api.server.BENCHMARK_HISTORY_FILE", bad_path),
            patch("rag_bench.api.server._benchmark_history", []),
        ):
            # Should not raise
            _save_benchmark_history()


class TestPaperCountCache:
    """Test paper count cache helpers."""

    def test_load_cached_count_exists(self, tmp_path):
        from rag_bench.api.server import _load_cached_paper_count

        cache_file = tmp_path / ".paper_count"
        cache_file.write_text("42")
        with patch("rag_bench.api.server._PAPER_COUNT_CACHE", cache_file):
            result = _load_cached_paper_count()
        assert result == 42

    def test_load_cached_count_missing(self, tmp_path):
        from rag_bench.api.server import _load_cached_paper_count

        cache_file = tmp_path / ".paper_count"
        with patch("rag_bench.api.server._PAPER_COUNT_CACHE", cache_file):
            result = _load_cached_paper_count()
        assert result is None

    def test_save_cached_count(self, tmp_path):
        from rag_bench.api.server import _save_cached_paper_count

        cache_file = tmp_path / ".paper_count"
        with patch("rag_bench.api.server._PAPER_COUNT_CACHE", cache_file):
            _save_cached_paper_count(123)
        assert cache_file.read_text() == "123"

    def test_save_cached_count_failure(self):
        from rag_bench.api.server import _save_cached_paper_count

        bad_path = Path("/nonexistent/dir/.paper_count")
        with patch("rag_bench.api.server._PAPER_COUNT_CACHE", bad_path):
            # Should not raise
            _save_cached_paper_count(99)
