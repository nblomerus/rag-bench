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

    def _patch_eval_dirs(self, tmp_path):
        """Return context managers patching eval directory constants to tmp_path."""
        return (
            patch("rag_bench.api.server.EVAL_PRODUCTION_DIR", tmp_path / "production"),
            patch("rag_bench.api.server.EVAL_MANUAL_DIR", tmp_path / "manual"),
            patch("rag_bench.api.server.EVAL_RESULTS_DIR", tmp_path),
        )

    def test_ragbench_with_results(self, client, tmp_path):
        """Return latest eval file for ragbench."""
        manual_dir = tmp_path / "manual"
        manual_dir.mkdir()
        data = {"summary": {"total_queries": 10, "retrieval_mrr": 0.5}, "metadata": {}}
        (manual_dir / "eval_20260101_120000.json").write_text(json.dumps(data))

        p1, p2, p3 = self._patch_eval_dirs(tmp_path)
        with p1, p2, p3:
            resp = client.get("/api/eval/benchmark/latest/ragbench")
        assert resp.status_code == 200
        body = resp.json()
        assert body["summary"]["total_queries"] == 10
        assert "_source_file" in body

    def test_ragbench_returns_most_recent(self, client, tmp_path):
        """Most recent file (by mtime) wins regardless of directory."""
        prod_dir = tmp_path / "production"
        prod_dir.mkdir()
        manual_dir = tmp_path / "manual"
        manual_dir.mkdir()

        prod_data = {"summary": {"total_queries": 100}, "metadata": {"run_type": "production"}}
        manual_data = {"summary": {"total_queries": 5}, "metadata": {"run_type": "manual"}}
        prod_file = prod_dir / "eval_20260201_120000.json"
        prod_file.write_text(json.dumps(prod_data))
        # Give manual file a strictly newer mtime
        import os

        manual_file = manual_dir / "eval_20260202_120000.json"
        manual_file.write_text(json.dumps(manual_data))
        os.utime(prod_file, (1000000, 1000000))  # old
        os.utime(manual_file, (2000000, 2000000))  # newer

        p1, p2, p3 = self._patch_eval_dirs(tmp_path)
        with p1, p2, p3:
            resp = client.get("/api/eval/benchmark/latest/ragbench")
        assert resp.status_code == 200
        assert resp.json()["summary"]["total_queries"] == 5
        assert resp.json()["_run_type"] == "manual"

    def test_ragbench_production_wins_on_tie(self, client, tmp_path):
        """Production wins when mtime matches manual."""
        prod_dir = tmp_path / "production"
        prod_dir.mkdir()
        manual_dir = tmp_path / "manual"
        manual_dir.mkdir()

        prod_data = {"summary": {"total_queries": 100}, "metadata": {"run_type": "production"}}
        manual_data = {"summary": {"total_queries": 5}, "metadata": {"run_type": "manual"}}
        prod_file = prod_dir / "eval_20260201_120000.json"
        prod_file.write_text(json.dumps(prod_data))
        manual_file = manual_dir / "eval_20260201_120000.json"
        manual_file.write_text(json.dumps(manual_data))
        # Force identical mtime
        import os

        os.utime(prod_file, (1000000, 1000000))
        os.utime(manual_file, (1000000, 1000000))

        p1, p2, p3 = self._patch_eval_dirs(tmp_path)
        with p1, p2, p3:
            resp = client.get("/api/eval/benchmark/latest/ragbench")
        assert resp.status_code == 200
        assert resp.json()["summary"]["total_queries"] == 100
        assert resp.json()["_run_type"] == "production"

    def test_ragbench_fallback_to_manual(self, client, tmp_path):
        """Falls back to manual when no production files exist."""
        manual_dir = tmp_path / "manual"
        manual_dir.mkdir()
        data = {"summary": {"total_queries": 7}, "metadata": {"run_type": "manual"}}
        (manual_dir / "eval_20260101_120000.json").write_text(json.dumps(data))

        p1, p2, p3 = self._patch_eval_dirs(tmp_path)
        with p1, p2, p3:
            resp = client.get("/api/eval/benchmark/latest/ragbench")
        assert resp.status_code == 200
        assert resp.json()["_run_type"] == "manual"

    def test_ragbench_no_results(self, client, tmp_path):
        """404 when no eval files exist."""
        p1, p2, p3 = self._patch_eval_dirs(tmp_path)
        with p1, p2, p3:
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

        with patch("rag_bench.api.server.load_ragtruth", return_value=mock_entries):
            resp = client.get("/api/eval/ragtruth/examples?sample=10")

        assert resp.status_code == 200
        body = resp.json()
        assert "examples" in body
        assert len(body["examples"]) <= 10
        assert "total_with_hallucinations" in body
        assert "total_clean" in body

    def test_examples_loader_failure(self, client):
        """500 when RAGTruth dataset fails to load."""
        with patch("rag_bench.api.server.load_ragtruth", side_effect=RuntimeError("No cache")):
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


class TestBenchmarkTrends:
    """Test GET /api/eval/benchmark/trends."""

    def _patch_eval_dirs(self, tmp_path):
        """Return context managers patching eval directory constants to tmp_path."""
        return (
            patch("rag_bench.api.server.EVAL_PRODUCTION_DIR", tmp_path / "production"),
            patch("rag_bench.api.server.EVAL_MANUAL_DIR", tmp_path / "manual"),
            patch("rag_bench.api.server.EVAL_RESULTS_DIR", tmp_path),
        )

    def test_trends_with_production_results(self, client, tmp_path):
        """Return trend data points from production eval files."""
        prod_dir = tmp_path / "production"
        prod_dir.mkdir()

        data_1 = {
            "summary": {
                "total_queries": 50,
                "retrieval_mrr": 0.65,
                "retrieval_ndcg_at_5": 0.70,
                "retrieval_hit_rate": 0.80,
                "avg_citation_precision": 0.75,
                "avg_completeness": 0.60,
                "avg_faithfulness": 3.5,
                "avg_latency_ms": 1200.0,
            },
            "metadata": {"timestamp": "2026-02-20 12:00:00", "run_type": "production"},
        }
        data_2 = {
            "summary": {
                "total_queries": 77,
                "retrieval_mrr": 0.68,
                "retrieval_ndcg_at_5": 0.72,
                "retrieval_hit_rate": 0.85,
                "avg_citation_precision": 0.80,
                "avg_completeness": 0.65,
                "avg_faithfulness": 4.0,
                "avg_latency_ms": 1100.0,
            },
            "metadata": {"timestamp": "2026-02-23 12:42:43", "run_type": "production"},
        }

        (prod_dir / "eval_20260220_120000.json").write_text(json.dumps(data_1))
        (prod_dir / "eval_20260223_124243.json").write_text(json.dumps(data_2))

        p1, p2, p3 = self._patch_eval_dirs(tmp_path)
        with p1, p2, p3:
            resp = client.get("/api/eval/benchmark/trends?run_type=production")

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["trends"]) == 2
        assert body["trends"][0]["total_queries"] == 50
        assert body["trends"][1]["total_queries"] == 77
        assert body["trends"][1]["retrieval_mrr"] == 0.68
        assert body["trends"][1]["avg_faithfulness"] == 4.0
        assert body["trends"][0]["run_type"] == "production"

    def test_trends_manual_filter(self, client, tmp_path):
        """Return only manual trends when filtered."""
        prod_dir = tmp_path / "production"
        prod_dir.mkdir()
        manual_dir = tmp_path / "manual"
        manual_dir.mkdir()

        prod_data = {
            "summary": {"total_queries": 100},
            "metadata": {"timestamp": "2026-02-20", "run_type": "production"},
        }
        manual_data = {
            "summary": {"total_queries": 5},
            "metadata": {"timestamp": "2026-02-21", "run_type": "manual"},
        }
        (prod_dir / "eval_20260220_000000.json").write_text(json.dumps(prod_data))
        (manual_dir / "eval_20260221_000000.json").write_text(json.dumps(manual_data))

        p1, p2, p3 = self._patch_eval_dirs(tmp_path)
        with p1, p2, p3:
            resp = client.get("/api/eval/benchmark/trends?run_type=manual")

        assert resp.status_code == 200
        trends = resp.json()["trends"]
        assert len(trends) == 1
        assert trends[0]["total_queries"] == 5
        assert trends[0]["run_type"] == "manual"

    def test_trends_all_filter(self, client, tmp_path):
        """Return all trends when using 'all' filter."""
        prod_dir = tmp_path / "production"
        prod_dir.mkdir()
        manual_dir = tmp_path / "manual"
        manual_dir.mkdir()

        prod_data = {
            "summary": {"total_queries": 100},
            "metadata": {"timestamp": "2026-02-20", "run_type": "production"},
        }
        manual_data = {
            "summary": {"total_queries": 5},
            "metadata": {"timestamp": "2026-02-21", "run_type": "manual"},
        }
        (prod_dir / "eval_20260220_000000.json").write_text(json.dumps(prod_data))
        (manual_dir / "eval_20260221_000000.json").write_text(json.dumps(manual_data))

        p1, p2, p3 = self._patch_eval_dirs(tmp_path)
        with p1, p2, p3:
            resp = client.get("/api/eval/benchmark/trends?run_type=all")

        assert resp.status_code == 200
        trends = resp.json()["trends"]
        assert len(trends) == 2

    def test_trends_production_fallback(self, client, tmp_path):
        """Fall back to all when production is empty."""
        manual_dir = tmp_path / "manual"
        manual_dir.mkdir()
        data = {
            "summary": {"total_queries": 10},
            "metadata": {"timestamp": "2026-02-20", "run_type": "manual"},
        }
        (manual_dir / "eval_20260220_000000.json").write_text(json.dumps(data))

        p1, p2, p3 = self._patch_eval_dirs(tmp_path)
        with p1, p2, p3:
            resp = client.get("/api/eval/benchmark/trends")  # default is production

        assert resp.status_code == 200
        # Should fall back to manual files
        assert len(resp.json()["trends"]) == 1

    def test_trends_invalid_run_type(self, client, tmp_path):
        """400 for invalid run_type."""
        p1, p2, p3 = self._patch_eval_dirs(tmp_path)
        with p1, p2, p3:
            resp = client.get("/api/eval/benchmark/trends?run_type=invalid")
        assert resp.status_code == 400

    def test_trends_empty_dir(self, client, tmp_path):
        """Return empty trends when no eval files exist."""
        p1, p2, p3 = self._patch_eval_dirs(tmp_path)
        with p1, p2, p3:
            resp = client.get("/api/eval/benchmark/trends")
        assert resp.status_code == 200
        assert resp.json()["trends"] == []

    def test_trends_no_dir(self, client, tmp_path):
        """Return empty trends when eval dirs don't exist."""
        empty = tmp_path / "nonexistent"
        with (
            patch("rag_bench.api.server.EVAL_PRODUCTION_DIR", empty / "production"),
            patch("rag_bench.api.server.EVAL_MANUAL_DIR", empty / "manual"),
            patch("rag_bench.api.server.EVAL_RESULTS_DIR", empty),
        ):
            resp = client.get("/api/eval/benchmark/trends")
        assert resp.status_code == 200
        assert resp.json()["trends"] == []

    def test_trends_corrupt_file_skipped(self, client, tmp_path):
        """Corrupt eval files are skipped without crashing."""
        prod_dir = tmp_path / "production"
        prod_dir.mkdir()

        valid = {
            "summary": {"total_queries": 10, "retrieval_mrr": 0.5},
            "metadata": {"timestamp": "2026-02-20", "run_type": "production"},
        }
        (prod_dir / "eval_20260220_000000.json").write_text(json.dumps(valid))
        (prod_dir / "eval_20260221_000000.json").write_text("not valid json!!!")

        p1, p2, p3 = self._patch_eval_dirs(tmp_path)
        with p1, p2, p3:
            resp = client.get("/api/eval/benchmark/trends?run_type=production")

        assert resp.status_code == 200
        assert len(resp.json()["trends"]) == 1

    def test_trends_missing_summary_fields_default(self, client, tmp_path):
        """Missing summary fields default to 0."""
        manual_dir = tmp_path / "manual"
        manual_dir.mkdir()

        data = {"summary": {}, "metadata": {"timestamp": "2026-02-20"}}
        (manual_dir / "eval_20260220_000000.json").write_text(json.dumps(data))

        p1, p2, p3 = self._patch_eval_dirs(tmp_path)
        with p1, p2, p3:
            resp = client.get("/api/eval/benchmark/trends?run_type=manual")

        assert resp.status_code == 200
        point = resp.json()["trends"][0]
        assert point["retrieval_mrr"] == 0.0
        assert point["avg_faithfulness"] == 0.0
        assert point["total_queries"] == 0


class TestEvalSchedule:
    """Test GET/POST /api/eval/schedule."""

    def test_schedule_status_default(self, client):
        """Default schedule status when nothing has been configured."""
        with (
            patch("rag_bench.api.server._eval_schedule_enabled", False),
            patch("rag_bench.api.server._eval_schedule_interval", 24),
            patch("rag_bench.api.server._eval_schedule_last_run", None),
            patch("rag_bench.api.server._eval_schedule_last_summary", {}),
        ):
            resp = client.get("/api/eval/schedule")

        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is False
        assert body["interval_hours"] == 24
        assert body["next_run"] is None
        assert body["last_run"] is None

    def test_schedule_status_with_last_run(self, client):
        """Schedule status shows next_run computed from last_run."""
        with (
            patch("rag_bench.api.server._eval_schedule_enabled", True),
            patch("rag_bench.api.server._eval_schedule_interval", 12),
            patch("rag_bench.api.server._eval_schedule_last_run", "2026-02-23T12:00:00"),
            patch("rag_bench.api.server._eval_schedule_last_summary", {"total_queries": 20, "retrieval_mrr": 0.7}),
        ):
            resp = client.get("/api/eval/schedule")

        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is True
        assert body["interval_hours"] == 12
        assert body["next_run"] == "2026-02-24T00:00:00"
        assert body["last_run"] == "2026-02-23T12:00:00"
        assert body["last_run_summary"]["retrieval_mrr"] == 0.7

    def test_schedule_enable(self, client):
        """POST to enable schedule updates state."""
        with (
            patch("rag_bench.api.server._eval_schedule_enabled", False),
            patch("rag_bench.api.server._eval_schedule_interval", 24),
            patch("rag_bench.api.server._eval_schedule_last_run", None),
            patch("rag_bench.api.server._eval_schedule_last_summary", {}),
            patch("rag_bench.api.server._start_eval_schedule") as mock_start,
        ):
            resp = client.post(
                "/api/eval/schedule",
                json={"enabled": True, "interval_hours": 8},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is True
        assert body["interval_hours"] == 8
        mock_start.assert_called_once()

    def test_schedule_disable(self, client):
        """POST to disable schedule doesn't start task."""
        with (
            patch("rag_bench.api.server._eval_schedule_enabled", True),
            patch("rag_bench.api.server._eval_schedule_interval", 24),
            patch("rag_bench.api.server._eval_schedule_last_run", None),
            patch("rag_bench.api.server._eval_schedule_last_summary", {}),
            patch("rag_bench.api.server._start_eval_schedule") as mock_start,
        ):
            resp = client.post(
                "/api/eval/schedule",
                json={"enabled": False, "interval_hours": 24},
            )

        assert resp.status_code == 200
        assert resp.json()["enabled"] is False
        mock_start.assert_not_called()

    def test_schedule_validation_interval_too_high(self, client):
        """Interval > 168 hours is rejected by Pydantic."""
        with (
            patch("rag_bench.api.server._eval_schedule_enabled", False),
            patch("rag_bench.api.server._eval_schedule_interval", 24),
        ):
            resp = client.post(
                "/api/eval/schedule",
                json={"enabled": True, "interval_hours": 999},
            )

        assert resp.status_code == 422

    def test_schedule_validation_interval_too_low(self, client):
        """Interval < 1 hour is rejected by Pydantic."""
        with (
            patch("rag_bench.api.server._eval_schedule_enabled", False),
            patch("rag_bench.api.server._eval_schedule_interval", 24),
        ):
            resp = client.post(
                "/api/eval/schedule",
                json={"enabled": True, "interval_hours": 0},
            )

        assert resp.status_code == 422


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
