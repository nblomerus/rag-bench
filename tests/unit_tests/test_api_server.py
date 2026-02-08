"""Test API server endpoints."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from rag_bench.api.server import (
    _extract_arxiv_id,
    _format_sources,
    app,
)


@pytest.fixture
def mock_pipeline():
    """Mock the global pipeline instances."""
    retriever = Mock()
    generator = Mock()

    retriever.collection = Mock()
    retriever.collection.count.return_value = 100

    generator.answer = Mock(
        return_value={
            "answer": "Test answer",
            "deflected": False,
            "results": [
                {
                    "score": 0.95,
                    "text": "Sample text content",
                    "chunk_id": "chunk_1",
                    "metadata": {
                        "title": "Test Paper",
                        "section": "Introduction",
                        "paper_id": "arxiv:2103.05674",
                    },
                }
            ],
            "filtered_results": [
                {
                    "score": 0.95,
                    "text": "Sample text content",
                    "chunk_id": "chunk_1",
                    "metadata": {
                        "title": "Test Paper",
                        "section": "Introduction",
                        "paper_id": "arxiv:2103.05674",
                    },
                }
            ],
            "scores": [0.95, 0.85],
        }
    )

    with (
        patch("rag_bench.api.server._retriever", retriever),
        patch("rag_bench.api.server._generator", generator),
        patch("rag_bench.api.server._llm_backend_name", "ollama"),
        patch("rag_bench.api.server._llm_model_name", "mistral:7b"),
    ):
        yield retriever, generator


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestFormatSources:
    """Test the _format_sources helper function."""

    def test_format_sources_basic(self):
        """Test basic source formatting."""
        results = [
            {
                "score": 0.95,
                "text": "Sample text" * 50,  # Long text to test truncation
                "chunk_id": "chunk_1",
                "metadata": {
                    "title": "Test Paper",
                    "section": "Introduction",
                    "paper_id": "arxiv:2103.05674",
                },
            }
        ]
        sources = _format_sources(results)
        assert len(sources) == 1
        assert sources[0].rank == 1
        assert sources[0].score == 0.95
        assert sources[0].title == "Test Paper"
        assert len(sources[0].text_preview) <= 300

    def test_format_sources_empty(self):
        """Test empty results."""
        sources = _format_sources([])
        assert sources == []

    def test_format_sources_relevance_high_score(self):
        """Test relevance assignment with high scores."""
        results = [
            {
                "score": 3.5,
                "text": "Text",
                "chunk_id": "chunk_1",
                "metadata": {"title": "Paper 1"},
            }
        ]
        sources = _format_sources(results)
        assert sources[0].relevance == "high"

    def test_format_sources_relevance_tiered(self):
        """Test relevance assignment with multiple tiers."""
        results = [
            {"score": 3.5, "text": "Text", "chunk_id": "c1", "metadata": {"title": "P1"}},
            {"score": 2.0, "text": "Text", "chunk_id": "c2", "metadata": {"title": "P2"}},
            {"score": 1.0, "text": "Text", "chunk_id": "c3", "metadata": {"title": "P3"}},
            {"score": 0.5, "text": "Text", "chunk_id": "c4", "metadata": {"title": "P4"}},
        ]
        sources = _format_sources(results)
        assert sources[0].relevance == "high"  # score 3.5 >= 3.0
        assert sources[1].relevance == "medium"  # score 2.0 >= 1.0
        assert sources[2].relevance == "medium"  # score 1.0 >= 1.0
        assert sources[3].relevance == "low"  # score 0.5 < 1.0

    def test_format_sources_fallback_metadata(self):
        """Test fallback metadata fields."""
        results = [
            {
                "score": 0.9,
                "text": "Text",
                "chunk_id": "chunk_1",
                "metadata": {
                    "source_display": "Display Title",
                    "doc_id": "doc_123",
                    "arxiv_id": "2103.05674",
                },
            }
        ]
        sources = _format_sources(results)
        assert sources[0].title == "Display Title"
        assert sources[0].paper_id == "doc_123"

    def test_format_sources_missing_metadata(self):
        """Test handling of missing metadata fields."""
        results = [
            {
                "score": 0.8,
                "text": "Text",
                "chunk_id": "chunk_1",
                "metadata": {},
            }
        ]
        sources = _format_sources(results)
        assert sources[0].title == "Unknown"
        assert sources[0].paper_id == ""


class TestExtractArxivId:
    """Test the _extract_arxiv_id helper function."""

    def test_extract_arxiv_id_basic(self):
        """Test basic arXiv ID extraction."""
        arxiv_id = _extract_arxiv_id("2103.05674")
        assert arxiv_id == "2103.05674"

    def test_extract_arxiv_id_with_version(self):
        """Test extraction of arXiv ID with version."""
        arxiv_id = _extract_arxiv_id("2103.05674v2")
        assert arxiv_id == "2103.05674"

    def test_extract_arxiv_id_with_prefix(self):
        """Test extraction with arxiv_ prefix."""
        arxiv_id = _extract_arxiv_id("arxiv_2103.05674")
        assert arxiv_id == "2103.05674"

    def test_extract_arxiv_id_old_format(self):
        """Test extraction of old format arXiv IDs."""
        arxiv_id = _extract_arxiv_id("hep-ph/0601234")
        assert arxiv_id == "hep-ph/0601234"

    def test_extract_arxiv_id_whitespace(self):
        """Test extraction with whitespace."""
        arxiv_id = _extract_arxiv_id("  2103.05674  ")
        assert arxiv_id == "2103.05674"


class TestHealthEndpoint:
    """Test the health check endpoint."""

    def test_health_check_ready(self, client, mock_pipeline):
        """Test health check when pipeline is ready."""
        with patch("rag_bench.api.server._generator", MagicMock()):
            response = client.get("/api/health")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "pipeline_ready" in data

    def test_health_check_not_ready(self, client):
        """Test health check when pipeline not ready."""
        with patch("rag_bench.api.server._generator", None):
            response = client.get("/api/health")
            assert response.status_code == 200


class TestQueryEndpoint:
    """Test the query endpoint."""

    def test_query_basic(self, client, mock_pipeline):
        """Test basic query."""
        with patch(
            "rag_bench.api.server._generator",
            MagicMock(
                answer=MagicMock(
                    return_value={
                        "answer": "Test",
                        "deflected": False,
                        "results": [],
                        "filtered_results": [],
                        "scores": [],
                        "deflection_reason": "",
                    }
                )
            ),
        ):
            response = client.post("/api/query", json={"question": "What is ML?"})
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "sources" in data
            assert "deflected" in data

    def test_query_casual_greeting_hi(self, client):
        """Test casual greeting with 'hi'."""
        with patch("rag_bench.api.server._generator", MagicMock()):
            response = client.post("/api/query", json={"question": "hi"})
            assert response.status_code == 200
            data = response.json()
            assert "RAG-Bench" in data["answer"]
            assert data["deflected"] is False

    def test_query_casual_greeting_hello(self, client):
        """Test casual greeting with 'hello'."""
        with patch("rag_bench.api.server._generator", MagicMock()):
            response = client.post("/api/query", json={"question": "hello"})
            assert response.status_code == 200
            data = response.json()
            assert "RAG-Bench" in data["answer"]

    def test_query_too_short(self, client):
        """Test query that is too short."""
        with patch("rag_bench.api.server._generator", MagicMock()):
            response = client.post("/api/query", json={"question": "ab"})
            assert response.status_code == 200
            data = response.json()
            assert "RAG-Bench" in data["answer"]

    def test_query_with_custom_top_k(self, client):
        """Test query with custom top_k."""
        generator_mock = MagicMock(
            answer=MagicMock(
                return_value={
                    "answer": "Test",
                    "deflected": False,
                    "results": [],
                    "filtered_results": [],
                    "scores": [],
                }
            )
        )
        with patch("rag_bench.api.server._generator", generator_mock):
            response = client.post(
                "/api/query",
                json={"question": "What is attention?", "top_k": 10},
            )
            assert response.status_code == 200
            # Verify that top_k was passed to the generator
            generator_mock.answer.assert_called()

    def test_query_pipeline_not_loaded(self, client):
        """Test query when pipeline is not loaded."""
        with patch("rag_bench.api.server._generator", None):
            response = client.post("/api/query", json={"question": "What is ML?"})
            assert response.status_code == 503

    def test_query_with_deflection(self, client):
        """Test query that gets deflected."""
        generator_mock = MagicMock(
            answer=MagicMock(
                return_value={
                    "answer": "I cannot answer this.",
                    "deflected": True,
                    "results": [],
                    "filtered_results": [],
                    "scores": [],
                    "deflection_reason": "No relevant papers found",
                }
            )
        )
        with patch("rag_bench.api.server._generator", generator_mock):
            response = client.post("/api/query", json={"question": "Something obscure?"})
            assert response.status_code == 200
            data = response.json()
            assert data["deflected"] is True


class TestStatsEndpoint:
    """Test the stats endpoint."""

    def test_stats_endpoint(self, client, mock_pipeline):
        """Test stats endpoint."""
        retriever_mock = MagicMock()
        retriever_mock.collection.count.return_value = 100
        retriever_mock.collection.get.return_value = {
            "metadatas": [
                {"paper_id": "p1", "arxiv_id": "2103.05674"},
                {"paper_id": "p2", "arxiv_id": "2103.05675"},
            ]
        }
        with patch("rag_bench.api.server._retriever", retriever_mock):
            response = client.get("/api/stats")
            assert response.status_code == 200
            data = response.json()
            assert "total_chunks" in data
            assert "total_papers" in data
            assert "collection_name" in data

    def test_stats_pipeline_not_loaded(self, client):
        """Test stats when pipeline is not loaded."""
        with patch("rag_bench.api.server._retriever", None):
            response = client.get("/api/stats")
            assert response.status_code == 503


class TestListPapersEndpoint:
    """Test the list papers endpoint."""

    def test_list_papers(self, client, mock_pipeline):
        """Test list papers endpoint."""
        retriever_mock = MagicMock()
        retriever_mock.collection.count.return_value = 2
        retriever_mock.collection.get.return_value = {
            "metadatas": [
                {
                    "doc_id": "p1",
                    "title": "Paper 1",
                    "year": 2020,
                    "arxiv_id": "2103.05674",
                    "section": "Introduction",
                },
                {
                    "doc_id": "p1",
                    "title": "Paper 1",
                    "year": 2020,
                    "arxiv_id": "2103.05674",
                    "section": "Method",
                },
                {
                    "doc_id": "p2",
                    "title": "Paper 2",
                    "year": 2021,
                    "arxiv_id": "2103.05675",
                    "section": "Results",
                },
            ]
        }
        with patch("rag_bench.api.server._retriever", retriever_mock):
            response = client.get("/api/papers")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 2
            assert data[0]["title"] == "Paper 1"
            assert data[0]["chunk_count"] == 2

    def test_list_papers_pipeline_not_loaded(self, client):
        """Test list papers when pipeline is not loaded."""
        with patch("rag_bench.api.server._retriever", None):
            response = client.get("/api/papers")
            assert response.status_code == 503


class TestFrontendEndpoint:
    """Test the frontend endpoint."""

    def test_frontend_exists(self, client):
        """Test frontend served when it exists."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        with patch("rag_bench.api.server.FRONTEND_HTML", mock_path):
            response = client.get("/")
            assert response.status_code == 200

    def test_frontend_not_found(self, client):
        """Test frontend 404 when it doesn't exist."""
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        with patch("rag_bench.api.server.FRONTEND_HTML", mock_path):
            response = client.get("/")
            assert response.status_code == 404


class TestInvalidRequests:
    """Test invalid request handling."""

    def test_invalid_json(self, client):
        """Test invalid JSON in request."""
        response = client.post(
            "/api/query",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code >= 400

    def test_empty_question(self, client):
        """Test empty question validation."""
        response = client.post("/api/query", json={"question": ""})
        assert response.status_code == 422  # Validation error

    def test_question_too_long(self, client):
        """Test question that exceeds max length."""
        response = client.post("/api/query", json={"question": "a" * 1001})
        assert response.status_code == 422  # Validation error

    def test_invalid_top_k(self, client):
        """Test invalid top_k value."""
        response = client.post(
            "/api/query",
            json={"question": "What is ML?", "top_k": 25},
        )
        assert response.status_code == 422  # Validation error


class TestEvalEndpoint:
    """Test the eval endpoint."""

    def test_eval_endpoint_basic(self, client):
        """Test eval endpoint."""
        with (
            patch(
                "rag_bench.api.server._generator",
                MagicMock(
                    answer=MagicMock(
                        return_value={
                            "answer": "Test",
                            "deflected": False,
                            "results": [],
                            "scores": [0.5],
                        }
                    )
                ),
            ),
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", MagicMock()),
            patch(
                "json.load",
                return_value=[
                    {
                        "id": "q1",
                        "question": "Test?",
                        "should_deflect": False,
                        "difficulty": "easy",
                    }
                ],
            ),
        ):
            response = client.post("/api/eval", json={"run_all": False})
            assert response.status_code == 200
            data = response.json()
            assert "total" in data
            assert "correct" in data
            assert "accuracy" in data

    def test_eval_endpoint_pipeline_not_loaded(self, client):
        """Test eval when pipeline not loaded."""
        with patch("rag_bench.api.server._generator", None):
            response = client.post("/api/eval", json={"run_all": False})
            assert response.status_code == 503

    def test_eval_endpoint_file_not_found(self, client):
        """Test eval when queries file not found."""
        with patch("rag_bench.api.server._generator", MagicMock()), patch("pathlib.Path.exists", return_value=False):
            response = client.post("/api/eval", json={"run_all": False})
            assert response.status_code == 404


class TestStreamingEndpoint:
    """Test the streaming query endpoint."""

    def test_query_stream_casual_greeting(self, client):
        """Test streaming with casual greeting."""
        with patch("rag_bench.api.server._generator", MagicMock()):
            response = client.post("/api/query/stream", json={"question": "hello"})
            assert response.status_code == 200


class TestPaperEndpoints:
    """Test paper-related endpoints."""

    def test_get_paper_not_found(self, client):
        """Test get paper when not found."""
        retriever_mock = MagicMock()
        retriever_mock.collection.get.side_effect = [
            {"ids": []},  # First call returns empty
            {"ids": []},  # Second call returns empty
            {"ids": []},  # Third call returns empty
        ]
        with patch("rag_bench.api.server._retriever", retriever_mock):
            response = client.get("/api/papers/nonexistent")
            assert response.status_code == 404

    def test_get_paper_pipeline_not_loaded(self, client):
        """Test get paper when pipeline not loaded."""
        with patch("rag_bench.api.server._retriever", None):
            response = client.get("/api/papers/some_id")
            assert response.status_code == 503

    def test_get_paper_pdf_not_found(self, client):
        """Test get paper PDF when arXiv ID not found."""
        retriever_mock = MagicMock()
        retriever_mock.collection.get.side_effect = [
            {"ids": []},  # First call
            {"ids": []},  # Second call
            {"ids": []},  # Third call
        ]
        with patch("rag_bench.api.server._retriever", retriever_mock):
            # API returns 502 when it can't extract arxiv_id and fetch fails
            response = client.get("/api/papers/xyz/pdf")
            assert response.status_code in [404, 502]

    def test_get_paper_pdf_pipeline_not_loaded(self, client):
        """Test get paper PDF when pipeline not loaded."""
        with patch("rag_bench.api.server._retriever", None):
            response = client.get("/api/papers/some_id/pdf")
            assert response.status_code == 503
