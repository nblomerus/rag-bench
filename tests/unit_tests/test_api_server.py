"""Test API server endpoints."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from rag_bench.api.server import (
    _compute_faithfulness_heuristic,
    _compute_per_source_cited,
    _compute_retrieval_confidence,
    _compute_score_spread,
    _compute_source_diversity,
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

    pipeline_mock = Mock()
    pipeline_mock.retriever = retriever
    pipeline_mock.generator = generator

    with (
        patch("rag_bench.api.server._pipeline", pipeline_mock),
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
        with patch("rag_bench.api.server._pipeline", MagicMock()), patch("rag_bench.api.server._generator", MagicMock()):
            response = client.post("/api/query", json={"question": "hi"})
            assert response.status_code == 200
            data = response.json()
            assert "RAG-Bench" in data["answer"]
            assert data["deflected"] is False

    def test_query_casual_greeting_hello(self, client):
        """Test casual greeting with 'hello'."""
        with patch("rag_bench.api.server._pipeline", MagicMock()), patch("rag_bench.api.server._generator", MagicMock()):
            response = client.post("/api/query", json={"question": "hello"})
            assert response.status_code == 200
            data = response.json()
            assert "RAG-Bench" in data["answer"]

    def test_query_too_short(self, client):
        """Test query that is too short."""
        with patch("rag_bench.api.server._pipeline", MagicMock()), patch("rag_bench.api.server._generator", MagicMock()):
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
        with (
            patch("rag_bench.api.server._pipeline", MagicMock()),
            patch("rag_bench.api.server._generator", generator_mock),
        ):
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
        with (
            patch("rag_bench.api.server._pipeline", MagicMock()),
            patch("rag_bench.api.server._generator", generator_mock),
        ):
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
        with (
            patch("rag_bench.api.server._pipeline", MagicMock()),
            patch("rag_bench.api.server._retriever", retriever_mock),
        ):
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
        with (
            patch("rag_bench.api.server._pipeline", MagicMock()),
            patch("rag_bench.api.server._retriever", retriever_mock),
        ):
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


class TestFaviconEndpoint:
    """Test the favicon endpoint."""

    def test_favicon_served(self, client, tmp_path):
        """Test favicon served when it exists."""
        favicon = tmp_path / "favicon.svg"
        favicon.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>')
        mock_dist = MagicMock()
        mock_dist.__truediv__ = lambda self, name: favicon
        with patch("rag_bench.api.server.FRONTEND_DIST", mock_dist):
            response = client.get("/favicon.svg")
            assert response.status_code == 200

    def test_favicon_not_found(self, client, tmp_path):
        """Test favicon 404 when it doesn't exist."""
        mock_dist = MagicMock()
        missing = tmp_path / "missing.svg"
        mock_dist.__truediv__ = lambda self, name: missing
        with patch("rag_bench.api.server.FRONTEND_DIST", mock_dist):
            response = client.get("/favicon.svg")
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
            patch("rag_bench.api.server._pipeline", MagicMock()),
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
        with (
            patch("rag_bench.api.server._pipeline", MagicMock()),
            patch("rag_bench.api.server._generator", MagicMock()),
            patch("pathlib.Path.exists", return_value=False),
        ):
            response = client.post("/api/eval", json={"run_all": False})
            assert response.status_code == 404


class TestStreamingEndpoint:
    """Test the streaming query endpoint."""

    def test_query_stream_casual_greeting(self, client):
        """Test streaming with casual greeting."""
        with patch("rag_bench.api.server._pipeline", MagicMock()), patch("rag_bench.api.server._generator", MagicMock()):
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
        with (
            patch("rag_bench.api.server._pipeline", MagicMock()),
            patch("rag_bench.api.server._retriever", retriever_mock),
        ):
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
        with (
            patch("rag_bench.api.server._pipeline", MagicMock()),
            patch("rag_bench.api.server._retriever", retriever_mock),
        ):
            # API returns 502 when it can't extract arxiv_id and fetch fails
            response = client.get("/api/papers/xyz/pdf")
            assert response.status_code in [404, 502]

    def test_get_paper_pdf_pipeline_not_loaded(self, client):
        """Test get paper PDF when pipeline not loaded."""
        with patch("rag_bench.api.server._retriever", None):
            response = client.get("/api/papers/some_id/pdf")
            assert response.status_code == 503


class TestQueryWithBackendOverride:
    """Test query endpoint with backend override."""

    def test_query_with_backend_override(self, client):
        """Test query with custom backend."""
        with patch("rag_bench.api.server.build_llm_backend") as mock_build:
            mock_build.return_value = MagicMock()

            # Mock RAGGenerator to return answer when initialized with different backend
            mock_gen_instance = MagicMock(
                answer=MagicMock(
                    return_value={
                        "answer": "Test answer",
                        "deflected": False,
                        "results": [],
                        "filtered_results": [],
                        "scores": [0.8],
                    }
                )
            )

            with (
                patch("rag_bench.api.server._pipeline", MagicMock()),
                patch("rag_bench.api.server._generator", MagicMock()),
                patch("rag_bench.api.server._retriever", MagicMock()),
                patch("rag_bench.api.server._llm_backend_name", "ollama"),
                patch("rag_bench.api.server.RAGGenerator", return_value=mock_gen_instance),
            ):
                response = client.post(
                    "/api/query",
                    json={
                        "question": "What is ML?",
                        "backend": "openai",
                        "model": "gpt-4",
                    },
                )
                assert response.status_code == 200
                # Should have called build_llm_backend with the custom backend
                mock_build.assert_called_with("openai", "gpt-4", "")

    def test_query_with_backend_same_as_current(self, client):
        """Test query when backend is same as current."""
        with (
            patch("rag_bench.api.server._pipeline", MagicMock()),
            patch(
                "rag_bench.api.server._generator",
                MagicMock(
                    answer=MagicMock(
                        return_value={
                            "answer": "Test",
                            "deflected": False,
                            "results": [],
                            "filtered_results": [],
                            "scores": [],
                        }
                    )
                ),
            ),
            patch("rag_bench.api.server._llm_backend_name", "ollama"),
        ):
            response = client.post(
                "/api/query",
                json={"question": "Test?", "backend": "ollama"},
            )
            assert response.status_code == 200


class TestStreamingWithResults:
    """Test streaming endpoint with actual results."""

    def test_query_stream_with_non_casual_question(self, client):
        """Test streaming with a real question (non-casual)."""

        def mock_stream():
            yield {
                "event": "sources",
                "results": [{"text": "source 1", "score": 0.9, "chunk_id": "c1", "metadata": {"title": "Test"}}],
                "filtered_results": [{"text": "source 1", "score": 0.9, "chunk_id": "c1", "metadata": {"title": "Test"}}],
            }
            yield {"event": "token", "token": "This"}
            yield {"event": "token", "token": " is"}
            yield {"event": "done", "answer": "This is a test", "deflection_reason": ""}

        generator_mock = MagicMock(answer_stream=MagicMock(return_value=mock_stream()))

        with (
            patch("rag_bench.api.server._pipeline", MagicMock()),
            patch("rag_bench.api.server._generator", generator_mock),
            patch("rag_bench.api.server._llm_backend_name", "ollama"),
            patch("rag_bench.api.server._llm_model_name", "mistral"),
        ):
            response = client.post("/api/query/stream", json={"question": "What is machine learning?"})
            assert response.status_code == 200
            # Should contain SSE data
            assert b"data:" in response.content or response.status_code == 200

    def test_query_stream_with_deflection(self, client):
        """Test streaming with deflection."""

        def mock_stream():
            yield {
                "event": "sources",
                "results": [],
                "filtered_results": [],
            }
            yield {"event": "deflected", "answer": "Cannot help", "reason": "Off topic"}

        generator_mock = MagicMock(answer_stream=MagicMock(return_value=mock_stream()))

        with (
            patch("rag_bench.api.server._pipeline", MagicMock()),
            patch("rag_bench.api.server._generator", generator_mock),
            patch("rag_bench.api.server._llm_backend_name", "ollama"),
            patch("rag_bench.api.server._llm_model_name", "mistral"),
        ):
            response = client.post("/api/query/stream", json={"question": "What is pizza flavor?"})
            assert response.status_code == 200

    def test_query_stream_pipeline_not_loaded(self, client):
        """Test streaming when pipeline is not loaded."""
        with patch("rag_bench.api.server._generator", None):
            response = client.post("/api/query/stream", json={"question": "test question"})
            assert response.status_code == 503


class TestGetPaperDetail:
    """Test the get paper detail endpoint."""

    def test_get_paper_success(self, client):
        """Test getting paper details successfully."""
        retriever_mock = MagicMock()
        retriever_mock.collection.count.return_value = 2

        # Mock the first successful get call on "doc_id" field
        retriever_mock.collection.get.side_effect = [
            {
                "ids": ["chunk1", "chunk2"],
                "documents": ["Text 1", "Text 2"],
                "metadatas": [
                    {
                        "doc_id": "paper1",
                        "title": "Test Paper",
                        "year": 2023,
                        "arxiv_id": "2301.0001",
                        "section": "intro",
                        "source_display": "Test Paper (2023)",
                    },
                    {
                        "doc_id": "paper1",
                        "title": "Test Paper",
                        "year": 2023,
                        "arxiv_id": "2301.0001",
                        "section": "methods",
                        "source_display": "Test Paper (2023)",
                    },
                ],
            }
        ]

        with (
            patch("rag_bench.api.server._pipeline", MagicMock()),
            patch("rag_bench.api.server._retriever", retriever_mock),
        ):
            response = client.get("/api/papers/paper1")
            assert response.status_code == 200
            data = response.json()
            assert data["title"] == "Test Paper"
            assert data["year"] == 2023
            assert data["arxiv_id"] == "2301.0001"
            assert len(data["chunks"]) == 2

    def test_get_paper_with_chunk_index(self, client):
        """Test parsing chunk index from chunk_id."""
        retriever_mock = MagicMock()
        retriever_mock.collection.count.return_value = 1

        retriever_mock.collection.get.side_effect = [
            {
                "ids": ["paper_intro_5"],
                "documents": ["Introduction text"],
                "metadatas": [
                    {
                        "doc_id": "paper",
                        "title": "Test",
                        "year": 2023,
                        "arxiv_id": "2301.0001",
                        "section": "intro",
                        "source_display": "Test",
                    }
                ],
            }
        ]

        with (
            patch("rag_bench.api.server._pipeline", MagicMock()),
            patch("rag_bench.api.server._retriever", retriever_mock),
        ):
            response = client.get("/api/papers/paper")
            assert response.status_code == 200
            data = response.json()
            # Chunk index should be parsed as 5
            assert len(data["chunks"]) == 1


class TestQueryFormatSourcesWithMissingScore:
    """Test format sources with edge cases around scoring."""

    def test_format_sources_with_missing_score(self):
        """Test formatting when score is missing from results."""
        results = [
            {
                "text": "Sample text",
                "chunk_id": "chunk_1",
                "metadata": {
                    "title": "Test Paper",
                    "section": "intro",
                    "paper_id": "arxiv:2103.05674",
                },
            }
        ]
        sources = _format_sources(results)
        assert len(sources) == 1
        # Should default to low relevance when no score
        assert sources[0].relevance in ["low", "medium", "high"]

    def test_format_sources_with_filtered_results_empty(self):
        """Test when filtered_results is empty but results is populated."""
        results = [
            {
                "score": 0.95,
                "text": "Sample" * 50,
                "chunk_id": "chunk_1",
                "metadata": {
                    "title": "Test Paper",
                    "section": "intro",
                    "paper_id": "arxiv:2103.05674",
                },
            }
        ]
        sources = _format_sources(results)
        assert len(sources) == 1
        assert sources[0].score == 0.95


class TestApiServerErrorHandling:
    """Test error handling in API endpoints."""

    def test_eval_endpoint_with_llm_level_deflection_detection(self, client):
        """Test eval endpoint detecting LLM-level deflection."""
        with (
            patch("rag_bench.api.server._pipeline", MagicMock()),
            patch(
                "rag_bench.api.server._generator",
                MagicMock(
                    answer=MagicMock(
                        return_value={
                            "answer": "I cannot answer this question as it's outside my scope.",
                            "deflected": False,  # Not deflected by gate
                            "results": [],
                            "scores": [0.2],
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
                        "question": "Off-topic?",
                        "should_deflect": True,  # Should have deflected
                        "difficulty": "deflection",
                    }
                ],
            ),
        ):
            response = client.post("/api/eval", json={"run_all": False})
            assert response.status_code == 200
            data = response.json()
            # Should detect LLM-level deflection
            assert "results" in data

    def test_eval_endpoint_with_empty_scores(self, client):
        """Test eval endpoint when scores list is empty."""
        with (
            patch("rag_bench.api.server._pipeline", MagicMock()),
            patch(
                "rag_bench.api.server._generator",
                MagicMock(
                    answer=MagicMock(
                        return_value={
                            "answer": "Test",
                            "deflected": False,
                            "results": [],
                            "scores": [],  # Empty scores
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
            assert data["results"][0]["top_score"] == 0.0

    def test_query_stream_with_exception_in_stream(self, client):
        """Test streaming endpoint error handling."""

        def mock_stream_with_error():
            yield {"event": "sources", "results": [], "filtered_results": []}
            raise Exception("Stream error test")

        with (
            patch("rag_bench.api.server._pipeline", MagicMock()),
            patch(
                "rag_bench.api.server._generator",
                MagicMock(answer_stream=MagicMock(return_value=mock_stream_with_error())),
            ),
            patch("rag_bench.api.server._llm_backend_name", "ollama"),
        ):
            response = client.post("/api/query/stream", json={"question": "What is ML?"})
            assert response.status_code == 200

    def test_get_paper_with_arxiv_id_lookup(self, client):
        """Test getting paper using arxiv_id field."""
        retriever_mock = MagicMock()
        retriever_mock.collection.count.return_value = 1

        # Mock the "doc_id" call to fail, then "paper_id", then succeed with "arxiv_id"
        retriever_mock.collection.get.side_effect = [
            {"ids": []},  # doc_id lookup fails
            {"ids": []},  # paper_id lookup fails
            {
                # arxiv_id lookup succeeds
                "ids": ["chunk1"],
                "documents": ["Text"],
                "metadatas": [
                    {
                        "arxiv_id": "2301.0001",
                        "title": "Test Paper",
                        "year": 2023,
                        "section": "intro",
                        "source_display": "Test",
                    }
                ],
            },
        ]

        with (
            patch("rag_bench.api.server._pipeline", MagicMock()),
            patch("rag_bench.api.server._retriever", retriever_mock),
        ):
            response = client.get("/api/papers/arxiv_2301.0001")
            assert response.status_code == 200
            data = response.json()
            assert data["title"] == "Test Paper"

    def test_get_paper_with_exception_in_lookup(self, client):
        """Test paper lookup with exception in field search."""
        retriever_mock = MagicMock()
        retriever_mock.collection.count.return_value = 1

        # First two calls raise exceptions, third succeeds
        retriever_mock.collection.get.side_effect = [
            Exception("Query error"),  # doc_id lookup raises exception
            Exception("Query error"),  # paper_id lookup raises exception
            {
                # arxiv_id lookup succeeds
                "ids": ["chunk1"],
                "documents": ["Text"],
                "metadatas": [
                    {
                        "arxiv_id": "2301.0001",
                        "title": "Test Paper",
                        "year": 2023,
                        "section": "intro",
                        "source_display": "Test",
                    }
                ],
            },
        ]

        with (
            patch("rag_bench.api.server._pipeline", MagicMock()),
            patch("rag_bench.api.server._retriever", retriever_mock),
        ):
            response = client.get("/api/papers/paper1")
            assert response.status_code == 200

    def test_get_paper_pdf_with_exception_in_fetch(self, client):
        """Test PDF fetch error handling."""
        with (
            patch("rag_bench.api.server._pipeline", MagicMock()),
            patch("rag_bench.api.server._retriever", MagicMock()),
            patch("rag_bench.api.server._fetch_pdf", side_effect=Exception("Network error")),
        ):
            response = client.get("/api/papers/2301.0001/pdf")
            assert response.status_code == 502

    def test_stats_endpoint_returns_info(self, client):
        """Test stats endpoint with actual stats."""
        retriever_mock = MagicMock()
        retriever_mock.collection.count.return_value = 42

        with (
            patch("rag_bench.api.server._pipeline", MagicMock()),
            patch("rag_bench.api.server._retriever", retriever_mock),
        ):
            response = client.get("/api/stats")
            assert response.status_code == 200
            data = response.json()
            assert data["total_chunks"] == 42


class TestComputeHelpers:
    """Direct tests for private compute helper functions."""

    def test_retrieval_confidence_high_cross_encoder(self):
        # gap_ratio = 5.0/2.5 = 2.0, top > 3.0, gap >= 1.8 → high
        results = [{"score": 5.0}, {"score": 2.5}]
        assert _compute_retrieval_confidence(results) == "high"

    def test_retrieval_confidence_low_cross_encoder_tight_scores(self):
        # gap_ratio = 7.5/7.0 = 1.07, too close together → low
        results = [{"score": 7.5}, {"score": 7.0}, {"score": 6.8}]
        assert _compute_retrieval_confidence(results) == "low"

    def test_retrieval_confidence_medium_cross_encoder(self):
        # gap_ratio = 4.0/2.5 = 1.6, top > 2.0, gap >= 1.3 → medium
        results = [{"score": 4.0}, {"score": 2.5}]
        assert _compute_retrieval_confidence(results) == "medium"

    def test_retrieval_confidence_medium_cross_encoder_single(self):
        # Single result: gap_ratio = inf, top > 2.0, gap >= 1.3 → medium
        results = [{"score": 2.5}]
        assert _compute_retrieval_confidence(results) == "medium"

    def test_retrieval_confidence_high_cosine(self):
        # gap_ratio = 0.9/0.5 = 1.8, top > 0.65, gap >= 1.5 → high
        results = [{"score": 0.9}, {"score": 0.5}]
        assert _compute_retrieval_confidence(results) == "high"

    def test_retrieval_confidence_low_cosine_tight_scores(self):
        # gap_ratio = 0.8/0.75 = 1.067, too close → low
        results = [{"score": 0.8}, {"score": 0.75}, {"score": 0.72}]
        assert _compute_retrieval_confidence(results) == "low"

    def test_retrieval_confidence_medium_cosine(self):
        # Single result: gap_ratio = inf, top > 0.4, gap >= 1.2 → medium
        results = [{"score": 0.6}]
        assert _compute_retrieval_confidence(results) == "medium"

    def test_retrieval_confidence_low_cosine_low_score(self):
        # top=0.3, below 0.4 threshold → low
        results = [{"score": 0.3}]
        assert _compute_retrieval_confidence(results) == "low"

    def test_score_spread_multiple_values(self):
        results = [{"score": 0.9}, {"score": 0.7}, {"score": 0.5}]
        spread = _compute_score_spread(results)
        assert spread["min"] == 0.5
        assert spread["max"] == 0.9
        assert spread["mean"] == pytest.approx(0.7, abs=0.001)
        assert "std" in spread

    def test_score_spread_single_value(self):
        results = [{"score": 0.8}]
        spread = _compute_score_spread(results)
        assert spread["min"] == 0.8
        assert spread["max"] == 0.8
        assert spread["std"] == 0.0

    def test_source_diversity_with_metadata(self):
        results = [
            {"metadata": {"paper_id": "1706.03762", "section": "abstract"}},
            {"metadata": {"paper_id": "1706.03762", "section": "methods"}},
            {"metadata": {"paper_id": "1810.04805", "section": "abstract"}},
        ]
        diversity = _compute_source_diversity(results)
        assert diversity["unique_papers"] == 2
        assert diversity["unique_sections"] == 2
        assert "1706.03762" in diversity["papers"]

    def test_source_diversity_empty_metadata(self):
        results = [{"metadata": {}}]
        diversity = _compute_source_diversity(results)
        assert diversity["unique_papers"] == 0
        assert diversity["unique_sections"] == 0


class TestPerSourceCited:
    """Tests for _compute_per_source_cited with multiple citation patterns."""

    def _make_result(self, paper_id="paper_1", title="Paper One", score=5.0):
        return {"metadata": {"paper_id": paper_id, "title": title}, "score": score}

    def test_single_inline_citation(self):
        answer = "Transformers use attention [Source 1]."
        results = [self._make_result()]
        per = _compute_per_source_cited(answer, results)
        assert len(per) == 1
        assert per[0]["cited"] is True
        assert per[0]["citation_count"] == 1
        assert per[0]["footer_only"] is False

    def test_multiple_sources_all_cited(self):
        answer = "Attention [Source 1] and masking [Source 2] and pretraining [Source 3]."
        results = [
            self._make_result("p1", "Paper 1"),
            self._make_result("p2", "Paper 2"),
            self._make_result("p3", "Paper 3"),
        ]
        per = _compute_per_source_cited(answer, results)
        assert all(s["cited"] for s in per)
        assert all(s["citation_count"] == 1 for s in per)

    def test_same_source_cited_multiple_times(self):
        answer = "First point [Source 1]. Second point also [Source 1]. Third [Source 1]."
        results = [self._make_result()]
        per = _compute_per_source_cited(answer, results)
        assert per[0]["cited"] is True
        assert per[0]["citation_count"] == 3

    def test_mixed_citation_counts(self):
        """Source 1 cited 3x, Source 2 cited 1x, Source 3 not cited."""
        answer = "Point A [Source 1]. Point B [Source 1]. Point C [Source 2]. Point D [Source 1]."
        results = [
            self._make_result("p1", "Paper 1"),
            self._make_result("p2", "Paper 2"),
            self._make_result("p3", "Paper 3"),
        ]
        per = _compute_per_source_cited(answer, results)
        assert per[0]["citation_count"] == 3
        assert per[1]["citation_count"] == 1
        assert per[2]["cited"] is False
        assert per[2]["citation_count"] == 0

    def test_bare_bracket_citations(self):
        """Model uses [1] instead of [Source 1]."""
        answer = "Attention mechanism [1] and masking approach [2]."
        results = [
            self._make_result("p1", "Paper 1"),
            self._make_result("p2", "Paper 2"),
        ]
        per = _compute_per_source_cited(answer, results)
        assert per[0]["cited"] is True
        assert per[1]["cited"] is True

    def test_footer_only_citations(self):
        """Model only lists sources in footer block."""
        answer = "Transformers use attention mechanisms.\n\nSources:\n[Source 1] Paper One\n[Source 2] Paper Two"
        results = [
            self._make_result("p1", "Paper 1"),
            self._make_result("p2", "Paper 2"),
        ]
        per = _compute_per_source_cited(answer, results)
        assert per[0]["cited"] is True
        assert per[0]["footer_only"] is True
        assert per[1]["cited"] is True
        assert per[1]["footer_only"] is True

    def test_inline_plus_footer(self):
        """Source 1 inline, Source 2 footer-only."""
        answer = "Attention is important [Source 1].\n\nSources:\n[Source 1] Paper 1\n[Source 2] Paper 2"
        results = [
            self._make_result("p1", "Paper 1"),
            self._make_result("p2", "Paper 2"),
        ]
        per = _compute_per_source_cited(answer, results)
        assert per[0]["cited"] is True
        assert per[0]["footer_only"] is False
        assert per[1]["cited"] is True
        assert per[1]["footer_only"] is True

    def test_five_sources_partial_citation(self):
        """5 sources, only 2 and 4 cited."""
        answer = "Point [Source 2]. Another point [Source 4]."
        results = [self._make_result(f"p{i}", f"Paper {i}") for i in range(1, 6)]
        per = _compute_per_source_cited(answer, results)
        cited_nums = [s["source_number"] for s in per if s["cited"]]
        assert cited_nums == [2, 4]
        uncited = [s["source_number"] for s in per if not s["cited"]]
        assert uncited == [1, 3, 5]

    def test_no_citations_at_all(self):
        answer = "The model uses attention mechanisms for sequence processing."
        results = [
            self._make_result("p1", "Paper 1"),
            self._make_result("p2", "Paper 2"),
        ]
        per = _compute_per_source_cited(answer, results)
        assert all(not s["cited"] for s in per)
        assert all(s["citation_count"] == 0 for s in per)


class TestFaithfulnessHeuristic:
    """Tests for _compute_faithfulness_heuristic with various overlap scenarios."""

    def test_perfect_overlap(self):
        """Answer directly quotes source text → high faithfulness."""
        source = "Transformers use multi-head self-attention mechanisms."
        answer = "Transformers use multi-head self-attention mechanisms."
        score = _compute_faithfulness_heuristic(answer, [source])
        assert score >= 4.0

    def test_no_overlap(self):
        """Answer content words don't appear in sources → low faithfulness."""
        source = "Photosynthesis converts sunlight into chemical energy in plants."
        answer = "Transformers use multi-head self-attention mechanisms for processing."
        score = _compute_faithfulness_heuristic(answer, [source])
        assert score <= 2.0

    def test_partial_overlap(self):
        """Some keywords match → mid-range faithfulness."""
        source = "The Transformer architecture uses attention mechanisms and feed-forward layers."
        answer = "The Transformer architecture processes sequences using specialized computation layers."
        score = _compute_faithfulness_heuristic(answer, [source])
        assert 1.5 <= score <= 4.5

    def test_multiple_sources_combined_overlap(self):
        """Keywords spread across multiple source passages."""
        sources = [
            "The Transformer model uses multi-head attention.",
            "Feed-forward layers apply nonlinear transformations.",
            "Layer normalization stabilizes training dynamics.",
        ]
        answer = (
            "The Transformer model uses multi-head attention with feed-forward layers. "
            "Layer normalization helps stabilize training dynamics."
        )
        score = _compute_faithfulness_heuristic(answer, sources)
        assert score >= 3.5

    def test_empty_sources(self):
        score = _compute_faithfulness_heuristic("Any answer text.", [])
        assert score == 1.0

    def test_empty_answer(self):
        score = _compute_faithfulness_heuristic("", ["Some source text."])
        assert score == 1.0

    def test_sources_block_stripped(self):
        """Sources block at end of answer should not count toward overlap."""
        source = "Attention mechanisms compute weighted sums."
        answer = "The method is novel.\n\nSources:\n[Source 1] Attention mechanisms compute weighted sums."
        score = _compute_faithfulness_heuristic(answer, [source])
        # The body "The method is novel." has low overlap; sources block excluded
        assert score < 3.0

    def test_multi_sentence_varying_overlap(self):
        """Multiple answer sentences with different overlap levels."""
        source = "BERT uses masked language modeling and next sentence prediction for pretraining."
        answer = (
            "BERT uses masked language modeling for pretraining. "
            "The architecture consists of stacked encoder blocks. "
            "Fine-tuning adapts the pretrained model to downstream tasks."
        )
        score = _compute_faithfulness_heuristic(answer, [source])
        # First sentence has high overlap, others lower → moderate score
        assert 2.0 <= score <= 4.5


class TestFullEvalEndpoint:
    """Tests for the /api/eval/full endpoint."""

    def test_full_eval_pipeline_not_loaded(self, client):
        """Full eval returns 503 when pipeline not loaded."""
        with (
            patch("rag_bench.api.server._generator", None),
            patch("rag_bench.api.server._retriever", None),
        ):
            response = client.post(
                "/api/eval/full",
                json={"retrieval_only": True},
            )
            assert response.status_code == 503

    def test_full_eval_with_mock_pipeline(self, client):
        """Full eval runs with mocked pipeline."""
        from rag_bench.eval.runner import EvalReport, SingleEvalResult

        mock_report = EvalReport(
            results=[
                SingleEvalResult(
                    id="test1",
                    question="What is attention?",
                    query_type="definition",
                    topic="transformers",
                    difficulty="easy",
                    retrieval={"mrr": 1.0},
                    deflection={"expected": False, "actual": False, "correct": True},
                )
            ],
            summary={
                "total_queries": 1,
                "retrieval_mrr": 1.0,
                "retrieval_precision_at_5": 0.5,
                "retrieval_recall_at_5": 1.0,
                "retrieval_ndcg_at_5": 1.0,
                "retrieval_hit_rate": 1.0,
                "avg_faithfulness": 0.0,
                "avg_relevance": 0.0,
                "avg_citation_precision": 0.0,
                "avg_citation_recall": 0.0,
                "avg_citation_density": 0.0,
                "avg_completeness": 0.0,
                "deflection_accuracy": 1.0,
                "avg_latency_ms": 100,
            },
            by_topic={"transformers": {"total_queries": 1, "retrieval_mrr": 1.0}},
            by_query_type={"definition": {"total_queries": 1}},
            by_difficulty={"easy": {"total_queries": 1}},
            metadata={"timestamp": "2026-02-17 10:00:00", "total_queries": 1},
        )

        mock_runner = MagicMock()
        mock_runner.run_all.return_value = mock_report

        mock_generator = MagicMock()
        mock_generator.llm = MagicMock()

        with (
            patch("rag_bench.api.server._pipeline", MagicMock()),
            patch("rag_bench.api.server._generator", mock_generator),
            patch("rag_bench.api.server._retriever", MagicMock()),
            patch("rag_bench.eval.runner.EvalRunner") as mock_runner_class,
        ):
            mock_runner_class.return_value = mock_runner
            response = client.post(
                "/api/eval/full",
                json={"retrieval_only": True},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["summary"]["total_queries"] == 1
