"""Test API schemas (request/response models)."""

import pytest

from rag_bench.api.schemas import (
    EvalRequest,
    EvalResponse,
    EvalResult,
    PaperChunk,
    PaperDetail,
    PaperSummary,
    QueryRequest,
    QueryResponse,
    SourceResult,
    StatsResponse,
)


class TestQueryRequest:
    """Test QueryRequest schema."""

    def test_query_request_basic(self):
        """Test basic query request."""
        req = QueryRequest(question="What is machine learning?")
        assert req.question == "What is machine learning?"
        assert req.top_k == 5
        assert req.backend is None
        assert req.model is None

    def test_query_request_with_top_k(self):
        """Test query request with custom top_k."""
        req = QueryRequest(question="What is attention?", top_k=10)
        assert req.top_k == 10

    def test_query_request_with_backend(self):
        """Test query request with custom backend."""
        req = QueryRequest(question="What is attention?", backend="openai", model="gpt-4")
        assert req.backend == "openai"
        assert req.model == "gpt-4"

    def test_query_request_validation_min_length(self):
        """Test query must have min length."""
        with pytest.raises(ValueError):
            QueryRequest(question="")

    def test_query_request_validation_max_length(self):
        """Test query must have max length."""
        with pytest.raises(ValueError):
            QueryRequest(question="a" * 1001)

    def test_query_request_validation_top_k_min(self):
        """Test top_k must be >= 1."""
        with pytest.raises(ValueError):
            QueryRequest(question="What is ml?", top_k=0)

    def test_query_request_validation_top_k_max(self):
        """Test top_k must be <= 20."""
        with pytest.raises(ValueError):
            QueryRequest(question="What is ml?", top_k=21)


class TestSourceResult:
    """Test SourceResult schema."""

    def test_source_result_basic(self):
        """Test basic source result."""
        source = SourceResult(
            rank=1,
            score=0.95,
            title="Attention is All You Need",
            section="Abstract",
            text_preview="In this paper we propose...",
            paper_id="arxiv:1706.03762",
            chunk_id="chunk_123",
            relevance="high",
        )
        assert source.rank == 1
        assert source.score == 0.95
        assert source.relevance == "high"

    def test_source_result_defaults(self):
        """Test source result with defaults."""
        source = SourceResult(rank=1, score=0.5)
        assert source.title == ""
        assert source.section == ""
        assert source.text_preview == ""
        assert source.paper_id == ""
        assert source.chunk_id == ""
        assert source.relevance == ""


class TestQueryResponse:
    """Test QueryResponse schema."""

    def test_query_response_basic(self):
        """Test basic query response."""
        sources = [
            SourceResult(rank=1, score=0.9, title="Paper 1"),
            SourceResult(rank=2, score=0.8, title="Paper 2"),
        ]
        response = QueryResponse(
            answer="The answer is...",
            sources=sources,
            deflected=False,
            deflection_reason="",
            scores=[0.9, 0.8],
            latency_ms=123.5,
            backend="ollama",
            model="mistral:7b",
        )
        assert response.answer == "The answer is..."
        assert len(response.sources) == 2
        assert response.deflected is False

    def test_query_response_deflected(self):
        """Test deflected query response."""
        response = QueryResponse(
            answer="I cannot answer this question.",
            sources=[],
            deflected=True,
            deflection_reason="No relevant papers found",
            scores=[],
            latency_ms=50.0,
            backend="ollama",
            model="mistral:7b",
        )
        assert response.deflected is True
        assert response.deflection_reason == "No relevant papers found"
        assert len(response.sources) == 0


class TestStatsResponse:
    """Test StatsResponse schema."""

    def test_stats_response(self):
        """Test stats response."""
        stats = StatsResponse(
            total_chunks=1000,
            total_papers=50,
            collection_name="arxiv",
            embedding_model="all-MiniLM-L6-v2",
            llm_backend="ollama",
            llm_model="mistral:7b",
        )
        assert stats.total_chunks == 1000
        assert stats.total_papers == 50
        assert stats.collection_name == "arxiv"


class TestEvalRequest:
    """Test EvalRequest schema."""

    def test_eval_request_default(self):
        """Test eval request with default."""
        req = EvalRequest()
        assert req.run_all is False

    def test_eval_request_with_run_all(self):
        """Test eval request with run_all enabled."""
        req = EvalRequest(run_all=True)
        assert req.run_all is True


class TestEvalResult:
    """Test EvalResult schema."""

    def test_eval_result(self):
        """Test eval result."""
        result = EvalResult(
            id="q1",
            question="What is attention?",
            passed=True,
            expected_deflect=False,
            actual_deflect=False,
            deflect_source="",
            top_score=0.95,
            difficulty="easy",
        )
        assert result.id == "q1"
        assert result.passed is True
        assert result.difficulty == "easy"

    def test_eval_result_failed(self):
        """Test eval result that failed."""
        result = EvalResult(
            id="q2",
            question="What is something obscure?",
            passed=False,
            expected_deflect=True,
            actual_deflect=False,
            deflect_source="relevance_gate",
            top_score=0.3,
            difficulty="hard",
        )
        assert result.passed is False
        assert result.actual_deflect is False


class TestEvalResponse:
    """Test EvalResponse schema."""

    def test_eval_response(self):
        """Test eval response."""
        results = [
            EvalResult(
                id="q1",
                question="Q1?",
                passed=True,
                expected_deflect=False,
                actual_deflect=False,
                top_score=0.9,
                difficulty="easy",
            ),
            EvalResult(
                id="q2",
                question="Q2?",
                passed=False,
                expected_deflect=True,
                actual_deflect=False,
                top_score=0.5,
                difficulty="hard",
            ),
        ]
        response = EvalResponse(
            total=2,
            correct=1,
            accuracy=0.5,
            results=results,
            by_difficulty={"easy": 1, "hard": 1},
        )
        assert response.total == 2
        assert response.correct == 1
        assert response.accuracy == 0.5


class TestPaperSummary:
    """Test PaperSummary schema."""

    def test_paper_summary_basic(self):
        """Test paper summary."""
        summary = PaperSummary(
            paper_id="arxiv:2103.05674",
            title="ELECTRA: Pre-training Text Encoders as Discriminators",
            year=2020,
            arxiv_id="2103.05674",
            chunk_count=42,
            sections=["Abstract", "Introduction", "Method"],
        )
        assert summary.paper_id == "arxiv:2103.05674"
        assert summary.year == 2020
        assert len(summary.sections) == 3

    def test_paper_summary_defaults(self):
        """Test paper summary with defaults."""
        summary = PaperSummary(
            paper_id="arxiv:2103.05674",
            title="Some Paper",
        )
        assert summary.year == 0
        assert summary.arxiv_id == ""
        assert summary.chunk_count == 0
        assert summary.sections == []


class TestPaperChunk:
    """Test PaperChunk schema."""

    def test_paper_chunk_basic(self):
        """Test paper chunk."""
        chunk = PaperChunk(
            chunk_id="chunk_123",
            text="This is the chunk text",
            section="Introduction",
            chunk_index=0,
        )
        assert chunk.chunk_id == "chunk_123"
        assert chunk.text == "This is the chunk text"
        assert chunk.section == "Introduction"

    def test_paper_chunk_defaults(self):
        """Test paper chunk with defaults."""
        chunk = PaperChunk(
            chunk_id="chunk_456",
            text="Some text",
        )
        assert chunk.section == ""
        assert chunk.chunk_index == 0


class TestPaperDetail:
    """Test PaperDetail schema."""

    def test_paper_detail_basic(self):
        """Test paper detail."""
        chunks = [
            PaperChunk(chunk_id="c1", text="Text 1"),
            PaperChunk(chunk_id="c2", text="Text 2"),
        ]
        detail = PaperDetail(
            paper_id="arxiv:2103.05674",
            title="ELECTRA",
            year=2020,
            arxiv_id="2103.05674",
            source_display="arXiv 2103.05674",
            chunks=chunks,
            sections=["Abstract", "Introduction"],
        )
        assert detail.paper_id == "arxiv:2103.05674"
        assert len(detail.chunks) == 2
        assert len(detail.sections) == 2

    def test_paper_detail_defaults(self):
        """Test paper detail with defaults."""
        detail = PaperDetail(
            paper_id="arxiv:2103.05674",
            title="ELECTRA",
        )
        assert detail.year == 0
        assert detail.arxiv_id == ""
        assert detail.source_display == ""
        assert detail.chunks == []
        assert detail.sections == []
