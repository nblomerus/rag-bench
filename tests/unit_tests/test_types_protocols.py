"""
Unit tests for rag_bench.core.types and rag_bench.core.protocols.

Tests cover:
- ChunkData creation, frozen immutability, to_dict()
- RetrievalResult and GenerationResult defaults and field access
- Protocol conformance: existing classes satisfy their protocols
"""

import pytest

from rag_bench.core.chunker import PaperChunker
from rag_bench.core.generator import (
    OllamaBackend,
    OpenAICompatibleBackend,
    TemplateFallbackBackend,
)
from rag_bench.core.protocols import Chunker, LLMBackend
from rag_bench.core.types import ChunkData, GenerationResult, RetrievalResult

# ═══════════════════════════════════════════════════════════════════════════
# ChunkData
# ═══════════════════════════════════════════════════════════════════════════


class TestChunkData:
    @pytest.fixture
    def sample_chunk(self):
        return ChunkData(
            chunk_id="c001",
            doc_id="doc_42",
            text="Attention is all you need.",
            section="abstract",
            metadata={"year": 2017, "source_display": "Vaswani et al."},
        )

    def test_fields(self, sample_chunk):
        assert sample_chunk.chunk_id == "c001"
        assert sample_chunk.doc_id == "doc_42"
        assert sample_chunk.text == "Attention is all you need."
        assert sample_chunk.section == "abstract"
        assert sample_chunk.metadata["year"] == 2017

    def test_frozen_immutability(self, sample_chunk):
        with pytest.raises(AttributeError):
            sample_chunk.text = "new text"

    def test_default_metadata(self):
        chunk = ChunkData(chunk_id="c", doc_id="d", text="t", section="s")
        assert chunk.metadata == {}

    def test_to_dict(self, sample_chunk):
        d = sample_chunk.to_dict()
        assert isinstance(d, dict)
        assert d["chunk_id"] == "c001"
        assert d["doc_id"] == "doc_42"
        assert d["text"] == "Attention is all you need."
        assert d["section"] == "abstract"
        assert d["metadata"]["year"] == 2017

    def test_to_dict_returns_copy_of_metadata(self, sample_chunk):
        d = sample_chunk.to_dict()
        d["metadata"]["injected"] = True
        assert "injected" not in sample_chunk.metadata

    def test_equality(self):
        a = ChunkData(chunk_id="c", doc_id="d", text="t", section="s")
        b = ChunkData(chunk_id="c", doc_id="d", text="t", section="s")
        assert a == b

    def test_not_hashable_due_to_dict_field(self, sample_chunk):
        # frozen dataclass with mutable dict field is not hashable
        with pytest.raises(TypeError):
            hash(sample_chunk)

    def test_not_hashable_even_empty_metadata(self):
        # Even empty dict metadata prevents hashing (dict.__hash__ is None)
        chunk = ChunkData(chunk_id="c", doc_id="d", text="t", section="s")
        with pytest.raises(TypeError):
            hash(chunk)


# ═══════════════════════════════════════════════════════════════════════════
# RetrievalResult
# ═══════════════════════════════════════════════════════════════════════════


class TestRetrievalResult:
    @pytest.fixture
    def sample_result(self):
        chunk = ChunkData(chunk_id="c1", doc_id="d1", text="hello", section="intro")
        return RetrievalResult(chunk=chunk, relevance_score=0.85)

    def test_fields(self, sample_result):
        assert sample_result.chunk.chunk_id == "c1"
        assert sample_result.relevance_score == 0.85

    def test_defaults(self, sample_result):
        assert sample_result.sources == []
        assert sample_result.rerank_score is None

    def test_with_sources_and_rerank(self):
        chunk = ChunkData(chunk_id="c", doc_id="d", text="t", section="s")
        r = RetrievalResult(
            chunk=chunk,
            relevance_score=0.9,
            sources=["bm25", "dense"],
            rerank_score=0.95,
        )
        assert r.sources == ["bm25", "dense"]
        assert r.rerank_score == 0.95


# ═══════════════════════════════════════════════════════════════════════════
# GenerationResult
# ═══════════════════════════════════════════════════════════════════════════


class TestGenerationResult:
    def test_non_deflected(self):
        r = GenerationResult(
            answer="Transformers use attention.",
            deflected=False,
            sources=["[1] Vaswani et al."],
        )
        assert r.answer == "Transformers use attention."
        assert r.deflected is False
        assert r.sources == ["[1] Vaswani et al."]

    def test_deflected(self):
        r = GenerationResult(
            answer="I cannot answer that.",
            deflected=True,
            deflection_reason="Off topic",
        )
        assert r.deflected is True
        assert r.deflection_reason == "Off topic"

    def test_defaults(self):
        r = GenerationResult(answer="test", deflected=False)
        assert r.sources == []
        assert r.deflection_reason is None
        assert r.results == []
        assert r.scores == []


# ═══════════════════════════════════════════════════════════════════════════
# Protocol conformance
# ═══════════════════════════════════════════════════════════════════════════


class TestProtocolConformance:
    """Verify that existing concrete classes satisfy their Protocol interfaces.

    Uses runtime_checkable isinstance() checks — these confirm the class
    has the right method signatures at the structural level.
    """

    def test_chunker_protocol(self):
        assert isinstance(PaperChunker(), Chunker)

    def test_retriever_protocol(self):
        """HybridRetriever satisfies Retriever protocol (has retrieve method)."""

        # Can't instantiate HybridRetriever without ChromaDB, so check the class
        # has the method signature
        assert hasattr(
            __import__("rag_bench.core.retriever", fromlist=["HybridRetriever"]).HybridRetriever,
            "retrieve",
        )

    def test_generator_protocol(self):
        """RAGGenerator has a generate() method matching Generator protocol."""

        RAGGenerator = __import__("rag_bench.core.generator", fromlist=["RAGGenerator"]).RAGGenerator
        assert hasattr(RAGGenerator, "generate")

    def test_llm_backend_protocol(self):
        """Concrete LLM backends satisfy LLMBackend protocol."""
        # TemplateFallbackBackend can be instantiated without dependencies
        assert isinstance(TemplateFallbackBackend(), LLMBackend)

        # Check the others have the generate method
        assert hasattr(OllamaBackend, "generate")
        assert hasattr(OpenAICompatibleBackend, "generate")
