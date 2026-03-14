"""
Unit tests for rag_bench.core.pipeline and adapter methods.

Tests cover:
- build_pipeline() factory with default and custom configs
- RAGPipeline.query() end-to-end (mocked components)
- HybridRetriever.retrieve() adapter method
- RAGGenerator.generate() adapter method
- RAGGenerator._retrieval_results_to_dicts() conversion
"""

from unittest.mock import MagicMock, patch

import pytest

from rag_bench.core.configs import (
    GeneratorConfig,
    PipelineConfig,
)
from rag_bench.core.generator import RAGGenerator, RelevanceGate, TemplateFallbackBackend
from rag_bench.core.pipeline import RAGPipeline, build_pipeline
from rag_bench.core.retriever import HybridRetriever
from rag_bench.core.types import ChunkData, GenerationResult, RetrievalResult

# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_chunk():
    return ChunkData(
        chunk_id="chunk_001",
        doc_id="doc_01",
        text="Transformers use self-attention mechanisms.",
        section="intro",
        metadata={"source_display": "Vaswani et al.", "section": "intro"},
    )


@pytest.fixture
def sample_retrieval_results(sample_chunk):
    return [
        RetrievalResult(chunk=sample_chunk, relevance_score=0.9, sources=["dense"]),
        RetrievalResult(
            chunk=ChunkData(
                chunk_id="chunk_002",
                doc_id="doc_02",
                text="BERT uses masked language modeling for pre-training.",
                section="methods",
                metadata={"source_display": "Devlin et al.", "section": "methods"},
            ),
            relevance_score=0.75,
            sources=["bm25"],
        ),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# build_pipeline()
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildPipeline:
    """Test the pipeline factory function with mocked heavy dependencies."""

    @patch("rag_bench.core.pipeline.HybridRetriever")
    @patch("rag_bench.core.pipeline.BGEEmbedder")
    @patch("rag_bench.core.pipeline.PaperChunker")
    @patch("rag_bench.core.pipeline.build_llm_backend")
    def test_build_with_defaults(self, mock_build_llm, mock_chunker, mock_embedder, mock_retriever):
        mock_build_llm.return_value = MagicMock()
        pipeline = build_pipeline()

        assert isinstance(pipeline, RAGPipeline)
        assert pipeline.config.name == "default"
        mock_chunker.assert_called_once()
        mock_embedder.assert_called_once()
        mock_retriever.assert_called_once()
        mock_build_llm.assert_called_once()

    @patch("rag_bench.core.pipeline.HybridRetriever")
    @patch("rag_bench.core.pipeline.BGEEmbedder")
    @patch("rag_bench.core.pipeline.PaperChunker")
    @patch("rag_bench.core.pipeline.build_llm_backend")
    def test_build_with_custom_config(self, mock_build_llm, mock_chunker, mock_embedder, mock_retriever):
        mock_build_llm.return_value = MagicMock()
        cfg = PipelineConfig(
            name="test_run",
            generator=GeneratorConfig(llm_backend="ollama", top_k=5),
        )
        pipeline = build_pipeline(cfg)

        assert pipeline.config.name == "test_run"
        mock_build_llm.assert_called_once_with("ollama", "gemma2:27b", "http://localhost:11434")

    @patch("rag_bench.core.pipeline.HybridRetriever")
    @patch("rag_bench.core.pipeline.BGEEmbedder")
    @patch("rag_bench.core.pipeline.PaperChunker")
    @patch("rag_bench.core.pipeline.build_llm_backend")
    def test_citation_boost_disabled(self, mock_build_llm, mock_chunker, mock_embedder, mock_retriever):
        mock_build_llm.return_value = MagicMock()
        cfg = PipelineConfig(
            generator=GeneratorConfig(enable_citation_boost=False),
        )
        pipeline = build_pipeline(cfg)
        # Generator should have been created without citation_booster
        # (booster=None when disabled)
        assert pipeline.generator is not None


# ═══════════════════════════════════════════════════════════════════════════
# RAGPipeline.query()
# ═══════════════════════════════════════════════════════════════════════════


class TestRAGPipelineQuery:
    @patch("rag_bench.core.pipeline.HybridRetriever")
    @patch("rag_bench.core.pipeline.BGEEmbedder")
    @patch("rag_bench.core.pipeline.PaperChunker")
    @patch("rag_bench.core.pipeline.build_llm_backend")
    def test_query_calls_retrieve_then_generate(
        self,
        mock_build_llm,
        mock_chunker,
        mock_embedder,
        mock_retriever,
        sample_retrieval_results,
    ):
        mock_build_llm.return_value = MagicMock()
        pipeline = build_pipeline()

        # Replace components with mocks for query() testing
        mock_ret = MagicMock()
        mock_gen = MagicMock()
        pipeline.retriever = mock_ret
        pipeline.generator = mock_gen

        mock_ret.retrieve.return_value = sample_retrieval_results
        mock_gen.generate.return_value = GenerationResult(
            answer="Attention is used.",
            deflected=False,
            sources=["[1] Vaswani"],
        )

        result = pipeline.query("What is attention?")

        mock_ret.retrieve.assert_called_once()
        mock_gen.generate.assert_called_once_with("What is attention?", context=sample_retrieval_results)
        assert isinstance(result, GenerationResult)
        assert result.answer == "Attention is used."

    @patch("rag_bench.core.pipeline.HybridRetriever")
    @patch("rag_bench.core.pipeline.BGEEmbedder")
    @patch("rag_bench.core.pipeline.PaperChunker")
    @patch("rag_bench.core.pipeline.build_llm_backend")
    def test_query_uses_custom_top_k(
        self,
        mock_build_llm,
        mock_chunker,
        mock_embedder,
        mock_retriever,
    ):
        mock_build_llm.return_value = MagicMock()
        pipeline = build_pipeline()

        mock_ret = MagicMock()
        mock_gen = MagicMock()
        pipeline.retriever = mock_ret
        pipeline.generator = mock_gen

        mock_ret.retrieve.return_value = []
        mock_gen.generate.return_value = GenerationResult(answer="", deflected=True)

        pipeline.query("test?", top_k=3)
        mock_ret.retrieve.assert_called_once_with("test?", top_k=3)


# ═══════════════════════════════════════════════════════════════════════════
# HybridRetriever.retrieve() adapter
# ═══════════════════════════════════════════════════════════════════════════


class TestRetrieverAdapter:
    """Test the retrieve() method that wraps query() with typed results."""

    def test_retrieve_converts_dicts_to_retrieval_results(self):
        # Create a partially-mocked retriever (bypass __init__)
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.query = MagicMock(
            return_value=[
                {
                    "chunk_id": "c1",
                    "text": "Attention mechanism explained.",
                    "score": 0.92,
                    "rerank_score": 0.95,
                    "metadata": {"doc_id": "d1", "section": "intro", "year": 2017},
                    "sources": ["dense", "bm25"],
                },
                {
                    "chunk_id": "c2",
                    "text": "BERT overview.",
                    "score": 0.80,
                    "rerank_score": None,
                    "metadata": {"doc_id": "d2", "section": "abstract"},
                    "sources": ["dense"],
                },
            ]
        )

        results = retriever.retrieve("What is attention?", top_k=5)

        retriever.query.assert_called_once_with("What is attention?", top_k=5)
        assert len(results) == 2

        r0 = results[0]
        assert isinstance(r0, RetrievalResult)
        assert isinstance(r0.chunk, ChunkData)
        assert r0.chunk.chunk_id == "c1"
        assert r0.chunk.doc_id == "d1"
        assert r0.chunk.text == "Attention mechanism explained."
        assert r0.chunk.section == "intro"
        assert r0.relevance_score == 0.92
        assert r0.rerank_score == 0.95
        assert r0.sources == ["dense", "bm25"]

        r1 = results[1]
        assert r1.chunk.chunk_id == "c2"
        assert r1.rerank_score is None

    def test_retrieve_handles_empty_results(self):
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.query = MagicMock(return_value=[])

        results = retriever.retrieve("nonexistent topic")
        assert results == []

    def test_retrieve_handles_missing_metadata(self):
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.query = MagicMock(
            return_value=[
                {
                    "chunk_id": "c1",
                    "text": "Some text.",
                    "score": 0.5,
                },
            ]
        )

        results = retriever.retrieve("query")
        r = results[0]
        assert r.chunk.doc_id == ""
        assert r.chunk.section == ""
        assert r.chunk.metadata == {}
        assert r.sources == []
        assert r.rerank_score is None


# ═══════════════════════════════════════════════════════════════════════════
# RAGGenerator._retrieval_results_to_dicts()
# ═══════════════════════════════════════════════════════════════════════════


class TestRetrievalResultsToDicts:
    def test_conversion(self, sample_retrieval_results):
        dicts = RAGGenerator._retrieval_results_to_dicts(sample_retrieval_results)

        assert len(dicts) == 2
        assert dicts[0]["chunk_id"] == "chunk_001"
        assert dicts[0]["text"] == "Transformers use self-attention mechanisms."
        assert dicts[0]["score"] == 0.9
        assert dicts[0]["sources"] == ["dense"]
        assert isinstance(dicts[0]["metadata"], dict)

    def test_empty_input(self):
        assert RAGGenerator._retrieval_results_to_dicts([]) == []


# ═══════════════════════════════════════════════════════════════════════════
# RAGGenerator.generate() adapter
# ═══════════════════════════════════════════════════════════════════════════


class TestGeneratorAdapter:
    """Test the generate() method that wraps the internal answer logic."""

    def _make_generator(self):
        """Create a RAGGenerator with mocked dependencies."""
        mock_retriever = MagicMock()
        llm = TemplateFallbackBackend()
        gate = RelevanceGate(min_top_score=0.3)

        gen = RAGGenerator(
            retriever=mock_retriever,
            llm_backend=llm,
            relevance_gate=gate,
            top_k=10,
        )
        return gen

    def test_generate_off_topic_deflects(self):
        gen = self._make_generator()
        chunk = ChunkData(chunk_id="c", doc_id="d", text="t", section="s")
        context = [RetrievalResult(chunk=chunk, relevance_score=0.9)]

        result = gen.generate("What is the best pizza topping?", context=context)

        assert isinstance(result, GenerationResult)
        assert result.deflected is True
        assert "AI/ML" in result.deflection_reason or "not appear" in result.deflection_reason

    def test_generate_opinion_deflects(self):
        gen = self._make_generator()
        chunk = ChunkData(
            chunk_id="c",
            doc_id="d",
            text="Transformers are neural network architectures.",
            section="intro",
        )
        context = [RetrievalResult(chunk=chunk, relevance_score=0.9)]

        result = gen.generate(
            "In your opinion, what is the best transformer architecture?",
            context=context,
        )

        assert result.deflected is True
        assert "opinion" in result.deflection_reason.lower()

    def test_generate_returns_generation_result(self, sample_retrieval_results):
        gen = self._make_generator()

        result = gen.generate(
            "How do transformers use attention mechanisms?",
            context=sample_retrieval_results,
        )

        assert isinstance(result, GenerationResult)
        assert isinstance(result.scores, list)
        assert len(result.results) == len(sample_retrieval_results)

    def test_generate_preserves_context_in_result(self, sample_retrieval_results):
        gen = self._make_generator()

        result = gen.generate(
            "How do transformers use attention?",
            context=sample_retrieval_results,
        )

        # Whether deflected or not, the results and scores should be populated
        assert result.results is sample_retrieval_results or len(result.results) == len(sample_retrieval_results)
        assert len(result.scores) == len(sample_retrieval_results)
