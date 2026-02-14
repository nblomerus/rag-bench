"""
Unit tests for rag_bench.core.retriever module.

Tests cover:
- BM25: tokenization, indexing, query
- CrossEncoderReranker: model loading, reranking strategies
- HybridRetriever: initialization, dense retrieval, fusion, full pipeline
- Edge cases: empty inputs, missing data, error scenarios
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_bench.core.retriever import BM25, CrossEncoderReranker, HybridRetriever

# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_chunks():
    """Sample chunks for indexing tests."""
    return [
        {
            "chunk_id": "chunk_001",
            "text": "Deep learning and neural networks are fundamental to AI.",
            "metadata": {"title": "Paper 1", "section": "intro"},
        },
        {
            "chunk_id": "chunk_002",
            "text": "Transformers use attention mechanisms for sequence modeling.",
            "metadata": {"title": "Paper 2", "section": "methods"},
        },
        {
            "chunk_id": "chunk_003",
            "text": "BERT is a transformer-based language model using masked attention.",
            "metadata": {"title": "Paper 3", "section": "architecture"},
        },
        {
            "chunk_id": "chunk_004",
            "text": "Reinforcement learning uses rewards to train agents.",
            "metadata": {"title": "Paper 4", "section": "intro"},
        },
    ]


@pytest.fixture
def sample_candidates():
    """Sample candidates for reranking tests."""
    return [
        {
            "chunk_id": "chunk_001",
            "text": "Attention mechanisms compute weighted sums over input sequences.",
            "score": 0.8,
            "metadata": {},
        },
        {
            "chunk_id": "chunk_002",
            "text": "Neural networks consist of layers of connected neurons.",
            "score": 0.6,
            "metadata": {},
        },
        {
            "chunk_id": "chunk_003",
            "text": "The transformer architecture revolutionized NLP with self-attention.",
            "score": 0.7,
            "metadata": {},
        },
    ]


# ══════════════════════════════════════════════════════════════════════════════
# BM25 Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestBM25:
    """Tests for BM25 class."""

    def test_init_default_parameters(self):
        """Test BM25 initialization with default parameters."""
        bm25 = BM25()
        assert bm25.k1 == 1.5
        assert bm25.b == 0.75
        assert bm25.doc_count == 0
        assert len(bm25.chunk_ids) == 0

    def test_init_custom_parameters(self):
        """Test BM25 initialization with custom parameters."""
        bm25 = BM25(k1=2.0, b=0.5)
        assert bm25.k1 == 2.0
        assert bm25.b == 0.5

    def test_tokenize_basic(self):
        """Test basic tokenization."""
        bm25 = BM25()
        tokens = bm25._tokenize("Hello world test")
        assert tokens == ["hello", "world", "test"]

    def test_tokenize_with_punctuation(self):
        """Test tokenization handles punctuation."""
        bm25 = BM25()
        tokens = bm25._tokenize("Hello, world! Test.")
        assert tokens == ["hello", "world", "test"]

    def test_tokenize_with_numbers(self):
        """Test tokenization includes alphanumeric tokens (stopwords filtered)."""
        bm25 = BM25()
        tokens = bm25._tokenize("BERT-128 model v2.5")
        assert "bert" in tokens
        assert "128" in tokens
        # "model" is a stopword at scale, filtered out
        assert "model" not in tokens
        assert "v2" in tokens
        # Single-char tokens ("5") are filtered
        assert "5" not in tokens

    def test_tokenize_contractions(self):
        """Test tokenization handles contractions."""
        bm25 = BM25()
        tokens = bm25._tokenize("They're won't can't")
        assert "they're" in tokens or "they" in tokens

    def test_tokenize_empty_string(self):
        """Test tokenization of empty string."""
        bm25 = BM25()
        tokens = bm25._tokenize("")
        assert tokens == []

    def test_index_basic(self, sample_chunks):
        """Test basic indexing."""
        bm25 = BM25()
        bm25.index(sample_chunks)

        assert bm25.doc_count == len(sample_chunks)
        assert len(bm25.chunk_ids) == len(sample_chunks)
        assert len(bm25.doc_len) == len(sample_chunks)
        assert len(bm25.doc_freqs) == len(sample_chunks)
        assert bm25.avgdl > 0
        assert len(bm25.idf) > 0

    def test_index_empty_list(self):
        """Test indexing empty list."""
        bm25 = BM25()
        bm25.index([])

        assert bm25.doc_count == 0
        assert bm25.avgdl == 0.0

    def test_index_chunk_ids_stored(self, sample_chunks):
        """Test that chunk IDs are correctly stored."""
        bm25 = BM25()
        bm25.index(sample_chunks)

        expected_ids = [c["chunk_id"] for c in sample_chunks]
        assert bm25.chunk_ids == expected_ids

    def test_index_computes_idf(self, sample_chunks):
        """Test that IDF scores are computed."""
        bm25 = BM25()
        bm25.index(sample_chunks)

        # Check some common terms have IDF scores
        assert len(bm25.idf) > 0
        # All IDF scores should be positive
        for _term, idf in bm25.idf.items():
            assert idf >= 0

    def test_query_basic(self, sample_chunks):
        """Test basic query."""
        bm25 = BM25()
        bm25.index(sample_chunks)

        results = bm25.query("attention transformers", top_k=2)

        assert len(results) <= 2
        assert all("chunk_id" in r for r in results)
        assert all("text" in r for r in results)
        assert all("score" in r for r in results)
        assert all(r["source"] == "bm25" for r in results)

    def test_query_returns_sorted_results(self, sample_chunks):
        """Test that query results are sorted by score."""
        bm25 = BM25()
        bm25.index(sample_chunks)

        results = bm25.query("transformer attention", top_k=4)

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_top_k_limit(self, sample_chunks):
        """Test that top_k parameter limits results."""
        bm25 = BM25()
        bm25.index(sample_chunks)

        results = bm25.query("learning", top_k=2)

        assert len(results) <= 2

    def test_query_no_matches(self, sample_chunks):
        """Test query with no matching terms."""
        bm25 = BM25()
        bm25.index(sample_chunks)

        results = bm25.query("xyzabc123notinanytext", top_k=10)

        # Should return empty or zero-scored results
        assert len(results) == 0 or all(r["score"] == 0 for r in results)

    def test_query_filters_zero_scores(self, sample_chunks):
        """Test that zero-score results are filtered."""
        bm25 = BM25()
        bm25.index(sample_chunks)

        results = bm25.query("nonexistent", top_k=10)

        # All returned results should have positive scores
        assert all(r["score"] > 0 for r in results)

    def test_query_empty_index(self):
        """Test query on empty index."""
        bm25 = BM25()
        bm25.index([])

        results = bm25.query("test query")

        assert results == []

    def test_query_metadata_included(self, sample_chunks):
        """Test that metadata is included in results."""
        bm25 = BM25()
        bm25.index(sample_chunks)

        results = bm25.query("neural", top_k=1)

        if results:
            assert "metadata" in results[0]

    def test_bm25_scoring_formula(self):
        """Test BM25 scoring is reasonable."""
        chunks = [
            {"chunk_id": "1", "text": "cat cat cat", "metadata": {}},
            {"chunk_id": "2", "text": "dog dog dog", "metadata": {}},
            {"chunk_id": "3", "text": "cat dog", "metadata": {}},
        ]
        bm25 = BM25()
        bm25.index(chunks)

        # Query for "cat" should score chunk 1 highest
        results = bm25.query("cat", top_k=3)

        assert results[0]["chunk_id"] == "1"  # Most "cat" occurrences


# ══════════════════════════════════════════════════════════════════════════════
# CrossEncoderReranker Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker class."""

    @patch("rag_bench.core.retriever.CrossEncoder")
    def test_init_with_model(self, mock_ce_class):
        """Test initialization with successful model loading."""
        mock_model = MagicMock()
        mock_ce_class.return_value = mock_model

        reranker = CrossEncoderReranker()

        assert reranker.model == mock_model
        mock_ce_class.assert_called_once()

    @patch("rag_bench.core.retriever.CrossEncoder")
    def test_init_fallback_on_error(self, mock_ce_class):
        """Test fallback when model loading fails."""
        mock_ce_class.side_effect = ImportError("No module")

        reranker = CrossEncoderReranker()

        assert reranker.model is None

    @patch("rag_bench.core.retriever.CrossEncoder")
    def test_rerank_with_model(self, mock_ce_class, sample_candidates):
        """Test reranking with cross-encoder model."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.3, 0.7])
        mock_ce_class.return_value = mock_model

        reranker = CrossEncoderReranker()
        results = reranker.rerank("test question", sample_candidates, top_k=2)

        assert len(results) == 2
        # Should be sorted by rerank_score
        assert results[0]["rerank_score"] > results[1]["rerank_score"]
        mock_model.predict.assert_called_once()

    @patch("rag_bench.core.retriever.CrossEncoder")
    def test_rerank_with_keywords(self, mock_ce_class, sample_candidates):
        """Test keyword-based fallback reranking."""
        mock_ce_class.side_effect = ImportError("No module")

        reranker = CrossEncoderReranker()
        results = reranker.rerank("attention transformer", sample_candidates, top_k=3)

        assert len(results) <= 3
        # Should have rerank_score added
        assert all("rerank_score" in r for r in results)
        # Should be sorted
        scores = [r["rerank_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    @patch("rag_bench.core.retriever.CrossEncoder")
    def test_rerank_empty_candidates(self, mock_ce_class):
        """Test reranking with empty candidates."""
        mock_model = MagicMock()
        mock_ce_class.return_value = mock_model

        reranker = CrossEncoderReranker()
        results = reranker.rerank("test", [], top_k=5)

        assert results == []
        mock_model.predict.assert_not_called()

    @patch("rag_bench.core.retriever.CrossEncoder")
    def test_rerank_respects_top_k(self, mock_ce_class, sample_candidates):
        """Test that top_k parameter is respected."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.8, 0.7])
        mock_ce_class.return_value = mock_model

        reranker = CrossEncoderReranker()
        results = reranker.rerank("test", sample_candidates, top_k=1)

        assert len(results) == 1

    @patch("rag_bench.core.retriever.CrossEncoder")
    def test_rerank_keyword_overlap_scoring(self, mock_ce_class):
        """Test keyword overlap scoring in fallback mode."""
        mock_ce_class.side_effect = ImportError("No module")

        candidates = [
            {"chunk_id": "1", "text": "attention mechanism in transformers", "score": 0.5},
            {"chunk_id": "2", "text": "convolutional neural networks", "score": 0.5},
        ]

        reranker = CrossEncoderReranker()
        results = reranker.rerank("attention transformers", candidates, top_k=2)

        # First result should have higher overlap
        assert results[0]["chunk_id"] == "1"

    @patch("rag_bench.core.retriever.CrossEncoder")
    def test_rerank_phrase_bonus(self, mock_ce_class):
        """Test that phrase matching gets bonus in fallback mode."""
        mock_ce_class.side_effect = ImportError("No module")

        candidates = [
            {"chunk_id": "1", "text": "the quick brown fox", "score": 0.5},
            {"chunk_id": "2", "text": "quick fox is brown", "score": 0.5},
        ]

        reranker = CrossEncoderReranker()
        results = reranker.rerank("quick brown fox", candidates, top_k=2)

        # First candidate has exact phrase, should score higher
        assert results[0]["rerank_score"] > results[1]["rerank_score"]


# ══════════════════════════════════════════════════════════════════════════════
# HybridRetriever Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestHybridRetriever:
    """Tests for HybridRetriever class."""

    @patch("rag_bench.core.retriever.CrossEncoderReranker")
    @patch("rag_bench.core.retriever.chromadb.PersistentClient")
    @patch("rag_bench.core.retriever._load_embedding_model")
    def test_init_basic(self, mock_load_model, mock_chroma, mock_reranker_class):
        """Test basic initialization."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1] * 768)
        mock_load_model.return_value = mock_model

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        mock_reranker = MagicMock()
        mock_reranker_class.return_value = mock_reranker

        retriever = HybridRetriever()

        assert retriever.embed_model == mock_model
        assert retriever.chroma_client == mock_client
        assert retriever.collection == mock_collection
        assert isinstance(retriever.bm25, BM25)

    @patch("rag_bench.core.retriever.CrossEncoderReranker")
    @patch("rag_bench.core.retriever.chromadb.PersistentClient")
    @patch("rag_bench.core.retriever._load_embedding_model")
    def test_init_builds_bm25_index(self, mock_load_model, mock_chroma, mock_reranker_class, sample_chunks):
        """Test that BM25 index is built from ChromaDB."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = len(sample_chunks)
        mock_collection.get.return_value = {
            "ids": [c["chunk_id"] for c in sample_chunks],
            "documents": [c["text"] for c in sample_chunks],
            "metadatas": [c["metadata"] for c in sample_chunks],
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        mock_reranker_class.return_value = MagicMock()

        retriever = HybridRetriever()

        assert retriever.bm25.doc_count == len(sample_chunks)

    @patch("rag_bench.core.retriever.CrossEncoderReranker")
    @patch("rag_bench.core.retriever.chromadb.PersistentClient")
    @patch("rag_bench.core.retriever._load_embedding_model")
    def test_dense_query(self, mock_load_model, mock_chroma, mock_reranker_class):
        """Test dense retrieval query."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1] * 768)
        mock_load_model.return_value = mock_model

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 3
        mock_collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        mock_collection.query.return_value = {
            "ids": [["chunk_1", "chunk_2"]],
            "documents": [["text 1", "text 2"]],
            "metadatas": [[{"title": "Paper 1"}, {"title": "Paper 2"}]],
            "distances": [[0.2, 0.4]],
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        mock_reranker_class.return_value = MagicMock()

        retriever = HybridRetriever()
        results = retriever._dense_query("test query", top_k=2)

        assert len(results) == 2
        assert results[0]["source"] == "dense"
        # Check score conversion (1 - distance)
        assert results[0]["score"] == 1.0 - 0.2

    @patch("rag_bench.core.retriever.CrossEncoderReranker")
    @patch("rag_bench.core.retriever.chromadb.PersistentClient")
    @patch("rag_bench.core.retriever._load_embedding_model")
    def test_reciprocal_rank_fusion(self, mock_load_model, mock_chroma, mock_reranker_class):
        """Test reciprocal rank fusion."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        mock_reranker_class.return_value = MagicMock()

        retriever = HybridRetriever(bm25_weight=0.5, dense_weight=0.5)

        bm25_results = [
            {"chunk_id": "chunk_1", "text": "text 1", "score": 10.0, "metadata": {}},
            {"chunk_id": "chunk_2", "text": "text 2", "score": 8.0, "metadata": {}},
        ]
        dense_results = [
            {"chunk_id": "chunk_2", "text": "text 2", "score": 0.9, "metadata": {}},
            {"chunk_id": "chunk_3", "text": "text 3", "score": 0.8, "metadata": {}},
        ]

        fused = retriever._reciprocal_rank_fusion(bm25_results, dense_results)

        # chunk_2 appears in both, should have highest RRF score
        assert fused[0]["chunk_id"] == "chunk_2"
        # All chunks should have rrf_score
        assert all("rrf_score" in r for r in fused)

    @patch("rag_bench.core.retriever.CrossEncoderReranker")
    @patch("rag_bench.core.retriever.chromadb.PersistentClient")
    @patch("rag_bench.core.retriever._load_embedding_model")
    def test_query_full_pipeline(self, mock_load_model, mock_chroma, mock_reranker_class):
        """Test full query pipeline."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1] * 768)
        mock_load_model.return_value = mock_model

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 2
        mock_collection.get.return_value = {
            "ids": ["chunk_1", "chunk_2"],
            "documents": ["text about attention", "text about transformers"],
            "metadatas": [{"title": "P1"}, {"title": "P2"}],
        }
        mock_collection.query.return_value = {
            "ids": [["chunk_1"]],
            "documents": [["text about attention"]],
            "metadatas": [[{"title": "P1"}]],
            "distances": [[0.1]],
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [{"chunk_id": "chunk_1", "text": "text", "rerank_score": 0.9, "metadata": {}}]
        mock_reranker_class.return_value = mock_reranker

        retriever = HybridRetriever()
        results = retriever.query("attention", top_k=1)

        assert len(results) == 1
        assert "score" in results[0]
        mock_reranker.rerank.assert_called_once()

    @patch("rag_bench.core.retriever.CrossEncoderReranker")
    @patch("rag_bench.core.retriever.chromadb.PersistentClient")
    @patch("rag_bench.core.retriever._load_embedding_model")
    def test_query_applies_threshold(self, mock_load_model, mock_chroma, mock_reranker_class):
        """Test that relevance threshold filters results."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1] * 768)
        mock_load_model.return_value = mock_model

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {"chunk_id": "1", "text": "t", "rerank_score": 0.8, "metadata": {}},
            {"chunk_id": "2", "text": "t", "rerank_score": 0.2, "metadata": {}},
        ]
        mock_reranker_class.return_value = mock_reranker

        retriever = HybridRetriever()
        results = retriever.query("test", top_k=10, relevance_threshold=0.5)

        # Only results >= 0.5 should be returned
        assert len(results) == 1
        assert results[0]["score"] >= 0.5

    @patch("rag_bench.core.retriever.CrossEncoderReranker")
    @patch("rag_bench.core.retriever.chromadb.PersistentClient")
    @patch("rag_bench.core.retriever._load_embedding_model")
    def test_query_with_citations(self, mock_load_model, mock_chroma, mock_reranker_class):
        """Test query_with_citations method."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1] * 768)
        mock_load_model.return_value = mock_model

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {
                "chunk_id": "1",
                "text": "text",
                "rerank_score": 0.8,
                "metadata": {"source_display": "Paper X", "section": "intro"},
            }
        ]
        mock_reranker_class.return_value = mock_reranker

        retriever = HybridRetriever()
        output = retriever.query_with_citations("test", top_k=1)

        assert "question" in output
        assert "results" in output
        assert "sources" in output
        assert "is_relevant" in output
        assert len(output["sources"]) > 0

    @patch("rag_bench.core.retriever.CrossEncoderReranker")
    @patch("rag_bench.core.retriever.chromadb.PersistentClient")
    @patch("rag_bench.core.retriever._load_embedding_model")
    def test_empty_collection(self, mock_load_model, mock_chroma, mock_reranker_class):
        """Test behavior with empty ChromaDB collection."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        mock_reranker_class.return_value = MagicMock()

        # Should not raise error
        retriever = HybridRetriever()
        assert retriever.bm25.doc_count == 0


# Additional tests for coverage
class TestRetrieverPrintResults:
    """Test print_results method for coverage."""

    @patch("rag_bench.core.retriever.CrossEncoderReranker")
    @patch("rag_bench.core.retriever.chromadb.PersistentClient")
    @patch("rag_bench.core.retriever._load_embedding_model")
    def test_print_results_output(self, mock_load_model, mock_chroma, mock_reranker_class):
        """Test print_results prints correctly with relevant results."""
        import sys
        from io import StringIO

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1] * 768)
        mock_load_model.return_value = mock_model

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 1
        mock_collection.get.return_value = {
            "ids": ["chunk_1"],
            "documents": ["test document"],
            "metadatas": [{"title": "Paper 1", "section": "intro"}],
        }
        mock_collection.query.return_value = {
            "ids": [["chunk_1"]],
            "documents": [["test document"]],
            "metadatas": [[{"title": "Paper 1", "section": "intro"}]],
            "distances": [[0.1]],
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {
                "chunk_id": "chunk_1",
                "text": "This is a test document with relevant content",
                "rerank_score": 0.95,
                "metadata": {"title": "Paper 1", "section": "intro", "source_display": "Paper 1"},
                "score": 0.9,
                "sources": ["dense"],
                "bm25_score": 0.8,
                "dense_score": 0.9,
                "rrf_score": 0.95,
            }
        ]
        mock_reranker_class.return_value = mock_reranker

        retriever = HybridRetriever()

        # Capture output
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        result = retriever.print_results("test question", top_k=1)

        output_str = captured_output.getvalue()
        sys.stdout = old_stdout

        # Verify output contains expected strings
        assert "test question" in output_str
        assert "Paper 1" in output_str
        assert result is not None
        assert result["is_relevant"]

    @patch("rag_bench.core.retriever.CrossEncoderReranker")
    @patch("rag_bench.core.retriever.chromadb.PersistentClient")
    @patch("rag_bench.core.retriever._load_embedding_model")
    def test_print_results_not_relevant(self, mock_load_model, mock_chroma, mock_reranker_class):
        """Test print_results with non-relevant (low score) results."""
        import sys
        from io import StringIO

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1] * 768)
        mock_load_model.return_value = mock_model

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = []
        mock_reranker_class.return_value = mock_reranker

        retriever = HybridRetriever()

        # Capture output
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        result = retriever.print_results("unrelated question", top_k=5)

        output_str = captured_output.getvalue()
        sys.stdout = old_stdout

        # Verify output indicates no relevant results
        assert "No sufficiently relevant results" in output_str
        assert result is not None
        assert not result["is_relevant"]


# ══════════════════════════════════════════════════════════════════════════════
# Additional Coverage Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestBM25Coverage:
    """Additional tests for BM25 coverage."""

    def test_tokenize_with_none_text(self):
        """Test _tokenize handles None input."""
        bm25 = BM25(k1=1.5, b=0.75)
        result = bm25._tokenize(None)
        assert result == []

    @patch("rag_bench.core.retriever.Path")
    def test_load_from_cache_failure(self, mock_path_class):
        """Test BM25 cache load failure."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.open.side_effect = Exception("Read error")

        bm25 = BM25()
        result = bm25.load_from_cache(mock_path)

        assert result is False


class TestRetrieverBranchCoverage:
    """Additional tests for branch coverage in retriever module."""

    @patch("rag_bench.core.retriever.chromadb.PersistentClient")
    @patch("rag_bench.core.retriever.CrossEncoderReranker")
    @patch("rag_bench.core.retriever._load_embedding_model")
    def test_build_bm25_cache_save_failure(self, mock_load_model, mock_reranker_class, mock_chroma_class):
        """Test BM25 build when cache save fails."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_load_model.return_value = mock_model

        mock_reranker = MagicMock()
        mock_reranker_class.return_value = mock_reranker

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 10
        mock_collection.get.return_value = {
            "ids": [f"chunk_{i}" for i in range(10)],
            "documents": [f"text {i}" for i in range(10)],
            "metadatas": [{"doc_id": "doc1"} for _ in range(10)],
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma_class.return_value = mock_client

        # Create retriever
        retriever = HybridRetriever()

        # Mock cache save to fail
        with patch.object(retriever.bm25, "save_to_cache") as mock_save:
            mock_save.side_effect = Exception("Write error")

            # Build BM25 - should handle save failure gracefully
            retriever._build_bm25_index()

            # Should complete without raising exception
            assert retriever.bm25.doc_count == 10
