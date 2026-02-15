"""Additional tests to improve retriever coverage."""

from unittest.mock import MagicMock, patch

import numpy as np

from rag_bench.core.retriever import HybridRetriever


@patch("rag_bench.core.retriever.chromadb.PersistentClient")
@patch("rag_bench.core.retriever.CrossEncoderReranker")
@patch("rag_bench.core.retriever._load_embedding_model")
def test_fetch_arxiv_chunks_success(mock_load_model, mock_reranker_class, mock_chroma_class):
    """Test fetch_paper_chunks successfully retrieves chunks."""
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 768
    mock_load_model.return_value = mock_model

    mock_reranker = MagicMock()
    mock_reranker_class.return_value = mock_reranker

    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0

    # Mock get for arxiv chunk fetching
    mock_collection.get.return_value = {
        "ids": ["chunk_1", "chunk_2"],
        "documents": ["Introduction text", "Methods text"],
        "metadatas": [
            {"arxiv_id": "1706.03762", "section": "introduction"},
            {"arxiv_id": "1706.03762", "section": "methods"},
        ],
    }

    mock_client.get_or_create_collection.return_value = mock_collection
    mock_chroma_class.return_value = mock_client

    retriever = HybridRetriever()
    chunks = retriever.fetch_paper_chunks("1706.03762", max_chunks=2)

    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == "chunk_1"
    assert chunks[0]["source"] == "injection"


@patch("rag_bench.core.retriever.chromadb.PersistentClient")
@patch("rag_bench.core.retriever.CrossEncoderReranker")
@patch("rag_bench.core.retriever._load_embedding_model")
def test_fetch_arxiv_chunks_exception(mock_load_model, mock_reranker_class, mock_chroma_class):
    """Test fetch_paper_chunks handles exceptions gracefully."""
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 768
    mock_load_model.return_value = mock_model

    mock_reranker = MagicMock()
    mock_reranker_class.return_value = mock_reranker

    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0

    # Make get raise an exception
    mock_collection.get.side_effect = Exception("Database error")

    mock_client.get_or_create_collection.return_value = mock_collection
    mock_chroma_class.return_value = mock_client

    retriever = HybridRetriever()
    chunks = retriever.fetch_paper_chunks("1706.03762")

    # Should return empty list on exception
    assert chunks == []


@patch("rag_bench.core.retriever.chromadb.PersistentClient")
@patch("rag_bench.core.retriever.CrossEncoderReranker")
@patch("rag_bench.core.retriever._load_embedding_model")
def test_fetch_arxiv_chunks_no_results(mock_load_model, mock_reranker_class, mock_chroma_class):
    """Test fetch_paper_chunks when no chunks found."""
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 768
    mock_load_model.return_value = mock_model

    mock_reranker = MagicMock()
    mock_reranker_class.return_value = mock_reranker

    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0

    # Return empty results
    mock_collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}

    mock_client.get_or_create_collection.return_value = mock_collection
    mock_chroma_class.return_value = mock_client

    retriever = HybridRetriever()
    chunks = retriever.fetch_paper_chunks("9999.99999")

    # Should return empty list
    assert chunks == []


@patch("rag_bench.core.retriever.chromadb.PersistentClient")
@patch("rag_bench.core.retriever.CrossEncoderReranker")
@patch("rag_bench.core.retriever._load_embedding_model")
def test_query_with_inject_chunks(mock_load_model, mock_reranker_class, mock_chroma_class):
    """Test query with inject_chunks parameter."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([0.1] * 768)
    mock_model.get_sentence_embedding_dimension.return_value = 768
    mock_load_model.return_value = mock_model

    # Injected chunks that should be preserved
    inject_chunks = [
        {
            "chunk_id": "injected_1",
            "text": "Injected foundational text",
            "metadata": {"arxiv_id": "1706.03762"},
            "score": 0.0,
            "source": "injection",
            "rrf_score": 0.0,
        }
    ]

    mock_reranker = MagicMock()
    # Reranker returns all chunks including injected
    mock_reranker.rerank.return_value = [
        {"chunk_id": "chunk_1", "text": "result 1", "rerank_score": 0.9, "metadata": {}, "rrf_score": 1.0},
        {
            "chunk_id": "injected_1",
            "text": "injected",
            "rerank_score": 0.8,
            "metadata": {},
            "source": "injection",
            "rrf_score": 0.0,
        },
    ]
    mock_reranker_class.return_value = mock_reranker

    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.count.return_value = 10

    # BM25 get for building index
    mock_collection.get.return_value = {
        "ids": ["chunk_1"],
        "documents": ["text 1"],
        "metadatas": [{"doc_id": "doc1"}],
    }

    # Dense query results
    mock_collection.query.return_value = {
        "ids": [["chunk_1"]],
        "documents": [["result 1"]],
        "metadatas": [[{"doc_id": "doc1"}]],
        "distances": [[0.1]],
    }

    mock_client.get_or_create_collection.return_value = mock_collection
    mock_chroma_class.return_value = mock_client

    retriever = HybridRetriever()

    results = retriever.query("attention", top_k=2, inject_chunks=inject_chunks)

    # Should include both regular and injected chunks
    chunk_ids = [r["chunk_id"] for r in results]
    assert "injected_1" in chunk_ids


@patch("rag_bench.core.retriever.chromadb.PersistentClient")
@patch("rag_bench.core.retriever.CrossEncoderReranker")
@patch("rag_bench.core.retriever._load_embedding_model")
def test_query_inject_chunks_survive_threshold(mock_load_model, mock_reranker_class, mock_chroma_class):
    """Test that injected chunks survive relevance threshold filtering."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([0.1] * 768)
    mock_model.get_sentence_embedding_dimension.return_value = 768
    mock_load_model.return_value = mock_model

    mock_reranker = MagicMock()
    # Return results with low scores
    mock_reranker.rerank.return_value = [
        {"chunk_id": "injected_1", "text": "injected", "rerank_score": 0.3, "metadata": {}, "source": "injection"},
    ]
    mock_reranker_class.return_value = mock_reranker

    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.count.return_value = 10

    mock_collection.get.return_value = {
        "ids": ["chunk_1"],
        "documents": ["text 1"],
        "metadatas": [{"doc_id": "doc1"}],
    }

    mock_collection.query.return_value = {
        "ids": [["chunk_1"]],
        "documents": [["result 1"]],
        "metadatas": [[{"doc_id": "doc1"}]],
        "distances": [[0.1]],
    }

    mock_client.get_or_create_collection.return_value = mock_collection
    mock_chroma_class.return_value = mock_client

    retriever = HybridRetriever()

    inject_chunks = [
        {
            "chunk_id": "injected_1",
            "text": "Injected text",
            "metadata": {},
            "score": 0.0,
            "source": "injection",
        }
    ]

    # Use high threshold that would normally filter out the injected chunk
    results = retriever.query("attention", top_k=5, relevance_threshold=0.8, inject_chunks=inject_chunks)

    # Injected chunk should still be present despite low score
    chunk_ids = [r["chunk_id"] for r in results]
    assert "injected_1" in chunk_ids
