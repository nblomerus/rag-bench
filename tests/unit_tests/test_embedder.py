"""
Unit tests for rag_bench.core.embedder module.

Tests cover:
- TfidfFallbackEmbedder: tokenization, encoding, normalization
- _load_embedding_model: model loading and fallback behavior
- Embedder: initialization, embedding, indexing, stats
- Edge cases: empty inputs, invalid parameters, error scenarios
"""

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_bench.core.embedder import Embedder, TfidfFallbackEmbedder, _load_embedding_model

# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_texts():
    """Sample text data for testing."""
    return [
        "This is a test sentence.",
        "Another test sentence for embedding.",
        "Machine learning is fascinating.",
    ]


@pytest.fixture
def sample_chunks():
    """Sample chunk dictionaries for indexing tests."""
    return [
        {
            "chunk_id": "chunk_001",
            "text": "Introduction to machine learning.",
            "doc_id": "paper_1",
            "section": "introduction",
            "metadata": {"page": 1, "author": "Test Author"},
        },
        {
            "chunk_id": "chunk_002",
            "text": "Deep learning architectures.",
            "doc_id": "paper_1",
            "section": "methods",
            "metadata": {"page": 2},
        },
        {
            "chunk_id": "chunk_003",
            "text": "Experimental results and analysis.",
            "doc_id": "paper_2",
            "section": "results",
            "metadata": {"page": 5},
        },
    ]


@pytest.fixture
def mock_embeddings():
    """Mock embedding vectors (768 dimensions matching BGE model)."""
    return np.array([[0.1] * 768, [0.2] * 768, [0.3] * 768])


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer model."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1] * 768])
    mock_model.get_sentence_embedding_dimension.return_value = 768
    return mock_model


@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client with collection."""
    mock_client = MagicMock()
    mock_collection = MagicMock()

    # Default mock behavior
    mock_collection.count.return_value = 0
    mock_collection.get.return_value = {"ids": [], "metadatas": []}
    mock_collection.add.return_value = None

    mock_client.get_or_create_collection.return_value = mock_collection

    # Return both client and collection for easy access
    return mock_client, mock_collection


# ══════════════════════════════════════════════════════════════════════════════
# TfidfFallbackEmbedder Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestTfidfFallbackEmbedder:
    """Tests for TfidfFallbackEmbedder class."""

    def test_init_default_dimension(self):
        """Test initialization with default dimension."""
        embedder = TfidfFallbackEmbedder()
        assert embedder.dim == 768

    def test_init_custom_dimension(self):
        """Test initialization with custom dimension."""
        embedder = TfidfFallbackEmbedder(dim=512)
        assert embedder.dim == 512

    def test_get_sentence_embedding_dimension(self):
        """Test get_sentence_embedding_dimension returns correct value."""
        embedder = TfidfFallbackEmbedder(dim=1024)
        assert embedder.get_sentence_embedding_dimension() == 1024

    def test_tokenize_basic(self):
        """Test basic tokenization."""
        embedder = TfidfFallbackEmbedder()
        tokens = embedder._tokenize("Hello world test")
        assert tokens == ["hello", "world", "test"]

    def test_tokenize_special_characters(self):
        """Test tokenization removes special characters."""
        embedder = TfidfFallbackEmbedder()
        tokens = embedder._tokenize("Hello, world! Test... #123")
        assert tokens == ["hello", "world", "test", "123"]

    def test_tokenize_filters_short_words(self):
        """Test tokenization filters single-character words."""
        embedder = TfidfFallbackEmbedder()
        tokens = embedder._tokenize("a big test i o u")
        assert tokens == ["big", "test"]

    def test_tokenize_empty_string(self):
        """Test tokenization of empty string."""
        embedder = TfidfFallbackEmbedder()
        tokens = embedder._tokenize("")
        assert tokens == []

    def test_tokenize_only_special_chars(self):
        """Test tokenization of string with only special characters."""
        embedder = TfidfFallbackEmbedder()
        tokens = embedder._tokenize("!@#$%^&*()")
        assert tokens == []

    def test_encode_single_string(self):
        """Test encoding a single string (converted to list)."""
        embedder = TfidfFallbackEmbedder(dim=100)
        result = embedder.encode("test sentence")

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 100)

    def test_encode_list_of_strings(self):
        """Test encoding multiple strings."""
        embedder = TfidfFallbackEmbedder(dim=100)
        texts = ["first text", "second text", "third text"]
        result = embedder.encode(texts)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 100)

    def test_encode_normalization_enabled(self):
        """Test that normalization produces unit vectors."""
        embedder = TfidfFallbackEmbedder(dim=100)
        result = embedder.encode(["test sentence"], normalize_embeddings=True)

        # Check L2 norm is approximately 1
        norm = np.linalg.norm(result[0])
        assert math.isclose(norm, 1.0, rel_tol=1e-6)

    def test_encode_normalization_disabled(self):
        """Test encoding without normalization."""
        embedder = TfidfFallbackEmbedder(dim=100)
        result = embedder.encode(["test sentence"], normalize_embeddings=False)

        # Without normalization, norm should not necessarily be 1
        norm = np.linalg.norm(result[0])
        # Just verify it's non-zero
        assert norm > 0

    def test_encode_empty_list(self):
        """Test encoding empty list of texts."""
        embedder = TfidfFallbackEmbedder(dim=100)
        result = embedder.encode([])

        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_encode_empty_strings(self):
        """Test encoding list with empty strings."""
        embedder = TfidfFallbackEmbedder(dim=100)
        result = embedder.encode(["", "", ""])

        assert result.shape == (3, 100)
        # Empty text should produce zero or normalized zero vector
        for embedding in result:
            # All zeros after normalization should be handled
            assert not np.any(np.isnan(embedding))

    def test_encode_hashing_consistency(self):
        """Test that same text produces same embedding."""
        embedder = TfidfFallbackEmbedder(dim=100)
        text = "consistent test text"

        result1 = embedder.encode([text])
        result2 = embedder.encode([text])

        np.testing.assert_array_equal(result1, result2)

    def test_encode_different_texts_different_embeddings(self):
        """Test that different texts produce different embeddings."""
        embedder = TfidfFallbackEmbedder(dim=100)

        result1 = embedder.encode(["machine learning"])
        result2 = embedder.encode(["deep neural networks"])

        # Embeddings should be different
        assert not np.allclose(result1, result2)

    def test_encode_progress_bar_parameter(self):
        """Test that show_progress_bar parameter is accepted (even if not used)."""
        embedder = TfidfFallbackEmbedder()
        # Should not raise error
        result = embedder.encode(["test"], show_progress_bar=True)
        assert result.shape == (1, 768)


# ══════════════════════════════════════════════════════════════════════════════
# _load_embedding_model Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestLoadEmbeddingModel:
    """Tests for _load_embedding_model function."""

    @patch("sentence_transformers.SentenceTransformer")
    def test_successful_model_loading(self, mock_st_class):
        """Test successful loading of SentenceTransformer model."""
        mock_model = MagicMock()
        mock_st_class.return_value = mock_model

        result = _load_embedding_model("test-model")

        assert result == mock_model
        mock_st_class.assert_called_once_with("test-model")

    @patch("sentence_transformers.SentenceTransformer")
    def test_fallback_on_import_error(self, mock_st_class):
        """Test fallback to TfidfFallbackEmbedder on ImportError."""
        mock_st_class.side_effect = ImportError("No module named 'sentence_transformers'")

        result = _load_embedding_model("test-model")

        assert isinstance(result, TfidfFallbackEmbedder)
        assert result.dim == 768

    @patch("sentence_transformers.SentenceTransformer")
    def test_fallback_on_os_error(self, mock_st_class):
        """Test fallback on OSError (e.g., network/download failure)."""
        mock_st_class.side_effect = OSError("Connection failed")

        result = _load_embedding_model("test-model")

        assert isinstance(result, TfidfFallbackEmbedder)

    @patch("sentence_transformers.SentenceTransformer")
    def test_fallback_on_generic_exception(self, mock_st_class):
        """Test fallback on any generic exception."""
        mock_st_class.side_effect = RuntimeError("Unexpected error")

        result = _load_embedding_model("test-model")

        assert isinstance(result, TfidfFallbackEmbedder)

    @patch("sentence_transformers.SentenceTransformer")
    @patch("rag_bench.core.embedder.logger")
    def test_logging_on_successful_load(self, mock_logger, mock_st_class):
        """Test that successful load logs info message."""
        mock_model = MagicMock()
        mock_st_class.return_value = mock_model

        _load_embedding_model("test-model")

        mock_logger.info.assert_called_once()
        assert "test-model" in str(mock_logger.info.call_args)

    @patch("sentence_transformers.SentenceTransformer")
    @patch("rag_bench.core.embedder.logger")
    def test_logging_on_fallback(self, mock_logger, mock_st_class):
        """Test that fallback logs warning message."""
        mock_st_class.side_effect = ImportError("Test error")

        _load_embedding_model("test-model")

        mock_logger.warning.assert_called_once()
        assert "falling" in str(mock_logger.warning.call_args).lower()


# ══════════════════════════════════════════════════════════════════════════════
# Embedder.__init__ Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestEmbedderInit:
    """Tests for Embedder initialization."""

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    def test_init_with_defaults(self, mock_load_model, mock_chroma):
        """Test initialization with default parameters."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_load_model.return_value = mock_model

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        embedder = Embedder()

        assert embedder.model == mock_model
        assert embedder.client == mock_client
        assert embedder.collection == mock_collection
        mock_load_model.assert_called_once_with("BAAI/bge-base-en-v1.5")

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    def test_init_with_custom_model(self, mock_load_model, mock_chroma):
        """Test initialization with custom model name."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_load_model.return_value = mock_model

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        Embedder(model_name="custom-model")

        mock_load_model.assert_called_once_with("custom-model")

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    def test_init_with_custom_chroma_path(self, mock_load_model, mock_chroma, tmp_path):
        """Test initialization with custom ChromaDB path."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_load_model.return_value = mock_model

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        custom_path = tmp_path / "custom_chroma"
        Embedder(chroma_path=str(custom_path))

        # Verify directory was created
        assert custom_path.exists()
        mock_chroma.assert_called_once_with(path=str(custom_path))

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    def test_init_with_custom_collection_name(self, mock_load_model, mock_chroma):
        """Test initialization with custom collection name."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_load_model.return_value = mock_model

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        Embedder(collection_name="my_collection")

        mock_client.get_or_create_collection.assert_called_once_with(
            name="my_collection",
            metadata={"hnsw:space": "cosine"},
        )

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    @pytest.mark.parametrize("metric", ["cosine", "l2", "ip"])
    def test_init_with_different_distance_metrics(self, mock_load_model, mock_chroma, metric):
        """Test initialization with different distance metrics."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_load_model.return_value = mock_model

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        Embedder(distance_metric=metric)

        mock_client.get_or_create_collection.assert_called_once_with(
            name="ai_ml_papers",
            metadata={"hnsw:space": metric},
        )

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    def test_init_creates_parent_directories(self, mock_load_model, mock_chroma, tmp_path):
        """Test that initialization creates parent directories for ChromaDB."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_load_model.return_value = mock_model

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        nested_path = tmp_path / "parent" / "child" / "chroma"
        Embedder(chroma_path=str(nested_path))

        # Verify nested directories were created
        assert nested_path.exists()


# ══════════════════════════════════════════════════════════════════════════════
# Embedder.embed_texts Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestEmbedderEmbedTexts:
    """Tests for Embedder.embed_texts method."""

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    def test_embed_texts_basic(self, mock_load_model, mock_chroma, sample_texts):
        """Test basic text embedding."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.array([[0.1] * 768] * len(sample_texts))
        mock_load_model.return_value = mock_model

        mock_client, mock_collection = MagicMock(), MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        embedder = Embedder()
        result = embedder.embed_texts(sample_texts, show_progress=False)

        assert isinstance(result, list)
        assert len(result) == len(sample_texts)
        assert all(len(emb) == 768 for emb in result)

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    def test_embed_texts_single_text(self, mock_load_model, mock_chroma):
        """Test embedding a single text."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.array([[0.5] * 768])
        mock_load_model.return_value = mock_model

        mock_client, mock_collection = MagicMock(), MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        embedder = Embedder()
        result = embedder.embed_texts(["single text"], show_progress=False)

        assert len(result) == 1
        assert len(result[0]) == 768

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    def test_embed_texts_empty_list(self, mock_load_model, mock_chroma):
        """Test embedding empty list of texts."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_load_model.return_value = mock_model

        mock_client, mock_collection = MagicMock(), MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        embedder = Embedder()
        result = embedder.embed_texts([], show_progress=False)

        assert result == []
        # encode should not be called for empty list
        mock_model.encode.assert_not_called()

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    def test_embed_texts_batch_size(self, mock_load_model, mock_chroma):
        """Test that batch_size parameter controls batching."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.array([[0.1] * 768])
        mock_load_model.return_value = mock_model

        mock_client, mock_collection = MagicMock(), MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        embedder = Embedder()
        texts = ["text1", "text2", "text3", "text4", "text5"]
        embedder.embed_texts(texts, batch_size=2, show_progress=False)

        # Should be called 3 times: ceil(5/2) = 3 batches
        assert mock_model.encode.call_count == 3

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    def test_embed_texts_normalization_enabled(self, mock_load_model, mock_chroma):
        """Test that normalization parameter is passed to model."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.array([[0.1] * 768])
        mock_load_model.return_value = mock_model

        mock_client, mock_collection = MagicMock(), MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        embedder = Embedder()
        embedder.embed_texts(["test"], normalize=True, show_progress=False)

        mock_model.encode.assert_called_once()
        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs["normalize_embeddings"] is True

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    def test_embed_texts_normalization_disabled(self, mock_load_model, mock_chroma):
        """Test that normalization can be disabled."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.array([[0.1] * 768])
        mock_load_model.return_value = mock_model

        mock_client, mock_collection = MagicMock(), MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        embedder = Embedder()
        embedder.embed_texts(["test"], normalize=False, show_progress=False)

        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs["normalize_embeddings"] is False

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    def test_embed_texts_show_progress_bar(self, mock_load_model, mock_chroma):
        """Test that show_progress parameter works."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.array([[0.1] * 768])
        mock_load_model.return_value = mock_model

        mock_client, mock_collection = MagicMock(), MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        embedder = Embedder()
        # Should not raise error with either value
        embedder.embed_texts(["test"], show_progress=True)
        embedder.embed_texts(["test"], show_progress=False)


# ══════════════════════════════════════════════════════════════════════════════
# Embedder.index_chunks Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestEmbedderIndexChunks:
    """Tests for Embedder.index_chunks method."""

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    def test_index_chunks_basic(self, mock_load_model, mock_chroma, sample_chunks):
        """Test basic chunk indexing."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.array([[0.1] * 768] * len(sample_chunks))
        mock_load_model.return_value = mock_model

        mock_client, mock_collection = MagicMock(), MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.get.return_value = {"ids": []}
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        embedder = Embedder()
        indexed_count = embedder.index_chunks(sample_chunks, skip_existing=False)

        assert indexed_count == len(sample_chunks)
        mock_collection.add.assert_called_once()

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    def test_index_chunks_empty_list(self, mock_load_model, mock_chroma):
        """Test indexing empty list of chunks."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_load_model.return_value = mock_model

        mock_client, mock_collection = MagicMock(), MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        embedder = Embedder()
        indexed_count = embedder.index_chunks([])

        assert indexed_count == 0
        mock_collection.add.assert_not_called()

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    def test_index_chunks_skip_existing_all_new(self, mock_load_model, mock_chroma, sample_chunks):
        """Test skip_existing when all chunks are new."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.array([[0.1] * 768] * len(sample_chunks))
        mock_load_model.return_value = mock_model

        mock_client, mock_collection = MagicMock(), MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.get.return_value = {"ids": []}  # No existing chunks
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        embedder = Embedder()
        indexed_count = embedder.index_chunks(sample_chunks, skip_existing=True)

        assert indexed_count == len(sample_chunks)
        mock_collection.add.assert_called_once()

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    def test_index_chunks_skip_existing_some_exist(self, mock_load_model, mock_chroma, sample_chunks):
        """Test skip_existing when some chunks already exist."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.array([[0.1] * 768])
        mock_load_model.return_value = mock_model

        mock_client, mock_collection = MagicMock(), MagicMock()
        mock_collection.count.return_value = 2
        # First two chunks already exist
        mock_collection.get.return_value = {"ids": ["chunk_001", "chunk_002"]}
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        embedder = Embedder()
        indexed_count = embedder.index_chunks(sample_chunks, skip_existing=True)

        # Only one new chunk should be indexed
        assert indexed_count == 1
        mock_collection.add.assert_called_once()

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    def test_index_chunks_skip_existing_all_exist(self, mock_load_model, mock_chroma, sample_chunks):
        """Test skip_existing when all chunks already exist."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_load_model.return_value = mock_model

        mock_client, mock_collection = MagicMock(), MagicMock()
        mock_collection.count.return_value = 3
        # All chunks exist
        mock_collection.get.return_value = {"ids": ["chunk_001", "chunk_002", "chunk_003"]}
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        embedder = Embedder()
        indexed_count = embedder.index_chunks(sample_chunks, skip_existing=True)

        assert indexed_count == 0
        mock_collection.add.assert_not_called()

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    def test_index_chunks_skip_existing_false(self, mock_load_model, mock_chroma, sample_chunks):
        """Test that skip_existing=False indexes all chunks."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.array([[0.1] * 768] * len(sample_chunks))
        mock_load_model.return_value = mock_model

        mock_client, mock_collection = MagicMock(), MagicMock()
        mock_collection.count.return_value = 2
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        embedder = Embedder()
        indexed_count = embedder.index_chunks(sample_chunks, skip_existing=False)

        # All chunks should be indexed regardless
        assert indexed_count == len(sample_chunks)
        # get() should not be called when skip_existing=False
        mock_collection.get.assert_not_called()

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    def test_index_chunks_batch_processing(self, mock_load_model, mock_chroma):
        """Test that batch_size parameter controls batching."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.array([[0.1] * 768])
        mock_load_model.return_value = mock_model

        mock_client, mock_collection = MagicMock(), MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.get.return_value = {"ids": []}
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        # Create 5 chunks
        chunks = [{"chunk_id": f"chunk_{i}", "text": f"text {i}", "doc_id": "doc", "section": "test"} for i in range(5)]

        embedder = Embedder()
        indexed_count = embedder.index_chunks(chunks, batch_size=2, skip_existing=False)

        assert indexed_count == 5
        # Should be called 3 times: ceil(5/2) = 3 batches
        assert mock_collection.add.call_count == 3

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    def test_index_chunks_metadata_sanitization(self, mock_load_model, mock_chroma):
        """Test that metadata is properly sanitized for ChromaDB."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.array([[0.1] * 768])
        mock_load_model.return_value = mock_model

        mock_client, mock_collection = MagicMock(), MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.get.return_value = {"ids": []}
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        # Chunk with various metadata types
        chunks = [
            {
                "chunk_id": "test_chunk",
                "text": "test text",
                "doc_id": "doc_1",
                "section": "intro",
                "metadata": {
                    "string_val": "text",
                    "int_val": 42,
                    "float_val": 3.14,
                    "bool_val": True,
                    "none_val": None,
                    "list_val": [1, 2, 3],  # Should be converted to string
                    "dict_val": {"nested": "value"},  # Should be converted to string
                },
            }
        ]

        embedder = Embedder()
        embedder.index_chunks(chunks, skip_existing=False)

        # Check metadata passed to add()
        call_args = mock_collection.add.call_args
        metadatas = call_args[1]["metadatas"]

        assert len(metadatas) == 1
        meta = metadatas[0]

        # Valid types preserved
        assert meta["string_val"] == "text"
        assert meta["int_val"] == 42
        assert meta["float_val"] == 3.14
        assert meta["bool_val"] is True

        # None converted to empty string
        assert meta["none_val"] == ""

        # Complex types converted to strings
        assert isinstance(meta["list_val"], str)
        assert isinstance(meta["dict_val"], str)

        # Required fields
        assert meta["section"] == "intro"
        assert meta["doc_id"] == "doc_1"

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    def test_index_chunks_metadata_missing_optional(self, mock_load_model, mock_chroma):
        """Test indexing chunks with missing optional metadata."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.array([[0.1] * 768])
        mock_load_model.return_value = mock_model

        mock_client, mock_collection = MagicMock(), MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.get.return_value = {"ids": []}
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        # Chunk without metadata field
        chunks = [
            {
                "chunk_id": "test_chunk",
                "text": "test text",
                "doc_id": "doc_1",
                "section": "intro",
            }
        ]

        embedder = Embedder()
        indexed_count = embedder.index_chunks(chunks, skip_existing=False)

        assert indexed_count == 1
        mock_collection.add.assert_called_once()

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    def test_index_chunks_handles_get_exception(self, mock_load_model, mock_chroma, sample_chunks):
        """Test that exceptions during get() are handled gracefully."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.array([[0.1] * 768] * len(sample_chunks))
        mock_load_model.return_value = mock_model

        mock_client, mock_collection = MagicMock(), MagicMock()
        mock_collection.count.return_value = 1
        mock_collection.get.side_effect = Exception("ChromaDB error")
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        embedder = Embedder()
        # Should not raise, should treat chunks as new
        indexed_count = embedder.index_chunks(sample_chunks, skip_existing=True)

        assert indexed_count == len(sample_chunks)


# ══════════════════════════════════════════════════════════════════════════════
# Embedder.get_collection_stats Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestEmbedderGetCollectionStats:
    """Tests for Embedder.get_collection_stats method."""

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    def test_get_collection_stats_empty_collection(self, mock_load_model, mock_chroma):
        """Test stats for empty collection."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_load_model.return_value = mock_model

        mock_client, mock_collection = MagicMock(), MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        embedder = Embedder()
        stats = embedder.get_collection_stats()

        assert stats == {
            "total_chunks": 0,
            "unique_papers": 0,
            "unique_sections": 0,
        }
        # get() should not be called for empty collection
        mock_collection.get.assert_not_called()

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    def test_get_collection_stats_with_data(self, mock_load_model, mock_chroma):
        """Test stats for collection with data."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_load_model.return_value = mock_model

        mock_client, mock_collection = MagicMock(), MagicMock()
        mock_collection.count.return_value = 5
        mock_collection.get.return_value = {
            "metadatas": [
                {"doc_id": "paper_1", "section": "intro"},
                {"doc_id": "paper_1", "section": "methods"},
                {"doc_id": "paper_2", "section": "intro"},
                {"doc_id": "paper_2", "section": "results"},
                {"doc_id": "paper_3", "section": "intro"},
            ]
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        embedder = Embedder()
        stats = embedder.get_collection_stats()

        assert stats["total_chunks"] == 5
        assert stats["unique_papers"] == 3  # paper_1, paper_2, paper_3
        assert stats["unique_sections"] == 3  # intro, methods, results

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    def test_get_collection_stats_large_collection(self, mock_load_model, mock_chroma):
        """Test stats for large collection (samples only 10k)."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_load_model.return_value = mock_model

        mock_client, mock_collection = MagicMock(), MagicMock()
        mock_collection.count.return_value = 50000
        # Simulate sampled data
        mock_collection.get.return_value = {
            "metadatas": [{"doc_id": f"paper_{i % 100}", "section": "intro"} for i in range(10000)]
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        embedder = Embedder()
        stats = embedder.get_collection_stats()

        assert stats["total_chunks"] == 50000
        # Only samples papers, might not capture all
        assert stats["unique_papers"] == 100

        # Verify limit parameter was used
        call_kwargs = mock_collection.get.call_args[1]
        assert call_kwargs["limit"] == 10000

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    def test_get_collection_stats_missing_metadata_fields(self, mock_load_model, mock_chroma):
        """Test stats when some metadata fields are missing."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_load_model.return_value = mock_model

        mock_client, mock_collection = MagicMock(), MagicMock()
        mock_collection.count.return_value = 3
        mock_collection.get.return_value = {
            "metadatas": [
                {"doc_id": "paper_1", "section": "intro"},
                {"section": "methods"},  # Missing doc_id
                {"doc_id": "paper_2"},  # Missing section
            ]
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        embedder = Embedder()
        stats = embedder.get_collection_stats()

        assert stats["total_chunks"] == 3
        # Should count empty strings from missing fields
        assert stats["unique_papers"] == 3  # paper_1, "", paper_2
        assert stats["unique_sections"] == 3  # intro, methods, ""


# ══════════════════════════════════════════════════════════════════════════════
# Edge Cases and Error Scenarios
# ══════════════════════════════════════════════════════════════════════════════


class TestEdgeCasesAndErrors:
    """Tests for edge cases and error handling."""

    def test_tfidf_embedder_very_long_text(self):
        """Test TfidfFallbackEmbedder with very long text."""
        embedder = TfidfFallbackEmbedder(dim=100)
        long_text = " ".join(["word"] * 10000)

        result = embedder.encode([long_text])

        assert result.shape == (1, 100)
        assert not np.any(np.isnan(result))

    def test_tfidf_embedder_unicode_text(self):
        """Test TfidfFallbackEmbedder with Unicode characters."""
        embedder = TfidfFallbackEmbedder(dim=100)
        unicode_text = "Hello 世界 🌍 café naïve"

        result = embedder.encode([unicode_text])

        assert result.shape == (1, 100)
        assert not np.any(np.isnan(result))

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    def test_embed_texts_with_very_large_batch_size(self, mock_load_model, mock_chroma):
        """Test embed_texts with batch size larger than number of texts."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.array([[0.1] * 768] * 3)
        mock_load_model.return_value = mock_model

        mock_client, mock_collection = MagicMock(), MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        embedder = Embedder()
        texts = ["text1", "text2", "text3"]
        result = embedder.embed_texts(texts, batch_size=1000, show_progress=False)

        assert len(result) == 3
        # Should be called only once since batch_size > len(texts)
        mock_model.encode.assert_called_once()

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    def test_index_chunks_with_empty_text(self, mock_load_model, mock_chroma):
        """Test indexing chunks with empty text field."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.array([[0.1] * 768])
        mock_load_model.return_value = mock_model

        mock_client, mock_collection = MagicMock(), MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.get.return_value = {"ids": []}
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        chunks = [{"chunk_id": "empty_chunk", "text": "", "doc_id": "doc_1", "section": "test"}]

        embedder = Embedder()
        indexed_count = embedder.index_chunks(chunks, skip_existing=False)

        # Should still index, even with empty text
        assert indexed_count == 1

    @patch("rag_bench.core.embedder.chromadb.PersistentClient")
    @patch("rag_bench.core.embedder._load_embedding_model")
    def test_index_chunks_large_batch_check(self, mock_load_model, mock_chroma):
        """Test that large chunk lists are checked in batches of 1000."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.array([[0.1] * 768])
        mock_load_model.return_value = mock_model

        mock_client, mock_collection = MagicMock(), MagicMock()
        mock_collection.count.return_value = 2000
        mock_collection.get.return_value = {"ids": []}
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client

        # Create 2500 chunks
        chunks = [
            {"chunk_id": f"chunk_{i}", "text": f"text {i}", "doc_id": "doc", "section": "test"} for i in range(2500)
        ]

        embedder = Embedder()
        embedder.index_chunks(chunks, skip_existing=True)

        # Should call get() 3 times: ceil(2500/1000) = 3
        assert mock_collection.get.call_count == 3
