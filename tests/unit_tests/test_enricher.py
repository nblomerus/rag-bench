"""
Unit tests for rag_bench.core.enricher module.

Tests cover:
- ContextualEnricher: header generation, caching, context building
- Full paper vs compressed context fallback
- Cache hit/miss behavior
- Error handling for LLM failures
- Integration with ChunkData (frozen dataclass)
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from rag_bench.core.enricher import ContextualEnricher, EnricherConfig
from rag_bench.core.types import ChunkData

# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_paper():
    """Sample paper for testing enrichment."""
    return {
        "doc_id": "arxiv_1234.5678",
        "title": "Attention Is All You Need",
        "authors": ["Vaswani, A.", "Shazeer, N.", "Parmar, N."],
        "year": 2017,
        "sections": {
            "abstract": "We propose a new architecture called the Transformer.",
            "introduction": "Sequence transduction models are based on complex recurrent networks. "
            "We propose a new simple architecture, the Transformer, based solely on attention mechanisms.",
            "methods": "The Transformer uses multi-head self-attention to compute representations. "
            "Each attention head learns different aspects of the input sequence relationships.",
        },
    }


@pytest.fixture
def sample_chunks():
    """Sample chunks that would come from chunking the sample paper."""
    return [
        ChunkData(
            chunk_id="arxiv_1234.5678_introduction_000",
            doc_id="arxiv_1234.5678",
            text="Attention Is All You Need — Introduction\n\n"
            "Sequence transduction models are based on complex recurrent networks.",
            section="introduction",
            metadata={"title": "Attention Is All You Need", "year": 2017},
        ),
        ChunkData(
            chunk_id="arxiv_1234.5678_methods_000",
            doc_id="arxiv_1234.5678",
            text="Attention Is All You Need — Methods\n\nThe Transformer uses multi-head self-attention.",
            section="methods",
            metadata={"title": "Attention Is All You Need", "year": 2017},
        ),
    ]


@pytest.fixture
def enricher(tmp_path):
    """Enricher with a temporary cache directory."""
    config = EnricherConfig(
        enabled=True,
        cache_dir=str(tmp_path / "cache"),
    )
    return ContextualEnricher(config=config)


@pytest.fixture
def mock_ollama_response():
    """Mock a successful Ollama response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "response": "This chunk is from 'Attention Is All You Need' by Vaswani et al. "
        "It introduces the Transformer architecture as an alternative to recurrent models."
    }
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


# ══════════════════════════════════════════════════════════════════════════════
# Header generation
# ══════════════════════════════════════════════════════════════════════════════


class TestHeaderGeneration:
    def test_enrich_prepends_header(self, enricher, sample_chunks, sample_paper, mock_ollama_response):
        """Enriched chunks should have the header prepended to their text."""
        with patch("rag_bench.core.enricher.requests.post", return_value=mock_ollama_response):
            enriched = enricher.enrich(sample_chunks, sample_paper)

        assert len(enriched) == len(sample_chunks)
        for original, result in zip(sample_chunks, enriched, strict=True):
            # Header should be prepended
            assert result.text.startswith("This chunk is from")
            # Original text should still be present
            assert original.text in result.text
            # Metadata should have the flag
            assert result.metadata["has_contextual_header"] is True

    def test_enrich_preserves_chunk_identity(self, enricher, sample_chunks, sample_paper, mock_ollama_response):
        """Enrichment should preserve chunk_id, doc_id, and section."""
        with patch("rag_bench.core.enricher.requests.post", return_value=mock_ollama_response):
            enriched = enricher.enrich(sample_chunks, sample_paper)

        for original, result in zip(sample_chunks, enriched, strict=True):
            assert result.chunk_id == original.chunk_id
            assert result.doc_id == original.doc_id
            assert result.section == original.section

    def test_enrich_empty_list(self, enricher, sample_paper):
        """Enriching an empty list should return an empty list."""
        result = enricher.enrich([], sample_paper)
        assert result == []

    def test_enrich_handles_llm_failure(self, enricher, sample_chunks, sample_paper):
        """If the LLM fails, chunks should pass through with empty header."""
        import requests as req

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = req.HTTPError("500 Server Error")

        with patch("rag_bench.core.enricher.requests.post", side_effect=req.RequestException("Connection refused")):
            enriched = enricher.enrich(sample_chunks, sample_paper)

        # Chunks should still be returned, just with empty headers
        assert len(enriched) == len(sample_chunks)
        for original, result in zip(sample_chunks, enriched, strict=True):
            # With empty header, text starts with \n\n then original
            assert original.text in result.text

    def test_enrich_skips_short_llm_response(self, enricher, sample_chunks, sample_paper):
        """If the LLM returns a very short response, it should be treated as empty."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "OK"}
        mock_resp.raise_for_status = MagicMock()

        with patch("rag_bench.core.enricher.requests.post", return_value=mock_resp):
            enriched = enricher.enrich(sample_chunks, sample_paper)

        # "OK" is < 20 chars, so header should be empty
        for result in enriched:
            assert not result.text.startswith("OK")


# ══════════════════════════════════════════════════════════════════════════════
# Caching
# ══════════════════════════════════════════════════════════════════════════════


class TestCaching:
    def test_cache_saves_and_loads(self, enricher, sample_chunks, sample_paper, mock_ollama_response):
        """Second call should use cached headers (no LLM calls)."""
        with patch("rag_bench.core.enricher.requests.post", return_value=mock_ollama_response) as mock_post:
            enricher.enrich(sample_chunks, sample_paper)
            first_call_count = mock_post.call_count

            # Second call — should be all cache hits
            enricher.enrich(sample_chunks, sample_paper)
            assert mock_post.call_count == first_call_count  # no new LLM calls

    def test_cache_file_created(self, enricher, sample_chunks, sample_paper, mock_ollama_response):
        """A cache JSON file should be created for each paper."""
        with patch("rag_bench.core.enricher.requests.post", return_value=mock_ollama_response):
            enricher.enrich(sample_chunks, sample_paper)

        cache_path = enricher._cache_path("arxiv_1234.5678")
        assert cache_path.exists()

        cache_data = json.loads(cache_path.read_text())
        assert len(cache_data) == len(sample_chunks)

    def test_cache_invalidates_on_text_change(self, enricher, sample_chunks, sample_paper, mock_ollama_response):
        """If chunk text changes, cache should miss and re-generate."""
        with patch("rag_bench.core.enricher.requests.post", return_value=mock_ollama_response) as mock_post:
            enricher.enrich(sample_chunks, sample_paper)
            first_call_count = mock_post.call_count

            # Modify one chunk's text
            modified_chunks = [
                ChunkData(
                    chunk_id=sample_chunks[0].chunk_id,
                    doc_id=sample_chunks[0].doc_id,
                    text="Completely different text now",
                    section=sample_chunks[0].section,
                    metadata=sample_chunks[0].metadata,
                ),
                sample_chunks[1],  # unchanged
            ]

            enricher.enrich(modified_chunks, sample_paper)
            # Should have made exactly 1 new LLM call (for the changed chunk)
            assert mock_post.call_count == first_call_count + 1


# ══════════════════════════════════════════════════════════════════════════════
# Context building
# ══════════════════════════════════════════════════════════════════════════════


class TestContextBuilding:
    def test_full_text_used_for_small_papers(self, enricher, sample_paper):
        """Small papers should use full text context."""
        context = enricher._build_document_context(sample_paper)
        # Should contain content from all sections
        assert "Transformer" in context
        assert "multi-head self-attention" in context

    def test_compressed_context_for_large_papers(self, tmp_path):
        """Papers exceeding context limit should use compressed context."""
        config = EnricherConfig(
            enabled=True,
            cache_dir=str(tmp_path / "cache"),
            max_context_tokens=1000,  # small enough to force compression on 10K+ papers
        )
        enricher = ContextualEnricher(config=config)

        large_paper = {
            "title": "Very Long Survey Paper",
            "authors": ["Author A"],
            "sections": {
                "abstract": "This is the abstract.",
                "introduction": "x " * 5000,  # 10K chars
                "methods": "y " * 5000,
            },
        }

        context = enricher._build_document_context(large_paper)
        # Should be compressed — sections truncated to ~200 chars
        assert "Very Long Survey Paper" in context
        assert len(context) < len("x " * 5000)  # much shorter than full

    def test_compressed_context_preserves_abstract(self, tmp_path):
        """Compressed context should keep the full abstract."""
        config = EnricherConfig(
            enabled=True,
            cache_dir=str(tmp_path / "cache"),
            max_context_tokens=1000,
        )
        enricher = ContextualEnricher(config=config)

        paper = {
            "title": "Test Paper",
            "authors": [],
            "sections": {
                "abstract": "This is a complete abstract with all the details.",
                "introduction": "x " * 5000,
            },
        }

        context = enricher._build_document_context(paper)
        assert "This is a complete abstract with all the details." in context

    def test_build_full_text_string_authors(self, enricher):
        """When authors is a string (not a list), it should be included directly (line 207)."""
        paper = {
            "title": "String Authors Paper",
            "authors": "Doe, J. and Smith, K.",
            "sections": {"abstract": "Short abstract."},
        }
        context = enricher._build_full_text(paper)
        assert "Doe, J. and Smith, K." in context

    def test_build_compressed_context_string_authors(self, tmp_path):
        """Compressed context: string authors path (line 229)."""
        config = EnricherConfig(
            enabled=True,
            cache_dir=str(tmp_path / "cache"),
            max_context_tokens=1000,
        )
        enricher = ContextualEnricher(config=config)

        paper = {
            "title": "Compressed String Authors",
            "authors": "Single Author String",
            "sections": {
                "abstract": "Abstract text.",
                "introduction": "x " * 5000,
            },
        }
        context = enricher._build_compressed_context(paper)
        assert "Single Author String" in context

    def test_build_compressed_context_hard_truncate(self, tmp_path):
        """Hard truncation when result still exceeds max_context_chars (line 251)."""
        # max_context_tokens=600 → available=88 → max_context_chars=308
        # Title of 500 chars alone exceeds this limit.
        config = EnricherConfig(
            enabled=True,
            cache_dir=str(tmp_path / "cache"),
            max_context_tokens=600,
        )
        enricher = ContextualEnricher(config=config)

        paper = {
            "title": "T" * 500,
            "authors": ["Author A"],
            "sections": {"abstract": "x " * 500},
        }
        context = enricher._build_compressed_context(paper)
        assert "[truncated]" in context


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════


class TestEnricherConfig:
    def test_default_config(self):
        """Default config should have enrichment disabled."""
        config = EnricherConfig()
        assert config.enabled is False
        assert "qwen2.5" in config.ollama_model

    def test_cache_dir_created(self, tmp_path):
        """Cache directory should be created on init."""
        cache_dir = tmp_path / "new_cache"
        assert not cache_dir.exists()

        config = EnricherConfig(enabled=True, cache_dir=str(cache_dir))
        ContextualEnricher(config=config)
        assert cache_dir.exists()
