"""
Unit tests for rag_bench.core.hybrid_ingest module.

Tests cover:
- arXiv ID extraction and normalization
- Loading scraped papers
- Loading HuggingFace papers
- Merging paper sources with deduplication
- Full hybrid ingestion pipeline
"""

import json
from unittest.mock import patch

import pytest

from rag_bench.core.hybrid_ingest import (
    extract_arxiv_id,
    hybrid_ingest,
    load_hf_papers,
    load_scraped_papers,
    merge_paper_sources,
    normalize_arxiv_id,
)

# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_scraped_papers():
    """Sample scraped paper documents."""
    return [
        {
            "doc_id": "arxiv_1706_03762",
            "title": "Attention Is All You Need",
            "arxiv_id": "1706.03762",
            "year": 2017,
            "full_text": "This paper introduces the Transformer...",
        },
        {
            "doc_id": "arxiv_1810_04805",
            "title": "BERT",
            "arxiv_id": "1810.04805",
            "year": 2018,
            "full_text": "BERT is a pre-trained model...",
        },
        {
            "doc_id": "arxiv_2005_11401",
            "title": "RAG",
            "arxiv_id": "2005.11401",
            "year": 2020,
            "full_text": "Retrieval-augmented generation...",
        },
    ]


@pytest.fixture
def sample_hf_papers():
    """Sample HuggingFace dataset papers (as raw rows)."""
    return [
        {
            "id": "1706.03762",
            "title": "Attention Is All You Need (HF version)",
            "published": "2017-06-12",
            "chunk": "HF version of the Transformer paper...",
            "chunk_id": "1706.03762-chunk-1",
        },
        {
            "id": "2106.09685",
            "title": "LoRA",
            "published": "2021-06-17",
            "chunk": "Low-Rank Adaptation...",
            "chunk_id": "2106.09685-chunk-1",
        },
        {
            "id": "short",
            "title": "Too Short",
            "published": "2023-01-01",
            "chunk": "x" * 50,  # Less than 100 chars
            "chunk_id": "short-chunk-1",
        },
    ]


@pytest.fixture
def temp_scraped_file(tmp_path, sample_scraped_papers):
    """Create a temporary scraped papers JSON file."""
    file_path = tmp_path / "scraped_papers.json"
    with open(file_path, "w") as f:
        json.dump(sample_scraped_papers, f)
    return file_path


# ══════════════════════════════════════════════════════════════════════════════
# extract_arxiv_id Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestExtractArxivId:
    """Tests for extract_arxiv_id function."""

    def test_extract_with_prefix_and_underscores(self):
        """Test extraction from arxiv_XXXX_XXXXX format."""
        result = extract_arxiv_id("arxiv_2106_09685")
        assert result == "2106.09685"

    def test_extract_with_prefix_and_dots(self):
        """Test extraction from arxiv_XXXX.XXXXX format."""
        result = extract_arxiv_id("arxiv_2106.09685")
        assert result == "2106.09685"

    def test_extract_without_prefix(self):
        """Test extraction from plain XXXX.XXXXX format."""
        result = extract_arxiv_id("2106.09685")
        assert result == "2106.09685"

    def test_extract_with_underscores_no_prefix(self):
        """Test extraction with underscores but no prefix."""
        result = extract_arxiv_id("1706_03762")
        assert result == "1706.03762"

    def test_extract_preserves_dots(self):
        """Test that dots are preserved if already present."""
        result = extract_arxiv_id("1706.03762")
        assert result == "1706.03762"


# ══════════════════════════════════════════════════════════════════════════════
# normalize_arxiv_id Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestNormalizeArxivId:
    """Tests for normalize_arxiv_id function."""

    def test_normalize_standard_format(self):
        """Test normalization of standard format."""
        result = normalize_arxiv_id("2106.09685")
        assert result == "2106.09685"

    def test_normalize_with_arxiv_prefix(self):
        """Test normalization with arxiv: prefix."""
        result = normalize_arxiv_id("arxiv:2106.09685")
        assert result == "2106.09685"

    def test_normalize_with_version(self):
        """Test normalization with version number."""
        result = normalize_arxiv_id("1706.03762v1")
        assert result == "1706.03762"

        result = normalize_arxiv_id("2005.11401v2")
        assert result == "2005.11401"

    def test_normalize_old_format_cs(self):
        """Test normalization of old format (cs/YYMMNNN)."""
        result = normalize_arxiv_id("cs/0506023")
        assert result == "0506.023"

    def test_normalize_old_format_other_category(self):
        """Test normalization of old format with different category."""
        result = normalize_arxiv_id("math/0601001")
        assert result == "0601.001"

    def test_normalize_combined_arxiv_prefix_and_version(self):
        """Test normalization with both prefix and version."""
        result = normalize_arxiv_id("arxiv:1706.03762v3")
        assert result == "1706.03762"

    def test_normalize_whitespace(self):
        """Test that whitespace is stripped."""
        result = normalize_arxiv_id("  2106.09685  ")
        assert result == "2106.09685"

    def test_normalize_case_insensitive_prefix(self):
        """Test that arxiv: prefix is case-insensitive."""
        result1 = normalize_arxiv_id("arxiv:1234.5678")
        result2 = normalize_arxiv_id("ARXIV:1234.5678")
        result3 = normalize_arxiv_id("ArXiv:1234.5678")

        assert result1 == result2 == result3 == "1234.5678"


# ══════════════════════════════════════════════════════════════════════════════
# load_scraped_papers Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestLoadScrapedPapers:
    """Tests for load_scraped_papers function."""

    def test_load_scraped_papers_success(self, temp_scraped_file):
        """Test successful loading of scraped papers."""
        result = load_scraped_papers(temp_scraped_file)

        assert isinstance(result, dict)
        assert len(result) == 3
        assert "1706.03762" in result
        assert "1810.04805" in result
        assert "2005.11401" in result

    def test_load_scraped_papers_structure(self, temp_scraped_file):
        """Test that loaded papers have correct structure."""
        result = load_scraped_papers(temp_scraped_file)

        paper = result["1706.03762"]
        assert "doc_id" in paper
        assert "title" in paper
        assert "arxiv_id" in paper
        assert "year" in paper
        assert "full_text" in paper

    def test_load_scraped_papers_nonexistent_file(self, tmp_path):
        """Test loading from nonexistent file."""
        nonexistent_path = tmp_path / "does_not_exist.json"
        result = load_scraped_papers(nonexistent_path)

        assert result == {}

    def test_load_scraped_papers_empty_json(self, tmp_path):
        """Test loading empty JSON file."""
        empty_file = tmp_path / "empty.json"
        with open(empty_file, "w") as f:
            json.dump([], f)

        result = load_scraped_papers(empty_file)

        assert result == {}

    def test_load_scraped_papers_normalizes_ids(self, tmp_path):
        """Test that IDs are normalized when loading."""
        papers = [
            {
                "doc_id": "arxiv_1706_03762",  # Underscores
                "arxiv_id": "1706.03762",
                "title": "Test",
                "full_text": "Text",
            }
        ]

        file_path = tmp_path / "papers.json"
        with open(file_path, "w") as f:
            json.dump(papers, f)

        result = load_scraped_papers(file_path)

        # Should be normalized to dots
        assert "1706.03762" in result


# ══════════════════════════════════════════════════════════════════════════════
# load_hf_papers Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestLoadHfPapers:
    """Tests for load_hf_papers function."""

    @patch("rag_bench.core.hybrid_ingest.tqdm")
    @patch("rag_bench.core.hybrid_ingest.load_arxiv_dataset")
    @patch("rag_bench.core.hybrid_ingest.parse_paper")
    def test_load_hf_papers_success(self, mock_parse, mock_load, mock_tqdm):
        """Test successful loading of HF papers."""
        # Create sample papers with sufficient text
        sample_papers = [
            {
                "id": "1706.03762",
                "title": "Attention Is All You Need",
                "published": "2017-06-12",
                "chunk": "x" * 200,  # Sufficient length
                "chunk_id": "1706.03762-chunk-1",
            },
            {
                "id": "2106.09685",
                "title": "LoRA",
                "published": "2021-06-17",
                "chunk": "y" * 200,  # Sufficient length
                "chunk_id": "2106.09685-chunk-1",
            },
        ]

        mock_load.return_value = sample_papers
        # Make tqdm return the iterable directly
        mock_tqdm.return_value = sample_papers

        # Mock parse_paper to return proper documents
        def parse_side_effect(row):
            return {
                "doc_id": f"arxiv_{row['id']}",
                "arxiv_id": row["id"],
                "title": row["title"],
                "year": int(row["published"][:4]),
                "full_text": row["chunk"],
            }

        mock_parse.side_effect = parse_side_effect

        result = load_hf_papers()

        assert isinstance(result, dict)
        assert len(result) == 2
        assert "1706.03762" in result
        assert "2106.09685" in result

    @patch("rag_bench.core.hybrid_ingest.load_arxiv_dataset")
    @patch("rag_bench.core.hybrid_ingest.parse_paper")
    def test_load_hf_papers_skips_short_text(self, mock_parse, mock_load):
        """Test that papers with short text are skipped."""
        mock_load.return_value = [
            {
                "id": "1234.5678",
                "title": "Test",
                "published": "2023-01-01",
                "chunk": "x" * 50,  # Less than 100 chars
                "chunk_id": "test-chunk",
            }
        ]

        mock_parse.return_value = {
            "doc_id": "arxiv_1234.5678",
            "arxiv_id": "1234.5678",
            "title": "Test",
            "year": 2023,
            "full_text": "x" * 50,
        }

        result = load_hf_papers()

        # Should skip the paper
        assert len(result) == 0

    @patch("rag_bench.core.hybrid_ingest.load_arxiv_dataset")
    @patch("rag_bench.core.hybrid_ingest.parse_paper")
    def test_load_hf_papers_skips_unknown_arxiv_id(self, mock_parse, mock_load):
        """Test that papers with unknown arXiv ID are skipped."""
        mock_load.return_value = [
            {
                "id": "unknown",
                "title": "Test",
                "published": "2023-01-01",
                "chunk": "x" * 200,
                "chunk_id": "test-chunk",
            }
        ]

        mock_parse.return_value = {
            "doc_id": "arxiv_unknown",
            "arxiv_id": "unknown",
            "title": "Test",
            "year": 2023,
            "full_text": "x" * 200,
        }

        result = load_hf_papers()

        # Should skip the paper
        assert len(result) == 0

    @patch("rag_bench.core.hybrid_ingest.load_arxiv_dataset")
    @patch("rag_bench.core.hybrid_ingest.parse_paper")
    def test_load_hf_papers_normalizes_ids(self, mock_parse, mock_load):
        """Test that arXiv IDs are normalized."""
        mock_load.return_value = [
            {
                "id": "arxiv:1706.03762v1",  # With prefix and version
                "title": "Test",
                "published": "2017-06-12",
                "chunk": "x" * 200,
                "chunk_id": "test-chunk",
            }
        ]

        mock_parse.return_value = {
            "doc_id": "arxiv_1706.03762v1",
            "arxiv_id": "arxiv:1706.03762v1",
            "title": "Test",
            "year": 2017,
            "full_text": "x" * 200,
        }

        result = load_hf_papers()

        # Should be normalized
        assert "1706.03762" in result

    @patch("rag_bench.core.hybrid_ingest.load_arxiv_dataset")
    @patch("rag_bench.core.hybrid_ingest.parse_paper")
    def test_load_hf_papers_custom_dataset(self, mock_parse, mock_load):
        """Test loading from custom dataset name."""
        mock_load.return_value = []

        load_hf_papers(dataset_name="custom/dataset", split="test")

        # Verify correct parameters passed
        mock_load.assert_called_once_with("custom/dataset", "test")


# ══════════════════════════════════════════════════════════════════════════════
# merge_paper_sources Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestMergePaperSources:
    """Tests for merge_paper_sources function."""

    def test_merge_no_overlap_prefer_scraped(self):
        """Test merging with no overlap, preferring scraped."""
        scraped = {
            "1706.03762": {"arxiv_id": "1706.03762", "source": "scraped"},
            "1810.04805": {"arxiv_id": "1810.04805", "source": "scraped"},
        }

        hf = {
            "2106.09685": {"arxiv_id": "2106.09685", "source": "hf"},
        }

        result = merge_paper_sources(scraped, hf, prefer_scraped=True)

        assert len(result) == 3
        arxiv_ids = [p["arxiv_id"] for p in result]
        assert "1706.03762" in arxiv_ids
        assert "1810.04805" in arxiv_ids
        assert "2106.09685" in arxiv_ids

    def test_merge_with_overlap_prefer_scraped(self):
        """Test merging with overlap, preferring scraped."""
        scraped = {
            "1706.03762": {"arxiv_id": "1706.03762", "source": "scraped", "quality": "high"},
        }

        hf = {
            "1706.03762": {"arxiv_id": "1706.03762", "source": "hf", "quality": "low"},
        }

        result = merge_paper_sources(scraped, hf, prefer_scraped=True)

        assert len(result) == 1
        assert result[0]["source"] == "scraped"
        assert result[0]["quality"] == "high"

    def test_merge_with_overlap_prefer_hf(self):
        """Test merging with overlap, preferring HF."""
        scraped = {
            "1706.03762": {"arxiv_id": "1706.03762", "source": "scraped", "quality": "high"},
        }

        hf = {
            "1706.03762": {"arxiv_id": "1706.03762", "source": "hf", "quality": "low"},
        }

        result = merge_paper_sources(scraped, hf, prefer_scraped=False)

        assert len(result) == 1
        assert result[0]["source"] == "hf"
        assert result[0]["quality"] == "low"

    def test_merge_scraped_only(self):
        """Test merging with only scraped papers."""
        scraped = {
            "1706.03762": {"arxiv_id": "1706.03762", "source": "scraped"},
            "1810.04805": {"arxiv_id": "1810.04805", "source": "scraped"},
        }

        result = merge_paper_sources(scraped, {}, prefer_scraped=True)

        assert len(result) == 2

    def test_merge_hf_only(self):
        """Test merging with only HF papers."""
        hf = {
            "2106.09685": {"arxiv_id": "2106.09685", "source": "hf"},
            "2005.11401": {"arxiv_id": "2005.11401", "source": "hf"},
        }

        result = merge_paper_sources({}, hf, prefer_scraped=True)

        assert len(result) == 2

    def test_merge_empty_sources(self):
        """Test merging with both sources empty."""
        result = merge_paper_sources({}, {}, prefer_scraped=True)

        assert len(result) == 0

    def test_merge_sorted_by_arxiv_id(self):
        """Test that results are sorted by arXiv ID."""
        scraped = {
            "2106.09685": {"arxiv_id": "2106.09685"},
            "1706.03762": {"arxiv_id": "1706.03762"},
            "1810.04805": {"arxiv_id": "1810.04805"},
        }

        result = merge_paper_sources(scraped, {}, prefer_scraped=True)

        arxiv_ids = [p["arxiv_id"] for p in result]
        assert arxiv_ids == sorted(arxiv_ids)


# ══════════════════════════════════════════════════════════════════════════════
# hybrid_ingest Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestHybridIngest:
    """Tests for hybrid_ingest function."""

    @patch("rag_bench.core.hybrid_ingest.load_hf_papers")
    @patch("rag_bench.core.hybrid_ingest.load_scraped_papers")
    def test_hybrid_ingest_basic(self, mock_load_scraped, mock_load_hf, tmp_path):
        """Test basic hybrid ingestion."""
        mock_load_scraped.return_value = {
            "1706.03762": {"arxiv_id": "1706.03762", "source": "scraped", "year": 2017},
        }

        mock_load_hf.return_value = {
            "2106.09685": {"arxiv_id": "2106.09685", "source": "hf", "year": 2021},
        }

        scraped_path = tmp_path / "scraped.json"

        result = hybrid_ingest(scraped_path)

        assert len(result) == 2

    @patch("rag_bench.core.hybrid_ingest.load_hf_papers")
    @patch("rag_bench.core.hybrid_ingest.load_scraped_papers")
    def test_hybrid_ingest_with_save(self, mock_load_scraped, mock_load_hf, tmp_path):
        """Test hybrid ingestion with saving to file."""
        mock_load_scraped.return_value = {
            "1706.03762": {"arxiv_id": "1706.03762", "source": "scraped"},
        }

        mock_load_hf.return_value = {
            "2106.09685": {"arxiv_id": "2106.09685", "source": "hf"},
        }

        scraped_path = tmp_path / "scraped.json"
        save_path = tmp_path / "output" / "merged.json"

        result = hybrid_ingest(scraped_path, save_path=save_path)

        assert len(result) == 2
        assert save_path.exists()

        # Verify saved content
        with open(save_path) as f:
            saved_data = json.load(f)

        assert len(saved_data) == 2

    @patch("rag_bench.core.hybrid_ingest.load_hf_papers")
    @patch("rag_bench.core.hybrid_ingest.load_scraped_papers")
    def test_hybrid_ingest_custom_dataset(self, mock_load_scraped, mock_load_hf, tmp_path):
        """Test hybrid ingestion with custom dataset."""
        mock_load_scraped.return_value = {}
        mock_load_hf.return_value = {}

        scraped_path = tmp_path / "scraped.json"

        hybrid_ingest(
            scraped_path,
            dataset_name="custom/dataset",
            split="test",
        )

        # Verify correct parameters passed to load_hf_papers
        mock_load_hf.assert_called_once_with("custom/dataset", "test")

    @patch("rag_bench.core.hybrid_ingest.load_hf_papers")
    @patch("rag_bench.core.hybrid_ingest.load_scraped_papers")
    def test_hybrid_ingest_prefer_hf(self, mock_load_scraped, mock_load_hf, tmp_path):
        """Test hybrid ingestion preferring HF over scraped."""
        # Same paper in both sources
        mock_load_scraped.return_value = {
            "1706.03762": {"arxiv_id": "1706.03762", "source": "scraped"},
        }

        mock_load_hf.return_value = {
            "1706.03762": {"arxiv_id": "1706.03762", "source": "hf"},
        }

        scraped_path = tmp_path / "scraped.json"

        result = hybrid_ingest(scraped_path, prefer_scraped=False)

        assert len(result) == 1
        assert result[0]["source"] == "hf"

    @patch("rag_bench.core.hybrid_ingest.load_hf_papers")
    @patch("rag_bench.core.hybrid_ingest.load_scraped_papers")
    def test_hybrid_ingest_logs_year_range(self, mock_load_scraped, mock_load_hf, tmp_path):
        """Test that hybrid ingestion logs year range."""
        mock_load_scraped.return_value = {
            "1706.03762": {"arxiv_id": "1706.03762", "year": 2017},
        }

        mock_load_hf.return_value = {
            "2106.09685": {"arxiv_id": "2106.09685", "year": 2021},
            "2307.09288": {"arxiv_id": "2307.09288", "year": 2023},
        }

        scraped_path = tmp_path / "scraped.json"

        result = hybrid_ingest(scraped_path)

        # Should have papers with years
        years = [p["year"] for p in result]
        assert min(years) == 2017
        assert max(years) == 2023
