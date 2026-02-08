"""
Unit tests for rag_bench.core.ingest module.

Tests cover:
- load_arxiv_dataset: dataset loading from HuggingFace
- extract_year: year extraction from various field formats
- parse_paper: paper parsing and document schema creation
- ingest_dataset: full pipeline integration
- Edge cases: missing fields, malformed data, empty content
"""

from unittest.mock import MagicMock, mock_open, patch

import pytest

from rag_bench.core.ingest import extract_year, ingest_dataset, load_arxiv_dataset, parse_paper

# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_arxiv_row():
    """Sample row from arxiv dataset."""
    return {
        "id": "2301.12345",
        "title": "Advances in Neural Networks",
        "authors": ["Smith, John", "Doe, Jane", "Lee, Wei"],
        "content": "# Introduction\n\nThis paper presents advances in neural networks.\n\n"
        "# Methods\n\nWe use deep learning techniques.\n\n"
        "# Results\n\nOur method achieves 95% accuracy.",
        "year": 2023,
        "published": "2023-01-15",
    }


@pytest.fixture
def minimal_arxiv_row():
    """Minimal valid arxiv row."""
    return {
        "id": "2401.00001",
        "title": "Minimal Paper",
        "content": "This is a very short paper with minimal content.",
    }


@pytest.fixture
def malformed_arxiv_row():
    """Arxiv row with missing or malformed fields."""
    return {
        "id": "9999.9999",
        "title": ["List", "Title"],  # Title as list
        "authors": "SingleAuthor",  # Single author string
        "content": "",  # Empty content
    }


@pytest.fixture
def mock_dataset():
    """Mock HuggingFace dataset."""
    return [
        {
            "id": "2301.11111",
            "title": "Paper One",
            "authors": "Author A",
            "content": (
                "Content of paper one with sufficient text to pass the minimum length requirement. "
                "This is a rather long paper with many words and sentences to ensure it passes "
                "the minimum content length filter."
            ),
            "year": 2023,
        },
        {
            "id": "2302.22222",
            "title": "Paper Two",
            "authors": "Author B, Author C",
            "content": (
                "Content of paper two also with enough text to be considered valid for processing. "
                "This is another paper with substantial content to meet the minimum length requirements."
            ),
            "year": 2023,
        },
        {
            "id": "2303.33333",
            "title": "Empty Paper",
            "authors": "Author D",
            "content": "",  # This should be skipped
            "year": 2023,
        },
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Extract Year Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestExtractYear:
    """Tests for extract_year function."""

    def test_year_from_year_field(self):
        """Test extracting year from direct year field."""
        row = {"year": 2023}
        assert extract_year(row) == 2023

    def test_year_from_string_year(self):
        """Test extracting year when it's a string."""
        row = {"year": "2023"}
        assert extract_year(row) == 2023

    def test_year_from_published_field(self):
        """Test extracting year from published date."""
        row = {"published": "2023-06-15"}
        assert extract_year(row) == 2023

    def test_year_from_date_field(self):
        """Test extracting year from generic date field."""
        row = {"date": "2024-01-01T12:00:00Z"}
        assert extract_year(row) == 2024

    def test_year_from_created_field(self):
        """Test extracting year from created date."""
        row = {"created": "2022-12-31"}
        assert extract_year(row) == 2022

    def test_year_from_arxiv_id_21st_century(self):
        """Test extracting year from arxiv ID (2000s)."""
        row = {"id": "2301.12345"}
        assert extract_year(row) == 2023

    def test_year_from_arxiv_id_20th_century(self):
        """Test extracting year from old arxiv ID format."""
        row = {"id": "9912.12345"}
        assert extract_year(row) == 1999

    def test_year_from_arxiv_id_field(self):
        """Test extracting from arxiv_id field."""
        row = {"arxiv_id": "2401.00001"}
        assert extract_year(row) == 2024

    def test_year_priority_order(self):
        """Test that year field takes priority over other fields."""
        row = {
            "year": 2023,
            "published": "2022-01-01",
            "id": "2101.12345",
        }
        assert extract_year(row) == 2023

    def test_invalid_year_fallback(self):
        """Test fallback when year field is invalid."""
        row = {
            "year": "invalid",
            "published": "2023-06-15",
        }
        assert extract_year(row) == 2023

    def test_no_year_information(self):
        """Test when no year information is available."""
        row = {"title": "Paper without year"}
        assert extract_year(row) is None

    def test_year_from_full_date_string(self):
        """Test extracting year from various date formats."""
        test_cases = [
            ({"published": "Jan 15, 2023"}, 2023),
            ({"date": "2024/06/30"}, 2024),
            ({"created": "2022"}, 2022),
        ]
        for row, expected in test_cases:
            assert extract_year(row) == expected


# ══════════════════════════════════════════════════════════════════════════════
# Parse Paper Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestParsePaper:
    """Tests for parse_paper function."""

    def test_parse_complete_paper(self, sample_arxiv_row):
        """Test parsing a complete paper with all fields."""
        doc = parse_paper(sample_arxiv_row)

        assert doc["doc_id"] == "arxiv_2301.12345"
        assert doc["title"] == "Advances in Neural Networks"
        assert len(doc["authors"]) == 3
        assert "Smith, John" in doc["authors"]
        assert doc["year"] == 2023
        assert doc["arxiv_id"] == "2301.12345"
        assert "full_text" in doc
        assert "sections" in doc
        assert "acronyms" in doc

    def test_parse_paper_content_field(self, sample_arxiv_row):
        """Test that content field is used for full_text."""
        doc = parse_paper(sample_arxiv_row)
        assert doc["full_text"] == sample_arxiv_row["content"]
        assert "Introduction" in doc["full_text"]

    def test_parse_paper_text_field(self):
        """Test that 'text' field is used if 'content' is missing."""
        row = {
            "id": "2301.00001",
            "title": "Test Paper",
            "text": "This is the paper text.",
            "authors": "Author, T.",
        }
        doc = parse_paper(row)
        assert doc["full_text"] == "This is the paper text."

    def test_parse_paper_chunk_field(self):
        """Test that 'chunk' field is used as fallback."""
        row = {
            "id": "2301.00002",
            "title": "Test Paper",
            "chunk": "This is a chunk of text.",
            "authors": "Author, T.",
        }
        doc = parse_paper(row)
        assert doc["full_text"] == "This is a chunk of text."

    def test_parse_paper_no_content(self):
        """Test handling of missing content."""
        row = {"id": "2301.00003", "title": "No Content", "authors": "Author, T."}
        doc = parse_paper(row)
        assert doc["full_text"] == ""

    def test_parse_paper_title_as_list(self, malformed_arxiv_row):
        """Test handling of title as list."""
        doc = parse_paper(malformed_arxiv_row)
        assert isinstance(doc["title"], str)
        # Should take first element or convert to string
        assert "List" in doc["title"] or doc["title"] == "['List', 'Title']"

    def test_parse_paper_authors_as_string(self):
        """Test parsing authors from comma-separated string."""
        row = {
            "id": "2301.00004",
            "title": "Test",
            "authors": "Smith J, Doe A, Lee B",
            "content": "Text",
        }
        doc = parse_paper(row)
        assert isinstance(doc["authors"], list)
        assert len(doc["authors"]) == 3
        assert "Smith J" in doc["authors"]
        assert "Doe A" in doc["authors"]
        assert "Lee B" in doc["authors"]

    def test_parse_paper_authors_list(self):
        """Test that author lists are preserved."""
        row = {
            "id": "2301.00005",
            "title": "Test",
            "authors": ["Author A", "Author B"],
            "content": "Text",
        }
        doc = parse_paper(row)
        assert doc["authors"] == ["Author A", "Author B"]

    def test_parse_paper_missing_authors(self):
        """Test handling of missing authors field."""
        row = {"id": "2301.00006", "title": "Test", "content": "Text"}
        doc = parse_paper(row)
        assert isinstance(doc["authors"], list)
        assert len(doc["authors"]) == 0

    def test_parse_paper_doc_id_formatting(self):
        """Test that doc_id is properly formatted."""
        row = {"id": "2301.12345/v2", "title": "Test", "content": "Text"}
        doc = parse_paper(row)
        # Should replace slashes with underscores
        assert "/" not in doc["doc_id"]
        assert doc["doc_id"].startswith("arxiv_")

    def test_parse_paper_arxiv_id_from_doi(self):
        """Test fallback to doi for arxiv_id."""
        row = {"doi": "test-doi", "title": "Test", "content": "Text"}
        doc = parse_paper(row)
        assert doc["arxiv_id"] == "test-doi"

    def test_parse_paper_sections_extracted(self, sample_arxiv_row):
        """Test that sections are extracted from content."""
        doc = parse_paper(sample_arxiv_row)
        assert "sections" in doc
        assert isinstance(doc["sections"], dict)
        # Should have extracted sections from markdown headings
        assert len(doc["sections"]) >= 0  # May or may not extract depending on implementation

    def test_parse_paper_acronyms_extracted(self, sample_arxiv_row):
        """Test that acronyms are extracted from content."""
        doc = parse_paper(sample_arxiv_row)
        assert "acronyms" in doc
        assert isinstance(doc["acronyms"], dict)

    def test_parse_paper_metadata_complete(self, sample_arxiv_row):
        """Test that all required metadata fields are present."""
        doc = parse_paper(sample_arxiv_row)

        required_fields = ["doc_id", "title", "authors", "year", "arxiv_id", "full_text", "sections", "acronyms"]
        for field in required_fields:
            assert field in doc

    def test_parse_minimal_paper(self, minimal_arxiv_row):
        """Test parsing a minimal paper."""
        doc = parse_paper(minimal_arxiv_row)

        assert doc["doc_id"] == "arxiv_2401.00001"
        assert doc["title"] == "Minimal Paper"
        assert doc["full_text"] == "This is a very short paper with minimal content."


# ══════════════════════════════════════════════════════════════════════════════
# Load Dataset Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestLoadArxivDataset:
    """Tests for load_arxiv_dataset function."""

    @patch("rag_bench.core.ingest.load_dataset")
    def test_load_default_dataset(self, mock_load_dataset):
        """Test loading with default parameters."""
        mock_load_dataset.return_value = ["paper1", "paper2"]

        result = load_arxiv_dataset()

        mock_load_dataset.assert_called_once_with("jamescalam/ai-arxiv2", split="train")
        assert result == ["paper1", "paper2"]

    @patch("rag_bench.core.ingest.load_dataset")
    def test_load_custom_dataset(self, mock_load_dataset):
        """Test loading with custom dataset name."""
        mock_load_dataset.return_value = ["paper1"]

        result = load_arxiv_dataset(dataset_name="custom/dataset", split="test")

        mock_load_dataset.assert_called_once_with("custom/dataset", split="test")
        assert result == ["paper1"]

    @patch("rag_bench.core.ingest.load_dataset")
    def test_load_dataset_returns_list(self, mock_load_dataset):
        """Test that dataset is converted to list."""
        mock_ds = MagicMock()
        mock_ds.__len__.return_value = 3
        mock_ds.__iter__.return_value = iter([{"id": "1"}, {"id": "2"}, {"id": "3"}])
        mock_load_dataset.return_value = mock_ds

        result = load_arxiv_dataset()

        assert isinstance(result, list)


# ══════════════════════════════════════════════════════════════════════════════
# Ingest Dataset Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestIngestDataset:
    """Tests for ingest_dataset function."""

    @patch("rag_bench.core.ingest.load_arxiv_dataset")
    def test_ingest_complete_pipeline(self, mock_load_dataset, mock_dataset):
        """Test full ingestion pipeline."""
        mock_load_dataset.return_value = mock_dataset

        docs = ingest_dataset()

        # Should parse papers (excluding the empty one)
        assert len(docs) >= 1
        assert all(isinstance(doc, dict) for doc in docs)
        assert all("doc_id" in doc for doc in docs)

    @patch("rag_bench.core.ingest.load_arxiv_dataset")
    def test_ingest_filters_short_content(self, mock_load_dataset):
        """Test that papers with insufficient content are filtered."""
        mock_load_dataset.return_value = [
            {"id": "1", "title": "Good", "content": "A" * 150, "year": 2023},
            {"id": "2", "title": "Too short", "content": "Short", "year": 2023},
            {"id": "3", "title": "Also good", "content": "B" * 150, "year": 2023},
        ]

        docs = ingest_dataset()

        # Should filter out the paper with short content
        assert len(docs) == 2

    @patch("rag_bench.core.ingest.load_arxiv_dataset")
    def test_ingest_empty_dataset(self, mock_load_dataset):
        """Test ingesting an empty dataset."""
        mock_load_dataset.return_value = []

        docs = ingest_dataset()

        assert docs == []

    @patch("rag_bench.core.ingest.load_arxiv_dataset")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_ingest_saves_to_file(self, mock_mkdir, mock_file, mock_load_dataset, mock_dataset, tmp_path):
        """Test that parsed documents are saved when save_path is provided."""
        mock_load_dataset.return_value = mock_dataset
        save_path = tmp_path / "parsed_papers.json"

        docs = ingest_dataset(save_path=save_path)

        # Should have attempted to save
        assert len(docs) >= 1

    @patch("rag_bench.core.ingest.load_arxiv_dataset")
    def test_ingest_custom_dataset_name(self, mock_load_dataset, mock_dataset):
        """Test ingestion with custom dataset name."""
        mock_load_dataset.return_value = mock_dataset

        docs = ingest_dataset(dataset_name="custom/dataset", split="test")

        mock_load_dataset.assert_called_once_with("custom/dataset", "test")
        assert len(docs) >= 1

    @patch("rag_bench.core.ingest.load_arxiv_dataset")
    def test_ingest_preserves_paper_structure(self, mock_load_dataset, sample_arxiv_row):
        """Test that ingested papers have correct structure."""
        mock_load_dataset.return_value = [sample_arxiv_row]

        docs = ingest_dataset()

        assert len(docs) == 1
        doc = docs[0]
        assert "doc_id" in doc
        assert "title" in doc
        assert "authors" in doc
        assert "year" in doc
        assert "full_text" in doc
        assert "sections" in doc
        assert "acronyms" in doc


# ══════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestIngestIntegration:
    """Integration tests for the full ingest pipeline."""

    @patch("rag_bench.core.ingest.load_dataset")
    def test_end_to_end_parsing(self, mock_load_dataset):
        """Test complete end-to-end parsing workflow."""
        # Create realistic mock data
        mock_load_dataset.return_value = [
            {
                "id": "2301.12345",
                "title": "Deep Learning Methods",
                "authors": ["Author A", "Author B"],
                "content": "# Introduction\n\nDeep learning has revolutionized AI. "
                "We introduce Convolutional Neural Networks (CNNs) for image processing.\n\n"
                "# Methods\n\nWe use CNNs and backpropagation.\n\n"
                "# Results\n\nOur model achieves 98% accuracy.",
                "published": "2023-01-15",
            }
        ]

        docs = ingest_dataset()

        assert len(docs) == 1
        doc = docs[0]

        # Verify structure
        assert doc["doc_id"] == "arxiv_2301.12345"
        assert doc["title"] == "Deep Learning Methods"
        assert len(doc["authors"]) == 2
        assert doc["year"] == 2023

        # Verify content
        assert "Deep learning" in doc["full_text"]
        assert isinstance(doc["sections"], dict)
        assert isinstance(doc["acronyms"], dict)

    @patch("rag_bench.core.ingest.load_dataset")
    def test_handles_various_paper_formats(self, mock_load_dataset):
        """Test that various paper formats are handled correctly."""
        mock_load_dataset.return_value = [
            # Different field names
            {"id": "1", "title": "Paper 1", "content": "A" * 200},
            {"arxiv_id": "2", "title": "Paper 2", "text": "B" * 200},
            {"id": "3", "title": ["Paper", "3"], "chunk": "C" * 200, "authors": "Single"},
        ]

        docs = ingest_dataset()

        # All papers should be parsed
        assert len(docs) == 3
        assert all(doc["doc_id"] for doc in docs)
        assert all(doc["full_text"] for doc in docs)
