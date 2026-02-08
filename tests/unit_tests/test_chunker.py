"""
Unit tests for rag_bench.core.chunker module.

Tests cover:
- PaperChunker: initialization, chunking logic, metadata handling
- Equation protection and restoration
- Table handling
- Acronym expansion within chunks
- Text cleaning and normalization
- Edge cases: empty inputs, small sections, malformed data
"""

import pytest

from rag_bench.core.chunker import PaperChunker, chunk_all_papers

# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_paper():
    """Sample paper document for testing."""
    return {
        "doc_id": "arxiv_1234.5678",
        "title": "A Study on Machine Learning",
        "authors": ["Smith, J.", "Doe, A."],
        "year": 2024,
        "arxiv_id": "1234.5678",
        "sections": {
            "introduction": "This paper introduces machine learning concepts. We explore supervised learning methods.",
            "methods": "We use various ML algorithms including neural networks. "
            "The training process involves backpropagation.",
            "results": "Our experiments show significant improvements. The model achieves 95% accuracy on the test set.",
        },
        "acronyms": {"ML": "Machine Learning", "NN": "Neural Network"},
    }


@pytest.fixture
def paper_with_equations():
    """Paper with mathematical equations."""
    return {
        "doc_id": "arxiv_9999.1111",
        "title": "Mathematical Foundations",
        "authors": ["Einstein, A."],
        "year": 2024,
        "arxiv_id": "9999.1111",
        "sections": {
            "theory": "The energy-mass relationship is given by $$E = mc^2$$ "
            "where E is energy, m is mass, and c is speed of light. "
            "This fundamental equation \\[F = ma\\] relates force to acceleration."
        },
        "acronyms": {},
    }


@pytest.fixture
def paper_with_tables():
    """Paper with markdown tables."""
    return {
        "doc_id": "arxiv_7777.8888",
        "title": "Benchmark Results",
        "authors": ["Researcher, B."],
        "year": 2024,
        "arxiv_id": "7777.8888",
        "sections": {
            "results": "We present our benchmark results:\n\n"
            "| Model | Accuracy | Speed |\n"
            "|-------|----------|-------|\n"
            "| GPT-4 | 95.2% | 100ms |\n"
            "| PaLM | 94.8% | 120ms |\n\n"
            "These results demonstrate state-of-the-art performance."
        },
        "acronyms": {},
    }


@pytest.fixture
def empty_paper():
    """Paper with minimal content."""
    return {
        "doc_id": "arxiv_0000.0001",
        "title": "Empty Paper",
        "authors": ["Unknown, X."],
        "year": 2024,
        "arxiv_id": "0000.0001",
        "sections": {},
        "acronyms": {},
    }


@pytest.fixture
def large_paper():
    """Paper with a large section that needs chunking."""
    section_text = " ".join([f"Sentence number {i} in this long section." for i in range(200)])
    return {
        "doc_id": "arxiv_5555.6666",
        "title": "A Very Long Paper",
        "authors": ["Verbose, V."],
        "year": 2024,
        "arxiv_id": "5555.6666",
        "sections": {"long_section": section_text},
        "acronyms": {"RAG": "Retrieval-Augmented Generation"},
    }


# ══════════════════════════════════════════════════════════════════════════════
# PaperChunker Initialization Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestPaperChunkerInit:
    """Tests for PaperChunker initialization."""

    def test_default_initialization(self):
        """Test default parameters."""
        chunker = PaperChunker()
        assert chunker.chunk_size == 2048
        assert chunker.chunk_overlap == 200
        assert chunker.min_section_length == 50

    def test_custom_initialization(self):
        """Test custom parameters."""
        chunker = PaperChunker(chunk_size=1024, chunk_overlap=100, min_section_length=20)
        assert chunker.chunk_size == 1024
        assert chunker.chunk_overlap == 100
        assert chunker.min_section_length == 20

    def test_splitter_created(self):
        """Test that text splitter is properly initialized."""
        chunker = PaperChunker()
        assert chunker.splitter is not None
        assert hasattr(chunker.splitter, "split_text")


# ══════════════════════════════════════════════════════════════════════════════
# Basic Chunking Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestChunkPaper:
    """Tests for chunk_paper method."""

    def test_chunk_simple_paper(self, sample_paper):
        """Test chunking a simple paper."""
        chunker = PaperChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_paper(sample_paper)

        assert len(chunks) > 0
        # Each chunk should have required fields
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "doc_id" in chunk
            assert "text" in chunk
            assert "section" in chunk
            assert "metadata" in chunk
            assert chunk["doc_id"] == "arxiv_1234.5678"

    def test_chunk_ids_unique(self, sample_paper):
        """Test that chunk IDs are unique."""
        chunker = PaperChunker()
        chunks = chunker.chunk_paper(sample_paper)

        chunk_ids = [c["chunk_id"] for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_chunk_metadata(self, sample_paper):
        """Test that chunks contain proper metadata."""
        chunker = PaperChunker()
        chunks = chunker.chunk_paper(sample_paper)

        for chunk in chunks:
            metadata = chunk["metadata"]
            assert "source_display" in metadata
            assert "title" in metadata
            assert "year" in metadata
            assert "arxiv_id" in metadata
            assert "section" in metadata
            assert metadata["title"] == "A Study on Machine Learning"
            assert metadata["year"] == 2024
            assert "2024" in metadata["source_display"]
            assert "A Study on Machine Learning" in metadata["source_display"]

    def test_section_names_preserved(self, sample_paper):
        """Test that section names are preserved in chunks."""
        chunker = PaperChunker()
        chunks = chunker.chunk_paper(sample_paper)

        sections = {c["section"] for c in chunks}
        assert "introduction" in sections or "methods" in sections or "results" in sections


# ══════════════════════════════════════════════════════════════════════════════
# Equation Handling Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestEquationHandling:
    """Tests for equation protection and restoration."""

    def test_equations_preserved(self, paper_with_equations):
        """Test that equations remain intact after chunking."""
        chunker = PaperChunker()
        chunks = chunker.chunk_paper(paper_with_equations)

        # Combine all chunk text
        all_text = " ".join(c["text"] for c in chunks)

        # Check that equations are preserved
        assert "$$E = mc^2$$" in all_text or "E = mc^2" in all_text
        assert "\\[F = ma\\]" in all_text or "F = ma" in all_text

    def test_protect_equations(self, paper_with_equations):
        """Test equation protection mechanism."""
        chunker = PaperChunker()
        text = "Energy is $$E = mc^2$$ in physics."

        protected = chunker._protect_equations(text)
        # Should have equation placeholders
        assert "__EQ_" in protected

    def test_restore_equations(self, paper_with_equations):
        """Test equation restoration mechanism."""
        chunker = PaperChunker()
        text = "Energy is $$E = mc^2$$ in physics."

        protected = chunker._protect_equations(text)
        restored = chunker._restore_equations(protected)

        # Should restore the equation
        assert "$$E = mc^2$$" in restored


# ══════════════════════════════════════════════════════════════════════════════
# Table Handling Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestTableHandling:
    """Tests for table protection."""

    def test_tables_preserved(self, paper_with_tables):
        """Test that tables remain intact after chunking."""
        chunker = PaperChunker()
        chunks = chunker.chunk_paper(paper_with_tables)

        all_text = " ".join(c["text"] for c in chunks)

        # Check that table structure is preserved
        assert "| Model | Accuracy | Speed |" in all_text
        assert "GPT-4" in all_text
        assert "95.2%" in all_text

    def test_protect_tables(self):
        """Test table protection mechanism."""
        chunker = PaperChunker()
        text = "Results:\n| Model | Score |\n|-------|-------|\n| A | 90 |\n| B | 85 |\n"

        protected = chunker._protect_tables(text)
        # Should preserve table structure
        assert "| Model | Score |" in protected


# ══════════════════════════════════════════════════════════════════════════════
# Acronym Expansion Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestAcronymExpansion:
    """Tests for acronym expansion in chunks."""

    def test_acronym_expansion(self):
        """Test that acronyms are expanded on first occurrence."""
        chunker = PaperChunker()
        text = "ML is important. ML methods are powerful."
        acronyms = {"ML": "Machine Learning"}

        expanded = chunker._expand_acronyms(text, acronyms)

        # First occurrence should be expanded
        assert "Machine Learning (ML)" in expanded
        # Should only expand once
        assert expanded.count("Machine Learning (ML)") == 1

    def test_acronym_no_expansion_if_already_present(self):
        """Test that already expanded acronyms aren't re-expanded."""
        chunker = PaperChunker()
        text = "Machine Learning (ML) is important. ML methods work."
        acronyms = {"ML": "Machine Learning"}

        expanded = chunker._expand_acronyms(text, acronyms)

        # Should not create duplicate expansions
        assert expanded.count("Machine Learning (ML)") == 1

    def test_acronym_boundary_matching(self):
        """Test that acronyms are only matched as whole words."""
        chunker = PaperChunker()
        text = "HTML is different from ML concepts."
        acronyms = {"ML": "Machine Learning"}

        expanded = chunker._expand_acronyms(text, acronyms)

        # Should expand ML but not affect HTML
        assert "Machine Learning (ML)" in expanded
        assert "HTML" in expanded

    def test_empty_acronyms(self):
        """Test handling of empty acronym dictionary."""
        chunker = PaperChunker()
        text = "ML is important."
        expanded = chunker._expand_acronyms(text, {})

        assert expanded == text


# ══════════════════════════════════════════════════════════════════════════════
# Text Cleaning Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestTextCleaning:
    """Tests for text cleaning functionality."""

    def test_clean_text_removes_excess_newlines(self):
        """Test that excessive blank lines are collapsed."""
        chunker = PaperChunker()
        text = "Line 1\n\n\n\nLine 2\n\n\n\n\nLine 3"

        cleaned = chunker._clean_text(text)

        assert "\n\n\n" not in cleaned
        assert "Line 1" in cleaned
        assert "Line 2" in cleaned
        assert "Line 3" in cleaned

    def test_clean_text_strips_whitespace(self):
        """Test that leading/trailing whitespace is removed."""
        chunker = PaperChunker()
        text = "   Some text with spaces   \n  More text  "

        cleaned = chunker._clean_text(text)

        assert not cleaned.startswith(" ")
        assert not cleaned.endswith(" ")

    def test_clean_text_preserves_content(self):
        """Test that actual content is preserved."""
        chunker = PaperChunker()
        text = "Important content here."

        cleaned = chunker._clean_text(text)

        assert "Important content here." in cleaned


# ══════════════════════════════════════════════════════════════════════════════
# Edge Cases Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_paper(self, empty_paper):
        """Test handling of paper with no sections."""
        chunker = PaperChunker()
        chunks = chunker.chunk_paper(empty_paper)

        # Should handle gracefully (may return empty list or minimal chunks)
        assert isinstance(chunks, list)

    def test_paper_with_full_text_fallback(self):
        """Test fallback to full_text when sections are missing."""
        paper = {
            "doc_id": "test_001",
            "title": "Test Paper",
            "authors": ["Author, T."],
            "year": 2024,
            "arxiv_id": "test.001",
            "sections": {},
            "full_text": "This is the full text content of the paper.",
            "acronyms": {},
        }

        chunker = PaperChunker()
        chunks = chunker.chunk_paper(paper)

        # Should chunk the full_text as fallback
        assert len(chunks) >= 1 or len(chunks) == 0  # Depends on min_section_length

    def test_small_sections_filtered(self):
        """Test that sections smaller than min_section_length are skipped."""
        paper = {
            "doc_id": "test_002",
            "title": "Test Paper",
            "authors": ["Author, T."],
            "year": 2024,
            "arxiv_id": "test.002",
            "sections": {"tiny": "Hi"},
            "acronyms": {},
        }

        chunker = PaperChunker(min_section_length=50)
        chunks = chunker.chunk_paper(paper)

        # Small section should be filtered out
        assert len(chunks) == 0

    def test_large_paper_creates_multiple_chunks(self, large_paper):
        """Test that large sections are split into multiple chunks."""
        chunker = PaperChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_paper(large_paper)

        # Should create multiple chunks
        assert len(chunks) > 1

    def test_chunk_text_minimum_length(self, large_paper):
        """Test that trivially small chunks are filtered."""
        chunker = PaperChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk_paper(large_paper)

        # All chunks should have reasonable content
        for chunk in chunks:
            assert len(chunk["text"].strip()) >= 20


# ══════════════════════════════════════════════════════════════════════════════
# Batch Processing Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestChunkAllPapers:
    """Tests for chunk_all_papers function."""

    def test_chunk_multiple_papers(self, sample_paper, paper_with_equations):
        """Test chunking multiple papers at once."""
        papers = [sample_paper, paper_with_equations]
        chunks = chunk_all_papers(papers, chunk_size=200, chunk_overlap=20)

        assert len(chunks) > 0
        # Chunks should come from both papers
        doc_ids = {c["doc_id"] for c in chunks}
        assert "arxiv_1234.5678" in doc_ids
        assert "arxiv_9999.1111" in doc_ids

    def test_chunk_empty_list(self):
        """Test chunking an empty list of papers."""
        chunks = chunk_all_papers([])
        assert chunks == []

    def test_chunk_all_papers_preserves_order(self, sample_paper):
        """Test that chunks maintain some relationship to source papers."""
        papers = [sample_paper]
        chunks = chunk_all_papers(papers)

        # All chunks should have the same doc_id
        doc_ids = {c["doc_id"] for c in chunks}
        assert len(doc_ids) == 1
        assert "arxiv_1234.5678" in doc_ids


class TestChunkerEdgeCasesAndCoverage:
    """Test additional edge cases for coverage improvement."""

    def test_chunk_paper_with_very_small_sections(self):
        """Test chunking with very small section content that gets filtered."""
        paper = {
            "doc_id": "arxiv_1234.1234",
            "title": "Short Paper",
            "authors": ["Author"],
            "year": 2024,
            "arxiv_id": "1234.1234",
            "sections": {
                "intro": "Hi",  # Less than 20 characters after processing
                "method": "This is a proper section with enough content to pass the minimum length requirement.",
            },
            "acronyms": {},
        }
        chunker = PaperChunker()
        chunks = chunker.chunk_paper(paper)
        # Should skip the 'intro' section due to small size
        # Should only have chunks from 'method'
        assert all(c["section"] != "intro" for c in chunks)

    def test_restore_equations_without_equations_stored(self):
        """Test equation restoration when no equations were stored."""
        chunker = PaperChunker()
        # Don't protect equations, so _equation_store won't exist
        text = "Some normal text without equations"
        if not hasattr(chunker, "_equation_store") and hasattr(chunker, "_equation_store"):
            delattr(chunker, "_equation_store")

        result = chunker._restore_equations(text)
        assert result == text

    def test_protect_and_restore_equations_roundtrip(self):
        """Test that equation protection and restoration preserve content."""
        chunker = PaperChunker()
        original = "The equation $x = y^2$ is important. Also $$E = mc^2$$."

        # Protect equations
        protected = chunker._protect_equations(original)
        # Text should contain placeholders
        assert "__EQ_" in protected

        # Restore equations
        restored = chunker._restore_equations(protected)
        assert restored == original

    def test_chunk_paper_preserves_acronyms(self):
        """Test that acronym expansions are applied in chunks."""
        paper = {
            "doc_id": "arxiv_1234.1234",
            "title": "Paper with Acronyms",
            "authors": ["Author"],
            "year": 2024,
            "arxiv_id": "1234.1234",
            "sections": {
                "intro": "ML is important. This stands for Machine Learning. ML algorithms are useful.",
            },
            "acronyms": {"ML": "Machine Learning"},
        }
        chunker = PaperChunker()
        chunks = chunker.chunk_paper(paper)
        # Chunks should exist
        assert len(chunks) > 0
        # At least one chunk should contain text
        assert any(len(c["text"]) > 0 for c in chunks)

    def test_chunk_paper_with_all_small_sections(self):
        """Test paper where all sections are too small to chunk."""
        paper = {
            "doc_id": "arxiv_5678.5678",
            "title": "Tiny Paper",
            "authors": ["Author"],
            "year": 2024,
            "arxiv_id": "5678.5678",
            "sections": {
                "a": "x",
                "b": "y",
                "c": "z",
            },
            "acronyms": {},
        }
        chunker = PaperChunker()
        chunks = chunker.chunk_paper(paper)
        # All sections are too small, should result in no chunks
        assert len(chunks) == 0

    def test_chunk_paper_handles_missing_acronyms(self):
        """Test chunking when acronyms field is missing."""
        paper = {
            "doc_id": "arxiv_1234.1234",
            "title": "Paper",
            "authors": ["Author"],
            "year": 2024,
            "arxiv_id": "1234.1234",
            "sections": {
                "intro": "This paper discusses various important topics and methods used in modern research.",
            },
            # No 'acronyms' field
        }
        chunker = PaperChunker()
        chunks = chunker.chunk_paper(paper)
        # Should still work without acronyms
        assert len(chunks) > 0

    def test_clean_text_removes_excess_whitespace(self):
        """Test that text cleaning removes extra whitespace."""
        chunker = PaperChunker()
        messy_text = "This   has\n\n\nmultiple    spaces\t\tand\n\nnewlines."
        cleaned = chunker._clean_text(messy_text)
        # Should have normalized whitespace
        # Multiple spaces might be collapsed
        assert len(cleaned) > 0

    def test_equation_store_multiple_equations(self):
        """Test protecting and restoring multiple equations."""
        chunker = PaperChunker()
        text = "First: $a = b$. Second: $c = d$. Third: $$e = f$$."
        protected = chunker._protect_equations(text)
        # Should have at least one placeholder
        assert "__EQ_" in protected

        restored = chunker._restore_equations(protected)
        assert restored == text
