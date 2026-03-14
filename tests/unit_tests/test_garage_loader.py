"""Tests for GaRAGe dataset loader."""

from unittest.mock import patch

import pytest

from rag_bench.eval.garage.loader import (
    GaRAGeEntry,
    GaRAGePassage,
    _cache_entries,
    _load_from_cache,
    _parse_entry,
    load_garage,
)


class TestParseEntry:
    """Test parsing raw HuggingFace rows into GaRAGeEntry objects."""

    def test_basic_parsing(self):
        raw = {
            "id": "test_1",
            "question": "What is attention?",
            "answer": "Attention is a mechanism...",
            "passages": [
                {"text": "Passage 1 text", "is_relevant": True, "id": "p1"},
                {"text": "Passage 2 text", "is_relevant": False, "id": "p2"},
            ],
            "question_tag": "answerable",
        }
        entry = _parse_entry(raw, 0)
        assert entry.id == "test_1"
        assert entry.question == "What is attention?"
        assert entry.gold_answer == "Attention is a mechanism..."
        assert len(entry.passages) == 2
        assert entry.passages[0].is_relevant is True
        assert entry.passages[1].is_relevant is False
        assert entry.should_deflect is False

    def test_unanswerable_detection(self):
        raw = {
            "question": "What is the recipe for cake?",
            "answer": "",
            "question_tag": "unanswerable",
        }
        entry = _parse_entry(raw, 0)
        assert entry.should_deflect is True

    def test_alternative_field_names(self):
        raw = {
            "example_id": "alt_1",
            "query": "Alternative question format",
            "gold_answer": "Alternative answer format",
            "contexts": ["Context text 1", "Context text 2"],
            "relevance_labels": [True, False],
        }
        entry = _parse_entry(raw, 0)
        assert entry.id == "alt_1"
        assert entry.question == "Alternative question format"
        assert entry.gold_answer == "Alternative answer format"
        assert len(entry.passages) == 2

    def test_missing_fields_defaults(self):
        raw = {}
        entry = _parse_entry(raw, 5)
        assert entry.id == "garage_5"
        assert entry.question == ""
        assert entry.gold_answer == ""
        assert len(entry.passages) == 0


class TestCacheOperations:
    """Test caching and loading of entries."""

    def test_round_trip(self, tmp_path):
        """Test that caching and loading preserves data."""
        entries = [
            GaRAGeEntry(
                id="test_1",
                question="What is X?",
                gold_answer="X is Y",
                passages=[
                    GaRAGePassage(text="passage text", is_relevant=True, passage_id="p1"),
                ],
                should_deflect=False,
                question_tag="answerable",
                topic_tag="ml",
            ),
        ]

        # Cache
        cache_file = tmp_path / "test_cache.json"
        with (
            patch("rag_bench.eval.garage.loader.GARAGE_CACHE_FILE", cache_file),
            patch("rag_bench.eval.garage.loader.GARAGE_CACHE_DIR", tmp_path),
        ):
            _cache_entries(entries)
            loaded = _load_from_cache()

        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].id == "test_1"
        assert loaded[0].question == "What is X?"
        assert loaded[0].passages[0].is_relevant is True


class TestLoadGarage:
    """Test the main load_garage function."""

    def test_sampling(self, tmp_path):
        """Test that sample_size limits results."""
        entries = [GaRAGeEntry(id=f"e_{i}", question=f"Q{i}", gold_answer=f"A{i}") for i in range(100)]

        cache_file = tmp_path / "test_cache.json"
        with (
            patch("rag_bench.eval.garage.loader.GARAGE_CACHE_FILE", cache_file),
            patch("rag_bench.eval.garage.loader.GARAGE_CACHE_DIR", tmp_path),
        ):
            _cache_entries(entries)
            loaded = load_garage(sample_size=10, seed=42)

        assert len(loaded) == 10

    def test_zero_sample_returns_all(self, tmp_path):
        entries = [GaRAGeEntry(id=f"e_{i}", question=f"Q{i}", gold_answer=f"A{i}") for i in range(5)]

        cache_file = tmp_path / "test_cache.json"
        with (
            patch("rag_bench.eval.garage.loader.GARAGE_CACHE_FILE", cache_file),
            patch("rag_bench.eval.garage.loader.GARAGE_CACHE_DIR", tmp_path),
        ):
            _cache_entries(entries)
            loaded = load_garage(sample_size=0, seed=42)

        assert len(loaded) == 5

    def test_force_download(self, tmp_path):
        """force_download=True skips cache and loads from HuggingFace."""
        cache_file = tmp_path / "cache.json"
        cache_file.write_text("[]")  # stale cache

        raw_data = [
            {"id": "new_1", "question": "Q?", "answer": "A", "passages": []},
        ]
        with (
            patch("rag_bench.eval.garage.loader.GARAGE_CACHE_FILE", cache_file),
            patch("rag_bench.eval.garage.loader.GARAGE_CACHE_DIR", tmp_path),
            patch("rag_bench.eval.garage.loader._try_load_from_huggingface", return_value=raw_data),
        ):
            loaded = load_garage(force_download=True)
        assert len(loaded) == 1
        assert loaded[0].id == "new_1"

    def test_no_cache_triggers_download(self, tmp_path):
        """When cache doesn't exist, downloads from HuggingFace."""
        cache_file = tmp_path / "nonexistent.json"
        raw_data = [
            {"id": "hf_1", "question": "Q?", "answer": "A"},
        ]
        with (
            patch("rag_bench.eval.garage.loader.GARAGE_CACHE_FILE", cache_file),
            patch("rag_bench.eval.garage.loader.GARAGE_CACHE_DIR", tmp_path),
            patch("rag_bench.eval.garage.loader._try_load_from_huggingface", return_value=raw_data),
        ):
            loaded = load_garage()
        assert len(loaded) == 1

    def test_corrupt_cache_triggers_download(self, tmp_path):
        """Corrupt JSON cache triggers re-download."""
        cache_file = tmp_path / "cache.json"
        cache_file.write_text("{{{bad json")

        raw_data = [{"id": "fix_1", "question": "Q?", "answer": "A"}]
        with (
            patch("rag_bench.eval.garage.loader.GARAGE_CACHE_FILE", cache_file),
            patch("rag_bench.eval.garage.loader.GARAGE_CACHE_DIR", tmp_path),
            patch("rag_bench.eval.garage.loader._try_load_from_huggingface", return_value=raw_data),
        ):
            loaded = load_garage()
        assert len(loaded) == 1


class TestTryLoadFromHuggingFace:
    """Tests for _try_load_from_huggingface."""

    def test_success(self):
        """Successfully loads data from HuggingFace."""
        from rag_bench.eval.garage.loader import _try_load_from_huggingface

        mock_split = [
            {"id": "1", "question": "Q1", "answer": "A1"},
            {"id": "2", "question": "Q2", "answer": "A2"},
        ]
        mock_ds = {"test": mock_split}
        with patch("rag_bench.eval.garage.loader.load_dataset", return_value=mock_ds):
            _try_load_from_huggingface()

    def test_import_error(self):
        """Raises when datasets library not installed."""
        from rag_bench.eval.garage.loader import _try_load_from_huggingface

        with patch("rag_bench.eval.garage.loader._HAS_DATASETS", False), pytest.raises(ImportError):
            _try_load_from_huggingface()

    def test_download_error(self):
        """Raises when HuggingFace download fails."""
        from rag_bench.eval.garage.loader import _try_load_from_huggingface

        with (
            patch("rag_bench.eval.garage.loader.load_dataset", side_effect=RuntimeError("Download failed")),
            pytest.raises(RuntimeError),
        ):
            _try_load_from_huggingface()


class TestParseEntryEdgeCases:
    """Additional edge cases for _parse_entry."""

    def test_string_passages_with_labels(self):
        """Test string passages with separate relevance labels."""
        raw = {
            "id": "e1",
            "question": "Q?",
            "answer": "A",
            "passages": ["passage text one", "passage text two"],
            "passage_labels": [True, False],
        }
        entry = _parse_entry(raw, 0)
        assert len(entry.passages) == 2
        assert entry.passages[0].is_relevant is True
        assert entry.passages[0].text == "passage text one"
        assert entry.passages[1].is_relevant is False

    def test_string_passages_without_labels(self):
        """String passages with no labels defaults to False."""
        raw = {
            "question": "Q?",
            "answer": "A",
            "passages": ["one", "two"],
        }
        entry = _parse_entry(raw, 0)
        assert entry.passages[0].is_relevant is False
        assert entry.passages[1].is_relevant is False

    def test_should_deflect_via_answerable_false(self):
        """answerable=False triggers should_deflect."""
        raw = {"question": "Q?", "answer": "", "answerable": False}
        entry = _parse_entry(raw, 0)
        assert entry.should_deflect is True

    def test_should_deflect_via_explicit_flag(self):
        """should_deflect=True in raw data."""
        raw = {"question": "Q?", "answer": "", "should_deflect": True}
        entry = _parse_entry(raw, 0)
        assert entry.should_deflect is True

    def test_no_answer_category(self):
        """no_answer and deflect categories trigger should_deflect."""
        for tag in ("no_answer", "deflect"):
            raw = {"question": "Q?", "answer": "", "question_tag": tag}
            entry = _parse_entry(raw, 0)
            assert entry.should_deflect is True, f"Tag '{tag}' should trigger deflect"

    def test_reference_answer_field(self):
        """Fallback to 'reference_answer' field."""
        raw = {"question": "Q?", "reference_answer": "ref ans"}
        entry = _parse_entry(raw, 0)
        assert entry.gold_answer == "ref ans"

    def test_dict_passage_with_content_key(self):
        """Dict passage with 'content' key."""
        raw = {
            "question": "Q?",
            "answer": "A",
            "passages": [{"content": "passage content", "relevant": True, "passage_id": "p99"}],
        }
        entry = _parse_entry(raw, 0)
        assert entry.passages[0].text == "passage content"
        assert entry.passages[0].is_relevant is True
        assert entry.passages[0].passage_id == "p99"

    def test_topic_and_domain_fields(self):
        """Test topic_tag from various field names."""
        raw = {"question": "Q?", "answer": "A", "domain": "NLP"}
        entry = _parse_entry(raw, 0)
        assert entry.topic_tag == "NLP"

        raw2 = {"question": "Q?", "answer": "A", "topic": "CV"}
        entry2 = _parse_entry(raw2, 0)
        assert entry2.topic_tag == "CV"
