"""Tests for RAGTruth dataset loader."""

from unittest.mock import MagicMock, patch

import pytest

from rag_bench.eval.ragtruth.loader import (
    HallucinationSpan,
    RAGTruthEntry,
    _cache_entries,
    _extract_source_text,
    _load_from_cache,
    _merge_and_parse,
    _parse_hallucination_spans,
    load_ragtruth,
)


class TestParseHallucinationSpans:
    """Test parsing hallucination span annotations."""

    def test_dict_spans(self):
        raw = [
            {"text": "hallucinated text", "start": 10, "end": 27, "label_type": "Evident Conflict"},
            {"hallucinated_text": "another span", "start_idx": 30, "end_idx": 42, "type": "Subtle Baseless"},
        ]
        spans = _parse_hallucination_spans(raw)
        assert len(spans) == 2
        assert spans[0].text == "hallucinated text"
        assert spans[0].label_type == "Evident Conflict"
        assert spans[1].text == "another span"
        assert spans[1].label_type == "Subtle Baseless"

    def test_string_spans(self):
        raw = ["span one", "span two"]
        spans = _parse_hallucination_spans(raw)
        assert len(spans) == 2
        assert spans[0].text == "span one"

    def test_empty_spans(self):
        spans = _parse_hallucination_spans([])
        assert len(spans) == 0

    def test_none_spans(self):
        spans = _parse_hallucination_spans(None)
        assert len(spans) == 0


class TestMergeAndParse:
    """Test merging source info with responses."""

    def test_basic_merge(self):
        source_info = [
            {"source_id": "s1", "source_info": "This is the context", "task_type": "QA"},
        ]
        responses = [
            {
                "response_id": "r1",
                "source_id": "s1",
                "response": "This is the answer",
                "prompt": "What is it?",
                "hallucination_spans": [{"text": "fabricated info", "label_type": "Evident Baseless"}],
                "has_hallucination": True,
            },
        ]
        entries = _merge_and_parse(source_info, responses)
        assert len(entries) == 1
        assert entries[0].id == "r1"
        assert entries[0].source_info == "This is the context"
        assert entries[0].has_hallucination is True
        assert len(entries[0].hallucination_spans) == 1

    def test_missing_source(self):
        """Response with no matching source should still parse."""
        responses = [
            {
                "response_id": "r1",
                "source_id": "missing",
                "response": "Answer text",
                "prompt": "Question?",
                "context": "Inline context",
            },
        ]
        entries = _merge_and_parse([], responses)
        assert len(entries) == 1
        assert entries[0].source_info == "Inline context"


class TestCacheOperations:
    """Test caching and loading of RAGTruth entries."""

    def test_round_trip(self, tmp_path):
        entries = [
            RAGTruthEntry(
                id="rt_1",
                source_id="s1",
                task_type="QA",
                source_info="context text",
                prompt="What is X?",
                reference_response="X is Y with some hallucinated info",
                hallucination_spans=[
                    HallucinationSpan(text="hallucinated info", start=20, end=37, label_type="Evident Baseless"),
                ],
                has_hallucination=True,
            ),
        ]

        cache_file = tmp_path / "test_cache.json"
        with (
            patch("rag_bench.eval.ragtruth.loader.RAGTRUTH_CACHE_FILE", cache_file),
            patch("rag_bench.eval.ragtruth.loader.RAGTRUTH_CACHE_DIR", tmp_path),
        ):
            _cache_entries(entries)
            loaded = _load_from_cache()

        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0].id == "rt_1"
        assert loaded[0].has_hallucination is True
        assert loaded[0].hallucination_spans[0].label_type == "Evident Baseless"


class TestLoadRagtruth:
    """Test the main load_ragtruth function."""

    def test_sampling(self, tmp_path):
        entries = [
            RAGTruthEntry(
                id=f"rt_{i}",
                source_id=f"s_{i}",
                task_type="QA",
                source_info=f"context {i}",
                prompt=f"question {i}",
                reference_response=f"answer {i}",
            )
            for i in range(50)
        ]

        cache_file = tmp_path / "test_cache.json"
        with (
            patch("rag_bench.eval.ragtruth.loader.RAGTRUTH_CACHE_FILE", cache_file),
            patch("rag_bench.eval.ragtruth.loader.RAGTRUTH_CACHE_DIR", tmp_path),
        ):
            _cache_entries(entries)
            loaded = load_ragtruth(sample_size=10, task_type="QA", seed=42)

        assert len(loaded) == 10

    def test_task_type_filter(self, tmp_path):
        entries = [
            RAGTruthEntry(id="qa_1", source_id="s1", task_type="QA", source_info="", prompt="", reference_response=""),
            RAGTruthEntry(
                id="sum_1", source_id="s2", task_type="Summarization", source_info="", prompt="", reference_response=""
            ),
            RAGTruthEntry(id="qa_2", source_id="s3", task_type="QA", source_info="", prompt="", reference_response=""),
        ]

        cache_file = tmp_path / "test_cache.json"
        with (
            patch("rag_bench.eval.ragtruth.loader.RAGTRUTH_CACHE_FILE", cache_file),
            patch("rag_bench.eval.ragtruth.loader.RAGTRUTH_CACHE_DIR", tmp_path),
        ):
            _cache_entries(entries)
            loaded = load_ragtruth(task_type="QA")

        assert len(loaded) == 2
        assert all(e.task_type == "QA" for e in loaded)


class TestExtractSourceText:
    """Test _extract_source_text helper."""

    def test_string_source_info(self):
        assert _extract_source_text({"source_info": "text here"}) == "text here"

    def test_dict_source_info_with_passages(self):
        data = {"source_info": {"question": "q", "passages": "some passages"}}
        assert _extract_source_text(data) == "some passages"

    def test_dict_source_info_with_text(self):
        data = {"source_info": {"text": "the text"}}
        assert _extract_source_text(data) == "the text"

    def test_context_fallback(self):
        data = {"context": "context text"}
        assert _extract_source_text(data) == "context text"

    def test_text_fallback(self):
        data = {"text": "raw text"}
        assert _extract_source_text(data) == "raw text"

    def test_empty_dict(self):
        assert _extract_source_text({}) == ""


class TestMergeAndParseAdvanced:
    """Additional merge and parse tests for edge cases."""

    def test_alternative_field_names(self):
        """Test responses with alternative field naming."""
        responses = [
            {
                "id": "r1",
                "source_id": "s1",
                "generated_text": "Generated answer text",
                "question": "What is X?",
                "hallucinations": [{"text": "bad stuff", "hallucination_type": "Evident Conflict"}],
                "is_hallucinated": True,
                "type": "QA",
            },
        ]
        entries = _merge_and_parse([], responses)
        assert len(entries) == 1
        assert entries[0].reference_response == "Generated answer text"
        assert entries[0].prompt == "What is X?"
        assert entries[0].has_hallucination is True
        assert len(entries[0].hallucination_spans) == 1
        assert entries[0].hallucination_spans[0].label_type == "Evident Conflict"

    def test_multiple_responses_one_source(self):
        """Multiple responses share one source."""
        sources = [{"source_id": "s1", "source_info": "Shared context"}]
        responses = [
            {"response_id": "r1", "source_id": "s1", "response": "A1", "prompt": "Q1"},
            {"response_id": "r2", "source_id": "s1", "response": "A2", "prompt": "Q2"},
        ]
        entries = _merge_and_parse(sources, responses)
        assert len(entries) == 2
        assert entries[0].source_info == "Shared context"
        assert entries[1].source_info == "Shared context"

    def test_response_with_no_id(self):
        """Response without id gets auto-generated id."""
        responses = [{"response": "ans", "prompt": "q"}]
        entries = _merge_and_parse([], responses)
        assert len(entries) == 1
        assert entries[0].id == "ragtruth_0"


class TestLoadRagtruthAdvanced:
    """Additional load_ragtruth tests."""

    def test_no_cache_triggers_download(self, tmp_path):
        """When cache doesn't exist, tries to download."""
        cache_file = tmp_path / "nonexistent.json"
        source_info = [{"source_id": "s1", "source_info": "ctx", "task_type": "QA"}]
        responses = [
            {"response_id": "r1", "source_id": "s1", "response": "a", "prompt": "q"},
        ]

        with (
            patch("rag_bench.eval.ragtruth.loader.RAGTRUTH_CACHE_FILE", cache_file),
            patch("rag_bench.eval.ragtruth.loader.RAGTRUTH_CACHE_DIR", tmp_path),
            patch("rag_bench.eval.ragtruth.loader._try_load_from_github", return_value=(source_info, responses)),
        ):
            loaded = load_ragtruth(task_type="QA")

        assert len(loaded) == 1

    def test_force_download(self, tmp_path):
        """force_download=True skips cache."""
        cache_file = tmp_path / "cache.json"
        # Write stale cache
        cache_file.write_text("[]")

        source_info = [{"source_id": "s1", "source_info": "new ctx", "task_type": "QA"}]
        responses = [
            {"response_id": "r1", "source_id": "s1", "response": "a", "prompt": "q"},
        ]

        with (
            patch("rag_bench.eval.ragtruth.loader.RAGTRUTH_CACHE_FILE", cache_file),
            patch("rag_bench.eval.ragtruth.loader.RAGTRUTH_CACHE_DIR", tmp_path),
            patch("rag_bench.eval.ragtruth.loader._try_load_from_github", return_value=(source_info, responses)),
        ):
            loaded = load_ragtruth(task_type="QA", force_download=True)

        assert len(loaded) == 1
        assert loaded[0].source_info == "new ctx"

    def test_github_fail_huggingface_fallback(self, tmp_path):
        """When GitHub fails, falls back to HuggingFace."""
        cache_file = tmp_path / "nonexistent.json"

        with (
            patch("rag_bench.eval.ragtruth.loader.RAGTRUTH_CACHE_FILE", cache_file),
            patch("rag_bench.eval.ragtruth.loader.RAGTRUTH_CACHE_DIR", tmp_path),
            patch("rag_bench.eval.ragtruth.loader._try_load_from_github", side_effect=RuntimeError("GitHub down")),
            patch("rag_bench.eval.ragtruth.loader._try_load_from_huggingface", return_value=([], [])),
        ):
            loaded = load_ragtruth(task_type="")

        assert len(loaded) == 0

    def test_all_task_types(self, tmp_path):
        """Empty task_type returns all entries."""
        entries = [
            RAGTruthEntry(id="qa", source_id="s", task_type="QA", source_info="", prompt="", reference_response=""),
            RAGTruthEntry(
                id="sum", source_id="s", task_type="Summarization", source_info="", prompt="", reference_response=""
            ),
        ]
        cache_file = tmp_path / "cache.json"
        with (
            patch("rag_bench.eval.ragtruth.loader.RAGTRUTH_CACHE_FILE", cache_file),
            patch("rag_bench.eval.ragtruth.loader.RAGTRUTH_CACHE_DIR", tmp_path),
        ):
            _cache_entries(entries)
            loaded = load_ragtruth(task_type="")
        assert len(loaded) == 2

    def test_corrupt_cache_triggers_download(self, tmp_path):
        """Corrupt cache file triggers re-download."""
        cache_file = tmp_path / "cache.json"
        cache_file.write_text("not valid json{{{")

        with (
            patch("rag_bench.eval.ragtruth.loader.RAGTRUTH_CACHE_FILE", cache_file),
            patch("rag_bench.eval.ragtruth.loader.RAGTRUTH_CACHE_DIR", tmp_path),
            patch("rag_bench.eval.ragtruth.loader._try_load_from_github", return_value=([], [])),
        ):
            loaded = load_ragtruth(task_type="")

        assert len(loaded) == 0


class TestDownloadJsonl:
    """Test _download_jsonl and network loaders."""

    def test_download_jsonl_success(self):
        """_download_jsonl parses JSONL content."""
        from rag_bench.eval.ragtruth.loader import _download_jsonl

        content = b'{"id": "1", "text": "hello"}\n{"id": "2", "text": "world"}\n'
        mock_resp = MagicMock()
        mock_resp.read.return_value = content
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = _download_jsonl("https://example.com/data.jsonl")

        assert len(result) == 2
        assert result[0]["id"] == "1"
        assert result[1]["text"] == "world"

    def test_download_jsonl_failure(self):
        """_download_jsonl raises on network error."""
        from rag_bench.eval.ragtruth.loader import _download_jsonl

        with patch("urllib.request.urlopen", side_effect=OSError("Connection refused")), pytest.raises(OSError):
            _download_jsonl("https://example.com/data.jsonl")

    def test_try_load_from_github(self):
        """_try_load_from_github downloads source_info and response."""
        from rag_bench.eval.ragtruth.loader import _try_load_from_github

        with patch("rag_bench.eval.ragtruth.loader._download_jsonl") as mock_dl:
            mock_dl.side_effect = [
                [{"source_id": "s1", "source_info": "ctx"}],
                [{"response_id": "r1", "source_id": "s1", "response": "ans"}],
            ]
            sources, responses = _try_load_from_github()

        assert len(sources) == 1
        assert len(responses) == 1
        assert mock_dl.call_count == 2

    def test_try_load_from_huggingface_success(self):
        """_try_load_from_huggingface loads from HF datasets."""
        import rag_bench.eval.ragtruth.loader as loader_mod

        with (
            patch.dict("sys.modules", {"datasets": MagicMock()}),
            patch.object(loader_mod, "_try_load_from_huggingface") as mock_fn,
        ):
            mock_fn.return_value = ([], [{"id": "1"}, {"id": "2"}])
            sources, responses = mock_fn()
        assert len(responses) == 2

    def test_try_load_from_huggingface_import_error(self):
        """_try_load_from_huggingface raises when datasets not installed."""
        from rag_bench.eval.ragtruth.loader import _try_load_from_huggingface

        with patch("rag_bench.eval.ragtruth.loader._HAS_DATASETS", False), pytest.raises(ImportError):
            _try_load_from_huggingface()


class TestParseHallucinationSpansEdge:
    """Edge cases for hallucination span parsing."""

    def test_dict_with_hallucination_type_key(self):
        """Test dict span with 'hallucination_type' key (3rd fallback)."""
        raw = [{"text": "bad", "hallucination_type": "Subtle Conflict"}]
        spans = _parse_hallucination_spans(raw)
        assert spans[0].label_type == "Subtle Conflict"

    def test_dict_with_context_key(self):
        """Source text from dict with 'context' key in nested value."""
        data = {"source_info": {"context": "ctx value"}}
        result = _extract_source_text(data)
        assert result == "ctx value"
