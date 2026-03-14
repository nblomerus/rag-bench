"""
Unit tests for rag_bench.core.entity_extractor module.

Tests cover:
- Triple extraction from LLM responses (happy path)
- JSON parsing with fallback (malformed JSON, markdown-wrapped)
- Caching: save, load, invalidation
- Error handling: LLM failures, empty responses, self-referential triples
- Entity/Triple data types
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests as req

from rag_bench.core.configs import ExtractorConfig
from rag_bench.core.entity_extractor import EntityExtractor
from rag_bench.core.graph_types import Entity, ExtractionResult, Triple
from rag_bench.core.types import ChunkData

# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_paper():
    return {
        "doc_id": "arxiv_2301.00001",
        "title": "Scaling Laws for Neural Language Models",
        "authors": ["Kaplan, J.", "McCandlish, S."],
        "year": 2020,
        "sections": {
            "abstract": "We study empirical scaling laws for language model performance.",
            "introduction": "Neural language models have shown remarkable capabilities.",
            "methods": "We train Transformer models of varying sizes on large text corpora.",
        },
    }


@pytest.fixture
def sample_chunks():
    return [
        ChunkData(
            chunk_id="arxiv_2301.00001_methods_000",
            doc_id="arxiv_2301.00001",
            text="We train Transformer models ranging from 768 to 1.5B parameters on "
            "the WebText2 dataset. Performance is measured using cross-entropy loss.",
            section="methods",
            metadata={"title": "Scaling Laws for Neural Language Models"},
        ),
        ChunkData(
            chunk_id="arxiv_2301.00001_results_000",
            doc_id="arxiv_2301.00001",
            text="GPT-3 outperforms GPT-2 on all benchmarks including SuperGLUE and MMLU.",
            section="results",
            metadata={"title": "Scaling Laws for Neural Language Models"},
        ),
    ]


@pytest.fixture
def extractor(tmp_path):
    config = ExtractorConfig(
        enabled=True,
        cache_dir=str(tmp_path / "cache"),
    )
    return EntityExtractor(config=config)


def _make_ollama_response(triples_json: list[dict]) -> MagicMock:
    """Helper to build a mock Ollama response with JSON triples."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": json.dumps(triples_json)}
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


SAMPLE_TRIPLES_JSON = [
    {
        "subject_name": "Transformer",
        "subject_type": "MODEL",
        "predicate": "TRAINED_ON",
        "object_name": "WebText2",
        "object_type": "DATASET",
    },
    {
        "subject_name": "GPT-3",
        "subject_type": "MODEL",
        "predicate": "OUTPERFORMS",
        "object_name": "GPT-2",
        "object_type": "MODEL",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# Extraction
# ══════════════════════════════════════════════════════════════════════════════


class TestExtraction:
    def test_extract_returns_triples(self, extractor, sample_chunks, sample_paper):
        """Successful extraction should return Triple objects."""
        mock_resp = _make_ollama_response(SAMPLE_TRIPLES_JSON)
        with patch("rag_bench.core.entity_extractor.requests.post", return_value=mock_resp):
            results = extractor.extract(sample_chunks, sample_paper)

        assert len(results) == 2
        assert all(isinstance(r, ExtractionResult) for r in results)
        # Each chunk gets the same mock response (2 triples each)
        assert all(len(r.triples) == 2 for r in results)

    def test_extract_triple_fields(self, extractor, sample_chunks, sample_paper):
        """Triples should have correct subject, predicate, object."""
        mock_resp = _make_ollama_response([SAMPLE_TRIPLES_JSON[0]])
        with patch("rag_bench.core.entity_extractor.requests.post", return_value=mock_resp):
            results = extractor.extract(sample_chunks[:1], sample_paper)

        triple = results[0].triples[0]
        assert triple.subject.name == "Transformer"
        assert triple.subject.entity_type == "MODEL"
        assert triple.predicate == "TRAINED_ON"
        assert triple.object.name == "WebText2"
        assert triple.object.entity_type == "DATASET"
        assert triple.source_chunk_id == "arxiv_2301.00001_methods_000"
        assert triple.source_doc_id == "arxiv_2301.00001"

    def test_extract_empty_list(self, extractor, sample_paper):
        """Extracting from empty chunk list returns empty."""
        results = extractor.extract([], sample_paper)
        assert results == []

    def test_extract_handles_llm_failure(self, extractor, sample_chunks, sample_paper):
        """LLM failure should return empty triples, not crash."""
        with patch(
            "rag_bench.core.entity_extractor.requests.post",
            side_effect=req.RequestException("Connection refused"),
        ):
            results = extractor.extract(sample_chunks, sample_paper)

        assert len(results) == 2
        assert all(len(r.triples) == 0 for r in results)
        assert all(r.parse_success is False for r in results)

    def test_extract_handles_empty_response(self, extractor, sample_chunks, sample_paper):
        """Empty JSON array from LLM → no triples."""
        mock_resp = _make_ollama_response([])
        with patch("rag_bench.core.entity_extractor.requests.post", return_value=mock_resp):
            results = extractor.extract(sample_chunks[:1], sample_paper)

        assert len(results[0].triples) == 0
        assert results[0].parse_success is True

    def test_extract_caps_triples_per_chunk(self, tmp_path, sample_chunks, sample_paper):
        """Should respect max_triples_per_chunk config."""
        config = ExtractorConfig(
            enabled=True,
            cache_dir=str(tmp_path / "cache"),
            max_triples_per_chunk=1,
        )
        extractor = EntityExtractor(config=config)
        mock_resp = _make_ollama_response(SAMPLE_TRIPLES_JSON)  # 2 triples

        with patch("rag_bench.core.entity_extractor.requests.post", return_value=mock_resp):
            results = extractor.extract(sample_chunks[:1], sample_paper)

        assert len(results[0].triples) == 1  # capped to 1

    def test_extract_skips_self_referential(self, extractor, sample_chunks, sample_paper):
        """Triples where subject == object should be filtered out."""
        bad_triple = [
            {
                "subject_name": "BERT",
                "subject_type": "MODEL",
                "predicate": "USES",
                "object_name": "BERT",
                "object_type": "MODEL",
            }
        ]
        mock_resp = _make_ollama_response(bad_triple)

        with patch("rag_bench.core.entity_extractor.requests.post", return_value=mock_resp):
            results = extractor.extract(sample_chunks[:1], sample_paper)

        assert len(results[0].triples) == 0


# ══════════════════════════════════════════════════════════════════════════════
# JSON parsing
# ══════════════════════════════════════════════════════════════════════════════


class TestJsonParsing:
    def test_parse_clean_json(self, extractor):
        """Clean JSON array should parse directly."""
        raw = json.dumps(SAMPLE_TRIPLES_JSON)
        triples = extractor._parse_triples(raw, "chunk_1", "doc_1")
        assert len(triples) == 2

    def test_parse_markdown_wrapped_json(self, extractor):
        """JSON wrapped in markdown code block should parse via fallback."""
        raw = "```json\n" + json.dumps(SAMPLE_TRIPLES_JSON) + "\n```"
        triples = extractor._parse_triples(raw, "chunk_1", "doc_1")
        assert len(triples) == 2

    def test_parse_json_with_preamble(self, extractor):
        """JSON with explanatory text before it should parse via fallback."""
        raw = "Here are the extracted triples:\n" + json.dumps(SAMPLE_TRIPLES_JSON)
        triples = extractor._parse_triples(raw, "chunk_1", "doc_1")
        assert len(triples) == 2

    def test_parse_completely_invalid(self, extractor):
        """Totally invalid output should return empty list."""
        triples = extractor._parse_triples("I don't understand the request", "c1", "d1")
        assert triples == []

    def test_parse_skips_incomplete_triples(self, extractor):
        """Triples missing required fields should be skipped."""
        raw = json.dumps(
            [
                {
                    "subject_name": "BERT",
                    "subject_type": "MODEL",
                    "predicate": "USES",
                    "object_name": "attention",
                    "object_type": "METHOD",
                },
                {
                    "subject_name": "",
                    "subject_type": "MODEL",
                    "predicate": "USES",
                    "object_name": "dropout",
                    "object_type": "METHOD",
                },  # empty subject
            ]
        )
        triples = extractor._parse_triples(raw, "c1", "d1")
        assert len(triples) == 1
        assert triples[0].subject.name == "BERT"


# ══════════════════════════════════════════════════════════════════════════════
# Caching
# ══════════════════════════════════════════════════════════════════════════════


class TestCaching:
    def test_cache_saves_and_loads(self, extractor, sample_chunks, sample_paper):
        """Second call should use cached results (no LLM calls)."""
        mock_resp = _make_ollama_response(SAMPLE_TRIPLES_JSON)

        with patch("rag_bench.core.entity_extractor.requests.post", return_value=mock_resp) as mock_post:
            extractor.extract(sample_chunks, sample_paper)
            first_call_count = mock_post.call_count

            # Second call — all cache hits
            extractor.extract(sample_chunks, sample_paper)
            assert mock_post.call_count == first_call_count

    def test_cache_file_created(self, extractor, sample_chunks, sample_paper):
        """Cache JSON file should be created for each paper."""
        mock_resp = _make_ollama_response(SAMPLE_TRIPLES_JSON)

        with patch("rag_bench.core.entity_extractor.requests.post", return_value=mock_resp):
            extractor.extract(sample_chunks, sample_paper)

        cache_path = extractor._cache_path("arxiv_2301.00001")
        assert cache_path.exists()

        cache_data = json.loads(cache_path.read_text())
        assert len(cache_data) == 2  # one entry per chunk

    def test_cache_invalidates_on_text_change(self, extractor, sample_chunks, sample_paper):
        """Changed chunk text should trigger re-extraction."""
        mock_resp = _make_ollama_response(SAMPLE_TRIPLES_JSON)

        with patch("rag_bench.core.entity_extractor.requests.post", return_value=mock_resp) as mock_post:
            extractor.extract(sample_chunks, sample_paper)
            first_call_count = mock_post.call_count

            modified_chunks = [
                ChunkData(
                    chunk_id=sample_chunks[0].chunk_id,
                    doc_id=sample_chunks[0].doc_id,
                    text="Completely different text about BERT and attention",
                    section=sample_chunks[0].section,
                    metadata=sample_chunks[0].metadata,
                ),
                sample_chunks[1],
            ]
            extractor.extract(modified_chunks, sample_paper)
            assert mock_post.call_count == first_call_count + 1


# ══════════════════════════════════════════════════════════════════════════════
# Data types
# ══════════════════════════════════════════════════════════════════════════════


class TestDataTypes:
    def test_entity_equality_case_insensitive(self):
        """Entities with same name (different case) should be equal."""
        e1 = Entity(name="Transformer", entity_type="MODEL")
        e2 = Entity(name="transformer", entity_type="MODEL")
        assert e1 == e2
        assert hash(e1) == hash(e2)

    def test_entity_different_type_not_equal(self):
        """Same name but different type should not be equal."""
        e1 = Entity(name="attention", entity_type="METHOD")
        e2 = Entity(name="attention", entity_type="METRIC")
        assert e1 != e2

    def test_triple_roundtrip(self):
        """Triple should survive to_dict → from_dict roundtrip."""
        triple = Triple(
            subject=Entity(name="GPT-4", entity_type="MODEL"),
            predicate="OUTPERFORMS",
            object=Entity(name="GPT-3.5", entity_type="MODEL"),
            source_chunk_id="chunk_1",
            source_doc_id="doc_1",
            confidence=0.95,
        )
        d = triple.to_dict()
        reconstructed = Triple.from_dict(d)
        assert reconstructed.subject.name == "GPT-4"
        assert reconstructed.predicate == "OUTPERFORMS"
        assert reconstructed.object.name == "GPT-3.5"
        assert reconstructed.confidence == 0.95

    def test_extraction_result_default(self):
        """ExtractionResult should have sensible defaults."""
        r = ExtractionResult(chunk_id="c1", doc_id="d1")
        assert r.triples == []
        assert r.parse_success is True
        assert r.raw_llm_response == ""


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════


class TestExtractorConfig:
    def test_default_config(self):
        config = ExtractorConfig()
        assert config.enabled is False
        assert config.max_triples_per_chunk == 10
        assert "qwen2.5" in config.ollama_model

    def test_cache_dir_created(self, tmp_path):
        cache_dir = tmp_path / "new_cache"
        assert not cache_dir.exists()
        config = ExtractorConfig(enabled=True, cache_dir=str(cache_dir))
        EntityExtractor(config=config)
        assert cache_dir.exists()
