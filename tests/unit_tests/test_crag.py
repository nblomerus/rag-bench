"""
Unit tests for rag_bench.core.crag module.

Tests cover:
- Confidence scoring (CORRECT / AMBIGUOUS / INCORRECT)
- Score concentration check (flat scores → downgrade)
- Knowledge refinement (filtering low-score results)
- Result merging (deduplication, interleave by score)
- HyDE generation (mocked Ollama)
- Full CRAG loop with mock retriever, including INCORRECT path
"""

from unittest.mock import MagicMock, patch

import requests

from rag_bench.core.crag import (
    ConfidenceLevel,
    CRAGConfig,
    CRAGRetriever,
    CRAGStats,
)
from rag_bench.core.types import ChunkData, RetrievalResult

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_result(chunk_id: str, rerank_score: float, doc_id: str = "d1") -> RetrievalResult:
    """Create a RetrievalResult with a given rerank score."""
    return RetrievalResult(
        chunk=ChunkData(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=f"Text for {chunk_id}",
            section="body",
        ),
        relevance_score=rerank_score,
        rerank_score=rerank_score,
        sources=["dense"],
    )


def _make_results(scores: list[float]) -> list[RetrievalResult]:
    """Create a list of results with given scores."""
    return [_make_result(f"c{i}", s) for i, s in enumerate(scores)]


class MockRetriever:
    """Fake retriever that returns pre-configured results."""

    def __init__(self, results: list[RetrievalResult]):
        self.results = results
        self.call_count = 0
        self.queries: list[str] = []

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        self.call_count += 1
        self.queries.append(query)
        return self.results[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# Confidence scoring
# ══════════════════════════════════════════════════════════════════════════════


class TestConfidenceScoring:
    def test_high_score_is_correct(self):
        """Top score >= 0.90 with clear gap → CORRECT."""
        config = CRAGConfig()
        crag = CRAGRetriever(base_retriever=MockRetriever([]), config=config)
        results = _make_results([0.98, 0.85, 0.70])
        assert crag._score_confidence(results) == ConfidenceLevel.CORRECT

    def test_mid_score_is_ambiguous(self):
        """Top score in [0.70, 0.90) → AMBIGUOUS."""
        config = CRAGConfig()
        crag = CRAGRetriever(base_retriever=MockRetriever([]), config=config)
        results = _make_results([0.82, 0.60, 0.50])
        assert crag._score_confidence(results) == ConfidenceLevel.AMBIGUOUS

    def test_low_score_is_incorrect(self):
        """Top score < 0.70 → INCORRECT."""
        config = CRAGConfig()
        crag = CRAGRetriever(base_retriever=MockRetriever([]), config=config)
        results = _make_results([0.55, 0.40, 0.30])
        assert crag._score_confidence(results) == ConfidenceLevel.INCORRECT

    def test_flat_scores_downgrade_to_ambiguous(self):
        """High but flat scores (no discrimination) → AMBIGUOUS."""
        config = CRAGConfig()
        crag = CRAGRetriever(base_retriever=MockRetriever([]), config=config)
        # All scores bunched at 0.91-0.92, gap < 0.005, top < 0.95
        results = _make_results([0.920, 0.918, 0.916, 0.914])
        assert crag._score_confidence(results) == ConfidenceLevel.AMBIGUOUS

    def test_very_high_flat_scores_stay_correct(self):
        """Very high flat scores (>0.95) → still CORRECT."""
        config = CRAGConfig()
        crag = CRAGRetriever(base_retriever=MockRetriever([]), config=config)
        results = _make_results([0.99, 0.988, 0.986])
        assert crag._score_confidence(results) == ConfidenceLevel.CORRECT

    def test_single_result(self):
        """Single result with high score → CORRECT."""
        config = CRAGConfig()
        crag = CRAGRetriever(base_retriever=MockRetriever([]), config=config)
        results = _make_results([0.95])
        assert crag._score_confidence(results) == ConfidenceLevel.CORRECT

    def test_custom_thresholds(self):
        """Custom thresholds should be respected."""
        config = CRAGConfig(correct_threshold=0.95, ambiguous_threshold=0.80)
        crag = CRAGRetriever(base_retriever=MockRetriever([]), config=config)
        # 0.92 is below the custom correct_threshold of 0.95
        results = _make_results([0.92, 0.80, 0.70])
        assert crag._score_confidence(results) == ConfidenceLevel.AMBIGUOUS


# ══════════════════════════════════════════════════════════════════════════════
# Knowledge refinement
# ══════════════════════════════════════════════════════════════════════════════


class TestKnowledgeRefinement:
    def test_filters_below_floor(self):
        """Results below refinement_floor should be removed."""
        config = CRAGConfig(refinement_floor=0.50)
        crag = CRAGRetriever(base_retriever=MockRetriever([]), config=config)
        results = _make_results([0.90, 0.60, 0.40, 0.20])
        refined = crag._refine(results, top_k=10)
        assert len(refined) == 2
        assert all(crag._get_score(r) >= 0.50 for r in refined)

    def test_keeps_at_least_one(self):
        """Even if all below floor, keep the best result."""
        config = CRAGConfig(refinement_floor=0.90)
        crag = CRAGRetriever(base_retriever=MockRetriever([]), config=config)
        results = _make_results([0.50, 0.40, 0.30])
        refined = crag._refine(results, top_k=10)
        assert len(refined) == 1
        assert refined[0].chunk.chunk_id == "c0"

    def test_respects_top_k(self):
        """Refinement should cap at top_k."""
        config = CRAGConfig(refinement_floor=0.10)
        crag = CRAGRetriever(base_retriever=MockRetriever([]), config=config)
        results = _make_results([0.9, 0.8, 0.7, 0.6, 0.5])
        refined = crag._refine(results, top_k=3)
        assert len(refined) == 3

    def test_empty_results(self):
        """Empty input → empty output."""
        config = CRAGConfig()
        crag = CRAGRetriever(base_retriever=MockRetriever([]), config=config)
        assert crag._refine([], top_k=10) == []


# ══════════════════════════════════════════════════════════════════════════════
# Result merging
# ══════════════════════════════════════════════════════════════════════════════


class TestResultMerging:
    def test_deduplicates_by_chunk_id(self):
        """Same chunk_id from both lists should appear only once."""
        config = CRAGConfig()
        crag = CRAGRetriever(base_retriever=MockRetriever([]), config=config)

        original = [_make_result("c1", 0.90), _make_result("c2", 0.80)]
        rewritten = [_make_result("c1", 0.85), _make_result("c3", 0.75)]

        merged = crag._merge_results(original, rewritten, top_k=10)
        ids = [r.chunk.chunk_id for r in merged]
        assert len(ids) == len(set(ids))  # no duplicates
        assert set(ids) == {"c1", "c2", "c3"}

    def test_interleave_by_score(self):
        """Merged results should be sorted by score (best first)."""
        config = CRAGConfig()
        crag = CRAGRetriever(base_retriever=MockRetriever([]), config=config)

        original = [_make_result("c1", 0.90), _make_result("c2", 0.70)]
        rewritten = [_make_result("c3", 0.85), _make_result("c4", 0.60)]

        merged = crag._merge_results(original, rewritten, top_k=10)
        scores = [crag._get_score(r) for r in merged]
        assert scores == sorted(scores, reverse=True)

    def test_respects_top_k(self):
        """Merge should cap at top_k."""
        config = CRAGConfig()
        crag = CRAGRetriever(base_retriever=MockRetriever([]), config=config)

        original = _make_results([0.9, 0.8, 0.7])
        rewritten = _make_results([0.85, 0.75, 0.65])
        # Different chunk_ids needed
        for i, r in enumerate(rewritten):
            object.__setattr__(r.chunk, "chunk_id", f"r{i}")

        merged = crag._merge_results(original, rewritten, top_k=4)
        assert len(merged) == 4

    def test_duplicate_keeps_higher_score(self):
        """When same chunk appears in both, the higher-scored copy wins."""
        config = CRAGConfig()
        crag = CRAGRetriever(base_retriever=MockRetriever([]), config=config)

        original = [_make_result("c1", 0.70)]
        rewritten = [_make_result("c1", 0.90)]

        merged = crag._merge_results(original, rewritten, top_k=10)
        assert len(merged) == 1
        assert crag._get_score(merged[0]) == 0.90


# ══════════════════════════════════════════════════════════════════════════════
# Full CRAG loop (with mock retriever, no Ollama)
# ══════════════════════════════════════════════════════════════════════════════


class TestCRAGLoop:
    def test_correct_confidence_skips_rewrite(self):
        """CORRECT confidence → no rewrite, base results returned."""
        results = _make_results([0.98, 0.90, 0.85])
        mock = MockRetriever(results)
        config = CRAGConfig(hyde_enabled=False)  # disable HyDE to test routing
        crag = CRAGRetriever(base_retriever=mock, config=config)

        output = crag.retrieve("test query", top_k=3)
        assert mock.call_count == 1  # only base retrieval
        assert len(output) == 3
        assert crag.stats.correct_count == 1

    def test_ambiguous_triggers_rewrite_when_hyde_disabled(self):
        """AMBIGUOUS with HyDE disabled → no second retrieval."""
        results = _make_results([0.82, 0.60, 0.50])
        mock = MockRetriever(results)
        config = CRAGConfig(hyde_enabled=False)
        crag = CRAGRetriever(base_retriever=mock, config=config)

        crag.retrieve("test query", top_k=3)
        assert mock.call_count == 1  # no rewrite without HyDE
        assert crag.stats.ambiguous_count == 1

    def test_empty_results_handled(self):
        """Empty base results → empty output."""
        mock = MockRetriever([])
        crag = CRAGRetriever(base_retriever=mock)
        output = crag.retrieve("test", top_k=5)
        assert output == []

    def test_refinement_applied_after_retrieval(self):
        """Low-score results should be filtered out."""
        results = _make_results([0.98, 0.95, 0.10, 0.05])
        mock = MockRetriever(results)
        config = CRAGConfig(refinement_floor=0.30)
        crag = CRAGRetriever(base_retriever=mock, config=config)

        output = crag.retrieve("test query", top_k=10)
        assert len(output) == 2  # only the 0.98 and 0.95 survive
        assert crag.stats.results_filtered == 2


# ══════════════════════════════════════════════════════════════════════════════
# Stats tracking
# ══════════════════════════════════════════════════════════════════════════════


class TestCRAGStats:
    def test_stats_summary(self):
        stats = CRAGStats(
            total_queries=10,
            correct_count=6,
            ambiguous_count=3,
            incorrect_count=1,
        )
        summary = stats.summary()
        assert summary["correct_pct"] == 0.6
        assert summary["ambiguous_pct"] == 0.3
        assert summary["incorrect_pct"] == 0.1

    def test_stats_empty(self):
        stats = CRAGStats()
        summary = stats.summary()
        assert summary["total_queries"] == 0
        assert summary["correct_pct"] == 0.0  # no div by zero


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════


class TestCRAGConfig:
    def test_default_config(self):
        config = CRAGConfig()
        assert config.correct_threshold == 0.90
        assert config.ambiguous_threshold == 0.70
        assert config.hyde_enabled is True
        assert config.max_rewrites == 1

    def test_config_from_configs_module(self):
        """CRAGConfig in configs.py should match defaults in crag.py."""
        from rag_bench.core.configs import CRAGConfig as ConfigsCRAGConfig

        c = ConfigsCRAGConfig()
        assert c.correct_threshold == 0.90
        assert c.hyde_enabled is True

    def test_pipeline_config_includes_crag(self):
        """PipelineConfig should have a crag field."""
        from rag_bench.core.configs import PipelineConfig

        pc = PipelineConfig()
        assert hasattr(pc, "crag")
        assert pc.crag.enabled is False

    def test_pipeline_config_roundtrip(self):
        """CRAGConfig should survive PipelineConfig serialization."""
        from rag_bench.core.configs import PipelineConfig

        pc = PipelineConfig()
        pc.crag.enabled = True
        pc.crag.correct_threshold = 0.85

        d = pc.to_dict()
        pc2 = PipelineConfig.from_dict(d)
        assert pc2.crag.enabled is True
        assert pc2.crag.correct_threshold == 0.85


# ══════════════════════════════════════════════════════════════════════════════
# Protocol conformance
# ══════════════════════════════════════════════════════════════════════════════


class TestProtocol:
    def test_conforms_to_retriever_protocol(self):
        """CRAGRetriever should satisfy the Retriever protocol."""
        from rag_bench.core.protocols import Retriever

        mock = MockRetriever([])
        crag = CRAGRetriever(base_retriever=mock)
        assert isinstance(crag, Retriever)


# ══════════════════════════════════════════════════════════════════════════════
# HyDE generation (mocked Ollama)
# ══════════════════════════════════════════════════════════════════════════════


class TestHyDEGeneration:
    def _make_crag(self, **kwargs):
        config = CRAGConfig(**kwargs)
        return CRAGRetriever(base_retriever=MockRetriever([]), config=config)

    def test_generate_hyde_returns_text(self):
        """Successful Ollama response → returns the generated doc."""
        crag = self._make_crag()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "GPT-4 uses RLHF and was trained on large datasets."}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp):
            result = crag._generate_hyde("What training method does GPT-4 use?")

        assert result is not None
        assert "GPT-4" in result

    def test_generate_hyde_too_short_returns_none(self):
        """Short Ollama response (< 20 chars) → None."""
        crag = self._make_crag()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "Short."}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp):
            result = crag._generate_hyde("What is attention?")

        assert result is None

    def test_generate_hyde_request_exception_returns_none(self):
        """Network failure → None (graceful degradation)."""
        crag = self._make_crag()
        with patch("requests.post", side_effect=requests.RequestException("timeout")):
            result = crag._generate_hyde("What is attention?")

        assert result is None

    def test_generate_hyde_empty_response_returns_none(self):
        """Empty string response → None."""
        crag = self._make_crag()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": ""}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp):
            result = crag._generate_hyde("What is BERT?")

        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# INCORRECT path — rewrite and retrieve
# ══════════════════════════════════════════════════════════════════════════════


class TestIncorrectPath:
    def test_incorrect_triggers_rewrite(self):
        """INCORRECT confidence → HyDE rewrite → second retrieval call."""
        low_results = _make_results([0.40, 0.30, 0.20])
        high_results = _make_results([0.95, 0.90, 0.85])
        # First call returns low scores; second (on HyDE doc) returns high
        call_count = [0]

        class SequentialRetriever:
            def retrieve(self, query, top_k=10):
                call_count[0] += 1
                if call_count[0] == 1:
                    return low_results[:top_k]
                return high_results[:top_k]

        config = CRAGConfig(hyde_enabled=True)
        crag = CRAGRetriever(base_retriever=SequentialRetriever(), config=config)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "A factual passage about the query topic providing details."}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp):
            crag.retrieve("Unknown query with low confidence", top_k=5)

        assert crag.stats.incorrect_count == 1
        assert crag.stats.rewrites_attempted == 1
        assert call_count[0] == 2  # base + HyDE

    def test_incorrect_hyde_disabled_no_rewrite(self):
        """INCORRECT with hyde_enabled=False → no rewrite."""
        low_results = _make_results([0.40, 0.30])
        mock = MockRetriever(low_results)
        config = CRAGConfig(hyde_enabled=False)
        crag = CRAGRetriever(base_retriever=mock, config=config)

        crag.retrieve("Unknown query", top_k=5)
        assert mock.call_count == 1  # only base retrieval
        assert crag.stats.incorrect_count == 1

    def test_incorrect_hyde_failure_falls_back(self):
        """INCORRECT, HyDE fails → original results returned."""
        low_results = _make_results([0.40, 0.30])
        mock = MockRetriever(low_results)
        config = CRAGConfig(hyde_enabled=True)
        crag = CRAGRetriever(base_retriever=mock, config=config)

        with patch("requests.post", side_effect=requests.RequestException("fail")):
            output = crag.retrieve("Unknown query", top_k=5)

        # Should still have some results (original low-scoring)
        assert len(output) >= 1
        assert mock.call_count == 1

    def test_rewrite_and_retrieve_merges_dedup(self):
        """Rewrite path deduplicates overlapping results."""
        shared_result = _make_result("shared_chunk", 0.50)
        orig_only = _make_result("orig_only", 0.45)
        hyde_only = _make_result("hyde_only", 0.85)

        class OverlapRetriever:
            def __init__(self):
                self.call_count = 0

            def retrieve(self, query, top_k=10):
                self.call_count += 1
                if self.call_count == 1:
                    return [shared_result, orig_only]
                return [shared_result, hyde_only]

        config = CRAGConfig(hyde_enabled=True, refinement_floor=0.0)
        ret = OverlapRetriever()
        crag = CRAGRetriever(base_retriever=ret, config=config)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "A detailed factual answer about the topic in question."}
        mock_resp.raise_for_status = MagicMock()

        # Force INCORRECT by patching confidence scoring
        with (
            patch.object(crag, "_score_confidence", return_value=ConfidenceLevel.INCORRECT),
            patch("requests.post", return_value=mock_resp),
        ):
            output = crag.retrieve("test query", top_k=10)

        ids = [r.chunk.chunk_id for r in output]
        # "shared_chunk" should appear only once
        assert ids.count("shared_chunk") == 1


# ══════════════════════════════════════════════════════════════════════════════
# get_score helper
# ══════════════════════════════════════════════════════════════════════════════


class TestGetScore:
    def test_prefers_rerank_score(self):
        result = RetrievalResult(
            chunk=ChunkData(chunk_id="c1", doc_id="d1", text="t", section="s"),
            relevance_score=0.5,
            rerank_score=0.9,
        )
        assert CRAGRetriever._get_score(result) == 0.9

    def test_falls_back_to_relevance_score(self):
        result = RetrievalResult(
            chunk=ChunkData(chunk_id="c1", doc_id="d1", text="t", section="s"),
            relevance_score=0.7,
            rerank_score=None,
        )
        assert CRAGRetriever._get_score(result) == 0.7

    def test_top_score_empty_returns_zero(self):
        config = CRAGConfig()
        crag = CRAGRetriever(base_retriever=MockRetriever([]), config=config)
        assert crag._top_score([]) == 0.0

    def test_two_results_top_score(self):
        config = CRAGConfig()
        crag = CRAGRetriever(base_retriever=MockRetriever([]), config=config)
        results = _make_results([0.80, 0.60])
        assert crag._top_score(results) == 0.80
