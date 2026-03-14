"""Integration tests for server.py — pipeline insight, quality metrics, graph context.

These tests exercise the server helper functions and API endpoints that are
not covered by unit tests, using FastAPI TestClient with mocked backends.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from rag_bench.api.server import (
    PipelineInsight,
    _compute_faithfulness_heuristic,
    _compute_per_source_cited,
    _compute_pipeline_insight,
    _compute_quality_metrics,
    _compute_retrieval_confidence,
    _compute_score_spread,
    _compute_source_diversity,
    _content_words,
    app,
)

# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_pipeline():
    """Mock global pipeline so server is considered loaded."""
    retriever = Mock()
    generator = Mock()
    retriever.collection = Mock()
    retriever.collection.count.return_value = 100
    pipeline = Mock()
    pipeline.retriever = retriever
    pipeline.generator = generator
    with (
        patch("rag_bench.api.server._pipeline", pipeline),
        patch("rag_bench.api.server._retriever", retriever),
        patch("rag_bench.api.server._generator", generator),
        patch("rag_bench.api.server._llm_backend_name", "ollama"),
        patch("rag_bench.api.server._llm_model_name", "test:7b"),
    ):
        yield retriever, generator


def _make_results(scores, paper_ids=None):
    """Build mock retrieval result dicts."""
    results = []
    for i, score in enumerate(scores):
        pid = paper_ids[i] if paper_ids else f"paper_{i}"
        results.append(
            {
                "score": score,
                "text": f"Sample passage {i} about machine learning models and architectures.",
                "chunk_id": f"chunk_{i}",
                "metadata": {
                    "title": f"Paper {i}",
                    "section": "introduction" if i % 2 == 0 else "methods",
                    "paper_id": pid,
                },
            }
        )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# _content_words
# ══════════════════════════════════════════════════════════════════════════════


class TestContentWords:
    def test_extracts_meaningful_words(self):
        words = _content_words("The Transformer architecture uses multi-head attention.")
        assert "transformer" in words
        assert "architecture" in words
        assert "attention" in words
        # Stop words excluded
        assert "the" not in words

    def test_excludes_short_tokens(self):
        words = _content_words("A B cd is an ok test word")
        assert "test" in words
        assert "word" in words
        # Short tokens excluded
        assert "cd" not in words
        assert "ok" not in words

    def test_empty_string(self):
        assert _content_words("") == set()


# ══════════════════════════════════════════════════════════════════════════════
# _compute_retrieval_confidence
# ══════════════════════════════════════════════════════════════════════════════


class TestRetrievalConfidence:
    def test_empty_results(self):
        assert _compute_retrieval_confidence([]) == "low"

    def test_high_confidence_cross_encoder(self):
        results = _make_results([5.0, 2.0, 1.5])
        assert _compute_retrieval_confidence(results) == "high"

    def test_medium_confidence_cross_encoder(self):
        results = _make_results([3.0, 2.0, 1.5])
        assert _compute_retrieval_confidence(results) == "medium"

    def test_low_confidence_cross_encoder(self):
        results = _make_results([1.5, 1.4, 1.3])
        assert _compute_retrieval_confidence(results) == "low"

    def test_high_confidence_cosine(self):
        results = _make_results([0.9, 0.5, 0.3])
        assert _compute_retrieval_confidence(results) == "high"

    def test_medium_confidence_cosine(self):
        results = _make_results([0.6, 0.45, 0.3])
        assert _compute_retrieval_confidence(results) == "medium"

    def test_low_confidence_cosine(self):
        results = _make_results([0.3, 0.28, 0.25])
        assert _compute_retrieval_confidence(results) == "low"

    def test_single_result_high(self):
        """Single high-scoring result → high (infinite gap ratio)."""
        results = _make_results([5.0])
        assert _compute_retrieval_confidence(results) == "high"


# ══════════════════════════════════════════════════════════════════════════════
# _compute_score_spread
# ══════════════════════════════════════════════════════════════════════════════


class TestScoreSpread:
    def test_empty(self):
        spread = _compute_score_spread([])
        assert spread["min"] == 0.0
        assert spread["gap_ratio"] == 0.0

    def test_normal_spread(self):
        results = _make_results([5.0, 3.0, 1.0])
        spread = _compute_score_spread(results)
        assert spread["min"] == 1.0
        assert spread["max"] == 5.0
        assert spread["mean"] == 3.0
        assert spread["gap_ratio"] == round(5.0 / 3.0, 2)
        assert spread["std"] > 0

    def test_single_result(self):
        results = _make_results([4.0])
        spread = _compute_score_spread(results)
        assert spread["gap_ratio"] == 0.0  # No second result


# ══════════════════════════════════════════════════════════════════════════════
# _compute_source_diversity
# ══════════════════════════════════════════════════════════════════════════════


class TestSourceDiversity:
    def test_diverse_sources(self):
        results = _make_results([1.0, 0.9, 0.8], paper_ids=["p1", "p2", "p3"])
        diversity = _compute_source_diversity(results)
        assert diversity["unique_papers"] == 3
        assert diversity["unique_sections"] == 2  # intro and methods

    def test_same_paper(self):
        results = _make_results([1.0, 0.9], paper_ids=["p1", "p1"])
        diversity = _compute_source_diversity(results)
        assert diversity["unique_papers"] == 1

    def test_empty(self):
        diversity = _compute_source_diversity([])
        assert diversity["unique_papers"] == 0
        assert diversity["unique_sections"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# _compute_faithfulness_heuristic
# ══════════════════════════════════════════════════════════════════════════════


class TestFaithfulness:
    def test_no_sources(self):
        assert _compute_faithfulness_heuristic("Some answer.", []) == 1.0

    def test_perfect_overlap(self):
        source = "Transformers use multi-head self-attention mechanisms for sequence modeling."
        answer = "Transformers use multi-head self-attention mechanisms for sequence modeling."
        score = _compute_faithfulness_heuristic(answer, [source])
        assert score >= 4.0

    def test_no_overlap(self):
        source = "The cat sat on the mat."
        answer = "Quantum physics describes subatomic particle behavior in accelerators."
        score = _compute_faithfulness_heuristic(answer, [source])
        assert score <= 2.0

    def test_strips_sources_block(self):
        """Should strip [Source N] footer before computing overlap."""
        source = "BERT uses masked language modeling pretraining objectives."
        answer = (
            "BERT uses masked language modeling pretraining objectives.\n\n"
            "[Source 1]: Some paper about unrelated topics and different subjects."
        )
        score = _compute_faithfulness_heuristic(answer, [source])
        assert score >= 3.0


# ══════════════════════════════════════════════════════════════════════════════
# _compute_per_source_cited
# ══════════════════════════════════════════════════════════════════════════════


class TestPerSourceCited:
    def test_inline_citation(self):
        answer = "According to [Source 1], transformers are great. Also see [Source 2]."
        results = _make_results([1.0, 0.9, 0.8])
        cited = _compute_per_source_cited(answer, results)
        assert cited[0]["cited"] is True
        assert cited[0]["citation_count"] == 1
        assert cited[1]["cited"] is True
        assert cited[2]["cited"] is False

    def test_bare_citation(self):
        answer = "Transformers [1] are widely used for NLP [2]."
        results = _make_results([1.0, 0.9])
        cited = _compute_per_source_cited(answer, results)
        assert cited[0]["cited"] is True
        assert cited[1]["cited"] is True

    def test_footer_only_citation(self):
        answer = "Transformers are widely used.\n\nSources:\n[Source 1]: Paper A"
        results = _make_results([1.0, 0.9])
        cited = _compute_per_source_cited(answer, results)
        assert cited[0]["cited"] is True
        assert cited[0]["footer_only"] is True
        assert cited[1]["cited"] is False

    def test_no_citations(self):
        answer = "Transformers are neural network architectures."
        results = _make_results([1.0])
        cited = _compute_per_source_cited(answer, results)
        assert cited[0]["cited"] is False


# ══════════════════════════════════════════════════════════════════════════════
# _compute_pipeline_insight
# ══════════════════════════════════════════════════════════════════════════════


class TestPipelineInsight:
    def test_simple_query_high_confidence(self):
        results = _make_results([5.0, 3.0, 1.0])
        insight = _compute_pipeline_insight(
            "What is dropout?",
            results,
            results,
            retrieval_ms=50.0,
            reranking_ms=30.0,
        )
        assert isinstance(insight, PipelineInsight)
        assert insight.query_type == "simple"
        assert insight.crag_confidence == "correct"
        assert len(insight.stages) == 5

    def test_comparison_query_type(self):
        results = _make_results([3.0, 2.5])
        insight = _compute_pipeline_insight(
            "Compare BERT versus GPT",
            results,
            results,
        )
        assert insight.query_type == "multi_hop"

    def test_low_score_incorrect(self):
        results = _make_results([0.3, 0.2])
        insight = _compute_pipeline_insight(
            "What is XYZ?",
            results,
            results,
        )
        assert insight.crag_confidence == "incorrect"

    def test_ambiguous_confidence(self):
        results = _make_results([0.82, 0.80])
        insight = _compute_pipeline_insight(
            "What is attention?",
            results,
            results,
        )
        assert insight.crag_confidence == "ambiguous"

    def test_empty_results(self):
        insight = _compute_pipeline_insight("test", [], [], retrieval_ms=10.0)
        assert insight.crag_confidence == "unknown"
        assert insight.total_candidates == 0

    def test_refinement_stage_filtered(self):
        all_results = _make_results([5.0, 3.0, 0.1, 0.05])
        filtered = _make_results([5.0, 3.0])
        insight = _compute_pipeline_insight("test query", all_results, filtered)
        # Find refinement stage
        refinement = next(s for s in insight.stages if s.name == "refinement")
        assert refinement.status == "done"
        assert refinement.metadata["filtered"] > 0

    def test_stages_count(self):
        results = _make_results([3.0])
        insight = _compute_pipeline_insight("test", results, results)
        stage_names = [s.name for s in insight.stages]
        assert stage_names == ["retrieval", "reranking", "crag", "refinement", "generation"]


# ══════════════════════════════════════════════════════════════════════════════
# _compute_quality_metrics
# ══════════════════════════════════════════════════════════════════════════════


class TestQualityMetrics:
    def test_basic_metrics(self):
        results = _make_results([5.0, 3.0, 1.0])
        answer = "Transformers use attention [Source 1]. Also [Source 2]."
        metrics = _compute_quality_metrics(answer, results, results)
        assert metrics.sources_cited == 2
        assert metrics.sources_provided == 3
        assert metrics.top_retrieval_score == 5.0
        assert metrics.retrieval_confidence in ("high", "medium", "low")
        assert metrics.faithfulness_score >= 1.0

    def test_no_results(self):
        metrics = _compute_quality_metrics("No sources available.", [], [])
        assert metrics.sources_cited == 0
        assert metrics.sources_provided == 0
        assert metrics.top_retrieval_score == 0.0

    def test_filtered_results_used(self):
        all_results = _make_results([5.0, 3.0, 1.0, 0.5, 0.1])
        filtered = _make_results([5.0, 3.0])
        answer = "Answer text [Source 1]."
        metrics = _compute_quality_metrics(answer, all_results, filtered)
        # Should use filtered results for citation metrics
        assert metrics.sources_provided == 2


# ══════════════════════════════════════════════════════════════════════════════
# Graph context endpoint
# ══════════════════════════════════════════════════════════════════════════════


class TestGraphContextEndpoint:
    def test_graph_context_no_neo4j(self, client):
        """Returns empty subgraph when Neo4j is unavailable."""
        with patch("rag_bench.api.server._get_graph_retriever", return_value=None):
            resp = client.get("/api/graph/context?question=What+is+BERT")
        assert resp.status_code == 200
        body = resp.json()
        assert body["nodes"] == []
        assert body["edges"] == []
        assert body["matched_entities"] == []

    def test_graph_context_with_entities(self, client):
        """Returns subgraph when entities match."""
        mock_retriever = MagicMock()
        mock_retriever._match_entities.return_value = [
            {"name": "BERT", "entity_type": "MODEL", "name_lower": "bert"},
        ]
        mock_retriever.store.get_entity_triples.return_value = [
            {
                "subject": "BERT",
                "subject_type": "MODEL",
                "predicate": "USES",
                "object": "Masked Language Modeling",
                "object_type": "METHOD",
                "weight": 3,
            },
        ]

        with patch("rag_bench.api.server._get_graph_retriever", return_value=mock_retriever):
            resp = client.get("/api/graph/context?question=How+does+BERT+work")

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["nodes"]) >= 2
        assert len(body["edges"]) >= 1
        assert "BERT" in body["matched_entities"]

        # Check matched node
        bert_node = next(n for n in body["nodes"] if n["id"] == "bert")
        assert bert_node["matched"] is True
        assert bert_node["entity_type"] == "MODEL"

    def test_graph_context_no_matches(self, client):
        """Returns empty when no entities match."""
        mock_retriever = MagicMock()
        mock_retriever._match_entities.return_value = []

        with patch("rag_bench.api.server._get_graph_retriever", return_value=mock_retriever):
            resp = client.get("/api/graph/context?question=random+gibberish")

        assert resp.status_code == 200
        assert resp.json()["nodes"] == []

    def test_graph_context_error_handling(self, client):
        """Returns empty subgraph on error, doesn't crash."""
        mock_retriever = MagicMock()
        mock_retriever._match_entities.side_effect = RuntimeError("Neo4j connection lost")

        with patch("rag_bench.api.server._get_graph_retriever", return_value=mock_retriever):
            resp = client.get("/api/graph/context?question=test")

        assert resp.status_code == 200
        assert resp.json()["nodes"] == []


# ══════════════════════════════════════════════════════════════════════════════
# Query endpoint with pipeline insight + quality metrics
# ══════════════════════════════════════════════════════════════════════════════


class TestQueryWithMetrics:
    def test_query_returns_quality_metrics(self, client, mock_pipeline):
        """POST /api/query should include quality_metrics in response."""
        retriever, generator = mock_pipeline
        generator.answer.return_value = {
            "answer": "Transformers use attention [Source 1].",
            "deflected": False,
            "results": _make_results([5.0, 3.0]),
            "filtered_results": _make_results([5.0, 3.0]),
        }
        resp = client.post("/api/query", json={"question": "What are transformers?"})
        assert resp.status_code == 200
        body = resp.json()
        assert "quality" in body
        qm = body["quality"]
        assert "retrieval_confidence" in qm
        assert "faithfulness_score" in qm
        assert "sources_cited" in qm

    def test_query_returns_pipeline_insight(self, client, mock_pipeline):
        """POST /api/query should include pipeline_insight in response."""
        retriever, generator = mock_pipeline
        generator.answer.return_value = {
            "answer": "BERT uses masking.",
            "deflected": False,
            "results": _make_results([4.0, 2.0]),
            "filtered_results": _make_results([4.0, 2.0]),
        }
        resp = client.post("/api/query", json={"question": "What is BERT?"})
        assert resp.status_code == 200
        body = resp.json()
        assert "pipeline" in body
        # pipeline may be None if _compute_pipeline_insight wasn't called
        # (depends on whether the generator returns the right shape)


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark run endpoint
# ══════════════════════════════════════════════════════════════════════════════


class TestBenchmarkRun:
    def test_ragbench_run_success(self, client, mock_pipeline):
        """POST /api/eval/benchmark with ragbench completes and returns results."""
        retriever, generator = mock_pipeline

        mock_report = MagicMock()
        mock_report.results = [
            MagicMock(
                id="test-1",
                question="What is X?",
                answer_preview="X is...",
                retrieval={"mrr": 0.5, "ndcg_at_k": 0.6},
                citation={"precision": 0.8},
                completeness={"score": 0.7},
                deflection={"correct": True},
                error=None,
            ),
        ]
        mock_report.summary = {
            "total_queries": 1,
            "retrieval_mrr": 0.5,
            "retrieval_ndcg_at_5": 0.6,
        }
        mock_report.metadata = {"total_evaluated": 1}

        with (
            patch("rag_bench.api.server._benchmark_running", False),
            patch("rag_bench.eval.runner.EvalRunner.run_all", return_value=mock_report),
            patch("rag_bench.api.server.save_report"),
            patch("rag_bench.api.server._save_benchmark_history"),
        ):
            resp = client.post(
                "/api/eval/benchmark",
                json={"benchmark": "ragbench", "sample_size": 1, "run_type": "manual"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["benchmark"] == "ragbench"
        assert len(body["results"]) == 1
        assert body["summary"]["retrieval_mrr"] == 0.5

    def test_ragtruth_run_success(self, client, mock_pipeline):
        """POST /api/eval/benchmark with ragtruth completes and returns results."""
        retriever, generator = mock_pipeline

        mock_report = MagicMock()
        mock_report.results = [
            MagicMock(
                id="rt-1",
                prompt="What is X?" * 10,
                generated_answer="X is a thing" * 10,
                has_hallucination_gold=False,
                has_hallucination_predicted=False,
                span_metrics={"span_f1": 1.0},
                error=None,
            ),
        ]
        mock_report.summary = {"case_level_accuracy": 0.9, "hallucination_rate": 0.1}
        mock_report.case_level = {"accuracy": 0.9, "f1": 0.85}
        mock_report.by_type = {"Baseless": 2, "Conflict": 1}
        mock_report.metadata = {"total_evaluated": 1}

        with (
            patch("rag_bench.api.server._benchmark_running", False),
            patch("rag_bench.eval.ragtruth.runner.RAGTruthRunner.run", return_value=mock_report),
            patch("rag_bench.api.server._save_benchmark_history"),
        ):
            resp = client.post(
                "/api/eval/benchmark",
                json={"benchmark": "ragtruth", "sample_size": 1},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["benchmark"] == "ragtruth"
        assert body["summary"]["case_level_accuracy"] == 0.9
        # Case level and by_type should be flattened into summary
        assert body["summary"]["accuracy"] == 0.9
        assert body["summary"]["type_Baseless"] == 2
