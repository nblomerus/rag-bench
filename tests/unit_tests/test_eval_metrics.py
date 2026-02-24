"""Tests for rag_bench.eval.metrics — pure metric computation functions."""

import pytest

from rag_bench.eval.citation_quality_queries import (
    CITATION_QUALITY_QUERIES,
    get_all_primary_sources,
    get_queries_by_topic,
    get_queries_by_type,
)
from rag_bench.eval.metrics import (
    citation_density,
    citation_precision,
    citation_recall,
    compute_citation_metrics,
    compute_completeness,
    compute_retrieval_metrics,
    count_unsupported_claims,
    detect_hallucinations,
    extract_cited_source_numbers,
    extract_paper_ids,
    hit_rate,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    source_coverage,
)

# ═══════════════════════════════════════════════════════════════════════════
# Retrieval Metrics
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractPaperIds:
    def test_arxiv_underscore_format(self):
        results = [
            {"arxiv_id": "arxiv_1706_03762"},
            {"arxiv_id": "arxiv_1810_04805"},
        ]
        assert extract_paper_ids(results) == ["1706.03762", "1810.04805"]

    def test_already_normalized(self):
        results = [{"arxiv_id": "1706.03762"}]
        assert extract_paper_ids(results) == ["1706.03762"]

    def test_deduplication_preserves_order(self):
        results = [
            {"arxiv_id": "arxiv_1706_03762"},
            {"arxiv_id": "arxiv_1810_04805"},
            {"arxiv_id": "arxiv_1706_03762"},  # duplicate
        ]
        assert extract_paper_ids(results) == ["1706.03762", "1810.04805"]

    def test_empty_results(self):
        assert extract_paper_ids([]) == []

    def test_paper_id_fallback(self):
        results = [{"paper_id": "2005.14165"}]
        assert extract_paper_ids(results) == ["2005.14165"]

    def test_metadata_fallback(self):
        results = [{"metadata": {"arxiv_id": "arxiv_2005_14165"}}]
        assert extract_paper_ids(results) == ["2005.14165"]

    def test_missing_ids(self):
        results = [{"text": "no id here"}]
        assert extract_paper_ids(results) == []


class TestPrecisionAtK:
    def test_perfect_precision(self):
        retrieved = ["A", "B"]
        expected = ["A", "B"]
        assert precision_at_k(retrieved, expected, k=2) == 1.0

    def test_zero_precision(self):
        retrieved = ["C", "D"]
        expected = ["A", "B"]
        assert precision_at_k(retrieved, expected, k=2) == 0.0

    def test_partial_precision(self):
        retrieved = ["A", "B", "C", "D", "E"]
        expected = ["A", "C"]
        assert precision_at_k(retrieved, expected, k=5) == 2 / 5

    def test_k_larger_than_retrieved(self):
        retrieved = ["A", "B"]
        expected = ["A"]
        assert precision_at_k(retrieved, expected, k=5) == 1 / 5

    def test_k_zero(self):
        assert precision_at_k(["A"], ["A"], k=0) == 0.0


class TestRecallAtK:
    def test_perfect_recall(self):
        retrieved = ["A", "B", "C"]
        expected = ["A", "B"]
        assert recall_at_k(retrieved, expected, k=3) == 1.0

    def test_zero_recall(self):
        retrieved = ["C", "D"]
        expected = ["A", "B"]
        assert recall_at_k(retrieved, expected, k=2) == 0.0

    def test_partial_recall(self):
        retrieved = ["A", "C", "D"]
        expected = ["A", "B"]
        assert recall_at_k(retrieved, expected, k=3) == 0.5

    def test_empty_expected(self):
        """No expected sources → perfect recall by definition."""
        assert recall_at_k(["A", "B"], [], k=2) == 1.0


class TestMRR:
    def test_first_result_relevant(self):
        assert mean_reciprocal_rank(["A", "B", "C"], ["A"]) == 1.0

    def test_second_result_relevant(self):
        assert mean_reciprocal_rank(["B", "A", "C"], ["A"]) == 0.5

    def test_third_result_relevant(self):
        assert mean_reciprocal_rank(["B", "C", "A"], ["A"]) == pytest.approx(1 / 3)

    def test_no_relevant_result(self):
        assert mean_reciprocal_rank(["B", "C", "D"], ["A"]) == 0.0

    def test_multiple_expected_first_match(self):
        """MRR uses rank of FIRST relevant result."""
        assert mean_reciprocal_rank(["B", "A", "C"], ["A", "C"]) == 0.5


class TestNDCG:
    def test_perfect_ranking(self):
        """All relevant at top → NDCG = 1.0."""
        retrieved = ["A", "B", "C"]
        expected = ["A", "B"]
        assert ndcg_at_k(retrieved, expected, k=3) == pytest.approx(1.0)

    def test_worst_ranking(self):
        """No relevant in top-K → NDCG = 0.0."""
        retrieved = ["C", "D", "E"]
        expected = ["A", "B"]
        assert ndcg_at_k(retrieved, expected, k=3) == 0.0

    def test_partial_relevance(self):
        """Some relevant mixed in."""
        retrieved = ["C", "A", "D", "B", "E"]
        expected = ["A", "B"]
        result = ndcg_at_k(retrieved, expected, k=5)
        assert 0 < result < 1.0

    def test_empty_expected(self):
        assert ndcg_at_k(["A", "B"], [], k=2) == 1.0


class TestHitRate:
    def test_hit(self):
        assert hit_rate(["A", "B", "C"], ["B"], k=3) == 1.0

    def test_miss(self):
        assert hit_rate(["A", "B", "C"], ["D"], k=3) == 0.0

    def test_hit_at_boundary(self):
        assert hit_rate(["A", "B", "C"], ["C"], k=3) == 1.0

    def test_miss_beyond_k(self):
        assert hit_rate(["A", "B", "C", "D"], ["D"], k=3) == 0.0


class TestComputeRetrievalMetrics:
    def test_full_pipeline(self):
        results = [
            {"arxiv_id": "arxiv_1706_03762", "score": 7.0},
            {"arxiv_id": "arxiv_1810_04805", "score": 6.0},
            {"arxiv_id": "arxiv_2005_14165", "score": 5.0},
        ]
        expected = ["1706.03762"]
        metrics = compute_retrieval_metrics(results, expected, k=3)
        assert metrics["mrr"] == 1.0
        assert metrics["recall_at_k"] == 1.0
        assert metrics["hit_rate"] == 1.0
        assert metrics["k"] == 3
        assert "1706.03762" in metrics["retrieved_papers"]


# ═══════════════════════════════════════════════════════════════════════════
# Citation Metrics
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractCitedSources:
    def test_single_citation(self):
        assert extract_cited_source_numbers("claim [Source 1] here") == [1]

    def test_multiple_citations(self):
        assert extract_cited_source_numbers("claim [Source 1] and [Source 3]") == [1, 3]

    def test_no_citations(self):
        assert extract_cited_source_numbers("no citations here") == []

    def test_duplicate_citations(self):
        assert extract_cited_source_numbers("[Source 1] and again [Source 1]") == [1]

    def test_sorted_output(self):
        assert extract_cited_source_numbers("[Source 3] before [Source 1]") == [1, 3]


class TestCitationPrecision:
    def test_all_cited_are_expected(self):
        answer = "claim [Source 1]"
        results = [{"arxiv_id": "1706.03762"}]
        expected = ["1706.03762"]
        assert citation_precision(answer, results, expected) == 1.0

    def test_none_cited_are_expected(self):
        answer = "claim [Source 1]"
        results = [{"arxiv_id": "9999.99999"}]
        expected = ["1706.03762"]
        assert citation_precision(answer, results, expected) == 0.0

    def test_no_citations(self):
        assert citation_precision("no citations", [], ["1706.03762"]) == 0.0

    def test_mixed(self):
        answer = "claim [Source 1] and [Source 2]"
        results = [{"arxiv_id": "1706.03762"}, {"arxiv_id": "9999.99999"}]
        expected = ["1706.03762"]
        assert citation_precision(answer, results, expected) == 0.5


class TestCitationRecall:
    def test_all_expected_cited(self):
        answer = "claim [Source 1]"
        results = [{"arxiv_id": "1706.03762"}]
        expected = ["1706.03762"]
        assert citation_recall(answer, results, expected) == 1.0

    def test_none_cited(self):
        answer = "no citations"
        results = [{"arxiv_id": "1706.03762"}]
        expected = ["1706.03762"]
        assert citation_recall(answer, results, expected) == 0.0

    def test_partial(self):
        answer = "claim [Source 1]"
        results = [{"arxiv_id": "1706.03762"}, {"arxiv_id": "1810.04805"}]
        expected = ["1706.03762", "1810.04805"]
        assert citation_recall(answer, results, expected) == 0.5

    def test_empty_expected(self):
        assert citation_recall("any answer", [], []) == 1.0


class TestSourceCoverage:
    def test_full_coverage(self):
        answer = "[Source 1] and [Source 2] and [Source 3]"
        assert source_coverage(answer, 3) == 1.0

    def test_partial_coverage(self):
        answer = "[Source 1] and [Source 3]"
        assert source_coverage(answer, 4) == 0.5

    def test_no_sources_provided(self):
        assert source_coverage("any answer", 0) == 0.0


class TestCitationDensity:
    def test_dense_citations(self):
        answer = "First claim [Source 1]. Second claim [Source 2]. Third claim [Source 3]."
        density = citation_density(answer)
        assert density == pytest.approx(1.0)

    def test_sparse_citations(self):
        answer = "First claim [Source 1]. Second claim. Third claim. Fourth claim."
        density = citation_density(answer)
        assert density == pytest.approx(0.25)

    def test_empty_answer(self):
        assert citation_density("") == 0.0

    def test_sources_block_excluded(self):
        answer = "Claim [Source 1].\n\nSources:\n1. Paper A"
        density = citation_density(answer)
        assert density > 0


class TestUnsupportedClaims:
    def test_all_claims_cited(self):
        answer = "The Transformer model uses attention [Source 1]. BERT uses masked language modeling [Source 2]."
        assert count_unsupported_claims(answer) == 0

    def test_uncited_factual_claims(self):
        answer = "The Transformer uses 8 attention heads. BERT has 768 dimensions."
        count = count_unsupported_claims(answer)
        assert count >= 1

    def test_short_sentences_ignored(self):
        answer = "Yes. No. Maybe."
        assert count_unsupported_claims(answer) == 0


class TestHallucinations:
    def test_no_hallucinations(self):
        answer = "BERT uses masked language modeling."
        assert detect_hallucinations(answer, ["500 billion"]) == []

    def test_detected_hallucination(self):
        answer = "BERT has 500 billion parameters."
        found = detect_hallucinations(answer, ["500 billion"])
        assert "500 billion" in found

    def test_case_insensitive(self):
        found = detect_hallucinations("The model has 500 Billion params", ["500 billion"])
        assert len(found) == 1

    def test_empty_excludes(self):
        assert detect_hallucinations("any answer", []) == []


class TestMultipleCitationScenarios:
    """Tests for answers citing multiple sources in various patterns."""

    def test_three_sources_all_cited_coverage(self):
        answer = "Point A [Source 1]. Point B [Source 2]. Point C [Source 3]."
        assert source_coverage(answer, 3) == 1.0

    def test_five_sources_two_cited_coverage(self):
        answer = "Point A [Source 1]. Point B [Source 4]."
        assert source_coverage(answer, 5) == 2 / 5

    def test_density_with_multi_cite_sentence(self):
        """One sentence cites multiple sources."""
        answer = "Transformers [Source 1] extend sequence models [Source 2]."
        density = citation_density(answer)
        assert density == pytest.approx(2.0)

    def test_density_across_many_sentences(self):
        """5 sentences, 3 have citations."""
        answer = (
            "Transformers use attention [Source 1]. "
            "This is efficient. "
            "BERT uses masking [Source 2]. "
            "Training is important. "
            "GPT uses autoregressive decoding [Source 3]."
        )
        density = citation_density(answer)
        assert density == pytest.approx(3 / 5)

    def test_unsupported_with_some_cited(self):
        """Mix of cited and uncited factual sentences."""
        answer = (
            "The Transformer has 65 million parameters [Source 1]. "
            "BERT achieves 0.85 accuracy on the benchmark. "
            "GPT-2 has 1.5 billion parameters [Source 2]."
        )
        # Middle sentence has number + technical term but no citation
        count = count_unsupported_claims(answer)
        assert count >= 1

    def test_all_factual_claims_cited(self):
        """Every factual sentence has a citation."""
        answer = (
            "The Transformer uses 8 attention heads [Source 1]. "
            "BERT has 110 million parameters [Source 2]. "
            "GPT-3 was trained on 300 billion tokens [Source 3]."
        )
        count = count_unsupported_claims(answer)
        assert count == 0

    def test_precision_all_cited_correct(self):
        """3 cited sources, all from expected papers."""
        answer = "A [Source 1]. B [Source 2]. C [Source 3]."
        results = [
            {"arxiv_id": "1706.03762"},
            {"arxiv_id": "1810.04805"},
            {"arxiv_id": "2005.14165"},
        ]
        expected = ["1706.03762", "1810.04805", "2005.14165"]
        assert citation_precision(answer, results, expected) == 1.0

    def test_precision_half_correct(self):
        """4 cited, 2 from expected papers."""
        answer = "A [Source 1]. B [Source 2]. C [Source 3]. D [Source 4]."
        results = [
            {"arxiv_id": "1706.03762"},
            {"arxiv_id": "9999.00001"},
            {"arxiv_id": "1810.04805"},
            {"arxiv_id": "9999.00002"},
        ]
        expected = ["1706.03762", "1810.04805"]
        assert citation_precision(answer, results, expected) == 0.5

    def test_recall_two_of_three_expected(self):
        """3 expected papers, only 2 cited in answer."""
        answer = "A [Source 1]. B [Source 2]."
        results = [
            {"arxiv_id": "1706.03762"},
            {"arxiv_id": "1810.04805"},
            {"arxiv_id": "2005.14165"},
        ]
        expected = ["1706.03762", "1810.04805", "2005.14165"]
        assert citation_recall(answer, results, expected) == pytest.approx(2 / 3)

    def test_full_pipeline_multi_source(self):
        """End-to-end compute_citation_metrics with 4 sources."""
        answer = (
            "Transformers [Source 1] are used in NLP. "
            "BERT [Source 2] introduced masking. "
            "GPT [Source 3] uses autoregressive generation."
        )
        results = [
            {"arxiv_id": "1706.03762"},
            {"arxiv_id": "1810.04805"},
            {"arxiv_id": "2005.14165"},
            {"arxiv_id": "1901.00001"},
        ]
        expected = ["1706.03762", "1810.04805", "2005.14165"]
        metrics = compute_citation_metrics(answer, results, expected)
        assert metrics["precision"] == 1.0  # all 3 cited are expected
        assert metrics["recall"] == 1.0  # all 3 expected are cited
        assert metrics["cited_sources"] == [1, 2, 3]
        assert metrics["source_coverage"] == 3 / 4  # 3 of 4 results cited
        assert metrics["density"] == pytest.approx(1.0)  # 3 citations / 3 sentences


class TestComputeCitationMetrics:
    def test_full_pipeline(self):
        answer = "The Transformer [Source 1] uses attention. BERT [Source 2] uses masking."
        results = [
            {"arxiv_id": "1706.03762"},
            {"arxiv_id": "1810.04805"},
        ]
        expected = ["1706.03762", "1810.04805"]
        metrics = compute_citation_metrics(answer, results, expected)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["cited_sources"] == [1, 2]
        assert len(metrics["cited_paper_ids"]) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Completeness
# ═══════════════════════════════════════════════════════════════════════════


class TestCompleteness:
    def test_all_keywords_found(self):
        result = compute_completeness("The model uses attention and softmax", ["attention", "softmax"])
        assert result["score"] == 1.0
        assert result["missing_keywords"] == []

    def test_none_found(self):
        result = compute_completeness("Something else", ["attention", "softmax"])
        assert result["score"] == 0.0
        assert len(result["missing_keywords"]) == 2

    def test_partial_match(self):
        result = compute_completeness("Uses attention mechanism", ["attention", "softmax", "Q"])
        assert result["score"] == pytest.approx(1 / 3)
        assert "softmax" in result["missing_keywords"]
        assert "Q" in result["missing_keywords"]

    def test_case_insensitive(self):
        result = compute_completeness("Uses ATTENTION and SOFTMAX", ["attention", "softmax"])
        assert result["score"] == 1.0

    def test_empty_expected(self):
        result = compute_completeness("any answer", [])
        assert result["score"] == 1.0

    def test_substring_match(self):
        """Should match partial words (e.g., 'retriev' matches 'retrieval')."""
        result = compute_completeness("retrieval augmented generation", ["retriev", "generat"])
        assert result["score"] == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Citation Quality Queries Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCitationQualityQueries:
    def test_constant_non_empty(self):
        assert len(CITATION_QUALITY_QUERIES) > 0
        assert "question" in CITATION_QUALITY_QUERIES[0]
        assert "primary_source" in CITATION_QUALITY_QUERIES[0]

    def test_get_queries_by_topic(self):
        results = get_queries_by_topic("transformers")
        assert len(results) > 0
        assert all(q["topic"] == "transformers" for q in results)

    def test_get_queries_by_topic_empty(self):
        results = get_queries_by_topic("nonexistent")
        assert results == []

    def test_get_queries_by_type(self):
        results = get_queries_by_type("definition")
        assert len(results) > 0
        assert all(q["query_type"] == "definition" for q in results)

    def test_get_queries_by_type_empty(self):
        results = get_queries_by_type("nonexistent")
        assert results == []

    def test_get_all_primary_sources(self):
        sources = get_all_primary_sources()
        assert isinstance(sources, set)
        assert "1706.03762" in sources
        assert len(sources) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Metrics Edge Case Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMetricsEdgeCases:
    def test_extract_paper_ids_non_dict_metadata(self):
        """Cover branch 34->36: metadata is not a dict instance."""
        results = [{"metadata": "not a dict"}]
        ids = extract_paper_ids(results)
        assert ids == []

    def test_ndcg_at_k_zero_k(self):
        """Cover line 112: idcg == 0 when k=0."""
        result = ndcg_at_k(["A", "B"], ["A"], k=0)
        assert result == 0.0

    def test_citation_recall_out_of_range_source(self):
        """Cover branch 215->213: cited source number exceeds results length."""
        answer = "claim [Source 5]"
        results = [{"arxiv_id": "1706.03762"}]
        expected = ["1706.03762"]
        recall = citation_recall(answer, results, expected)
        assert recall == 0.0

    def test_citation_recall_with_metadata_fallback(self):
        """Cover lines 172-174: metadata fallback in _source_number_to_paper_id."""
        answer = "claim [Source 1]"
        results = [{"metadata": {"arxiv_id": "1706.03762"}}]
        expected = ["1706.03762"]
        recall = citation_recall(answer, results, expected)
        assert recall == 1.0

    def test_compute_citation_metrics_out_of_range_source(self):
        """Cover branch 303->301: source number out of range."""
        answer = "claim [Source 10]"
        results = [{"arxiv_id": "1706.03762"}]
        expected = ["1706.03762"]
        metrics = compute_citation_metrics(answer, results, expected)
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0

    def test_count_unsupported_no_factual_sentence(self):
        """Cover branch 272->258: sentence with no factual indicators."""
        # First sentence has citation, second has no numbers/names/technical terms
        answer = "First claim [Source 1]. the quick brown fox over the lazy dog in the forest."
        count = count_unsupported_claims(answer)
        # The second sentence should not be counted (no factual indicators)
        assert count == 0

    def test_compute_retrieval_metrics_with_acceptable_sources(self):
        """Cover the acceptable_sources path in compute_retrieval_metrics."""
        results = [{"arxiv_id": "arxiv_1706_03762", "score": 7.0}]
        metrics = compute_retrieval_metrics(
            results,
            expected_sources=["1706.03762"],
            acceptable_sources=["1706.03762", "1810.04805"],
            k=5,
        )
        assert metrics["mrr"] == 1.0
        assert metrics["precision_at_k"] == 1.0 / 5

    def test_extract_cited_bare_format(self):
        """Cover line 179: bare [N] citation format in body."""
        nums = extract_cited_source_numbers("claim [1] and [3] here")
        assert nums == [1, 3]

    def test_extract_cited_bare_format_out_of_range(self):
        """Bare [N] format filters out numbers > 20."""
        nums = extract_cited_source_numbers("claim [25] here")
        assert nums == []

    def test_extract_cited_footer_only_standard(self):
        """Cover line 184: footer-only [Source N] citations."""
        answer = "Here is the claim.\n\nSources:\n[Source 1] Paper A\n[Source 3] Paper B"
        nums = extract_cited_source_numbers(answer)
        assert nums == [1, 3]

    def test_source_number_to_paper_id_non_dict_metadata(self):
        """Cover branch 198→200: metadata is a string, not dict."""
        from rag_bench.eval.metrics import _source_number_to_paper_id

        result = _source_number_to_paper_id(1, [{"metadata": "not a dict"}])
        assert result == ""
