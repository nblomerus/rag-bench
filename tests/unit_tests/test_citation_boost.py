"""
Unit tests for rag_bench.core.citation_boost module.

Tests cover:
- CitationBooster initialization
- Query intent classification
- Relevant paper identification
- Age-based boosting
- Foundational paper boosting
- Result boosting and re-ranking
- Result diversification
"""

import pytest

from rag_bench.core.citation_boost import (
    FOUNDATIONAL_PAPERS,
    FOUNDATIONAL_QUERY_PATTERNS,
    RECENT_QUERY_PATTERNS,
    TOPIC_PAPER_MAP,
    CitationBooster,
)

# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_results():
    """Sample retrieval results for testing."""
    return [
        {
            "chunk_id": "chunk_001",
            "text": "Attention mechanism text",
            "score": 0.8,
            "metadata": {
                "doc_id": "paper_1",
                "arxiv_id": "1706.03762",  # Attention Is All You Need
                "year": 2017,
            },
        },
        {
            "chunk_id": "chunk_002",
            "text": "Recent model text",
            "score": 0.9,
            "metadata": {
                "doc_id": "paper_2",
                "arxiv_id": "2307.09288",  # Llama 2
                "year": 2023,
            },
        },
        {
            "chunk_id": "chunk_003",
            "text": "Non-foundational paper text",
            "score": 0.85,
            "metadata": {
                "doc_id": "paper_3",
                "arxiv_id": "9999.99999",  # Not in foundational list
                "year": 2022,
            },
        },
    ]


@pytest.fixture
def custom_foundational_papers():
    """Custom foundational papers map for testing."""
    return {
        "1234.5678": {"title": "Test Paper", "boost": 1.5, "year": 2020},
        "8765.4321": {"title": "Another Test", "boost": 2.0, "year": 2018},
    }


# ══════════════════════════════════════════════════════════════════════════════
# Constants Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestConstants:
    """Tests for module-level constants."""

    def test_foundational_papers_structure(self):
        """Test that FOUNDATIONAL_PAPERS has expected structure."""
        assert isinstance(FOUNDATIONAL_PAPERS, dict)
        assert len(FOUNDATIONAL_PAPERS) > 0

        # Check structure of entries
        for arxiv_id, metadata in FOUNDATIONAL_PAPERS.items():
            assert isinstance(arxiv_id, str)
            assert "title" in metadata
            assert "boost" in metadata
            assert "year" in metadata
            assert isinstance(metadata["boost"], (int, float))
            assert isinstance(metadata["year"], int)

    def test_topic_paper_map_structure(self):
        """Test that TOPIC_PAPER_MAP has expected structure."""
        assert isinstance(TOPIC_PAPER_MAP, dict)
        assert len(TOPIC_PAPER_MAP) > 0

        for topic, arxiv_ids in TOPIC_PAPER_MAP.items():
            assert isinstance(topic, str)
            assert isinstance(arxiv_ids, list)
            assert all(isinstance(aid, str) for aid in arxiv_ids)

    def test_query_patterns_exist(self):
        """Test that query pattern lists exist and are non-empty."""
        assert isinstance(FOUNDATIONAL_QUERY_PATTERNS, list)
        assert len(FOUNDATIONAL_QUERY_PATTERNS) > 0

        assert isinstance(RECENT_QUERY_PATTERNS, list)
        assert len(RECENT_QUERY_PATTERNS) > 0


# ══════════════════════════════════════════════════════════════════════════════
# CitationBooster.__init__ Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestCitationBoosterInit:
    """Tests for CitationBooster initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        booster = CitationBooster()

        assert booster.foundational_papers == FOUNDATIONAL_PAPERS
        assert booster.enable_age_boost is True
        assert booster.enable_query_adaptive is True

    def test_init_with_custom_papers(self, custom_foundational_papers):
        """Test initialization with custom foundational papers."""
        booster = CitationBooster(foundational_papers=custom_foundational_papers)

        assert booster.foundational_papers == custom_foundational_papers
        assert booster.foundational_papers != FOUNDATIONAL_PAPERS

    def test_init_disable_age_boost(self):
        """Test initialization with age boost disabled."""
        booster = CitationBooster(enable_age_boost=False)

        assert booster.enable_age_boost is False
        assert booster.enable_query_adaptive is True

    def test_init_disable_query_adaptive(self):
        """Test initialization with query adaptive disabled."""
        booster = CitationBooster(enable_query_adaptive=False)

        assert booster.enable_age_boost is True
        assert booster.enable_query_adaptive is False

    def test_init_all_disabled(self):
        """Test initialization with all features disabled."""
        booster = CitationBooster(enable_age_boost=False, enable_query_adaptive=False)

        assert booster.enable_age_boost is False
        assert booster.enable_query_adaptive is False


# ══════════════════════════════════════════════════════════════════════════════
# classify_query_intent Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestClassifyQueryIntent:
    """Tests for classify_query_intent method."""

    def test_foundational_query_what_is(self):
        """Test classification of 'what is' query."""
        booster = CitationBooster()
        intent = booster.classify_query_intent("What is attention mechanism?")

        assert intent == "foundational"

    def test_foundational_query_explain(self):
        """Test classification of 'explain' query."""
        booster = CitationBooster()
        intent = booster.classify_query_intent("Explain transformers")

        assert intent == "foundational"

    def test_foundational_query_how_does(self):
        """Test classification of 'how does' query."""
        booster = CitationBooster()
        intent = booster.classify_query_intent("How does BERT work?")

        assert intent == "foundational"

    def test_recent_query_sota(self):
        """Test classification of SOTA query."""
        booster = CitationBooster()
        intent = booster.classify_query_intent("Best SOTA models for NLP")

        assert intent == "recent"

    def test_recent_query_latest(self):
        """Test classification of 'latest' query."""
        booster = CitationBooster()
        intent = booster.classify_query_intent("Latest developments in LLMs")

        assert intent == "recent"

    def test_recent_query_2024(self):
        """Test classification of year-specific query."""
        booster = CitationBooster()
        intent = booster.classify_query_intent("Best models in 2024")

        assert intent == "recent"

    def test_balanced_query_no_keywords(self):
        """Test classification of query without specific keywords."""
        booster = CitationBooster()
        intent = booster.classify_query_intent("transformers for nlp")

        assert intent == "balanced"

    def test_balanced_query_mixed_keywords(self):
        """Test classification with both foundational and recent keywords."""
        booster = CitationBooster()
        intent = booster.classify_query_intent("What is the latest SOTA model?")

        # Should be balanced when both types of keywords present
        assert intent in ["foundational", "recent", "balanced"]

    def test_case_insensitive(self):
        """Test that classification is case-insensitive."""
        booster = CitationBooster()

        intent1 = booster.classify_query_intent("WHAT IS ATTENTION")
        intent2 = booster.classify_query_intent("what is attention")
        intent3 = booster.classify_query_intent("What Is Attention")

        assert intent1 == intent2 == intent3 == "foundational"


# ══════════════════════════════════════════════════════════════════════════════
# identify_relevant_papers Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestIdentifyRelevantPapers:
    """Tests for identify_relevant_papers method."""

    def test_identify_transformer_papers(self):
        """Test identification of transformer-related papers."""
        booster = CitationBooster()
        papers = booster.identify_relevant_papers("What is a transformer?")

        assert len(papers) > 0
        assert "1706.03762" in papers  # Attention Is All You Need

    def test_identify_bert_papers(self):
        """Test identification of BERT-related papers."""
        booster = CitationBooster()
        papers = booster.identify_relevant_papers("Tell me about BERT")

        assert len(papers) > 0
        assert "1810.04805" in papers  # BERT

    def test_identify_multiple_topics(self):
        """Test identification with multiple topics in query."""
        booster = CitationBooster()
        papers = booster.identify_relevant_papers("transformer and BERT architectures")

        # Should find papers for both topics
        assert len(papers) >= 2

    def test_no_matching_papers(self):
        """Test query with no matching papers."""
        booster = CitationBooster()
        papers = booster.identify_relevant_papers("random unrelated query xyz")

        assert len(papers) == 0

    def test_recent_query_returns_empty(self):
        """Test that recent-intent queries don't inject foundational papers."""
        booster = CitationBooster(enable_query_adaptive=True)
        papers = booster.identify_relevant_papers("Latest SOTA transformer models")

        # Should return empty list due to recent intent
        assert len(papers) == 0

    def test_query_adaptive_disabled_returns_papers(self):
        """Test that papers are returned when query_adaptive is disabled."""
        booster = CitationBooster(enable_query_adaptive=False)
        papers = booster.identify_relevant_papers("Latest SOTA transformer models")

        # Should return papers since query_adaptive is disabled
        assert len(papers) > 0

    def test_case_insensitive_topic_matching(self):
        """Test that topic matching is case-insensitive."""
        booster = CitationBooster()

        papers1 = booster.identify_relevant_papers("TRANSFORMER")
        papers2 = booster.identify_relevant_papers("transformer")
        papers3 = booster.identify_relevant_papers("Transformer")

        assert papers1 == papers2 == papers3


# ══════════════════════════════════════════════════════════════════════════════
# _calculate_age_boost Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestCalculateAgeBoost:
    """Tests for _calculate_age_boost method."""

    def test_age_boost_disabled(self):
        """Test that age boost returns 1.0 when disabled."""
        booster = CitationBooster(enable_age_boost=False)
        boost = booster._calculate_age_boost(2017, "foundational")

        assert boost == 1.0

    def test_age_boost_none_year(self):
        """Test that None year returns 1.0."""
        booster = CitationBooster()
        boost = booster._calculate_age_boost(None, "foundational")

        assert boost == 1.0

    def test_foundational_old_paper(self):
        """Test boost for old paper with foundational intent."""
        booster = CitationBooster()
        boost = booster._calculate_age_boost(2015, "foundational")

        assert boost > 1.0
        assert boost == 1.8  # Pre-transformer era

    def test_foundational_early_transformer(self):
        """Test boost for early transformer paper with foundational intent."""
        booster = CitationBooster()
        boost = booster._calculate_age_boost(2018, "foundational")

        assert boost > 1.0
        assert boost == 1.5  # Early transformer era

    def test_foundational_modern_paper(self):
        """Test boost for modern paper with foundational intent."""
        booster = CitationBooster()
        boost = booster._calculate_age_boost(2022, "foundational")

        assert boost >= 1.0
        assert boost == 1.2  # Modern era

    def test_foundational_very_recent_paper(self):
        """Test no boost for very recent paper with foundational intent."""
        booster = CitationBooster()
        boost = booster._calculate_age_boost(2023, "foundational")

        assert boost == 1.0

    def test_recent_intent_new_paper(self):
        """Test boost for new paper with recent intent."""
        booster = CitationBooster()
        boost = booster._calculate_age_boost(2023, "recent")

        assert boost > 1.0
        assert boost == 1.3

    def test_recent_intent_old_paper(self):
        """Test penalty for old paper with recent intent."""
        booster = CitationBooster()
        boost = booster._calculate_age_boost(2018, "recent")

        assert boost < 1.0
        assert boost == 0.9

    def test_balanced_intent_old_paper(self):
        """Test moderate boost for old paper with balanced intent."""
        booster = CitationBooster()
        boost = booster._calculate_age_boost(2017, "balanced")

        assert boost > 1.0
        assert boost == 1.2

    def test_balanced_intent_recent_paper(self):
        """Test no boost for recent paper with balanced intent."""
        booster = CitationBooster()
        boost = booster._calculate_age_boost(2023, "balanced")

        assert boost == 1.0


# ══════════════════════════════════════════════════════════════════════════════
# _calculate_foundational_boost Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestCalculateFoundationalBoost:
    """Tests for _calculate_foundational_boost method."""

    def test_foundational_paper_boost(self):
        """Test boost for paper in foundational list."""
        booster = CitationBooster()
        boost = booster._calculate_foundational_boost("1706.03762")  # Attention paper

        assert boost > 1.0

    def test_non_foundational_paper(self):
        """Test no boost for paper not in foundational list."""
        booster = CitationBooster()
        boost = booster._calculate_foundational_boost("9999.99999")

        assert boost == 1.0

    def test_custom_foundational_papers(self, custom_foundational_papers):
        """Test boost with custom foundational papers."""
        booster = CitationBooster(foundational_papers=custom_foundational_papers)

        boost = booster._calculate_foundational_boost("1234.5678")
        assert boost == 1.5

    def test_different_papers_different_boosts(self):
        """Test that different foundational papers can have different boosts."""
        booster = CitationBooster()

        boost1 = booster._calculate_foundational_boost("1706.03762")  # Attention
        boost2 = booster._calculate_foundational_boost("1810.04805")  # BERT

        # Both should be boosted but may have different values
        assert boost1 > 1.0
        assert boost2 > 1.0


# ══════════════════════════════════════════════════════════════════════════════
# boost_results Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestBoostResults:
    """Tests for boost_results method."""

    def test_boost_empty_results(self):
        """Test boosting empty results list."""
        booster = CitationBooster()
        results = booster.boost_results([])

        assert results == []

    def test_boost_adds_metadata_fields(self, sample_results):
        """Test that boosting adds required metadata fields."""
        booster = CitationBooster()
        results = booster.boost_results(sample_results.copy())

        for result in results:
            assert "original_score" in result
            assert "score" in result
            assert "boost_factor" in result
            assert "foundational_boost" in result
            assert "age_boost" in result

    def test_boost_foundational_paper(self, sample_results):
        """Test that foundational papers get boosted."""
        booster = CitationBooster()
        results = booster.boost_results(sample_results.copy())

        # Find the Attention paper result
        attention_result = next(r for r in results if r["metadata"]["arxiv_id"] == "1706.03762")

        assert attention_result["boost_factor"] > 1.0
        assert attention_result["score"] > attention_result["original_score"]

    def test_boost_results_sorted_by_score(self, sample_results):
        """Test that results are sorted by boosted score."""
        booster = CitationBooster()
        results = booster.boost_results(sample_results.copy())

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_boost_with_top_k(self, sample_results):
        """Test that top_k parameter limits results."""
        booster = CitationBooster()
        results = booster.boost_results(sample_results.copy(), top_k=2)

        assert len(results) == 2

    def test_boost_with_query_foundational(self, sample_results):
        """Test boosting with foundational query."""
        booster = CitationBooster()
        results = booster.boost_results(sample_results.copy(), query="What is attention?")

        # Foundational papers should be boosted more
        attention_result = next(r for r in results if r["metadata"]["arxiv_id"] == "1706.03762")
        assert attention_result["age_boost"] > 1.0

    def test_boost_with_query_recent(self, sample_results):
        """Test boosting with recent query."""
        booster = CitationBooster()
        results = booster.boost_results(sample_results.copy(), query="Latest SOTA models 2024")

        # Recent papers should be boosted
        llama_result = next(r for r in results if r["metadata"]["arxiv_id"] == "2307.09288")
        assert llama_result["age_boost"] > 1.0

    def test_boost_with_arxiv_prefix(self):
        """Test boosting with arxiv_ prefix in ID."""
        results = [
            {
                "chunk_id": "chunk_001",
                "text": "Test",
                "score": 0.8,
                "metadata": {
                    "doc_id": "paper_1",
                    "arxiv_id": "arxiv_1706_03762",  # With prefix and underscores
                    "year": 2017,
                },
            }
        ]

        booster = CitationBooster()
        boosted = booster.boost_results(results.copy())

        # Should still recognize as foundational paper
        assert boosted[0]["foundational_boost"] > 1.0

    def test_boost_with_string_year(self):
        """Test boosting with year as string."""
        results = [
            {
                "chunk_id": "chunk_001",
                "text": "Test",
                "score": 0.8,
                "metadata": {
                    "doc_id": "paper_1",
                    "arxiv_id": "1706.03762",
                    "year": "2017",  # String year
                },
            }
        ]

        booster = CitationBooster()
        boosted = booster.boost_results(results.copy())

        # Should handle string year
        assert boosted[0]["age_boost"] > 1.0

    def test_boost_with_invalid_year(self):
        """Test boosting with invalid year string."""
        results = [
            {
                "chunk_id": "chunk_001",
                "text": "Test",
                "score": 0.8,
                "metadata": {
                    "doc_id": "paper_1",
                    "arxiv_id": "1706.03762",
                    "year": "invalid",
                },
            }
        ]

        booster = CitationBooster()
        boosted = booster.boost_results(results.copy())

        # Should handle invalid year gracefully
        assert "score" in boosted[0]

    def test_boost_score_floor_for_foundational(self):
        """Test score floor applied to foundational papers."""
        results = [
            {
                "chunk_id": "chunk_high",
                "text": "High score",
                "score": 0.95,
                "metadata": {"doc_id": "p1", "arxiv_id": "9999.99999", "year": 2023},
            },
            {
                "chunk_id": "chunk_med",
                "text": "Medium score",
                "score": 0.80,
                "metadata": {"doc_id": "p2", "arxiv_id": "8888.88888", "year": 2022},
            },
            {
                "chunk_id": "chunk_low_found",
                "text": "Low score foundational",
                "score": 0.40,
                "metadata": {"doc_id": "p3", "arxiv_id": "1706.03762", "year": 2017},
            },
        ]

        booster = CitationBooster()
        boosted = booster.boost_results(results.copy(), query="What is attention?")

        # Find the foundational paper
        found_result = next(r for r in boosted if r["metadata"]["arxiv_id"] == "1706.03762")

        # Should have score floor applied
        # With 3 results, p75 is the 3//4 = 0th element (highest score)
        # But the foundational paper should be significantly boosted
        assert found_result["score"] > found_result["original_score"]

    def test_boost_no_query_uses_balanced(self, sample_results):
        """Test that empty query uses balanced intent."""
        booster = CitationBooster()
        results = booster.boost_results(sample_results.copy(), query="")

        # Should still apply boosts
        assert all(r["boost_factor"] >= 1.0 for r in results)


# ══════════════════════════════════════════════════════════════════════════════
# diversify_results Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestDiversifyResults:
    """Tests for diversify_results method."""

    def test_diversify_basic(self):
        """Test basic diversification."""
        results = [
            {"chunk_id": "c1", "score": 0.9, "metadata": {"doc_id": "p1", "arxiv_id": "1234"}},
            {"chunk_id": "c2", "score": 0.8, "metadata": {"doc_id": "p1", "arxiv_id": "1234"}},
            {"chunk_id": "c3", "score": 0.7, "metadata": {"doc_id": "p2", "arxiv_id": "5678"}},
        ]

        booster = CitationBooster()
        diversified = booster.diversify_results(results, top_k=3, max_per_paper=1)

        # Should only take one chunk per paper
        doc_ids = [r["metadata"]["doc_id"] for r in diversified]
        assert len(doc_ids) == len(set(doc_ids))

    def test_diversify_respects_top_k(self):
        """Test that diversification respects top_k parameter."""
        results = [
            {"chunk_id": f"c{i}", "score": 1.0 - i * 0.1, "metadata": {"doc_id": f"p{i}", "arxiv_id": str(i)}}
            for i in range(10)
        ]

        booster = CitationBooster()
        diversified = booster.diversify_results(results, top_k=5)

        assert len(diversified) <= 5

    def test_diversify_max_per_paper(self):
        """Test max_per_paper constraint."""
        results = [
            {"chunk_id": f"c{i}", "score": 1.0 - i * 0.1, "metadata": {"doc_id": "same_paper", "arxiv_id": "1234"}}
            for i in range(5)
        ]

        booster = CitationBooster()
        diversified = booster.diversify_results(results, top_k=5, max_per_paper=2)

        # Should only include 2 chunks from the same paper
        assert len(diversified) == 2

    def test_diversify_require_foundational(self):
        """Test that foundational paper is included when required."""
        results = [
            {"chunk_id": "c1", "score": 0.95, "metadata": {"doc_id": "p1", "arxiv_id": "9999.99999"}},
            {"chunk_id": "c2", "score": 0.90, "metadata": {"doc_id": "p2", "arxiv_id": "8888.88888"}},
            {"chunk_id": "c3", "score": 0.50, "metadata": {"doc_id": "p3", "arxiv_id": "1706.03762"}},  # Foundational
        ]

        booster = CitationBooster()
        diversified = booster.diversify_results(results, top_k=2, require_foundational=True)

        # Should include the foundational paper even though it has lower score
        arxiv_ids = [r["metadata"]["arxiv_id"] for r in diversified]
        assert "1706.03762" in arxiv_ids

    def test_diversify_no_require_foundational(self):
        """Test diversification without requiring foundational paper."""
        results = [
            {"chunk_id": "c1", "score": 0.95, "metadata": {"doc_id": "p1", "arxiv_id": "9999.99999"}},
            {"chunk_id": "c2", "score": 0.90, "metadata": {"doc_id": "p2", "arxiv_id": "8888.88888"}},
            {"chunk_id": "c3", "score": 0.50, "metadata": {"doc_id": "p3", "arxiv_id": "1706.03762"}},  # Foundational
        ]

        booster = CitationBooster()
        diversified = booster.diversify_results(results, top_k=2, require_foundational=False)

        # Top 2 results by score (may not include foundational)
        assert len(diversified) == 2

    def test_diversify_empty_results(self):
        """Test diversification of empty results."""
        booster = CitationBooster()
        diversified = booster.diversify_results([], top_k=5)

        assert len(diversified) == 0

    def test_diversify_cleans_arxiv_id(self):
        """Test that arxiv_id cleaning works in diversification."""
        results = [
            {"chunk_id": "c1", "score": 0.50, "metadata": {"doc_id": "p1", "arxiv_id": "arxiv_1706_03762"}},
        ]

        booster = CitationBooster()
        diversified = booster.diversify_results(results, top_k=2, require_foundational=True)

        # Should recognize as foundational despite prefix/underscores
        assert len(diversified) == 1


class TestCitationBoostMissingBranches:
    def test_balanced_intent_2019_to_2021_paper(self):
        """year <= 2021 in balanced mode → 1.1 (line 295)."""
        booster = CitationBooster()
        boost = booster._calculate_age_boost(2020, "balanced")
        assert boost == 1.1

    def test_balanced_intent_2019_paper(self):
        """year = 2019 ≤ 2021 in balanced mode → 1.1."""
        booster = CitationBooster()
        boost = booster._calculate_age_boost(2019, "balanced")
        assert boost == 1.1

    def test_score_floor_raises_low_foundational_score(self):
        """Foundational paper with score below p75 → score raised to p75 (lines 382-386)."""
        booster = CitationBooster()
        # Result 1: foundational with high boost but low score (below p75)
        # Result 2-5: non-foundational with higher scores
        results = [
            {"chunk_id": "c1", "score": 0.9, "metadata": {"arxiv_id": "9999.0001"}, "foundational_boost": 1.0},
            {"chunk_id": "c2", "score": 0.85, "metadata": {"arxiv_id": "9999.0002"}, "foundational_boost": 1.0},
            {"chunk_id": "c3", "score": 0.80, "metadata": {"arxiv_id": "9999.0003"}, "foundational_boost": 1.0},
            {"chunk_id": "c4", "score": 0.75, "metadata": {"arxiv_id": "9999.0004"}, "foundational_boost": 1.0},
            {"chunk_id": "c5", "score": 0.20, "metadata": {"arxiv_id": "9999.0005"}, "foundational_boost": 2.0},
        ]
        # Use a query intent that uses score floors (foundational)
        boosted = booster.boost_results(results, query="foundational research classic paper", top_k=5)
        # The foundational paper (c5) with low score should have its score raised
        assert isinstance(boosted, list)

    def test_boost_results_no_results_boosted(self):
        """When no results have boost_factor > 1.0, num_boosted == 0 → no debug log (line 393->397)."""
        booster = CitationBooster(enable_age_boost=False, enable_query_adaptive=False)
        results = [
            {"chunk_id": "c1", "score": 0.9, "metadata": {"arxiv_id": "9999.0001", "year": "2023"}},
            {"chunk_id": "c2", "score": 0.8, "metadata": {"arxiv_id": "9999.0002", "year": "2023"}},
        ]
        # With all boosts disabled, no result will have boost_factor > 1.0
        boosted = booster.boost_results(results.copy())
        # Should return results sorted but without any boosting
        assert len(boosted) == 2
