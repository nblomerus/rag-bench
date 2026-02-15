"""
citation_boost.py — Boost retrieval scores for foundational/highly-cited papers.

This module provides metadata-based boosting strategies to improve citation
quality by prioritizing foundational papers in retrieval results.

Usage:
    from rag_bench.core.citation_boost import CitationBooster

    booster = CitationBooster()
    results = retriever.query("What is attention?", top_k=20)
    boosted = booster.boost_results(results, top_k=5)
"""

import logging
from typing import Literal

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Curated list of foundational papers (manually tagged)
# ═══════════════════════════════════════════════════════════════════════════
FOUNDATIONAL_PAPERS = {
    # Transformers & Attention
    "1706.03762": {"title": "Attention Is All You Need", "boost": 2.0, "year": 2017},  # Increased from 1.5
    "1810.04805": {"title": "BERT", "boost": 1.8, "year": 2018},  # Increased from 1.5
    "1910.10683": {"title": "T5", "boost": 1.6, "year": 2019},  # Increased from 1.4
    "2005.14165": {"title": "GPT-3", "boost": 1.6, "year": 2020},  # Increased from 1.4
    "2307.09288": {"title": "Llama 2", "boost": 1.4, "year": 2023},  # Increased from 1.3
    "2310.06825": {"title": "Mistral 7B", "boost": 1.4, "year": 2023},  # Increased from 1.3
    "2401.04088": {"title": "Mixtral", "boost": 1.4, "year": 2024},  # Increased from 1.3
    # RAG & Retrieval
    "2005.11401": {"title": "RAG", "boost": 1.5, "year": 2020},
    "2210.07316": {"title": "REPLUG", "boost": 1.3, "year": 2022},
    "2212.10496": {"title": "Self-RAG", "boost": 1.3, "year": 2022},
    # Vision
    "1512.03385": {"title": "ResNet", "boost": 1.5, "year": 2015},
    "1409.1556": {"title": "VGG", "boost": 1.4, "year": 2014},
    "2010.11929": {"title": "Vision Transformer", "boost": 1.4, "year": 2020},
    # Optimization & Training
    "1412.6980": {"title": "Adam", "boost": 1.5, "year": 2014},
    "1502.03167": {"title": "Batch Normalization", "boost": 1.4, "year": 2015},
    "1607.06450": {"title": "Layer Normalization", "boost": 1.3, "year": 2016},
    # Parameter Efficiency
    "2106.09685": {"title": "LoRA", "boost": 1.4, "year": 2021},
    "2110.04366": {"title": "Prefix Tuning", "boost": 1.3, "year": 2021},
    # Reinforcement Learning
    "1707.06347": {"title": "PPO", "boost": 1.5, "year": 2017},
    "2203.02155": {"title": "InstructGPT", "boost": 1.4, "year": 2022},
    # Evaluation
    "2002.08512": {"title": "BLEU Score", "boost": 1.3, "year": 2002},
    "1803.05449": {"title": "ROUGE", "boost": 1.3, "year": 2004},
}


# ═══════════════════════════════════════════════════════════════════════════
# Query Classification for Intent-Based Boosting
# ═══════════════════════════════════════════════════════════════════════════
TOPIC_PAPER_MAP: dict[str, list[str]] = {
    # Maps topic keywords (lowercased) to arxiv_ids from FOUNDATIONAL_PAPERS.
    # When a query mentions a topic, the corresponding paper's chunks are
    # fetched directly from ChromaDB and injected into the rerank pool.
    "transformer": ["1706.03762"],
    "attention": ["1706.03762"],
    "self-attention": ["1706.03762"],
    "self attention": ["1706.03762"],
    "multi-head attention": ["1706.03762"],
    "bert": ["1810.04805"],
    "masked language model": ["1810.04805"],
    "t5": ["1910.10683"],
    "text-to-text": ["1910.10683"],
    "gpt-3": ["2005.14165"],
    "gpt3": ["2005.14165"],
    "in-context learning": ["2005.14165"],
    "llama": ["2307.09288"],
    "mistral": ["2310.06825"],
    "mixtral": ["2401.04088"],
    "mixture of experts": ["2401.04088"],
    "rag": ["2005.11401"],
    "retrieval augmented generation": ["2005.11401"],
    "retrieval-augmented": ["2005.11401"],
    "resnet": ["1512.03385"],
    "residual network": ["1512.03385"],
    "vgg": ["1409.1556"],
    "vision transformer": ["2010.11929"],
    "vit": ["2010.11929"],
    "adam optimizer": ["1412.6980"],
    "batch normalization": ["1502.03167"],
    "batch norm": ["1502.03167"],
    "layer normalization": ["1607.06450"],
    "layer norm": ["1607.06450"],
    "lora": ["2106.09685"],
    "low-rank adaptation": ["2106.09685"],
    "prefix tuning": ["2110.04366"],
    "ppo": ["1707.06347"],
    "proximal policy": ["1707.06347"],
    "instructgpt": ["2203.02155"],
    "rlhf": ["2203.02155", "1707.06347"],
}


FOUNDATIONAL_QUERY_PATTERNS = [
    "what is",
    "what are",
    "explain",
    "introduce",
    "definition of",
    "basics of",
    "fundamental",
    "how does",
    "how do",
    "original paper",
    "first proposed",
    "seminal work",
    "who invented",
    "who created",
    "history of",
]

RECENT_QUERY_PATTERNS = [
    "sota",
    "state of the art",
    "state-of-the-art",
    "latest",
    "recent",
    "2023",
    "2024",
    "2025",
    "current",
    "modern",
    "new",
    "advances in",
    "improvements",
    "compared to",
]


class CitationBooster:
    """
    Boost retrieval scores to prioritize foundational papers.

    Provides multiple boosting strategies:
    1. Foundational tagging - Boost manually curated foundational papers
    2. Age-based boosting - Favor older papers (likely more foundational)
    3. Query-adaptive boosting - Adjust based on query intent
    """

    def __init__(
        self,
        foundational_papers: dict | None = None,
        enable_age_boost: bool = True,
        enable_query_adaptive: bool = True,
    ):
        """
        Initialize the citation booster.

        Args:
            foundational_papers: Dict mapping arxiv_id -> metadata (uses default if None)
            enable_age_boost: Whether to boost older papers
            enable_query_adaptive: Whether to adapt boosting based on query intent
        """
        self.foundational_papers = foundational_papers or FOUNDATIONAL_PAPERS
        self.enable_age_boost = enable_age_boost
        self.enable_query_adaptive = enable_query_adaptive

        logger.info(
            f"CitationBooster initialized: {len(self.foundational_papers)} foundational papers, "
            f"age_boost={enable_age_boost}, query_adaptive={enable_query_adaptive}"
        )

    def classify_query_intent(self, query: str) -> Literal["foundational", "recent", "balanced"]:
        """
        Classify whether query asks for foundational knowledge or recent work.

        Args:
            query: User's search query

        Returns:
            "foundational", "recent", or "balanced"
        """
        query_lower = query.lower()

        foundational_score = sum(1 for pattern in FOUNDATIONAL_QUERY_PATTERNS if pattern in query_lower)

        recent_score = sum(1 for pattern in RECENT_QUERY_PATTERNS if pattern in query_lower)

        if foundational_score > recent_score:
            return "foundational"
        elif recent_score > foundational_score:
            return "recent"
        else:
            return "balanced"

    def identify_relevant_papers(self, query: str) -> list[str]:
        """
        Identify foundational papers relevant to the query based on topic matching.

        Returns a list of arxiv_ids whose chunks should be directly fetched
        and injected into the candidate pool.
        """
        query_lower = query.lower()
        matched_ids: set[str] = set()

        for topic, arxiv_ids in TOPIC_PAPER_MAP.items():
            if topic in query_lower:
                matched_ids.update(arxiv_ids)

        if not matched_ids:
            return []

        # Don't inject foundational papers for "recent/SOTA" queries
        if self.enable_query_adaptive:
            intent = self.classify_query_intent(query)
            if intent == "recent":
                return []

        return list(matched_ids)

    def _calculate_age_boost(self, year: int | None, query_intent: str) -> float:
        """
        Calculate boost factor based on paper age.

        Args:
            year: Publication year
            query_intent: "foundational", "recent", or "balanced"

        Returns:
            Boost multiplier (1.0 = no boost)
        """
        if not self.enable_age_boost or year is None:
            return 1.0

        if query_intent == "foundational":
            # Strongly favor older papers for foundational queries
            if year <= 2017:
                return 1.8  # Pre-transformer era (increased from 1.4)
            elif year <= 2020:
                return 1.5  # Early transformer era (increased from 1.2)
            elif year <= 2022:
                return 1.2  # Modern era (increased from 1.1)
            else:
                return 1.0  # Very recent

        elif query_intent == "recent":
            # Favor recent papers
            if year >= 2023:
                return 1.3
            elif year >= 2021:
                return 1.1
            else:
                return 0.9  # Slight penalty for old papers

        else:  # balanced
            # Moderate boost for foundational era
            if year <= 2018:
                return 1.2
            elif year <= 2021:
                return 1.1
            else:
                return 1.0

    def _calculate_foundational_boost(self, arxiv_id: str) -> float:
        """
        Calculate boost for manually tagged foundational papers.

        Args:
            arxiv_id: arXiv ID of the paper

        Returns:
            Boost multiplier (1.0 = no boost)
        """
        if arxiv_id in self.foundational_papers:
            return self.foundational_papers[arxiv_id].get("boost", 1.3)
        return 1.0

    def boost_results(
        self,
        results: list[dict],
        query: str = "",
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Apply citation boosting to retrieval results.

        Args:
            results: List of retrieval results with 'score', 'metadata' fields
            query: Original user query (for intent classification)
            top_k: Number of results to return after boosting (None = all)

        Returns:
            Boosted and re-sorted results
        """
        if not results:
            return results

        # Classify query intent
        query_intent = "balanced"
        if self.enable_query_adaptive and query:
            query_intent = self.classify_query_intent(query)
            logger.debug(f"Query intent: {query_intent}")

        # Apply boosting
        for result in results:
            metadata = result.get("metadata", {})
            original_score = result.get("score", 0.0)

            # Extract paper metadata
            arxiv_id = metadata.get("arxiv_id", "")
            # Clean arxiv_id (remove arxiv_ prefix if present)
            arxiv_id_clean = arxiv_id.replace("arxiv_", "").replace("_", ".")

            year = metadata.get("year")
            if isinstance(year, str):
                try:
                    year = int(year)
                except ValueError:
                    year = None

            # Calculate boost factors
            foundational_boost = self._calculate_foundational_boost(arxiv_id_clean)
            age_boost = self._calculate_age_boost(year, query_intent)

            # Combine boosts (multiplicative)
            total_boost = foundational_boost * age_boost

            # Apply boost
            boosted_score = original_score * total_boost

            # Store both scores for transparency
            result["original_score"] = original_score
            result["score"] = boosted_score
            result["boost_factor"] = total_boost
            result["foundational_boost"] = foundational_boost
            result["age_boost"] = age_boost

        # Score floor for foundational papers on foundational-intent queries.
        # Ensures injected foundational papers survive downstream score-gap
        # filters even when the cross-encoder gave them modest raw scores.
        if query_intent == "foundational":
            scores_desc = sorted((r["score"] for r in results), reverse=True)
            p75 = scores_desc[len(scores_desc) // 4] if len(scores_desc) >= 4 else scores_desc[-1] if scores_desc else 0.0

            for result in results:
                if result.get("foundational_boost", 1.0) > 1.0 and result["score"] < p75:
                    logger.debug(
                        f"Score floor: {result['score']:.2f} -> {p75:.2f} "
                        f"for {result.get('metadata', {}).get('arxiv_id', '?')}"
                    )
                    result["score"] = p75

        # Re-sort by boosted score
        results.sort(key=lambda x: x["score"], reverse=True)

        # Log boosting effects
        num_boosted = sum(1 for r in results if r.get("boost_factor", 1.0) > 1.0)
        if num_boosted > 0:
            logger.debug(f"Boosted {num_boosted}/{len(results)} results (intent={query_intent})")

        # Return top-k
        if top_k is not None:
            return results[:top_k]
        return results

    def diversify_results(
        self,
        results: list[dict],
        top_k: int = 5,
        max_per_paper: int = 2,
        require_foundational: bool = True,
    ) -> list[dict]:
        """
        Diversify results to avoid over-representation of single papers.

        Args:
            results: List of retrieval results
            top_k: Number of results to return
            max_per_paper: Maximum chunks from same paper
            require_foundational: Ensure at least one foundational paper if available

        Returns:
            Diversified results
        """
        selected = []
        seen_papers = {}  # doc_id -> count
        has_foundational = False

        for result in results:
            if len(selected) >= top_k:
                break

            doc_id = result.get("metadata", {}).get("doc_id", "")
            arxiv_id = result.get("metadata", {}).get("arxiv_id", "").replace("arxiv_", "").replace("_", ".")

            # Check if this paper is over-represented
            if seen_papers.get(doc_id, 0) >= max_per_paper:
                continue

            # Track foundational papers
            if arxiv_id in self.foundational_papers:
                has_foundational = True

            selected.append(result)
            seen_papers[doc_id] = seen_papers.get(doc_id, 0) + 1

        # If requiring foundational and we don't have one, try to add one
        if require_foundational and not has_foundational and len(results) > len(selected):
            for result in results:
                arxiv_id = result.get("metadata", {}).get("arxiv_id", "").replace("arxiv_", "").replace("_", ".")
                if arxiv_id in self.foundational_papers:
                    # Replace lowest scoring non-foundational result
                    if len(selected) >= top_k:
                        selected[-1] = result
                    else:
                        selected.append(result)
                    logger.debug(f"Added foundational paper: {arxiv_id}")
                    break

        return selected[:top_k]
