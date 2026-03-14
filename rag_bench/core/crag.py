"""
crag.py — Corrective RAG (CRAG) retriever wrapper.

Implements the CRAG loop as a drop-in Retriever:

1. Retrieve with the base retriever
2. Score confidence using reranker scores
3. If low confidence → rewrite query (HyDE) → re-retrieve → merge
4. Refine: strip low-relevance results

The CRAGRetriever conforms to the Retriever protocol, so the rest
of the pipeline (generator, eval runner) doesn't know CRAG exists.

Reference: Yan et al., "Corrective Retrieval Augmented Generation" (2024)

Usage:
    base = HybridRetriever(config=retriever_config)
    crag = CRAGRetriever(base_retriever=base, config=crag_config)
    results = crag.retrieve("What optimizer does GPT-4 use?")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum

import requests

from rag_bench.core.types import RetrievalResult

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Retrieval confidence classification."""

    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


@dataclass
class CRAGConfig:
    """Configuration for the CRAG retriever wrapper.

    Threshold calibration notes (from benchmark analysis):
    - Rerank scores are sigmoid-normalized to [0, 1]
    - P25 of top-1 scores = 0.97; clear drop-off below 0.85
    - Scores < 0.70 strongly correlate with irrelevant top results
    """

    # Confidence thresholds (on top-1 rerank score)
    correct_threshold: float = 0.90
    ambiguous_threshold: float = 0.70

    # Score floor for knowledge refinement — results below this are dropped
    refinement_floor: float = 0.30

    # HyDE query rewriting
    hyde_enabled: bool = True
    ollama_model: str = "qwen2.5:14b-instruct-q4_K_M"
    ollama_base_url: str = "http://localhost:11434"
    hyde_max_tokens: int = 256
    hyde_temperature: float = 0.3

    # Re-retrieval
    max_rewrites: int = 1

    # Merge strategy: how to combine original + rewritten results
    # "interleave" = round-robin by score, "rewritten_first" = prefer rewrite
    merge_strategy: str = "interleave"


@dataclass
class CRAGStats:
    """Tracks CRAG decisions across queries for analysis."""

    total_queries: int = 0
    correct_count: int = 0
    ambiguous_count: int = 0
    incorrect_count: int = 0
    rewrites_attempted: int = 0
    results_filtered: int = 0

    def summary(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "correct_pct": self.correct_count / max(1, self.total_queries),
            "ambiguous_pct": self.ambiguous_count / max(1, self.total_queries),
            "incorrect_pct": self.incorrect_count / max(1, self.total_queries),
            "rewrites_attempted": self.rewrites_attempted,
            "results_filtered": self.results_filtered,
        }


class CRAGRetriever:
    """Corrective RAG wrapper around any base retriever.

    Conforms to the Retriever protocol — drop-in replacement.

    Parameters
    ----------
    base_retriever
        Any object with a ``retrieve(query, top_k) -> list[RetrievalResult]``
        method (i.e., satisfies the Retriever protocol).
    config : CRAGConfig
        Tuning parameters for confidence thresholds, HyDE, etc.
    """

    def __init__(self, base_retriever, config: CRAGConfig | None = None):
        self.base = base_retriever
        self.config = config or CRAGConfig()
        self.stats = CRAGStats()

    # -- Public API (Retriever protocol) ------------------------------------

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Retrieve with corrective loop.

        1. Base retrieval
        2. Confidence scoring
        3. If AMBIGUOUS/INCORRECT: HyDE rewrite → re-retrieve → merge
        4. Knowledge refinement (filter low-score results)
        """
        t0 = time.time()
        self.stats.total_queries += 1

        # Step 1: Base retrieval
        results = self.base.retrieve(query, top_k=top_k)
        if not results:
            return results

        # Step 2: Score confidence
        confidence = self._score_confidence(results)

        if confidence == ConfidenceLevel.CORRECT:
            self.stats.correct_count += 1
            logger.debug(f"CRAG: CORRECT ({self._top_score(results):.3f}) — {query[:60]}")
        elif confidence == ConfidenceLevel.AMBIGUOUS:
            self.stats.ambiguous_count += 1
            logger.info(f"CRAG: AMBIGUOUS ({self._top_score(results):.3f}) — {query[:60]}")
            # AMBIGUOUS: don't rewrite — original results are usually decent.
            # Only apply refinement (step 4) to clean up low-scoring tail.
        else:
            self.stats.incorrect_count += 1
            logger.info(f"CRAG: INCORRECT ({self._top_score(results):.3f}) — rewriting: {query[:60]}")
            # INCORRECT: base retrieval failed — HyDE rewrite is our best shot
            if self.config.hyde_enabled:
                results = self._rewrite_and_retrieve(query, results, top_k)

        # Step 4: Knowledge refinement
        results = self._refine(results, top_k)

        elapsed = (time.time() - t0) * 1000
        logger.debug(f"CRAG: {confidence.value} | {len(results)} results | {elapsed:.0f}ms")

        return results

    # -- Confidence scoring -------------------------------------------------

    def _score_confidence(self, results: list[RetrievalResult]) -> ConfidenceLevel:
        """Classify retrieval confidence from reranker scores.

        Uses two signals:
        1. Top-1 rerank score — primary signal
        2. Score concentration — if top scores are bunched together,
           the model isn't discriminating well (lower confidence)
        """
        top_score = self._top_score(results)

        if top_score >= self.config.correct_threshold:
            # Additional check: is the top result clearly better than the rest?
            # If scores are flat (no discrimination), downgrade to AMBIGUOUS
            if len(results) >= 3:
                top3_mean = sum(self._get_score(r) for r in results[:3]) / 3
                gap = top_score - top3_mean
                # If gap < 0.01 and we're near the threshold, be cautious
                if gap < 0.005 and top_score < 0.95:
                    return ConfidenceLevel.AMBIGUOUS
            return ConfidenceLevel.CORRECT

        if top_score >= self.config.ambiguous_threshold:
            return ConfidenceLevel.AMBIGUOUS

        return ConfidenceLevel.INCORRECT

    # -- HyDE query rewriting -----------------------------------------------

    def _rewrite_and_retrieve(
        self,
        original_query: str,
        original_results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Generate a hypothetical answer (HyDE) and re-retrieve.

        HyDE insight: questions and answers live in different regions of
        embedding space.  A hypothetical answer is closer to real answer
        passages, so searching with it improves recall.
        """
        self.stats.rewrites_attempted += 1

        hyde_doc = self._generate_hyde(original_query)
        if not hyde_doc:
            return original_results

        # Re-retrieve using the hypothetical document as query
        new_results = self.base.retrieve(hyde_doc, top_k=top_k)

        # Merge original and new results
        merged = self._merge_results(original_results, new_results, top_k)

        logger.info(f"CRAG: HyDE rewrite produced {len(new_results)} results, merged to {len(merged)}")

        return merged

    def _generate_hyde(self, query: str) -> str | None:
        """Generate a hypothetical document that would answer the query.

        Uses Ollama to produce a short, factual passage — not a full answer,
        just enough to shift the embedding toward answer-space.
        """
        prompt = (
            f"Write a short, factual paragraph (3-5 sentences) that would "
            f"directly answer this question about AI/ML research. Write as if "
            f"you are quoting from an academic paper. Do not hedge or say "
            f'"I don\'t know" — just write the factual content.\n\n'
            f"Question: {query}\n\n"
            f"Factual paragraph:"
        )

        try:
            resp = requests.post(
                f"{self.config.ollama_base_url}/api/generate",
                json={
                    "model": self.config.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.hyde_temperature,
                        "num_predict": self.config.hyde_max_tokens,
                    },
                },
                timeout=60,
            )
            resp.raise_for_status()
            hyde_doc = resp.json().get("response", "").strip()

            if len(hyde_doc) < 20:
                logger.warning("CRAG: HyDE generation too short, skipping")
                return None

            logger.debug(f"CRAG: HyDE generated ({len(hyde_doc)} chars): {hyde_doc[:100]}...")
            return hyde_doc

        except requests.RequestException as e:
            logger.warning(f"CRAG: HyDE generation failed: {e}")
            return None

    # -- Merge & refinement -------------------------------------------------

    def _merge_results(
        self,
        original: list[RetrievalResult],
        rewritten: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Merge original and rewritten results, deduplicating by chunk_id.

        Interleave strategy: combine both lists, sort by score, deduplicate.
        This lets the best results from either retrieval float to the top.
        """
        seen_ids: set[str] = set()
        merged: list[RetrievalResult] = []

        # Combine and sort by score (descending)
        all_results = list(original) + list(rewritten)
        all_results.sort(key=lambda r: self._get_score(r), reverse=True)

        for result in all_results:
            cid = result.chunk.chunk_id
            if cid not in seen_ids:
                seen_ids.add(cid)
                merged.append(result)
                if len(merged) >= top_k:
                    break

        return merged

    def _refine(
        self,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Knowledge refinement: strip low-relevance results.

        Removes results below the score floor, but always keeps at least 1
        result (the best we have is better than nothing).
        """
        floor = self.config.refinement_floor
        before_count = len(results)

        refined = [r for r in results if self._get_score(r) >= floor]

        # Always keep at least the top result
        if not refined and results:
            refined = [results[0]]

        filtered_count = before_count - len(refined)
        if filtered_count > 0:
            self.stats.results_filtered += filtered_count
            logger.debug(f"CRAG: Refined {before_count} → {len(refined)} results")

        return refined[:top_k]

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _get_score(result: RetrievalResult) -> float:
        """Get the best available score for a result."""
        if result.rerank_score is not None:
            return result.rerank_score
        return result.relevance_score

    def _top_score(self, results: list[RetrievalResult]) -> float:
        """Get the top-1 score from results."""
        if not results:
            return 0.0
        return self._get_score(results[0])
