"""
RAGTruth benchmark runner.

Orchestrates running the RAGTruth evaluation:
1. Load dataset entries (QA task type)
2. For each entry: inject context → generate answer → detect hallucinations → compare
3. Aggregate results into a report
"""

import logging
import re
import time
from dataclasses import dataclass, field

from .loader import RAGTruthEntry, load_ragtruth
from .metrics import (
    case_level_accuracy,
    compute_hallucination_rate,
    hallucination_by_type,
    span_level_f1,
)

logger = logging.getLogger(__name__)


@dataclass
class RAGTruthSingleResult:
    """Result for a single RAGTruth entry."""

    id: str
    source_id: str
    task_type: str
    prompt: str

    # Ground truth
    has_hallucination_gold: bool = False
    gold_spans: list[dict] = field(default_factory=list)  # [{text, label_type}]
    gold_span_types: list[str] = field(default_factory=list)

    # Predictions
    generated_answer: str = ""
    has_hallucination_predicted: bool = False
    predicted_spans: list[str] = field(default_factory=list)
    predicted_span_types: list[str] = field(default_factory=list)

    # Metrics
    span_metrics: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class RAGTruthReport:
    """Aggregated RAGTruth evaluation report."""

    summary: dict = field(default_factory=dict)
    hallucination_rate: dict = field(default_factory=dict)
    case_level: dict = field(default_factory=dict)
    by_type: dict = field(default_factory=dict)
    results: list[RAGTruthSingleResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class RAGTruthRunner:
    """
    Runs the RAGTruth benchmark against a RAG pipeline.

    Args:
        generator: An object with an `answer(question, top_k, context_override)` method.
        judge: Optional JudgeLLM for hallucination detection. If provided, uses the
               judge to detect hallucinations in generated responses. Otherwise,
               uses heuristic detection.
    """

    def __init__(self, generator, judge=None):
        self.generator = generator
        self.judge = judge

    def run_single(self, entry: RAGTruthEntry) -> RAGTruthSingleResult:
        """Run evaluation on a single RAGTruth entry."""
        result = RAGTruthSingleResult(
            id=entry.id,
            source_id=entry.source_id,
            task_type=entry.task_type,
            prompt=entry.prompt,
            has_hallucination_gold=entry.has_hallucination,
            gold_spans=[{"text": s.text, "label_type": s.label_type} for s in entry.hallucination_spans],
            gold_span_types=[s.label_type for s in entry.hallucination_spans if s.label_type],
        )

        try:
            start = time.time()

            # Generate answer using the RAG pipeline with injected context
            gen_result = self.generator.answer(
                question=entry.prompt,
                top_k=5,
                context_override=entry.source_info,
            )

            result.latency_ms = (time.time() - start) * 1000
            result.generated_answer = gen_result.get("answer", "")

            # Detect hallucinations in the generated response
            if self.judge:
                detection = self._detect_with_judge(result.generated_answer, entry.source_info)
            else:
                detection = self._detect_heuristic(result.generated_answer, entry.source_info)

            result.has_hallucination_predicted = detection.get("has_hallucination", False)
            result.predicted_spans = detection.get("spans", [])
            result.predicted_span_types = detection.get("span_types", [])

            # Compute span-level metrics
            gold_span_texts = [s.text for s in entry.hallucination_spans]
            result.span_metrics = span_level_f1(
                predicted_spans=result.predicted_spans,
                gold_spans=gold_span_texts,
                full_text=result.generated_answer,
            )

        except Exception as e:
            result.error = str(e)
            logger.error("RAGTruth eval error for %s: %s", entry.id, e)

        return result

    def _detect_with_judge(self, answer: str, context: str) -> dict:
        """Use JudgeLLM to detect hallucinations."""
        try:
            # Score faithfulness — low scores indicate hallucination
            faith_result = self.judge.score_faithfulness(
                question="",  # RAGTruth evaluates faithfulness to context
                answer=answer,
                source_passages=[context],
            )
            score = faith_result.get("score", 3.0)

            # Score < 3 suggests hallucination
            has_hallucination = score < 3.0

            # Try to extract specific spans from reasoning
            reasoning = faith_result.get("reasoning", "")
            spans = self._extract_spans_from_reasoning(reasoning, answer)

            return {
                "has_hallucination": has_hallucination,
                "spans": spans,
                "span_types": ["Evident Conflict"] * len(spans) if spans else [],
                "faithfulness_score": score,
            }
        except Exception as e:
            logger.warning("Judge detection failed, falling back to heuristic: %s", e)
            return self._detect_heuristic(answer, context)

    def _detect_heuristic(self, answer: str, context: str) -> dict:
        """
        Heuristic hallucination detection using content-word overlap.

        Sentences in the answer that have very low overlap with the context
        are flagged as potential hallucinations.
        """
        if not answer or not context:
            return {"has_hallucination": False, "spans": [], "span_types": []}

        context_words = set(re.findall(r"\b\w+\b", context.lower()))
        # Remove common stop words for better signal
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "shall",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "that",
            "this",
            "these",
            "those",
            "it",
            "its",
            "and",
            "but",
            "or",
            "nor",
            "not",
            "so",
            "yet",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "only",
            "own",
            "same",
            "than",
            "too",
            "very",
        }
        context_content = context_words - stop_words

        # Split answer into sentences
        sentences = re.split(r"(?<=[.!?])\s+", answer)
        hallucinated_spans = []

        for sent in sentences:
            sent_words = set(re.findall(r"\b\w+\b", sent.lower())) - stop_words
            if len(sent_words) < 3:
                continue  # Skip very short sentences

            overlap = sent_words & context_content
            overlap_ratio = len(overlap) / len(sent_words) if sent_words else 1.0

            # If less than 20% of content words appear in context, flag as suspicious
            if overlap_ratio < 0.20:
                hallucinated_spans.append(sent.strip())

        return {
            "has_hallucination": len(hallucinated_spans) > 0,
            "spans": hallucinated_spans,
            "span_types": ["Evident Baseless"] * len(hallucinated_spans),
        }

    @staticmethod
    def _extract_spans_from_reasoning(reasoning: str, answer: str) -> list[str]:
        """Try to extract specific hallucinated text from judge reasoning."""
        spans = []

        # Look for quoted text in reasoning that appears in the answer
        quoted = re.findall(r'"([^"]{10,})"', reasoning)
        for q in quoted:
            if q.lower() in answer.lower():
                spans.append(q)

        return spans

    def run(
        self,
        sample_size: int = 50,
        task_type: str = "QA",
        seed: int = 42,
        force_download: bool = False,
    ) -> RAGTruthReport:
        """
        Run the full RAGTruth benchmark evaluation.

        Args:
            sample_size: Number of entries to evaluate (0 = all).
            task_type: Filter by task type (default "QA").
            seed: Random seed for sampling.
            force_download: Re-download dataset.

        Returns:
            RAGTruthReport with metrics, per-entry results, and breakdowns.
        """
        start_time = time.time()
        entries = load_ragtruth(
            sample_size=sample_size,
            task_type=task_type,
            seed=seed,
            force_download=force_download,
        )

        results = []
        for i, entry in enumerate(entries):
            logger.info("RAGTruth [%d/%d] %s", i + 1, len(entries), entry.prompt[:60])
            result = self.run_single(entry)
            results.append(result)

        total_time = (time.time() - start_time) * 1000

        # Aggregate
        report = self._aggregate(results)
        report.metadata = {
            "benchmark": "ragtruth",
            "task_type": task_type,
            "total_evaluated": len(results),
            "sample_size": sample_size,
            "total_time_ms": round(total_time, 1),
            "errors": sum(1 for r in results if r.error),
        }
        return report

    def _aggregate(self, results: list[RAGTruthSingleResult]) -> RAGTruthReport:
        """Aggregate individual results into a report."""
        report = RAGTruthReport(results=results)

        valid = [r for r in results if not r.error]
        if not valid:
            return report

        # Hallucination rate
        rate_data = [
            {
                "has_hallucination_predicted": r.has_hallucination_predicted,
                "has_hallucination_gold": r.has_hallucination_gold,
            }
            for r in valid
        ]
        report.hallucination_rate = compute_hallucination_rate(rate_data)

        # Case-level accuracy
        report.case_level = case_level_accuracy(rate_data)

        # Span-level F1 (average across entries)
        span_f1s = [r.span_metrics.get("span_f1", 0) for r in valid if r.span_metrics]
        avg_span_f1 = sum(span_f1s) / len(span_f1s) if span_f1s else 0.0

        # Hallucination by type
        type_data = [
            {"gold_span_types": r.gold_span_types, "predicted_span_types": r.predicted_span_types} for r in valid
        ]
        report.by_type = hallucination_by_type(type_data)

        # Summary
        report.summary = {
            "hallucination_rate": report.hallucination_rate.get("predicted_rate", 0.0),
            "gold_hallucination_rate": report.hallucination_rate.get("gold_rate", 0.0),
            "case_level_accuracy": report.case_level.get("accuracy", 0.0),
            "case_level_f1": report.case_level.get("f1", 0.0),
            "avg_span_f1": round(avg_span_f1, 4),
            "avg_latency_ms": round(sum(r.latency_ms for r in valid) / len(valid), 1) if valid else 0.0,
            "total_evaluated": len(valid),
        }

        return report
