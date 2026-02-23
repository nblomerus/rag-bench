"""
GaRAGe benchmark runner.

Orchestrates running the GaRAGe evaluation:
1. Load dataset entries
2. For each entry: inject passages as context → generate answer → compute metrics
3. Aggregate results into a report
"""

import logging
import time
from dataclasses import dataclass, field

from .loader import GaRAGeEntry, load_garage
from .metrics import compute_attribution_f1, compute_deflection_metrics, compute_raf, compute_uraf

logger = logging.getLogger(__name__)


@dataclass
class GaRAGeSingleResult:
    """Result for a single GaRAGe entry."""

    id: str
    question: str
    gold_answer: str
    generated_answer: str = ""
    should_deflect: bool = False
    did_deflect: bool = False
    question_tag: str = ""
    topic_tag: str = ""

    # Metrics
    raf: dict = field(default_factory=dict)
    uraf: dict = field(default_factory=dict)
    attribution: dict = field(default_factory=dict)

    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class GaRAGeReport:
    """Aggregated GaRAGe evaluation report."""

    summary: dict = field(default_factory=dict)
    deflection: dict = field(default_factory=dict)
    by_category: dict = field(default_factory=dict)
    results: list[GaRAGeSingleResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class GaRAGeRunner:
    """
    Runs the GaRAGe benchmark against a RAG pipeline.

    Args:
        generator: An object with an `answer(question, top_k, context_override)` method
                   that returns a dict with 'answer' and 'deflected' keys.
        judge: Optional JudgeLLM for additional scoring (not required for core metrics).
    """

    def __init__(self, generator, judge=None):
        self.generator = generator
        self.judge = judge

    def run_single(self, entry: GaRAGeEntry) -> GaRAGeSingleResult:
        """Run evaluation on a single GaRAGe entry."""
        result = GaRAGeSingleResult(
            id=entry.id,
            question=entry.question,
            gold_answer=entry.gold_answer,
            should_deflect=entry.should_deflect,
            question_tag=entry.question_tag,
            topic_tag=entry.topic_tag,
        )

        try:
            # Build context from passages
            context_texts = [p.text for p in entry.passages]
            context = "\n\n---\n\n".join(f"[Source {i + 1}]: {text}" for i, text in enumerate(context_texts))

            start = time.time()

            # Generate answer using the RAG pipeline with injected context
            gen_result = self.generator.answer(
                question=entry.question,
                top_k=len(entry.passages),
                context_override=context,
            )

            result.latency_ms = (time.time() - start) * 1000
            result.generated_answer = gen_result.get("answer", "")
            result.did_deflect = gen_result.get("deflected", False)

            # Compute metrics (skip for deflected questions unless they shouldn't deflect)
            passages_dicts = [
                {"text": p.text, "is_relevant": p.is_relevant, "passage_id": p.passage_id} for p in entry.passages
            ]

            if not result.did_deflect:
                result.raf = compute_raf(result.generated_answer, passages_dicts, entry.gold_answer)
                result.uraf = compute_uraf(result.generated_answer, passages_dicts, entry.gold_answer)
                result.attribution = compute_attribution_f1(result.generated_answer, passages_dicts)

        except Exception as e:
            result.error = str(e)
            logger.error("GaRAGe eval error for %s: %s", entry.id, e)

        return result

    def run(
        self,
        sample_size: int = 50,
        seed: int = 42,
        force_download: bool = False,
    ) -> GaRAGeReport:
        """
        Run the full GaRAGe benchmark evaluation.

        Args:
            sample_size: Number of entries to evaluate (0 = all).
            seed: Random seed for sampling.
            force_download: Re-download dataset from HuggingFace.

        Returns:
            GaRAGeReport with metrics, per-entry results, and aggregations.
        """
        start_time = time.time()
        entries = load_garage(sample_size=sample_size, seed=seed, force_download=force_download)

        results = []
        for i, entry in enumerate(entries):
            logger.info("GaRAGe [%d/%d] %s", i + 1, len(entries), entry.question[:60])
            result = self.run_single(entry)
            results.append(result)

        total_time = (time.time() - start_time) * 1000

        # Aggregate
        report = self._aggregate(results)
        report.metadata = {
            "benchmark": "garage",
            "total_evaluated": len(results),
            "sample_size": sample_size,
            "total_time_ms": round(total_time, 1),
            "errors": sum(1 for r in results if r.error),
        }
        return report

    def _aggregate(self, results: list[GaRAGeSingleResult]) -> GaRAGeReport:
        """Aggregate individual results into a report."""
        report = GaRAGeReport(results=results)

        # Filter to non-error, non-deflected results for metric aggregation
        answered = [r for r in results if not r.error and not r.did_deflect]

        # Core metric averages
        if answered:
            report.summary = {
                "raf_score": round(self._mean([r.raf.get("raf_score", 0) for r in answered]), 4),
                "uraf_score": round(self._mean([r.uraf.get("uraf_score", 0) for r in answered]), 4),
                "attribution_f1": round(self._mean([r.attribution.get("attribution_f1", 0) for r in answered]), 4),
                "attribution_precision": round(
                    self._mean([r.attribution.get("attribution_precision", 0) for r in answered]), 4
                ),
                "attribution_recall": round(
                    self._mean([r.attribution.get("attribution_recall", 0) for r in answered]), 4
                ),
                "avg_latency_ms": round(self._mean([r.latency_ms for r in answered]), 1),
                "total_answered": len(answered),
            }
        else:
            report.summary = {
                "raf_score": 0.0,
                "uraf_score": 0.0,
                "attribution_f1": 0.0,
                "attribution_precision": 0.0,
                "attribution_recall": 0.0,
                "avg_latency_ms": 0.0,
                "total_answered": 0,
            }

        # Deflection metrics
        deflection_data = [
            {"should_deflect": r.should_deflect, "did_deflect": r.did_deflect} for r in results if not r.error
        ]
        report.deflection = compute_deflection_metrics(deflection_data)

        # Group by category
        categories = {}
        for r in answered:
            tag = r.question_tag or "unknown"
            if tag not in categories:
                categories[tag] = []
            categories[tag].append(r)

        report.by_category = {}
        for tag, group in categories.items():
            report.by_category[tag] = {
                "count": len(group),
                "raf_score": round(self._mean([r.raf.get("raf_score", 0) for r in group]), 4),
                "attribution_f1": round(self._mean([r.attribution.get("attribution_f1", 0) for r in group]), 4),
            }

        return report

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0
