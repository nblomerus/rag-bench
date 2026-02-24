"""
Evaluation orchestrator for RAG-Bench.

Coordinates running benchmark queries through the pipeline, collecting results,
computing all metrics, and producing the final report data structure.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

from rag_bench.eval.benchmark import BenchmarkEntry, get_benchmark
from rag_bench.eval.judge import JudgeLLM
from rag_bench.eval.metrics import (
    compute_citation_metrics,
    compute_completeness,
    compute_retrieval_metrics,
)

logger = logging.getLogger(__name__)


def _clear_cuda_cache() -> None:
    """Free fragmented CUDA memory if torch is available."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


@dataclass
class SingleEvalResult:
    """Result for a single benchmark query."""

    id: str
    question: str
    query_type: str
    topic: str
    difficulty: str
    retrieval: dict = field(default_factory=dict)
    citation: dict = field(default_factory=dict)
    completeness: dict = field(default_factory=dict)
    faithfulness: dict = field(default_factory=dict)
    relevance: dict = field(default_factory=dict)
    deflection: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    answer_preview: str = ""
    error: str | None = None


@dataclass
class EvalReport:
    """Aggregated evaluation report."""

    summary: dict = field(default_factory=dict)
    by_topic: dict = field(default_factory=dict)
    by_query_type: dict = field(default_factory=dict)
    by_difficulty: dict = field(default_factory=dict)
    results: list[SingleEvalResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class EvalRunner:
    def __init__(
        self,
        retriever,
        generator,
        judge: JudgeLLM | None = None,
        benchmark: list[BenchmarkEntry] | None = None,
    ):
        self.retriever = retriever
        self.generator = generator
        self.judge = judge
        self.benchmark = benchmark or get_benchmark()

    def run_single(self, entry: BenchmarkEntry) -> SingleEvalResult:
        """Run one benchmark entry through the full pipeline."""
        result = SingleEvalResult(
            id=entry.id,
            question=entry.question,
            query_type=entry.query_type,
            topic=entry.topic,
            difficulty=entry.difficulty,
        )

        try:
            start = time.time()
            try:
                gen_result = self.generator.answer(entry.question, top_k=5)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"CUDA OOM for {entry.id}, clearing cache and retrying")
                    _clear_cuda_cache()
                    gen_result = self.generator.answer(entry.question, top_k=5)
                else:
                    raise
            result.latency_ms = (time.time() - start) * 1000

            answer = gen_result.get("answer", "")
            all_results = gen_result.get("results", [])
            filtered_results = gen_result.get("filtered_results", [])
            deflected = gen_result.get("deflected", False)

            result.answer_preview = answer[:200]

            # Deflection check
            result.deflection = {
                "expected": entry.should_deflect,
                "actual": deflected,
                "correct": entry.should_deflect == deflected,
            }

            # Retrieval metrics (always compute, even for deflections)
            if entry.expected_sources:
                result.retrieval = compute_retrieval_metrics(
                    all_results,
                    entry.expected_sources,
                    entry.acceptable_sources or None,
                    k=5,
                )

            # Skip generation metrics for deflections
            if deflected:
                return result

            # Citation metrics
            sources_for_citation = filtered_results if filtered_results else all_results[:5]
            result.citation = compute_citation_metrics(
                answer,
                sources_for_citation,
                entry.expected_sources,
                entry.acceptable_sources or None,
                entry.expected_answer_excludes or None,
            )

            # Completeness
            result.completeness = compute_completeness(answer, entry.expected_answer_contains)

            # Judge (if available) — each call isolated so one failure doesn't skip the other
            if self.judge and sources_for_citation:
                source_texts = [r.get("text", r.get("text_preview", "")) for r in sources_for_citation]
                try:
                    result.faithfulness = self.judge.score_faithfulness(entry.question, answer, source_texts)
                except Exception as e:
                    logger.warning(f"Judge faithfulness failed for {entry.id}: {e}")
                try:
                    result.relevance = self.judge.score_relevance(entry.question, answer)
                except Exception as e:
                    logger.warning(f"Judge relevance failed for {entry.id}: {e}")

        except Exception as e:
            logger.error(f"Error evaluating {entry.id}: {e}")
            result.error = str(e)

        return result

    def run_all(
        self,
        filter_topic: str | None = None,
        filter_type: str | None = None,
        filter_difficulty: str | None = None,
        retrieval_only: bool = False,
    ) -> EvalReport:
        """Run all (or filtered) benchmark entries and aggregate."""
        entries = self.benchmark

        if filter_topic:
            entries = [e for e in entries if e.topic == filter_topic]
        if filter_type:
            entries = [e for e in entries if e.query_type == filter_type]
        if filter_difficulty:
            entries = [e for e in entries if e.difficulty == filter_difficulty]

        results = []
        total = len(entries)
        for i, entry in enumerate(entries):
            logger.info(f"[{i + 1}/{total}] Evaluating: {entry.id}")

            result = self._run_retrieval_only(entry) if retrieval_only else self.run_single(entry)
            results.append(result)

            # Free fragmented CUDA memory every 10 entries to prevent OOM
            if (i + 1) % 10 == 0:
                _clear_cuda_cache()

        report = EvalReport(
            results=results,
            summary=self._aggregate_results(results),
            by_topic=self._group_by(results, "topic"),
            by_query_type=self._group_by(results, "query_type"),
            by_difficulty=self._group_by(results, "difficulty"),
            metadata={
                "total_queries": len(results),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "retrieval_only": retrieval_only,
                "filters": {
                    "topic": filter_topic,
                    "type": filter_type,
                    "difficulty": filter_difficulty,
                },
            },
        )
        return report

    def _run_retrieval_only(self, entry: BenchmarkEntry) -> SingleEvalResult:
        """Run retrieval only — skip generation + judge."""
        result = SingleEvalResult(
            id=entry.id,
            question=entry.question,
            query_type=entry.query_type,
            topic=entry.topic,
            difficulty=entry.difficulty,
        )

        try:
            start = time.time()
            retrieval_results = self.retriever.query(entry.question, top_k=5)
            result.latency_ms = (time.time() - start) * 1000

            if entry.expected_sources:
                result.retrieval = compute_retrieval_metrics(
                    retrieval_results,
                    entry.expected_sources,
                    entry.acceptable_sources or None,
                    k=5,
                )
        except Exception as e:
            logger.error(f"Error in retrieval-only for {entry.id}: {e}")
            result.error = str(e)

        return result

    def _aggregate_results(self, results: list[SingleEvalResult]) -> dict:
        """Compute summary statistics across all results."""
        if not results:
            return {}

        non_deflection = [r for r in results if not r.deflection.get("expected", False)]

        def _safe_mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        # Retrieval metrics (non-deflection only)
        retrieval_mrrs = [r.retrieval.get("mrr", 0.0) for r in non_deflection if r.retrieval]
        retrieval_p5 = [r.retrieval.get("precision_at_k", 0.0) for r in non_deflection if r.retrieval]
        retrieval_r5 = [r.retrieval.get("recall_at_k", 0.0) for r in non_deflection if r.retrieval]
        retrieval_ndcg = [r.retrieval.get("ndcg_at_k", 0.0) for r in non_deflection if r.retrieval]
        retrieval_hit = [r.retrieval.get("hit_rate", 0.0) for r in non_deflection if r.retrieval]

        # Faithfulness & relevance (non-deflection with judge scores)
        faith_scores = [r.faithfulness.get("score", 0.0) for r in non_deflection if r.faithfulness]
        rel_scores = [r.relevance.get("score", 0.0) for r in non_deflection if r.relevance]

        # Citation metrics (non-deflection only)
        cit_prec = [r.citation.get("precision", 0.0) for r in non_deflection if r.citation]
        cit_rec = [r.citation.get("recall", 0.0) for r in non_deflection if r.citation]
        cit_dens = [r.citation.get("density", 0.0) for r in non_deflection if r.citation]

        # Completeness
        comp_scores = [r.completeness.get("score", 0.0) for r in non_deflection if r.completeness]

        # Deflection accuracy
        deflection_correct = sum(1 for r in results if r.deflection.get("correct", False))
        deflection_total = sum(1 for r in results if r.deflection)

        # Latency
        latencies = [r.latency_ms for r in results if r.latency_ms > 0]

        return {
            "total_queries": len(results),
            "retrieval_mrr": round(_safe_mean(retrieval_mrrs), 4),
            "retrieval_precision_at_5": round(_safe_mean(retrieval_p5), 4),
            "retrieval_recall_at_5": round(_safe_mean(retrieval_r5), 4),
            "retrieval_ndcg_at_5": round(_safe_mean(retrieval_ndcg), 4),
            "retrieval_hit_rate": round(_safe_mean(retrieval_hit), 4),
            "avg_faithfulness": round(_safe_mean(faith_scores), 2),
            "avg_relevance": round(_safe_mean(rel_scores), 2),
            "avg_citation_precision": round(_safe_mean(cit_prec), 4),
            "avg_citation_recall": round(_safe_mean(cit_rec), 4),
            "avg_citation_density": round(_safe_mean(cit_dens), 2),
            "avg_completeness": round(_safe_mean(comp_scores), 4),
            "deflection_accuracy": round(deflection_correct / deflection_total if deflection_total else 0.0, 4),
            "avg_latency_ms": round(_safe_mean(latencies), 1),
        }

    def _group_by(
        self,
        results: list[SingleEvalResult],
        key: str,
    ) -> dict[str, dict]:
        """Group results by a field and aggregate each group."""
        groups: dict[str, list[SingleEvalResult]] = defaultdict(list)
        for r in results:
            value = getattr(r, key, "unknown")
            groups[value].append(r)

        return {k: self._aggregate_results(v) for k, v in groups.items()}
