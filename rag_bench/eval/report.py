"""
Report generation for RAG-Bench evaluation.

Produces terminal summary, Markdown report, and JSON serialization.
"""

import json
import os
import time
from dataclasses import asdict

from rag_bench.eval.runner import EvalReport


def generate_terminal_summary(report: EvalReport) -> str:
    """Produce a colored terminal table with key metrics."""
    s = report.summary
    meta = report.metadata

    w = 53  # inner width

    def row(text: str) -> str:
        return f"\u2551 {text}{' ' * (w - len(text) - 2)} \u2551"

    def sep() -> str:
        return f"\u2560{'═' * w}\u2563"

    lines = [
        f"\u2554{'═' * w}\u2557",
        f"\u2551{'RAG-Bench Evaluation Report':^{w}}\u2551",
        sep(),
        row(f"Total Queries:  {s.get('total_queries', 0)}"),
        row(f"Timestamp:      {meta.get('timestamp', 'N/A')}"),
        sep(),
        row("RETRIEVAL"),
        row(f"  MRR:              {s.get('retrieval_mrr', 0):.4f}"),
        row(f"  Precision@5:      {s.get('retrieval_precision_at_5', 0):.4f}"),
        row(f"  Recall@5:         {s.get('retrieval_recall_at_5', 0):.4f}"),
        row(f"  NDCG@5:           {s.get('retrieval_ndcg_at_5', 0):.4f}"),
        row(f"  Hit Rate:         {s.get('retrieval_hit_rate', 0):.4f}"),
        sep(),
        row("FAITHFULNESS & RELEVANCE"),
        row(f"  Faithfulness:     {s.get('avg_faithfulness', 0):.1f} / 5.0"),
        row(f"  Relevance:        {s.get('avg_relevance', 0):.1f} / 5.0"),
        sep(),
        row("CITATION QUALITY"),
        row(f"  Citation Prec:    {s.get('avg_citation_precision', 0):.4f}"),
        row(f"  Citation Recall:  {s.get('avg_citation_recall', 0):.4f}"),
        row(f"  Avg Density:      {s.get('avg_citation_density', 0):.1f} / sentence"),
        sep(),
        row("DEFLECTION & PERFORMANCE"),
        row(f"  Deflection Acc:   {s.get('deflection_accuracy', 0) * 100:.1f}%"),
        row(f"  Completeness:     {s.get('avg_completeness', 0):.4f}"),
        row(f"  Avg Latency:      {s.get('avg_latency_ms', 0):.0f} ms"),
        f"\u255a{'═' * w}\u255d",
    ]
    return "\n".join(lines)


def generate_markdown_report(report: EvalReport) -> str:
    """Produce a Markdown document with full evaluation results."""
    s = report.summary
    meta = report.metadata
    sections = []

    # Title
    sections.append("# RAG-Bench Evaluation Report\n")
    sections.append(f"**Timestamp:** {meta.get('timestamp', 'N/A')}  ")
    sections.append(f"**Total Queries:** {s.get('total_queries', 0)}  ")
    if meta.get("retrieval_only"):
        sections.append("**Mode:** Retrieval Only  ")
    sections.append("")

    # Summary table
    sections.append("## Summary Metrics\n")
    sections.append("| Metric | Value |")
    sections.append("|--------|-------|")
    sections.append(f"| Retrieval MRR | {s.get('retrieval_mrr', 0):.4f} |")
    sections.append(f"| Retrieval P@5 | {s.get('retrieval_precision_at_5', 0):.4f} |")
    sections.append(f"| Retrieval R@5 | {s.get('retrieval_recall_at_5', 0):.4f} |")
    sections.append(f"| Retrieval NDCG@5 | {s.get('retrieval_ndcg_at_5', 0):.4f} |")
    sections.append(f"| Retrieval Hit Rate | {s.get('retrieval_hit_rate', 0):.4f} |")
    sections.append(f"| Avg Faithfulness | {s.get('avg_faithfulness', 0):.2f} / 5.0 |")
    sections.append(f"| Avg Relevance | {s.get('avg_relevance', 0):.2f} / 5.0 |")
    sections.append(f"| Citation Precision | {s.get('avg_citation_precision', 0):.4f} |")
    sections.append(f"| Citation Recall | {s.get('avg_citation_recall', 0):.4f} |")
    sections.append(f"| Citation Density | {s.get('avg_citation_density', 0):.2f} / sentence |")
    sections.append(f"| Completeness | {s.get('avg_completeness', 0):.4f} |")
    sections.append(f"| Deflection Accuracy | {s.get('deflection_accuracy', 0) * 100:.1f}% |")
    sections.append(f"| Avg Latency | {s.get('avg_latency_ms', 0):.0f} ms |")
    sections.append("")

    # Breakdown by topic
    if report.by_topic:
        sections.append("## Retrieval by Topic\n")
        sections.append("| Topic | MRR | P@5 | R@5 | Hit Rate | Queries |")
        sections.append("|-------|-----|-----|-----|----------|---------|")
        for topic, metrics in sorted(report.by_topic.items()):
            sections.append(
                f"| {topic} | {metrics.get('retrieval_mrr', 0):.3f} "
                f"| {metrics.get('retrieval_precision_at_5', 0):.3f} "
                f"| {metrics.get('retrieval_recall_at_5', 0):.3f} "
                f"| {metrics.get('retrieval_hit_rate', 0):.3f} "
                f"| {metrics.get('total_queries', 0)} |"
            )
        sections.append("")

    # Breakdown by query type
    if report.by_query_type:
        sections.append("## Metrics by Query Type\n")
        sections.append("| Type | MRR | Faithfulness | Citation Prec | Completeness | Queries |")
        sections.append("|------|-----|-------------|---------------|-------------|---------|")
        for qtype, metrics in sorted(report.by_query_type.items()):
            sections.append(
                f"| {qtype} | {metrics.get('retrieval_mrr', 0):.3f} "
                f"| {metrics.get('avg_faithfulness', 0):.1f} "
                f"| {metrics.get('avg_citation_precision', 0):.3f} "
                f"| {metrics.get('avg_completeness', 0):.3f} "
                f"| {metrics.get('total_queries', 0)} |"
            )
        sections.append("")

    # Breakdown by difficulty
    if report.by_difficulty:
        sections.append("## Metrics by Difficulty\n")
        sections.append("| Difficulty | MRR | Faithfulness | Completeness | Avg Latency | Queries |")
        sections.append("|-----------|-----|-------------|-------------|------------|---------|")
        for diff, metrics in sorted(report.by_difficulty.items()):
            sections.append(
                f"| {diff} | {metrics.get('retrieval_mrr', 0):.3f} "
                f"| {metrics.get('avg_faithfulness', 0):.1f} "
                f"| {metrics.get('avg_completeness', 0):.3f} "
                f"| {metrics.get('avg_latency_ms', 0):.0f} ms "
                f"| {metrics.get('total_queries', 0)} |"
            )
        sections.append("")

    # Failed queries
    failed = [
        r
        for r in report.results
        if (
            r.retrieval.get("mrr", 1.0) < 0.5
            or r.faithfulness.get("score", 5.0) < 3
            or r.citation.get("recall", 1.0) < 0.5
            or (r.deflection and not r.deflection.get("correct", True))
            or r.error
        )
    ]
    if failed:
        sections.append("## Failed / Low-Scoring Queries\n")
        for r in failed:
            sections.append(f"### {r.id}\n")
            sections.append(f"**Question:** {r.question}  ")
            if r.error:
                sections.append(f"**Error:** {r.error}  ")
            if r.retrieval:
                sections.append(f"**MRR:** {r.retrieval.get('mrr', 0):.3f}  ")
            if r.faithfulness:
                sections.append(f"**Faithfulness:** {r.faithfulness.get('score', 0):.1f}  ")
            if r.citation:
                sections.append(f"**Citation Recall:** {r.citation.get('recall', 0):.3f}  ")
            if r.deflection and not r.deflection.get("correct", True):
                sections.append(
                    f"**Deflection:** expected={r.deflection.get('expected')}, actual={r.deflection.get('actual')}  "
                )
            sections.append("")

    return "\n".join(sections)


def report_to_json(report: EvalReport) -> dict:
    """Serialize the full EvalReport to a JSON-compatible dict."""
    data = asdict(report)
    # Round floats for readability
    return _round_floats(data)


def _round_floats(obj, precision=4):
    """Recursively round floats in a nested structure."""
    if isinstance(obj, float):
        return round(obj, precision)
    elif isinstance(obj, dict):
        return {k: _round_floats(v, precision) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_round_floats(item, precision) for item in obj]
    return obj


def save_report(
    report: EvalReport,
    output_dir: str = "eval_results",
    run_type: str = "manual",
) -> tuple[str, str]:
    """
    Save both JSON and Markdown reports to output_dir/<run_type>/.
    Returns (json_path, md_path).
    """
    report.metadata["run_type"] = run_type
    target_dir = os.path.join(output_dir, run_type)
    os.makedirs(target_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(target_dir, f"eval_{timestamp}.json")
    md_path = os.path.join(target_dir, f"eval_{timestamp}.md")

    # JSON
    json_data = report_to_json(report)
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)

    # Markdown
    md_content = generate_markdown_report(report)
    with open(md_path, "w") as f:
        f.write(md_content)

    return json_path, md_path
