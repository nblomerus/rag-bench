#!/usr/bin/env python3
"""
query.py — Interactive query interface for RAG-Bench.

Supports:
- Interactive REPL mode (default)
- Single query mode (--query "...")
- JSON output for programmatic use (--json)
- Backend selection (--backend ollama|openai|template)
- Config file for full pipeline configuration (--config)

Usage:
    python query.py                                    # Interactive mode
    python query.py --query "How does attention work?"  # Single query
    python query.py --backend ollama                    # Use Ollama LLM
    python query.py --json --query "..."               # JSON output
    python query.py --eval                             # Run evaluation suite
    python query.py --eval --backend ollama            # Eval with Ollama
    python query.py --config my_config.json            # Use pipeline config file
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_bench.config import (  # noqa: E402
    DEFAULT_TOP_K,
    EVAL_DIR,
    RERANKER_MODEL,
)
from rag_bench.core.configs import (  # noqa: E402
    GeneratorConfig,
    PipelineConfig,
    RetrieverConfig,
)
from rag_bench.core.pipeline import RAGPipeline, build_pipeline  # noqa: E402

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Suppress verbose HTTP logs from httpx and httpcore
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def interactive_mode(pipeline: RAGPipeline, json_output: bool = False):
    """Run the interactive query REPL."""
    print("\n" + "=" * 60)
    print("  RAG-Bench: AI/ML Research Paper Assistant")
    print("=" * 60)
    print("  Ask questions about AI/ML research papers.")
    print("  Type 'quit' or 'exit' to stop.")
    print("  Type 'eval' to run the evaluation set.")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("  Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if question.lower() == "eval":
            run_eval_subset(pipeline)
            continue

        if json_output:
            result = pipeline.query(question)
            print(json.dumps(asdict(result), indent=2, default=str))
        else:
            # Use the old print_answer for rich terminal output
            pipeline.generator.print_answer(question)


def run_eval_subset(pipeline: RAGPipeline, run_all: bool = False):
    """Run evaluation on sample queries.

    Args:
        pipeline: The assembled RAG pipeline
        run_all: If True, run ALL queries. If False, only easy + deflection + adversarial.
    """
    eval_path = EVAL_DIR / "eval_queries.json"
    if not eval_path.exists():
        logger.error("Eval queries not found at eval/eval_queries.json")
        return

    with open(eval_path) as f:
        queries = json.load(f)

    if run_all:
        test_queries = queries
        logger.info(f"Running ALL {len(test_queries)} evaluation queries...")
    else:
        test_queries = [q for q in queries if q.get("difficulty") in ("easy", "deflection", "adversarial")]
        logger.info(f"Running {len(test_queries)} evaluation queries (easy + deflection + adversarial)...")

    correct = 0
    total = len(test_queries)
    results_by_difficulty = {}
    top_k = pipeline.config.generator.top_k

    for q in test_queries:
        retrieval_results = pipeline.retriever.retrieve(q["question"], top_k=top_k)
        gen_result = pipeline.generator.generate(q["question"], context=retrieval_results)
        should_deflect = q.get("should_deflect", False)

        # Two-layer deflection detection:
        # 1. Retrieval-level gate (gen_result.deflected)
        # 2. LLM-level deflection (model says "I don't have information")
        did_deflect = gen_result.deflected
        llm_deflected = False

        if not did_deflect and should_deflect:
            # Check if the LLM itself refused to answer
            answer_lower = gen_result.answer.lower()
            deflection_phrases = [
                "i don't have information",
                "i don't have sufficient information",
                "not covered in",
                "not appear in the",
                "no information about",
                "not mentioned in",
                "don't have enough information",
                "sources do not contain",
                "sources don't contain",
                "premise appears incorrect",
                "premise may be incorrect",
                "cannot find information",
                "not in my knowledge base",
                "not specified in",
                "not detailed in",
                "not disclosed",
                "does not specify",
                "does not mention",
                "does not provide",
                "do not contain",
                "do not specify",
                "do not provide",
                "no specific information",
                "insufficient information",
                "not explicitly stated",
                "not explicitly mentioned",
            ]
            llm_deflected = any(phrase in answer_lower for phrase in deflection_phrases)
            if llm_deflected:
                did_deflect = True

        is_correct = did_deflect == should_deflect
        if is_correct:
            correct += 1

        # Track per-difficulty stats
        diff = q.get("difficulty", "unknown")
        if diff not in results_by_difficulty:
            results_by_difficulty[diff] = {"correct": 0, "total": 0}
        results_by_difficulty[diff]["total"] += 1
        if is_correct:
            results_by_difficulty[diff]["correct"] += 1

        status = "PASS" if is_correct else "FAIL"
        scores = gen_result.scores
        scores_str = f"top={scores[0]:.3f}" if scores else "no scores"
        deflect_source = ""
        if did_deflect:
            deflect_source = " (gate)" if gen_result.deflected else " (llm)"

        logger.info(
            f"[{status}] {q['id']}: {scores_str} | deflect={did_deflect}{deflect_source} (expected={should_deflect})"
        )
        logger.info(f"  {q['question'][:65]}")

        # For FAIL cases, print debug info to help diagnose
        if not is_correct:
            # Show source paper titles
            for i, r in enumerate(retrieval_results[:3], 1):
                title = r.chunk.metadata.get("title", r.chunk.metadata.get("source_display", "?"))[:60]
                logger.debug(f"  Source {i}: {title} (score={r.relevance_score:.2f})")
            # Show first 200 chars of LLM answer
            answer_preview = gen_result.answer[:200].replace("\n", " ")
            logger.debug(f"  Answer: {answer_preview}...")

    # Summary
    logger.info("=" * 50)
    logger.info(f"Overall: {correct}/{total} ({correct / total * 100:.0f}%)")
    for diff, stats in sorted(results_by_difficulty.items()):
        pct = stats["correct"] / stats["total"] * 100 if stats["total"] else 0
        logger.info(f"  {diff:>15}: {stats['correct']}/{stats['total']} ({pct:.0f}%)")


def _config_from_args(args) -> PipelineConfig:
    """Build a PipelineConfig from CLI arguments."""
    if args.config:
        return PipelineConfig.load(args.config)

    return PipelineConfig(
        name="cli",
        retriever=RetrieverConfig(
            reranker_model=args.reranker,
        ),
        generator=GeneratorConfig(
            llm_backend=args.backend,
            llm_model=args.model or "gemma2:27b",
            llm_base_url=args.base_url or "",
            top_k=args.top_k,
            relevance_threshold=args.threshold,
            enable_citation_boost=not args.no_citation_boost,
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description="RAG-Bench Query Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Single query mode (non-interactive)",
    )
    parser.add_argument(
        "--backend",
        "-b",
        choices=["template", "ollama", "openai"],
        default="template",
        help="LLM backend for answer generation (default: template)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="",
        help="Model name for the LLM backend",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="",
        help="Base URL for the LLM backend API",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of sources to retrieve (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--reranker",
        type=str,
        default=RERANKER_MODEL,
        help=f"Cross-encoder model for reranking (default: {RERANKER_MODEL})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Minimum relevance score before deflecting (default: 0.3)",
    )
    parser.add_argument(
        "--eval",
        "-e",
        action="store_true",
        help="Run the evaluation suite (easy + deflection + adversarial queries)",
    )
    parser.add_argument(
        "--eval-all",
        action="store_true",
        help="Run ALL evaluation queries (including medium and hard)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--no-citation-boost",
        action="store_true",
        help="Disable citation quality boosting for foundational papers",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="",
        help="Path to a PipelineConfig JSON file",
    )
    args = parser.parse_args()
    setup_logging(args.verbose)

    # Build pipeline from config (CLI args or config file)
    config = _config_from_args(args)
    logger.info(f"Building pipeline '{config.name}'...")
    pipeline = build_pipeline(config)
    logger.info("Pipeline ready")

    # Eval mode, single query, or interactive
    if args.eval or args.eval_all:
        run_eval_subset(pipeline, run_all=args.eval_all)
    elif args.query:
        if args.json:
            result = pipeline.query(args.query, top_k=args.top_k)
            print(json.dumps(asdict(result), indent=2, default=str))
        else:
            pipeline.generator.print_answer(args.query)
    else:
        interactive_mode(pipeline, json_output=args.json)


if __name__ == "__main__":
    main()
