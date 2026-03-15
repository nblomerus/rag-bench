"""
CLI entry point for RAG-Bench evaluation.

Usage:
    python -m rag_bench.eval.run                          # Full evaluation
    python -m rag_bench.eval.run --retrieval-only          # Fast: retrieval only
    python -m rag_bench.eval.run --topic transformers      # Filter by topic
    python -m rag_bench.eval.run --type comparison         # Filter by query type
    python -m rag_bench.eval.run --output-dir ./results    # Custom output dir
    python -m rag_bench.eval.run --no-judge                # Skip LLM judge
"""

import argparse
import logging
import os
import time

from rag_bench.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
)
from rag_bench.core.citation_boost import CitationBooster
from rag_bench.core.generator import RAGGenerator, RelevanceGate, build_llm_backend
from rag_bench.core.retriever import HybridRetriever
from rag_bench.eval.judge import JudgeLLM
from rag_bench.eval.report import generate_terminal_summary, save_report
from rag_bench.eval.runner import EvalRunner

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG-Bench Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only run retrieval metrics, skip generation and LLM judge",
    )
    parser.add_argument("--topic", type=str, default=None, help="Filter by topic")
    parser.add_argument("--type", type=str, default=None, help="Filter by query type")
    parser.add_argument("--difficulty", type=str, default=None, help="Filter by difficulty")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Directory to save reports (default: eval_results)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM-as-judge scoring (faster, no faithfulness/relevance scores)",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()

    logger.info("Initializing RAG pipeline for evaluation...")
    start_init = time.time()

    # Initialize retriever
    reranker_model = os.environ.get("RAG_RERANKER_MODEL", RERANKER_MODEL)
    retriever = HybridRetriever(
        embedding_model=EMBEDDING_MODEL,
        reranker_model=reranker_model,
        chroma_path=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    # Initialize LLM backend
    llm_backend = os.environ.get("RAG_LLM_BACKEND", "ollama")
    llm_model = os.environ.get("RAG_LLM_MODEL", "qwen2.5:14b")
    llm_base_url = os.environ.get("RAG_LLM_BASE_URL", "http://localhost:11434")
    llm = build_llm_backend(llm_backend, llm_model, llm_base_url)

    # Initialize citation booster
    citation_booster = CitationBooster(
        enable_age_boost=True,
        enable_query_adaptive=True,
    )

    # Initialize generator
    gate = RelevanceGate()
    generator = RAGGenerator(
        retriever=retriever,
        llm_backend=llm,
        relevance_gate=gate,
        top_k=DEFAULT_TOP_K,
        citation_booster=citation_booster,
    )

    # Initialize judge
    judge = None
    if not args.no_judge and not args.retrieval_only:
        judge = JudgeLLM(llm)
        logger.info("LLM judge enabled")
    else:
        logger.info("LLM judge disabled")

    init_time = time.time() - start_init
    logger.info(f"Pipeline initialized in {init_time:.1f}s")

    # Create runner and run evaluation
    runner = EvalRunner(
        retriever=retriever,
        generator=generator,
        judge=judge,
    )

    logger.info("Starting evaluation...")
    report = runner.run_all(
        filter_topic=args.topic,
        filter_type=args.type,
        filter_difficulty=args.difficulty,
        retrieval_only=args.retrieval_only,
    )

    # Print terminal summary
    print("\n" + generate_terminal_summary(report))

    # Save reports
    json_path, md_path = save_report(report, args.output_dir, run_type="manual")
    print("\nReports saved:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")


if __name__ == "__main__":
    main()
