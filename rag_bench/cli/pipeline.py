"""
pipeline.py — RAG-Bench Data Ingestion & Indexing Pipeline.

Runs the complete pipeline:
1. Load AI/ML papers from scraped PDFs and/or HuggingFace
2. Parse papers into structured documents
3. Chunk with equation/table/acronym awareness
4. Embed with BGE and index in ChromaDB
5. Run test queries to verify the index

Usage:
    rag-pipeline                          # Run full pipeline (scraped papers only)
    rag-pipeline --hybrid                 # Run with hybrid ingestion (scraped + HF dataset)
    rag-pipeline --enrich                 # Run with contextual enrichment (requires Ollama)
    rag-pipeline --step ingest --hybrid   # Run only hybrid ingestion
    rag-pipeline --step chunk             # Run only chunking (requires ingest first)
    rag-pipeline --step enrich --enrich   # Run only enrichment (requires chunk first + Ollama)
    rag-pipeline --step index             # Run only indexing (requires chunk first)
    rag-pipeline --step test              # Run only test queries (requires index first)
"""

import argparse
import collections
import json
import logging
import sys
import time

from rag_bench.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP_CHARS,
    CHUNK_SIZE_CHARS,
    COLLECTION_NAME,
    DATA_DIR,
    DATASET_NAME,
    DISTANCE_METRIC,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    EVAL_DIR,
    LOG_DIR,
    MIN_SECTION_LENGTH,
)
from rag_bench.core.chunker import chunk_all_papers
from rag_bench.core.configs import EmbedderConfig, EnricherConfig, RetrieverConfig
from rag_bench.core.embedder import Embedder
from rag_bench.core.enricher import ContextualEnricher
from rag_bench.core.hybrid_ingest import hybrid_ingest
from rag_bench.core.retriever import HybridRetriever
from rag_bench.core.types import ChunkData


def setup_logging():
    """Configure logging for the pipeline."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "pipeline.log"),
        ],
    )


def step_ingest(use_hybrid: bool = False) -> list[dict]:
    """
    Step 1: Load and parse papers.

    Args:
        use_hybrid: If True, merge scraped papers with HuggingFace dataset
    """
    logger = logging.getLogger("pipeline.ingest")
    logger.info("=" * 60)
    logger.info("STEP 1: Ingesting papers")
    logger.info("=" * 60)

    start = time.time()

    scraped_path = DATA_DIR / "scraped_papers.json"

    if use_hybrid:
        # Hybrid mode: merge scraped + HuggingFace dataset
        logger.info("🔄 Using HYBRID ingestion mode")
        logger.info("   Combining scraped papers + HuggingFace ai-arxiv2 dataset")

        if not scraped_path.exists():
            logger.warning(f"Scraped papers not found: {scraped_path}")
            logger.warning("Proceeding with HuggingFace dataset only")

        docs = hybrid_ingest(
            scraped_path=scraped_path,
            dataset_name=DATASET_NAME,
            split="train",
            save_path=DATA_DIR / "parsed_papers.json",
            prefer_scraped=True,
        )
    else:
        # Original mode: scraped papers only
        logger.info("📄 Using scraped papers only")

        if not scraped_path.exists():
            logger.error(f"Scraped papers not found: {scraped_path}")
            logger.error("Please run: python scripts/scrape_arxiv.py --mode extended")
            logger.error("Or use --hybrid flag to include HuggingFace dataset")
            sys.exit(1)

        logger.info(f"Loading scraped papers from {scraped_path}")
        with open(scraped_path) as f:
            docs = json.load(f)

        # Save as parsed_papers.json for consistency
        with open(DATA_DIR / "parsed_papers.json", "w") as f:
            json.dump(docs, f, indent=2, default=str)

        logger.info(f"Loaded {len(docs)} papers from local scrape")

    elapsed = time.time() - start

    logger.info(f"Ingestion complete: {len(docs)} papers in {elapsed:.1f}s")

    years = [d["year"] for d in docs if d["year"]]
    if years:
        logger.info(f"Year range: {min(years)} - {max(years)}")

    section_counts = {}
    for doc in docs:
        for section in doc["sections"]:
            section_counts[section] = section_counts.get(section, 0) + 1

    logger.info(f"Most common sections: {sorted(section_counts.items(), key=lambda x: -x[1])[:10]}")

    return docs


def step_chunk(docs: list[dict]) -> list[dict]:
    """Step 2: Chunk all papers."""
    logger = logging.getLogger("pipeline.chunk")
    logger.info("=" * 60)
    logger.info("STEP 2: Chunking papers")
    logger.info("=" * 60)

    start = time.time()
    chunks = chunk_all_papers(
        docs,
        chunk_size=CHUNK_SIZE_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
        min_section_length=MIN_SECTION_LENGTH,
    )
    elapsed = time.time() - start

    logger.info(f"Chunking complete: {len(chunks)} chunks in {elapsed:.1f}s")

    # Save chunks for debugging/caching
    chunks_path = DATA_DIR / "chunks.json"
    with open(chunks_path, "w") as f:
        json.dump(chunks, f, indent=2, default=str)
    logger.info(f"Saved chunks to {chunks_path}")

    chunk_lengths = [len(c["text"]) for c in chunks]
    logger.info(
        f"Chunk length stats: "
        f"min={min(chunk_lengths)}, "
        f"max={max(chunk_lengths)}, "
        f"avg={sum(chunk_lengths) / len(chunk_lengths):.0f}, "
        f"median={sorted(chunk_lengths)[len(chunk_lengths) // 2]}"
    )

    return chunks


def step_enrich(chunks: list[dict], docs: list[dict], enricher_config: EnricherConfig | None = None) -> list[dict]:
    """Step 2.5: Add contextual headers to chunks using local LLM.

    This step calls Ollama to generate a 2-3 sentence context summary
    for each chunk, situating it within its source paper.  Results are
    cached on disk so re-runs skip already-enriched chunks.
    """
    logger = logging.getLogger("pipeline.enrich")
    logger.info("=" * 60)
    logger.info("STEP 2.5: Contextual enrichment")
    logger.info("=" * 60)

    config = enricher_config or EnricherConfig(enabled=True)
    enricher = ContextualEnricher(config=config)

    # Build a lookup from doc_id -> paper dict
    docs_by_id: dict[str, dict] = {}
    for doc in docs:
        docs_by_id[doc["doc_id"]] = doc

    # Group chunks by doc_id so we can enrich per-paper
    chunks_by_doc: dict[str, list[dict]] = collections.defaultdict(list)
    for chunk in chunks:
        chunks_by_doc[chunk["doc_id"]].append(chunk)

    start = time.time()
    enriched_chunks: list[dict] = []
    total_papers = len(chunks_by_doc)

    for i, (doc_id, doc_chunks) in enumerate(chunks_by_doc.items()):
        paper = docs_by_id.get(doc_id)
        if not paper:
            logger.warning(f"Paper {doc_id} not found in docs, skipping enrichment")
            enriched_chunks.extend(doc_chunks)
            continue

        # Convert dicts to ChunkData for the enricher
        chunk_objects = [
            ChunkData(
                chunk_id=c["chunk_id"],
                doc_id=c["doc_id"],
                text=c["text"],
                section=c.get("section", ""),
                metadata=c.get("metadata", {}),
            )
            for c in doc_chunks
        ]

        enriched = enricher.enrich(chunk_objects, paper)
        enriched_chunks.extend(c.to_dict() for c in enriched)

        if (i + 1) % 50 == 0:
            logger.info(f"  Papers enriched: {i + 1}/{total_papers}")

    elapsed = time.time() - start
    logger.info(f"Enrichment complete: {len(enriched_chunks)} chunks in {elapsed:.1f}s")

    # Save enriched chunks
    enriched_path = DATA_DIR / "chunks_enriched.json"
    with open(enriched_path, "w") as f:
        json.dump(enriched_chunks, f, indent=2, default=str)
    logger.info(f"Saved enriched chunks to {enriched_path}")

    return enriched_chunks


def step_index(chunks: list[dict]) -> Embedder:
    """Step 3: Embed and index chunks."""
    logger = logging.getLogger("pipeline.index")
    logger.info("=" * 60)
    logger.info("STEP 3: Embedding and indexing chunks")
    logger.info("=" * 60)

    start = time.time()
    embedder_config = EmbedderConfig(
        model_name=EMBEDDING_MODEL,
        chroma_path=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
        distance_metric=DISTANCE_METRIC,
        batch_size=EMBEDDING_BATCH_SIZE,
    )
    embedder = Embedder(config=embedder_config)

    embedder.index_chunks(chunks, batch_size=embedder_config.batch_size)
    elapsed = time.time() - start

    # Invalidate the paper count cache so the API recomputes it on next startup
    cache_file = CHROMA_DIR / ".paper_count"
    if cache_file.exists():
        cache_file.unlink()
        logger.info("Paper count cache invalidated")

    stats = embedder.get_collection_stats()
    logger.info(f"Indexing complete in {elapsed:.1f}s")
    logger.info(f"Collection stats: {stats}")

    return embedder


def step_test():
    """Step 4: Run test queries against the index."""
    # Import here to avoid circular imports at module load time

    logger = logging.getLogger("pipeline.test")
    logger.info("=" * 60)
    logger.info("STEP 4: Running test queries")
    logger.info("=" * 60)

    retriever_config = RetrieverConfig(
        embedding_model=EMBEDDING_MODEL,
        chroma_path=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )
    retriever = HybridRetriever(config=retriever_config)

    eval_path = EVAL_DIR / "eval_queries.json"
    with open(eval_path) as f:
        eval_queries = json.load(f)

    test_queries = [q for q in eval_queries if q["difficulty"] in ("easy", "deflection")][:8]

    results_log = []
    for q in test_queries:
        result = retriever.print_results(q["question"], top_k=5)
        results_log.append(
            {
                "id": q["id"],
                "question": q["question"],
                "difficulty": q["difficulty"],
                "top_score": result["top_score"],
                "is_relevant": result["is_relevant"],
                "num_results": len(result["results"]),
                "should_deflect": q.get("should_deflect", False),
            }
        )

    results_path = EVAL_DIR / "smoke_test_results.json"
    with open(results_path, "w") as f:
        json.dump(results_log, f, indent=2)
    logger.info(f"Saved test results to {results_path}")

    logger.info("\n" + "=" * 60)
    logger.info("SMOKE TEST SUMMARY")
    logger.info("=" * 60)
    for r in results_log:
        status = "PASS" if r["is_relevant"] != r["should_deflect"] else "FAIL"
        logger.info(
            f"  {status} [{r['id']}] score={r['top_score']:.3f} "
            f"relevant={r['is_relevant']} "
            f"deflect={r['should_deflect']} "
            f"| {r['question'][:60]}"
        )

    return results_log


def main():
    parser = argparse.ArgumentParser(description="RAG-Bench Pipeline")
    parser.add_argument(
        "--step",
        choices=["ingest", "chunk", "enrich", "index", "test", "all"],
        default="all",
        help="Which pipeline step to run (default: all)",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Use hybrid ingestion (merge scraped papers + HuggingFace ai-arxiv2 dataset)",
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        help="Enable contextual enrichment (requires Ollama running locally)",
    )
    parser.add_argument(
        "--enrich-model",
        default="qwen2.5:14b-instruct-q4_K_M",
        help="Ollama model for enrichment (default: qwen2.5:14b-instruct-q4_K_M)",
    )
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("pipeline")

    logger.info("RAG-Bench: Data Ingestion & Indexing")
    logger.info(f"Dataset: {DATASET_NAME}")
    logger.info(f"Embedding model: {EMBEDDING_MODEL}")
    logger.info(f"Chunk size: {CHUNK_SIZE_CHARS} chars (~{CHUNK_SIZE_CHARS // 4} tokens)")
    logger.info(f"ChromaDB path: {CHROMA_DIR}")

    overall_start = time.time()

    if args.step in ("ingest", "all"):
        docs = step_ingest(use_hybrid=args.hybrid)
    else:
        docs_path = DATA_DIR / "parsed_papers.json"
        if docs_path.exists():
            with open(docs_path) as f:
                docs = json.load(f)
            logger.info(f"Loaded {len(docs)} cached parsed papers")
        else:
            docs = []

    if args.step in ("chunk", "all"):
        if not docs:
            logger.error("No documents to chunk. Run ingest first.")
            sys.exit(1)
        chunks = step_chunk(docs)
    else:
        chunks_path = DATA_DIR / "chunks.json"
        if chunks_path.exists():
            with open(chunks_path) as f:
                chunks = json.load(f)
            logger.info(f"Loaded {len(chunks)} cached chunks")
        else:
            chunks = []

    if args.step in ("enrich", "all") and args.enrich:
        if not chunks:
            logger.error("No chunks to enrich. Run chunk first.")
            sys.exit(1)
        if not docs:
            # Need paper data for enrichment context
            docs_path = DATA_DIR / "parsed_papers.json"
            if docs_path.exists():
                with open(docs_path) as f:
                    docs = json.load(f)
            else:
                logger.error("No parsed papers found. Run ingest first.")
                sys.exit(1)
        enricher_config = EnricherConfig(
            enabled=True,
            ollama_model=args.enrich_model,
        )
        chunks = step_enrich(chunks, docs, enricher_config)
    elif args.step == "enrich" and not args.enrich:
        logger.warning("--step enrich requires --enrich flag. Skipping.")

    if args.step in ("index", "all"):
        if not chunks:
            logger.error("No chunks to index. Run chunk first.")
            sys.exit(1)
        step_index(chunks)

    if args.step in ("test", "all"):
        step_test()

    total_elapsed = time.time() - overall_start
    logger.info(f"\nPipeline complete in {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
