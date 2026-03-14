"""
server.py — FastAPI backend for RAG-Bench.

Exposes the RAG pipeline as a REST API with:
- POST /api/query       -> Ask a question, get grounded answer with citations
- POST /api/query/stream -> SSE streaming for real-time token output
- GET  /api/stats       -> Corpus statistics (paper count, chunk count, etc.)
- POST /api/eval        -> Run the evaluation suite
- GET  /api/papers/{id}/pdf -> Fetch & cache arXiv PDF for paper viewer
- GET  /api/health      -> Health check

Usage:
    rag-serve                                     # Runs on port 8001 (avoids Docker conflict)
    RAG_API_PORT=9000 python -m rag_bench.api.server  # Custom port
"""

import asyncio
import json
import os
import random
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import httpx
import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator

from rag_bench.api.schemas import (
    BenchmarkEvalRequest,
    BenchmarkEvalResponse,
    BenchmarkHistoryEntry,
    BenchmarkHistoryResponse,
    BenchmarkResultItem,
    EvalRequest,
    EvalResponse,
    EvalResult,
    EvalScheduleRequest,
    EvalScheduleStatus,
    EvalSummary,
    FullEvalRequest,
    FullEvalResponse,
    FullEvalResult,
    GraphEdge,
    GraphNode,
    GraphSubgraph,
    PaperChunk,
    PaperDetail,
    PaperSummary,
    PipelineInsight,
    PipelineStage,
    QualityMetrics,
    QueryRequest,
    QueryResponse,
    SourceResult,
    StatsResponse,
    TrendDataPoint,
    TrendsResponse,
)
from rag_bench.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EVAL_DIR,
    EVAL_SCHEDULE_HOURS,
    PROJECT_ROOT,
    RAG_ENV,
    VERSION,
)
from rag_bench.core.citation_boost import CitationBooster
from rag_bench.core.configs import GraphStoreConfig, PipelineConfig
from rag_bench.core.generator import (
    DEFLECTION_PHRASES,
    RAGGenerator,
    RelevanceGate,
    build_llm_backend,
)
from rag_bench.core.graph_retriever import GraphRetriever
from rag_bench.core.graph_store import GraphStore
from rag_bench.core.pipeline import RAGPipeline, build_pipeline
from rag_bench.core.retriever import HybridRetriever
from rag_bench.eval.benchmark import get_benchmark
from rag_bench.eval.judge import JudgeLLM
from rag_bench.eval.metrics import (
    citation_density,
    count_unsupported_claims,
    extract_cited_source_numbers,
    source_coverage,
)
from rag_bench.eval.ragtruth.loader import load_ragtruth
from rag_bench.eval.ragtruth.runner import RAGTruthRunner
from rag_bench.eval.report import save_report
from rag_bench.eval.runner import EvalRunner
from rag_bench.observability import (
    ACTIVE_REQUESTS,
    BUILD_INFO,
    CITATION_COVERAGE,
    CORPUS_CHUNKS,
    CORPUS_PAPERS,
    PIPELINE_READY,
    QUERIES_TOTAL,
    REQUEST_DURATION,
    RETRIEVAL_TOP_SCORE,
    UNIQUE_USERS,
    RequestTracker,
    get_logger,
    setup_logging,
)
from rag_bench.utils.text import fix_encoding, strip_chunk_preamble

setup_logging(
    log_level=os.environ.get("LOG_LEVEL", "INFO"),
    json_logs=RAG_ENV == "production",
)
logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Application setup
# ═══════════════════════════════════════════════════════════════════════════

# Global pipeline instance (loaded once at startup via build_pipeline)
_pipeline: RAGPipeline | None = None

# Convenience aliases — set during lifespan, used by endpoints
_retriever: HybridRetriever | None = None
_generator: RAGGenerator | None = None
_citation_booster: CitationBooster | None = None
_llm_backend_name: str = ""
_llm_model_name: str = ""
_total_papers: int = 0  # Updated by background task
_papers_counting: bool = True  # False once the full scan completes
_start_time: float = time.time()
_tracker = RequestTracker()


_PAPER_COUNT_CACHE = CHROMA_DIR / ".paper_count"


def _load_cached_paper_count() -> int | None:
    """Return the cached paper count, or None if the cache doesn't exist."""
    try:
        return int(_PAPER_COUNT_CACHE.read_text().strip())
    except Exception:
        return None


def _save_cached_paper_count(count: int) -> None:
    try:
        _PAPER_COUNT_CACHE.write_text(str(count))
    except Exception as e:
        logger.warning(f"Could not write paper count cache: {e}")


async def _count_papers_background():
    """Scan all ChromaDB chunks to get the exact unique paper count. Runs after startup."""
    global _total_papers, _papers_counting
    if not _pipeline:
        return

    collection = _pipeline.retriever.collection
    total_chunks = collection.count()
    batch_size = 200_000
    offset = 0
    unique_papers: set[str] = set()

    while offset < total_chunks:
        batch = await asyncio.to_thread(
            collection.get,
            limit=batch_size,
            offset=offset,
            include=["metadatas"],
        )
        for meta in batch.get("metadatas", []):
            if meta:
                pid = meta.get("paper_id") or meta.get("arxiv_id") or meta.get("doc_id") or meta.get("title", "")
                if pid:
                    unique_papers.add(pid)
        offset += batch_size
        _total_papers = len(unique_papers)

    _papers_counting = False
    _save_cached_paper_count(_total_papers)
    logger.info(f"Paper count complete: {_total_papers:,} unique papers across {total_chunks:,} chunks")


@asynccontextmanager
async def lifespan(app):
    """Load the RAG pipeline at startup, clean up on shutdown."""
    global \
        _pipeline, \
        _retriever, \
        _generator, \
        _citation_booster, \
        _llm_backend_name, \
        _llm_model_name, \
        _total_papers, \
        _papers_counting

    logger.info("Loading RAG pipeline...")
    start = time.time()

    config = PipelineConfig.from_env()
    _pipeline = build_pipeline(config)

    # Set convenience aliases for endpoints that access components directly
    _retriever = _pipeline.retriever
    _generator = _pipeline.generator
    _citation_booster = _pipeline.generator.citation_booster if hasattr(_pipeline.generator, "citation_booster") else None
    _llm_backend_name = config.generator.llm_backend
    _llm_model_name = config.generator.llm_model

    total_chunks = _retriever.collection.count()
    cached = _load_cached_paper_count()
    if cached is not None:
        _total_papers = cached
        _papers_counting = False
        logger.info(f"Corpus: {total_chunks:,} chunks, {_total_papers:,} papers (cached)")
    else:
        logger.info(f"Corpus: {total_chunks:,} chunks — no cache, starting background paper count")
        asyncio.create_task(_count_papers_background())

    elapsed = time.time() - start
    logger.info(f"RAG pipeline loaded in {elapsed:.1f}s")

    # Set Prometheus gauges
    PIPELINE_READY.set(1)
    CORPUS_CHUNKS.set(total_chunks)
    CORPUS_PAPERS.set(_total_papers)
    BUILD_INFO.info(
        {
            "version": VERSION,
            "llm_backend": _llm_backend_name,
            "llm_model": _llm_model_name,
            "embedding_model": EMBEDDING_MODEL,
        }
    )

    # Start scheduled auto-eval if enabled
    if _eval_schedule_enabled:
        _start_eval_schedule()
        logger.info("Scheduled auto-eval enabled (every %dh)", _eval_schedule_interval)

    yield  # App runs here

    # Cancel scheduled eval on shutdown
    if _eval_schedule_task and not _eval_schedule_task.done():
        _eval_schedule_task.cancel()

    PIPELINE_READY.set(0)
    logger.info("Shutting down RAG pipeline")


app = FastAPI(
    title="RAG-Bench API",
    description="AI/ML Research Paper Question Answering with Grounded Citations",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — allow React dev server and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus auto-instrumentation for all endpoints
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attach a unique request ID and log request lifecycle."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)

    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000

    response.headers["X-Request-ID"] = request_id

    # Skip noisy paths
    path = request.url.path
    if path not in ("/metrics", "/api/health", "/api/metrics/summary"):
        logger.info(
            "request_completed",
            method=request.method,
            path=path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 1),
        )

    return response


def _format_sources(results: list[dict]) -> list[SourceResult]:
    """Convert raw retrieval results to API response format."""
    sources = []
    top_score = results[0].get("score", 0.0) if results else 0.0

    for i, r in enumerate(results, 1):
        score = r.get("score", 0.0)
        meta = r.get("metadata", {})

        # Assign relevance tier based on score
        if top_score > 2.0:
            if score >= 3.0:
                relevance = "high"
            elif score >= 1.0:
                relevance = "medium"
            else:
                relevance = "low"
        else:
            if score >= 0.5:
                relevance = "high"
            elif score >= 0.2:
                relevance = "medium"
            else:
                relevance = "low"

        sources.append(
            SourceResult(
                rank=i,
                score=round(score, 4),
                title=meta.get("title", meta.get("source_display", "Unknown")),
                section=meta.get("section", ""),
                text_preview=strip_chunk_preamble(fix_encoding(r.get("text", "")))[:300],
                paper_id=meta.get("paper_id", meta.get("doc_id", meta.get("arxiv_id", ""))),
                chunk_id=r.get("chunk_id", ""),
                relevance=relevance,
            )
        )
    return sources


# ═══════════════════════════════════════════════════════════════════════════
# Per-response quality metrics (fast, deterministic — no LLM calls)
# ═══════════════════════════════════════════════════════════════════════════


_STOP_WORDS = frozenset(
    [
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
        "shall",
        "should",
        "may",
        "might",
        "must",
        "can",
        "could",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "as",
        "or",
        "and",
        "but",
        "not",
        "no",
        "so",
        "if",
        "than",
        "then",
        "their",
        "there",
        "they",
        "we",
        "our",
        "us",
        "which",
        "who",
        "what",
        "how",
        "when",
        "where",
        "why",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
        "s",
        "t",
        "re",
        "ve",
        "ll",
        "d",
        "m",
    ]
)


def _content_words(text: str) -> set[str]:
    """Extract meaningful content words (3+ chars, not stop words)."""

    tokens = re.findall(r"[a-zA-Z]\w{2,}", text.lower())
    return {t for t in tokens if t not in _STOP_WORDS}


def _compute_faithfulness_heuristic(answer: str, source_passages: list[str]) -> float:
    """
    Content-word overlap between answer sentences and source passages, mapped to 1-5.

    Filters stop words and short tokens so common function words don't inflate the score.
    Only considers sentences with at least 4 content words to avoid noise from
    one-word headers or transition phrases.
    """
    if not source_passages:
        return 1.0

    # Strip the trailing sources block from the answer
    answer_body = re.split(r"\n\s*(?:\[Source[^\]]*\]:|Sources?:)", answer, maxsplit=1)[0]
    source_words = _content_words(" ".join(source_passages))
    if not source_words:
        return 1.0

    sentences = [s.strip() for s in re.split(r"[.!?]+", answer_body) if len(s.strip()) > 15]
    if not sentences:
        return 1.0

    overlap_scores = []
    for sentence in sentences:
        words = _content_words(sentence)
        if len(words) >= 4:  # skip very short/header sentences
            overlap_scores.append(len(words & source_words) / len(words))

    if not overlap_scores:
        return 1.0
    avg_overlap = sum(overlap_scores) / len(overlap_scores)
    return round(1.0 + avg_overlap * 4.0, 1)


def _compute_quality_metrics(answer: str, all_results: list[dict], filtered_results: list[dict]) -> QualityMetrics:
    """Compute inline quality metrics for a single query response."""
    sources_for_metrics = filtered_results if filtered_results else all_results[:5]
    num_provided = len(sources_for_metrics)
    cited = extract_cited_source_numbers(answer)
    source_passages = [r.get("text", "") for r in sources_for_metrics if r.get("text")]

    return QualityMetrics(
        retrieval_confidence=_compute_retrieval_confidence(all_results),
        citation_coverage=round(source_coverage(answer, num_provided), 4) if num_provided else 0.0,
        citation_density=round(citation_density(answer), 2),
        unsupported_claims=count_unsupported_claims(answer),
        sources_cited=len(cited),
        sources_provided=num_provided,
        top_retrieval_score=round(all_results[0].get("score", 0.0), 4) if all_results else 0.0,
        score_spread=_compute_score_spread(all_results),
        source_diversity=_compute_source_diversity(sources_for_metrics),
        per_source_cited=_compute_per_source_cited(answer, sources_for_metrics),
        faithfulness_score=_compute_faithfulness_heuristic(answer, source_passages),
    )


def _compute_retrieval_confidence(results: list[dict]) -> str:
    """Classify retrieval confidence as high/medium/low.

    Uses both absolute score and score gap (top vs second) so a clear winner
    at moderate absolute scores is still rated 'high'.
    """
    if not results:
        return "low"
    scores = [r.get("score", 0.0) for r in results]
    top = scores[0]
    second = scores[1] if len(scores) > 1 else 0.0
    # gap_ratio: how much better is the top result than the second
    gap_ratio = top / second if second > 0.0 else float("inf")

    # Cross-encoder scores typically in 1-10 range; cosine < 1.0
    if top > 2.0:  # cross-encoder range
        if top > 3.0 and gap_ratio >= 1.8:
            return "high"
        elif top > 2.0 and gap_ratio >= 1.3:
            return "medium"
    else:  # cosine range
        if top > 0.65 and gap_ratio >= 1.5:
            return "high"
        elif top > 0.4 and gap_ratio >= 1.2:
            return "medium"
    return "low"


def _compute_score_spread(results: list[dict]) -> dict:
    """Return min/max/mean/std/gap_ratio of retrieval scores.

    gap_ratio is top_score / second_score — a high ratio (>1.8) means the top
    result is a clear winner, indicating focused retrieval for the query.
    """
    if not results:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "gap_ratio": 0.0}
    scores = [r.get("score", 0.0) for r in results]
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    second = scores[1] if len(scores) > 1 else 0.0
    gap_ratio = round(scores[0] / second, 2) if second > 0.0 else 0.0
    return {
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "mean": round(mean, 4),
        "std": round(variance**0.5, 4),
        "gap_ratio": gap_ratio,
    }


def _compute_source_diversity(results: list[dict]) -> dict:
    """Return unique papers and sections from results."""
    papers = set()
    sections = set()
    for r in results:
        meta = r.get("metadata", {})
        pid = meta.get("paper_id") or meta.get("arxiv_id") or meta.get("doc_id") or ""
        if pid:
            papers.add(pid)
        sec = meta.get("section", "")
        if sec:
            sections.add(sec)
    return {
        "unique_papers": len(papers),
        "unique_sections": len(sections),
        "papers": sorted(papers),
    }


def _compute_per_source_cited(answer: str, results: list[dict]) -> list[dict]:
    """For each source, check if it was cited in the answer and how many times.

    Detects three citation styles:
      - [Source N] inline in body
      - [N] bare number inline in body
      - Footer-only: source listed in the trailing sources block (counted as cited once)
    """
    parts = re.split(r"\n\s*(?:\[Source[^\]]*\]:|Sources?:)", answer, maxsplit=1)
    body = parts[0]
    footer = parts[1] if len(parts) > 1 else ""

    per_source = []
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})

        # Standard [Source N] in body
        standard_count = len(re.findall(rf"\[Source\s+{i}\]", body))
        # Bare [N] in body only (avoid matching footer list items)
        bare_count = len(re.findall(rf"\[{i}\]", body))
        inline_count = standard_count + bare_count

        # Footer reference (only credited if no inline citation found)
        footer_cited = False
        if inline_count == 0:
            footer_cited = bool(re.search(rf"\[Source\s+{i}\]", footer) or re.search(rf"\[{i}\]", footer))

        count = inline_count + (1 if footer_cited else 0)
        per_source.append(
            {
                "source_number": i,
                "paper_id": meta.get("paper_id") or meta.get("arxiv_id") or meta.get("doc_id", ""),
                "title": meta.get("title", "Unknown"),
                "cited": count > 0,
                "citation_count": count,
                "footer_only": footer_cited,
                "score": round(r.get("score", 0.0), 4),
            }
        )
    return per_source


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline insight computation
# ═══════════════════════════════════════════════════════════════════════════


def _compute_pipeline_insight(
    question: str,
    all_results: list[dict],
    filtered_results: list[dict],
    retrieval_ms: float = 0.0,
    reranking_ms: float = 0.0,
) -> PipelineInsight:
    """Compute pipeline decision trace from retrieval results.

    Shows what the CRAG and agent subsystems decided (or would decide)
    for this query, based on rerank scores and query patterns.
    """
    scores = [r.get("score", 0.0) for r in all_results] if all_results else []
    top_score = scores[0] if scores else 0.0

    # ── Query classification (mirrors agent._classify_query) ──
    q_lower = question.lower()
    comparison_kw = (
        "compare",
        "difference",
        "versus",
        "vs",
        "contrast",
        "better",
        "worse",
        "advantage",
        "disadvantage",
        r"how does .* differ",
        "what is the difference",
    )
    multi_hop_kw = (
        "relationship between",
        r"how does .* relate",
        "what led to",
        "evolution of",
        "trace the",
        "step by step",
        r"explain how .* affects",
    )
    query_type = "simple"
    for pat in comparison_kw:
        if re.search(pat, q_lower):
            query_type = "multi_hop"
            break
    if query_type == "simple":
        for pat in multi_hop_kw:
            if re.search(pat, q_lower):
                query_type = "multi_hop"
                break

    # ── CRAG confidence (mirrors crag._score_confidence) ──
    crag_correct_threshold = 0.90
    crag_ambiguous_threshold = 0.70
    crag_refinement_floor = 0.30

    if top_score >= crag_correct_threshold:
        crag_confidence = "correct"
        # Check for flat scores (score concentration)
        if len(scores) >= 3:
            top3_mean = sum(scores[:3]) / 3
            gap = top_score - top3_mean
            if gap < 0.005 and top_score < 0.95:
                crag_confidence = "ambiguous"
    elif top_score >= crag_ambiguous_threshold:
        crag_confidence = "ambiguous"
    elif scores:
        crag_confidence = "incorrect"
    else:
        crag_confidence = "unknown"

    # CRAG action based on confidence
    if crag_confidence == "correct":
        crag_action = "pass_through"
    elif crag_confidence == "ambiguous":
        crag_action = "refine_only"
    else:
        crag_action = "hyde_rewrite"

    # ── Count sources by retrieval type ──
    source_types: dict[str, int] = {}
    for r in all_results:
        for src in r.get("sources", []):
            source_types[src] = source_types.get(src, 0) + 1

    # ── Count filtered results ──
    refined_count = sum(1 for s in scores if s >= crag_refinement_floor)

    # ── Build stage list ──
    stages = []

    # Stage 1: Retrieval
    stages.append(
        PipelineStage(
            name="retrieval",
            status="done",
            duration_ms=round(retrieval_ms, 1),
            detail=f"Retrieved {len(all_results)} candidates via hybrid search (BM25 + dense)",
            metadata={"candidates": len(all_results)},
        )
    )

    # Stage 2: Reranking
    stages.append(
        PipelineStage(
            name="reranking",
            status="done",
            duration_ms=round(reranking_ms, 1),
            detail=f"Cross-encoder reranking, top score: {top_score:.3f}",
            metadata={"top_score": round(top_score, 4), "model": "bge-reranker-v2-m3"},
        )
    )

    # Stage 3: CRAG evaluation
    crag_detail_map = {
        "correct": f"High confidence ({top_score:.3f}) — results are trustworthy",
        "ambiguous": f"Medium confidence ({top_score:.3f}) — applying knowledge refinement",
        "incorrect": f"Low confidence ({top_score:.3f}) — would trigger HyDE rewrite",
        "unknown": "No results to evaluate",
    }
    stages.append(
        PipelineStage(
            name="crag",
            status=crag_confidence,
            duration_ms=0.0,
            detail=crag_detail_map.get(crag_confidence, ""),
            metadata={
                "confidence": crag_confidence,
                "action": crag_action,
                "top_score": round(top_score, 4),
                "threshold_correct": crag_correct_threshold,
                "threshold_ambiguous": crag_ambiguous_threshold,
            },
        )
    )

    # Stage 4: Knowledge refinement
    filtered_out = len(all_results) - refined_count
    stages.append(
        PipelineStage(
            name="refinement",
            status="done" if filtered_out > 0 else "skipped",
            duration_ms=0.0,
            detail=(
                f"Filtered {filtered_out} low-relevance results (floor: {crag_refinement_floor})"
                if filtered_out > 0
                else "All results above quality floor"
            ),
            metadata={"filtered": filtered_out, "remaining": refined_count, "floor": crag_refinement_floor},
        )
    )

    # Stage 5: Generation
    stages.append(
        PipelineStage(
            name="generation",
            status="pending",
            duration_ms=0.0,
            detail="Generating grounded answer with citations",
            metadata={},
        )
    )

    return PipelineInsight(
        query_type=query_type,
        crag_confidence=crag_confidence,
        crag_top_score=round(top_score, 4),
        crag_action=crag_action,
        stages=stages,
        total_candidates=len(all_results),
        final_results=len(filtered_results) if filtered_results else refined_count,
        sources_by_type=source_types,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/health")
async def health_check():
    """Health check — is the pipeline loaded?"""
    uptime = round(time.time() - _start_time)
    total_chunks = _retriever.collection.count() if _pipeline else 0
    return {
        "status": "ok" if _pipeline else "loading",
        "pipeline_ready": _pipeline is not None,
        "version": VERSION,
        "uptime_seconds": uptime,
        "corpus": {
            "chunks": total_chunks,
            "papers": _total_papers,
            "papers_counting": _papers_counting,
        },
        "llm": {
            "backend": _llm_backend_name,
            "model": _llm_model_name,
        },
    }


@app.get("/api/metrics/summary")
async def metrics_summary():
    """Return curated metrics summary for the frontend Production tab."""
    return _tracker.summary()


@app.post("/api/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest, raw_request: Request):
    """Ask a question and get a grounded answer with citations."""
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline still loading")

    # Quick check for casual / non-research queries
    casual_patterns = {
        "hi",
        "hello",
        "hey",
        "yo",
        "sup",
        "howdy",
        "hola",
        "greetings",
        "good morning",
        "good afternoon",
        "good evening",
        "what's up",
        "how are you",
        "thanks",
        "thank you",
        "bye",
        "goodbye",
        "help",
    }
    q_lower = request.question.strip().lower().rstrip("!?.,")
    if q_lower in casual_patterns or len(q_lower) < 3:
        return QueryResponse(
            answer=(
                "Hello! I'm RAG-Bench, an AI/ML research paper assistant. "
                "Ask me a question about machine learning, transformers, "
                "attention mechanisms, training techniques, or any AI research topic — "
                "and I'll find relevant papers and give you a grounded answer with citations."
            ),
            sources=[],
            deflected=False,
            deflection_reason="",
            scores=[],
            latency_ms=0.0,
            backend="n/a",
            model="n/a",
        )

    # Optionally override LLM backend per request
    generator = _generator
    backend_name = _llm_backend_name
    model_name = _llm_model_name

    if request.backend and request.backend != _llm_backend_name:
        llm = build_llm_backend(
            request.backend,
            request.model or "",
            "",
        )
        # Create temporary generator with citation booster if enabled
        booster = _citation_booster if request.enable_citation_boost else None
        generator = RAGGenerator(
            retriever=_retriever,
            llm_backend=llm,
            relevance_gate=RelevanceGate(),
            top_k=request.top_k,
            citation_booster=booster,
        )
        backend_name = request.backend
        model_name = request.model or ""
    elif not request.enable_citation_boost:
        # Create generator without booster if disabled
        generator = RAGGenerator(
            retriever=_retriever,
            llm_backend=_generator.llm_backend,
            relevance_gate=RelevanceGate(),
            top_k=request.top_k,
            citation_booster=None,
        )

    ACTIVE_REQUESTS.inc()
    start = time.time()
    try:
        result = generator.answer(request.question, top_k=request.top_k)
    except Exception:
        ACTIVE_REQUESTS.dec()
        QUERIES_TOTAL.labels(status="error").inc()
        raise
    latency = (time.time() - start) * 1000
    ACTIVE_REQUESTS.dec()

    display_results = result.get("filtered_results", result.get("results", []))

    # Compute per-response quality metrics (fast, no LLM)
    quality = QualityMetrics()
    if not result["deflected"]:
        try:
            quality = _compute_quality_metrics(
                result["answer"],
                result.get("results", []),
                result.get("filtered_results", []),
            )
        except Exception as e:
            logger.warning(f"Quality metrics computation failed: {e}")

    # Record Prometheus metrics
    status = "deflected" if result["deflected"] else "success"
    QUERIES_TOTAL.labels(status=status).inc()
    REQUEST_DURATION.labels(endpoint="/api/query", method="POST").observe(latency / 1000)
    if result.get("scores"):
        RETRIEVAL_TOP_SCORE.observe(result["scores"][0])
    if quality.citation_coverage:
        CITATION_COVERAGE.observe(quality.citation_coverage)

    # Record in-memory tracker for frontend
    client_ip = raw_request.headers.get("x-forwarded-for", raw_request.client.host if raw_request.client else "")
    _tracker.record_query(
        latency_ms=latency,
        status=status,
        question_preview=request.question,
        retrieval_ms=result.get("retrieval_ms", 0.0),
        generation_ms=result.get("generation_ms", 0.0),
        reranking_ms=result.get("reranking_ms", 0.0),
        citation_coverage=quality.citation_coverage,
        top_score=result["scores"][0] if result.get("scores") else 0.0,
        client_ip=client_ip,
    )
    UNIQUE_USERS.set(len(_tracker.unique_ips))

    return QueryResponse(
        answer=result["answer"],
        sources=_format_sources(display_results),
        deflected=result["deflected"],
        deflection_reason=result.get("deflection_reason", ""),
        scores=result.get("scores", []),
        latency_ms=round(latency, 1),
        backend=backend_name,
        model=model_name,
        quality=quality,
    )


@app.post("/api/query/stream")
async def query_rag_stream(request: QueryRequest, raw_request: Request):
    """Stream the RAG answer via Server-Sent Events (SSE)."""
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline still loading")

    # Handle casual queries fast
    casual_patterns = {
        "hi",
        "hello",
        "hey",
        "yo",
        "sup",
        "howdy",
        "hola",
        "greetings",
        "good morning",
        "good afternoon",
        "good evening",
        "what's up",
        "how are you",
        "thanks",
        "thank you",
        "bye",
        "goodbye",
        "help",
    }
    q_lower = request.question.strip().lower().rstrip("!?.,")

    if q_lower in casual_patterns or len(q_lower) < 3:
        greeting = (
            "Hello! I'm RAG-Bench, an AI/ML research paper assistant. "
            "Ask me a question about machine learning, transformers, "
            "attention mechanisms, training techniques, or any AI research topic — "
            "and I'll find relevant papers and give you a grounded answer with citations."
        )

        async def greeting_stream():
            yield f"data: {json.dumps({'event': 'sources', 'sources': []})}\n\n"
            yield f"data: {json.dumps({'event': 'token', 'token': greeting})}\n\n"
            done_msg = {
                "event": "done",
                "answer": greeting,
                "deflected": False,
                "latency_ms": 0,
                "backend": "n/a",
                "model": "n/a",
            }
            yield f"data: {json.dumps(done_msg)}\n\n"

        return StreamingResponse(
            greeting_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    client_ip = raw_request.headers.get("x-forwarded-for", raw_request.client.host if raw_request.client else "")

    async def event_stream():
        try:
            start = time.time()

            _STOP = object()

            def _next_or_stop(g):
                try:
                    return next(g)
                except StopIteration:
                    return _STOP

            loop = asyncio.get_event_loop()

            # Use appropriate generator based on citation boost setting
            generator = _generator
            if not request.enable_citation_boost:
                # Create temporary generator without booster
                generator = RAGGenerator(
                    retriever=_retriever,
                    llm_backend=_generator.llm,
                    relevance_gate=RelevanceGate(),
                    top_k=request.top_k,
                    citation_booster=None,
                )

            gen = generator.answer_stream(request.question, top_k=request.top_k)

            # Capture results and timing from events
            _stream_all_results: list[dict] = []
            _stream_filtered_results: list[dict] = []
            _stream_retrieval_ms = 0.0
            _stream_reranking_ms = 0.0
            _stream_pipeline_stages: list[dict] = []
            _stream_query_type = "simple"
            _stream_crag: dict = {}
            _stream_sources_by_type: dict = {}

            while True:
                evt = await loop.run_in_executor(None, _next_or_stop, gen)
                if evt is _STOP:
                    break

                if evt["event"] == "pipeline":
                    # Real-time pipeline stage event from generator
                    stage_data = {
                        "stage": evt.get("stage", ""),
                        "status": evt.get("status", ""),
                        "detail": evt.get("detail", ""),
                        "data": evt.get("data", {}),
                    }
                    # Update or append stage
                    existing = next((s for s in _stream_pipeline_stages if s["stage"] == stage_data["stage"]), None)
                    if existing:
                        existing.update(stage_data)
                    else:
                        _stream_pipeline_stages.append(stage_data)

                    yield f"data: {json.dumps({'event': 'pipeline', **stage_data})}\n\n"

                elif evt["event"] == "sources":
                    _stream_all_results = evt.get("results", [])
                    _stream_filtered_results = evt.get("filtered_results", [])
                    _stream_retrieval_ms = evt.get("retrieval_ms", 0.0)
                    _stream_reranking_ms = evt.get("reranking_ms", 0.0)
                    _stream_query_type = evt.get("query_type", "simple")
                    _stream_crag = evt.get("crag", {})
                    _stream_sources_by_type = evt.get("sources_by_type", {})
                    # Don't send sources yet — wait until after generation
                    # so _strip_uncited_sources can filter to only cited ones.

                elif evt["event"] == "deflected":
                    latency = (time.time() - start) * 1000
                    deflection_msg = {
                        "event": "done",
                        "answer": evt["answer"],
                        "deflected": True,
                        "reason": evt["reason"],
                        "latency_ms": round(latency, 1),
                        "backend": _llm_backend_name,
                        "model": _llm_model_name,
                    }
                    yield f"data: {json.dumps(deflection_msg)}\n\n"
                    QUERIES_TOTAL.labels(status="deflected").inc()
                    REQUEST_DURATION.labels(endpoint="/api/query/stream", method="POST").observe(latency / 1000)
                    _tracker.record_query(
                        latency_ms=latency,
                        status="deflected",
                        question_preview=request.question,
                        retrieval_ms=_stream_retrieval_ms,
                        reranking_ms=_stream_reranking_ms,
                        client_ip=client_ip,
                    )
                    UNIQUE_USERS.set(len(_tracker.unique_ips))
                    return

                elif evt["event"] == "token":
                    yield f"data: {json.dumps({'event': 'token', 'token': evt['token']})}\n\n"

                elif evt["event"] == "done":
                    # Update filtered results if the generator stripped uncited sources
                    if evt.get("filtered_results"):
                        _stream_filtered_results = evt["filtered_results"]

                    # Send sources after generation so only cited ones are shown
                    display_results = _stream_filtered_results or _stream_all_results
                    sources = _format_sources(display_results)
                    yield f"data: {json.dumps({'event': 'sources', 'sources': [s.model_dump() for s in sources]})}\n\n"

                    latency = (time.time() - start) * 1000
                    quality = QualityMetrics()
                    try:
                        quality = _compute_quality_metrics(
                            evt["answer"],
                            _stream_all_results,
                            _stream_filtered_results,
                        )
                    except Exception as qe:
                        logger.warning(f"Stream quality metrics failed: {qe}")
                    # Build final pipeline summary from accumulated stages
                    pipeline_summary = {
                        "query_type": _stream_query_type,
                        "crag_confidence": _stream_crag.get("confidence", ""),
                        "crag_top_score": _stream_crag.get("top_score", 0.0),
                        "crag_action": _stream_crag.get("action", ""),
                        "stages": _stream_pipeline_stages,
                        "total_candidates": len(_stream_all_results),
                        "final_results": (
                            len(_stream_filtered_results) if _stream_filtered_results else len(_stream_all_results)
                        ),
                        "sources_by_type": _stream_sources_by_type,
                    }

                    completion_msg = {
                        "event": "done",
                        "answer": evt["answer"],
                        "deflected": False,
                        "latency_ms": round(latency, 1),
                        "backend": _llm_backend_name,
                        "model": _llm_model_name,
                        "quality": quality.model_dump(),
                        "pipeline": pipeline_summary,
                    }
                    yield f"data: {json.dumps(completion_msg)}\n\n"

                    # Record metrics
                    QUERIES_TOTAL.labels(status="success").inc()
                    REQUEST_DURATION.labels(endpoint="/api/query/stream", method="POST").observe(latency / 1000)
                    if quality.citation_coverage:
                        CITATION_COVERAGE.observe(quality.citation_coverage)
                    _tracker.record_query(
                        latency_ms=latency,
                        status="success",
                        question_preview=request.question,
                        retrieval_ms=evt.get("retrieval_ms", 0.0),
                        generation_ms=evt.get("generation_ms", 0.0),
                        reranking_ms=evt.get("reranking_ms", 0.0),
                        citation_coverage=quality.citation_coverage,
                        client_ip=client_ip,
                    )
                    UNIQUE_USERS.set(len(_tracker.unique_ips))

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"
            QUERIES_TOTAL.labels(status="error").inc()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Return corpus and pipeline statistics (computed at startup)."""
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline still loading")

    return StatsResponse(
        total_chunks=_retriever.collection.count(),
        total_papers=_total_papers,
        papers_counting=_papers_counting,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
        llm_backend=_llm_backend_name,
        llm_model=_llm_model_name,
    )


@app.post("/api/eval", response_model=EvalResponse)
async def run_evaluation(request: EvalRequest):
    """Run the evaluation suite and return detailed results."""
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline still loading")

    eval_path = EVAL_DIR / "eval_queries.json"
    if not eval_path.exists():
        raise HTTPException(status_code=404, detail="eval_queries.json not found")

    with open(eval_path) as f:
        queries = json.load(f)

    if not request.run_all:
        queries = [q for q in queries if q.get("difficulty") in ("easy", "deflection", "adversarial")]

    results = []
    correct = 0
    by_difficulty = {}

    for q in queries:
        result = _generator.answer(q["question"], top_k=5)
        expected_deflect = q.get("should_deflect", False)
        did_deflect = result["deflected"]
        deflect_source = "gate" if did_deflect else ""

        # LLM-level deflection detection
        if not did_deflect and expected_deflect:
            answer_lower = result["answer"].lower()
            if any(phrase in answer_lower for phrase in DEFLECTION_PHRASES):
                did_deflect = True
                deflect_source = "llm"

        is_correct = did_deflect == expected_deflect
        if is_correct:
            correct += 1

        diff = q.get("difficulty", "unknown")
        if diff not in by_difficulty:
            by_difficulty[diff] = {"correct": 0, "total": 0}
        by_difficulty[diff]["total"] += 1
        if is_correct:
            by_difficulty[diff]["correct"] += 1

        top_score = result["scores"][0] if result.get("scores") else 0.0

        results.append(
            EvalResult(
                id=q["id"],
                question=q["question"],
                passed=is_correct,
                expected_deflect=expected_deflect,
                actual_deflect=did_deflect,
                deflect_source=deflect_source,
                top_score=round(top_score, 3),
                difficulty=diff,
            )
        )

    return EvalResponse(
        total=len(queries),
        correct=correct,
        accuracy=round(correct / len(queries) * 100, 1) if queries else 0,
        results=results,
        by_difficulty={
            k: {
                "correct": v["correct"],
                "total": v["total"],
                "accuracy": round(v["correct"] / v["total"] * 100, 1) if v["total"] else 0,
            }
            for k, v in sorted(by_difficulty.items())
        },
    )


@app.post("/api/eval/full", response_model=FullEvalResponse)
async def run_full_evaluation(request: FullEvalRequest):
    """Run the comprehensive evaluation suite with retrieval, citation, and faithfulness metrics."""
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline still loading")

    # Create judge from existing LLM backend (skip if retrieval_only)
    judge = None
    if not request.retrieval_only:
        judge = JudgeLLM(_generator.llm)

    runner = EvalRunner(
        retriever=_retriever,
        generator=_generator,
        judge=judge,
    )

    report = runner.run_all(
        filter_topic=request.topic,
        filter_type=request.query_type,
        filter_difficulty=request.difficulty,
        retrieval_only=request.retrieval_only,
    )

    # Convert to response schema
    results = []
    for r in report.results:
        results.append(
            FullEvalResult(
                id=r.id,
                question=r.question,
                query_type=r.query_type,
                topic=r.topic,
                difficulty=r.difficulty,
                latency_ms=r.latency_ms,
                answer_preview=r.answer_preview,
                error=r.error,
            )
        )

    return FullEvalResponse(
        summary=EvalSummary(**report.summary) if report.summary else EvalSummary(),
        by_topic=report.by_topic,
        by_query_type=report.by_query_type,
        by_difficulty=report.by_difficulty,
        results=results,
        metadata=report.metadata,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Paper viewer endpoints
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/papers", response_model=list[PaperSummary])
async def list_papers():
    """List all papers in the corpus with basic metadata."""
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline still loading")

    total = _retriever.collection.count()
    all_data = _retriever.collection.get(
        limit=total,
        include=["metadatas"],
    )

    papers = {}
    for meta in all_data.get("metadatas", []):
        pid = meta.get("doc_id", meta.get("paper_id", meta.get("arxiv_id", "")))
        if not pid:
            continue
        if pid not in papers:
            papers[pid] = {
                "paper_id": pid,
                "title": meta.get("title", "Unknown"),
                "year": meta.get("year", 0),
                "arxiv_id": meta.get("arxiv_id", ""),
                "chunk_count": 0,
                "sections": set(),
            }
        papers[pid]["chunk_count"] += 1
        section = meta.get("section", "")
        if section:
            papers[pid]["sections"].add(section)

    result = []
    for p in sorted(papers.values(), key=lambda x: x["title"]):
        result.append(
            PaperSummary(
                paper_id=p["paper_id"],
                title=p["title"],
                year=p["year"],
                arxiv_id=p["arxiv_id"],
                chunk_count=p["chunk_count"],
                sections=sorted(p["sections"]),
            )
        )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# PDF proxy — fetch arXiv PDFs with local caching
# NOTE: This endpoint MUST be declared before the catch-all
#       /api/papers/{paper_id:path} route.
# ═══════════════════════════════════════════════════════════════════════════
PDF_CACHE_DIR = PROJECT_ROOT / "data" / "pdfs"
PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _extract_arxiv_id(paper_id: str) -> str:
    """Extract a bare arXiv ID from various paper_id formats."""
    pid = paper_id.strip()
    if pid.lower().startswith("arxiv_"):
        pid = pid[6:]
    if pid and pid[-1].isdigit() and "v" in pid:
        base, _, ver = pid.rpartition("v")
        if ver.isdigit():
            pid = base
    return pid


async def _fetch_pdf(arxiv_id: str):
    """Return path to a cached PDF, downloading from arXiv if needed."""
    safe_name = arxiv_id.replace("/", "_").replace("\\", "_")
    cached = PDF_CACHE_DIR / f"{safe_name}.pdf"

    if cached.exists() and cached.stat().st_size > 1000:
        return cached

    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    logger.info(f"Downloading PDF: {url}")

    async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
        resp = await client.get(url)
        if resp.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"arXiv returned {resp.status_code} for {arxiv_id}",
            )
        content_type = resp.headers.get("content-type", "")
        if "pdf" not in content_type and len(resp.content) < 5000:
            raise HTTPException(
                status_code=502,
                detail=f"arXiv did not return a PDF for {arxiv_id}",
            )
        cached.write_bytes(resp.content)
        logger.info(f"Cached PDF: {cached} ({len(resp.content)} bytes)")

    return cached


@app.get("/api/papers/{paper_id:path}/pdf")
async def get_paper_pdf(paper_id: str):
    """Fetch the arXiv PDF for a paper (cached locally)."""
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline still loading")

    arxiv_id = ""

    for field in ("doc_id", "paper_id", "arxiv_id"):
        try:
            data = _retriever.collection.get(
                where={field: paper_id},
                limit=1,
                include=["metadatas"],
            )
            if data and data.get("ids"):
                meta = data["metadatas"][0]
                arxiv_id = meta.get("arxiv_id", "")
                break
        except Exception:
            continue

    if not arxiv_id:
        arxiv_id = _extract_arxiv_id(paper_id)

    if not arxiv_id:
        raise HTTPException(status_code=404, detail="No arXiv ID found for this paper")

    try:
        pdf_path = await _fetch_pdf(arxiv_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF fetch failed for {arxiv_id}: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to fetch PDF: {e}") from e

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"{arxiv_id}.pdf",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Paper detail (catch-all — must come AFTER /pdf route)
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/papers/{paper_id:path}", response_model=PaperDetail)
async def get_paper(paper_id: str):
    """Get all chunks for a specific paper, ordered by section and chunk index."""
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline still loading")

    total = _retriever.collection.count()

    for field in ("doc_id", "paper_id", "arxiv_id"):
        try:
            data = _retriever.collection.get(
                where={field: paper_id},
                limit=total,
                include=["documents", "metadatas"],
            )
            if data and data.get("ids"):
                break
        except Exception:
            continue
    else:
        raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found")

    if not data.get("ids"):
        raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found")

    chunks = []
    title = ""
    year = 0
    arxiv_id = ""
    source_display = ""
    sections = set()

    for idx, (cid, doc, meta) in enumerate(
        zip(
            data["ids"],
            data.get("documents", []),
            data.get("metadatas", []),
            strict=False,
        )
    ):
        section = meta.get("section", "")
        if section:
            sections.add(section)
        if not title:
            title = meta.get("title", "Unknown")
            year = meta.get("year", 0)
            arxiv_id = meta.get("arxiv_id", "")
            source_display = meta.get("source_display", "")

        chunk_idx = 0
        parts = cid.rsplit("_", 1)
        if len(parts) == 2:
            try:
                chunk_idx = int(parts[1])
            except ValueError:
                chunk_idx = idx

        chunks.append(
            PaperChunk(
                chunk_id=cid,
                text=fix_encoding(doc or ""),
                section=section,
                chunk_index=chunk_idx,
            )
        )

    section_order = {
        "abstract": 0,
        "introduction": 1,
        "background": 2,
        "related_work": 3,
        "related work": 3,
        "method": 4,
        "methods": 4,
        "methodology": 4,
        "approach": 4,
        "model": 5,
        "architecture": 5,
        "experiments": 6,
        "results": 7,
        "evaluation": 7,
        "discussion": 8,
        "analysis": 8,
        "conclusion": 9,
        "conclusions": 9,
        "references": 10,
    }
    chunks.sort(
        key=lambda c: (
            section_order.get(c.section.lower(), 5),
            c.chunk_index,
        )
    )

    return PaperDetail(
        paper_id=paper_id,
        title=title,
        year=year,
        arxiv_id=arxiv_id,
        source_display=source_display,
        chunks=chunks,
        sections=sorted(sections, key=lambda s: section_order.get(s.lower(), 5)),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Frontend — serve the single-file React app
# ═══════════════════════════════════════════════════════════════════════════
# ── Benchmark Evaluation Endpoints ──

_benchmark_running = False
_benchmark_history: list[dict] = []
BENCHMARK_HISTORY_FILE = PROJECT_ROOT / "data" / "benchmark_history.json"


def _load_benchmark_history() -> list[dict]:
    """Load benchmark history from disk."""
    global _benchmark_history
    if BENCHMARK_HISTORY_FILE.exists():
        try:
            with open(BENCHMARK_HISTORY_FILE) as f:
                _benchmark_history = json.load(f)
        except Exception:
            _benchmark_history = []
    return _benchmark_history


def _save_benchmark_history():
    """Save benchmark history to disk."""
    try:
        BENCHMARK_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(BENCHMARK_HISTORY_FILE, "w") as f:
            json.dump(_benchmark_history, f, indent=2)
    except Exception as e:
        logger.error("Failed to save benchmark history: %s", e)


# Load history on module import
_load_benchmark_history()


@app.post("/api/eval/benchmark", response_model=BenchmarkEvalResponse)
async def run_benchmark_evaluation(request: BenchmarkEvalRequest):
    """Run a RAG-Bench or RAGTruth benchmark evaluation."""
    global _benchmark_running

    if _benchmark_running:
        raise HTTPException(status_code=409, detail="A benchmark evaluation is already running")

    if not _generator:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    if request.benchmark not in ("ragbench", "ragtruth"):
        raise HTTPException(status_code=400, detail="Benchmark must be 'ragbench' or 'ragtruth'")

    _benchmark_running = True
    try:
        if request.benchmark == "ragbench":
            benchmark_entries = get_benchmark()
            if request.sample_size > 0 and request.sample_size < len(benchmark_entries):
                rng = random.Random(42)
                benchmark_entries = rng.sample(benchmark_entries, request.sample_size)

            judge = JudgeLLM(_generator.llm)
            eval_runner = EvalRunner(
                retriever=_retriever,
                generator=_generator,
                judge=judge,
                benchmark=benchmark_entries,
            )
            report = await asyncio.get_event_loop().run_in_executor(
                None,
                eval_runner.run_all,
            )

            results = [
                BenchmarkResultItem(
                    id=r.id,
                    question=r.question,
                    answer=r.answer_preview,
                    metrics={
                        "ndcg_at_5": r.retrieval.get("ndcg_at_k", 0),
                        "mrr": r.retrieval.get("mrr", 0),
                        "citation_precision": r.citation.get("precision", 0),
                        "completeness": r.completeness.get("score", 0),
                        "deflection_correct": r.deflection.get("correct", False),
                    },
                    error=r.error,
                )
                for r in report.results
            ]

            summary = report.summary
            accuracy = report.summary.get("retrieval_ndcg_at_5", None)

        else:  # ragtruth
            runner = RAGTruthRunner(generator=_generator)
            report = await asyncio.get_event_loop().run_in_executor(
                None, lambda: runner.run(sample_size=request.sample_size)
            )

            results = [
                BenchmarkResultItem(
                    id=r.id,
                    question=r.prompt[:200],
                    answer=r.generated_answer[:200],
                    metrics={
                        "has_hallucination_gold": r.has_hallucination_gold,
                        "has_hallucination_predicted": r.has_hallucination_predicted,
                        "span_f1": r.span_metrics.get("span_f1", 0),
                    },
                    error=r.error,
                )
                for r in report.results
            ]

            # Flatten case_level and by_type into summary (nested dicts break React rendering)
            summary = {**report.summary}
            for k, v in report.case_level.items():
                if isinstance(v, (int, float)):
                    summary[k] = v
            for hall_type, count in report.by_type.items():
                if isinstance(count, (int, float)):
                    summary[f"type_{hall_type}"] = count
            accuracy = report.summary.get("case_level_accuracy", None)

        metadata = report.metadata

        # Save to history
        history_entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "benchmark": request.benchmark,
            "total_evaluated": metadata.get("total_evaluated", 0),
            "accuracy": accuracy,
            "summary": {k: v for k, v in summary.items() if isinstance(v, (int, float, str, bool))},
        }
        _benchmark_history.insert(0, history_entry)
        # Keep last 50 runs
        if len(_benchmark_history) > 50:
            _benchmark_history[:] = _benchmark_history[:50]
        _save_benchmark_history()

        # Save ragbench evals to disk
        if request.benchmark == "ragbench":
            save_report(report, str(PROJECT_ROOT / "eval_results"), run_type=request.run_type)  # type: ignore[arg-type]

        return BenchmarkEvalResponse(
            benchmark=request.benchmark,
            summary=summary,
            results=results,
            metadata=metadata,
        )

    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Benchmark module not available: {e}") from e
    except Exception as e:
        logger.error("Benchmark evaluation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {e}") from e
    finally:
        _benchmark_running = False


@app.get("/api/eval/benchmark/status")
async def benchmark_status():
    """Check if a benchmark evaluation is currently running."""
    return {"running": _benchmark_running}


@app.get("/api/eval/benchmark/history", response_model=BenchmarkHistoryResponse)
async def benchmark_history():
    """Return past benchmark evaluation runs."""
    return BenchmarkHistoryResponse(runs=[BenchmarkHistoryEntry(**h) for h in _benchmark_history])


EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results"
EVAL_PRODUCTION_DIR = EVAL_RESULTS_DIR / "production"
EVAL_MANUAL_DIR = EVAL_RESULTS_DIR / "manual"


def _find_latest_eval(prefer_production: bool = True):
    """Find the most recent eval JSON file.

    When *prefer_production* is True and a production result exists with the
    same timestamp as a manual result, the production file wins.  Otherwise
    the globally most-recent file (by mtime) is returned regardless of
    directory.
    """
    candidates = []
    for d in (EVAL_PRODUCTION_DIR, EVAL_MANUAL_DIR, EVAL_RESULTS_DIR):
        if not d.exists():
            continue
        candidates.extend(f for f in d.glob("eval_*.json") if f.parent == d)

    if not candidates:
        return None

    # Sort by mtime; on tie, production wins when preferred
    def _sort_key(p):
        is_prod = 1 if (prefer_production and p.parent == EVAL_PRODUCTION_DIR) else 0
        return (p.stat().st_mtime, is_prod)

    return max(candidates, key=_sort_key)


def _collect_eval_files(run_type: str = "all") -> list:
    """Collect eval JSON files filtered by run_type ('production', 'manual', or 'all')."""
    files = []
    if run_type in ("production", "all") and EVAL_PRODUCTION_DIR.exists():
        files.extend(EVAL_PRODUCTION_DIR.glob("eval_*.json"))
    if run_type in ("manual", "all") and EVAL_MANUAL_DIR.exists():
        files.extend(EVAL_MANUAL_DIR.glob("eval_*.json"))
    # Include legacy root-level files (backward compat)
    if EVAL_RESULTS_DIR.exists():
        files.extend(f for f in EVAL_RESULTS_DIR.glob("eval_*.json") if f.parent == EVAL_RESULTS_DIR)
    return sorted(files, key=lambda p: p.name)


@app.get("/api/eval/benchmark/latest/{benchmark}")
async def benchmark_latest(benchmark: str):
    """Return the latest saved evaluation results for a benchmark."""
    if benchmark == "ragbench":
        latest_file = _find_latest_eval(prefer_production=True)
        if not latest_file:
            raise HTTPException(status_code=404, detail="No RAG-Bench evaluation results found")
        try:
            with open(latest_file) as f:
                data = json.load(f)
            data["_source_file"] = latest_file.name
            data["_run_type"] = data.get("metadata", {}).get("run_type", "manual")
            return data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load results: {e}") from e

    elif benchmark == "ragtruth":
        # Find the latest ragtruth entry in benchmark history
        for entry in _benchmark_history:
            if entry.get("benchmark") == "ragtruth" and entry.get("summary"):
                return entry
        raise HTTPException(status_code=404, detail="No RAGTruth evaluation results found")

    else:
        raise HTTPException(status_code=400, detail="Benchmark must be 'ragbench' or 'ragtruth'")


@app.get("/api/eval/benchmark/examples")
async def benchmark_examples():
    """Return benchmark test examples that users can try individually."""
    entries = get_benchmark()
    examples = []
    for e in entries:
        examples.append(
            {
                "id": e.id,
                "question": e.question,
                "expected_sources": e.expected_sources,
                "expected_answer_contains": e.expected_answer_contains,
                "query_type": e.query_type,
                "topic": e.topic,
                "difficulty": e.difficulty,
                "should_deflect": e.should_deflect,
            }
        )
    return {"examples": examples}


@app.get("/api/eval/ragtruth/examples")
async def ragtruth_examples(sample: int = 20, seed: int = 42):
    """Return RAGTruth examples for the Try-It panel.

    Returns a mix of hallucinated and clean entries from the cached dataset.
    """
    try:
        entries = load_ragtruth(sample_size=0, task_type="QA", seed=seed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load RAGTruth: {e}") from e

    # Split into hallucinated vs clean based on metadata labels
    with_hallu = []
    clean = []
    for e in entries:
        labels = e.metadata.get("labels", [])
        if labels:
            with_hallu.append(e)
        else:
            clean.append(e)

    # Sample a balanced set
    rng = random.Random(seed)
    half = sample // 2
    sampled_hallu = rng.sample(with_hallu, min(half, len(with_hallu)))
    sampled_clean = rng.sample(clean, min(sample - len(sampled_hallu), len(clean)))
    sampled = sampled_hallu + sampled_clean
    rng.shuffle(sampled)

    examples = []
    for e in sampled:
        labels = e.metadata.get("labels", [])
        examples.append(
            {
                "id": e.id,
                "prompt": e.prompt,
                "context": e.source_info[:800],  # truncate for transport
                "response": e.reference_response,
                "has_hallucination": bool(labels),
                "hallucination_spans": [
                    {"text": lbl.get("text", ""), "label_type": lbl.get("label_type", "")} for lbl in labels
                ],
            }
        )
    return {"examples": examples, "total_with_hallucinations": len(with_hallu), "total_clean": len(clean)}


@app.post("/api/eval/ragtruth/detect")
async def ragtruth_detect(body: dict):
    """Run hallucination detection on a single context+response pair.

    Expects: { "context": "...", "response": "..." }
    Returns detection results from the heuristic detector.
    """
    context = body.get("context", "")
    response = body.get("response", "")
    if not context or not response:
        raise HTTPException(status_code=400, detail="Both 'context' and 'response' are required")

    runner = RAGTruthRunner(generator=None)
    start = time.time()
    detection = runner._detect_heuristic(response, context)
    latency = (time.time() - start) * 1000

    return {
        "has_hallucination": detection.get("has_hallucination", False),
        "flagged_spans": [
            {"text": span, "type": stype}
            for span, stype in zip(
                detection.get("spans", []),
                detection.get("span_types", []),
                strict=False,
            )
        ],
        "latency_ms": round(latency, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Eval Trends (regression tracking)
# ═══════════════════════════════════════════════════════════════════════════


@app.get("/api/eval/benchmark/trends", response_model=TrendsResponse)
async def benchmark_trends(run_type: str = "production"):
    """Return historical eval metrics for trend visualization.

    Args:
        run_type: Filter by run type — 'production' (default), 'manual', or 'all'.
    """
    if run_type not in ("production", "manual", "all"):
        raise HTTPException(status_code=400, detail="run_type must be 'production', 'manual', or 'all'")

    files = _collect_eval_files(run_type)

    # Fallback: if production requested but empty, return all (graceful migration)
    if not files and run_type == "production":
        files = _collect_eval_files("all")

    trends = []
    for path in files:
        try:
            with open(path) as f:
                data = json.load(f)
            summary = data.get("summary", {})
            meta = data.get("metadata", {})
            timestamp = meta.get("timestamp", path.stem.replace("eval_", ""))
            trends.append(
                TrendDataPoint(
                    timestamp=timestamp,
                    run_type=meta.get("run_type", "manual"),
                    retrieval_mrr=summary.get("retrieval_mrr", 0.0),
                    retrieval_ndcg_at_5=summary.get("retrieval_ndcg_at_5", 0.0),
                    retrieval_hit_rate=summary.get("retrieval_hit_rate", 0.0),
                    avg_citation_precision=summary.get("avg_citation_precision", 0.0),
                    avg_citation_recall=summary.get("avg_citation_recall", 0.0),
                    avg_completeness=summary.get("avg_completeness", 0.0),
                    avg_faithfulness=summary.get("avg_faithfulness", 0.0),
                    deflection_accuracy=summary.get("deflection_accuracy", 0.0),
                    avg_latency_ms=summary.get("avg_latency_ms", 0.0),
                    total_queries=summary.get("total_queries", 0),
                )
            )
        except Exception as e:
            logger.warning("Failed to read eval file %s: %s", path.name, e)

    return TrendsResponse(trends=trends)


# ═══════════════════════════════════════════════════════════════════════════
# Scheduled Auto-Eval
# ═══════════════════════════════════════════════════════════════════════════

_eval_schedule_enabled = EVAL_SCHEDULE_HOURS > 0
_eval_schedule_interval = EVAL_SCHEDULE_HOURS
_eval_schedule_task: asyncio.Task | None = None
_eval_schedule_last_run: str | None = None
_eval_schedule_last_summary: dict = {}


async def _scheduled_eval_loop():
    """Background loop that runs RAG-Bench evals on a schedule."""
    global _eval_schedule_last_run, _eval_schedule_last_summary

    while True:
        await asyncio.sleep(_eval_schedule_interval * 3600)

        if not _eval_schedule_enabled or not _generator or _benchmark_running:
            continue

        logger.info("Starting scheduled auto-eval")
        try:
            benchmark_entries = get_benchmark()
            rng = random.Random(42)
            sample = rng.sample(benchmark_entries, min(20, len(benchmark_entries)))

            judge = JudgeLLM(_generator.llm)
            runner = EvalRunner(
                retriever=_retriever,
                generator=_generator,
                judge=judge,
                benchmark=sample,
            )

            report = await asyncio.get_event_loop().run_in_executor(None, runner.run_all)

            eval_dir = str(PROJECT_ROOT / "eval_results")
            save_report(report, eval_dir, run_type="production")

            _eval_schedule_last_run = datetime.now().isoformat(timespec="seconds")
            _eval_schedule_last_summary = report.summary
            logger.info("Scheduled auto-eval completed: %d queries", len(report.results))

        except Exception as e:
            logger.error("Scheduled auto-eval failed: %s", e, exc_info=True)


def _start_eval_schedule():
    """Start the eval schedule background task if not already running."""
    global _eval_schedule_task
    if _eval_schedule_task is None or _eval_schedule_task.done():
        _eval_schedule_task = asyncio.create_task(_scheduled_eval_loop())


@app.get("/api/eval/schedule", response_model=EvalScheduleStatus)
async def eval_schedule_status():
    """Return the current auto-eval schedule configuration."""
    next_run = None
    if _eval_schedule_enabled and _eval_schedule_last_run:
        try:
            last = datetime.fromisoformat(_eval_schedule_last_run)
            next_run = (last + timedelta(hours=_eval_schedule_interval)).isoformat(timespec="seconds")
        except ValueError:
            pass

    return EvalScheduleStatus(
        enabled=_eval_schedule_enabled,
        interval_hours=_eval_schedule_interval,
        next_run=next_run,
        last_run=_eval_schedule_last_run,
        last_run_summary={k: v for k, v in _eval_schedule_last_summary.items() if isinstance(v, (int, float))},
    )


@app.post("/api/eval/schedule", response_model=EvalScheduleStatus)
async def eval_schedule_configure(request: EvalScheduleRequest):
    """Enable or disable the auto-eval schedule."""
    global _eval_schedule_enabled, _eval_schedule_interval

    _eval_schedule_enabled = request.enabled
    _eval_schedule_interval = request.interval_hours

    if _eval_schedule_enabled:
        _start_eval_schedule()

    return await eval_schedule_status()


FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
FRONTEND_HTML = FRONTEND_DIST / "index.html"

# ═══════════════════════════════════════════════════════════════════════════
# Knowledge Graph visualization
# ═══════════════════════════════════════════════════════════════════════════

_graph_store = None
_graph_retriever = None


def _get_graph_retriever():
    """Lazy-init the graph retriever (only when Neo4j is available)."""
    global _graph_store, _graph_retriever
    if _graph_retriever is not None:
        return _graph_retriever
    try:
        neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        config = GraphStoreConfig(uri=neo4j_uri)
        _graph_store = GraphStore(config=config)
        _graph_retriever = GraphRetriever(store=_graph_store)
        logger.info(f"Graph retriever initialized ({neo4j_uri})")
        return _graph_retriever
    except Exception as e:
        logger.debug(f"Graph retriever unavailable: {e}")
        return None


@app.get("/api/graph/context", response_model=GraphSubgraph)
async def get_graph_context(question: str):
    """Return the local knowledge graph subgraph for a query.

    Matches entities in the question, fetches their 1-hop neighborhood,
    and returns nodes + edges for frontend visualization.
    """
    retriever = _get_graph_retriever()
    if retriever is None:
        return GraphSubgraph()

    try:
        loop = asyncio.get_event_loop()

        # Match entities from the question
        matched = await loop.run_in_executor(None, retriever._match_entities, question)
        if not matched:
            return GraphSubgraph()

        # Build the subgraph: nodes + edges
        nodes_by_id: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []
        matched_names = []

        for entity in matched[:5]:  # Cap at 5 entities
            name = entity["name"]
            name_lower = entity["name_lower"]
            matched_names.append(name)

            # Add matched entity as a node
            nodes_by_id[name_lower] = GraphNode(
                id=name_lower,
                name=name,
                entity_type=entity["entity_type"],
                matched=True,
            )

            # Fetch triples involving this entity
            triples = await loop.run_in_executor(None, retriever.store.get_entity_triples, name, 30)

            for t in triples:
                s_id = t["subject"].lower()
                o_id = t["object"].lower()

                # Add subject node if new
                if s_id not in nodes_by_id:
                    nodes_by_id[s_id] = GraphNode(
                        id=s_id,
                        name=t["subject"],
                        entity_type=t["subject_type"],
                        matched=False,
                    )

                # Add object node if new
                if o_id not in nodes_by_id:
                    nodes_by_id[o_id] = GraphNode(
                        id=o_id,
                        name=t["object"],
                        entity_type=t["object_type"],
                        matched=False,
                    )

                # Add edge (deduplicate by source+target+predicate)
                edge_key = (s_id, o_id, t["predicate"])
                if not any((e.source, e.target, e.predicate) == edge_key for e in edges):
                    edges.append(
                        GraphEdge(
                            source=s_id,
                            target=o_id,
                            predicate=t["predicate"],
                            weight=t["weight"],
                        )
                    )

        return GraphSubgraph(
            nodes=list(nodes_by_id.values()),
            edges=edges,
            matched_entities=matched_names,
        )

    except Exception as e:
        logger.warning(f"Graph context error: {e}")
        return GraphSubgraph()


# Serve built frontend assets (JS, CSS) from dist/assets/
if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="frontend-assets")


@app.get("/sw.js", response_class=FileResponse)
async def serve_service_worker():
    """Serve the service worker from dist root."""
    sw_path = FRONTEND_DIST / "sw.js"
    if not sw_path.exists():
        raise HTTPException(status_code=404, detail="Service worker not found")
    return FileResponse(sw_path, media_type="application/javascript")


@app.get("/favicon.svg", response_class=FileResponse)
async def serve_favicon():
    """Serve the favicon from dist root."""
    favicon_path = FRONTEND_DIST / "favicon.svg"
    if not favicon_path.exists():
        raise HTTPException(status_code=404, detail="Favicon not found")
    return FileResponse(favicon_path, media_type="image/svg+xml")


@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """Serve the React frontend."""
    if not FRONTEND_HTML.exists():
        raise HTTPException(status_code=404, detail="Frontend not built. Run: cd frontend && npm run build")
    return FileResponse(FRONTEND_HTML, media_type="text/html")


def run():
    """Entry point for `rag-serve` CLI command."""
    port = int(os.environ.get("RAG_API_PORT", 8001))
    # Disable reload in Docker containers to prevent file-watcher infinite loops
    # during model initialization. Enable only for local development if needed.
    reload = os.environ.get("RELOAD", "").lower() in ("true", "1", "yes")
    log_level = os.environ.get("LOG_LEVEL", "info")

    uvicorn.run(
        "rag_bench.api.server:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
        log_level=log_level,
    )


if __name__ == "__main__":
    run()
