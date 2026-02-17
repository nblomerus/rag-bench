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
import logging
import os
import time
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from rag_bench.api.schemas import (
    EvalRequest,
    EvalResponse,
    EvalResult,
    EvalSummary,
    FullEvalRequest,
    FullEvalResponse,
    FullEvalResult,
    PaperChunk,
    PaperDetail,
    PaperSummary,
    QualityMetrics,
    QueryRequest,
    QueryResponse,
    SourceResult,
    StatsResponse,
)
from rag_bench.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    PROJECT_ROOT,
    RERANKER_MODEL,
)
from rag_bench.core.citation_boost import CitationBooster
from rag_bench.core.generator import (
    DEFLECTION_PHRASES,
    RAGGenerator,
    RelevanceGate,
    build_llm_backend,
)
from rag_bench.core.retriever import HybridRetriever
from rag_bench.utils.text import fix_encoding

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ═══════════════════════════════════════════════════════════════════════════
# Configuration from environment
# ═══════════════════════════════════════════════════════════════════════════
LLM_BACKEND = os.environ.get("RAG_LLM_BACKEND", "ollama")
LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "mistral:7b-instruct-q4_K_M")
LLM_BASE_URL = os.environ.get("RAG_LLM_BASE_URL", "http://localhost:11434")
RERANKER_MODEL_OVERRIDE = os.environ.get("RAG_RERANKER_MODEL", RERANKER_MODEL)


# ═══════════════════════════════════════════════════════════════════════════
# Application setup
# ═══════════════════════════════════════════════════════════════════════════

# Global pipeline instances (loaded once at startup)
_retriever: HybridRetriever | None = None
_generator: RAGGenerator | None = None
_citation_booster: CitationBooster | None = None
_llm_backend_name: str = ""
_llm_model_name: str = ""
_total_papers: int = 0  # Updated by background task
_papers_counting: bool = True  # False once the full scan completes


async def _count_papers_background():
    """Scan all ChromaDB chunks to get the exact unique paper count. Runs after startup."""
    global _total_papers, _papers_counting
    if not _retriever:
        return

    collection = _retriever.collection
    total_chunks = collection.count()
    batch_size = 50_000
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
    logger.info(f"Paper count complete: {_total_papers:,} unique papers across {total_chunks:,} chunks")


@asynccontextmanager
async def lifespan(app):
    """Load the RAG pipeline at startup, clean up on shutdown."""
    global _retriever, _generator, _citation_booster, _llm_backend_name, _llm_model_name, _total_papers, _papers_counting

    logger.info("Loading RAG pipeline...")
    start = time.time()

    _retriever = HybridRetriever(
        embedding_model=EMBEDDING_MODEL,
        reranker_model=RERANKER_MODEL_OVERRIDE,
        chroma_path=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    # Initialize citation booster for improved citation quality
    _citation_booster = CitationBooster(
        enable_age_boost=True,
        enable_query_adaptive=True,
    )
    logger.info("Citation booster initialized with %d foundational papers", len(_citation_booster.foundational_papers))

    _llm_backend_name = LLM_BACKEND
    _llm_model_name = LLM_MODEL

    llm = build_llm_backend(LLM_BACKEND, LLM_MODEL, LLM_BASE_URL)
    gate = RelevanceGate()

    _generator = RAGGenerator(
        retriever=_retriever,
        llm_backend=llm,
        relevance_gate=gate,
        top_k=DEFAULT_TOP_K,
        citation_booster=_citation_booster,
    )

    total_chunks = _retriever.collection.count()
    logger.info(f"Corpus: {total_chunks:,} chunks — starting background paper count")
    asyncio.create_task(_count_papers_background())

    elapsed = time.time() - start
    logger.info(f"RAG pipeline loaded in {elapsed:.1f}s")

    yield  # App runs here

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
                text_preview=fix_encoding(r.get("text", ""))[:300],
                paper_id=meta.get("paper_id", meta.get("doc_id", meta.get("arxiv_id", ""))),
                chunk_id=r.get("chunk_id", ""),
                relevance=relevance,
            )
        )
    return sources


# ═══════════════════════════════════════════════════════════════════════════
# Per-response quality metrics (fast, deterministic — no LLM calls)
# ═══════════════════════════════════════════════════════════════════════════


def _compute_quality_metrics(answer: str, all_results: list[dict], filtered_results: list[dict]) -> QualityMetrics:
    """Compute inline quality metrics for a single query response."""
    from rag_bench.eval.metrics import (
        citation_density,
        count_unsupported_claims,
        extract_cited_source_numbers,
        source_coverage,
    )

    sources_for_metrics = filtered_results if filtered_results else all_results[:5]
    num_provided = len(sources_for_metrics)
    cited = extract_cited_source_numbers(answer)

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
    )


def _compute_retrieval_confidence(results: list[dict]) -> str:
    """Classify retrieval confidence as high/medium/low."""
    if not results:
        return "low"
    top_score = results[0].get("score", 0.0)
    above_half = sum(1 for r in results if r.get("score", 0.0) >= top_score / 2)

    # Cross-encoder scores are typically > 1.0, cosine < 1.0
    if top_score > 2.0:  # cross-encoder range
        if top_score > 6.0 and above_half >= 3:
            return "high"
        elif top_score > 3.0 or above_half >= 1:
            return "medium"
    else:  # cosine range
        if top_score > 0.7 and above_half >= 3:
            return "high"
        elif top_score > 0.5 or above_half >= 1:
            return "medium"
    return "low"


def _compute_score_spread(results: list[dict]) -> dict:
    """Return min/max/mean/std of retrieval scores."""
    if not results:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    scores = [r.get("score", 0.0) for r in results]
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    return {
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "mean": round(mean, 4),
        "std": round(variance**0.5, 4),
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
    """For each source, check if it was cited in the answer and how many times."""
    import re

    per_source = []
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        count = len(re.findall(rf"\[Source\s+{i}\]", answer))
        per_source.append(
            {
                "source_number": i,
                "paper_id": meta.get("paper_id") or meta.get("arxiv_id") or meta.get("doc_id", ""),
                "title": meta.get("title", "Unknown"),
                "cited": count > 0,
                "citation_count": count,
                "score": round(r.get("score", 0.0), 4),
            }
        )
    return per_source


# ═══════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/health")
async def health_check():
    """Health check — is the pipeline loaded?"""
    return {
        "status": "ok" if _generator else "loading",
        "pipeline_ready": _generator is not None,
    }


@app.post("/api/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Ask a question and get a grounded answer with citations."""
    if not _generator:
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

    start = time.time()
    result = generator.answer(request.question, top_k=request.top_k)
    latency = (time.time() - start) * 1000

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
async def query_rag_stream(request: QueryRequest):
    """Stream the RAG answer via Server-Sent Events (SSE)."""
    if not _generator:
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

            while True:
                evt = await loop.run_in_executor(None, _next_or_stop, gen)
                if evt is _STOP:
                    break

                if evt["event"] == "sources":
                    display_results = evt.get("filtered_results", evt.get("results", []))
                    sources = _format_sources(display_results)
                    yield f"data: {json.dumps({'event': 'sources', 'sources': [s.model_dump() for s in sources]})}\n\n"

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
                    return

                elif evt["event"] == "token":
                    yield f"data: {json.dumps({'event': 'token', 'token': evt['token']})}\n\n"

                elif evt["event"] == "done":
                    latency = (time.time() - start) * 1000
                    completion_msg = {
                        "event": "done",
                        "answer": evt["answer"],
                        "deflected": False,
                        "latency_ms": round(latency, 1),
                        "backend": _llm_backend_name,
                        "model": _llm_model_name,
                    }
                    yield f"data: {json.dumps(completion_msg)}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"

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
    if not _retriever:
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
    if not _generator:
        raise HTTPException(status_code=503, detail="Pipeline still loading")

    eval_path = PROJECT_ROOT / "eval" / "eval_queries.json"
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
    if not _generator or not _retriever:
        raise HTTPException(status_code=503, detail="Pipeline still loading")

    from rag_bench.eval.judge import JudgeLLM
    from rag_bench.eval.runner import EvalRunner

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
    if not _retriever:
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
    if not _retriever:
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
    if not _retriever:
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
FRONTEND_HTML = PROJECT_ROOT / "frontend" / "index.html"


@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """Serve the React frontend."""
    if not FRONTEND_HTML.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
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
