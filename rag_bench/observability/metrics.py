"""Prometheus metric definitions for RAG-Bench.

All metrics are defined here and imported by modules that need to observe them.
The /metrics endpoint is exposed via prometheus-fastapi-instrumentator in server.py.
"""

from prometheus_client import Counter, Gauge, Histogram, Info

# ── Counters ──
QUERIES_TOTAL = Counter(
    "rag_queries_total",
    "Total RAG queries processed",
    ["status"],  # success | deflected | error
)

# ── Histograms ──
REQUEST_DURATION = Histogram(
    "rag_request_duration_seconds",
    "End-to-end request duration",
    ["endpoint", "method"],
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)

RETRIEVAL_DURATION = Histogram(
    "rag_retrieval_duration_seconds",
    "Hybrid retrieval pipeline duration",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

GENERATION_DURATION = Histogram(
    "rag_generation_duration_seconds",
    "LLM generation duration",
    buckets=(0.5, 1, 2.5, 5, 10, 30, 60),
)

RERANKING_DURATION = Histogram(
    "rag_reranking_duration_seconds",
    "Cross-encoder reranking duration",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2),
)

RETRIEVAL_TOP_SCORE = Histogram(
    "rag_retrieval_top_score",
    "Distribution of top retrieval scores",
    buckets=(0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 3, 5, 8),
)

CITATION_COVERAGE = Histogram(
    "rag_citation_coverage",
    "Distribution of citation coverage ratios",
    buckets=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
)

# ── Gauges ──
ACTIVE_REQUESTS = Gauge(
    "rag_active_requests",
    "Number of in-flight requests",
)

PIPELINE_READY = Gauge(
    "rag_pipeline_ready",
    "Whether the RAG pipeline is loaded (1=ready, 0=loading)",
)

CORPUS_CHUNKS = Gauge(
    "rag_corpus_chunks_total",
    "Total chunks in ChromaDB",
)

CORPUS_PAPERS = Gauge(
    "rag_corpus_papers_total",
    "Total unique papers in corpus",
)

UNIQUE_USERS = Gauge(
    "rag_unique_users",
    "Unique IP addresses seen since startup",
)

# ── Info ──
BUILD_INFO = Info(
    "rag_build",
    "Build and configuration information",
)
