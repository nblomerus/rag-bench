"""Observability package — structured logging, Prometheus metrics, and request tracking."""

from rag_bench.observability.logging import get_logger, setup_logging
from rag_bench.observability.metrics import (
    ACTIVE_REQUESTS,
    BUILD_INFO,
    CITATION_COVERAGE,
    CONCURRENT_USERS,
    CORPUS_CHUNKS,
    CORPUS_PAPERS,
    GENERATION_DURATION,
    PIPELINE_READY,
    QUERIES_REJECTED,
    QUERIES_TOTAL,
    QUERY_CAPACITY,
    QUEUED_QUERIES,
    REQUEST_DURATION,
    REQUESTS_BY_ENDPOINT,
    RERANKING_DURATION,
    RETRIEVAL_DURATION,
    RETRIEVAL_TOP_SCORE,
    UNIQUE_USERS,
)
from rag_bench.observability.tracker import RequestTracker

__all__ = [
    "setup_logging",
    "get_logger",
    "RequestTracker",
    "QUERIES_TOTAL",
    "REQUEST_DURATION",
    "RETRIEVAL_DURATION",
    "GENERATION_DURATION",
    "RERANKING_DURATION",
    "RETRIEVAL_TOP_SCORE",
    "CITATION_COVERAGE",
    "ACTIVE_REQUESTS",
    "PIPELINE_READY",
    "CORPUS_CHUNKS",
    "CORPUS_PAPERS",
    "UNIQUE_USERS",
    "BUILD_INFO",
    "QUEUED_QUERIES",
    "QUERY_CAPACITY",
    "QUERIES_REJECTED",
    "REQUESTS_BY_ENDPOINT",
    "CONCURRENT_USERS",
]
