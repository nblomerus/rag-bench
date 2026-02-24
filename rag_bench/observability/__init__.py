"""Observability package — structured logging, Prometheus metrics, and request tracking."""

from rag_bench.observability.logging import get_logger, setup_logging
from rag_bench.observability.metrics import (
    ACTIVE_REQUESTS,
    BUILD_INFO,
    CITATION_COVERAGE,
    CORPUS_CHUNKS,
    CORPUS_PAPERS,
    GENERATION_DURATION,
    PIPELINE_READY,
    QUERIES_TOTAL,
    REQUEST_DURATION,
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
]
