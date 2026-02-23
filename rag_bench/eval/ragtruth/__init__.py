"""RAGTruth benchmark evaluation — Hallucination detection for RAG systems."""

from .loader import RAGTruthEntry, load_ragtruth
from .metrics import case_level_accuracy, compute_hallucination_rate, hallucination_by_type, span_level_f1
from .runner import RAGTruthReport, RAGTruthRunner

__all__ = [
    "load_ragtruth",
    "RAGTruthEntry",
    "compute_hallucination_rate",
    "span_level_f1",
    "case_level_accuracy",
    "hallucination_by_type",
    "RAGTruthRunner",
    "RAGTruthReport",
]
