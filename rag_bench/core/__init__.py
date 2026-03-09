"""rag_bench.core — Public API for the modular RAG pipeline."""

# Data types
# Config dataclasses
from rag_bench.core.configs import (
    ChunkerConfig,
    EmbedderConfig,
    GeneratorConfig,
    PipelineConfig,
    RetrieverConfig,
)

# Pipeline assembly
from rag_bench.core.pipeline import RAGPipeline, build_pipeline

# Protocols (interfaces)
from rag_bench.core.protocols import (
    Chunker,
    Embedder,
    Generator,
    LLMBackend,
    Reranker,
    Retriever,
)
from rag_bench.core.types import ChunkData, GenerationResult, RetrievalResult

__all__ = [
    # Types
    "ChunkData",
    "RetrievalResult",
    "GenerationResult",
    # Protocols
    "Chunker",
    "Embedder",
    "Retriever",
    "Reranker",
    "Generator",
    "LLMBackend",
    # Configs
    "ChunkerConfig",
    "EmbedderConfig",
    "RetrieverConfig",
    "GeneratorConfig",
    "PipelineConfig",
    # Pipeline
    "RAGPipeline",
    "build_pipeline",
]
