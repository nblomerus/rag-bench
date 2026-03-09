"""
configs.py — Typed configuration dataclasses for each pipeline component.

Defaults mirror the existing values in rag_bench/config.py for backward
compatibility.  A top-level PipelineConfig bundles them all and supports
JSON serialization so experiment configs can be saved alongside results.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Per-component configs
# ---------------------------------------------------------------------------


@dataclass
class ChunkerConfig:
    """Configuration for document chunking."""

    chunk_size: int = 1024
    chunk_overlap: int = 128
    min_section_length: int = 50
    min_chunk_length: int = 100

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})")


@dataclass
class EmbedderConfig:
    """Configuration for embedding and vector store indexing."""

    model_name: str = "BAAI/bge-base-en-v1.5"
    chroma_path: str = "./chroma_db"
    collection_name: str = "ai_ml_papers"
    distance_metric: str = "cosine"
    batch_size: int = 254


@dataclass
class RetrieverConfig:
    """Configuration for hybrid retrieval."""

    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    chroma_path: str = "./chroma_db"
    collection_name: str = "ai_ml_papers"
    bm25_weight: float = 0.4
    dense_weight: float = 0.6
    first_stage_k: int = 500
    rerank_candidates: int = 200

    def __post_init__(self):
        if abs(self.bm25_weight + self.dense_weight - 1.0) > 0.01:
            raise ValueError(f"bm25_weight ({self.bm25_weight}) + dense_weight ({self.dense_weight}) must sum to 1.0")


@dataclass
class GeneratorConfig:
    """Configuration for answer generation."""

    llm_backend: str = "template"  # "ollama" | "openai" | "template"
    llm_model: str = "gemma2:27b"
    llm_base_url: str = "http://localhost:11434"
    top_k: int = 10
    relevance_threshold: float = 0.3
    enable_citation_boost: bool = True
    system_prompt: str = ""  # empty = use default from generator.py


# ---------------------------------------------------------------------------
# Top-level pipeline config
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Full pipeline configuration.  Serializable to/from JSON."""

    name: str = "default"
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)

    # -- serialization helpers ------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PipelineConfig:
        return cls(
            name=data.get("name", "default"),
            chunker=ChunkerConfig(**data["chunker"]) if "chunker" in data else ChunkerConfig(),
            embedder=EmbedderConfig(**data["embedder"]) if "embedder" in data else EmbedderConfig(),
            retriever=RetrieverConfig(**data["retriever"]) if "retriever" in data else RetrieverConfig(),
            generator=GeneratorConfig(**data["generator"]) if "generator" in data else GeneratorConfig(),
        )

    def save(self, path: Path | str) -> None:
        """Write config as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path | str) -> PipelineConfig:
        """Read config from JSON."""
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    @classmethod
    def from_env(cls) -> PipelineConfig:
        """Build a config from environment variables (matches current server.py behaviour)."""
        return cls(
            name="env",
            retriever=RetrieverConfig(
                chroma_path=os.environ.get("RAG_CHROMA_DIR", "./chroma_db"),
                embedding_model=os.environ.get("RAG_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"),
                reranker_model=os.environ.get("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            ),
            embedder=EmbedderConfig(
                chroma_path=os.environ.get("RAG_CHROMA_DIR", "./chroma_db"),
                model_name=os.environ.get("RAG_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"),
            ),
            generator=GeneratorConfig(
                llm_backend=os.environ.get("RAG_LLM_BACKEND", "ollama"),
                llm_model=os.environ.get("RAG_LLM_MODEL", "gemma2:27b"),
                llm_base_url=os.environ.get("RAG_LLM_BASE_URL", "http://localhost:11434"),
            ),
        )
