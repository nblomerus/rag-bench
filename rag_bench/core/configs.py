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
    """Configuration for document chunking.

    Shared parameters (min_section_length, min_chunk_length) live here.
    Strategy-specific parameters live in ``strategy_config`` and are
    passed to the strategy constructor as keyword arguments.

    For backward compatibility, ``chunk_size`` and ``chunk_overlap`` are
    still accepted — they are forwarded into ``strategy_config`` for the
    default "recursive" strategy if ``strategy_config`` is empty.
    """

    strategy: str = "recursive"
    min_section_length: int = 50
    min_chunk_length: int = 100
    strategy_config: dict = field(default_factory=dict)

    # Legacy fields — used to populate strategy_config for "recursive"
    chunk_size: int = 1024
    chunk_overlap: int = 128

    def __post_init__(self):
        # Backward compat: if no explicit strategy_config and using
        # recursive, populate from the legacy fields.
        if not self.strategy_config and self.strategy == "recursive":
            self.strategy_config = {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            }


@dataclass
class EnricherConfig:
    """Configuration for contextual header enrichment.

    When ``enabled`` is False (default), the enrichment step is skipped
    entirely — zero overhead.  Set to True and point ``ollama_model`` at
    a running Ollama instance to generate contextual headers.
    """

    enabled: bool = False
    ollama_model: str = "qwen2.5:14b-instruct-q4_K_M"
    ollama_base_url: str = "http://localhost:11434"
    cache_dir: str = ".enricher_cache"
    max_context_tokens: int = 30_000
    request_timeout: int = 120
    batch_log_interval: int = 25


@dataclass
class ExtractorConfig:
    """Configuration for LLM-based entity/relation extraction (GraphRAG).

    The extractor calls Ollama to extract (entity, relation, entity) triples
    from each chunk.  Results are cached on disk (one JSON file per paper)
    so re-runs skip already-processed chunks.
    """

    enabled: bool = False
    ollama_model: str = "qwen2.5:14b-instruct-q4_K_M"
    ollama_base_url: str = "http://localhost:11434"
    cache_dir: str = ".extractor_cache"
    max_triples_per_chunk: int = 10  # cap to avoid noisy over-extraction
    request_timeout: int = 120
    batch_log_interval: int = 25


@dataclass
class GraphStoreConfig:
    """Configuration for Neo4j graph storage (GraphRAG).

    Credentials default to the docker-compose.yml values.
    ``batch_size`` controls how many triples are written per Cypher
    transaction — larger batches are faster but use more memory.
    """

    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "ragbench2024"
    database: str = "neo4j"  # Community Edition only has "neo4j"
    batch_size: int = 500


@dataclass
class CRAGConfig:
    """Configuration for Corrective RAG (CRAG) retrieval wrapper.

    Thresholds calibrated from benchmark rerank score distribution:
    - Scores are sigmoid-normalized [0, 1]
    - P25 = 0.97, clear drop below 0.85, strong drop below 0.70
    """

    enabled: bool = False
    correct_threshold: float = 0.90
    ambiguous_threshold: float = 0.70
    refinement_floor: float = 0.30
    hyde_enabled: bool = True
    ollama_model: str = "qwen2.5:14b-instruct-q4_K_M"
    ollama_base_url: str = "http://localhost:11434"
    hyde_max_tokens: int = 256
    hyde_temperature: float = 0.3
    max_rewrites: int = 1
    merge_strategy: str = "interleave"


@dataclass
class AgentConfig:
    """Configuration for the agentic RAG orchestrator.

    The agent wraps retrieval + generation in a plan-execute-evaluate loop.
    Rule-based classification routes simple queries directly; multi-hop
    queries get LLM-based decomposition.
    """

    enabled: bool = False
    max_iterations: int = 3
    sufficiency_threshold: float = 0.85
    min_results_for_answer: int = 2
    ollama_model: str = "qwen2.5:14b-instruct-q4_K_M"
    ollama_base_url: str = "http://localhost:11434"


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
    enricher: EnricherConfig = field(default_factory=EnricherConfig)
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    graph_store: GraphStoreConfig = field(default_factory=GraphStoreConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    crag: CRAGConfig = field(default_factory=CRAGConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    # -- serialization helpers ------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PipelineConfig:
        return cls(
            name=data.get("name", "default"),
            chunker=ChunkerConfig(**data["chunker"]) if "chunker" in data else ChunkerConfig(),
            enricher=EnricherConfig(**data["enricher"]) if "enricher" in data else EnricherConfig(),
            extractor=ExtractorConfig(**data["extractor"]) if "extractor" in data else ExtractorConfig(),
            graph_store=GraphStoreConfig(**data["graph_store"]) if "graph_store" in data else GraphStoreConfig(),
            embedder=EmbedderConfig(**data["embedder"]) if "embedder" in data else EmbedderConfig(),
            retriever=RetrieverConfig(**data["retriever"]) if "retriever" in data else RetrieverConfig(),
            generator=GeneratorConfig(**data["generator"]) if "generator" in data else GeneratorConfig(),
            crag=CRAGConfig(**data["crag"]) if "crag" in data else CRAGConfig(),
            agent=AgentConfig(**data["agent"]) if "agent" in data else AgentConfig(),
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
