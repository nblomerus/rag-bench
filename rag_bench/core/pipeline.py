"""
pipeline.py — Composable RAG pipeline and factory function.

Assembles chunker, embedder, retriever, and generator from a
PipelineConfig so you can swap any component or compare configurations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rag_bench.core.chunker import PaperChunker
from rag_bench.core.citation_boost import CitationBooster
from rag_bench.core.configs import PipelineConfig
from rag_bench.core.embedder import Embedder as BGEEmbedder
from rag_bench.core.generator import (
    RAGGenerator,
    RelevanceGate,
    build_llm_backend,
)
from rag_bench.core.retriever import HybridRetriever
from rag_bench.core.types import GenerationResult

logger = logging.getLogger(__name__)


@dataclass
class RAGPipeline:
    """Assembled pipeline with all components.

    Holds references to every stage so callers can use the full pipeline
    via ``query()`` or access individual components directly.
    """

    config: PipelineConfig
    chunker: PaperChunker
    embedder: BGEEmbedder
    retriever: HybridRetriever
    generator: RAGGenerator

    def query(self, question: str, top_k: int | None = None) -> GenerationResult:
        """Run the full pipeline: retrieve → generate."""
        k = top_k or self.config.generator.top_k
        results = self.retriever.retrieve(question, top_k=k)
        return self.generator.generate(question, context=results)


def build_pipeline(config: PipelineConfig | None = None) -> RAGPipeline:
    """Factory: assemble a complete RAG pipeline from a config.

    If *config* is ``None`` a default ``PipelineConfig()`` is used, which
    mirrors the values in ``rag_bench/config.py``.
    """
    if config is None:
        config = PipelineConfig()

    logger.info(f"Building pipeline '{config.name}'")

    chunker = PaperChunker(config=config.chunker)

    embedder = BGEEmbedder(config=config.embedder)

    retriever = HybridRetriever(config=config.retriever)

    llm = build_llm_backend(
        config.generator.llm_backend,
        config.generator.llm_model,
        config.generator.llm_base_url,
    )
    gate = RelevanceGate(min_top_score=config.generator.relevance_threshold)
    booster = CitationBooster() if config.generator.enable_citation_boost else None

    generator = RAGGenerator(
        retriever=retriever,
        llm_backend=llm,
        relevance_gate=gate,
        top_k=config.generator.top_k,
        citation_booster=booster,
        system_prompt=config.generator.system_prompt or None,
    )

    logger.info(f"Pipeline '{config.name}' ready")
    return RAGPipeline(
        config=config,
        chunker=chunker,
        embedder=embedder,
        retriever=retriever,
        generator=generator,
    )
