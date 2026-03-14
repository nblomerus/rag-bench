"""
protocols.py — Interface definitions for each RAG pipeline stage.

Uses Python Protocols (structural subtyping) so existing classes
satisfy the interface without inheriting from anything.
"""

from typing import Protocol, runtime_checkable

import numpy as np

from rag_bench.core.types import ChunkData, GenerationResult, RetrievalResult


@runtime_checkable
class ChunkingStrategy(Protocol):
    """Low-level text splitting strategy.

    Accepts a plain text string and returns a list of text segments.
    PaperChunker delegates the actual splitting to a ChunkingStrategy
    while owning all pre/post-processing (equation protection, acronym
    expansion, metadata assembly, etc.).
    """

    def split_text(self, text: str) -> list[str]: ...


@runtime_checkable
class Chunker(Protocol):
    """Splits a parsed paper into indexable chunks."""

    def chunk_paper(self, doc: dict) -> list[ChunkData]: ...


@runtime_checkable
class Embedder(Protocol):
    """Embeds text and indexes chunks into a vector store."""

    def embed_texts(self, texts: list[str], **kwargs) -> np.ndarray: ...

    def index_chunks(self, chunks: list[ChunkData], **kwargs) -> int: ...


@runtime_checkable
class Retriever(Protocol):
    """Retrieves relevant chunks for a query."""

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]: ...


@runtime_checkable
class Reranker(Protocol):
    """Reranks retrieval candidates for fine-grained relevance."""

    def rerank(self, query: str, candidates: list[RetrievalResult], top_k: int) -> list[RetrievalResult]: ...


@runtime_checkable
class LLMBackend(Protocol):
    """Generates text from a prompt via any LLM provider."""

    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 1024) -> str: ...


@runtime_checkable
class Generator(Protocol):
    """Produces a cited answer from a query and retrieved context."""

    def generate(self, query: str, context: list[RetrievalResult]) -> GenerationResult: ...
