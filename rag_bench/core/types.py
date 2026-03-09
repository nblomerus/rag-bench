"""
types.py — Core data types for RAG-Bench, including chunk representations and retrieval results.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChunkData:
    """
    Represents a chunk of data that can be retrieved, including its unique identifier and the text content.
    """

    chunk_id: str
    doc_id: str
    text: str
    section: str
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to plain dict for backward compatibility with code that expects dicts."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "section": self.section,
            "metadata": dict(self.metadata),
        }


@dataclass
class RetrievalResult:
    """
    Represents a single retrieval result for a given query, including the retrieved chunk and its relevance score.
    """

    chunk: ChunkData
    relevance_score: float
    sources: list[str] = field(default_factory=list)
    rerank_score: float | None = None


@dataclass
class GenerationResult:
    """
    Represents the result of a generation step, including the generated text and any associated metadata.
    """

    answer: str
    deflected: bool
    sources: list[str] = field(default_factory=list)
    deflection_reason: str | None = None
    results: list[RetrievalResult] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
