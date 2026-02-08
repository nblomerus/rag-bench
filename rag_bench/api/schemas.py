"""
schemas.py — Pydantic request/response models for the RAG-Bench API.
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="The question to ask")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of sources to retrieve")
    backend: str | None = Field(default=None, description="Override LLM backend (ollama/openai/template)")
    model: str | None = Field(default=None, description="Override model name")


class SourceResult(BaseModel):
    rank: int
    score: float
    title: str = ""
    section: str = ""
    text_preview: str = ""
    paper_id: str = ""
    chunk_id: str = ""
    relevance: str = ""


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResult]
    deflected: bool
    deflection_reason: str = ""
    scores: list[float]
    latency_ms: float
    backend: str
    model: str


class StatsResponse(BaseModel):
    total_chunks: int
    total_papers: int
    collection_name: str
    embedding_model: str
    llm_backend: str
    llm_model: str


class EvalRequest(BaseModel):
    run_all: bool = Field(default=False, description="Run all queries including medium and hard")


class EvalResult(BaseModel):
    id: str
    question: str
    passed: bool
    expected_deflect: bool
    actual_deflect: bool
    deflect_source: str = ""
    top_score: float
    difficulty: str


class EvalResponse(BaseModel):
    total: int
    correct: int
    accuracy: float
    results: list[EvalResult]
    by_difficulty: dict


class PaperSummary(BaseModel):
    paper_id: str
    title: str
    year: int = 0
    arxiv_id: str = ""
    chunk_count: int = 0
    sections: list[str] = []


class PaperChunk(BaseModel):
    chunk_id: str
    text: str
    section: str = ""
    chunk_index: int = 0


class PaperDetail(BaseModel):
    paper_id: str
    title: str
    year: int = 0
    arxiv_id: str = ""
    source_display: str = ""
    chunks: list[PaperChunk] = []
    sections: list[str] = []
