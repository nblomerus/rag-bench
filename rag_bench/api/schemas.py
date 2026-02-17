"""
schemas.py — Pydantic request/response models for the RAG-Bench API.
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="The question to ask")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of sources to retrieve")
    backend: str | None = Field(default=None, description="Override LLM backend (ollama/openai/template)")
    model: str | None = Field(default=None, description="Override model name")
    enable_citation_boost: bool = Field(
        default=True, description="Enable citation quality boosting for foundational papers"
    )


class SourceResult(BaseModel):
    rank: int
    score: float
    title: str = ""
    section: str = ""
    text_preview: str = ""
    paper_id: str = ""
    chunk_id: str = ""
    relevance: str = ""


class QualityMetrics(BaseModel):
    retrieval_confidence: str = "unknown"
    citation_coverage: float = 0.0
    citation_density: float = 0.0
    unsupported_claims: int = 0
    sources_cited: int = 0
    sources_provided: int = 0
    top_retrieval_score: float = 0.0
    score_spread: dict = {}
    source_diversity: dict = {}
    per_source_cited: list[dict] = []


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResult]
    deflected: bool
    deflection_reason: str = ""
    scores: list[float]
    latency_ms: float
    backend: str
    model: str
    quality: QualityMetrics = QualityMetrics()


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


# ── Full Eval Request/Response ──


class FullEvalRequest(BaseModel):
    retrieval_only: bool = Field(default=False, description="Only run retrieval metrics, skip generation")
    topic: str | None = Field(default=None, description="Filter by topic")
    query_type: str | None = Field(default=None, description="Filter by query type")
    difficulty: str | None = Field(default=None, description="Filter by difficulty")


class RetrievalMetrics(BaseModel):
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    hit_rate: float = 0.0
    retrieved_papers: list[str] = []
    expected_papers: list[str] = []
    k: int = 5


class FaithfulnessMetrics(BaseModel):
    score: float = 0.0
    reasoning: str = ""


class CitationMetrics(BaseModel):
    precision: float = 0.0
    recall: float = 0.0
    source_coverage: float = 0.0
    density: float = 0.0
    unsupported_claims: int = 0
    hallucination_flags: list[str] = []


class CompletenessMetrics(BaseModel):
    expected_keywords_found: int = 0
    expected_keywords_total: int = 0
    score: float = 0.0
    missing_keywords: list[str] = []


class DeflectionMetrics(BaseModel):
    expected: bool = False
    actual: bool = False
    correct: bool = False


class FullEvalResult(BaseModel):
    id: str
    question: str
    query_type: str = ""
    topic: str = ""
    difficulty: str = ""
    retrieval: RetrievalMetrics = RetrievalMetrics()
    citation: CitationMetrics = CitationMetrics()
    completeness: CompletenessMetrics = CompletenessMetrics()
    faithfulness: FaithfulnessMetrics = FaithfulnessMetrics()
    relevance: FaithfulnessMetrics = FaithfulnessMetrics()
    deflection: DeflectionMetrics = DeflectionMetrics()
    latency_ms: float = 0.0
    answer_preview: str = ""
    error: str | None = None


class EvalSummary(BaseModel):
    total_queries: int = 0
    retrieval_mrr: float = 0.0
    retrieval_precision_at_5: float = 0.0
    retrieval_recall_at_5: float = 0.0
    retrieval_ndcg_at_5: float = 0.0
    retrieval_hit_rate: float = 0.0
    avg_faithfulness: float = 0.0
    avg_relevance: float = 0.0
    avg_citation_precision: float = 0.0
    avg_citation_recall: float = 0.0
    avg_citation_density: float = 0.0
    avg_completeness: float = 0.0
    deflection_accuracy: float = 0.0
    avg_latency_ms: float = 0.0


class FullEvalResponse(BaseModel):
    summary: EvalSummary
    by_topic: dict = {}
    by_query_type: dict = {}
    by_difficulty: dict = {}
    results: list[FullEvalResult]
    metadata: dict = {}
