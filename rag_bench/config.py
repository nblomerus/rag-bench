"""
RAG-Bench Configuration
All hyperparameters, paths, and model settings for the pipeline.
"""

import os
from pathlib import Path

# ── Paths ──
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EVAL_DIR = PROJECT_ROOT / "rag_bench" / "eval"
LOG_DIR = PROJECT_ROOT / "logs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# ── Dataset ──
DATASET_NAME = "jamescalam/ai-arxiv2"
DATASET_SPLIT = "train"

# ── Embedding Model ──
EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
EMBEDDING_DIM = 768
EMBEDDING_NORMALIZE = True  # BGE models require L2 normalization

# ── ChromaDB ──
COLLECTION_NAME = "ai_ml_papers"
DISTANCE_METRIC = "cosine"  # cosine similarity for normalized embeddings

# ── Chunking ──
CHUNK_SIZE = 256  # tokens (characters used as proxy; ~4 chars/token)
CHUNK_OVERLAP = 32  # token overlap between chunks
CHUNK_SIZE_CHARS = 1024  # approximate character equivalent of 256 tokens
CHUNK_OVERLAP_CHARS = 128  # approximate character equivalent of 32 tokens
MIN_SECTION_LENGTH = 50  # skip sections shorter than this (characters)
MIN_CHUNK_LENGTH = 100  # skip chunks shorter than this (characters)

# Separators in priority order — equations first to keep them atomic
CHUNK_SEPARATORS = [
    "\n$$",  # LaTeX display math
    "$$\n",  # LaTeX display math end
    "\n\\[",  # LaTeX display math alt
    "\\]\n",  # LaTeX display math alt end
    "\n\n",  # paragraph break
    "\n",  # line break
    ". ",  # sentence break
    " ",  # word break
    "",  # character break (last resort)
]

# Sections to exclude from indexing (noise that degrades retrieval at scale)
SECTION_BLOCKLIST = frozenset(
    {
        "references",
        "bibliography",
        "acknowledgments",
        "acknowledgements",
        "acknowledgment",
        "acknowledgement",
        "preamble",
        "author_contributions",
        "funding",
        "competing_interests",
        "data_availability",
        "ethics_statement",
    }
)

# ── Retrieval ──
DEFAULT_TOP_K = 10
RELEVANCE_THRESHOLD = 0.3  # minimum cosine similarity to consider relevant
FIRST_STAGE_K = 500  # candidates per first-stage retriever (BM25 + dense)
RERANK_CANDIDATES = 200  # candidates passed to cross-encoder reranker
BM25_WEIGHT = 0.4  # RRF fusion weight for BM25
DENSE_WEIGHT = 0.6  # RRF fusion weight for dense retrieval

# ── Reranker ──
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# ── Indexing ──
EMBEDDING_BATCH_SIZE = 254  # batch size for embedding computation (increased for GPU)

# ── Logging ──
LOG_LEVEL = "INFO"

# ── Version ──
_version_file = PROJECT_ROOT / "version"
VERSION = _version_file.read_text().strip() if _version_file.exists() else "0.0.0"

# ── Environment ──
RAG_ENV = os.environ.get("RAG_ENV", "development")

# ── Scheduled Eval ──
EVAL_SCHEDULE_HOURS = int(os.environ.get("RAG_EVAL_SCHEDULE_HOURS", "24"))
