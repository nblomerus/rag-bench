"""
RAG-Bench Configuration
All hyperparameters, paths, and model settings for the pipeline.
"""

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
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768
EMBEDDING_NORMALIZE = True  # BGE models require L2 normalization

# ── ChromaDB ──
COLLECTION_NAME = "ai_ml_papers"
DISTANCE_METRIC = "cosine"  # cosine similarity for normalized embeddings

# ── Chunking ──
CHUNK_SIZE = 512  # tokens (characters used as proxy; ~4 chars/token)
CHUNK_OVERLAP = 50  # token overlap between chunks
CHUNK_SIZE_CHARS = 2048  # approximate character equivalent of 512 tokens
CHUNK_OVERLAP_CHARS = 200  # approximate character equivalent of 50 tokens
MIN_SECTION_LENGTH = 50  # skip sections shorter than this (characters)

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

# ── Retrieval ──
DEFAULT_TOP_K = 10
RELEVANCE_THRESHOLD = 0.3  # minimum cosine similarity to consider relevant

# ── Indexing ──
EMBEDDING_BATCH_SIZE = 64  # batch size for embedding computation

# ── Logging ──
LOG_LEVEL = "INFO"
