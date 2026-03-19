"""
retriever.py — Hybrid retriever: BM25 + Dense + Cross-Encoder Reranking.

Three-stage retrieval:
1. BM25 (sparse): keyword matching for exact terms, equations, acronyms
2. Dense (BGE): semantic similarity via ChromaDB
3. Cross-encoder reranking: fine-grained relevance scoring

The hybrid approach handles edge cases that pure dense retrieval misses:
- Exact equation lookups (BM25 wins)
- Conceptual similarity (dense wins)
- Final ranking precision (cross-encoder wins)
"""

import json
import logging
import math
import pickle
import re
import time
from array import array as c_array
from collections import Counter
from pathlib import Path

import chromadb
import numpy as np
import torch
from sentence_transformers import CrossEncoder

from rag_bench.config import (
    BM25_WEIGHT,
    DENSE_WEIGHT,
    FIRST_STAGE_K,
    RERANK_CANDIDATES,
    RERANKER_MODEL,
)
from rag_bench.core.configs import RetrieverConfig
from rag_bench.core.embedder import _load_embedding_model
from rag_bench.core.types import ChunkData, RetrievalResult

try:
    from rag_bench.observability.metrics import RERANKING_DURATION, RETRIEVAL_DURATION

    _HAS_METRICS = True
except Exception:
    _HAS_METRICS = False

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# BM25 Implementation (no external dependency needed)
# ═══════════════════════════════════════════════════════════════════════════
class BM25:
    """
    Okapi BM25 sparse retriever over in-memory document store.

    Built from scratch to avoid extra dependencies. Operates on
    the same chunk collection as the dense retriever.
    """

    # Stopwords: standard English + ML-ubiquitous terms with near-zero IDF at scale
    STOPWORDS = frozenset(
        {
            # Standard English stopwords
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "shall",
            "should",
            "may",
            "might",
            "must",
            "can",
            "could",
            "and",
            "but",
            "or",
            "nor",
            "not",
            "no",
            "so",
            "if",
            "than",
            "that",
            "this",
            "these",
            "those",
            "it",
            "its",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "about",
            "between",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "up",
            "down",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "every",
            "both",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "only",
            "own",
            "same",
            "too",
            "very",
            "just",
            "also",
            "we",
            "our",
            "us",
            "they",
            "their",
            "them",
            "he",
            "she",
            "him",
            "her",
            "his",
        }
    )

    # Cache format version — bump when the on-disk layout changes so stale
    # caches are automatically rebuilt instead of crashing on load.
    _CACHE_VERSION = 3

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: np.ndarray = np.array([], dtype=np.float32)
        self.avgdl = 0.0
        self.idf: dict[str, float] = {}
        self.doc_count = 0
        self.chunk_ids: list[str] = []
        # Inverted index: term -> (doc_indices np.uint32, term_freqs np.float32)
        self.inv_index: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize with stopword removal for better signal at scale."""
        if text is None:
            return []
        text = text.lower()
        tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)
        return [t for t in tokens if t not in self.STOPWORDS and len(t) > 1]

    def index(self, chunks):
        """Build the BM25 inverted index from an iterable of chunk dicts.

        Accepts any iterable (list, generator, etc.) to support streaming
        chunks without buffering all text in memory at once.
        """

        self.chunk_ids = []

        # First pass: collect term frequencies per doc and doc lengths.
        # Use array.array for postings instead of list[tuple] — each entry
        # uses ~6 bytes (uint32 + uint16) vs ~72 bytes for a Python tuple,
        # cutting peak memory by ~12x for large corpora.
        df = Counter()  # document frequency per term
        postings_docs: dict[str, c_array] = {}  # term -> doc indices (uint32)
        postings_tfs: dict[str, c_array] = {}  # term -> term freqs  (uint16)
        doc_lens = c_array("f")  # float32

        for doc_idx, chunk in enumerate(chunks):
            self.chunk_ids.append(chunk["chunk_id"])
            tokens = self._tokenize(chunk["text"])
            doc_lens.append(len(tokens))

            tf = Counter(tokens)
            for term, freq in tf.items():
                df[term] += 1
                if term not in postings_docs:
                    postings_docs[term] = c_array("I")  # unsigned 32-bit
                    postings_tfs[term] = c_array("H")  # unsigned 16-bit
                postings_docs[term].append(doc_idx)
                postings_tfs[term].append(min(freq, 65535))

        self.doc_count = len(self.chunk_ids)

        self.doc_len = np.frombuffer(doc_lens, dtype=np.float32).copy()
        self.avgdl = float(self.doc_len.mean()) if self.doc_count > 0 else 0.0

        # Compute IDF
        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1.0)

        # Build inverted index with numpy arrays
        self.inv_index = {}
        for term in postings_docs:
            doc_ids = np.frombuffer(postings_docs[term], dtype=np.uint32).copy()
            tfs = np.frombuffer(postings_tfs[term], dtype=np.uint16).astype(np.float32)
            self.inv_index[term] = (doc_ids, tfs)

        logger.info(f"BM25 inverted index built: {self.doc_count} docs, {len(self.idf)} unique terms")

    def query(self, question: str, top_k: int = 20) -> list[dict]:
        """Score documents against the query using the inverted index."""
        query_tokens = self._tokenize(question)
        if not query_tokens or self.doc_count == 0:
            return []

        scores = np.zeros(self.doc_count, dtype=np.float64)

        for token in query_tokens:
            if token not in self.inv_index:
                continue

            idf = self.idf[token]
            doc_ids, tfs = self.inv_index[token]

            # Vectorized BM25 scoring over matching documents only
            dl = self.doc_len[doc_ids]
            numerator = tfs * (self.k1 + 1)
            denominator = tfs + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            scores[doc_ids] += idf * numerator / denominator

        # Fast top-k via argpartition (O(n) instead of O(n log n) full sort)
        actual_k = min(top_k, self.doc_count)
        top_indices = np.argpartition(scores, -actual_k)[-actual_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            results.append(
                {
                    "chunk_id": self.chunk_ids[idx],
                    "score": float(scores[idx]),
                    "source": "bm25",
                }
            )

        return results

    def save_to_cache(self, path: Path):
        """Serialize the inverted index to disk."""
        state = {
            "_cache_version": self._CACHE_VERSION,
            "doc_count": self.doc_count,
            "avgdl": self.avgdl,
            "k1": self.k1,
            "b": self.b,
            "idf": self.idf,
            "doc_len": self.doc_len,
            "chunk_ids": self.chunk_ids,
            "inv_index": self.inv_index,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"BM25 cache saved to {path} ({path.stat().st_size / 1024 / 1024:.0f} MB)")

    def load_from_cache(self, path: Path) -> bool:
        """Load a previously serialized BM25 index. Returns True on success."""
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)  # noqa: S301
            if state.get("_cache_version") != self._CACHE_VERSION:
                logger.info("BM25 cache version mismatch, rebuilding...")
                return False
            self.doc_count = state["doc_count"]
            self.avgdl = state["avgdl"]
            self.k1 = state["k1"]
            self.b = state["b"]
            self.idf = state["idf"]
            self.doc_len = state["doc_len"]
            self.chunk_ids = state["chunk_ids"]
            self.inv_index = state["inv_index"]
            return True
        except Exception as e:
            logger.warning(f"Failed to load BM25 cache: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════
# Cross-Encoder Reranker
# ═══════════════════════════════════════════════════════════════════════════
class CrossEncoderReranker:
    """
    Reranks candidate passages using a cross-encoder model.

    Cross-encoders process (query, passage) pairs jointly, giving much
    more accurate relevance scores than bi-encoders, but are too slow
    for first-stage retrieval over thousands of docs.

    Default: BAAI/bge-reranker-v2-m3 (high quality for scientific text)
    Fallback: simple keyword overlap scoring if model unavailable
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model = None
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        """Load cross-encoder model, preferring GPU when available."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = CrossEncoder(self.model_name, device=device)
            logger.info(f"Loaded cross-encoder: {self.model_name} (on {device})")
        except Exception as e:
            if isinstance(e, RuntimeError) and "out of memory" in str(e).lower() and device == "cuda":
                logger.warning("GPU OOM loading cross-encoder, falling back to CPU")
                self.model = CrossEncoder(self.model_name, device="cpu")
                logger.info(f"Loaded cross-encoder: {self.model_name} (on CPU, OOM fallback)")
            else:
                logger.warning(f"Could not load cross-encoder ({e}). Falling back to keyword overlap scoring.")
                self.model = None

    def rerank(
        self,
        question: str,
        candidates: list[dict],
        top_k: int = 10,
    ) -> list[dict]:
        """
        Rerank candidates by relevance to the question.

        Args:
            question: The user's query
            candidates: List of chunk dicts with 'text' field
            top_k: Number of results to return after reranking

        Returns:
            Reranked list of chunk dicts with updated 'rerank_score'
        """
        if not candidates:
            return []

        if self.model is not None:
            return self._rerank_with_model(question, candidates, top_k)
        else:
            return self._rerank_with_keywords(question, candidates, top_k)

    def _rerank_with_model(self, question: str, candidates: list[dict], top_k: int) -> list[dict]:
        """Rerank using cross-encoder model."""
        pairs = [(question, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)

        # Sort by rerank score descending
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]

    def _rerank_with_keywords(self, question: str, candidates: list[dict], top_k: int) -> list[dict]:
        """Fallback: rerank using keyword overlap + position scoring."""
        q_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))

        for c in candidates:
            c_tokens = re.findall(r"[a-z0-9]+", c["text"].lower())
            c_token_set = set(c_tokens)

            # Overlap ratio
            overlap = len(q_tokens & c_token_set) / max(len(q_tokens), 1)

            # Exact phrase bonus
            q_lower = question.lower()
            phrase_bonus = 0.0
            # Check for 2-gram and 3-gram matches
            q_words = q_lower.split()
            for n in [3, 2]:
                for i in range(len(q_words) - n + 1):
                    phrase = " ".join(q_words[i : i + n])
                    if phrase in c["text"].lower():
                        phrase_bonus += 0.1 * n

            # Combine existing score with keyword score
            existing_score = c.get("score", 0.0)
            c["rerank_score"] = existing_score * 0.4 + overlap * 0.4 + phrase_bonus * 0.2

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]


# ═══════════════════════════════════════════════════════════════════════════
# Hybrid Retriever (combines all three stages)
# ═══════════════════════════════════════════════════════════════════════════
class HybridRetriever:
    """
    Three-stage hybrid retriever:
    1. BM25 (sparse) -> top candidates
    2. Dense (BGE via ChromaDB) -> top candidates
    3. Reciprocal Rank Fusion -> merged candidates
    4. Cross-encoder reranking -> final top-k

    Usage:
        retriever = HybridRetriever(...)
        results = retriever.query("How does attention work?", top_k=5)
    """

    # BGE query instruction prefix for improved retrieval accuracy
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        reranker_model: str = RERANKER_MODEL,
        chroma_path: str | Path = "./chroma_db",
        collection_name: str = "ai_ml_papers",
        bm25_weight: float = BM25_WEIGHT,
        dense_weight: float = DENSE_WEIGHT,
        *,
        config: RetrieverConfig | None = None,
    ):
        if config is not None:
            embedding_model = config.embedding_model
            reranker_model = config.reranker_model
            chroma_path = config.chroma_path
            collection_name = config.collection_name
            bm25_weight = config.bm25_weight
            dense_weight = config.dense_weight
        # Dense retriever (BGE + ChromaDB)
        logger.info("Initializing dense retriever...")
        self.embed_model = _load_embedding_model(embedding_model)
        self._chroma_path = Path(chroma_path)
        self.chroma_client = chromadb.PersistentClient(path=str(self._chroma_path))
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)

        # BM25 retriever (with disk cache)
        logger.info("Building BM25 index...")
        self.bm25 = BM25()
        self._build_bm25_index()

        # Cross-encoder reranker
        logger.info("Initializing cross-encoder reranker...")
        self.reranker = CrossEncoderReranker(model_name=reranker_model)

        # Fusion weights
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

        # Paper title index for canonical paper boosting
        logger.info("Building paper title index...")
        self._build_title_index()

        total_chunks = self.collection.count()
        logger.info(
            f"HybridRetriever ready: {total_chunks} chunks, BM25 weight={bm25_weight}, Dense weight={dense_weight}"
        )

    def _build_title_index(self):
        """Build paper title -> arxiv_id lookup for canonical paper boosting."""
        cache_path = self._chroma_path / "title_index.json"

        if cache_path.exists():
            with open(cache_path) as f:
                self._paper_titles = json.load(f)
            logger.info(f"Title index loaded from cache ({len(self._paper_titles)} papers)")
        else:
            count = self.collection.count()
            papers: dict[str, str] = {}
            batch_size = 100_000
            for offset in range(0, count, batch_size):
                batch_limit = min(batch_size, count - offset)
                data = self.collection.get(
                    offset=offset,
                    limit=batch_limit,
                    include=["metadatas"],
                )
                for meta in data["metadatas"]:
                    pid = meta.get("arxiv_id", "")
                    title = meta.get("title", "")
                    if pid and title and pid not in papers:
                        papers[pid] = title
                logger.info(f"  Title index: scanned {min(offset + batch_size, count)}/{count}")

            self._paper_titles = papers
            with open(cache_path, "w") as f:
                json.dump(papers, f)
            logger.info(f"Title index built and cached ({len(papers)} papers)")

        # Build inverted index: word -> list of arxiv_ids
        self._title_words: dict[str, list[str]] = {}
        for pid, title in self._paper_titles.items():
            for word in set(re.findall(r"[a-z0-9]+", title.lower())):
                if len(word) >= 2:
                    self._title_words.setdefault(word, []).append(pid)
        logger.info(f"Title inverted index: {len(self._title_words)} unique terms")

    _TITLE_STOP_WORDS = frozenset(
        {
            "what",
            "how",
            "does",
            "did",
            "do",
            "is",
            "are",
            "was",
            "were",
            "the",
            "a",
            "an",
            "and",
            "or",
            "if",
            "to",
            "of",
            "in",
            "on",
            "at",
            "by",
            "it",
            "be",
            "as",
            "so",
            "up",
            "no",
            "we",
            "its",
            "not",
            "but",
            "they",
            "can",
            "which",
            "about",
            "have",
            "has",
            "this",
            "that",
            "than",
            "into",
            "between",
            "over",
            "from",
            "with",
            "many",
            "more",
        }
    )

    def _find_title_matches(self, question: str, max_papers: int = 3) -> list[str]:
        """Find canonical papers whose titles best match the query using IDF scoring."""
        query_terms = set(re.findall(r"[a-z0-9]+", question.lower())) - self._TITLE_STOP_WORDS
        total_papers = max(len(self._paper_titles), 1)

        paper_scores: Counter = Counter()
        for term in query_terms:
            matching_papers = self._title_words.get(term, [])
            n = len(matching_papers)
            if n == 0 or n > total_papers * 0.5:
                continue
            weight = math.log(total_papers / n)
            for pid in matching_papers:
                paper_scores[pid] += weight

        min_score = math.log(total_papers / 200)
        ranked = [(pid, s) for pid, s in paper_scores.most_common(max_papers * 3) if s >= min_score]
        return [pid for pid, _ in ranked[:max_papers]]

    def _search_abstracts(self, question: str, max_papers: int = 3) -> list[str]:
        """Find canonical papers by querying abstract sections with dense search."""
        prefixed_query = self.BGE_QUERY_PREFIX + question
        raw_embedding = self.embed_model.encode(prefixed_query, normalize_embeddings=True)
        query_embedding = np.array(raw_embedding).flatten().tolist()

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                where={"section": "abstract"},
                n_results=max_papers * 5,
                include=["metadatas", "distances"],
            )
        except Exception:
            return []

        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        seen: set[str] = set()
        papers: list[str] = []
        for i in range(len(results["ids"][0])):
            pid = results["metadatas"][0][i].get("arxiv_id", "")
            if pid and pid not in seen:
                seen.add(pid)
                papers.append(pid)
                if len(papers) >= max_papers:
                    break
        return papers

    def _build_bm25_index(self):
        """Load BM25 index from cache, or build from ChromaDB and cache."""
        count = self.collection.count()
        if count == 0:
            logger.warning("Collection is empty — BM25 index will be empty")
            return

        cache_path = self._chroma_path / "bm25_cache.pkl"

        # Try loading from cache
        if cache_path.exists():
            t0 = time.monotonic()
            if self.bm25.load_from_cache(cache_path) and self.bm25.doc_count == count:
                elapsed = time.monotonic() - t0
                logger.info(
                    f"BM25 index loaded from cache in {elapsed:.1f}s ({count} chunks, {len(self.bm25.idf)} terms)"
                )
                return
            logger.info("BM25 cache stale (chunk count changed), rebuilding...")

        # Build from ChromaDB — stream chunks via generator to avoid
        # holding all 1.6M+ chunk texts in memory simultaneously.
        batch_size = 200_000

        def _chunk_stream():
            loaded = 0
            for offset in range(0, count, batch_size):
                batch_limit = min(batch_size, count - offset)
                batch_data = self.collection.get(
                    offset=offset,
                    limit=batch_limit,
                    include=["documents"],
                )
                for i in range(len(batch_data["ids"])):
                    yield {
                        "chunk_id": batch_data["ids"][i],
                        "text": batch_data["documents"][i],
                    }
                loaded += len(batch_data["ids"])
                logger.info(f"  Loaded {loaded}/{count} chunks for BM25")

        logger.info(f"Streaming {count} chunks for BM25 indexing...")
        self.bm25.index(_chunk_stream())

        # Save cache for next startup
        try:
            self.bm25.save_to_cache(cache_path)
        except Exception as e:
            logger.warning(f"Could not save BM25 cache: {e}")

    def _dense_query(self, question: str, top_k: int = 50) -> list[dict]:
        """Run dense retrieval via ChromaDB with BGE query instruction prefix."""
        # BGE models use an instruction prefix on queries (not documents) for
        # improved retrieval accuracy
        prefixed_query = self.BGE_QUERY_PREFIX + question
        raw_embedding = self.embed_model.encode(
            prefixed_query,
            normalize_embeddings=True,
        )
        query_embedding = np.array(raw_embedding).flatten().tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        formatted = []
        for i in range(len(results["ids"][0])):
            score = 1.0 - results["distances"][0][i]  # cosine distance -> similarity
            formatted.append(
                {
                    "chunk_id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "score": score,
                    "metadata": results["metadatas"][0][i],
                    "source": "dense",
                }
            )

        return formatted

    def _reciprocal_rank_fusion(
        self,
        bm25_results: list[dict],
        dense_results: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """
        Merge BM25 and dense results using Reciprocal Rank Fusion (RRF).

        RRF score = sum( 1 / (k + rank_i) ) for each ranking the doc appears in.
        This is robust and doesn't require score normalization between retrievers.
        """
        # Build a combined map of chunk_id -> chunk data + RRF score
        chunk_map = {}

        for rank, result in enumerate(bm25_results):
            cid = result["chunk_id"]
            rrf_score = self.bm25_weight / (k + rank + 1)

            if cid not in chunk_map:
                chunk_map[cid] = {
                    "chunk_id": cid,
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "rrf_score": 0.0,
                    "bm25_score": result["score"],
                    "dense_score": 0.0,
                    "sources": [],
                }
            chunk_map[cid]["rrf_score"] += rrf_score
            chunk_map[cid]["sources"].append("bm25")

        for rank, result in enumerate(dense_results):
            cid = result["chunk_id"]
            rrf_score = self.dense_weight / (k + rank + 1)

            if cid not in chunk_map:
                chunk_map[cid] = {
                    "chunk_id": cid,
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "rrf_score": 0.0,
                    "bm25_score": 0.0,
                    "dense_score": 0.0,
                    "sources": [],
                }
            chunk_map[cid]["rrf_score"] += rrf_score
            chunk_map[cid]["dense_score"] = result["score"]
            chunk_map[cid]["sources"].append("dense")

        # Sort by RRF score
        fused = sorted(chunk_map.values(), key=lambda x: x["rrf_score"], reverse=True)
        return fused

    def fetch_paper_chunks(
        self,
        arxiv_id: str,
        max_chunks: int = 3,
        preferred_sections: list[str] | None = None,
        query: str | None = None,
    ) -> list[dict]:
        """
        Fetch chunks for a specific paper directly from ChromaDB.

        Used for foundational paper injection — bypasses BM25/dense retrieval
        to guarantee the paper appears in the candidate pool.

        When ``query`` is provided, uses semantic similarity to select the most
        relevant chunks from the paper and returns real similarity scores.
        Otherwise falls back to static section-priority ordering with score 0.0.
        """
        # ── Query-aware path: select chunks by semantic similarity ──
        if query is not None:
            try:
                prefixed_query = self.BGE_QUERY_PREFIX + query
                raw_embedding = self.embed_model.encode(
                    prefixed_query,
                    normalize_embeddings=True,
                )
                query_embedding = np.array(raw_embedding).flatten().tolist()

                data = self.collection.query(
                    query_embeddings=[query_embedding],
                    where={"arxiv_id": arxiv_id},
                    n_results=max_chunks,
                    include=["documents", "metadatas", "distances"],
                )
            except Exception as e:
                logger.warning(f"Query-aware fetch failed for arxiv_id={arxiv_id}: {e}")
                return []

            if not data or not data.get("ids") or not data["ids"][0]:
                logger.debug(f"No chunks found for arxiv_id={arxiv_id}")
                return []

            chunks = []
            for i in range(len(data["ids"][0])):
                score = 1.0 - data["distances"][0][i]
                chunks.append(
                    {
                        "chunk_id": data["ids"][0][i],
                        "text": data["documents"][0][i],
                        "metadata": data["metadatas"][0][i],
                        "score": score,
                        "source": "injection",
                        "sources": ["injection"],
                        "rrf_score": 0.0,
                        "bm25_score": 0.0,
                        "dense_score": score,
                    }
                )

            score_strs = [f"{c['score']:.3f}" for c in chunks]
            logger.debug(f"Query-aware fetch: {len(chunks)} chunks for arxiv_id={arxiv_id} (scores: {score_strs})")
            return chunks

        # ── Static fallback path (backward compatibility) ──
        if preferred_sections is None:
            preferred_sections = ["abstract", "introduction", "background", "model", "architecture", "method"]

        try:
            data = self.collection.get(
                where={"arxiv_id": arxiv_id},
                limit=50,
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.warning(f"Failed to fetch chunks for arxiv_id={arxiv_id}: {e}")
            return []

        if not data or not data.get("ids"):
            logger.debug(f"No chunks found for arxiv_id={arxiv_id}")
            return []

        chunks = []
        for i in range(len(data["ids"])):
            chunks.append(
                {
                    "chunk_id": data["ids"][i],
                    "text": data["documents"][i],
                    "metadata": data["metadatas"][i],
                    "score": 0.0,
                    "source": "injection",
                    "sources": ["injection"],
                    "rrf_score": 0.0,
                    "bm25_score": 0.0,
                    "dense_score": 0.0,
                }
            )

        # Prioritize preferred sections
        section_priority = {s: idx for idx, s in enumerate(preferred_sections)}

        def sort_key(chunk):
            section = chunk["metadata"].get("section", "").lower()
            return section_priority.get(section, len(preferred_sections))

        chunks.sort(key=sort_key)
        selected = chunks[:max_chunks]

        logger.debug(
            f"Fetched {len(selected)} chunks for arxiv_id={arxiv_id} "
            f"(sections: {[c['metadata'].get('section', '?') for c in selected]})"
        )
        return selected

    def query(
        self,
        question: str,
        top_k: int = 10,
        first_stage_k: int = FIRST_STAGE_K,
        relevance_threshold: float = 0.0,
        inject_chunks: list[dict] | None = None,
    ) -> list[dict]:
        """
        Full hybrid retrieval pipeline.

        Args:
            question: User query
            top_k: Final number of results after reranking
            first_stage_k: Number of candidates from each first-stage retriever
            relevance_threshold: Minimum rerank score to include

        Returns:
            List of result dicts sorted by relevance, each containing:
            - chunk_id, text, score, rerank_score, metadata, sources
        """
        _retrieval_start = time.time()

        # Stage 1: First-stage retrieval (parallel in concept)
        bm25_results = self.bm25.query(question, top_k=first_stage_k)
        dense_results = self._dense_query(question, top_k=first_stage_k)

        # Enrich BM25 results with text/metadata from ChromaDB
        if bm25_results:
            bm25_ids = [r["chunk_id"] for r in bm25_results]
            enriched = self.collection.get(ids=bm25_ids, include=["documents", "metadatas"])
            id_to_data = {}
            for i, cid in enumerate(enriched["ids"]):
                id_to_data[cid] = (enriched["documents"][i], enriched["metadatas"][i])
            for r in bm25_results:
                text, meta = id_to_data.get(r["chunk_id"], ("", {}))
                r["text"] = text
                r["metadata"] = meta

        logger.debug(f"First stage: BM25={len(bm25_results)}, Dense={len(dense_results)}")

        # Stage 2: Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(bm25_results, dense_results)

        # Take top candidates for reranking (reranker is expensive)
        rerank_limit = min(len(fused), RERANK_CANDIDATES)
        rerank_candidates = fused[:rerank_limit]

        # Stage 2.5: Inject foundational paper chunks into the rerank pool
        if inject_chunks:
            existing_ids = {r["chunk_id"] for r in rerank_candidates}
            new_injections = [c for c in inject_chunks if c["chunk_id"] not in existing_ids]
            if new_injections:
                # Replace bottom-ranked candidates to keep pool size constant
                n_inject = min(len(new_injections), rerank_limit // 4)  # Max 25% of slots
                if n_inject > 0:
                    rerank_candidates = rerank_candidates[: rerank_limit - n_inject] + new_injections[:n_inject]
                    logger.debug(
                        f"Injected {n_inject} foundational chunks into rerank pool (pool size: {len(rerank_candidates)})"
                    )

        # Stage 3: Cross-encoder reranking
        _rerank_start = time.time()
        reranked = self.reranker.rerank(question, rerank_candidates, top_k=top_k)
        _rerank_elapsed = time.time() - _rerank_start

        # Ensure injected chunks survive reranking regardless of rank.
        # The reranker sorts candidates in-place, so rerank_candidates still
        # has all entries with their rerank_score — we just re-add any
        # injected chunks that fell below the top_k cut.
        if inject_chunks:
            returned_ids = {r["chunk_id"] for r in reranked}
            for c in rerank_candidates:
                if c.get("source") == "injection" and c["chunk_id"] not in returned_ids:
                    reranked.append(c)

        # Use rerank_score as the final score
        for r in reranked:
            r["score"] = r.get("rerank_score", r.get("rrf_score", 0.0))

        # Apply threshold (injected chunks bypass — they were intentionally added)
        if relevance_threshold > 0:
            reranked = [r for r in reranked if r["score"] >= relevance_threshold or r.get("source") == "injection"]

        # Record timing metrics for observability
        _retrieval_elapsed = time.time() - _retrieval_start
        if _HAS_METRICS:
            RETRIEVAL_DURATION.observe(_retrieval_elapsed)
            RERANKING_DURATION.observe(_rerank_elapsed)

        # Attach timing so callers (e.g. generator) can report per-stage breakdown
        self._last_retrieval_ms = _retrieval_elapsed * 1000
        self._last_reranking_ms = _rerank_elapsed * 1000

        return reranked

    def query_with_citations(
        self,
        question: str,
        top_k: int = 5,
        relevance_threshold: float = 0.3,
    ) -> dict:
        """
        Retrieve results formatted with numbered citation sources.

        Returns a dict with:
        - results: list of scored chunks
        - sources: formatted source strings for citation
        - is_relevant: whether any results meet the relevance threshold
        """
        results = self.query(
            question=question,
            top_k=top_k,
            relevance_threshold=0.0,  # Get all, filter below
        )

        # Build citation source strings
        sources = []
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            source_str = (
                f"[Source {i}] {meta.get('source_display', 'Unknown')} — {meta.get('section', 'Unknown section')}"
            )
            sources.append(source_str)

        # Determine if the top result is relevant enough
        top_score = results[0]["score"] if results else 0.0
        is_relevant = top_score >= relevance_threshold

        return {
            "question": question,
            "results": results,
            "sources": sources,
            "is_relevant": is_relevant,
            "top_score": top_score,
        }

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Protocol-conforming retrieval — returns typed RetrievalResult objects.

        Wraps the existing query() method, converting raw dicts into the
        typed dataclasses defined in types.py.
        """
        raw_results = self.query(query, top_k=top_k)
        return [
            RetrievalResult(
                chunk=ChunkData(
                    chunk_id=r["chunk_id"],
                    doc_id=r.get("metadata", {}).get("doc_id", ""),
                    text=r.get("text", ""),
                    section=r.get("metadata", {}).get("section", ""),
                    metadata=r.get("metadata", {}),
                ),
                relevance_score=r.get("score", 0.0),
                rerank_score=r.get("rerank_score"),
                sources=r.get("sources", []),
            )
            for r in raw_results
        ]

    def print_results(self, question: str, top_k: int = 5):
        """Pretty-print hybrid retrieval results."""
        output = self.query_with_citations(question, top_k=top_k)

        print(f"\n{'=' * 70}")
        print(f"Query: {question}")
        print(f"{'=' * 70}")

        if not output["is_relevant"]:
            print(f"\n  No sufficiently relevant results (top score: {output['top_score']:.3f})")
            print("  The system would deflect this query.\n")
            return output

        print(f"\nTop {len(output['results'])} results:\n")

        for i, r in enumerate(output["results"], 1):
            retrieval_sources = ", ".join(r.get("sources", ["unknown"]))
            print(f"  [{i}] Score: {r['score']:.4f} (via {retrieval_sources})")
            print(f"      Paper: {r['metadata'].get('title', 'Unknown')[:70]}")
            print(f"      Section: {r['metadata'].get('section', '?')}")
            print(f"      Text: {r['text'][:150].strip()}...")
            if "bm25_score" in r:
                print(
                    f"      BM25={r.get('bm25_score', 0):.3f}  "
                    f"Dense={r.get('dense_score', 0):.3f}  "
                    f"RRF={r.get('rrf_score', 0):.4f}"
                )
            print()

        print("Sources:")
        for source in output["sources"]:
            print(f"  {source}")

        print()
        return output
