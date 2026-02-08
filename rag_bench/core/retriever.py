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

import logging
import math
import re
from collections import Counter
from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import CrossEncoder

from rag_bench.core.embedder import _load_embedding_model

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

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len = []  # length of each document
        self.avgdl = 0.0  # average document length
        self.doc_freqs = []  # term frequency per document
        self.idf = {}  # inverse document frequency per term
        self.doc_count = 0
        self.chunk_ids = []  # parallel array of chunk IDs
        self.chunk_texts = []  # parallel array of chunk texts
        self.chunk_metadata = []  # parallel array of metadata

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenizer: lowercase, split on non-alphanumeric."""
        text = text.lower()
        tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)
        return tokens

    def index(self, chunks: list[dict]):
        """Build the BM25 index from a list of chunk dicts."""
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.chunk_texts = [c["text"] for c in chunks]
        self.chunk_metadata = [c.get("metadata", {}) for c in chunks]

        # Build term frequencies
        df = Counter()  # document frequency per term
        self.doc_freqs = []
        self.doc_len = []

        for chunk in chunks:
            tokens = self._tokenize(chunk["text"])
            self.doc_len.append(len(tokens))

            tf = Counter(tokens)
            self.doc_freqs.append(tf)

            # Count document frequency (how many docs contain each term)
            for term in tf:
                df[term] += 1

        self.doc_count = len(chunks)
        self.avgdl = sum(self.doc_len) / max(self.doc_count, 1)

        # Compute IDF for each term
        self.idf = {}
        for term, freq in df.items():
            # IDF with smoothing to avoid negative values
            self.idf[term] = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1.0)

        logger.info(f"BM25 index built: {self.doc_count} docs, {len(self.idf)} unique terms")

    def query(self, question: str, top_k: int = 20) -> list[dict]:
        """Score all documents against the query and return top-k."""
        query_tokens = self._tokenize(question)
        scores = []

        for i in range(self.doc_count):
            score = 0.0
            tf = self.doc_freqs[i]
            dl = self.doc_len[i]

            for token in query_tokens:
                if token not in tf:
                    continue

                term_freq = tf[token]
                idf = self.idf.get(token, 0.0)

                # BM25 scoring formula
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * numerator / denominator

            scores.append(score)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break  # No more relevant results
            results.append(
                {
                    "chunk_id": self.chunk_ids[idx],
                    "text": self.chunk_texts[idx],
                    "score": float(scores[idx]),
                    "metadata": self.chunk_metadata[idx],
                    "source": "bm25",
                }
            )

        return results


# ═══════════════════════════════════════════════════════════════════════════
# Cross-Encoder Reranker
# ═══════════════════════════════════════════════════════════════════════════
class CrossEncoderReranker:
    """
    Reranks candidate passages using a cross-encoder model.

    Cross-encoders process (query, passage) pairs jointly, giving much
    more accurate relevance scores than bi-encoders, but are too slow
    for first-stage retrieval over thousands of docs.

    Uses: cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, accurate)
    Fallback: simple keyword overlap scoring if model unavailable
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = None
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        """Try to load the cross-encoder model."""
        try:
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Loaded cross-encoder: {self.model_name}")
        except Exception as e:
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
    1. BM25 (sparse) -> top 50 candidates
    2. Dense (BGE via ChromaDB) -> top 50 candidates
    3. Reciprocal Rank Fusion -> merged candidates
    4. Cross-encoder reranking -> final top-k

    Usage:
        retriever = HybridRetriever(...)
        results = retriever.query("How does attention work?", top_k=5)
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        chroma_path: str | Path = "./chroma_db",
        collection_name: str = "ai_ml_papers",
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
    ):
        # Dense retriever (BGE + ChromaDB)
        logger.info("Initializing dense retriever...")
        self.embed_model = _load_embedding_model(embedding_model)
        chroma_path = Path(chroma_path)
        self.chroma_client = chromadb.PersistentClient(path=str(chroma_path))
        self.collection = self.chroma_client.get_collection(name=collection_name)

        # BM25 retriever
        logger.info("Building BM25 index...")
        self.bm25 = BM25()
        self._build_bm25_index()

        # Cross-encoder reranker
        logger.info("Initializing cross-encoder reranker...")
        self.reranker = CrossEncoderReranker(model_name=reranker_model)

        # Fusion weights
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

        total_chunks = self.collection.count()
        logger.info(
            f"HybridRetriever ready: {total_chunks} chunks, BM25 weight={bm25_weight}, Dense weight={dense_weight}"
        )

    def _build_bm25_index(self):
        """Load all chunks from ChromaDB and build the BM25 index."""
        count = self.collection.count()
        if count == 0:
            logger.warning("Collection is empty — BM25 index will be empty")
            return

        # Fetch all documents from ChromaDB
        all_data = self.collection.get(
            limit=count,
            include=["documents", "metadatas"],
        )

        chunks = []
        for i in range(len(all_data["ids"])):
            chunks.append(
                {
                    "chunk_id": all_data["ids"][i],
                    "text": all_data["documents"][i],
                    "metadata": all_data["metadatas"][i],
                }
            )

        self.bm25.index(chunks)

    def _dense_query(self, question: str, top_k: int = 50) -> list[dict]:
        """Run dense retrieval via ChromaDB."""
        raw_embedding = self.embed_model.encode(
            question,
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

    def query(
        self,
        question: str,
        top_k: int = 10,
        first_stage_k: int = 50,
        relevance_threshold: float = 0.0,
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
        # Stage 1: First-stage retrieval (parallel in concept)
        bm25_results = self.bm25.query(question, top_k=first_stage_k)
        dense_results = self._dense_query(question, top_k=first_stage_k)

        logger.debug(f"First stage: BM25={len(bm25_results)}, Dense={len(dense_results)}")

        # Stage 2: Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(bm25_results, dense_results)

        # Take top candidates for reranking (reranker is expensive)
        rerank_candidates = fused[: min(len(fused), first_stage_k)]

        # Stage 3: Cross-encoder reranking
        reranked = self.reranker.rerank(question, rerank_candidates, top_k=top_k)

        # Use rerank_score as the final score
        for r in reranked:
            r["score"] = r.get("rerank_score", r.get("rrf_score", 0.0))

        # Apply threshold
        if relevance_threshold > 0:
            reranked = [r for r in reranked if r["score"] >= relevance_threshold]

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
