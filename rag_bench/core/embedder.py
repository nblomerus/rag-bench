"""
embedder.py — Embed chunks and index them in ChromaDB.

Uses BAAI/bge-base-en-v1.5 for dense embeddings with cosine similarity.
Stores in ChromaDB with HNSW indexing for fast approximate nearest neighbor search.

Falls back to TF-IDF embeddings when the SentenceTransformer model can't be
downloaded (e.g., in sandboxed environments without HuggingFace access).
"""

import logging
import math
import re
from collections import Counter
from pathlib import Path

import chromadb
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag_bench.core.configs import EmbedderConfig

logger = logging.getLogger(__name__)


class TfidfFallbackEmbedder:
    """
    Lightweight TF-IDF embedder for use when SentenceTransformer is unavailable.
    Produces fixed-dimension embeddings via hashed TF-IDF.

    This is NOT production quality — it's a fallback so the pipeline can run
    end-to-end without network access. Replace with BGE on your own machine.
    """

    def __init__(self, dim: int = 768):
        self.dim = dim

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + lowercased tokenization."""
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        return [w for w in text.split() if len(w) > 1]

    def encode(
        self,
        sentences,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        **kwargs,
    ):
        """Encode sentences into fixed-dimension TF-IDF vectors."""
        if isinstance(sentences, str):
            sentences = [sentences]

        embeddings = []
        for text in sentences:
            tokens = self._tokenize(text)
            tf = Counter(tokens)
            vec = [0.0] * self.dim

            for token, count in tf.items():
                # Hash token to a dimension index
                idx = hash(token) % self.dim
                # Simple TF weighting
                vec[idx] += count / max(len(tokens), 1)

            # L2 normalize
            if normalize_embeddings:
                norm = math.sqrt(sum(x * x for x in vec)) or 1.0
                vec = [x / norm for x in vec]

            embeddings.append(vec)

        return np.array(embeddings)

    def get_sentence_embedding_dimension(self):
        return self.dim


def _load_embedding_model(model_name: str):
    """Try to load SentenceTransformer; fall back to TF-IDF if unavailable."""
    try:
        # Check CUDA availability with detailed logging
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            logger.info(f"CUDA available: {device_count} device(s) - {device_name}")
            device = "cuda"
        else:
            logger.warning("CUDA not available - check PyTorch installation and GPU drivers")
            device = "cpu"

        model = SentenceTransformer(model_name, device=device)
        logger.info(f"Loaded SentenceTransformer: {model_name} on {device}")
        return model
    except Exception as e:
        logger.warning(f"Could not load SentenceTransformer ({e}). Falling back to TF-IDF hashed embedder.")
        return TfidfFallbackEmbedder(dim=768)


class Embedder:
    """
    Handles embedding computation and ChromaDB indexing.

    Attributes:
        model: SentenceTransformer model (or TF-IDF fallback) for encoding text
        client: ChromaDB persistent client
        collection: ChromaDB collection for storing chunks
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        chroma_path: str | Path = "./chroma_db",
        collection_name: str = "ai_ml_papers",
        distance_metric: str = "cosine",
        *,
        config: EmbedderConfig | None = None,
    ):
        if config is not None:
            model_name = config.model_name
            chroma_path = config.chroma_path
            collection_name = config.collection_name
            distance_metric = config.distance_metric
        logger.info(f"Loading embedding model: {model_name}")
        self.model = _load_embedding_model(model_name)
        logger.info(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

        # Initialize ChromaDB
        chroma_path = Path(chroma_path)
        chroma_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing ChromaDB at {chroma_path}")
        self.client = chromadb.PersistentClient(path=str(chroma_path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric},
        )
        logger.info(f"Collection '{collection_name}' ready. Existing documents: {self.collection.count()}")

    def embed_texts(
        self,
        texts: list[str],
        normalize: bool = True,
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> list[list[float]]:
        """
        Compute embeddings for a list of texts.

        Args:
            texts: Texts to embed
            normalize: Whether to L2-normalize (required for BGE + cosine)
            batch_size: Encoding batch size
            show_progress: Whether to show a progress bar

        Returns:
            List of embedding vectors
        """
        logger.info(f"Embedding {len(texts)} texts (batch_size={batch_size})")

        all_embeddings = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding", unit="batch")

        for i in iterator:
            batch = texts[i : i + batch_size]
            embeddings = self.model.encode(
                batch,
                normalize_embeddings=normalize,
                show_progress_bar=False,
                convert_to_tensor=False,  # Keep as numpy for ChromaDB
                batch_size=batch_size,  # Pass through batch size to model
            )
            all_embeddings.extend(embeddings.tolist())

        return all_embeddings

    def index_chunks(
        self,
        chunks: list[dict],
        batch_size: int = 256,
        skip_existing: bool = True,
    ) -> int:
        """
        Embed and index chunks into ChromaDB.

        Args:
            chunks: List of chunk dicts with chunk_id, text, metadata
            batch_size: Processing batch size
            skip_existing: If True, skip chunks already in the collection

        Returns:
            Number of newly indexed chunks
        """
        # Use upsert mode - let ChromaDB handle duplicates efficiently at database level
        # This avoids loading existing chunks into memory for comparison
        if not skip_existing:
            logger.info(f"Indexing {len(chunks)} chunks (upsert mode - no duplicate check)")
        else:
            logger.info(f"Indexing {len(chunks)} chunks (upsert mode - DB will skip duplicates)")

        if not chunks:
            logger.info("No chunks to index")
            return 0

        indexed = 0

        # Process in batches for embedding and indexing
        for i in tqdm(range(0, len(chunks), batch_size), desc="Indexing", unit="batch"):
            batch = chunks[i : i + batch_size]

            texts = [c["text"] for c in batch]
            ids = [c["chunk_id"] for c in batch]
            metadatas = []

            for c in batch:
                # ChromaDB metadata must be flat (str, int, float, bool)
                meta = {}
                for k, v in c.get("metadata", {}).items():
                    if v is None:
                        meta[k] = ""
                    elif isinstance(v, (str, int, float, bool)):
                        meta[k] = v
                    else:
                        meta[k] = str(v)
                # Also store the section name
                meta["section"] = c.get("section", "")
                meta["doc_id"] = c.get("doc_id", "")
                metadatas.append(meta)

            # Compute embeddings
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_tensor=False,  # Keep as numpy for ChromaDB
                batch_size=batch_size,  # Use specified batch size for embedding computation
            )

            # Upsert to ChromaDB (adds new, updates existing)
            self.collection.upsert(
                documents=texts,
                embeddings=embeddings.tolist(),
                ids=ids,
                metadatas=metadatas,
            )

            indexed += len(batch)

        logger.info(f"Indexed {indexed} chunks. Total in collection: {self.collection.count()}")
        return indexed

    def get_collection_stats(self) -> dict:
        """Return stats about the current collection."""
        count = self.collection.count()

        # Sample some docs to get unique paper count
        if count > 0:
            sample_size = min(count, 10000)
            results = self.collection.get(
                limit=sample_size,
                include=["metadatas"],
            )
            unique_papers = set()
            unique_sections = set()
            for meta in results["metadatas"]:
                unique_papers.add(meta.get("doc_id", ""))
                unique_sections.add(meta.get("section", ""))

            return {
                "total_chunks": count,
                "unique_papers": len(unique_papers),
                "unique_sections": len(unique_sections),
            }

        return {"total_chunks": 0, "unique_papers": 0, "unique_sections": 0}
