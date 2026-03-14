"""
semantic.py — Semantic chunking strategy.

Splits text at natural semantic boundaries by:
1. Splitting into sentences
2. Computing sentence embeddings
3. Measuring cosine similarity between consecutive sentences
4. Splitting where similarity drops significantly below the local mean

Uses a relative (standard-deviation-based) threshold so it adapts to
each section's baseline similarity level.  A fixed threshold fails on
enumerative content (references, author lists) where inter-sentence
similarity is uniformly low — the SD method only splits at genuine
topic transitions regardless of the absolute similarity level.
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

try:
    from sentence_transformers import SentenceTransformer

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False

logger = logging.getLogger(__name__)

# Common abbreviations that shouldn't trigger a sentence split
_ABBREVIATIONS = frozenset({"dr", "mr", "mrs", "ms", "prof", "jr", "sr", "vs", "etc", "approx", "i.e", "e.g"})

# Simple sentence-ending pattern: punctuation followed by whitespace
_SENTENCE_END_RE = re.compile(r"([.!?])\s+")


@dataclass
class SemanticConfig:
    """Typed config for SemanticStrategy."""

    embedding_model: str = "BAAI/bge-base-en-v1.5"
    zscore_threshold: float = 1.0
    min_chunk_size: int = 200
    max_chunk_size: int = 2048
    window_size: int = 3

    def __post_init__(self):
        if self.zscore_threshold < 0:
            raise ValueError(f"zscore_threshold must be non-negative, got {self.zscore_threshold}")
        if self.min_chunk_size < 0:
            raise ValueError(f"min_chunk_size must be non-negative, got {self.min_chunk_size}")
        if self.max_chunk_size <= self.min_chunk_size:
            raise ValueError(
                f"max_chunk_size ({self.max_chunk_size}) must be greater than min_chunk_size ({self.min_chunk_size})"
            )


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SemanticStrategy:
    """Split text at semantic boundaries using embedding similarity.

    Computes embeddings for each sentence, then splits where the cosine
    similarity between consecutive sentence groups drops significantly
    below the local mean (more than ``zscore_threshold`` standard
    deviations).

    This adaptive approach works on both narrative text (high baseline
    similarity) and enumerative text (low baseline) because it detects
    *relative* drops rather than requiring an absolute threshold.

    The ``window_size`` parameter controls how many sentences are averaged
    on each side of a candidate split point.  A window of 1 compares
    individual sentences; larger windows smooth out noise from short or
    formulaic sentences.

    Args:
        embedding_model: Name of a sentence-transformers model.
        zscore_threshold: Split where similarity is this many SDs below the
            mean.  Higher = fewer splits.  Default 1.0.
        min_chunk_size: Merge small chunks until they reach this size (chars).
        max_chunk_size: Force-split chunks that exceed this size (chars).
        window_size: Number of sentences to average on each side of a split.
        embed_fn: Optional callable that embeds a list of strings. If
            provided, ``embedding_model`` is ignored.  Useful for sharing
            an already-loaded model or injecting a mock in tests.
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        zscore_threshold: float = 1.0,
        min_chunk_size: int = 200,
        max_chunk_size: int = 2048,
        window_size: int = 3,
        embed_fn: Callable[[list[str]], np.ndarray] | None = None,
        # Legacy alias — maps to zscore_threshold for backward compat
        similarity_threshold: float | None = None,
    ):
        if similarity_threshold is not None:
            zscore_threshold = similarity_threshold

        config = SemanticConfig(
            embedding_model=embedding_model,
            zscore_threshold=zscore_threshold,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            window_size=window_size,
        )
        self.embedding_model = config.embedding_model
        self.zscore_threshold = config.zscore_threshold
        self.min_chunk_size = config.min_chunk_size
        self.max_chunk_size = config.max_chunk_size
        self.window_size = config.window_size

        self._embed_fn = embed_fn
        self._model = None  # lazy-loaded

    # -- public API -----------------------------------------------------------

    def split_text(self, text: str) -> list[str]:
        sentences = self._split_sentences(text)

        if len(sentences) <= 1:
            return [text] if text.strip() else []

        embeddings = self._embed(sentences)
        split_indices = self._find_split_points(embeddings)
        raw_chunks = self._group_sentences(sentences, split_indices)
        chunks = self._enforce_size_limits(raw_chunks)

        return chunks

    # -- internal helpers -----------------------------------------------------

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences, respecting common abbreviations."""
        # Split on sentence-ending punctuation followed by whitespace
        tokens = _SENTENCE_END_RE.split(text)

        # _SENTENCE_END_RE.split gives [text, punct, text, punct, ...].
        # Re-attach the punctuation to the preceding text.
        sentences: list[str] = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i + 1] in ".!?":
                sentence = tokens[i] + tokens[i + 1]
                i += 2
            else:
                sentence = tokens[i]
                i += 1

            # Check if the "sentence" actually ends with an abbreviation
            if sentences and self._ends_with_abbreviation(sentences[-1]):
                sentences[-1] = sentences[-1] + " " + sentence
            elif sentence.strip():
                sentences.append(sentence.strip())

        return sentences

    @staticmethod
    def _ends_with_abbreviation(text: str) -> bool:
        """Check if text ends with a known abbreviation."""
        # Get the last word before the period
        stripped = text.rstrip(".")
        last_word = stripped.split()[-1].lower() if stripped.split() else ""
        return last_word.rstrip(".") in _ABBREVIATIONS

    def _embed(self, sentences: list[str]) -> np.ndarray:
        """Embed a list of sentences, returning (n, dim) array."""
        if self._embed_fn is not None:
            return self._embed_fn(sentences)

        if self._model is None:
            logger.info(f"Loading embedding model: {self.embedding_model}")
            self._model = SentenceTransformer(self.embedding_model)

        return self._model.encode(sentences, normalize_embeddings=True)

    def _find_split_points(self, embeddings: np.ndarray) -> list[int]:
        """Find indices where semantic similarity drops significantly.

        Computes cosine similarity between windowed sentence groups, then
        splits where the similarity is more than ``zscore_threshold``
        standard deviations below the mean.

        Returns a sorted list of sentence indices where a split should
        occur *before* that sentence.
        """
        n = len(embeddings)
        if n <= 1:
            return []

        similarities = self._compute_similarities(embeddings)

        sim_array = np.array(similarities)
        mean = sim_array.mean()
        std = sim_array.std()

        # If all similarities are identical (std ≈ 0), no meaningful
        # splits exist — return empty to keep the text as one chunk.
        if std < 1e-6:
            return []

        cutoff = mean - self.zscore_threshold * std
        logger.debug(
            f"Similarity stats: mean={mean:.3f} std={std:.3f} cutoff={cutoff:.3f} (zscore={self.zscore_threshold})"
        )

        split_points = []
        for i, sim in enumerate(similarities):
            if sim < cutoff:
                split_points.append(i + 1)

        return split_points

    def _compute_similarities(self, embeddings: np.ndarray) -> list[float]:
        """Compute cosine similarity between consecutive windowed groups."""
        n = len(embeddings)
        similarities = []
        for i in range(n - 1):
            left_start = max(0, i - self.window_size + 1)
            right_end = min(n, i + 1 + self.window_size)

            left_avg = embeddings[left_start : i + 1].mean(axis=0)
            right_avg = embeddings[i + 1 : right_end].mean(axis=0)

            sim = _cosine_similarity(left_avg, right_avg)
            similarities.append(sim)
        return similarities

    def _group_sentences(self, sentences: list[str], split_indices: list[int]) -> list[str]:
        """Group sentences into chunks at the given split indices."""
        if not split_indices:
            return [" ".join(sentences)]

        chunks = []
        prev = 0
        for idx in split_indices:
            chunk_text = " ".join(sentences[prev:idx])
            if chunk_text.strip():
                chunks.append(chunk_text)
            prev = idx

        # Remaining sentences
        tail = " ".join(sentences[prev:])
        if tail.strip():
            chunks.append(tail)

        return chunks

    def _enforce_size_limits(self, chunks: list[str]) -> list[str]:
        """Merge small chunks and split oversized ones.

        Pass 1 — forward merge: accumulate undersized chunks into a buffer
                 until the buffer reaches min_chunk_size.
        Pass 2 — backward merge: if the last chunk is still undersized,
                 merge it into its predecessor.
        Pass 3 — force-split anything over max_chunk_size at word boundaries.
        """
        if not chunks:
            return chunks

        # Pass 1: forward merge
        merged: list[str] = []
        buffer = ""
        for chunk in chunks:
            candidate = (buffer + " " + chunk).strip() if buffer else chunk
            if len(buffer) < self.min_chunk_size and len(candidate) <= self.max_chunk_size:
                buffer = candidate
            else:
                if buffer:
                    merged.append(buffer)
                buffer = chunk
        if buffer:
            merged.append(buffer)

        # Pass 2: backward merge — absorb a small trailing chunk
        if len(merged) >= 2 and len(merged[-1]) < self.min_chunk_size:
            candidate = merged[-2] + " " + merged[-1]
            if len(candidate) <= self.max_chunk_size:
                merged[-2] = candidate
                merged.pop()

        # Pass 3: force-split anything still over max_chunk_size
        result: list[str] = []
        for chunk in merged:
            if len(chunk) <= self.max_chunk_size:
                result.append(chunk)
            else:
                words = chunk.split()
                current = ""
                for word in words:
                    if current and len(current) + len(word) + 1 > self.max_chunk_size:
                        result.append(current)
                        current = word
                    else:
                        current = f"{current} {word}".strip()
                if current:
                    result.append(current)

        return result
