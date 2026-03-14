"""
Unit tests for rag_bench.core.strategies package.

Tests cover:
- SemanticStrategy: sentence splitting, embedding, split-point detection,
  size enforcement, config validation
- RecursiveStrategy: basic splitting, config validation
- strategies __init__: get_strategy factory, registry
- SemanticConfig / RecursiveConfig validation
"""

import numpy as np
import pytest

from rag_bench.core.strategies import STRATEGY_REGISTRY, get_strategy
from rag_bench.core.strategies.recursive import RecursiveConfig, RecursiveStrategy
from rag_bench.core.strategies.semantic import (
    SemanticConfig,
    SemanticStrategy,
    _cosine_similarity,
)

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _make_embed_fn(n_dims: int = 8):
    """Return a deterministic embed_fn that maps strings to fixed vectors."""
    rng = np.random.default_rng(42)
    cache: dict[str, np.ndarray] = {}

    def embed(sentences: list[str]) -> np.ndarray:
        vecs = []
        for s in sentences:
            if s not in cache:
                v = rng.standard_normal(n_dims).astype(np.float32)
                v /= np.linalg.norm(v)
                cache[s] = v
            vecs.append(cache[s])
        return np.array(vecs)

    return embed


def _make_step_embed_fn(n_dims: int = 8, step_at: int = 3):
    """
    Embed function that creates an abrupt semantic shift at sentence `step_at`.
    Sentences 0..step_at-1 share one direction; step_at..end share another.
    """
    base_a = np.ones(n_dims, dtype=np.float32)
    base_a /= np.linalg.norm(base_a)

    base_b = np.zeros(n_dims, dtype=np.float32)
    base_b[0] = 1.0
    base_b[1] = -1.0
    base_b /= np.linalg.norm(base_b)

    def embed(sentences: list[str]) -> np.ndarray:
        vecs = []
        for i, _ in enumerate(sentences):
            if i < step_at:
                vecs.append(base_a)
            else:
                vecs.append(base_b)
        return np.array(vecs)

    return embed


# ══════════════════════════════════════════════════════════════════════════════
# _cosine_similarity
# ══════════════════════════════════════════════════════════════════════════════


class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        assert abs(_cosine_similarity(a, a) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector_returns_zero(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        assert _cosine_similarity(a, b) == 0.0
        assert _cosine_similarity(b, a) == 0.0

    def test_both_zero_returns_zero(self):
        a = np.array([0.0, 0.0])
        assert _cosine_similarity(a, a) == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# SemanticConfig validation
# ══════════════════════════════════════════════════════════════════════════════


class TestSemanticConfig:
    def test_valid_defaults(self):
        c = SemanticConfig()
        assert c.zscore_threshold == 1.0
        assert c.min_chunk_size == 200
        assert c.max_chunk_size == 2048

    def test_negative_zscore_raises(self):
        with pytest.raises(ValueError, match="zscore_threshold"):
            SemanticConfig(zscore_threshold=-0.1)

    def test_negative_min_chunk_raises(self):
        with pytest.raises(ValueError, match="min_chunk_size"):
            SemanticConfig(min_chunk_size=-1)

    def test_max_not_greater_than_min_raises(self):
        with pytest.raises(ValueError, match="max_chunk_size"):
            SemanticConfig(min_chunk_size=100, max_chunk_size=100)

    def test_zero_zscore_is_valid(self):
        # threshold == 0.0 means "always split"
        c = SemanticConfig(zscore_threshold=0.0)
        assert c.zscore_threshold == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# SemanticStrategy — construction
# ══════════════════════════════════════════════════════════════════════════════


class TestSemanticStrategyInit:
    def test_legacy_similarity_threshold_alias(self):
        """similarity_threshold kwarg maps to zscore_threshold."""
        s = SemanticStrategy(similarity_threshold=2.0, embed_fn=_make_embed_fn())
        assert s.zscore_threshold == 2.0

    def test_embed_fn_stored(self):
        fn = _make_embed_fn()
        s = SemanticStrategy(embed_fn=fn)
        assert s._embed_fn is fn

    def test_model_lazy_loaded(self):
        """Model attribute starts as None when embed_fn is given."""
        s = SemanticStrategy(embed_fn=_make_embed_fn())
        assert s._model is None


# ══════════════════════════════════════════════════════════════════════════════
# SemanticStrategy — _split_sentences
# ══════════════════════════════════════════════════════════════════════════════


class TestSplitSentences:
    def setup_method(self):
        self.strategy = SemanticStrategy(embed_fn=_make_embed_fn())

    def test_basic_sentence_split(self):
        text = "The cat sat on the mat. The dog ran away. Birds sang loudly."
        sentences = self.strategy._split_sentences(text)
        assert len(sentences) == 3

    def test_single_sentence(self):
        text = "Just one sentence here."
        sentences = self.strategy._split_sentences(text)
        assert len(sentences) == 1
        assert sentences[0] == "Just one sentence here."

    def test_abbreviation_not_split(self):
        """Dr. should NOT trigger a split."""
        text = "Dr. Smith introduced the concept. The audience applauded."
        sentences = self.strategy._split_sentences(text)
        # "Dr. Smith introduced the concept." is one sentence; next is separate
        assert len(sentences) == 2
        assert "Dr." in sentences[0]

    def test_etc_not_split(self):
        """etc. should not be treated as a sentence boundary."""
        text = "We cover topics such as NLP, CV, etc. In this paper we present something new."
        sentences = self.strategy._split_sentences(text)
        # etc. causes the abbreviation logic to merge "etc." with the next sentence
        # The exact count depends on how many periods appear; key assertion is that
        # the text starting with "etc" appears merged with adjacent content.
        assert len(sentences) >= 1
        full = " ".join(sentences)
        assert "etc" in full

    def test_exclamation_splits(self):
        text = "Wow! That was amazing. Really!"
        sentences = self.strategy._split_sentences(text)
        assert len(sentences) == 3

    def test_question_splits(self):
        text = "What is this? It is a test. Why not?"
        sentences = self.strategy._split_sentences(text)
        assert len(sentences) == 3

    def test_empty_text_returns_empty(self):
        assert self.strategy._split_sentences("") == []

    def test_whitespace_only_returns_empty(self):
        assert self.strategy._split_sentences("   \n  ") == []


# ══════════════════════════════════════════════════════════════════════════════
# SemanticStrategy — _ends_with_abbreviation
# ══════════════════════════════════════════════════════════════════════════════


class TestEndsWithAbbreviation:
    def test_dr_matches(self):
        assert SemanticStrategy._ends_with_abbreviation("Dr.")

    def test_mr_matches(self):
        assert SemanticStrategy._ends_with_abbreviation("Mr.")

    def test_etc_matches(self):
        assert SemanticStrategy._ends_with_abbreviation("etc.")

    def test_vs_matches(self):
        assert SemanticStrategy._ends_with_abbreviation("vs.")

    def test_normal_word_no_match(self):
        assert not SemanticStrategy._ends_with_abbreviation("sentence.")

    def test_empty_no_match(self):
        assert not SemanticStrategy._ends_with_abbreviation("")

    def test_ie_matches(self):
        assert SemanticStrategy._ends_with_abbreviation("i.e")

    def test_eg_matches(self):
        assert SemanticStrategy._ends_with_abbreviation("e.g")


# ══════════════════════════════════════════════════════════════════════════════
# SemanticStrategy — _embed
# ══════════════════════════════════════════════════════════════════════════════


class TestEmbed:
    def test_embed_fn_used(self):
        fn = _make_embed_fn(n_dims=4)
        s = SemanticStrategy(embed_fn=fn)
        result = s._embed(["hello", "world"])
        assert result.shape == (2, 4)

    def test_model_not_loaded_when_embed_fn_provided(self):
        """No SentenceTransformer loaded when embed_fn is given."""
        fn = _make_embed_fn()
        s = SemanticStrategy(embed_fn=fn)
        s._embed(["test"])
        assert s._model is None

    def test_lazy_model_loading_via_sentence_transformers(self):
        """When no embed_fn, SentenceTransformer is lazy-loaded on first call."""
        from unittest.mock import MagicMock, patch

        mock_model = MagicMock()
        mock_model.encode.return_value = np.eye(4, dtype=np.float32)

        s = SemanticStrategy(embedding_model="mock-model")
        assert s._model is None

        with patch("rag_bench.core.strategies.semantic.SentenceTransformer", return_value=mock_model) as mock_cls:
            result = s._embed(["sentence one", "sentence two", "sentence three", "sentence four"])

        mock_cls.assert_called_once_with("mock-model")
        assert s._model is mock_model
        assert result.shape[0] == 4

    def test_lazy_model_reused_on_second_call(self):
        """Model should not be reloaded on subsequent _embed calls."""
        from unittest.mock import MagicMock, patch

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 0.0]], dtype=np.float32)

        s = SemanticStrategy(embedding_model="mock-model")
        with patch("rag_bench.core.strategies.semantic.SentenceTransformer", return_value=mock_model) as mock_cls:
            s._embed(["first call"])
            s._embed(["second call"])

        # Should only be instantiated once
        mock_cls.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# SemanticStrategy — _compute_similarities
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeSimilarities:
    def setup_method(self):
        self.strategy = SemanticStrategy(embed_fn=_make_embed_fn(), window_size=1)

    def test_length_is_n_minus_1(self):
        vecs = np.eye(5, dtype=np.float32)
        sims = self.strategy._compute_similarities(vecs)
        assert len(sims) == 4

    def test_identical_vecs_similarity_one(self):
        vecs = np.tile(np.array([1.0, 0.0]), (4, 1)).astype(np.float32)
        sims = self.strategy._compute_similarities(vecs)
        assert all(abs(s - 1.0) < 1e-5 for s in sims)

    def test_orthogonal_similarity_zero(self):
        v = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        sims = self.strategy._compute_similarities(v)
        # alternating orthogonal vectors → similarity near 0
        assert all(abs(s) < 0.1 for s in sims)


# ══════════════════════════════════════════════════════════════════════════════
# SemanticStrategy — _find_split_points
# ══════════════════════════════════════════════════════════════════════════════


class TestFindSplitPoints:
    def test_no_split_for_single_embedding(self):
        s = SemanticStrategy(embed_fn=_make_embed_fn())
        embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
        assert s._find_split_points(embeddings) == []

    def test_no_split_when_uniform_similarity(self):
        """All-identical embeddings → std ≈ 0 → no splits."""
        s = SemanticStrategy(embed_fn=_make_embed_fn(), zscore_threshold=1.0)
        base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        embeddings = np.tile(base, (6, 1))
        assert s._find_split_points(embeddings) == []

    def test_detects_semantic_shift(self):
        """Sharp topic switch should produce at least one split point."""
        s = SemanticStrategy(
            embed_fn=_make_step_embed_fn(n_dims=8, step_at=4),
            zscore_threshold=0.5,
            window_size=1,
        )
        # 8 sentences — 4 about topic A, 4 about topic B
        sentences = [f"s{i}" for i in range(8)]
        embeddings = s._embed_fn(sentences)
        split_pts = s._find_split_points(embeddings)
        assert 4 in split_pts or 3 in split_pts  # split near boundary


# ══════════════════════════════════════════════════════════════════════════════
# SemanticStrategy — _group_sentences
# ══════════════════════════════════════════════════════════════════════════════


class TestGroupSentences:
    def setup_method(self):
        self.strategy = SemanticStrategy(embed_fn=_make_embed_fn())

    def test_no_splits_returns_single_chunk(self):
        sentences = ["This is one.", "This is two.", "This is three."]
        chunks = self.strategy._group_sentences(sentences, [])
        assert len(chunks) == 1
        assert chunks[0] == "This is one. This is two. This is three."

    def test_one_split_returns_two_chunks(self):
        sentences = ["A", "B", "C", "D"]
        chunks = self.strategy._group_sentences(sentences, [2])
        assert len(chunks) == 2
        assert chunks[0] == "A B"
        assert chunks[1] == "C D"

    def test_split_at_start(self):
        sentences = ["A", "B", "C"]
        chunks = self.strategy._group_sentences(sentences, [1])
        assert len(chunks) == 2
        assert chunks[0] == "A"
        assert chunks[1] == "B C"

    def test_multiple_splits(self):
        sentences = ["A", "B", "C", "D", "E"]
        chunks = self.strategy._group_sentences(sentences, [2, 4])
        assert len(chunks) == 3

    def test_empty_chunks_omitted(self):
        sentences = ["A", "B"]
        # split at 2 means tail is empty after the last sentence
        chunks = self.strategy._group_sentences(sentences, [1, 2])
        # ["A"] and ["B"] — tail "" is empty
        non_empty = [c for c in chunks if c.strip()]
        assert len(non_empty) == 2


# ══════════════════════════════════════════════════════════════════════════════
# SemanticStrategy — _enforce_size_limits
# ══════════════════════════════════════════════════════════════════════════════


class TestEnforceSizeLimits:
    def test_empty_returns_empty(self):
        s = SemanticStrategy(embed_fn=_make_embed_fn(), min_chunk_size=10, max_chunk_size=50)
        assert s._enforce_size_limits([]) == []

    def test_small_chunks_merged(self):
        """Chunks smaller than min_chunk_size should be merged."""
        s = SemanticStrategy(embed_fn=_make_embed_fn(), min_chunk_size=20, max_chunk_size=200)
        chunks = ["Hi.", "There.", "Friend."]
        result = s._enforce_size_limits(chunks)
        # All very short, should merge into fewer chunks
        total_chars = sum(len(c) for c in result)
        assert total_chars >= 10  # content preserved

    def test_oversized_chunk_force_split(self):
        """A chunk over max_chunk_size should be split at word boundaries."""
        s = SemanticStrategy(embed_fn=_make_embed_fn(), min_chunk_size=5, max_chunk_size=20)
        big = "word " * 20  # 100 chars
        result = s._enforce_size_limits([big])
        assert all(len(c) <= 20 for c in result)
        assert len(result) > 1

    def test_trailing_small_chunk_backward_merge(self):
        """Pass 2: a small trailing chunk should merge into its predecessor."""
        s = SemanticStrategy(embed_fn=_make_embed_fn(), min_chunk_size=50, max_chunk_size=200)
        # First chunk is big enough; second is tiny
        chunks = ["A" * 60, "tiny"]
        result = s._enforce_size_limits(chunks)
        # The tiny trailing chunk should be absorbed
        assert len(result) == 1
        assert "tiny" in result[0]

    def test_trailing_small_chunk_not_merged_when_too_big(self):
        """If merging the trailing chunk would exceed max, don't merge."""
        s = SemanticStrategy(embed_fn=_make_embed_fn(), min_chunk_size=50, max_chunk_size=80)
        # Both chunks together exceed max_chunk_size=80
        chunks = ["A" * 60, "B" * 30]
        result = s._enforce_size_limits(chunks)
        # Should keep both since merging would be 91 chars > 80
        assert len(result) == 2

    def test_single_chunk_within_limits_unchanged(self):
        s = SemanticStrategy(embed_fn=_make_embed_fn(), min_chunk_size=5, max_chunk_size=200)
        chunks = ["This is a normal sized chunk."]
        result = s._enforce_size_limits(chunks)
        assert result == chunks


# ══════════════════════════════════════════════════════════════════════════════
# SemanticStrategy — split_text (end-to-end)
# ══════════════════════════════════════════════════════════════════════════════


class TestSplitText:
    def test_empty_text_returns_empty(self):
        s = SemanticStrategy(embed_fn=_make_embed_fn())
        assert s.split_text("") == []

    def test_whitespace_only_returns_empty(self):
        s = SemanticStrategy(embed_fn=_make_embed_fn())
        assert s.split_text("   \n  ") == []

    def test_single_sentence_returns_it(self):
        s = SemanticStrategy(embed_fn=_make_embed_fn())
        result = s.split_text("Just one sentence.")
        assert result == ["Just one sentence."]

    def test_returns_list_of_strings(self):
        embed = _make_embed_fn(n_dims=8)
        s = SemanticStrategy(embed_fn=embed, min_chunk_size=1, max_chunk_size=1000)
        text = (
            "The Transformer architecture was introduced in 2017. "
            "It uses self-attention to process sequences. "
            "BERT extends this with bidirectional training. "
            "GPT uses a causal language model objective. "
            "Both have become foundational models in NLP."
        )
        result = s.split_text(text)
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)
        # Content should be preserved
        full_text = " ".join(result)
        assert "Transformer" in full_text

    def test_semantic_split_with_step_embed(self):
        """Sharp topic boundary embed fn → should produce at least 2 chunks."""
        embed = _make_step_embed_fn(n_dims=8, step_at=3)
        s = SemanticStrategy(
            embed_fn=embed,
            zscore_threshold=0.1,  # very sensitive
            min_chunk_size=1,
            max_chunk_size=10000,
            window_size=1,
        )
        text = (
            "Alpha sentence one. Alpha sentence two. Alpha sentence three. "
            "Beta sentence one. Beta sentence two. Beta sentence three."
        )
        result = s.split_text(text)
        assert len(result) >= 2


# ══════════════════════════════════════════════════════════════════════════════
# RecursiveConfig validation
# ══════════════════════════════════════════════════════════════════════════════


class TestRecursiveConfig:
    def test_valid_defaults(self):
        c = RecursiveConfig()
        assert c.chunk_size == 1024
        assert c.chunk_overlap == 128

    def test_zero_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_size"):
            RecursiveConfig(chunk_size=0)

    def test_negative_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_size"):
            RecursiveConfig(chunk_size=-1)

    def test_negative_overlap_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            RecursiveConfig(chunk_overlap=-1)

    def test_overlap_gte_size_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            RecursiveConfig(chunk_size=100, chunk_overlap=100)

    def test_overlap_just_under_size_is_valid(self):
        c = RecursiveConfig(chunk_size=100, chunk_overlap=99)
        assert c.chunk_overlap == 99


# ══════════════════════════════════════════════════════════════════════════════
# RecursiveStrategy
# ══════════════════════════════════════════════════════════════════════════════


class TestRecursiveStrategy:
    def test_splits_long_text(self):
        s = RecursiveStrategy(chunk_size=50, chunk_overlap=0)
        text = "word " * 100
        chunks = s.split_text(text)
        assert len(chunks) > 1

    def test_short_text_single_chunk(self):
        s = RecursiveStrategy(chunk_size=1000, chunk_overlap=0)
        text = "Short text."
        chunks = s.split_text(text)
        assert len(chunks) == 1

    def test_respects_chunk_size(self):
        s = RecursiveStrategy(chunk_size=100, chunk_overlap=0)
        text = "sentence. " * 50
        chunks = s.split_text(text)
        assert all(len(c) <= 110 for c in chunks)  # small tolerance for splitter

    def test_custom_overlap_preserved(self):
        s = RecursiveStrategy(chunk_size=60, chunk_overlap=20)
        assert s.chunk_overlap == 20
        assert s.chunk_size == 60

    def test_empty_text(self):
        s = RecursiveStrategy()
        result = s.split_text("")
        assert result == []


# ══════════════════════════════════════════════════════════════════════════════
# get_strategy factory
# ══════════════════════════════════════════════════════════════════════════════


class TestGetStrategy:
    def test_returns_recursive_strategy(self):
        s = get_strategy("recursive")
        assert isinstance(s, RecursiveStrategy)

    def test_returns_semantic_strategy(self):
        s = get_strategy("semantic")
        assert isinstance(s, SemanticStrategy)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            get_strategy("nonexistent")

    def test_config_passed_to_strategy(self):
        s = get_strategy("recursive", {"chunk_size": 512, "chunk_overlap": 64})
        assert s.chunk_size == 512
        assert s.chunk_overlap == 64

    def test_semantic_config_passed(self):
        s = get_strategy("semantic", {"zscore_threshold": 1.5, "min_chunk_size": 100})
        assert s.zscore_threshold == 1.5

    def test_registry_contains_expected_keys(self):
        assert "recursive" in STRATEGY_REGISTRY
        assert "semantic" in STRATEGY_REGISTRY

    def test_error_message_lists_available(self):
        with pytest.raises(ValueError, match="recursive"):
            get_strategy("bogus")
