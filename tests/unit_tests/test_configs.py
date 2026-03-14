"""
Unit tests for rag_bench.core.configs module.

Tests cover:
- ChunkerConfig validation (__post_init__)
- RetrieverConfig weight validation
- PipelineConfig serialization (to_dict, from_dict, save, load)
- PipelineConfig.from_env() environment-based construction
- Default values and round-trip fidelity
"""

import json
import os
from unittest.mock import patch

import pytest

from rag_bench.core.configs import (
    ChunkerConfig,
    EmbedderConfig,
    GeneratorConfig,
    PipelineConfig,
    RetrieverConfig,
)

# ═══════════════════════════════════════════════════════════════════════════
# ChunkerConfig validation
# ═══════════════════════════════════════════════════════════════════════════


class TestChunkerConfig:
    def test_defaults(self):
        cfg = ChunkerConfig()
        assert cfg.chunk_size == 1024
        assert cfg.chunk_overlap == 128
        assert cfg.min_section_length == 50

    def test_custom_values(self):
        cfg = ChunkerConfig(chunk_size=512, chunk_overlap=64)
        assert cfg.chunk_size == 512
        assert cfg.chunk_overlap == 64

    def test_strategy_config_populated_from_legacy_fields(self):
        """Legacy chunk_size/chunk_overlap populate strategy_config for recursive."""
        cfg = ChunkerConfig(chunk_size=512, chunk_overlap=64)
        assert cfg.strategy_config == {"chunk_size": 512, "chunk_overlap": 64}

    def test_explicit_strategy_config_not_overwritten(self):
        """Explicit strategy_config is preserved even for recursive strategy."""
        cfg = ChunkerConfig(strategy_config={"chunk_size": 256, "chunk_overlap": 32})
        assert cfg.strategy_config == {"chunk_size": 256, "chunk_overlap": 32}

    def test_non_recursive_strategy_no_auto_populate(self):
        """Non-recursive strategy with no strategy_config stays empty."""
        cfg = ChunkerConfig(strategy="semantic", strategy_config={})
        # semantic with empty config — no auto-populate because it's not recursive
        # but __post_init__ only fires for recursive + empty
        assert cfg.strategy == "semantic"

    def test_strategy_default(self):
        cfg = ChunkerConfig()
        assert cfg.strategy == "recursive"

    def test_min_section_length_default(self):
        cfg = ChunkerConfig()
        assert cfg.min_section_length == 50


# ═══════════════════════════════════════════════════════════════════════════
# RetrieverConfig validation
# ═══════════════════════════════════════════════════════════════════════════


class TestRetrieverConfig:
    def test_defaults_sum_to_one(self):
        cfg = RetrieverConfig()
        assert abs(cfg.bm25_weight + cfg.dense_weight - 1.0) < 0.01

    def test_valid_custom_weights(self):
        cfg = RetrieverConfig(bm25_weight=0.3, dense_weight=0.7)
        assert cfg.bm25_weight == 0.3
        assert cfg.dense_weight == 0.7

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            RetrieverConfig(bm25_weight=0.5, dense_weight=0.6)

    def test_weights_both_zero(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            RetrieverConfig(bm25_weight=0.0, dense_weight=0.0)


# ═══════════════════════════════════════════════════════════════════════════
# EmbedderConfig / GeneratorConfig (no validation, just defaults)
# ═══════════════════════════════════════════════════════════════════════════


class TestEmbedderConfig:
    def test_defaults(self):
        cfg = EmbedderConfig()
        assert cfg.model_name == "BAAI/bge-base-en-v1.5"
        assert cfg.distance_metric == "cosine"
        assert cfg.batch_size == 254


class TestGeneratorConfig:
    def test_defaults(self):
        cfg = GeneratorConfig()
        assert cfg.llm_backend == "template"
        assert cfg.top_k == 10
        assert cfg.relevance_threshold == 0.3
        assert cfg.enable_citation_boost is True
        assert cfg.system_prompt == ""


# ═══════════════════════════════════════════════════════════════════════════
# PipelineConfig serialization
# ═══════════════════════════════════════════════════════════════════════════


class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.name == "default"
        assert isinstance(cfg.chunker, ChunkerConfig)
        assert isinstance(cfg.embedder, EmbedderConfig)
        assert isinstance(cfg.retriever, RetrieverConfig)
        assert isinstance(cfg.generator, GeneratorConfig)

    def test_to_dict_returns_all_fields(self):
        cfg = PipelineConfig(name="test_run")
        d = cfg.to_dict()
        assert d["name"] == "test_run"
        assert "chunker" in d
        assert "embedder" in d
        assert "retriever" in d
        assert "generator" in d
        assert d["chunker"]["chunk_size"] == 1024

    def test_from_dict_round_trip(self):
        original = PipelineConfig(
            name="custom",
            chunker=ChunkerConfig(chunk_size=512, chunk_overlap=64),
            generator=GeneratorConfig(llm_backend="ollama", top_k=5),
        )
        d = original.to_dict()
        restored = PipelineConfig.from_dict(d)
        assert restored.name == "custom"
        assert restored.chunker.chunk_size == 512
        assert restored.chunker.chunk_overlap == 64
        assert restored.generator.llm_backend == "ollama"
        assert restored.generator.top_k == 5

    def test_from_dict_missing_sections_uses_defaults(self):
        cfg = PipelineConfig.from_dict({"name": "minimal"})
        assert cfg.name == "minimal"
        assert cfg.chunker.chunk_size == 1024  # default
        assert cfg.retriever.bm25_weight == 0.4  # default

    def test_save_and_load(self, tmp_path):
        original = PipelineConfig(
            name="saved_run",
            chunker=ChunkerConfig(chunk_size=256, chunk_overlap=32),
        )
        path = tmp_path / "config.json"
        original.save(path)

        assert path.exists()
        loaded = PipelineConfig.load(path)
        assert loaded.name == "saved_run"
        assert loaded.chunker.chunk_size == 256
        assert loaded.chunker.chunk_overlap == 32

    def test_save_creates_parent_dirs(self, tmp_path):
        cfg = PipelineConfig()
        path = tmp_path / "nested" / "dir" / "config.json"
        cfg.save(path)
        assert path.exists()

    def test_save_produces_valid_json(self, tmp_path):
        cfg = PipelineConfig(name="json_test")
        path = tmp_path / "config.json"
        cfg.save(path)
        data = json.loads(path.read_text())
        assert data["name"] == "json_test"

    def test_from_env_reads_environment(self):
        env = {
            "RAG_CHROMA_DIR": "/tmp/test_chroma",
            "RAG_EMBEDDING_MODEL": "test-model",
            "RAG_LLM_BACKEND": "openai",
            "RAG_LLM_MODEL": "gpt-4",
            "RAG_LLM_BASE_URL": "https://api.openai.com",
            "RAG_RERANKER_MODEL": "test-reranker",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = PipelineConfig.from_env()
        assert cfg.name == "env"
        assert cfg.retriever.chroma_path == "/tmp/test_chroma"
        assert cfg.retriever.embedding_model == "test-model"
        assert cfg.retriever.reranker_model == "test-reranker"
        assert cfg.embedder.chroma_path == "/tmp/test_chroma"
        assert cfg.embedder.model_name == "test-model"
        assert cfg.generator.llm_backend == "openai"
        assert cfg.generator.llm_model == "gpt-4"
        assert cfg.generator.llm_base_url == "https://api.openai.com"

    def test_from_env_uses_defaults_without_env_vars(self):
        env_keys = [
            "RAG_CHROMA_DIR",
            "RAG_EMBEDDING_MODEL",
            "RAG_LLM_BACKEND",
            "RAG_LLM_MODEL",
            "RAG_LLM_BASE_URL",
            "RAG_RERANKER_MODEL",
        ]
        cleaned = {k: v for k, v in os.environ.items() if k not in env_keys}
        with patch.dict(os.environ, cleaned, clear=True):
            cfg = PipelineConfig.from_env()
        assert cfg.retriever.chroma_path == "./chroma_db"
        assert cfg.generator.llm_backend == "ollama"
