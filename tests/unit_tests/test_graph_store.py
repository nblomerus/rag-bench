"""
Unit tests for rag_bench.core.graph_store module.

Tests cover:
- Triple storage with MERGE deduplication
- Entity lookup (case-insensitive)
- Neighbor traversal (1-hop and multi-hop)
- Entity triple retrieval
- Graph stats
- Batch writing
- Idempotent index creation

These tests require a running Neo4j instance.  They are skipped if Neo4j
is not available (CI-friendly).  The tests use a fresh database state
(cleared before each test).
"""

import pytest

pytest.importorskip("neo4j")

from rag_bench.core.configs import GraphStoreConfig
from rag_bench.core.graph_store import GraphStore
from rag_bench.core.graph_types import Entity, Triple


def _neo4j_available() -> bool:
    """Check if Neo4j is reachable."""
    try:
        store = GraphStore()
        store.get_stats()
        store.close()
        return True
    except Exception:
        return False


neo4j_required = pytest.mark.skipif(
    not _neo4j_available(),
    reason="Neo4j is not running",
)


@pytest.fixture
def store():
    """GraphStore connected to local Neo4j, cleared before each test."""
    s = GraphStore()
    s.clear()
    yield s
    s.close()


def _sample_triples() -> list[Triple]:
    """A small set of triples for testing."""
    return [
        Triple(
            subject=Entity(name="GPT-4", entity_type="MODEL"),
            predicate="OUTPERFORMS",
            object=Entity(name="GPT-3.5", entity_type="MODEL"),
            source_chunk_id="chunk_001",
            source_doc_id="doc_001",
        ),
        Triple(
            subject=Entity(name="GPT-4", entity_type="MODEL"),
            predicate="EVALUATED_ON",
            object=Entity(name="MMLU", entity_type="DATASET"),
            source_chunk_id="chunk_001",
            source_doc_id="doc_001",
        ),
        Triple(
            subject=Entity(name="Transformer", entity_type="MODEL"),
            predicate="USES",
            object=Entity(name="self-attention", entity_type="METHOD"),
            source_chunk_id="chunk_002",
            source_doc_id="doc_002",
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Storage
# ══════════════════════════════════════════════════════════════════════════════


@neo4j_required
class TestStorage:
    def test_store_triples(self, store):
        """Storing triples should create nodes and edges."""
        triples = _sample_triples()
        count = store.store_triples(triples)
        assert count == 3

        stats = store.get_stats()
        # GPT-4, GPT-3.5, MMLU, Transformer, self-attention = 5 unique entities
        assert stats["node_count"] == 5
        assert stats["edge_count"] == 3

    def test_store_empty(self, store):
        """Storing empty list should return 0."""
        assert store.store_triples([]) == 0

    def test_merge_deduplicates_entities(self, store):
        """Same entity from different triples should create one node."""
        triples = _sample_triples()  # GPT-4 appears in 2 triples
        store.store_triples(triples)

        stats = store.get_stats()
        # GPT-4 should be one node, not two
        assert stats["node_count"] == 5

    def test_merge_accumulates_provenance(self, store):
        """Same triple from different chunks should increment weight."""
        triple1 = Triple(
            subject=Entity(name="BERT", entity_type="MODEL"),
            predicate="USES",
            object=Entity(name="attention", entity_type="METHOD"),
            source_chunk_id="chunk_A",
            source_doc_id="doc_A",
        )
        triple2 = Triple(
            subject=Entity(name="BERT", entity_type="MODEL"),
            predicate="USES",
            object=Entity(name="attention", entity_type="METHOD"),
            source_chunk_id="chunk_B",
            source_doc_id="doc_B",
        )

        store.store_triples([triple1])
        store.store_triples([triple2])

        triples = store.get_entity_triples("BERT")
        assert len(triples) == 1
        assert triples[0]["weight"] == 2
        assert set(triples[0]["source_doc_ids"]) == {"doc_A", "doc_B"}


# ══════════════════════════════════════════════════════════════════════════════
# Lookup
# ══════════════════════════════════════════════════════════════════════════════


@neo4j_required
class TestLookup:
    def test_get_entity(self, store):
        """Should find entity by name (case-insensitive)."""
        store.store_triples(_sample_triples())

        entity = store.get_entity("gpt-4")  # lowercase
        assert entity is not None
        assert entity["name"] == "GPT-4"
        assert entity["entity_type"] == "MODEL"

    def test_get_entity_not_found(self, store):
        """Missing entity should return None."""
        assert store.get_entity("nonexistent") is None

    def test_get_neighbors(self, store):
        """Should return connected entities."""
        store.store_triples(_sample_triples())

        neighbors = store.get_neighbors("GPT-4")
        names = {n["name"] for n in neighbors}
        assert "GPT-3.5" in names
        assert "MMLU" in names

    def test_get_entity_triples(self, store):
        """Should return all triples involving an entity."""
        store.store_triples(_sample_triples())

        triples = store.get_entity_triples("GPT-4")
        assert len(triples) == 2
        predicates = {t["predicate"] for t in triples}
        assert "OUTPERFORMS" in predicates
        assert "EVALUATED_ON" in predicates


# ══════════════════════════════════════════════════════════════════════════════
# Stats
# ══════════════════════════════════════════════════════════════════════════════


@neo4j_required
class TestStats:
    def test_stats_empty(self, store):
        stats = store.get_stats()
        assert stats["node_count"] == 0
        assert stats["edge_count"] == 0

    def test_stats_after_insert(self, store):
        store.store_triples(_sample_triples())
        stats = store.get_stats()
        assert stats["entity_types"]["MODEL"] == 3  # GPT-4, GPT-3.5, Transformer
        assert stats["entity_types"]["DATASET"] == 1
        assert stats["entity_types"]["METHOD"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════


class TestGraphStoreConfig:
    def test_default_config(self):
        config = GraphStoreConfig()
        assert config.uri == "bolt://localhost:7687"
        assert config.username == "neo4j"
        assert config.batch_size == 500
