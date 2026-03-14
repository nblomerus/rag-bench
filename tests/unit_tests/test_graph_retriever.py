"""
Unit tests for rag_bench.core.graph_retriever module.

Tests cover:
- Entity matching from queries (n-gram matching)
- Graph context generation (facts formatting)
- Integration with GraphStore (requires Neo4j)
- Mock-based tests for entity cache, query expansion, community summaries
- Edge cases: no matches, empty graph
"""

import pytest

pytest.importorskip("neo4j")

from contextlib import contextmanager
from unittest.mock import MagicMock

from rag_bench.core.graph_retriever import GraphRetriever, GraphRetrieverConfig
from rag_bench.core.graph_store import GraphStore
from rag_bench.core.graph_types import Entity, Triple


def _neo4j_available() -> bool:
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


def _seed_graph(store: GraphStore):
    """Seed the graph with known test data."""
    triples = [
        Triple(
            subject=Entity(name="GPT-4", entity_type="MODEL"),
            predicate="OUTPERFORMS",
            object=Entity(name="GPT-3.5", entity_type="MODEL"),
            source_chunk_id="c1",
            source_doc_id="doc_001",
        ),
        Triple(
            subject=Entity(name="GPT-4", entity_type="MODEL"),
            predicate="EVALUATED_ON",
            object=Entity(name="MMLU", entity_type="DATASET"),
            source_chunk_id="c2",
            source_doc_id="doc_001",
        ),
        Triple(
            subject=Entity(name="GPT-4", entity_type="MODEL"),
            predicate="USES",
            object=Entity(name="RLHF", entity_type="METHOD"),
            source_chunk_id="c3",
            source_doc_id="doc_002",
        ),
        Triple(
            subject=Entity(name="Transformer", entity_type="MODEL"),
            predicate="USES",
            object=Entity(name="self-attention", entity_type="METHOD"),
            source_chunk_id="c4",
            source_doc_id="doc_003",
        ),
        Triple(
            subject=Entity(name="BERT", entity_type="MODEL"),
            predicate="EXTENDS",
            object=Entity(name="Transformer", entity_type="MODEL"),
            source_chunk_id="c5",
            source_doc_id="doc_004",
        ),
    ]
    store.store_triples(triples)


@pytest.fixture
def seeded_store():
    """GraphStore with test data, cleared before each test."""
    store = GraphStore()
    store.clear()
    _seed_graph(store)
    yield store
    store.close()


@pytest.fixture
def retriever(seeded_store):
    """GraphRetriever connected to seeded store."""
    r = GraphRetriever(store=seeded_store)
    yield r


# ══════════════════════════════════════════════════════════════════════════════
# Entity matching
# ══════════════════════════════════════════════════════════════════════════════


@neo4j_required
class TestEntityMatching:
    def test_matches_exact_entity(self, retriever):
        """Should find exact entity names in the query."""
        matched = retriever._match_entities("What datasets was GPT-4 evaluated on?")
        names = {e["name"] for e in matched}
        assert "GPT-4" in names

    def test_matches_case_insensitive(self, retriever):
        """Entity matching should be case-insensitive."""
        matched = retriever._match_entities("how does bert handle long sequences?")
        names = {e["name"] for e in matched}
        assert "BERT" in names

    def test_matches_multiple_entities(self, retriever):
        """Should find multiple entities in one query."""
        matched = retriever._match_entities("Compare GPT-4 and BERT on MMLU")
        names = {e["name"] for e in matched}
        assert "GPT-4" in names
        assert "BERT" in names
        assert "MMLU" in names

    def test_no_match_returns_empty(self, retriever):
        """Unknown entities should return empty."""
        matched = retriever._match_entities("What is the meaning of life?")
        assert matched == []

    def test_respects_max_entities(self, seeded_store):
        """Should cap at max_entities_per_query."""
        config = GraphRetrieverConfig(max_entities_per_query=1)
        retriever = GraphRetriever(store=seeded_store, config=config)
        matched = retriever._match_entities("Compare GPT-4 and BERT on MMLU")
        assert len(matched) <= 1


# ══════════════════════════════════════════════════════════════════════════════
# Graph context generation
# ══════════════════════════════════════════════════════════════════════════════


@neo4j_required
class TestGraphContext:
    def test_returns_injectable_chunk(self, retriever):
        """get_graph_context should return chunk-like dicts."""
        chunks = retriever.get_graph_context("What datasets was GPT-4 evaluated on?")
        assert len(chunks) == 1

        chunk = chunks[0]
        assert "chunk_id" in chunk
        assert "text" in chunk
        assert "doc_id" in chunk
        assert chunk["source"] == "injection"
        assert chunk["doc_id"] == "knowledge_graph"

    def test_context_contains_facts(self, retriever):
        """The chunk text should contain graph facts."""
        chunks = retriever.get_graph_context("Tell me about GPT-4")
        text = chunks[0]["text"]
        # Should mention GPT-4's relationships
        assert "GPT-4" in text
        # Should have at least one fact
        assert "- " in text

    def test_empty_for_no_match(self, retriever):
        """No entity matches → empty list."""
        chunks = retriever.get_graph_context("What is quantum computing?")
        assert chunks == []

    def test_context_has_metadata(self, retriever):
        """Chunk metadata should include entity/fact counts."""
        chunks = retriever.get_graph_context("How does Transformer use attention?")
        meta = chunks[0]["metadata"]
        assert meta["source"] == "knowledge_graph"
        assert meta["entities_matched"] >= 1
        assert meta["facts_count"] >= 1


# ══════════════════════════════════════════════════════════════════════════════
# Formatting
# ══════════════════════════════════════════════════════════════════════════════


class TestFormatting:
    def test_format_graph_chunk(self):
        """Formatted text should be readable."""
        facts = [
            "GPT-4 (MODEL) outperforms GPT-3.5 (MODEL)",
            "GPT-4 (MODEL) evaluated on MMLU (DATASET)",
        ]
        entities = [{"name": "GPT-4", "entity_type": "MODEL", "name_lower": "gpt-4"}]
        text = GraphRetriever._format_graph_chunk(facts, entities)

        assert "Knowledge graph facts about: GPT-4" in text
        assert "- GPT-4 (MODEL) outperforms GPT-3.5 (MODEL)" in text
        assert "- GPT-4 (MODEL) evaluated on MMLU (DATASET)" in text

    def test_deduplicates_facts(self):
        """Duplicate facts should be removed."""
        facts = ["fact A", "fact A", "fact B"]
        entities = [{"name": "X", "entity_type": "MODEL", "name_lower": "x"}]
        text = GraphRetriever._format_graph_chunk(facts, entities)
        assert text.count("fact A") == 1


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════


class TestConfig:
    def test_default_config(self):
        config = GraphRetrieverConfig()
        assert config.max_hops == 1
        assert config.max_entities_per_query == 5


# ══════════════════════════════════════════════════════════════════════════════
# Mocked GraphRetriever — no Neo4j required
# ══════════════════════════════════════════════════════════════════════════════


def _make_mock_store(entities=None, triples=None):
    """Build a MagicMock GraphStore with pre-configured session responses."""
    store = MagicMock()

    # Build session context manager that returns records for entity cache query
    mock_session = MagicMock()

    entity_records = [
        {"name": e["name"], "name_lower": e["name"].lower(), "entity_type": e["entity_type"]} for e in (entities or [])
    ]

    triple_records = triples or []

    def session_run(query, **kwargs):
        if "MATCH (e:Entity)" in query and "name_lower" not in query:
            # Entity cache query
            return entity_records
        if "get_entity_triples" in str(query) or "RELATED_TO" in str(query):
            return triple_records
        return []

    mock_session.run.side_effect = session_run
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    @contextmanager
    def _session():
        yield mock_session

    store._session = _session
    return store, mock_session


def _make_retriever_with_cache(entities):
    """Create a GraphRetriever with a pre-populated entity cache (no Neo4j)."""
    store = MagicMock()
    retriever = GraphRetriever.__new__(GraphRetriever)
    retriever.config = GraphRetrieverConfig()
    retriever.store = store
    retriever._owns_store = False
    retriever._community_detector = None
    retriever._entity_cache = {
        e["name"].lower(): {
            "name": e["name"],
            "name_lower": e["name"].lower(),
            "entity_type": e["entity_type"],
        }
        for e in entities
    }
    return retriever


class TestEntityMatchingMocked:
    """Entity matching tests using a pre-populated cache — no Neo4j needed."""

    def test_empty_cache_returns_empty(self):
        r = _make_retriever_with_cache([])
        assert r._match_entities("What is GPT-4?") == []

    def test_single_entity_matched(self):
        r = _make_retriever_with_cache([{"name": "GPT-4", "entity_type": "MODEL"}])
        matched = r._match_entities("Tell me about GPT-4")
        assert len(matched) == 1
        assert matched[0]["name"] == "GPT-4"

    def test_entity_not_in_query_not_matched(self):
        r = _make_retriever_with_cache([{"name": "GPT-4", "entity_type": "MODEL"}])
        matched = r._match_entities("What is quantum computing?")
        assert matched == []

    def test_short_entity_skipped_by_min_length(self):
        r = _make_retriever_with_cache([{"name": "NN", "entity_type": "MODEL"}])
        # "nn" is 2 chars — below default min_entity_length=3
        r.config = GraphRetrieverConfig(min_entity_length=3)
        matched = r._match_entities("NN is useful")
        assert matched == []

    def test_long_ngram_preferred_over_short(self):
        """A 2-gram entity match should beat its 1-gram components."""
        r = _make_retriever_with_cache(
            [
                {"name": "Self Attention", "entity_type": "METHOD"},
                {"name": "Attention", "entity_type": "METHOD"},
            ]
        )
        matched = r._match_entities("self attention mechanism")
        names = [m["name"] for m in matched]
        # "Self Attention" (2-gram) should match; "Attention" is subsumed
        assert "Self Attention" in names

    def test_max_entities_capped(self):
        entities = [{"name": f"Model{i}", "entity_type": "MODEL"} for i in range(10)]
        r = _make_retriever_with_cache(entities)
        r.config = GraphRetrieverConfig(max_entities_per_query=3)
        text = " ".join(f"Model{i}" for i in range(10))
        matched = r._match_entities(text)
        assert len(matched) <= 3

    def test_case_insensitive_match(self):
        r = _make_retriever_with_cache([{"name": "BERT", "entity_type": "MODEL"}])
        matched = r._match_entities("bert is a great model")
        assert any(m["name"] == "BERT" for m in matched)


class TestEntityCacheMocked:
    """Tests for _get_entity_cache loading from Neo4j (mocked)."""

    def test_cache_built_once(self):
        """Second call to _get_entity_cache should return cached result."""
        r = _make_retriever_with_cache([{"name": "GPT-4", "entity_type": "MODEL"}])
        # Cache is already set — calling again should not touch the store
        r.store.assert_not_called()
        cache1 = r._get_entity_cache()
        cache2 = r._get_entity_cache()
        assert cache1 is cache2

    def test_cache_none_triggers_neo4j_query(self):
        """If _entity_cache is None, it should be populated from Neo4j."""
        mock_store = MagicMock()
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        entity_records = [
            {"name": "BERT", "name_lower": "bert", "entity_type": "MODEL"},
        ]
        mock_session.run.return_value = entity_records

        @contextmanager
        def _session():
            yield mock_session

        mock_store._session = _session

        retriever = GraphRetriever.__new__(GraphRetriever)
        retriever.config = GraphRetrieverConfig()
        retriever.store = mock_store
        retriever._owns_store = False
        retriever._community_detector = None
        retriever._entity_cache = None

        cache = retriever._get_entity_cache()
        assert "bert" in cache
        assert cache["bert"]["name"] == "BERT"


class TestFetchEntityContextMocked:
    """Tests for _fetch_entity_context using mocked store.get_entity_triples."""

    def _make_retriever(self, triples):
        store = MagicMock()
        store.get_entity_triples.return_value = triples
        r = GraphRetriever.__new__(GraphRetriever)
        r.config = GraphRetrieverConfig(min_edge_weight=1)
        r.store = store
        r._owns_store = False
        r._community_detector = None
        r._entity_cache = {}
        return r

    def test_returns_facts_and_doc_ids(self):
        triples = [
            {
                "subject": "GPT-4",
                "subject_type": "MODEL",
                "predicate": "USES",
                "object": "RLHF",
                "object_type": "METHOD",
                "weight": 2,
                "source_doc_ids": ["doc1", "doc2"],
            }
        ]
        r = self._make_retriever(triples)
        facts, doc_ids = r._fetch_entity_context("GPT-4")
        assert len(facts) == 1
        assert "gpt-4" in facts[0].lower()
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids

    def test_low_weight_triple_skipped(self):
        triples = [
            {
                "subject": "GPT-4",
                "subject_type": "MODEL",
                "predicate": "USES",
                "object": "RLHF",
                "object_type": "METHOD",
                "weight": 0,
                "source_doc_ids": [],
            }
        ]
        r = self._make_retriever(triples)
        facts, doc_ids = r._fetch_entity_context("GPT-4")
        assert facts == []

    def test_no_source_doc_ids(self):
        triples = [
            {
                "subject": "GPT-4",
                "subject_type": "MODEL",
                "predicate": "USES",
                "object": "RLHF",
                "object_type": "METHOD",
                "weight": 1,
                "source_doc_ids": None,
            }
        ]
        r = self._make_retriever(triples)
        facts, doc_ids = r._fetch_entity_context("GPT-4")
        assert len(facts) == 1
        assert len(doc_ids) == 0

    def test_predicate_underscores_replaced(self):
        triples = [
            {
                "subject": "GPT-4",
                "subject_type": "MODEL",
                "predicate": "EVALUATED_ON",
                "object": "MMLU",
                "object_type": "DATASET",
                "weight": 1,
                "source_doc_ids": [],
            }
        ]
        r = self._make_retriever(triples)
        facts, _ = r._fetch_entity_context("GPT-4")
        assert "evaluated on" in facts[0]


class TestGetGraphContextMocked:
    """Test get_graph_context end-to-end with mocked store."""

    def _make_retriever(self, entities, triples):
        r = _make_retriever_with_cache(entities)
        r.store.get_entity_triples.return_value = triples
        return r

    def test_no_entity_match_returns_empty(self):
        r = _make_retriever_with_cache([{"name": "GPT-4", "entity_type": "MODEL"}])
        r.store.get_entity_triples.return_value = []
        chunks = r.get_graph_context("What is quantum computing?")
        assert chunks == []

    def test_no_facts_no_community_returns_empty(self):
        r = _make_retriever_with_cache([{"name": "GPT-4", "entity_type": "MODEL"}])
        r.store.get_entity_triples.return_value = []
        chunks = r.get_graph_context("Tell me about GPT-4")
        assert chunks == []

    def test_returns_chunk_with_facts(self):
        triples = [
            {
                "subject": "GPT-4",
                "subject_type": "MODEL",
                "predicate": "USES",
                "object": "RLHF",
                "object_type": "METHOD",
                "weight": 1,
                "source_doc_ids": ["doc1"],
            }
        ]
        r = self._make_retriever([{"name": "GPT-4", "entity_type": "MODEL"}], triples)
        chunks = r.get_graph_context("Tell me about GPT-4")
        assert len(chunks) == 1
        chunk = chunks[0]
        assert "chunk_id" in chunk
        assert "text" in chunk
        assert "GPT-4" in chunk["text"]
        assert chunk["source"] == "injection"
        assert chunk["metadata"]["facts_count"] == 1

    def test_community_summaries_included(self):
        triples = [
            {
                "subject": "GPT-4",
                "subject_type": "MODEL",
                "predicate": "USES",
                "object": "RLHF",
                "object_type": "METHOD",
                "weight": 1,
                "source_doc_ids": [],
            }
        ]
        r = self._make_retriever([{"name": "GPT-4", "entity_type": "MODEL"}], triples)
        # Inject a community detector
        mock_community = MagicMock()
        mock_community.summary = "Large language models and RLHF training."
        mock_detector = MagicMock()
        mock_detector.get_community_for_entity.return_value = mock_community
        r._community_detector = mock_detector

        chunks = r.get_graph_context("Tell me about GPT-4")
        assert len(chunks) == 1
        assert "Research context:" in chunks[0]["text"]

    def test_community_no_summary_not_included(self):
        triples = [
            {
                "subject": "GPT-4",
                "subject_type": "MODEL",
                "predicate": "USES",
                "object": "RLHF",
                "object_type": "METHOD",
                "weight": 1,
                "source_doc_ids": [],
            }
        ]
        r = self._make_retriever([{"name": "GPT-4", "entity_type": "MODEL"}], triples)
        mock_community = MagicMock()
        mock_community.summary = ""  # empty summary
        mock_detector = MagicMock()
        mock_detector.get_community_for_entity.return_value = mock_community
        r._community_detector = mock_detector

        chunks = r.get_graph_context("Tell me about GPT-4")
        assert "Research context:" not in chunks[0]["text"]

    def test_community_none_not_included(self):
        triples = [
            {
                "subject": "GPT-4",
                "subject_type": "MODEL",
                "predicate": "USES",
                "object": "RLHF",
                "object_type": "METHOD",
                "weight": 1,
                "source_doc_ids": [],
            }
        ]
        r = self._make_retriever([{"name": "GPT-4", "entity_type": "MODEL"}], triples)
        mock_detector = MagicMock()
        mock_detector.get_community_for_entity.return_value = None
        r._community_detector = mock_detector

        chunks = r.get_graph_context("Tell me about GPT-4")
        assert "Research context:" not in chunks[0]["text"]

    def test_chunk_id_encodes_entity_names(self):
        triples = [
            {
                "subject": "GPT-4",
                "subject_type": "MODEL",
                "predicate": "USES",
                "object": "RLHF",
                "object_type": "METHOD",
                "weight": 1,
                "source_doc_ids": [],
            }
        ]
        r = self._make_retriever([{"name": "GPT-4", "entity_type": "MODEL"}], triples)
        chunks = r.get_graph_context("Tell me about GPT-4")
        assert "gpt" in chunks[0]["chunk_id"]


class TestExpandQueryMocked:
    """Tests for expand_query using mocked store.get_neighbors."""

    def test_no_entities_returns_original(self):
        r = _make_retriever_with_cache([])
        result = r.expand_query("What is quantum computing?")
        assert result == "What is quantum computing?"

    def test_neighbors_appended(self):
        r = _make_retriever_with_cache([{"name": "GPT-4", "entity_type": "MODEL"}])
        r.store.get_neighbors.return_value = [{"name": "RLHF"}, {"name": "MMLU"}]
        result = r.expand_query("Tell me about GPT-4")
        assert "[Related:" in result
        assert "GPT-4" in result  # original query retained

    def test_no_new_neighbors_returns_original(self):
        r = _make_retriever_with_cache([{"name": "GPT-4", "entity_type": "MODEL"}])
        # Neighbors are already in the query
        r.store.get_neighbors.return_value = [{"name": "GPT-4"}]
        result = r.expand_query("Tell me about GPT-4")
        # "gpt-4" is in the query, so no new names → original returned
        assert result == "Tell me about GPT-4"

    def test_context_manager(self):
        store = MagicMock()
        store.close = MagicMock()
        r = GraphRetriever.__new__(GraphRetriever)
        r.config = GraphRetrieverConfig()
        r.store = store
        r._owns_store = False
        r._community_detector = None
        r._entity_cache = {}

        with r as ret:
            assert ret is r


class TestFormatGraphChunkExtended:
    def test_with_community_summaries(self):
        facts = ["A uses B"]
        entities = [{"name": "A", "entity_type": "MODEL"}]
        summaries = ["Cluster about deep learning."]
        text = GraphRetriever._format_graph_chunk(facts, entities, summaries)
        assert "Research context: Cluster about deep learning." in text

    def test_deduplicates_community_summaries(self):
        facts = ["A uses B"]
        entities = [{"name": "A", "entity_type": "MODEL"}]
        summaries = ["Summary X", "Summary X", "Summary Y"]
        text = GraphRetriever._format_graph_chunk(facts, entities, summaries)
        assert text.count("Summary X") == 1

    def test_no_community_summaries(self):
        facts = ["A uses B"]
        entities = [{"name": "A", "entity_type": "MODEL"}]
        text = GraphRetriever._format_graph_chunk(facts, entities, None)
        assert "Research context:" not in text

    def test_empty_community_summaries_list(self):
        facts = ["A uses B"]
        entities = [{"name": "A", "entity_type": "MODEL"}]
        text = GraphRetriever._format_graph_chunk(facts, entities, [])
        assert "Research context:" not in text


class TestGraphRetrieverContextManager:
    def test_owns_store_closes_on_exit(self):
        store = MagicMock()
        store.close = MagicMock()
        r = GraphRetriever.__new__(GraphRetriever)
        r.config = GraphRetrieverConfig()
        r.store = store
        r._owns_store = True
        r._community_detector = None
        r._entity_cache = {}
        r.close()
        store.close.assert_called_once()

    def test_not_owned_store_not_closed(self):
        store = MagicMock()
        r = GraphRetriever.__new__(GraphRetriever)
        r.config = GraphRetrieverConfig()
        r.store = store
        r._owns_store = False
        r._community_detector = None
        r._entity_cache = {}
        r.close()
        store.close.assert_not_called()
