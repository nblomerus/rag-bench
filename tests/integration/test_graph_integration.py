"""Integration tests for graph modules — no Neo4j required.

Tests GraphStore, GraphRetriever, and CommunityDetector with mocked
Neo4j driver so they run on CI without a real database.
"""

from unittest.mock import MagicMock, Mock, patch

from rag_bench.core.configs import GraphStoreConfig
from rag_bench.core.graph_types import Entity, Triple

# ══════════════════════════════════════════════════════════════════════════════
# Mock helpers
# ══════════════════════════════════════════════════════════════════════════════


def _mock_neo4j_driver():
    """Create a mock Neo4j driver with session support."""
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value = session
    # Make session work as context manager
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=False)
    return driver, session


def _make_triple(
    subj="BERT", subj_type="MODEL", pred="USES", obj="Attention", obj_type="METHOD", chunk_id="c1", doc_id="d1"
):
    return Triple(
        subject=Entity(name=subj, entity_type=subj_type),
        predicate=pred,
        object=Entity(name=obj, entity_type=obj_type),
        source_chunk_id=chunk_id,
        source_doc_id=doc_id,
        confidence=0.9,
    )


# ══════════════════════════════════════════════════════════════════════════════
# GraphStore
# ══════════════════════════════════════════════════════════════════════════════


class TestGraphStoreWithMockedNeo4j:
    @patch("rag_bench.core.graph_store.GraphDatabase")
    def test_init_creates_indexes(self, mock_gdb):
        """GraphStore.__init__ should call _ensure_indexes."""
        driver, session = _mock_neo4j_driver()
        mock_gdb.driver.return_value = driver

        from rag_bench.core.graph_store import GraphStore

        GraphStore(config=GraphStoreConfig())
        # Should have called session.run for index creation
        assert session.run.call_count >= 2
        calls = [str(c) for c in session.run.call_args_list]
        assert any("entity_name_lower" in c for c in calls)

    @patch("rag_bench.core.graph_store.GraphDatabase")
    def test_store_triples_batched(self, mock_gdb):
        """store_triples should batch writes."""
        driver, session = _mock_neo4j_driver()
        mock_gdb.driver.return_value = driver

        from rag_bench.core.graph_store import GraphStore

        store = GraphStore(config=GraphStoreConfig(batch_size=2))
        triples = [_make_triple("A", "MODEL", "USES", "B", "METHOD", f"c{i}", f"d{i}") for i in range(5)]
        count = store.store_triples(triples)
        assert count == 5
        # With batch_size=2 and 5 triples: 3 batches (2+2+1)
        # Plus 2 index creation calls = 5 total session.run calls
        write_calls = [c for c in session.run.call_args_list if "UNWIND" in str(c)]
        assert len(write_calls) == 3

    @patch("rag_bench.core.graph_store.GraphDatabase")
    def test_store_triples_empty(self, mock_gdb):
        """store_triples with empty list returns 0."""
        driver, session = _mock_neo4j_driver()
        mock_gdb.driver.return_value = driver

        from rag_bench.core.graph_store import GraphStore

        store = GraphStore()
        assert store.store_triples([]) == 0

    @patch("rag_bench.core.graph_store.GraphDatabase")
    def test_get_entity_found(self, mock_gdb):
        """get_entity returns dict when entity exists."""
        driver, session = _mock_neo4j_driver()
        mock_gdb.driver.return_value = driver

        mock_record = {"name": "BERT", "entity_type": "MODEL"}
        session.run.return_value.single.return_value = mock_record

        from rag_bench.core.graph_store import GraphStore

        store = GraphStore()
        result = store.get_entity("bert")
        assert result == {"name": "BERT", "entity_type": "MODEL"}

    @patch("rag_bench.core.graph_store.GraphDatabase")
    def test_get_entity_not_found(self, mock_gdb):
        """get_entity returns None when entity doesn't exist."""
        driver, session = _mock_neo4j_driver()
        mock_gdb.driver.return_value = driver

        session.run.return_value.single.return_value = None

        from rag_bench.core.graph_store import GraphStore

        store = GraphStore()
        result = store.get_entity("nonexistent")
        assert result is None

    @patch("rag_bench.core.graph_store.GraphDatabase")
    def test_get_neighbors_1hop(self, mock_gdb):
        """get_neighbors with max_hops=1 uses simple pattern."""
        driver, session = _mock_neo4j_driver()
        mock_gdb.driver.return_value = driver

        mock_records = [
            {"name": "Attention", "entity_type": "METHOD", "predicate": "USES", "weight": 5, "direction": "out"},
        ]
        session.run.return_value = [MagicMock(**{k: v for k, v in r.items()}) for r in mock_records]
        # Make records dict-able
        session.run.return_value = mock_records

        from rag_bench.core.graph_store import GraphStore

        store = GraphStore()
        neighbors = store.get_neighbors("BERT", max_hops=1, limit=10)
        assert len(neighbors) == 1
        assert neighbors[0]["name"] == "Attention"

    @patch("rag_bench.core.graph_store.GraphDatabase")
    def test_get_neighbors_multihop(self, mock_gdb):
        """get_neighbors with max_hops>1 uses variable-length pattern."""
        driver, session = _mock_neo4j_driver()
        mock_gdb.driver.return_value = driver

        session.run.return_value = []

        from rag_bench.core.graph_store import GraphStore

        store = GraphStore()
        store.get_neighbors("BERT", max_hops=3, limit=10)
        # Should have called with f-string containing max_hops
        cypher_call = str(session.run.call_args_list[-1])
        assert "RELATED_TO*1..3" in cypher_call

    @patch("rag_bench.core.graph_store.GraphDatabase")
    def test_get_entity_triples(self, mock_gdb):
        """get_entity_triples returns list of dicts."""
        driver, session = _mock_neo4j_driver()
        mock_gdb.driver.return_value = driver

        mock_records = [
            {
                "subject": "BERT",
                "subject_type": "MODEL",
                "predicate": "USES",
                "object": "MLM",
                "object_type": "METHOD",
                "weight": 3,
                "source_doc_ids": ["doc1"],
            },
        ]
        session.run.return_value = mock_records

        from rag_bench.core.graph_store import GraphStore

        store = GraphStore()
        triples = store.get_entity_triples("BERT", limit=10)
        assert len(triples) == 1
        assert triples[0]["subject"] == "BERT"

    @patch("rag_bench.core.graph_store.GraphDatabase")
    def test_get_stats(self, mock_gdb):
        """get_stats returns node/edge counts and type distributions."""
        driver, session = _mock_neo4j_driver()
        mock_gdb.driver.return_value = driver

        # Set up return values for the 4 queries in get_stats
        node_result = MagicMock()
        node_result.single.return_value = {"n": 100}
        edge_result = MagicMock()
        edge_result.single.return_value = {"n": 200}
        type_result = [{"t": "MODEL", "n": 50}, {"t": "METHOD", "n": 30}]
        pred_result = [{"p": "USES", "n": 80}, {"p": "EXTENDS", "n": 40}]

        session.run.side_effect = [
            MagicMock(),  # index 1
            MagicMock(),  # index 2
            node_result,
            edge_result,
            type_result,
            pred_result,
        ]

        from rag_bench.core.graph_store import GraphStore

        store = GraphStore()
        stats = store.get_stats()
        assert stats["node_count"] == 100
        assert stats["edge_count"] == 200
        assert stats["entity_types"]["MODEL"] == 50
        assert stats["top_predicates"]["USES"] == 80

    @patch("rag_bench.core.graph_store.GraphDatabase")
    def test_clear(self, mock_gdb):
        """clear() runs DETACH DELETE."""
        driver, session = _mock_neo4j_driver()
        mock_gdb.driver.return_value = driver

        from rag_bench.core.graph_store import GraphStore

        store = GraphStore()
        store.clear()
        calls = [str(c) for c in session.run.call_args_list]
        assert any("DETACH DELETE" in c for c in calls)

    @patch("rag_bench.core.graph_store.GraphDatabase")
    def test_context_manager(self, mock_gdb):
        """GraphStore supports context manager protocol."""
        driver, session = _mock_neo4j_driver()
        mock_gdb.driver.return_value = driver

        from rag_bench.core.graph_store import GraphStore

        with GraphStore() as store:
            assert store is not None
        driver.close.assert_called_once()

    @patch("rag_bench.core.graph_store.GraphDatabase")
    def test_find_path(self, mock_gdb):
        """find_path returns path or None."""
        driver, session = _mock_neo4j_driver()
        mock_gdb.driver.return_value = driver

        session.run.return_value = [
            {"name": "BERT", "entity_type": "MODEL"},
            {"name": "GPT", "entity_type": "MODEL"},
        ]

        from rag_bench.core.graph_store import GraphStore

        store = GraphStore()
        path = store.find_path("BERT", "GPT")
        assert path is not None
        assert len(path) == 2

    @patch("rag_bench.core.graph_store.GraphDatabase")
    def test_find_path_no_path(self, mock_gdb):
        """find_path returns None when no path exists."""
        driver, session = _mock_neo4j_driver()
        mock_gdb.driver.return_value = driver

        session.run.return_value = []

        from rag_bench.core.graph_store import GraphStore

        store = GraphStore()
        path = store.find_path("BERT", "Unrelated")
        assert path is None


# ══════════════════════════════════════════════════════════════════════════════
# GraphRetriever
# ══════════════════════════════════════════════════════════════════════════════


class TestGraphRetrieverWithMockedStore:
    def _make_retriever(self, entity_cache=None, triples=None, neighbors=None):
        """Build a GraphRetriever with a mocked store."""
        mock_store = MagicMock()
        mock_store.get_entity_triples.return_value = triples or []
        mock_store.get_neighbors.return_value = neighbors or []

        # Mock the session for entity cache building
        mock_session = MagicMock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        mock_store._session.return_value = mock_session

        if entity_cache is not None:
            mock_session.run.return_value = [
                {"name": e["name"], "name_lower": e["name"].lower(), "entity_type": e["entity_type"]}
                for e in entity_cache
            ]

        with patch("rag_bench.core.graph_retriever.GraphStore"):
            from rag_bench.core.graph_retriever import GraphRetriever, GraphRetrieverConfig

            retriever = GraphRetriever(
                store=mock_store,
                config=GraphRetrieverConfig(min_entity_length=3),
            )
        return retriever

    def test_match_entities_finds_known_entities(self):
        entities = [
            {"name": "BERT", "entity_type": "MODEL"},
            {"name": "GPT-2", "entity_type": "MODEL"},
            {"name": "Attention", "entity_type": "METHOD"},
        ]
        retriever = self._make_retriever(entity_cache=entities)
        matched = retriever._match_entities("How does BERT compare to GPT-2?")
        names = [e["name"] for e in matched]
        assert "BERT" in names

    def test_match_entities_empty_cache(self):
        retriever = self._make_retriever(entity_cache=[])
        matched = retriever._match_entities("What is attention?")
        assert matched == []

    def test_match_entities_respects_max(self):
        entities = [{"name": f"Entity{i}", "entity_type": "MODEL"} for i in range(20)]
        retriever = self._make_retriever(entity_cache=entities)
        retriever.config.max_entities_per_query = 3
        query = " ".join(f"Entity{i}" for i in range(20))
        matched = retriever._match_entities(query)
        assert len(matched) <= 3

    def test_get_graph_context_with_facts(self):
        entities = [{"name": "BERT", "entity_type": "MODEL"}]
        triples = [
            {
                "subject": "BERT",
                "subject_type": "MODEL",
                "predicate": "USES",
                "object": "MLM",
                "object_type": "METHOD",
                "weight": 3,
                "source_doc_ids": ["doc1"],
            },
        ]
        retriever = self._make_retriever(entity_cache=entities, triples=triples)
        chunks = retriever.get_graph_context("Tell me about BERT")
        assert len(chunks) == 1
        assert "BERT" in chunks[0]["text"]
        assert chunks[0]["section"] == "graph_context"
        assert chunks[0]["metadata"]["entities_matched"] == 1

    def test_get_graph_context_no_matches(self):
        retriever = self._make_retriever(entity_cache=[])
        chunks = retriever.get_graph_context("Random question with no entities")
        assert chunks == []

    def test_get_graph_context_no_facts(self):
        entities = [{"name": "BERT", "entity_type": "MODEL"}]
        retriever = self._make_retriever(entity_cache=entities, triples=[])
        chunks = retriever.get_graph_context("Tell me about BERT")
        assert chunks == []

    def test_expand_query_adds_neighbors(self):
        entities = [{"name": "BERT", "entity_type": "MODEL"}]
        neighbors = [
            {"name": "RoBERTa", "entity_type": "MODEL", "predicate": "EXTENDS", "weight": 5, "direction": "out"},
        ]
        retriever = self._make_retriever(entity_cache=entities, neighbors=neighbors)
        expanded = retriever.expand_query("How does BERT work?")
        assert "RoBERTa" in expanded
        assert "[Related:" in expanded

    def test_expand_query_no_entities(self):
        retriever = self._make_retriever(entity_cache=[])
        result = retriever.expand_query("random query")
        assert result == "random query"

    def test_expand_query_no_new_neighbors(self):
        """If all neighbors are already in the query, don't expand."""
        entities = [{"name": "BERT", "entity_type": "MODEL"}]
        neighbors = [
            {"name": "BERT", "entity_type": "MODEL", "predicate": "USES", "weight": 5, "direction": "out"},
        ]
        retriever = self._make_retriever(entity_cache=entities, neighbors=neighbors)
        result = retriever.expand_query("Tell me about BERT")
        assert result == "Tell me about BERT"

    def test_format_graph_chunk_basic(self):
        from rag_bench.core.graph_retriever import GraphRetriever

        facts = ["BERT (MODEL) uses MLM (METHOD)", "BERT (MODEL) extends Transformer (MODEL)"]
        entities = [{"name": "BERT", "entity_type": "MODEL"}]
        text = GraphRetriever._format_graph_chunk(facts, entities)
        assert "BERT" in text
        assert "uses MLM" in text

    def test_format_graph_chunk_with_community(self):
        from rag_bench.core.graph_retriever import GraphRetriever

        facts = ["BERT (MODEL) uses MLM (METHOD)"]
        entities = [{"name": "BERT", "entity_type": "MODEL"}]
        summaries = ["This cluster represents language model pretraining methods."]
        text = GraphRetriever._format_graph_chunk(facts, entities, summaries)
        assert "Research context:" in text
        assert "pretraining" in text

    def test_format_graph_chunk_deduplicates(self):
        from rag_bench.core.graph_retriever import GraphRetriever

        facts = ["BERT uses MLM", "BERT uses MLM"]  # duplicate
        entities = [{"name": "BERT", "entity_type": "MODEL"}]
        text = GraphRetriever._format_graph_chunk(facts, entities)
        assert text.count("BERT uses MLM") == 1

    def test_close_owns_store(self):
        """When store is created internally, close() closes it."""
        mock_store = MagicMock()
        with patch("rag_bench.core.graph_retriever.GraphStore"):
            from rag_bench.core.graph_retriever import GraphRetriever

            retriever = GraphRetriever(store=mock_store)
            retriever._owns_store = True
            retriever.close()
            mock_store.close.assert_called_once()

    def test_close_doesnt_own_store(self):
        """When store is passed in, close() does NOT close it."""
        mock_store = MagicMock()
        with patch("rag_bench.core.graph_retriever.GraphStore"):
            from rag_bench.core.graph_retriever import GraphRetriever

            retriever = GraphRetriever(store=mock_store)
            retriever._owns_store = False
            retriever.close()
            mock_store.close.assert_not_called()

    def test_context_manager(self):
        mock_store = MagicMock()
        with patch("rag_bench.core.graph_retriever.GraphStore"):
            from rag_bench.core.graph_retriever import GraphRetriever

            with GraphRetriever(store=mock_store) as retriever:
                assert retriever is not None

    def test_fetch_entity_context_filters_by_weight(self):
        entities = [{"name": "BERT", "entity_type": "MODEL"}]
        triples = [
            {
                "subject": "BERT",
                "subject_type": "MODEL",
                "predicate": "USES",
                "object": "MLM",
                "object_type": "METHOD",
                "weight": 5,
                "source_doc_ids": ["d1"],
            },
            {
                "subject": "BERT",
                "subject_type": "MODEL",
                "predicate": "RELATED",
                "object": "Noise",
                "object_type": "METHOD",
                "weight": 0,
                "source_doc_ids": [],
            },
        ]
        retriever = self._make_retriever(entity_cache=entities, triples=triples)
        retriever.config.min_edge_weight = 2
        facts, doc_ids = retriever._fetch_entity_context("BERT")
        assert len(facts) == 1  # Only weight=5 triple passes
        assert "d1" in doc_ids


# ══════════════════════════════════════════════════════════════════════════════
# CommunityDetector
# ══════════════════════════════════════════════════════════════════════════════


class TestCommunityDetectorWithMockedStore:
    def _make_detector(self, tmp_path=None):
        """Build a CommunityDetector with a mocked store."""
        mock_store = MagicMock()
        mock_session = MagicMock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        mock_store._session.return_value = mock_session

        from rag_bench.core.community_detection import (
            CommunityDetector,
            CommunityDetectorConfig,
        )

        config = CommunityDetectorConfig(
            min_community_size=2,
            cache_path=str(tmp_path / "cache.json") if tmp_path else "/tmp/test_cache.json",
        )
        return CommunityDetector(store=mock_store, config=config), mock_store, mock_session

    def test_detect_small_graph_returns_empty(self, tmp_path):
        """Graph with fewer nodes than min_community_size returns empty."""
        import networkx as nx

        detector, store, session = self._make_detector(tmp_path)

        G = nx.Graph()
        G.add_node("a", name="A", entity_type="MODEL")

        with patch.object(detector, "_export_to_networkx", return_value=G):
            communities = detector.detect()
        assert communities == []

    def test_detect_finds_communities(self, tmp_path):
        """Detection on a graph with clear clusters finds communities."""
        import networkx as nx

        detector, store, session = self._make_detector(tmp_path)

        G = nx.Graph()
        # Cluster 1: tightly connected
        for name in ["bert", "roberta", "albert"]:
            G.add_node(name, name=name.upper(), entity_type="MODEL")
        G.add_edge("bert", "roberta", weight=5, predicate="EXTENDS")
        G.add_edge("bert", "albert", weight=4, predicate="EXTENDS")
        G.add_edge("roberta", "albert", weight=3, predicate="EXTENDS")

        # Cluster 2: tightly connected
        for name in ["cnn", "resnet", "vgg"]:
            G.add_node(name, name=name.upper(), entity_type="MODEL")
        G.add_edge("cnn", "resnet", weight=5, predicate="EXTENDS")
        G.add_edge("cnn", "vgg", weight=4, predicate="EXTENDS")
        G.add_edge("resnet", "vgg", weight=3, predicate="EXTENDS")

        with (
            patch.object(detector, "_export_to_networkx", return_value=G),
            patch.object(detector, "_store_community_ids"),
        ):
            communities = detector.detect()

        assert len(communities) >= 2
        # Sorted by size (largest first)
        sizes = [c.size for c in communities]
        assert sizes == sorted(sizes, reverse=True)

    def test_community_has_predicates(self, tmp_path):
        """Communities should have top_predicates populated."""
        import networkx as nx

        detector, store, session = self._make_detector(tmp_path)

        G = nx.Graph()
        for name in ["bert", "roberta", "albert"]:
            G.add_node(name, name=name.upper(), entity_type="MODEL")
        G.add_edge("bert", "roberta", weight=5, predicate="EXTENDS")
        G.add_edge("bert", "albert", weight=4, predicate="USES")
        G.add_edge("roberta", "albert", weight=3, predicate="EXTENDS")

        with (
            patch.object(detector, "_export_to_networkx", return_value=G),
            patch.object(detector, "_store_community_ids"),
        ):
            communities = detector.detect()

        for c in communities:
            assert isinstance(c.top_predicates, list)

    def test_generate_summaries_with_cache(self, tmp_path):
        """Cached summaries are reused without LLM call."""
        import json

        from rag_bench.core.community_detection import Community

        detector, store, session = self._make_detector(tmp_path)

        # Pre-populate cache
        cache_path = tmp_path / "cache.json"
        cache_path.write_text(json.dumps({"0": "A cluster about NLP models."}))

        communities = [
            Community(
                community_id=0, entities=[{"name": "BERT", "entity_type": "MODEL"}], size=3, top_predicates=["EXTENDS"]
            ),
        ]

        result = detector.generate_summaries(communities)
        assert result[0].summary == "A cluster about NLP models."

    def test_generate_summaries_calls_ollama(self, tmp_path):
        """Uncached communities trigger Ollama call."""
        from rag_bench.core.community_detection import Community

        detector, store, session = self._make_detector(tmp_path)

        communities = [
            Community(
                community_id=0, entities=[{"name": "BERT", "entity_type": "MODEL"}], size=3, top_predicates=["EXTENDS"]
            ),
        ]

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "This cluster represents masked language model variants."}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp):
            result = detector.generate_summaries(communities)

        assert "masked language model" in result[0].summary

    def test_generate_summaries_handles_failure(self, tmp_path):
        """LLM failure leaves summary empty."""
        import requests

        from rag_bench.core.community_detection import Community

        detector, store, session = self._make_detector(tmp_path)

        communities = [
            Community(
                community_id=0, entities=[{"name": "BERT", "entity_type": "MODEL"}], size=3, top_predicates=["EXTENDS"]
            ),
        ]

        with patch("requests.post", side_effect=requests.RequestException("Connection refused")):
            result = detector.generate_summaries(communities)

        assert result[0].summary == ""

    def test_get_community_predicates(self, tmp_path):
        """_get_community_predicates returns most common predicates."""
        import networkx as nx

        from rag_bench.core.community_detection import CommunityDetector

        G = nx.Graph()
        G.add_node("a")
        G.add_node("b")
        G.add_node("c")
        G.add_edge("a", "b", predicate="USES", weight=1)
        G.add_edge("a", "c", predicate="USES", weight=1)
        G.add_edge("b", "c", predicate="EXTENDS", weight=1)

        preds = CommunityDetector._get_community_predicates(G, {"a", "b", "c"})
        assert preds[0] == "USES"  # 2 occurrences vs 1

    def test_summary_cache_roundtrip(self, tmp_path):
        """Cache save/load roundtrip preserves data."""
        detector, store, session = self._make_detector(tmp_path)

        detector._save_summary_cache({"0": "Test summary"})
        loaded = detector._load_summary_cache()
        assert loaded["0"] == "Test summary"

    def test_load_missing_cache(self, tmp_path):
        """Loading nonexistent cache returns empty dict."""
        from rag_bench.core.community_detection import (
            CommunityDetector,
            CommunityDetectorConfig,
        )

        mock_store = MagicMock()
        config = CommunityDetectorConfig(
            cache_path=str(tmp_path / "nonexistent" / "cache.json"),
        )
        detector = CommunityDetector(store=mock_store, config=config)
        assert detector._load_summary_cache() == {}

    def test_get_community_for_entity(self, tmp_path):
        """get_community_for_entity queries Neo4j and returns Community."""
        detector, store, session = self._make_detector(tmp_path)

        # First query: find community_id
        cid_result = MagicMock()
        cid_result.single.return_value = {"cid": 0}
        # Second query: find all members
        members_result = [
            {"name": "BERT", "entity_type": "MODEL"},
            {"name": "RoBERTa", "entity_type": "MODEL"},
        ]
        session.run.side_effect = [cid_result, members_result]

        community = detector.get_community_for_entity("BERT")
        assert community is not None
        assert community.community_id == 0
        assert community.size == 2

    def test_get_community_for_entity_not_found(self, tmp_path):
        """get_community_for_entity returns None when no community."""
        detector, store, session = self._make_detector(tmp_path)

        cid_result = MagicMock()
        cid_result.single.return_value = None
        session.run.return_value = cid_result
        # Also make single() on the session.run result return None
        session.run.return_value.single.return_value = None

        community = detector.get_community_for_entity("Unknown")
        assert community is None
