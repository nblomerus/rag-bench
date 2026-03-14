"""
Unit tests for rag_bench.core.community_detection module.

Tests cover:
- Community detection on a small seeded graph (Neo4j, skipped if unavailable)
- Community storage in Neo4j
- Export to NetworkX
- Community lookup by entity
- Mocked tests: generate_summaries, cache, small graph, get_community_predicates
"""

import pytest

pytest.importorskip("neo4j")
pytest.importorskip("networkx")

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import requests as req_lib

from rag_bench.core.community_detection import (
    Community,
    CommunityDetector,
    CommunityDetectorConfig,
)
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


def _seed_graph_with_communities(store: GraphStore):
    """Create two clear clusters connected by a single bridge edge."""
    # Cluster A: Transformer ecosystem
    cluster_a = [
        Triple(Entity("Transformer", "MODEL"), "USES", Entity("self-attention", "METHOD"), "c1", "d1"),
        Triple(Entity("Transformer", "MODEL"), "USES", Entity("positional encoding", "METHOD"), "c2", "d1"),
        Triple(Entity("BERT", "MODEL"), "EXTENDS", Entity("Transformer", "MODEL"), "c3", "d2"),
        Triple(Entity("GPT-2", "MODEL"), "EXTENDS", Entity("Transformer", "MODEL"), "c4", "d3"),
        Triple(Entity("BERT", "MODEL"), "TRAINED_ON", Entity("BookCorpus", "DATASET"), "c5", "d2"),
        Triple(Entity("self-attention", "METHOD"), "PART_OF", Entity("Transformer", "MODEL"), "c6", "d1"),
    ]

    # Cluster B: CNN ecosystem
    cluster_b = [
        Triple(Entity("ResNet", "MODEL"), "USES", Entity("skip connections", "METHOD"), "c10", "d4"),
        Triple(Entity("VGG", "MODEL"), "USES", Entity("convolution", "METHOD"), "c11", "d5"),
        Triple(Entity("ResNet", "MODEL"), "EVALUATED_ON", Entity("ImageNet", "DATASET"), "c12", "d4"),
        Triple(Entity("VGG", "MODEL"), "EVALUATED_ON", Entity("ImageNet", "DATASET"), "c13", "d5"),
        Triple(Entity("ResNet", "MODEL"), "OUTPERFORMS", Entity("VGG", "MODEL"), "c14", "d6"),
        Triple(Entity("convolution", "METHOD"), "PART_OF", Entity("VGG", "MODEL"), "c15", "d5"),
    ]

    # Bridge: weak connection between clusters
    bridge = [
        Triple(Entity("BERT", "MODEL"), "COMPARED_WITH", Entity("ResNet", "MODEL"), "c20", "d7"),
    ]

    store.store_triples(cluster_a + cluster_b + bridge)


@pytest.fixture
def store():
    s = GraphStore()
    s.clear()
    _seed_graph_with_communities(s)
    yield s
    s.close()


@pytest.fixture
def detector(store):
    config = CommunityDetectorConfig(min_community_size=3)
    return CommunityDetector(store=store, config=config)


@neo4j_required
class TestCommunityDetection:
    def test_detects_communities(self, detector):
        """Should find at least 2 communities in our graph."""
        communities = detector.detect()
        assert len(communities) >= 2

    def test_community_sizes(self, detector):
        """Largest communities should have >= 3 entities."""
        communities = detector.detect()
        for c in communities:
            assert c.size >= 3

    def test_communities_are_thematic(self, detector):
        """The two clusters should be in different communities."""
        communities = detector.detect()

        # Find which community Transformer and ResNet belong to
        transformer_cid = None
        resnet_cid = None
        for c in communities:
            names = {e["name"].lower() for e in c.entities}
            if "transformer" in names:
                transformer_cid = c.community_id
            if "resnet" in names:
                resnet_cid = c.community_id

        # They should be in different communities
        assert transformer_cid is not None
        assert resnet_cid is not None
        assert transformer_cid != resnet_cid

    def test_stores_community_ids_in_neo4j(self, detector, store):
        """After detection, entities should have community_id property."""
        detector.detect()

        with store._session() as session:
            result = session.run("MATCH (e:Entity {name_lower: 'transformer'}) RETURN e.community_id AS cid")
            record = result.single()
            assert record is not None
            assert record["cid"] is not None

    def test_get_community_for_entity(self, detector):
        """Should retrieve the community an entity belongs to."""
        detector.detect()

        community = detector.get_community_for_entity("Transformer")
        assert community is not None
        assert community.size >= 3
        names = {e["name"].lower() for e in community.entities}
        assert "transformer" in names


@neo4j_required
class TestNetworkXExport:
    def test_export_has_correct_size(self, detector):
        """NetworkX graph should have same node/edge count as Neo4j."""
        G = detector._export_to_networkx()
        stats = detector.store.get_stats()
        assert G.number_of_nodes() == stats["node_count"]

    def test_edges_have_weight(self, detector):
        """Exported edges should have weight attribute."""
        G = detector._export_to_networkx()
        for _u, _v, data in G.edges(data=True):
            assert "weight" in data


class TestCommunityDataclass:
    def test_default_community(self):
        c = Community(community_id=0, entities=[], size=0)
        assert c.summary == ""
        assert c.top_predicates == []


# ══════════════════════════════════════════════════════════════════════════════
# Mocked community detection tests — no Neo4j required
# ══════════════════════════════════════════════════════════════════════════════


def _make_mock_store():
    store = MagicMock()
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    @contextmanager
    def _session():
        yield mock_session

    store._session = _session
    return store, mock_session


def _make_detector(store=None, config=None):
    if store is None:
        store, _ = _make_mock_store()
    return CommunityDetector(store=store, config=config or CommunityDetectorConfig())


class TestDetectMocked:
    """Tests for detect() using an in-memory NetworkX graph."""

    def test_empty_graph_returns_empty(self):
        """Graph with fewer nodes than min_community_size → []."""
        store, session = _make_mock_store()
        # Return 1 node, 0 edges
        session.run.side_effect = [
            [{"id": "a", "name": "A", "entity_type": "MODEL"}],
            [],  # edges query
        ]
        config = CommunityDetectorConfig(min_community_size=3)
        detector = CommunityDetector(store=store, config=config)
        # _export_to_networkx returns a graph with 1 node
        # detect() checks < min_community_size → []
        with patch.object(detector, "_export_to_networkx") as mock_export:
            G = nx.Graph()
            G.add_node("a", name="A", entity_type="MODEL")
            mock_export.return_value = G
            result = detector.detect()
        assert result == []

    def test_store_community_ids_called(self):
        store, session = _make_mock_store()
        session.run.return_value = []
        config = CommunityDetectorConfig(min_community_size=3)
        detector = CommunityDetector(store=store, config=config)

        # Build a real graph that will produce communities
        G = _build_two_cluster_graph()
        with (
            patch.object(detector, "_export_to_networkx", return_value=G),
            patch.object(detector, "_store_community_ids") as mock_store_ids,
        ):
            communities = detector.detect()
            mock_store_ids.assert_called_once_with(communities)

    def test_communities_sorted_by_size(self):
        store, _ = _make_mock_store()
        config = CommunityDetectorConfig(min_community_size=2)
        detector = CommunityDetector(store=store, config=config)

        G = _build_two_cluster_graph()
        with (
            patch.object(detector, "_export_to_networkx", return_value=G),
            patch.object(detector, "_store_community_ids"),
        ):
            communities = detector.detect()

        sizes = [c.size for c in communities]
        assert sizes == sorted(sizes, reverse=True)

    def test_communities_have_predicates(self):
        store, _ = _make_mock_store()
        config = CommunityDetectorConfig(min_community_size=2)
        detector = CommunityDetector(store=store, config=config)

        G = _build_two_cluster_graph()
        with (
            patch.object(detector, "_export_to_networkx", return_value=G),
            patch.object(detector, "_store_community_ids"),
        ):
            communities = detector.detect()

        for c in communities:
            assert isinstance(c.top_predicates, list)


def _build_two_cluster_graph() -> nx.Graph:
    """Build a NetworkX graph with two clear clusters."""
    G = nx.Graph()
    # Cluster A
    for name in ["transformer", "bert", "gpt2", "attention", "positional"]:
        G.add_node(name, name=name.title(), entity_type="MODEL")
    edges_a = [
        ("transformer", "bert", "EXTENDS"),
        ("transformer", "gpt2", "EXTENDS"),
        ("transformer", "attention", "USES"),
        ("transformer", "positional", "USES"),
        ("bert", "attention", "USES"),
    ]
    for src, tgt, pred in edges_a:
        G.add_edge(src, tgt, weight=2, predicate=pred)

    # Cluster B
    for name in ["resnet", "vgg", "imagenet", "convolution", "skip"]:
        G.add_node(name, name=name.title(), entity_type="MODEL")
    edges_b = [
        ("resnet", "vgg", "OUTPERFORMS"),
        ("resnet", "imagenet", "EVALUATED_ON"),
        ("vgg", "imagenet", "EVALUATED_ON"),
        ("resnet", "skip", "USES"),
        ("vgg", "convolution", "USES"),
    ]
    for src, tgt, pred in edges_b:
        G.add_edge(src, tgt, weight=2, predicate=pred)

    # Bridge
    G.add_edge("bert", "resnet", weight=1, predicate="COMPARED_WITH")
    return G


class TestGenerateSummariesMocked:
    """Tests for generate_summaries using mocked Ollama."""

    def _make_detector_with_cache(self, cache_content=None):
        store, _ = _make_mock_store()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = str(Path(tmpdir) / "summaries.json")
            config = CommunityDetectorConfig(
                cache_path=cache_path,
                ollama_base_url="http://localhost:11434",
            )
            detector = CommunityDetector(store=store, config=config)
            if cache_content:
                Path(cache_path).write_text(json.dumps(cache_content))
            yield detector, cache_path

    def test_uses_cache_if_available(self):
        store, _ = _make_mock_store()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = str(Path(tmpdir) / "summaries.json")
            config = CommunityDetectorConfig(cache_path=cache_path)
            detector = CommunityDetector(store=store, config=config)

            # Pre-populate cache
            Path(cache_path).write_text(json.dumps({"0": "Cached summary."}))

            community = Community(community_id=0, entities=[], size=3)
            with patch("requests.post") as mock_post:
                result = detector.generate_summaries([community])
            mock_post.assert_not_called()
            assert result[0].summary == "Cached summary."

    def test_ollama_called_for_uncached_community(self):
        store, _ = _make_mock_store()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = str(Path(tmpdir) / "summaries.json")
            config = CommunityDetectorConfig(cache_path=cache_path)
            detector = CommunityDetector(store=store, config=config)

            community = Community(
                community_id=1,
                entities=[{"name": "BERT", "entity_type": "MODEL"}],
                size=3,
                top_predicates=["USES", "EXTENDS"],
            )

            mock_resp = MagicMock()
            mock_resp.json.return_value = {"response": "This is a detailed summary."}
            mock_resp.raise_for_status = MagicMock()

            with patch("requests.post", return_value=mock_resp) as mock_post:
                result = detector.generate_summaries([community])

            mock_post.assert_called_once()
            assert result[0].summary == "This is a detailed summary."

    def test_short_response_ignored(self):
        """Responses <= 20 chars should not be stored."""
        store, _ = _make_mock_store()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = str(Path(tmpdir) / "summaries.json")
            config = CommunityDetectorConfig(cache_path=cache_path)
            detector = CommunityDetector(store=store, config=config)

            community = Community(
                community_id=2,
                entities=[{"name": "BERT", "entity_type": "MODEL"}],
                size=3,
                top_predicates=["USES"],
            )

            mock_resp = MagicMock()
            mock_resp.json.return_value = {"response": "Short."}
            mock_resp.raise_for_status = MagicMock()

            with patch("requests.post", return_value=mock_resp):
                result = detector.generate_summaries([community])

            assert result[0].summary == ""  # not set

    def test_request_exception_handled(self):
        """Network error should not raise — community summary stays empty."""
        store, _ = _make_mock_store()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = str(Path(tmpdir) / "summaries.json")
            config = CommunityDetectorConfig(cache_path=cache_path)
            detector = CommunityDetector(store=store, config=config)

            community = Community(
                community_id=3,
                entities=[{"name": "GPT", "entity_type": "MODEL"}],
                size=3,
                top_predicates=["USES"],
            )

            with patch("requests.post", side_effect=req_lib.RequestException("timeout")):
                result = detector.generate_summaries([community])

            assert result[0].summary == ""

    def test_cache_saved_after_generation(self):
        store, _ = _make_mock_store()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = str(Path(tmpdir) / "summaries.json")
            config = CommunityDetectorConfig(cache_path=cache_path)
            detector = CommunityDetector(store=store, config=config)

            community = Community(
                community_id=4,
                entities=[{"name": "Transformer", "entity_type": "MODEL"}],
                size=5,
                top_predicates=["USES"],
            )

            mock_resp = MagicMock()
            mock_resp.json.return_value = {"response": "This cluster covers transformer-based language models."}
            mock_resp.raise_for_status = MagicMock()

            with patch("requests.post", return_value=mock_resp):
                detector.generate_summaries([community])

            saved = json.loads(Path(cache_path).read_text())
            assert "4" in saved
            assert "transformer" in saved["4"].lower()


class TestSummaryCacheMocked:
    def test_load_missing_cache_returns_empty(self):
        store, _ = _make_mock_store()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CommunityDetectorConfig(cache_path=str(Path(tmpdir) / "nonexistent.json"))
            detector = CommunityDetector(store=store, config=config)
            assert detector._load_summary_cache() == {}

    def test_load_corrupt_cache_returns_empty(self):
        store, _ = _make_mock_store()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "bad.json"
            cache_path.write_text("not json at all {{")
            config = CommunityDetectorConfig(cache_path=str(cache_path))
            detector = CommunityDetector(store=store, config=config)
            assert detector._load_summary_cache() == {}

    def test_save_creates_directories(self):
        store, _ = _make_mock_store()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = str(Path(tmpdir) / "new_dir" / "summaries.json")
            config = CommunityDetectorConfig(cache_path=cache_path)
            detector = CommunityDetector(store=store, config=config)
            detector._save_summary_cache({"1": "test summary"})
            assert Path(cache_path).exists()
            data = json.loads(Path(cache_path).read_text())
            assert data["1"] == "test summary"


class TestGetCommunityPredicates:
    def test_returns_most_common_predicates(self):
        G = nx.Graph()
        G.add_node("a")
        G.add_node("b")
        G.add_node("c")
        G.add_edge("a", "b", predicate="USES", weight=1)
        G.add_edge("b", "c", predicate="USES", weight=1)
        G.add_edge("a", "c", predicate="EXTENDS", weight=1)

        members = {"a", "b", "c"}
        preds = CommunityDetector._get_community_predicates(G, members)
        assert preds[0] == "USES"  # most common

    def test_only_intra_community_edges(self):
        G = nx.Graph()
        for n in ["a", "b", "c", "d"]:
            G.add_node(n)
        G.add_edge("a", "b", predicate="USES", weight=1)
        G.add_edge("a", "d", predicate="EXTERNAL", weight=1)  # d not in members

        members = {"a", "b", "c"}
        preds = CommunityDetector._get_community_predicates(G, members)
        assert "EXTERNAL" not in preds

    def test_empty_members_returns_empty(self):
        G = nx.Graph()
        preds = CommunityDetector._get_community_predicates(G, set())
        assert preds == []


class TestGetCommunityForEntityMocked:
    def test_entity_without_community_returns_none(self):
        store, session = _make_mock_store()

        # single() returns None (no community_id)
        mock_result = MagicMock()
        mock_result.single.return_value = None
        session.run.return_value = mock_result

        detector = CommunityDetector(store=store)
        result = detector.get_community_for_entity("UnknownEntity")
        assert result is None

    def test_entity_with_community_returns_community(self):
        store, session = _make_mock_store()

        mock_result = MagicMock()
        mock_result.single.return_value = {"cid": 7}
        session.run.side_effect = [
            mock_result,  # first query: get cid
            [
                {"name": "BERT", "entity_type": "MODEL"},
                {"name": "GPT-2", "entity_type": "MODEL"},
            ],  # second query: members
        ]

        detector = CommunityDetector(store=store)
        community = detector.get_community_for_entity("BERT")
        assert community is not None
        assert community.community_id == 7
        assert community.size == 2

    def test_entity_with_none_cid_returns_none(self):
        store, session = _make_mock_store()

        mock_result = MagicMock()
        mock_result.single.return_value = {"cid": None}
        session.run.return_value = mock_result

        detector = CommunityDetector(store=store)
        result = detector.get_community_for_entity("SomeEntity")
        assert result is None


class TestStoreCommunityIdsMocked:
    def test_runs_neo4j_update_per_community(self):
        store, session = _make_mock_store()
        session.run.return_value = None

        communities = [
            Community(
                community_id=0,
                entities=[{"name": "BERT", "entity_type": "MODEL"}, {"name": "GPT-2", "entity_type": "MODEL"}],
                size=2,
            ),
            Community(
                community_id=1,
                entities=[{"name": "ResNet", "entity_type": "MODEL"}],
                size=1,
            ),
        ]
        detector = CommunityDetector(store=store)
        detector._store_community_ids(communities)

        # session.run should be called once per community
        assert session.run.call_count == len(communities)
