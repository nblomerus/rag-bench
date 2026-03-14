"""
community_detection.py — Louvain community detection over the knowledge graph.

Exports the Neo4j graph to NetworkX, runs weighted Louvain community
detection, stores community IDs back as node properties in Neo4j,
and optionally generates LLM summaries for each community.

Communities group tightly-connected entities — e.g., {Transformer,
self-attention, multi-head attention} form a natural cluster.  These
communities serve two purposes in GraphRAG:

1. **Thematic retrieval**: When a query matches an entity, we can also
   return its community summary as context (broader topic overview).
2. **Graph-level summaries**: The community summaries can answer
   high-level questions like "What are the main research themes?"
   without needing to traverse the full graph.

Usage:
    detector = CommunityDetector(store=graph_store)
    communities = detector.detect()           # Run Louvain + store in Neo4j
    detector.generate_summaries(communities)  # LLM summaries via Ollama
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import requests
from networkx.algorithms.community import louvain_communities

from rag_bench.core.graph_store import GraphStore

logger = logging.getLogger(__name__)


@dataclass
class Community:
    """A detected community of entities."""

    community_id: int
    entities: list[dict]  # [{name, entity_type}, ...]
    size: int
    summary: str = ""
    top_predicates: list[str] = field(default_factory=list)


@dataclass
class CommunityDetectorConfig:
    """Configuration for community detection."""

    resolution: float = 1.0  # Louvain resolution (higher = more communities)
    min_community_size: int = 3  # Skip tiny communities
    max_summary_entities: int = 10  # Max entities to show in summary prompt
    ollama_model: str = "qwen2.5:14b-instruct-q4_K_M"
    ollama_base_url: str = "http://localhost:11434"
    cache_path: str = ".community_cache/summaries.json"


class CommunityDetector:
    """Detect and summarize communities in the knowledge graph.

    Parameters
    ----------
    store : GraphStore
        An open Neo4j connection.
    config : CommunityDetectorConfig
        Tuning parameters.
    """

    def __init__(
        self,
        store: GraphStore,
        config: CommunityDetectorConfig | None = None,
    ):
        self.store = store
        self.config = config or CommunityDetectorConfig()

    # -- Public API --------------------------------------------------------

    def detect(self) -> list[Community]:
        """Run community detection and store results in Neo4j.

        Returns a list of Community objects sorted by size (largest first).
        """
        t0 = time.time()

        # Step 1: Export to NetworkX
        G = self._export_to_networkx()
        logger.info(f"Exported graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        if G.number_of_nodes() < self.config.min_community_size:
            logger.warning("Graph too small for community detection")
            return []

        # Step 2: Run Louvain
        partition = louvain_communities(
            G,
            weight="weight",
            resolution=self.config.resolution,
            seed=42,
        )

        # Step 3: Build Community objects
        communities: list[Community] = []
        for cid, members in enumerate(partition):
            if len(members) < self.config.min_community_size:
                continue

            entities = []
            for name_lower in members:
                node_data = G.nodes[name_lower]
                entities.append(
                    {
                        "name": node_data.get("name", name_lower),
                        "entity_type": node_data.get("entity_type", "UNKNOWN"),
                    }
                )

            # Sort entities by degree (most connected first)
            entities.sort(
                key=lambda e: G.degree(e["name"].lower(), weight="weight"),
                reverse=True,
            )

            # Find top predicates within this community
            top_preds = self._get_community_predicates(G, members)

            communities.append(
                Community(
                    community_id=cid,
                    entities=entities,
                    size=len(members),
                    top_predicates=top_preds,
                )
            )

        communities.sort(key=lambda c: c.size, reverse=True)

        # Step 4: Store community IDs in Neo4j
        self._store_community_ids(communities)

        elapsed = time.time() - t0
        logger.info(
            f"Detected {len(communities)} communities (min size {self.config.min_community_size}) in {elapsed:.2f}s"
        )

        return communities

    def generate_summaries(self, communities: list[Community]) -> list[Community]:
        """Generate LLM summaries for each community.

        Calls Ollama to produce a 1-2 sentence description of what
        each community represents.  Results are cached on disk.
        """
        cache = self._load_summary_cache()
        generated = 0
        t0 = time.time()

        for community in communities:
            cache_key = str(community.community_id)

            if cache_key in cache:
                community.summary = cache[cache_key]
                continue

            # Build prompt with top entities
            top_entities = community.entities[: self.config.max_summary_entities]
            entity_list = "\n".join(f"- {e['name']} ({e['entity_type']})" for e in top_entities)
            pred_list = ", ".join(community.top_predicates[:5])

            prompt = (
                f"This is a cluster of {community.size} related entities from "
                f"AI/ML research papers. The entities are connected by relationships "
                f"like: {pred_list}.\n\n"
                f"Key entities:\n{entity_list}\n\n"
                f"Write a concise 1-2 sentence summary describing what this "
                f"cluster of entities represents (the research topic or theme). "
                f"Be specific and technical.\n\nSummary:"
            )

            try:
                resp = requests.post(
                    f"{self.config.ollama_base_url}/api/generate",
                    json={
                        "model": self.config.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.2, "num_predict": 100},
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                summary = resp.json().get("response", "").strip()

                if len(summary) > 20:
                    community.summary = summary
                    cache[cache_key] = summary
                    generated += 1

            except requests.RequestException as e:
                logger.warning(f"Failed to generate summary for community {community.community_id}: {e}")

        self._save_summary_cache(cache)

        elapsed = time.time() - t0
        logger.info(f"Generated {generated} community summaries in {elapsed:.1f}s")

        return communities

    # -- Neo4j export/import -----------------------------------------------

    def _export_to_networkx(self) -> nx.Graph:
        """Export the Neo4j Entity graph to an undirected NetworkX graph."""
        G = nx.Graph()

        with self.store._session() as session:
            # Add nodes
            result = session.run(
                "MATCH (e:Entity) RETURN e.name_lower AS id, e.name AS name,        e.entity_type AS entity_type"
            )
            for record in result:
                G.add_node(
                    record["id"],
                    name=record["name"],
                    entity_type=record["entity_type"],
                )

            # Add edges with weight
            result = session.run(
                "MATCH (s:Entity)-[r:RELATED_TO]->(o:Entity) "
                "RETURN s.name_lower AS source, o.name_lower AS target, "
                "       r.weight AS weight, r.predicate AS predicate"
            )
            for record in result:
                src, tgt = record["source"], record["target"]
                if G.has_node(src) and G.has_node(tgt):
                    # Accumulate weight if multiple edges between same pair
                    if G.has_edge(src, tgt):
                        G[src][tgt]["weight"] += record["weight"]
                    else:
                        G.add_edge(
                            src,
                            tgt,
                            weight=record["weight"],
                            predicate=record["predicate"],
                        )

        return G

    def _store_community_ids(self, communities: list[Community]):
        """Write community_id as a node property in Neo4j."""
        with self.store._session() as session:
            for community in communities:
                name_lowers = [e["name"].lower() for e in community.entities]
                session.run(
                    "UNWIND $names AS n MATCH (e:Entity {name_lower: n}) SET e.community_id = $cid",
                    names=name_lowers,
                    cid=community.community_id,
                )

        logger.info(f"Stored community IDs for {len(communities)} communities in Neo4j")

    @staticmethod
    def _get_community_predicates(G: nx.Graph, members: set) -> list[str]:
        """Find the most common predicates within a community."""
        pred_counts: Counter = Counter()

        for u, v, data in G.edges(members, data=True):
            if u in members and v in members:
                pred = data.get("predicate", "UNKNOWN")
                pred_counts[pred] += 1

        return [p for p, _ in pred_counts.most_common(5)]

    # -- Cache -------------------------------------------------------------

    def _load_summary_cache(self) -> dict[str, str]:
        path = Path(self.config.cache_path)
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_summary_cache(self, cache: dict[str, str]):
        path = Path(self.config.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=2))

    # -- Retrieval integration ---------------------------------------------

    def get_community_for_entity(self, entity_name: str) -> Community | None:
        """Look up the community an entity belongs to."""
        with self.store._session() as session:
            result = session.run(
                "MATCH (e:Entity {name_lower: $name}) RETURN e.community_id AS cid",
                name=entity_name.lower(),
            )
            record = result.single()
            if record and record["cid"] is not None:
                cid = record["cid"]
                # Fetch all members of this community
                members = session.run(
                    "MATCH (e:Entity {community_id: $cid}) "
                    "RETURN e.name AS name, e.entity_type AS entity_type "
                    "ORDER BY e.name",
                    cid=cid,
                )
                entities = [dict(r) for r in members]
                return Community(
                    community_id=cid,
                    entities=entities,
                    size=len(entities),
                )
        return None
