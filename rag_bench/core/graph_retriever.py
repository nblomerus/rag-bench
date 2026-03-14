"""
graph_retriever.py — Knowledge graph augmentation for retrieval.

Given a user query, the GraphRetriever:
1. Identifies entity mentions in the query (fast keyword matching against
   the Neo4j entity index — no LLM call needed).
2. Fetches the local neighborhood (1-2 hops) for each matched entity.
3. Formats the graph facts as synthetic "context chunks" that can be
   injected into the existing HybridRetriever's reranking pool.

This is designed as an *augmentation* layer, not a replacement.  The
vector search finds relevant text passages; the graph adds structured
facts and relationships that may not appear verbatim in any single chunk.

Usage:
    graph_ret = GraphRetriever(store=graph_store)
    graph_chunks = graph_ret.get_graph_context("What datasets was GPT-4 evaluated on?")
    results = hybrid_retriever.query(question, inject_chunks=graph_chunks)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from rag_bench.core.configs import GraphStoreConfig
from rag_bench.core.graph_store import GraphStore

# Lazy import to avoid circular dependency
_CommunityDetector = None

logger = logging.getLogger(__name__)


@dataclass
class GraphRetrieverConfig:
    """Configuration for graph-augmented retrieval."""

    max_hops: int = 1  # Neighborhood depth
    max_neighbors: int = 20  # Max neighbors per entity
    max_entities_per_query: int = 5  # Cap entity matches per query
    min_entity_length: int = 3  # Skip very short entity names
    min_edge_weight: int = 1  # Minimum edge weight to include


class GraphRetriever:
    """Augments retrieval with knowledge graph context.

    Parameters
    ----------
    store : GraphStore
        An open connection to the Neo4j graph.
    config : GraphRetrieverConfig
        Tuning parameters for entity matching and traversal.
    """

    def __init__(
        self,
        store: GraphStore | None = None,
        config: GraphRetrieverConfig | None = None,
        graph_store_config: GraphStoreConfig | None = None,
        community_detector=None,
    ):
        self.config = config or GraphRetrieverConfig()

        if store is not None:
            self.store = store
            self._owns_store = False
        else:
            self.store = GraphStore(config=graph_store_config)
            self._owns_store = True

        self._community_detector = community_detector

        # Cache known entity names for fast query matching
        self._entity_cache: dict[str, dict] | None = None

    def close(self):
        if self._owns_store:
            self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # -- Public API --------------------------------------------------------

    def get_graph_context(self, question: str) -> list[dict]:
        """Extract graph context for a query, formatted as injectable chunks.

        Returns a list of chunk-like dicts compatible with
        HybridRetriever.query(inject_chunks=...).
        """
        t0 = time.time()

        # Step 1: Find entities mentioned in the query
        matched_entities = self._match_entities(question)
        if not matched_entities:
            logger.debug(f"No graph entities matched for: {question[:80]}")
            return []

        logger.info(f"Graph entities matched: {[e['name'] for e in matched_entities]}")

        # Step 2: Fetch neighborhood for each entity
        all_facts: list[str] = []
        source_doc_ids: set[str] = set()

        for entity in matched_entities:
            facts, doc_ids = self._fetch_entity_context(entity["name"])
            all_facts.extend(facts)
            source_doc_ids.update(doc_ids)

        # Step 2b: Fetch community summaries for matched entities
        community_summaries: list[str] = []
        if self._community_detector is not None:
            for entity in matched_entities:
                community = self._community_detector.get_community_for_entity(entity["name"])
                if community and community.summary:
                    community_summaries.append(community.summary)

        if not all_facts and not community_summaries:
            return []

        # Step 3: Format as a synthetic chunk
        graph_text = self._format_graph_chunk(all_facts, matched_entities, community_summaries)

        chunk = {
            "chunk_id": "graph_context_"
            + "_".join(e["name"].lower().replace(" ", "_")[:20] for e in matched_entities[:3]),
            "doc_id": "knowledge_graph",
            "text": graph_text,
            "section": "graph_context",
            "metadata": {
                "source": "knowledge_graph",
                "entities_matched": len(matched_entities),
                "facts_count": len(all_facts),
            },
            "source": "injection",
            "sources": list(source_doc_ids)[:10],
        }

        elapsed = time.time() - t0
        logger.info(f"Graph context: {len(all_facts)} facts for {len(matched_entities)} entities in {elapsed:.3f}s")

        return [chunk]

    def expand_query(self, question: str) -> str:
        """Expand a query with related entity names from the graph.

        Appends neighbor entity names to the query string so that BM25
        and dense retrieval can match chunks mentioning related concepts.
        For example: "How does attention work?" might become
        "How does attention work? [Related: Transformer, multi-head, BERT]"

        This is more effective than chunk injection because it influences
        the first-stage retrieval (BM25 + dense) rather than competing
        in the reranking stage.
        """
        matched_entities = self._match_entities(question)
        if not matched_entities:
            return question

        # Collect neighbor names
        related_names: set[str] = set()
        for entity in matched_entities:
            neighbors = self.store.get_neighbors(
                entity["name"],
                max_hops=1,
                limit=self.config.max_neighbors,
            )
            for n in neighbors:
                name = n.get("name", "")
                if name and name.lower() not in question.lower():
                    related_names.add(name)

        if not related_names:
            return question

        # Append as a hint (capped to avoid overwhelming the query)
        expansion = " ".join(list(related_names)[:8])
        expanded = f"{question} [Related: {expansion}]"
        logger.debug(f"Query expanded: {question[:50]} → +{len(related_names)} entities")
        return expanded

    # -- Entity matching ---------------------------------------------------

    def _match_entities(self, question: str) -> list[dict]:
        """Find known entities mentioned in the query.

        Uses a simple but effective approach: tokenize the query into
        n-grams (1-4 words) and check each against the entity index.
        This is fast (~1ms) and doesn't need an LLM call.
        """
        entity_index = self._get_entity_cache()
        if not entity_index:
            return []

        question_lower = question.lower()
        # Clean the question for matching
        question_clean = re.sub(r"[^\w\s\-.]", " ", question_lower)
        words = question_clean.split()

        matched: dict[str, dict] = {}  # name_lower -> entity dict

        # Check n-grams from longest to shortest (prefer longer matches)
        for n in range(min(4, len(words)), 0, -1):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i : i + n])

                if len(ngram) < self.config.min_entity_length:
                    continue

                if ngram in entity_index and ngram not in matched:
                    matched[ngram] = entity_index[ngram]

                    if len(matched) >= self.config.max_entities_per_query:
                        return list(matched.values())

        return list(matched.values())

    def _get_entity_cache(self) -> dict[str, dict]:
        """Load or return cached entity name index.

        Queries Neo4j once, then caches in memory.  For a graph with
        ~100K entities this uses ~10MB of memory.
        """
        if self._entity_cache is not None:
            return self._entity_cache

        logger.info("Building entity name cache from Neo4j...")
        t0 = time.time()

        self._entity_cache = {}
        with self.store._session() as session:
            result = session.run(
                "MATCH (e:Entity) RETURN e.name AS name, e.name_lower AS name_lower,        e.entity_type AS entity_type"
            )
            for record in result:
                name_lower = record["name_lower"]
                if len(name_lower) >= self.config.min_entity_length:
                    self._entity_cache[name_lower] = {
                        "name": record["name"],
                        "name_lower": name_lower,
                        "entity_type": record["entity_type"],
                    }

        elapsed = time.time() - t0
        logger.info(f"Entity cache built: {len(self._entity_cache)} entities in {elapsed:.2f}s")

        return self._entity_cache

    # -- Graph traversal ---------------------------------------------------

    def _fetch_entity_context(self, entity_name: str) -> tuple[list[str], set[str]]:
        """Fetch structured facts about an entity from the graph.

        Returns (list of natural-language fact strings, set of source doc_ids).
        """
        facts: list[str] = []
        doc_ids: set[str] = set()

        triples = self.store.get_entity_triples(
            entity_name,
            limit=self.config.max_neighbors,
        )

        for t in triples:
            if t["weight"] < self.config.min_edge_weight:
                continue

            predicate = t["predicate"].replace("_", " ").lower()
            fact = f"{t['subject']} ({t['subject_type']}) {predicate} {t['object']} ({t['object_type']})"
            facts.append(fact)

            if t.get("source_doc_ids"):
                doc_ids.update(t["source_doc_ids"])

        return facts, doc_ids

    # -- Formatting --------------------------------------------------------

    @staticmethod
    def _format_graph_chunk(
        facts: list[str],
        entities: list[dict],
        community_summaries: list[str] | None = None,
    ) -> str:
        """Format graph facts as a readable text chunk.

        This text gets embedded and reranked alongside real chunks,
        so it needs to be natural-language-ish for the cross-encoder.
        """
        entity_names = ", ".join(e["name"] for e in entities)

        lines = [
            f"Knowledge graph facts about: {entity_names}",
            "",
        ]

        # Community context (thematic overview)
        if community_summaries:
            seen_summaries = set()
            for summary in community_summaries:
                if summary not in seen_summaries:
                    lines.append(f"Research context: {summary}")
                    seen_summaries.add(summary)
            lines.append("")

        # Individual facts
        seen = set()
        for fact in facts:
            if fact not in seen:
                lines.append(f"- {fact}")
                seen.add(fact)

        return "\n".join(lines)
