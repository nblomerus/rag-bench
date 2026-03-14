"""
graph_store.py — Neo4j wrapper for GraphRAG knowledge graph storage.

Persists Entity nodes and relationship edges extracted by EntityExtractor.
Uses MERGE semantics so duplicate entities are deduplicated at write time,
and multiple chunks that produce the same triple enrich a single edge with
additional provenance (source_chunk_ids).

Node design
-----------
Each Entity becomes a node with:
- A label matching its type: :MODEL, :DATASET, :METHOD, etc.
- Properties: name, name_lower (for case-insensitive lookup)

Edge design
-----------
Each Triple becomes a relationship typed by its predicate (e.g. -[:USES]->).
Edge properties: source_doc_ids (list), source_chunk_ids (list), weight (int).
Weight counts how many times this triple was extracted across the corpus —
a simple proxy for confidence/importance.

Performance
-----------
Writes are batched (default 500 triples per transaction) via UNWIND for
throughput.  Indexes on (name_lower) are created at init for fast lookups.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

from neo4j import GraphDatabase

from rag_bench.core.configs import GraphStoreConfig
from rag_bench.core.graph_types import Triple

logger = logging.getLogger(__name__)


class GraphStore:
    """Neo4j-backed knowledge graph store.

    Parameters
    ----------
    config : GraphStoreConfig
        Connection URI, credentials, and batch size.
    """

    def __init__(self, config: GraphStoreConfig | None = None):
        self.config = config or GraphStoreConfig()
        self._driver = GraphDatabase.driver(
            self.config.uri,
            auth=(self.config.username, self.config.password),
        )
        self._ensure_indexes()

    def close(self):
        """Close the Neo4j driver and release resources."""
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # -- Schema setup ------------------------------------------------------

    def _ensure_indexes(self):
        """Create indexes for fast entity lookups (idempotent)."""
        # A composite index on (name_lower, entity_type) would be ideal,
        # but Community Edition only supports single-property indexes on
        # node labels.  We create one index per entity type label.
        #
        # We use a generic Entity label that all nodes share, plus the
        # type-specific label.  Index on the generic label's name_lower.
        with self._session() as session:
            session.run("CREATE INDEX entity_name_lower IF NOT EXISTS FOR (e:Entity) ON (e.name_lower)")
            session.run("CREATE INDEX entity_doc_id IF NOT EXISTS FOR (e:Entity) ON (e.doc_id)")
        logger.info("Neo4j indexes ensured")

    # -- Write operations --------------------------------------------------

    def store_triples(self, triples: list[Triple]) -> int:
        """Persist a batch of triples into Neo4j.

        Uses MERGE to deduplicate entities and accumulate provenance on
        edges.  Returns the number of triples written.

        Triples are processed in batches of ``config.batch_size``.
        """
        if not triples:
            return 0

        total = 0
        for i in range(0, len(triples), self.config.batch_size):
            batch = triples[i : i + self.config.batch_size]
            self._write_batch(batch)
            total += len(batch)

        logger.info(f"Stored {total} triples in Neo4j")
        return total

    def _write_batch(self, triples: list[Triple]):
        """Write a batch of triples in a single transaction.

        The Cypher uses UNWIND over a parameter list so the entire batch
        is one round-trip to Neo4j.  MERGE ensures idempotent writes.
        """
        # Serialize triples to parameter dicts
        params = []
        for t in triples:
            params.append(
                {
                    "s_name": t.subject.name,
                    "s_name_lower": t.subject.name.lower(),
                    "s_type": t.subject.entity_type,
                    "predicate": t.predicate,
                    "o_name": t.object.name,
                    "o_name_lower": t.object.name.lower(),
                    "o_type": t.object.entity_type,
                    "chunk_id": t.source_chunk_id,
                    "doc_id": t.source_doc_id,
                    "confidence": t.confidence,
                }
            )

        # We use a single RELATED_TO relationship type and store the
        # predicate as a property.  Neo4j Community doesn't support dynamic
        # relationship types in MERGE, and entity_type is stored as a node
        # property rather than a label for the same reason.
        query = """
        UNWIND $triples AS t
        MERGE (s:Entity {name_lower: t.s_name_lower, entity_type: t.s_type})
        ON CREATE SET s.name = t.s_name
        MERGE (o:Entity {name_lower: t.o_name_lower, entity_type: t.o_type})
        ON CREATE SET o.name = t.o_name
        MERGE (s)-[r:RELATED_TO {predicate: t.predicate}]->(o)
        ON CREATE SET
            r.weight = 1,
            r.source_chunk_ids = [t.chunk_id],
            r.source_doc_ids = [t.doc_id],
            r.confidence_sum = t.confidence
        ON MATCH SET
            r.weight = r.weight + 1,
            r.source_chunk_ids = CASE
                WHEN NOT t.chunk_id IN r.source_chunk_ids
                THEN r.source_chunk_ids + t.chunk_id
                ELSE r.source_chunk_ids
            END,
            r.source_doc_ids = CASE
                WHEN NOT t.doc_id IN r.source_doc_ids
                THEN r.source_doc_ids + t.doc_id
                ELSE r.source_doc_ids
            END,
            r.confidence_sum = r.confidence_sum + t.confidence
        """

        with self._session() as session:
            session.run(query, triples=params)

    # -- Read operations ---------------------------------------------------

    def get_entity(self, name: str) -> dict | None:
        """Look up an entity by name (case-insensitive)."""
        with self._session() as session:
            result = session.run(
                "MATCH (e:Entity {name_lower: $name_lower}) RETURN e.name AS name, e.entity_type AS entity_type",
                name_lower=name.lower(),
            )
            record = result.single()
            if record:
                return {"name": record["name"], "entity_type": record["entity_type"]}
        return None

    def get_neighbors(
        self,
        name: str,
        max_hops: int = 1,
        limit: int = 50,
    ) -> list[dict]:
        """Get entities connected to the given entity within max_hops.

        Returns a list of dicts with: name, entity_type, predicate,
        direction ("out" or "in"), weight, hops.
        """
        with self._session() as session:
            # For 1-hop we use a simple pattern; for multi-hop we use
            # variable-length paths.
            if max_hops == 1:
                result = session.run(
                    """
                    MATCH (e:Entity {name_lower: $name_lower})-[r:RELATED_TO]-(neighbor:Entity)
                    RETURN neighbor.name AS name,
                           neighbor.entity_type AS entity_type,
                           r.predicate AS predicate,
                           r.weight AS weight,
                           CASE WHEN startNode(r) = e THEN 'out' ELSE 'in' END AS direction
                    ORDER BY r.weight DESC
                    LIMIT $limit
                    """,
                    name_lower=name.lower(),
                    limit=limit,
                )
            else:
                result = session.run(
                    f"""
                    MATCH path = (e:Entity {{name_lower: $name_lower}})-[:RELATED_TO*1..{max_hops}]-(neighbor:Entity)
                    WHERE e <> neighbor
                    WITH neighbor, relationships(path) AS rels, length(path) AS hops
                    UNWIND rels AS r
                    RETURN DISTINCT neighbor.name AS name,
                           neighbor.entity_type AS entity_type,
                           r.predicate AS predicate,
                           r.weight AS weight,
                           hops
                    ORDER BY hops ASC, r.weight DESC
                    LIMIT $limit
                    """,
                    name_lower=name.lower(),
                    limit=limit,
                )

            return [dict(record) for record in result]

    def get_entity_triples(
        self,
        name: str,
        limit: int = 50,
    ) -> list[dict]:
        """Get all triples involving an entity (as subject or object).

        Returns dicts with: subject, predicate, object, weight, source_doc_ids.
        """
        with self._session() as session:
            result = session.run(
                """
                MATCH (s:Entity)-[r:RELATED_TO]->(o:Entity)
                WHERE s.name_lower = $name_lower OR o.name_lower = $name_lower
                RETURN s.name AS subject, s.entity_type AS subject_type,
                       r.predicate AS predicate,
                       o.name AS object, o.entity_type AS object_type,
                       r.weight AS weight,
                       r.source_doc_ids AS source_doc_ids
                ORDER BY r.weight DESC
                LIMIT $limit
                """,
                name_lower=name.lower(),
                limit=limit,
            )
            return [dict(record) for record in result]

    def find_path(
        self,
        entity_a: str,
        entity_b: str,
        max_hops: int = 4,
    ) -> list[dict] | None:
        """Find the shortest path between two entities.

        Returns a list of alternating nodes and relationships, or None.
        """
        with self._session() as session:
            result = session.run(
                f"""
                MATCH path = shortestPath(
                    (a:Entity {{name_lower: $a}})-[:RELATED_TO*1..{max_hops}]-(b:Entity {{name_lower: $b}})
                )
                UNWIND relationships(path) AS r
                WITH nodes(path) AS ns, collect({{
                    predicate: r.predicate,
                    weight: r.weight
                }}) AS rels
                UNWIND range(0, size(ns)-1) AS i
                RETURN ns[i].name AS name, ns[i].entity_type AS entity_type
                """,
                a=entity_a.lower(),
                b=entity_b.lower(),
            )
            records = [dict(r) for r in result]
            return records if records else None

    # -- Stats -------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return graph statistics: node count, edge count, top entity types."""
        with self._session() as session:
            node_count = session.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]

            edge_count = session.run("MATCH ()-[r:RELATED_TO]->() RETURN count(r) AS n").single()["n"]

            type_counts = {}
            result = session.run("MATCH (e:Entity) RETURN e.entity_type AS t, count(*) AS n ORDER BY n DESC")
            for record in result:
                type_counts[record["t"]] = record["n"]

            predicate_counts = {}
            result = session.run(
                "MATCH ()-[r:RELATED_TO]->() RETURN r.predicate AS p, count(*) AS n ORDER BY n DESC LIMIT 20"
            )
            for record in result:
                predicate_counts[record["p"]] = record["n"]

        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "entity_types": type_counts,
            "top_predicates": predicate_counts,
        }

    def clear(self):
        """Delete all nodes and relationships. Use with care."""
        with self._session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.warning("Cleared all data from Neo4j")

    # -- Internal ----------------------------------------------------------

    @contextmanager
    def _session(self):
        """Yield a Neo4j session, auto-closing on exit."""
        session = self._driver.session(database=self.config.database)
        try:
            yield session
        finally:
            session.close()
