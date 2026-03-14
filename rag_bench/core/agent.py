"""
agent.py — Agentic RAG orchestrator.

Wraps the retrieval and generation pipeline in an agent loop that can:
1. Classify query complexity (simple vs multi-hop vs unanswerable)
2. Plan retrieval strategy (which tools, in what order)
3. Execute retrieval steps with multiple tools
4. Evaluate sufficiency (via CRAG confidence scoring)
5. Reflect and replan if context is insufficient
6. Generate a grounded answer with citations

Design decisions:
- Rule-based query classification (no LLM call) for simple/entity queries
- LLM-based decomposition only for detected multi-hop queries
- CRAG confidence scoring reused for sufficiency evaluation
- Max 3 iterations to bound latency
- Conforms to Generator protocol for drop-in benchmarking

Usage:
    agent = RAGAgent(retriever=hybrid, generator=rag_gen, graph_retriever=graph_ret)
    result = agent.run("Compare attention in BERT vs GPT-2")
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum

import requests

from rag_bench.core.types import ChunkData, GenerationResult, RetrievalResult

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Types
# ══════════════════════════════════════════════════════════════════════════════


class QueryType(Enum):
    """Classification of query complexity."""

    SIMPLE = "simple"  # Single-hop factual: "What optimizer does GPT-4 use?"
    MULTI_HOP = "multi_hop"  # Requires combining info: "Compare X and Y"
    ENTITY_HEAVY = "entity"  # Entity-focused: "What papers cite Transformer?"


@dataclass
class PlanStep:
    """A single retrieval step in the agent's plan."""

    tool: str  # "hybrid_search", "graph_search", "decomposed_search"
    query: str  # The query to execute
    rationale: str = ""  # Why this step (for logging/debugging)


@dataclass
class Plan:
    """The agent's retrieval plan."""

    query_type: QueryType
    steps: list[PlanStep]
    original_query: str


@dataclass
class AgentConfig:
    """Configuration for the RAG agent."""

    max_iterations: int = 3
    sufficiency_threshold: float = 0.85  # Min top rerank score to consider sufficient
    min_results_for_answer: int = 2  # Need at least N results above threshold

    # LLM for decomposition / reflection
    ollama_model: str = "qwen2.5:14b-instruct-q4_K_M"
    ollama_base_url: str = "http://localhost:11434"

    # Multi-hop detection keywords
    comparison_keywords: tuple = (
        "compare",
        "difference",
        "versus",
        "vs",
        "contrast",
        "better",
        "worse",
        "advantage",
        "disadvantage",
        "how does .* differ",
        "what is the difference",
    )
    multi_hop_keywords: tuple = (
        "relationship between",
        "how does .* relate",
        "what led to",
        "evolution of",
        "trace the",
        "step by step",
        "explain how .* affects",
    )


@dataclass
class AgentStats:
    """Tracks agent behavior across queries."""

    total_queries: int = 0
    simple_count: int = 0
    multi_hop_count: int = 0
    entity_count: int = 0
    total_iterations: int = 0
    replans: int = 0
    decompositions: int = 0

    def summary(self) -> dict:
        total = max(1, self.total_queries)
        return {
            "total_queries": self.total_queries,
            "simple_pct": self.simple_count / total,
            "multi_hop_pct": self.multi_hop_count / total,
            "entity_pct": self.entity_count / total,
            "avg_iterations": self.total_iterations / total,
            "replans": self.replans,
            "decompositions": self.decompositions,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Agent
# ══════════════════════════════════════════════════════════════════════════════


class RAGAgent:
    """Agentic RAG orchestrator.

    Parameters
    ----------
    retriever
        Base retriever (HybridRetriever or CRAGRetriever).
    generator
        Answer generator (RAGGenerator).
    graph_retriever : optional
        GraphRetriever for entity-aware queries.
    config : AgentConfig
        Agent tuning parameters.
    """

    def __init__(
        self,
        retriever,
        generator,
        graph_retriever=None,
        config: AgentConfig | None = None,
    ):
        self.retriever = retriever
        self.generator = generator
        self.graph_retriever = graph_retriever
        self.config = config or AgentConfig()
        self.stats = AgentStats()

    # -- Public API ---------------------------------------------------------

    def run(self, query: str, top_k: int = 10) -> GenerationResult:
        """Execute the full agent loop: retrieve → evaluate → escalate if needed.

        Key insight: always start with standard retrieval. Only escalate
        to decomposition/graph expansion if the base retrieval is insufficient.
        This prevents diluting good results for queries that don't need help.
        """
        t0 = time.time()
        self.stats.total_queries += 1

        # Classify for stats tracking (but don't act on it yet)
        query_type = self._classify_query(query)
        if query_type == QueryType.SIMPLE:
            self.stats.simple_count += 1
        elif query_type == QueryType.MULTI_HOP:
            self.stats.multi_hop_count += 1
        else:
            self.stats.entity_count += 1

        # Step 1: Always start with standard retrieval
        all_context = self.retriever.retrieve(query, top_k=top_k)
        iteration = 1
        self.stats.total_iterations += 1

        # Step 2: Check sufficiency — only escalate if needed
        if not self._is_sufficient(all_context):
            logger.info(f"Agent: {query_type.value} query, base retrieval insufficient — escalating: {query[:60]}")

            # Build escalation plan based on query type
            plan = self._plan(query)

            for iteration in range(1, self.config.max_iterations):
                self.stats.total_iterations += 1
                new_results = self._execute_plan(plan, top_k)
                all_context = self._merge_context(all_context, new_results, top_k)

                if self._is_sufficient(all_context):
                    break

                # Try replanning with different strategy
                if iteration < self.config.max_iterations - 1:
                    new_plan = self._replan(query, all_context, plan, iteration)
                    if new_plan and new_plan.steps:
                        plan = new_plan
                        self.stats.replans += 1
                    else:
                        break

        # Step 5: Generate
        result = self.generator.generate(query, all_context)

        elapsed = (time.time() - t0) * 1000
        logger.info(f"Agent: {query_type.value} | {iteration} iter | {len(all_context)} chunks | {elapsed:.0f}ms")

        return result

    def generate(self, query: str, context: list[RetrievalResult]) -> GenerationResult:
        """Generator protocol conformance — delegates to run() ignoring pre-built context."""
        return self.run(query)

    # -- Query classification -----------------------------------------------

    def _classify_query(self, query: str) -> QueryType:
        """Classify query complexity using pattern matching.

        Rule-based classification avoids an LLM call on every query.
        Only multi-hop queries need LLM decomposition.
        """
        q_lower = query.lower()

        # Check for comparison/multi-hop patterns
        for pattern in self.config.comparison_keywords:
            if re.search(pattern, q_lower):
                return QueryType.MULTI_HOP

        for pattern in self.config.multi_hop_keywords:
            if re.search(pattern, q_lower):
                return QueryType.MULTI_HOP

        # Check for entity-heavy queries (if graph is available)
        if self.graph_retriever is not None:
            entities = self.graph_retriever._match_entities(query)
            if len(entities) >= 2:
                return QueryType.ENTITY_HEAVY

        return QueryType.SIMPLE

    # -- Planning -----------------------------------------------------------

    def _plan(self, query: str) -> Plan:
        """Create an initial retrieval plan based on query classification."""
        query_type = self._classify_query(query)

        if query_type == QueryType.SIMPLE:
            self.stats.simple_count += 1
            return Plan(
                query_type=query_type,
                original_query=query,
                steps=[
                    PlanStep(
                        tool="hybrid_search",
                        query=query,
                        rationale="Direct retrieval for simple factual query",
                    )
                ],
            )

        if query_type == QueryType.ENTITY_HEAVY:
            self.stats.entity_count += 1
            steps = [
                PlanStep(
                    tool="hybrid_search",
                    query=query,
                    rationale="Hybrid search for passage-level context",
                ),
            ]
            if self.graph_retriever is not None:
                # Use query expansion to boost entity-related retrieval
                expanded = self.graph_retriever.expand_query(query)
                if expanded != query:
                    steps.append(
                        PlanStep(
                            tool="hybrid_search",
                            query=expanded,
                            rationale="Graph-expanded query for related entities",
                        )
                    )
            return Plan(
                query_type=query_type,
                original_query=query,
                steps=steps,
            )

        # MULTI_HOP: decompose into sub-queries
        self.stats.multi_hop_count += 1
        sub_queries = self._decompose_query(query)

        if sub_queries:
            self.stats.decompositions += 1
            steps = [
                PlanStep(
                    tool="hybrid_search",
                    query=sq,
                    rationale=f"Sub-query {i + 1} of {len(sub_queries)}",
                )
                for i, sq in enumerate(sub_queries)
            ]
        else:
            # Decomposition failed — fall back to direct search
            steps = [
                PlanStep(
                    tool="hybrid_search",
                    query=query,
                    rationale="Fallback: direct search (decomposition failed)",
                )
            ]

        return Plan(
            query_type=query_type,
            original_query=query,
            steps=steps,
        )

    def _decompose_query(self, query: str) -> list[str] | None:
        """Use LLM to decompose a complex query into 2-3 simpler sub-queries.

        Returns None if decomposition fails or isn't useful.
        """
        prompt = (
            "Decompose this complex question into 2-3 simpler, independent "
            "sub-questions that together would answer the original. Each "
            "sub-question should be self-contained and searchable.\n\n"
            "Return ONLY a JSON array of strings, nothing else.\n\n"
            f"Question: {query}\n\n"
            "Sub-questions:"
        )

        try:
            resp = requests.post(
                f"{self.config.ollama_base_url}/api/generate",
                json={
                    "model": self.config.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 256},
                },
                timeout=30,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()

            # Parse JSON array
            sub_queries = self._parse_json_array(raw)
            if sub_queries and 2 <= len(sub_queries) <= 4:
                logger.info(f"Agent: Decomposed into {len(sub_queries)} sub-queries")
                return sub_queries

        except (requests.RequestException, ValueError) as e:
            logger.warning(f"Agent: Decomposition failed: {e}")

        return None

    # -- Execution ----------------------------------------------------------

    def _execute_plan(self, plan: Plan, top_k: int) -> list[RetrievalResult]:
        """Execute all steps in the plan, collecting results."""
        all_results: list[RetrievalResult] = []

        for step in plan.steps:
            if step.tool == "hybrid_search":
                results = self.retriever.retrieve(step.query, top_k=top_k)
                all_results.extend(results)
            elif step.tool == "graph_context" and self.graph_retriever is not None:
                chunks = self.graph_retriever.get_graph_context(step.query)
                for c in chunks:
                    all_results.append(
                        RetrievalResult(
                            chunk=ChunkData(
                                chunk_id=c["chunk_id"],
                                doc_id=c.get("doc_id", ""),
                                text=c.get("text", ""),
                                section=c.get("section", ""),
                                metadata=c.get("metadata", {}),
                            ),
                            relevance_score=0.5,  # Default score for graph context
                            sources=c.get("sources", []),
                        )
                    )

        return all_results

    # -- Evaluation ---------------------------------------------------------

    def _is_sufficient(self, context: list[RetrievalResult]) -> bool:
        """Check if retrieved context is sufficient to answer.

        Uses rerank scores as a proxy (same insight as CRAG):
        - Need at least N results above the sufficiency threshold
        """
        if not context:
            return False

        good_results = sum(
            1 for r in context if (r.rerank_score or r.relevance_score) >= self.config.sufficiency_threshold
        )

        return good_results >= self.config.min_results_for_answer

    # -- Replanning ---------------------------------------------------------

    def _replan(
        self,
        query: str,
        context: list[RetrievalResult],
        prev_plan: Plan,
        iteration: int,
    ) -> Plan | None:
        """Generate a new plan based on what we've found so far.

        Strategies tried in order:
        1. If no graph was used → try graph-expanded search
        2. If no decomposition → try decomposing
        3. Otherwise → no new plan (go with best effort)
        """
        prev_queries = {s.query for s in prev_plan.steps}

        # Strategy 1: Try graph expansion if we haven't yet
        if self.graph_retriever is not None:
            expanded = self.graph_retriever.expand_query(query)
            if expanded != query and expanded not in prev_queries:
                return Plan(
                    query_type=prev_plan.query_type,
                    original_query=query,
                    steps=[
                        PlanStep(
                            tool="hybrid_search",
                            query=expanded,
                            rationale="Replan: graph-expanded query",
                        )
                    ],
                )

        # Strategy 2: Try decomposition if we haven't yet
        if prev_plan.query_type != QueryType.MULTI_HOP:
            sub_queries = self._decompose_query(query)
            if sub_queries:
                new_sqs = [sq for sq in sub_queries if sq not in prev_queries]
                if new_sqs:
                    self.stats.decompositions += 1
                    return Plan(
                        query_type=QueryType.MULTI_HOP,
                        original_query=query,
                        steps=[
                            PlanStep(
                                tool="hybrid_search",
                                query=sq,
                                rationale="Replan: decomposed sub-query",
                            )
                            for sq in new_sqs
                        ],
                    )

        return None  # No new strategy available

    # -- Context management -------------------------------------------------

    def _merge_context(
        self,
        existing: list[RetrievalResult],
        new: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Merge new results into existing context, deduplicating."""
        seen_ids = {r.chunk.chunk_id for r in existing}
        merged = list(existing)

        for r in new:
            if r.chunk.chunk_id not in seen_ids:
                seen_ids.add(r.chunk.chunk_id)
                merged.append(r)

        # Sort by score (best first) and cap
        merged.sort(
            key=lambda r: r.rerank_score if r.rerank_score is not None else r.relevance_score,
            reverse=True,
        )

        return merged[:top_k]

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _parse_json_array(raw: str) -> list[str] | None:
        """Parse a JSON array from potentially messy LLM output."""
        # Try direct parse
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and all(isinstance(s, str) for s in parsed):
                return parsed
        except json.JSONDecodeError:
            pass

        # Try extracting array from surrounding text
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list) and all(isinstance(s, str) for s in parsed):
                    return parsed
            except json.JSONDecodeError:
                pass

        return None
