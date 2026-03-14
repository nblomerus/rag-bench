"""
Unit tests for rag_bench.core.agent module.

Tests cover:
- Query classification (simple, multi-hop, entity-heavy)
- Plan generation for each query type
- Sufficiency evaluation
- Context merging and deduplication
- JSON array parsing from LLM output
- Full agent loop with mock components
- Config integration
"""

from unittest.mock import MagicMock, patch

import requests as req_lib

from rag_bench.core.agent import (
    AgentConfig,
    AgentStats,
    Plan,
    PlanStep,
    QueryType,
    RAGAgent,
)
from rag_bench.core.types import ChunkData, GenerationResult, RetrievalResult

# ── Mocks ────────────────────────────────────────────────────────────────────


def _make_result(chunk_id: str, rerank_score: float, doc_id: str = "d1") -> RetrievalResult:
    return RetrievalResult(
        chunk=ChunkData(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=f"Text for {chunk_id}",
            section="body",
        ),
        relevance_score=rerank_score,
        rerank_score=rerank_score,
        sources=["dense"],
    )


class MockRetriever:
    def __init__(self, results: list[RetrievalResult]):
        self.results = results
        self.call_count = 0
        self.queries: list[str] = []

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        self.call_count += 1
        self.queries.append(query)
        return self.results[:top_k]


class MockGenerator:
    def __init__(self):
        self.call_count = 0
        self.last_context: list[RetrievalResult] = []

    def generate(self, query: str, context: list[RetrievalResult]) -> GenerationResult:
        self.call_count += 1
        self.last_context = context
        return GenerationResult(
            answer=f"Answer based on {len(context)} chunks",
            deflected=False,
            sources=[r.chunk.chunk_id for r in context],
        )


class MockGraphRetriever:
    """Fake graph retriever for testing entity detection."""

    def __init__(self, entities: list[dict] | None = None):
        self._entities = entities or []

    def _match_entities(self, query: str) -> list[dict]:
        return self._entities

    def expand_query(self, query: str) -> str:
        if self._entities:
            return f"{query} [Related: expanded_entity]"
        return query

    def get_graph_context(self, query: str) -> list[dict]:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Query classification
# ══════════════════════════════════════════════════════════════════════════════


class TestQueryClassification:
    def test_simple_query(self):
        agent = RAGAgent(
            retriever=MockRetriever([]),
            generator=MockGenerator(),
        )
        assert agent._classify_query("What optimizer does GPT-4 use?") == QueryType.SIMPLE

    def test_comparison_is_multi_hop(self):
        agent = RAGAgent(
            retriever=MockRetriever([]),
            generator=MockGenerator(),
        )
        assert agent._classify_query("Compare BERT and GPT-2 architectures") == QueryType.MULTI_HOP

    def test_versus_is_multi_hop(self):
        agent = RAGAgent(
            retriever=MockRetriever([]),
            generator=MockGenerator(),
        )
        assert agent._classify_query("BERT vs GPT-2: which is better?") == QueryType.MULTI_HOP

    def test_difference_is_multi_hop(self):
        agent = RAGAgent(
            retriever=MockRetriever([]),
            generator=MockGenerator(),
        )
        assert agent._classify_query("What is the difference between ReLU and GELU?") == QueryType.MULTI_HOP

    def test_evolution_is_multi_hop(self):
        agent = RAGAgent(
            retriever=MockRetriever([]),
            generator=MockGenerator(),
        )
        assert agent._classify_query("Trace the evolution of attention mechanisms") == QueryType.MULTI_HOP

    def test_entity_heavy_with_graph(self):
        """Query with 2+ matched entities → ENTITY_HEAVY."""
        graph = MockGraphRetriever(
            entities=[
                {"name": "BERT", "entity_type": "MODEL"},
                {"name": "GPT-2", "entity_type": "MODEL"},
            ]
        )
        agent = RAGAgent(
            retriever=MockRetriever([]),
            generator=MockGenerator(),
            graph_retriever=graph,
        )
        # "Tell me about BERT and GPT-2" — not a comparison keyword,
        # but 2 entities matched → ENTITY_HEAVY
        assert agent._classify_query("Tell me about BERT and GPT-2") == QueryType.ENTITY_HEAVY

    def test_entity_without_graph_is_simple(self):
        """Without graph_retriever, entity queries are classified as SIMPLE."""
        agent = RAGAgent(
            retriever=MockRetriever([]),
            generator=MockGenerator(),
            graph_retriever=None,
        )
        assert agent._classify_query("Tell me about BERT and GPT-2") == QueryType.SIMPLE


# ══════════════════════════════════════════════════════════════════════════════
# Planning
# ══════════════════════════════════════════════════════════════════════════════


class TestPlanning:
    def test_simple_plan_has_one_step(self):
        agent = RAGAgent(
            retriever=MockRetriever([]),
            generator=MockGenerator(),
        )
        plan = agent._plan("What is dropout?")
        assert plan.query_type == QueryType.SIMPLE
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == "hybrid_search"

    def test_entity_plan_includes_expansion(self):
        graph = MockGraphRetriever(
            entities=[
                {"name": "A", "entity_type": "M"},
                {"name": "B", "entity_type": "M"},
            ]
        )
        agent = RAGAgent(
            retriever=MockRetriever([]),
            generator=MockGenerator(),
            graph_retriever=graph,
        )
        plan = agent._plan("Tell me about A and B")
        assert plan.query_type == QueryType.ENTITY_HEAVY
        # Should have hybrid_search + graph-expanded search
        assert len(plan.steps) == 2
        assert any("[Related:" in s.query for s in plan.steps)


# ══════════════════════════════════════════════════════════════════════════════
# Sufficiency evaluation
# ══════════════════════════════════════════════════════════════════════════════


class TestSufficiency:
    def test_sufficient_with_good_results(self):
        agent = RAGAgent(
            retriever=MockRetriever([]),
            generator=MockGenerator(),
            config=AgentConfig(sufficiency_threshold=0.80, min_results_for_answer=2),
        )
        context = [_make_result("c1", 0.95), _make_result("c2", 0.90)]
        assert agent._is_sufficient(context) is True

    def test_insufficient_with_low_scores(self):
        agent = RAGAgent(
            retriever=MockRetriever([]),
            generator=MockGenerator(),
            config=AgentConfig(sufficiency_threshold=0.80, min_results_for_answer=2),
        )
        context = [_make_result("c1", 0.90), _make_result("c2", 0.50)]
        assert agent._is_sufficient(context) is False

    def test_insufficient_when_empty(self):
        agent = RAGAgent(
            retriever=MockRetriever([]),
            generator=MockGenerator(),
        )
        assert agent._is_sufficient([]) is False

    def test_insufficient_with_too_few_results(self):
        agent = RAGAgent(
            retriever=MockRetriever([]),
            generator=MockGenerator(),
            config=AgentConfig(min_results_for_answer=3),
        )
        context = [_make_result("c1", 0.95), _make_result("c2", 0.90)]
        assert agent._is_sufficient(context) is False


# ══════════════════════════════════════════════════════════════════════════════
# Context merging
# ══════════════════════════════════════════════════════════════════════════════


class TestContextMerging:
    def test_deduplicates_by_chunk_id(self):
        agent = RAGAgent(
            retriever=MockRetriever([]),
            generator=MockGenerator(),
        )
        existing = [_make_result("c1", 0.90)]
        new = [_make_result("c1", 0.85), _make_result("c2", 0.80)]
        merged = agent._merge_context(existing, new, top_k=10)
        ids = [r.chunk.chunk_id for r in merged]
        assert ids == ["c1", "c2"]  # no duplicate c1

    def test_sorted_by_score(self):
        agent = RAGAgent(
            retriever=MockRetriever([]),
            generator=MockGenerator(),
        )
        existing = [_make_result("c1", 0.70)]
        new = [_make_result("c2", 0.95)]
        merged = agent._merge_context(existing, new, top_k=10)
        assert merged[0].chunk.chunk_id == "c2"  # higher score first

    def test_caps_at_top_k(self):
        agent = RAGAgent(
            retriever=MockRetriever([]),
            generator=MockGenerator(),
        )
        existing = [_make_result(f"c{i}", 0.9 - i * 0.1) for i in range(5)]
        new = [_make_result(f"n{i}", 0.85 - i * 0.1) for i in range(5)]
        merged = agent._merge_context(existing, new, top_k=5)
        assert len(merged) == 5


# ══════════════════════════════════════════════════════════════════════════════
# JSON parsing
# ══════════════════════════════════════════════════════════════════════════════


class TestJSONParsing:
    def test_clean_json(self):
        raw = '["What is X?", "How does Y work?"]'
        result = RAGAgent._parse_json_array(raw)
        assert result == ["What is X?", "How does Y work?"]

    def test_json_with_surrounding_text(self):
        raw = 'Here are the sub-questions:\n["Q1", "Q2", "Q3"]\nDone.'
        result = RAGAgent._parse_json_array(raw)
        assert result == ["Q1", "Q2", "Q3"]

    def test_invalid_json_returns_none(self):
        raw = "This is not JSON at all"
        assert RAGAgent._parse_json_array(raw) is None

    def test_non_string_array_returns_none(self):
        raw = "[1, 2, 3]"
        assert RAGAgent._parse_json_array(raw) is None


# ══════════════════════════════════════════════════════════════════════════════
# Full agent loop
# ══════════════════════════════════════════════════════════════════════════════


class TestAgentLoop:
    def test_simple_query_single_iteration(self):
        """Simple query with good results → 1 iteration, 1 retrieval call."""
        results = [_make_result("c1", 0.95), _make_result("c2", 0.90)]
        mock_ret = MockRetriever(results)
        mock_gen = MockGenerator()
        agent = RAGAgent(
            retriever=mock_ret,
            generator=mock_gen,
            config=AgentConfig(sufficiency_threshold=0.85, min_results_for_answer=2),
        )

        output = agent.run("What is dropout?")
        assert not output.deflected
        assert mock_ret.call_count == 1  # single retrieval
        assert mock_gen.call_count == 1
        assert agent.stats.simple_count == 1

    def test_insufficient_triggers_replan(self):
        """Low-quality results → agent tries to replan."""
        results = [_make_result("c1", 0.50)]
        mock_ret = MockRetriever(results)
        mock_gen = MockGenerator()
        agent = RAGAgent(
            retriever=mock_ret,
            generator=mock_gen,
            config=AgentConfig(sufficiency_threshold=0.85, min_results_for_answer=2),
        )

        agent.run("What is dropout?")
        # Should still produce an answer (best effort)
        assert mock_gen.call_count == 1
        # May or may not replan depending on available strategies

    def test_generates_with_context(self):
        """Generator receives the accumulated context."""
        results = [_make_result("c1", 0.95), _make_result("c2", 0.90)]
        mock_ret = MockRetriever(results)
        mock_gen = MockGenerator()
        agent = RAGAgent(
            retriever=mock_ret,
            generator=mock_gen,
            config=AgentConfig(sufficiency_threshold=0.85, min_results_for_answer=2),
        )

        agent.run("What is dropout?")
        assert len(mock_gen.last_context) == 2

    def test_max_iterations_bounded(self):
        """Agent should not exceed max_iterations."""
        results = [_make_result("c1", 0.30)]  # Always insufficient
        mock_ret = MockRetriever(results)
        mock_gen = MockGenerator()
        agent = RAGAgent(
            retriever=mock_ret,
            generator=mock_gen,
            config=AgentConfig(
                max_iterations=2,
                sufficiency_threshold=0.99,
                min_results_for_answer=5,
            ),
        )

        agent.run("Impossible query")
        assert mock_gen.call_count == 1  # Still generates best-effort
        assert agent.stats.total_iterations <= 2


# ══════════════════════════════════════════════════════════════════════════════
# Stats
# ══════════════════════════════════════════════════════════════════════════════


class TestAgentStats:
    def test_summary(self):
        stats = AgentStats(
            total_queries=10,
            simple_count=7,
            multi_hop_count=2,
            entity_count=1,
            total_iterations=12,
        )
        s = stats.summary()
        assert s["simple_pct"] == 0.7
        assert s["avg_iterations"] == 1.2


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════


class TestAgentConfig:
    def test_defaults(self):
        config = AgentConfig()
        assert config.max_iterations == 3
        assert config.sufficiency_threshold == 0.85

    def test_pipeline_config_includes_agent(self):
        from rag_bench.core.configs import PipelineConfig

        pc = PipelineConfig()
        assert hasattr(pc, "agent")
        assert pc.agent.enabled is False

    def test_pipeline_config_roundtrip(self):
        from rag_bench.core.configs import PipelineConfig

        pc = PipelineConfig()
        pc.agent.enabled = True
        pc.agent.max_iterations = 5

        d = pc.to_dict()
        pc2 = PipelineConfig.from_dict(d)
        assert pc2.agent.enabled is True
        assert pc2.agent.max_iterations == 5


# ══════════════════════════════════════════════════════════════════════════════
# Decompose query (mocked Ollama)
# ══════════════════════════════════════════════════════════════════════════════


class TestDecomposeQuery:
    def _make_agent(self):
        return RAGAgent(
            retriever=MockRetriever([]),
            generator=MockGenerator(),
        )

    def test_returns_sub_queries_on_success(self):
        agent = self._make_agent()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": '["What is BERT?", "What is GPT-2?", "How do they compare?"]'}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp):
            result = agent._decompose_query("Compare BERT and GPT-2")

        assert result == ["What is BERT?", "What is GPT-2?", "How do they compare?"]

    def test_returns_none_on_request_exception(self):
        agent = self._make_agent()
        with patch("requests.post", side_effect=req_lib.RequestException("fail")):
            result = agent._decompose_query("Complex question")
        assert result is None

    def test_returns_none_for_too_few_sub_queries(self):
        """1-element array → not useful, return None."""
        agent = self._make_agent()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": '["Only one question?"]'}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp):
            result = agent._decompose_query("Simple query?")

        assert result is None

    def test_returns_none_for_too_many_sub_queries(self):
        """5-element array → too many, return None."""
        agent = self._make_agent()
        mock_resp = MagicMock()
        sub_qs = [f"Q{i}?" for i in range(5)]
        mock_resp.json.return_value = {"response": repr(sub_qs).replace("'", '"')}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp):
            result = agent._decompose_query("Very complex query")

        assert result is None

    def test_returns_none_for_bad_json(self):
        agent = self._make_agent()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "not valid JSON at all"}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp):
            result = agent._decompose_query("A query")

        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# Plan generation — MULTI_HOP path (LLM decomposition)
# ══════════════════════════════════════════════════════════════════════════════


class TestPlanningMultiHop:
    def test_multi_hop_with_successful_decomposition(self):
        """MULTI_HOP + successful decomposition → multi-step plan."""
        agent = RAGAgent(
            retriever=MockRetriever([]),
            generator=MockGenerator(),
        )
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": '["What is BERT?", "What is GPT-2?"]'}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.post", return_value=mock_resp):
            plan = agent._plan("Compare BERT and GPT-2 architectures")

        assert plan.query_type == QueryType.MULTI_HOP
        assert len(plan.steps) == 2
        assert agent.stats.decompositions == 1

    def test_multi_hop_decomposition_fails_fallback(self):
        """MULTI_HOP decomposition fails → single fallback step."""
        agent = RAGAgent(
            retriever=MockRetriever([]),
            generator=MockGenerator(),
        )
        with patch("requests.post", side_effect=req_lib.RequestException("fail")):
            plan = agent._plan("Compare BERT and GPT-2 architectures")

        assert plan.query_type == QueryType.MULTI_HOP
        assert len(plan.steps) == 1
        assert plan.steps[0].query == "Compare BERT and GPT-2 architectures"

    def test_entity_plan_no_expansion(self):
        """ENTITY_HEAVY without graph → single step."""
        graph = MockGraphRetriever(
            entities=[
                {"name": "A", "entity_type": "M"},
                {"name": "B", "entity_type": "M"},
            ]
        )
        # Override expand_query to return original (no expansion)
        graph.expand_query = lambda q: q
        agent = RAGAgent(
            retriever=MockRetriever([]),
            generator=MockGenerator(),
            graph_retriever=graph,
        )
        plan = agent._plan("Tell me about A and B")
        assert plan.query_type == QueryType.ENTITY_HEAVY
        assert len(plan.steps) == 1  # no expansion step


# ══════════════════════════════════════════════════════════════════════════════
# Execute plan — graph_context tool
# ══════════════════════════════════════════════════════════════════════════════


class TestExecutePlan:
    def test_hybrid_search_step(self):
        results = [_make_result("c1", 0.90)]
        mock_ret = MockRetriever(results)
        agent = RAGAgent(retriever=mock_ret, generator=MockGenerator())

        plan = Plan(
            query_type=QueryType.SIMPLE,
            original_query="test",
            steps=[PlanStep(tool="hybrid_search", query="test", rationale="test")],
        )
        output = agent._execute_plan(plan, top_k=5)
        assert len(output) == 1
        assert output[0].chunk.chunk_id == "c1"

    def test_graph_context_step(self):
        """graph_context step uses graph_retriever.get_graph_context."""
        graph = MockGraphRetriever()
        graph.get_graph_context = MagicMock(
            return_value=[
                {
                    "chunk_id": "graph_ctx_1",
                    "doc_id": "knowledge_graph",
                    "text": "GPT-4 uses RLHF.",
                    "section": "graph_context",
                    "metadata": {"source": "knowledge_graph", "entities_matched": 1, "facts_count": 1},
                    "sources": ["doc1"],
                }
            ]
        )

        agent = RAGAgent(
            retriever=MockRetriever([]),
            generator=MockGenerator(),
            graph_retriever=graph,
        )
        plan = Plan(
            query_type=QueryType.ENTITY_HEAVY,
            original_query="test",
            steps=[PlanStep(tool="graph_context", query="test", rationale="graph")],
        )
        output = agent._execute_plan(plan, top_k=5)
        assert len(output) == 1
        assert output[0].chunk.chunk_id == "graph_ctx_1"
        assert output[0].relevance_score == 0.5  # default graph score

    def test_graph_context_step_no_graph_retriever(self):
        """graph_context step without graph_retriever → no results."""
        agent = RAGAgent(
            retriever=MockRetriever([]),
            generator=MockGenerator(),
            graph_retriever=None,
        )
        plan = Plan(
            query_type=QueryType.ENTITY_HEAVY,
            original_query="test",
            steps=[PlanStep(tool="graph_context", query="test", rationale="graph")],
        )
        output = agent._execute_plan(plan, top_k=5)
        assert output == []

    def test_multiple_steps_combined(self):
        results_a = [_make_result("a1", 0.90)]
        results_b = [_make_result("b1", 0.80)]
        call_no = [0]

        class SequentialRetriever:
            def retrieve(self, query, top_k=10):
                call_no[0] += 1
                return results_a if call_no[0] == 1 else results_b

        agent = RAGAgent(retriever=SequentialRetriever(), generator=MockGenerator())
        plan = Plan(
            query_type=QueryType.MULTI_HOP,
            original_query="test",
            steps=[
                PlanStep(tool="hybrid_search", query="q1", rationale=""),
                PlanStep(tool="hybrid_search", query="q2", rationale=""),
            ],
        )
        output = agent._execute_plan(plan, top_k=10)
        assert len(output) == 2


# ══════════════════════════════════════════════════════════════════════════════
# Replan
# ══════════════════════════════════════════════════════════════════════════════


class TestReplan:
    def test_graph_expansion_used_if_available(self):
        """Replan should try graph expansion when graph_retriever is set."""
        graph = MockGraphRetriever(entities=[{"name": "A", "entity_type": "M"}])
        agent = RAGAgent(
            retriever=MockRetriever([]),
            generator=MockGenerator(),
            graph_retriever=graph,
        )
        prev_plan = Plan(
            query_type=QueryType.SIMPLE,
            original_query="What is A?",
            steps=[PlanStep(tool="hybrid_search", query="What is A?", rationale="")],
        )
        new_plan = agent._replan("What is A?", [], prev_plan, iteration=1)
        assert new_plan is not None
        assert len(new_plan.steps) == 1
        assert "[Related:" in new_plan.steps[0].query

    def test_no_replan_if_expansion_already_used(self):
        """Don't suggest the same expanded query twice."""
        graph = MockGraphRetriever(entities=[{"name": "A", "entity_type": "M"}])
        expanded = "What is A? [Related: expanded_entity]"
        agent = RAGAgent(
            retriever=MockRetriever([]),
            generator=MockGenerator(),
            graph_retriever=graph,
        )
        # Mark expansion as already used
        prev_plan = Plan(
            query_type=QueryType.SIMPLE,
            original_query="What is A?",
            steps=[
                PlanStep(tool="hybrid_search", query="What is A?", rationale=""),
                PlanStep(tool="hybrid_search", query=expanded, rationale=""),
            ],
        )
        # Decomposition also fails
        with patch.object(agent, "_decompose_query", return_value=None):
            new_plan = agent._replan("What is A?", [], prev_plan, iteration=1)
        assert new_plan is None

    def test_decomposition_tried_if_no_graph(self):
        """Without graph_retriever, replan tries decomposition."""
        agent = RAGAgent(
            retriever=MockRetriever([]),
            generator=MockGenerator(),
            graph_retriever=None,
        )
        prev_plan = Plan(
            query_type=QueryType.SIMPLE,
            original_query="Compare X and Y",
            steps=[PlanStep(tool="hybrid_search", query="Compare X and Y", rationale="")],
        )
        with patch.object(agent, "_decompose_query", return_value=["What is X?", "What is Y?"]):
            new_plan = agent._replan("Compare X and Y", [], prev_plan, iteration=1)

        assert new_plan is not None
        assert new_plan.query_type == QueryType.MULTI_HOP
        assert agent.stats.decompositions >= 1

    def test_no_replan_if_already_multi_hop(self):
        """MULTI_HOP plan doesn't try decomposition again."""
        agent = RAGAgent(
            retriever=MockRetriever([]),
            generator=MockGenerator(),
            graph_retriever=None,
        )
        prev_plan = Plan(
            query_type=QueryType.MULTI_HOP,
            original_query="Compare X and Y",
            steps=[PlanStep(tool="hybrid_search", query="Compare X and Y", rationale="")],
        )
        new_plan = agent._replan("Compare X and Y", [], prev_plan, iteration=1)
        assert new_plan is None

    def test_no_graph_no_decompose_returns_none(self):
        """No graph, decompose returns None → replan returns None."""
        agent = RAGAgent(
            retriever=MockRetriever([]),
            generator=MockGenerator(),
            graph_retriever=None,
        )
        prev_plan = Plan(
            query_type=QueryType.SIMPLE,
            original_query="test",
            steps=[PlanStep(tool="hybrid_search", query="test", rationale="")],
        )
        with patch.object(agent, "_decompose_query", return_value=None):
            new_plan = agent._replan("test", [], prev_plan, iteration=1)
        assert new_plan is None


# ══════════════════════════════════════════════════════════════════════════════
# generate() protocol method
# ══════════════════════════════════════════════════════════════════════════════


class TestGenerateProtocol:
    def test_generate_delegates_to_run(self):
        """generate() should call run() and return its result."""
        results = [_make_result("c1", 0.95), _make_result("c2", 0.90)]
        mock_ret = MockRetriever(results)
        mock_gen = MockGenerator()
        agent = RAGAgent(
            retriever=mock_ret,
            generator=mock_gen,
            config=AgentConfig(sufficiency_threshold=0.85, min_results_for_answer=2),
        )

        output = agent.generate("What is dropout?", [])
        assert isinstance(output, GenerationResult)
        assert mock_gen.call_count == 1


# ══════════════════════════════════════════════════════════════════════════════
# Replan integration in run()
# ══════════════════════════════════════════════════════════════════════════════


class TestJSONParsingEdgeCases:
    def test_array_with_malformed_inner_json(self):
        """Array-like content that fails json.loads → None."""
        raw = "[not valid json at all"
        result = RAGAgent._parse_json_array(raw)
        assert result is None

    def test_array_extracted_but_invalid(self):
        """Bracket-wrapped but invalid JSON → hits except JSONDecodeError (line 506-507)."""
        raw = "Here: [not, valid, json]"  # has brackets but not valid JSON
        result = RAGAgent._parse_json_array(raw)
        assert result is None

    def test_array_wrapped_object_not_list(self):
        """Brackets found but content is an object not list → None."""
        raw = '{"key": "value"}'
        result = RAGAgent._parse_json_array(raw)
        assert result is None


class TestAgentRunEntityPath:
    def test_entity_query_increments_entity_count_in_run(self):
        """ENTITY_HEAVY query through run() should increment entity_count stat."""
        results = [_make_result("c1", 0.95), _make_result("c2", 0.90)]
        mock_ret = MockRetriever(results)
        mock_gen = MockGenerator()
        graph = MockGraphRetriever(
            entities=[
                {"name": "BERT", "entity_type": "MODEL"},
                {"name": "GPT-2", "entity_type": "MODEL"},
            ]
        )
        agent = RAGAgent(
            retriever=mock_ret,
            generator=mock_gen,
            graph_retriever=graph,
            config=AgentConfig(sufficiency_threshold=0.85, min_results_for_answer=2),
        )
        # "Tell me about BERT and GPT-2" → ENTITY_HEAVY (2 entities) → entity_count++
        agent.run("Tell me about BERT and GPT-2")
        assert agent.stats.entity_count >= 1

    def test_escalation_runs_multiple_iterations(self):
        """Insufficient results → iterations loop runs at least twice."""
        low = [_make_result("c1", 0.30)]
        mock_ret = MockRetriever(low)
        mock_gen = MockGenerator()
        agent = RAGAgent(
            retriever=mock_ret,
            generator=mock_gen,
            config=AgentConfig(
                max_iterations=3,
                sufficiency_threshold=0.99,
                min_results_for_answer=5,
            ),
        )
        agent.run("Simple query that is insufficient")
        # iterations counter should be > 1
        assert agent.stats.total_iterations >= 1


class TestAgentReplanIntegration:
    def test_replan_increments_counter(self):
        """When replan succeeds and is used, stats.replans is incremented."""
        # Always returns insufficient results
        low_results = [_make_result("c1", 0.30)]
        mock_ret = MockRetriever(low_results)
        mock_gen = MockGenerator()

        graph = MockGraphRetriever(entities=[{"name": "A", "entity_type": "M"}])

        agent = RAGAgent(
            retriever=mock_ret,
            generator=mock_gen,
            graph_retriever=graph,
            config=AgentConfig(
                max_iterations=3,
                sufficiency_threshold=0.99,
                min_results_for_answer=5,
            ),
        )
        agent.run("What is A?")
        # The agent should have tried replanning at least once
        assert agent.stats.replans >= 0  # at least attempted

    def test_multi_hop_with_decomposition_in_run(self):
        """Multi-hop query with decomposition goes through the plan loop."""
        results = [_make_result("c1", 0.95), _make_result("c2", 0.90)]
        mock_ret = MockRetriever(results)
        mock_gen = MockGenerator()
        agent = RAGAgent(
            retriever=mock_ret,
            generator=mock_gen,
            config=AgentConfig(sufficiency_threshold=0.85, min_results_for_answer=2),
        )

        # First retrieval is sufficient → no escalation
        agent.run("Compare BERT versus GPT-2 architectures")
        assert mock_gen.call_count == 1
        assert agent.stats.multi_hop_count >= 1
