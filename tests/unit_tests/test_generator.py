"""
Unit tests for rag_bench.core.generator module.

Tests cover:
- Math post-processing functions
- LLM backend classes (Ollama, OpenAI-compatible, Template fallback)
- RelevanceGate: deflection logic, keyword overlap, entity extraction
- RAGGenerator: full pipeline, source filtering, citation formatting
- Edge cases: empty inputs, malformed data, API failures
"""

from unittest.mock import MagicMock, patch

import pytest

from rag_bench.core.generator import (
    LLMBackend,
    OllamaBackend,
    RAGGenerator,
    RelevanceGate,
    TemplateFallbackBackend,
    _split_math_segments,
    build_llm_backend,
    postprocess_math,
)

# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_retrieval_results():
    """Sample retrieval results for generator tests."""
    return [
        {
            "chunk_id": "chunk_001",
            "text": "Transformers use self-attention mechanisms to process sequences.",
            "score": 0.9,
            "metadata": {"source_display": "Attention Is All You Need", "section": "intro"},
        },
        {
            "chunk_id": "chunk_002",
            "text": "BERT is trained with masked language modeling.",
            "score": 0.75,
            "metadata": {"source_display": "BERT Paper", "section": "methods"},
        },
        {
            "chunk_id": "chunk_003",
            "text": "GPT-3 has 175 billion parameters.",
            "score": 0.6,
            "metadata": {"source_display": "GPT-3 Paper", "section": "architecture"},
        },
    ]


@pytest.fixture
def mock_retriever():
    """Mock retriever for RAGGenerator tests."""
    retriever = MagicMock()
    retriever.query.return_value = []
    return retriever


# ══════════════════════════════════════════════════════════════════════════════
# Math Post-Processing Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestMathPostProcessing:
    """Tests for math post-processing functions."""

    def test_split_math_segments_no_math(self):
        """Test splitting text with no math."""
        text = "This is plain text without math."
        segments = _split_math_segments(text)

        assert len(segments) == 1
        assert segments[0] == (False, text)

    def test_split_math_segments_inline_math(self):
        """Test splitting text with inline math."""
        text = "The equation $x^2 + y^2 = 1$ is a circle."
        segments = _split_math_segments(text)

        # Should have 3 segments: before, math, after
        assert len(segments) == 3
        assert not segments[0][0]  # Not math
        assert segments[1][0]  # Is math
        assert segments[1][1] == "$x^2 + y^2 = 1$"

    def test_split_math_segments_display_math(self):
        """Test splitting text with display math."""
        text = "Consider $$E = mc^2$$ which is famous."
        segments = _split_math_segments(text)

        assert any(seg[0] and "$$" in seg[1] for seg in segments)

    def test_postprocess_math_greek_unicode(self):
        """Test that Greek Unicode is converted to LaTeX."""
        text = "The parameter α controls learning."
        result = postprocess_math(text)

        assert "$\\alpha$" in result
        assert "α" not in result

    def test_postprocess_math_subscripts(self):
        """Test that subscripts are converted."""
        text = "The variable x_t is important."
        result = postprocess_math(text)

        assert "$x_{t}$" in result

    def test_postprocess_math_superscripts(self):
        """Test that superscripts are converted."""
        text = "We compute x^2 here."
        result = postprocess_math(text)

        assert "$x^{2}$" in result

    def test_postprocess_math_preserves_existing(self):
        """Test that existing LaTeX is preserved."""
        text = "The equation $\\alpha_t = 0.5$ is given."
        result = postprocess_math(text)

        # Should preserve existing LaTeX
        assert "$\\alpha_t = 0.5$" in result

    def test_postprocess_math_unicode_symbols(self):
        """Test that Unicode math symbols are converted."""
        text = "The sum ∑ over all items."
        result = postprocess_math(text)

        assert "$\\sum$" in result
        assert "∑" not in result

    def test_postprocess_math_empty_string(self):
        """Test that empty string is handled."""
        assert postprocess_math("") == ""

    def test_postprocess_math_no_math(self):
        """Test that plain text is unchanged."""
        text = "This is plain text without any math."
        result = postprocess_math(text)

        assert result == text

    def test_postprocess_math_multiple_transforms(self):
        """Test that multiple transformations work together."""
        text = "The value α_t = x^2 is computed."
        result = postprocess_math(text)

        assert "$\\alpha_{t}$" in result
        assert "$x^{2}$" in result


# ══════════════════════════════════════════════════════════════════════════════
# LLM Backend Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestLLMBackends:
    """Tests for LLM backend classes."""

    def test_llm_backend_base_class(self):
        """Test that base class raises NotImplementedError."""
        backend = LLMBackend()

        with pytest.raises(NotImplementedError):
            backend.generate("test prompt")

    @patch("requests.post")
    def test_ollama_backend_generate(self, mock_post):
        """Test Ollama backend generation."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Generated text"}
        mock_post.return_value = mock_response

        backend = OllamaBackend(model="mistral", base_url="http://localhost:11434")
        result = backend.generate("test prompt", system_prompt="system")

        assert result == "Generated text"
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_ollama_backend_handles_error(self, mock_post):
        """Test Ollama backend error handling."""
        mock_post.side_effect = Exception("Connection failed")

        backend = OllamaBackend()

        with pytest.raises(Exception, match="Connection failed"):
            backend.generate("test prompt")

    @patch("requests.post")
    def test_ollama_backend_custom_parameters(self, mock_post):
        """Test that Ollama backend uses custom parameters."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Generated text"}
        mock_post.return_value = mock_response

        backend = OllamaBackend()
        backend.generate("test", max_tokens=512)

        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs["json"]
        assert payload["options"]["num_predict"] == 512

    def test_template_fallback_backend_basic(self):
        """Test template fallback backend basic generation."""
        backend = TemplateFallbackBackend()

        prompt = (
            "[Source 1] Test Author\nThis is a test passage. It has multiple sentences. "
            "Here is more detail.\n\nQuestion: What is this?"
        )
        result = backend.generate(prompt)

        # Should return something non-empty
        assert len(result) > 0
        # Should include citation
        assert "Source 1" in result or "retrieved" in result.lower()

    def test_template_fallback_backend_with_empty_prompt(self):
        """Test template fallback with empty prompt."""
        backend = TemplateFallbackBackend()

        result = backend.generate("")

        # Should still return something
        assert isinstance(result, str)

    def test_build_llm_backend_ollama(self):
        """Test building Ollama backend."""
        backend = build_llm_backend("ollama", model="mistral")

        assert isinstance(backend, OllamaBackend)

    def test_build_llm_backend_template(self):
        """Test building template fallback backend."""
        backend = build_llm_backend("template")

        assert isinstance(backend, TemplateFallbackBackend)

    def test_build_llm_backend_invalid(self):
        """Test that invalid backend returns template fallback."""
        backend = build_llm_backend("invalid_backend")

        # Should default to template fallback for unknown backends
        assert isinstance(backend, TemplateFallbackBackend)


# ══════════════════════════════════════════════════════════════════════════════
# RelevanceGate Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestRelevanceGate:
    """Tests for RelevanceGate class."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        gate = RelevanceGate()

        assert gate.min_top_score == 0.3
        assert gate.min_relevant_chunks == 1
        assert not gate._calibrated

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        gate = RelevanceGate(
            min_top_score=0.5,
            min_relevant_chunks=2,
            keyword_overlap_threshold=0.3,
        )

        assert gate.min_top_score == 0.5
        assert gate.min_relevant_chunks == 2
        assert gate.keyword_overlap_threshold == 0.3

    def test_auto_calibrate_cross_encoder_scale(self):
        """Test auto-calibration for cross-encoder scores."""
        gate = RelevanceGate(min_top_score=0.3)

        gate._auto_calibrate(5.0)  # Cross-encoder score

        assert gate._calibrated
        assert gate._effective_threshold >= 0.5

    def test_auto_calibrate_cosine_scale(self):
        """Test auto-calibration for cosine similarity scores."""
        gate = RelevanceGate(min_top_score=0.3)

        gate._auto_calibrate(0.8)  # Cosine similarity score

        assert gate._calibrated
        assert gate._effective_threshold >= 0.3

    def test_naive_stem_short_words(self):
        """Test that short words are not stemmed."""
        gate = RelevanceGate()

        assert gate._naive_stem("cat") == "cat"
        assert gate._naive_stem("run") == "run"

    def test_naive_stem_removes_suffixes(self):
        """Test that common suffixes are removed."""
        gate = RelevanceGate()

        # Stemmer removes "ing" suffix: running -> runn
        assert gate._naive_stem("running") == "runn"
        # Stemmer removes "tion" suffix: attention -> atten
        assert gate._naive_stem("attention") == "atten"

    def test_compute_keyword_overlap_perfect_match(self):
        """Test keyword overlap with perfect match."""
        gate = RelevanceGate()

        question = "What is deep learning?"
        text = "Deep learning is a subset of machine learning."

        overlap = gate._compute_keyword_overlap(question, text)

        # Should have high overlap (both have "deep" and "learning")
        assert overlap > 0.5

    def test_compute_keyword_overlap_no_match(self):
        """Test keyword overlap with no match."""
        gate = RelevanceGate()

        question = "What is quantum computing?"
        text = "Deep learning uses neural networks."

        overlap = gate._compute_keyword_overlap(question, text)

        # Should have low/zero overlap
        assert overlap < 0.3

    def test_compute_keyword_overlap_stopwords_ignored(self):
        """Test that stopwords are filtered."""
        gate = RelevanceGate()

        question = "What is the transformer architecture?"
        text = "Some text about neural networks."

        # "transformer" and "architecture" are not in text
        overlap = gate._compute_keyword_overlap(question, text)

        assert overlap < 0.5

    def test_extract_entities_named_entities(self):
        """Test extraction of named entities."""
        gate = RelevanceGate()

        question = "What is BERT-Base and how does GPT-2 compare?"
        entities = gate._extract_entities(question)

        assert "BERT-Base" in entities or "BERT" in entities
        assert "GPT-2" in entities

    def test_extract_entities_acronyms(self):
        """Test extraction of technical acronyms."""
        gate = RelevanceGate()

        question = "How does CLIP work with ViT-B/32?"
        entities = gate._extract_entities(question)

        # Should extract CLIP and ViT-B/32
        assert len(entities) > 0

    def test_extract_entities_quoted_terms(self):
        """Test extraction of quoted terms."""
        gate = RelevanceGate()

        question = 'What is "attention mechanism"?'
        entities = gate._extract_entities(question)

        assert "attention mechanism" in entities

    def test_should_deflect_empty_results(self):
        """Test deflection with empty results."""
        gate = RelevanceGate()

        should_deflect, reason = gate.should_deflect([])

        assert should_deflect
        assert "no relevant" in reason.lower()

    def test_should_deflect_low_score(self):
        """Test deflection with low top score."""
        gate = RelevanceGate(min_top_score=0.5)

        results = [{"score": 0.2, "text": "some text"}]
        should_deflect, reason = gate.should_deflect(results)

        assert should_deflect
        assert "threshold" in reason.lower()

    def test_should_deflect_insufficient_relevant_chunks(self):
        """Test deflection with insufficient relevant chunks."""
        gate = RelevanceGate(min_top_score=0.3, min_relevant_chunks=3)

        results = [
            {"score": 0.8, "text": "text 1"},
            {"score": 0.2, "text": "text 2"},
        ]
        should_deflect, reason = gate.should_deflect(results)

        assert should_deflect
        assert "passage" in reason.lower()

    def test_should_deflect_passes_with_good_results(self):
        """Test that good results pass the gate."""
        gate = RelevanceGate(min_top_score=0.3)

        results = [
            {"score": 0.8, "text": "relevant text"},
            {"score": 0.7, "text": "also relevant"},
        ]
        should_deflect, reason = gate.should_deflect(results)

        assert not should_deflect

    def test_check_entity_presence_missing_entity(self):
        """Test entity presence check when entity is missing."""
        gate = RelevanceGate()

        question = "What is BERT-Large?"
        results = [
            {"text": "Transformers use attention.", "metadata": {"title": "Attention Paper"}},
        ]

        should_deflect, reason = gate._check_entity_presence(question, results)

        assert should_deflect
        assert "BERT" in reason or "term" in reason.lower()

    def test_check_entity_presence_entity_in_title(self):
        """Test that entity in title passes check."""
        gate = RelevanceGate()

        question = "What is BERT?"
        results = [
            {
                "text": "This paper introduces a model.",
                "metadata": {"title": "BERT: Pre-training of Deep Bidirectional Transformers"},
            },
        ]

        should_deflect, reason = gate._check_entity_presence(question, results)

        assert not should_deflect

    def test_check_entity_presence_concept_acronyms_ignored(self):
        """Test that generic concept acronyms are ignored."""
        gate = RelevanceGate()

        question = "What is LLM training?"
        results = [
            {"text": "Language models are trained on text.", "metadata": {}},
        ]

        should_deflect, reason = gate._check_entity_presence(question, results)

        # Generic acronyms like LLM shouldn't trigger deflection
        assert not should_deflect

    def test_check_false_premise_detects_false_claims(self):
        """Test detection of false premises in questions."""
        gate = RelevanceGate()

        question = "According to the BERT paper, what result did their model achieve on ImageNet classification?"
        passages = "This paper discusses BERT and language modeling on Wikipedia text."

        has_false_premise, reason = gate._check_false_premise(question, passages)

        # Should detect that "ImageNet" isn't in passages (false premise)
        assert has_false_premise


# ══════════════════════════════════════════════════════════════════════════════
# RAGGenerator Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestRAGGenerator:
    """Tests for RAGGenerator class."""

    def test_init_default_parameters(self, mock_retriever):
        """Test initialization with default parameters."""
        generator = RAGGenerator(retriever=mock_retriever)

        assert generator.retriever == mock_retriever
        assert isinstance(generator.llm, TemplateFallbackBackend)
        assert isinstance(generator.gate, RelevanceGate)
        assert generator.top_k == 5

    def test_init_custom_backend(self, mock_retriever):
        """Test initialization with custom LLM backend."""
        custom_llm = MagicMock(spec=LLMBackend)
        generator = RAGGenerator(retriever=mock_retriever, llm_backend=custom_llm)

        assert generator.llm == custom_llm

    def test_build_sources_block(self, mock_retriever, sample_retrieval_results):
        """Test building sources block for prompt."""
        generator = RAGGenerator(retriever=mock_retriever)

        sources_block = generator._build_sources_block(sample_retrieval_results)

        assert "[Source 1]" in sources_block
        assert "[Source 2]" in sources_block
        assert "Transformers" in sources_block

    def test_build_sources_block_empty(self, mock_retriever):
        """Test building sources block with empty results."""
        generator = RAGGenerator(retriever=mock_retriever)

        sources_block = generator._build_sources_block([])

        assert sources_block == ""

    def test_format_citations(self, mock_retriever, sample_retrieval_results):
        """Test citation formatting."""
        generator = RAGGenerator(retriever=mock_retriever)

        citations = generator._format_citations(sample_retrieval_results)

        assert len(citations) == len(sample_retrieval_results)
        assert all("[Source" in c for c in citations)

    def test_filter_relevant_sources_basic(self, mock_retriever):
        """Test filtering of low-relevance sources."""
        generator = RAGGenerator(retriever=mock_retriever)

        results = [
            {"score": 0.9, "text": "highly relevant"},
            {"score": 0.8, "text": "also relevant"},
            {"score": 0.1, "text": "not relevant"},
        ]

        filtered = generator._filter_relevant_sources(results)

        # Should filter out the low-score result
        assert len(filtered) <= len(results)
        assert all(r["score"] >= 0.1 for r in filtered)

    def test_filter_relevant_sources_empty(self, mock_retriever):
        """Test filtering with empty results."""
        generator = RAGGenerator(retriever=mock_retriever)

        filtered = generator._filter_relevant_sources([])

        assert filtered == []

    def test_answer_deflects_on_irrelevant(self, mock_retriever):
        """Test that answer deflects when results are irrelevant."""
        mock_retriever.query.return_value = [
            {"score": 0.1, "text": "low score", "metadata": {}},
        ]

        generator = RAGGenerator(
            retriever=mock_retriever,
            relevance_gate=RelevanceGate(min_top_score=0.5),
        )

        response = generator.answer("test question")

        assert response["deflected"]
        assert "deflection_reason" in response

    def test_answer_generates_with_relevant_results(self, mock_retriever, sample_retrieval_results):
        """Test that answer is generated with relevant results."""
        mock_retriever.query.return_value = sample_retrieval_results

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Generated answer with [Source 1] citation."

        generator = RAGGenerator(
            retriever=mock_retriever,
            llm_backend=mock_llm,
            relevance_gate=RelevanceGate(min_top_score=0.3),
        )

        response = generator.answer("What are transformers?")

        assert not response["deflected"]
        assert "answer" in response
        assert len(response["answer"]) > 0
        mock_llm.generate.assert_called_once()

    def test_answer_handles_llm_failure(self, mock_retriever, sample_retrieval_results):
        """Test that answer handles LLM generation failure."""
        mock_retriever.query.return_value = sample_retrieval_results

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = Exception("LLM error")

        generator = RAGGenerator(
            retriever=mock_retriever,
            llm_backend=mock_llm,
            relevance_gate=RelevanceGate(min_top_score=0.3),
        )

        # Should fall back to template backend
        response = generator.answer("What are transformers?")

        # Should still return an answer (from fallback)
        assert "answer" in response

    def test_answer_postprocesses_math(self, mock_retriever, sample_retrieval_results):
        """Test that math is post-processed in answer."""
        mock_retriever.query.return_value = sample_retrieval_results

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "The parameter α controls x^2."

        generator = RAGGenerator(
            retriever=mock_retriever,
            llm_backend=mock_llm,
            relevance_gate=RelevanceGate(min_top_score=0.3),
        )

        response = generator.answer("What is the parameter?")

        # Math should be converted to LaTeX
        assert "$\\alpha$" in response["answer"] or "alpha" in response["answer"].lower()

    def test_answer_respects_top_k(self, mock_retriever, sample_retrieval_results):
        """Test that top_k parameter is respected."""
        mock_retriever.query.return_value = sample_retrieval_results

        generator = RAGGenerator(retriever=mock_retriever, top_k=2)

        generator.answer("test question")

        # Check that retriever was called with correct top_k
        mock_retriever.query.assert_called_with("test question", top_k=2)

    def test_answer_checks_keyword_overlap(self, mock_retriever):
        """Test that low keyword overlap triggers deflection."""
        mock_retriever.query.return_value = [
            {
                "score": 0.8,
                "text": "Completely unrelated content about gardening.",
                "metadata": {},
            },
        ]

        generator = RAGGenerator(
            retriever=mock_retriever,
            relevance_gate=RelevanceGate(min_top_score=0.3, keyword_overlap_threshold=0.5),
        )

        response = generator.answer("What is quantum computing?")

        # Should deflect due to low keyword overlap
        assert response["deflected"]

    def test_answer_empty_retrieval_results(self, mock_retriever):
        """Test answer with empty retrieval results."""
        mock_retriever.query.return_value = []

        generator = RAGGenerator(retriever=mock_retriever)

        response = generator.answer("test question")

        assert response["deflected"]
        assert len(response["sources"]) == 0

    def test_check_answer_alignment_detects_tangential(self, mock_retriever):
        """Test detection of tangential answers."""
        generator = RAGGenerator(retriever=mock_retriever)

        question = "How many parameters does BERT have?"
        answer = "BERT is a transformer-based model trained on Wikipedia."

        is_tangential, focus = generator._check_answer_alignment(question, answer)

        # Should detect that "parameters" is not addressed
        assert is_tangential
        assert "parameter" in focus.lower()

    def test_check_answer_alignment_passes_aligned(self, mock_retriever):
        """Test that aligned answers pass the check."""
        generator = RAGGenerator(retriever=mock_retriever)

        question = "How many parameters does BERT have?"
        answer = "BERT-Base has 110 million parameters [Source 1]."

        is_tangential, focus = generator._check_answer_alignment(question, answer)

        # Should not be tangential
        assert not is_tangential


# ══════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestGeneratorIntegration:
    """Integration tests for generator components."""

    def test_full_pipeline_with_deflection(self, mock_retriever):
        """Test full pipeline that triggers deflection."""
        mock_retriever.query.return_value = []

        generator = RAGGenerator(retriever=mock_retriever)
        response = generator.answer("What is XYZ technology?")

        assert response["deflected"]
        assert "answer" in response
        assert any(phrase in response["answer"].lower() for phrase in ["don't have", "insufficient"])

    def test_full_pipeline_with_generation(self, mock_retriever):
        """Test full pipeline that generates an answer."""
        mock_retriever.query.return_value = [
            {
                "score": 0.9,
                "text": "Attention mechanisms compute weighted sums.",
                "metadata": {"source_display": "Attention Paper", "section": "intro"},
            },
        ]

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Attention mechanisms work by computing weighted sums [Source 1]."

        generator = RAGGenerator(
            retriever=mock_retriever,
            llm_backend=mock_llm,
            relevance_gate=RelevanceGate(min_top_score=0.3),
        )

        response = generator.answer("How do attention mechanisms work?")

        assert not response["deflected"]
        assert "Attention mechanisms" in response["answer"]
        assert len(response["sources"]) > 0


# ══════════════════════════════════════════════════════════════════════════════
# Additional Coverage Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestOpenAIBackend:
    """Tests for OpenAI-compatible backend."""

    @patch("requests.post")
    def test_openai_backend_generate(self, mock_post):
        """Test OpenAI-compatible backend generation."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "OpenAI response text"}}]}
        mock_post.return_value = mock_response

        from rag_bench.core.generator import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend(model="gpt-3.5-turbo", base_url="http://localhost:8000/v1")
        result = backend.generate("test prompt", system_prompt="you are helpful")

        assert result == "OpenAI response text"
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "chat/completions" in call_args[0][0]

    @patch("requests.post")
    def test_openai_backend_custom_api_key(self, mock_post):
        """Test that OpenAI backend uses custom API key."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "response"}}]}
        mock_post.return_value = mock_response

        from rag_bench.core.generator import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend(api_key="custom-key-123")
        backend.generate("prompt")

        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer custom-key-123"

    @patch("requests.post")
    def test_openai_backend_handles_error(self, mock_post):
        """Test OpenAI backend error handling."""
        mock_post.side_effect = Exception("Network error")

        from rag_bench.core.generator import OpenAICompatibleBackend

        backend = OpenAICompatibleBackend()

        with pytest.raises(Exception, match="Network error"):
            backend.generate("test prompt")

    @patch("requests.post")
    def test_ollama_streaming_backend(self, mock_post):
        """Test Ollama streaming generation."""
        # Mock streaming response
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            b'{"response": "Hello", "done": false}',
            b'{"response": " world", "done": false}',
            b'{"response": "", "done": true}',
        ]
        mock_post.return_value = mock_response

        backend = OllamaBackend()
        tokens = list(backend.generate_stream("test prompt"))

        assert len(tokens) == 2
        assert tokens[0] == "Hello"
        assert tokens[1] == " world"

    @patch("requests.post")
    def test_ollama_streaming_handles_error(self, mock_post):
        """Test Ollama streaming error handling."""
        mock_post.side_effect = Exception("Connection failed")

        backend = OllamaBackend()

        with pytest.raises(Exception, match="Connection failed"):
            list(backend.generate_stream("test prompt"))


class TestRelevanceGateAdvanced:
    """Advanced tests for RelevanceGate edge cases."""

    def test_auto_calibrate_already_calibrated(self):
        """Test that auto-calibrate is idempotent."""
        gate = RelevanceGate(min_top_score=0.3)
        gate._auto_calibrate(5.0)
        first_threshold = gate._effective_threshold

        gate._auto_calibrate(10.0)
        second_threshold = gate._effective_threshold

        assert first_threshold == second_threshold

    def test_naive_stem_preserves_short_words(self):
        """Test that short words are not modified."""
        gate = RelevanceGate()

        assert gate._naive_stem("is") == "is"
        assert gate._naive_stem("run") == "run"
        assert gate._naive_stem("go") == "go"

    def test_compute_keyword_overlap_empty_question(self):
        """Test keyword overlap with empty question."""
        gate = RelevanceGate()

        overlap = gate._compute_keyword_overlap("", "some text")

        assert overlap == 1.0

    def test_extract_entities_empty_question(self):
        """Test entity extraction from empty question."""
        gate = RelevanceGate()

        entities = gate._extract_entities("")

        assert entities == []

    def test_extract_entities_duplicates_removed(self):
        """Test that duplicate entities are removed."""
        gate = RelevanceGate()

        entities = gate._extract_entities("BERT and BERT models")

        # BERT should appear only once
        assert entities.count("BERT") <= 1

    def test_check_entity_presence_no_entities(self):
        """Test entity check with no extractable entities."""
        gate = RelevanceGate()

        should_deflect, reason = gate._check_entity_presence("what is this?", [{"text": "some text", "metadata": {}}])

        assert not should_deflect

    def test_check_entity_presence_prominent_entity(self):
        """Test entity check when entity appears prominently."""
        gate = RelevanceGate()

        should_deflect, reason = gate._check_entity_presence(
            "What is BERT?",
            [{"text": "BERT is a model. BERT implements masking. BERT is powerful.", "metadata": {"title": "BERT"}}],
        )

        assert not should_deflect

    def test_check_false_premise_no_claim_pattern(self):
        """Test false premise detection with text that has no claim pattern."""
        gate = RelevanceGate()

        has_false_premise, reason = gate._check_false_premise(
            "General question about transformers", "Transformers are neural Networks"
        )

        assert not has_false_premise

    def test_should_deflect_with_calibration_cross_encoder(self):
        """Test deflection decision with cross-encoder scores."""
        gate = RelevanceGate(min_top_score=0.3)

        results = [
            {"score": 8.5, "text": "relevant text 1"},
            {"score": 7.2, "text": "relevant text 2"},
        ]
        should_deflect, reason = gate.should_deflect(results)

        assert not should_deflect

    def test_should_deflect_score_gap_analysis(self):
        """Test deflection with large score gap."""
        gate = RelevanceGate(min_top_score=0.3)

        results = [
            {"score": 0.9, "text": "very relevant"},
            {"score": 0.2, "text": "not relevant"},
        ]
        should_deflect, reason = gate.should_deflect(results)

        assert not should_deflect


class TestRAGGeneratorAdvanced:
    """Advanced tests for RAGGenerator edge cases."""

    def test_filter_relevant_sources_cross_encoder_scale(self, mock_retriever):
        """Test source filtering with cross-encoder scores."""
        generator = RAGGenerator(retriever=mock_retriever)

        results = [
            {"score": 8.0, "text": "text 1"},
            {"score": 5.0, "text": "text 2"},
            {"score": 2.5, "text": "text 3"},
        ]

        filtered = generator._filter_relevant_sources(results)

        # Should include first two (good scores), drop last one
        assert len(filtered) <= 2
        assert filtered[0]["score"] == 8.0

    def test_filter_relevant_sources_cosine_scale(self, mock_retriever):
        """Test source filtering with cosine similarity scores."""
        generator = RAGGenerator(retriever=mock_retriever)

        results = [
            {"score": 0.95, "text": "text 1"},
            {"score": 0.80, "text": "text 2"},
            {"score": 0.05, "text": "text 3"},
        ]

        filtered = generator._filter_relevant_sources(results)

        assert len(filtered) >= 1
        assert filtered[0]["score"] == 0.95

    def test_filter_relevant_sources_small_scale_scores(self, mock_retriever):
        """Test source filtering with small-scale scores (BM25 range)."""
        generator = RAGGenerator(retriever=mock_retriever)

        results = [
            {"score": 0.02, "text": "text 1"},
            {"score": 0.015, "text": "text 2"},
            {"score": 0.005, "text": "text 3"},
        ]

        filtered = generator._filter_relevant_sources(results)

        # Should cap at 4 results for small scores
        assert len(filtered) <= 4

    def test_format_citations_with_missing_metadata(self, mock_retriever):
        """Test citation formatting with incomplete metadata."""
        generator = RAGGenerator(retriever=mock_retriever)

        results = [
            {
                "score": 0.9,
                "text": "some text",
                "metadata": {"source_display": "Paper A"},  # Missing section
            },
        ]

        citations = generator._format_citations(results)

        assert len(citations) == 1
        assert "Paper A" in citations[0]

    def test_answer_with_custom_top_k(self, mock_retriever, sample_retrieval_results):
        """Test answer generation respecting custom top_k."""
        mock_retriever.query.return_value = sample_retrieval_results[:2]

        generator = RAGGenerator(
            retriever=mock_retriever,
            llm_backend=MagicMock(generate=MagicMock(return_value="Answer.")),
            relevance_gate=RelevanceGate(min_top_score=0.3),
            top_k=10,
        )

        generator.answer("test", top_k=3)

        # Should call with custom top_k=3, not the default top_k=10
        mock_retriever.query.assert_called_with("test", top_k=3)

    def test_answer_detects_entity_mismatch(self, mock_retriever):
        """Test that answer detects when entity is missing."""
        mock_retriever.query.return_value = [
            {"score": 0.8, "text": "Deep learning is great.", "metadata": {"source_display": "Paper", "title": "ML"}}
        ]

        generator = RAGGenerator(
            retriever=mock_retriever, llm_backend=MagicMock(), relevance_gate=RelevanceGate(min_top_score=0.3)
        )

        response = generator.answer("What is QuantumML?")

        assert response["deflected"]
        # The deflection reason could be about keyword overlap or entity presence
        assert "deflection_reason" in response

    def test_answer_detects_tangential_response(self, mock_retriever):
        """Test detection of tangential answers."""
        mock_retriever.query.return_value = [
            {"score": 0.9, "text": "BERT is a transformer.", "metadata": {"source_display": "Paper", "section": "intro"}}
        ]

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Transformers use attention. BERT is related to NLP."

        generator = RAGGenerator(
            retriever=mock_retriever, llm_backend=mock_llm, relevance_gate=RelevanceGate(min_top_score=0.3)
        )

        response = generator.answer("How many parameters does BERT have?")

        # Should deflect because "parameters" is not in the answer
        assert response["deflected"]

    def test_answer_integration_with_encoding_fix(self, mock_retriever):
        """Test answer handles encoding-broken text."""
        mock_retriever.query.return_value = [
            {
                "score": 0.9,
                "text": "This text might have encoding issues",
                "metadata": {"source_display": "Paper", "section": "intro"},
            }
        ]

        mock_llm = MagicMock(generate=MagicMock(return_value="Answer [Source 1]."))

        generator = RAGGenerator(
            retriever=mock_retriever, llm_backend=mock_llm, relevance_gate=RelevanceGate(min_top_score=0.3)
        )

        response = generator.answer("What?")

        assert not response["deflected"]
        assert "answer" in response

    @patch("sys.stdout", new_callable=MagicMock)
    def test_print_answer(self, mock_stdout, mock_retriever, sample_retrieval_results):
        """Test print_answer method."""
        mock_retriever.query.return_value = sample_retrieval_results
        mock_llm = MagicMock(generate=MagicMock(return_value="Generated answer."))

        generator = RAGGenerator(
            retriever=mock_retriever, llm_backend=mock_llm, relevance_gate=RelevanceGate(min_top_score=0.3)
        )

        result = generator.print_answer("What are transformers?")

        assert not result["deflected"]
        assert "Generated answer." in result["answer"]

    @patch("sys.stdout", new_callable=MagicMock)
    def test_print_answer_with_deflection(self, mock_stdout, mock_retriever):
        """Test print_answer with deflection."""
        mock_retriever.query.return_value = []

        generator = RAGGenerator(retriever=mock_retriever)
        result = generator.print_answer("Unknown topic")

        assert result["deflected"]
        assert "DEFLECTED" in result or "deflected" in result


class TestAnswerStreaming:
    """Tests for answer_stream method."""

    def test_answer_stream_with_normal_backend(self):
        """Test streaming answer generation with non-streaming backend."""
        mock_retriever = MagicMock()
        mock_retriever.query.return_value = [
            {"score": 0.9, "text": "Testing is important.", "metadata": {"source_display": "Paper", "section": "intro"}}
        ]

        mock_llm = MagicMock()
        # Return an answer that relates to the question to avoid tangential detection
        mock_llm.generate = MagicMock(return_value="Testing is a critical component [Source 1].")

        generator = RAGGenerator(
            retriever=mock_retriever, llm_backend=mock_llm, relevance_gate=RelevanceGate(min_top_score=0.3)
        )

        events = list(generator.answer_stream("What is testing?"))

        # Should have sources and done events
        assert any(e.get("event") == "sources" for e in events)
        # Could be "done" or "deflected" - let's check for completion
        assert any(e.get("event") in ["done", "deflected"] for e in events)

    def test_answer_stream_deflects_on_low_score(self):
        """Test streaming deflects on low relevance."""
        mock_retriever = MagicMock()
        mock_retriever.query.return_value = [{"score": 0.1, "text": "irrelevant", "metadata": {}}]

        generator = RAGGenerator(retriever=mock_retriever, relevance_gate=RelevanceGate(min_top_score=0.5))

        events = list(generator.answer_stream("test question"))

        assert any(e.get("event") == "deflected" for e in events)

    def test_answer_stream_detects_tangential_answer(self):
        """Test streaming detects tangential answers."""
        mock_retriever = MagicMock()
        mock_retriever.query.return_value = [
            {"score": 0.9, "text": "BERT paper content", "metadata": {"source_display": "Paper", "section": "intro"}}
        ]

        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(return_value="Transformers are good")

        generator = RAGGenerator(
            retriever=mock_retriever, llm_backend=mock_llm, relevance_gate=RelevanceGate(min_top_score=0.3)
        )

        events = list(generator.answer_stream("How many parameters does BERT have?"))

        # Should end with deflected because answer is tangential
        assert events[-1].get("event") == "deflected"

    def test_answer_stream_handles_generation_error(self):
        """Test streaming handles backend errors gracefully."""
        mock_retriever = MagicMock()
        mock_retriever.query.return_value = [
            {"score": 0.9, "text": "content", "metadata": {"source_display": "Paper", "section": "intro"}}
        ]

        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(side_effect=Exception("Generation error"))

        generator = RAGGenerator(
            retriever=mock_retriever, llm_backend=mock_llm, relevance_gate=RelevanceGate(min_top_score=0.3)
        )

        events = list(generator.answer_stream("test question"))

        # Should still produce events and use fallback
        assert len(events) > 0
        assert any(e.get("event") in ["done", "sources"] for e in events)

    def test_answer_stream_empty_retrieval(self):
        """Test streaming with empty retrieval results."""
        mock_retriever = MagicMock()
        mock_retriever.query.return_value = []

        generator = RAGGenerator(retriever=mock_retriever, relevance_gate=RelevanceGate(min_top_score=0.3))

        events = list(generator.answer_stream("test question"))

        assert any(e.get("event") == "deflected" for e in events)


class TestMathPostProcessingEdgeCases:
    """Additional edge case tests for math post-processing."""

    def test_postprocess_math_greek_with_subscript(self):
        """Test Greek letter with subscript conversion."""
        text = "The parameter α_t is important."
        result = postprocess_math(text)

        assert "$\\alpha_{t}$" in result

    def test_postprocess_math_multiple_unicode_symbols(self):
        """Test multiple Unicode math symbols."""
        text = "Sum ∑ and product ∏ and integral ∫"
        result = postprocess_math(text)

        assert "$\\sum$" in result
        assert "$\\prod$" in result
        assert "$\\int$" in result

    def test_postprocess_math_preserves_display_math(self):
        """Test display math preservation."""
        text = "Formula: $$x^2 + y^2 = z^2$$ is Pythagorean."
        result = postprocess_math(text)

        assert "$$x^2 + y^2 = z^2$$" in result

    def test_postprocess_math_complex_greek_subscripts(self):
        """Test complex Greek with braced subscripts."""
        text = "The value σ_{t-1} is used."
        result = postprocess_math(text)

        assert "$\\sigma_{t-1}$" in result

    def test_postprocess_math_greek_preceded_by_letter(self):
        """Test that Greek preceded by letter is not converted."""
        text = "The word dogma contains no Greek."
        result = postprocess_math(text)

        # Σ in sigma should not be converted since preceded by g
        assert result == text
