"""
LLM-as-Judge for answer faithfulness and relevance evaluation.

Uses the existing LLMBackend to evaluate answer quality with structured prompts.
Falls back to keyword heuristics when LLM response cannot be parsed.
"""

import json
import logging
import re

from rag_bench.core.generator import LLMBackend

logger = logging.getLogger(__name__)


FAITHFULNESS_PROMPT = """You are evaluating the faithfulness of a RAG system's answer.

Source passages provided to the system:
{sources}

Answer produced by the system:
{answer}

Rate the faithfulness on a scale of 1-5:
5 = Every factual claim in the answer is directly supported by the source passages
4 = Nearly all claims are supported; minor inferences are reasonable
3 = A mix of supported and unsupported claims
2 = Several claims are not supported by or contradict the sources
1 = The answer is mostly fabricated or contradicts the sources

Respond in exactly this format:
Score: [1-5]
Reasoning: [Your explanation in 1-3 sentences]"""


RELEVANCE_PROMPT = """You are evaluating whether an answer addresses the question asked.

Question: {question}

Answer: {answer}

Rate the relevance on a scale of 1-5:
5 = Directly and completely answers the question
4 = Mostly answers the question with minor gaps
3 = Partially answers the question with significant gaps
2 = Only tangentially related to the question
1 = Does not address the question at all

Respond in exactly this format:
Score: [1-5]
Reasoning: [Your explanation in 1-3 sentences]"""


class JudgeLLM:
    """LLM-based evaluator for answer faithfulness and relevance."""

    def __init__(self, llm_backend: LLMBackend):
        self.llm = llm_backend

    def score_faithfulness(
        self,
        question: str,
        answer: str,
        source_passages: list[str],
    ) -> dict:
        """Rate how faithfully the answer reflects the source passages (1-5)."""
        sources_text = "\n\n".join(f"[Source {i + 1}]: {p}" for i, p in enumerate(source_passages))
        prompt = FAITHFULNESS_PROMPT.format(sources=sources_text, answer=answer)

        for attempt in range(2):
            try:
                raw = self.llm.generate(
                    prompt,
                    system_prompt="You are an evaluation judge. Be concise and precise.",
                    max_tokens=256,
                )
                score, reasoning = self._parse_score(raw)
                return {"score": score, "reasoning": reasoning, "raw_response": raw}
            except Exception as e:
                if attempt == 0:
                    logger.warning(f"Judge faithfulness attempt 1 failed, retrying: {e}")
                else:
                    logger.warning(f"Judge faithfulness failed after retry: {e}")
        return self._faithfulness_heuristic(answer, source_passages)

    def score_relevance(
        self,
        question: str,
        answer: str,
    ) -> dict:
        """Rate how well the answer addresses the question (1-5)."""
        prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)

        for attempt in range(2):
            try:
                raw = self.llm.generate(
                    prompt,
                    system_prompt="You are an evaluation judge. Be concise and precise.",
                    max_tokens=256,
                )
                score, reasoning = self._parse_score(raw)
                return {"score": score, "reasoning": reasoning, "raw_response": raw}
            except Exception as e:
                if attempt == 0:
                    logger.warning(f"Judge relevance attempt 1 failed, retrying: {e}")
                else:
                    logger.warning(f"Judge relevance failed after retry: {e}")
        return {"score": 3.0, "reasoning": "LLM judge failed after retry", "raw_response": ""}

    def _parse_score(self, response: str) -> tuple[float, str]:
        """
        Extract numeric score and reasoning from judge LLM output.

        Tries multiple patterns:
        1. "Score: N" or "Score: N/5"
        2. First line is just a number
        3. JSON-like {"score": N, "reasoning": "..."}

        Returns (score, reasoning). Falls back to (3.0, "Could not parse") on failure.
        """
        response = response.strip()

        # Pattern 1: "Score: N" or "Score: N/5"
        m = re.search(r"Score:\s*(\d+(?:\.\d+)?)\s*(?:/\s*5)?", response, re.IGNORECASE)
        if m:
            score = float(m.group(1))
            score = max(1.0, min(5.0, score))
            # Extract reasoning after "Reasoning:"
            rm = re.search(r"Reasoning:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
            reasoning = rm.group(1).strip() if rm else ""
            return score, reasoning

        # Pattern 2: First line is just a number
        first_line = response.split("\n")[0].strip()
        try:
            score = float(first_line.rstrip("/5"))
            score = max(1.0, min(5.0, score))
            rest = "\n".join(response.split("\n")[1:]).strip()
            return score, rest
        except ValueError:
            pass

        # Pattern 3: JSON
        try:
            data = json.loads(response)
            score = float(data.get("score", 3.0))
            score = max(1.0, min(5.0, score))
            reasoning = str(data.get("reasoning", ""))
            return score, reasoning
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        return 3.0, "Could not parse judge response"

    def _faithfulness_heuristic(
        self,
        answer: str,
        source_passages: list[str],
    ) -> dict:
        """
        Fallback: keyword overlap between answer sentences and source text.
        """
        if not source_passages:
            return {"score": 1.0, "reasoning": "No sources provided", "raw_response": ""}

        source_text = " ".join(source_passages).lower()
        source_words = set(source_text.split())

        answer_sentences = [s.strip() for s in re.split(r"[.!?]+", answer) if s.strip()]
        if not answer_sentences:
            return {"score": 1.0, "reasoning": "Empty answer", "raw_response": ""}

        overlap_scores = []
        for sentence in answer_sentences:
            words = set(sentence.lower().split())
            if not words:
                continue
            overlap = len(words & source_words) / len(words)
            overlap_scores.append(overlap)

        avg_overlap = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0
        # Map 0-1 overlap to 1-5 score
        score = 1.0 + avg_overlap * 4.0
        score = max(1.0, min(5.0, round(score, 1)))

        return {
            "score": score,
            "reasoning": f"Heuristic fallback: {avg_overlap:.0%} keyword overlap with sources",
            "raw_response": "",
        }
