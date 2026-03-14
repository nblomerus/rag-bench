"""
entity_extractor.py — LLM-based entity & relation extraction for GraphRAG.

Extracts (entity, relation, entity) triples from paper chunks using a local
LLM via Ollama.  Each chunk is processed independently — the LLM sees the
chunk text plus a light document context (title + abstract) to help it
resolve abbreviations and pronouns.

Triples are cached on disk (one JSON file per paper, keyed by chunk text
hash) so re-running skips already-processed chunks.  This is the same
caching pattern used by ContextualEnricher.

Design notes
------------
* Schema-guided extraction: the prompt lists expected entity types
  (MODEL, DATASET, METHOD, ...) and relation types (USES, OUTPERFORMS, ...)
  to get consistent, queryable output.  The LLM can still use free-form
  types for edge cases.
* JSON output parsing with fallback: tries json.loads first, then regex
  extraction for partial/malformed JSON.
* Caps triples per chunk (default 10) to avoid noisy over-extraction.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path

import requests

from rag_bench.core.configs import ExtractorConfig
from rag_bench.core.graph_types import (
    ENTITY_TYPES,
    RELATION_TYPES,
    Entity,
    ExtractionResult,
    Triple,
)
from rag_bench.core.types import ChunkData

logger = logging.getLogger(__name__)


# -- Prompt ----------------------------------------------------------------

_EXTRACTION_PROMPT = """\
You are an expert at extracting structured knowledge from AI/ML research papers.

Given a chunk from a research paper, extract all meaningful (subject, predicate, object) triples.

**Entity types** (use these when possible):
{entity_types}

**Relation types** (use these when possible):
{relation_types}

**Rules:**
- Entity names should be normalized: use the canonical name (e.g. "Transformer" not "the transformer model")
- Each triple must have: subject (name + type), predicate, object (name + type)
- Only extract factual relationships stated or strongly implied in the text
- Skip vague or speculative statements
- Return at most {max_triples} triples

**Paper context:**
Title: {title}
Abstract: {abstract}

**Chunk to extract from:**
{chunk_text}

Respond with a JSON array of triples. Each triple is an object with keys:
"subject_name", "subject_type", "predicate", "object_name", "object_type"

Example:
[
  {{"subject_name": "GPT-4", "subject_type": "MODEL", "predicate": "OUTPERFORMS",
    "object_name": "GPT-3.5", "object_type": "MODEL"}},
  {{"subject_name": "GPT-4", "subject_type": "MODEL", "predicate": "EVALUATED_ON",
    "object_name": "MMLU", "object_type": "DATASET"}}
]

If no meaningful triples can be extracted, return an empty array: []

JSON output:"""


# -- Extractor -------------------------------------------------------------


class EntityExtractor:
    """Extract entity/relation triples from chunks using a local LLM.

    Parameters
    ----------
    config : ExtractorConfig
        Model, URL, cache location, and extraction parameters.
    """

    def __init__(self, config: ExtractorConfig | None = None):
        self.config = config or ExtractorConfig()
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- public API --------------------------------------------------------

    def extract(
        self,
        chunks: list[ChunkData],
        paper: dict,
    ) -> list[ExtractionResult]:
        """Extract triples from a list of chunks belonging to one paper.

        Returns one ExtractionResult per chunk.  Results are cached on disk
        so repeated calls skip already-processed chunks.
        """
        if not chunks:
            return []

        title = paper.get("title", "Unknown")
        abstract = self._get_abstract(paper)
        doc_id = paper.get("doc_id", "unknown")
        cache = self._load_cache(doc_id)

        results: list[ExtractionResult] = []
        cache_hits = 0
        generated = 0
        t0 = time.time()

        for _i, chunk in enumerate(chunks):
            chunk_hash = self._hash_text(chunk.text)

            if chunk_hash in cache:
                # Reconstruct from cached triples
                cached_triples = [Triple.from_dict(t) for t in cache[chunk_hash]]
                results.append(
                    ExtractionResult(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        triples=cached_triples,
                    )
                )
                cache_hits += 1
            else:
                result = self._extract_from_chunk(chunk, title, abstract)
                results.append(result)
                # Cache the triples as dicts
                cache[chunk_hash] = [t.to_dict() for t in result.triples]
                generated += 1

            # Progress logging
            total_done = cache_hits + generated
            if total_done % self.config.batch_log_interval == 0 and total_done > 0:
                elapsed = time.time() - t0
                rate = generated / elapsed if elapsed > 0 and generated > 0 else 0
                logger.info(
                    f"  Extraction progress: {total_done}/{len(chunks)} "
                    f"({cache_hits} cached, {generated} generated, "
                    f"{rate:.1f} chunks/s)"
                )

        # Save updated cache
        self._save_cache(doc_id, cache)

        elapsed = time.time() - t0
        total_triples = sum(len(r.triples) for r in results)
        logger.info(
            f"Extracted {total_triples} triples from {len(chunks)} chunks "
            f"for '{title[:60]}' in {elapsed:.1f}s "
            f"({cache_hits} cached, {generated} generated)"
        )

        return results

    # -- LLM interaction ---------------------------------------------------

    def _extract_from_chunk(
        self,
        chunk: ChunkData,
        title: str,
        abstract: str,
    ) -> ExtractionResult:
        """Call Ollama to extract triples from a single chunk."""
        prompt = _EXTRACTION_PROMPT.format(
            entity_types=", ".join(ENTITY_TYPES),
            relation_types=", ".join(RELATION_TYPES),
            max_triples=self.config.max_triples_per_chunk,
            title=title,
            abstract=abstract[:1000],  # cap abstract to save tokens
            chunk_text=chunk.text[:3000],  # cap chunk text
        )

        try:
            resp = requests.post(
                f"{self.config.ollama_base_url}/api/generate",
                json={
                    "model": self.config.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 1024,  # JSON triples need more tokens
                    },
                },
                timeout=self.config.request_timeout,
            )
            resp.raise_for_status()
            raw_response = resp.json().get("response", "").strip()

            triples = self._parse_triples(raw_response, chunk.chunk_id, chunk.doc_id)
            return ExtractionResult(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                triples=triples,
                raw_llm_response=raw_response,
                parse_success=True,
            )

        except requests.RequestException as e:
            logger.error(f"Ollama request failed for chunk {chunk.chunk_id}: {e}")
            return ExtractionResult(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                triples=[],
                parse_success=False,
            )

    # -- JSON parsing with fallback ----------------------------------------

    def _parse_triples(
        self,
        raw: str,
        chunk_id: str,
        doc_id: str,
    ) -> list[Triple]:
        """Parse LLM JSON output into Triple objects.

        Tries strict json.loads first, then falls back to regex extraction
        for partial/malformed JSON (common with smaller LLMs).
        """
        triples = []

        # Try parsing the full response as JSON
        raw_dicts = self._try_parse_json(raw)

        if raw_dicts is None:
            # Fallback: extract JSON array from within the response
            raw_dicts = self._try_extract_json_array(raw)

        if raw_dicts is None:
            logger.warning(
                f"Failed to parse JSON from LLM response for chunk {chunk_id}. Response starts with: {raw[:100]}"
            )
            return []

        for item in raw_dicts[: self.config.max_triples_per_chunk]:
            triple = self._dict_to_triple(item, chunk_id, doc_id)
            if triple is not None:
                triples.append(triple)

        return triples

    @staticmethod
    def _try_parse_json(raw: str) -> list[dict] | None:
        """Try to parse the raw string as a JSON array."""
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        return None

    @staticmethod
    def _try_extract_json_array(raw: str) -> list[dict] | None:
        """Extract a JSON array from within a larger string.

        LLMs sometimes wrap JSON in markdown code blocks or add
        explanatory text before/after.
        """
        # Try to find a JSON array in the response
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _dict_to_triple(
        item: dict,
        chunk_id: str,
        doc_id: str,
    ) -> Triple | None:
        """Convert a parsed dict to a Triple, with validation."""
        try:
            subject_name = item.get("subject_name", "").strip()
            subject_type = item.get("subject_type", "OTHER").strip().upper()
            predicate = item.get("predicate", "").strip().upper()
            object_name = item.get("object_name", "").strip()
            object_type = item.get("object_type", "OTHER").strip().upper()

            # Validate: all fields must be non-empty
            if not all([subject_name, predicate, object_name]):
                return None

            # Skip self-referential triples
            if subject_name.lower() == object_name.lower():
                return None

            return Triple(
                subject=Entity(name=subject_name, entity_type=subject_type),
                predicate=predicate,
                object=Entity(name=object_name, entity_type=object_type),
                source_chunk_id=chunk_id,
                source_doc_id=doc_id,
            )

        except (AttributeError, TypeError):
            return None

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _get_abstract(paper: dict) -> str:
        """Extract abstract from paper sections."""
        sections = paper.get("sections", {})
        for key in ("abstract", "Abstract", "ABSTRACT"):
            if key in sections:
                return sections[key]
        return ""

    @staticmethod
    def _hash_text(text: str) -> str:
        """SHA-256 hash of chunk text, used as cache key."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _cache_path(self, doc_id: str) -> Path:
        safe_id = doc_id.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_id}.json"

    def _load_cache(self, doc_id: str) -> dict[str, list[dict]]:
        """Load cached extraction results for a paper."""
        path = self._cache_path(doc_id)
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load cache for {doc_id}: {e}")
        return {}

    def _save_cache(self, doc_id: str, cache: dict[str, list[dict]]) -> None:
        path = self._cache_path(doc_id)
        try:
            path.write_text(json.dumps(cache, indent=2))
        except OSError as e:
            logger.warning(f"Failed to save cache for {doc_id}: {e}")
