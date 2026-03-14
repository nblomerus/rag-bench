"""
enricher.py — Contextual header enrichment for chunks.

After chunking, each chunk loses its broader document context.  A chunk
that says "the authors found this approach effective" doesn't embed well
because it lacks the *who* and *what*.

The ContextualEnricher calls a local LLM (via Ollama) to generate a
short contextual header for each chunk — situating it within the full
document.  The header is prepended to the chunk text before embedding,
dramatically improving retrieval for pronoun-heavy or abbreviation-heavy
passages.

Headers are cached on disk (keyed by a hash of the chunk text) so
re-running the pipeline skips already-enriched chunks.

Design notes
------------
* Separate from PaperChunker (single-responsibility: chunker splits,
  enricher contextualises).
* Accepts any list of ChunkData — works with both recursive and semantic
  chunking strategies.
* Falls back to a compressed document context (title + section headers +
  current section) when the full paper exceeds the LLM context window.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from rag_bench.core.types import ChunkData

logger = logging.getLogger(__name__)

# Conservative estimate: 1 token ≈ 3.5 chars for English text.
# We leave headroom for the prompt template + generated output.
_CHARS_PER_TOKEN = 3.5
_PROMPT_OVERHEAD_TOKENS = 512  # template + output budget
_DEFAULT_HEADER_TAG = "<context>"
_DEFAULT_HEADER_END_TAG = "</context>"


@dataclass
class EnricherConfig:
    """Configuration for contextual header enrichment."""

    enabled: bool = False
    ollama_model: str = "qwen2.5:14b-instruct-q4_K_M"
    ollama_base_url: str = "http://localhost:11434"
    cache_dir: str = ".enricher_cache"
    max_context_tokens: int = 30_000  # leave 2K for prompt + output
    request_timeout: int = 120  # seconds per LLM call
    batch_log_interval: int = 25  # log progress every N chunks


_PROMPT_TEMPLATE = """\
<document>
{document_context}
</document>

Here is a chunk from that document:

<chunk>
{chunk_text}
</chunk>

Write a short context (2-3 sentences) that situates this chunk within \
the full document. Mention the paper title, the specific topic of this \
chunk, and how it relates to the paper's overall contribution. \
Be factual and concise — this context will be prepended to the chunk \
for search indexing.

Respond with ONLY the context sentences, nothing else."""


class ContextualEnricher:
    """Generate and prepend contextual headers to chunks using a local LLM.

    Parameters
    ----------
    config : EnricherConfig
        All knobs for the enricher (model, URL, cache location, etc.).
    """

    def __init__(self, config: EnricherConfig | None = None):
        self.config = config or EnricherConfig()
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Max chars we can fit in the document context portion of the prompt
        available_tokens = self.config.max_context_tokens - _PROMPT_OVERHEAD_TOKENS
        self._max_context_chars = int(available_tokens * _CHARS_PER_TOKEN)

    # -- public API -----------------------------------------------------------

    def enrich(
        self,
        chunks: list[ChunkData],
        paper: dict,
    ) -> list[ChunkData]:
        """Add contextual headers to a list of chunks from one paper.

        For each chunk:
        1. Check the on-disk cache — return cached header if available.
        2. Build a document context (full paper or compressed fallback).
        3. Call the LLM to generate a 2-3 sentence header.
        4. Prepend the header to the chunk text.
        5. Save the header to cache.

        Returns a new list of ChunkData (originals are frozen/immutable).
        """
        if not chunks:
            return []

        doc_context = self._build_document_context(paper)
        cache = self._load_cache(paper.get("doc_id", "unknown"))

        enriched: list[ChunkData] = []
        cache_hits = 0
        generated = 0
        t0 = time.time()

        for _i, chunk in enumerate(chunks):
            chunk_hash = self._hash_text(chunk.text)

            # Check cache
            if chunk_hash in cache:
                header = cache[chunk_hash]
                cache_hits += 1
            else:
                header = self._generate_header(chunk.text, doc_context)
                cache[chunk_hash] = header
                generated += 1

            # Build enriched chunk with header prepended
            enriched_text = f"{header}\n\n{chunk.text}"
            new_chunk = ChunkData(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=enriched_text,
                section=chunk.section,
                metadata={**chunk.metadata, "has_contextual_header": True},
            )
            enriched.append(new_chunk)

            # Progress logging
            total_done = cache_hits + generated
            if total_done % self.config.batch_log_interval == 0:
                elapsed = time.time() - t0
                rate = generated / elapsed if elapsed > 0 and generated > 0 else 0
                logger.info(
                    f"  Enrichment progress: {total_done}/{len(chunks)} "
                    f"({cache_hits} cached, {generated} generated, "
                    f"{rate:.1f} chunks/s)"
                )

        # Save updated cache
        self._save_cache(paper.get("doc_id", "unknown"), cache)

        elapsed = time.time() - t0
        logger.info(
            f"Enriched {len(chunks)} chunks for '{paper.get('title', '?')[:60]}' "
            f"in {elapsed:.1f}s ({cache_hits} cached, {generated} generated)"
        )

        return enriched

    # -- context building -----------------------------------------------------

    def _build_document_context(self, paper: dict) -> str:
        """Build the document context string for the LLM prompt.

        Strategy:
        - Try full paper text first (best quality).
        - If too long, fall back to: title + abstract + section headers +
          first paragraph of each section (compressed context).
        """
        full_text = self._build_full_text(paper)

        if len(full_text) <= self._max_context_chars:
            return full_text

        # Compressed fallback
        logger.debug(
            f"Paper '{paper.get('title', '?')[:50]}' too long "
            f"({len(full_text):,} chars > {self._max_context_chars:,}), "
            f"using compressed context"
        )
        return self._build_compressed_context(paper)

    def _build_full_text(self, paper: dict) -> str:
        """Reconstruct the full paper text from sections."""
        parts = [f"Title: {paper.get('title', 'Unknown')}"]

        authors = paper.get("authors", [])
        if authors:
            if isinstance(authors, list):
                parts.append(f"Authors: {', '.join(authors[:5])}")
            else:
                parts.append(f"Authors: {authors}")

        sections = paper.get("sections", {})
        for name, text in sections.items():
            label = name.replace("_", " ").title()
            parts.append(f"\n## {label}\n{text}")

        return "\n".join(parts)

    def _build_compressed_context(self, paper: dict) -> str:
        """Build a compressed context for papers that exceed the token limit.

        Includes: title, authors, abstract (full), section headers with
        first ~200 chars of each section.
        """
        parts = [f"Title: {paper.get('title', 'Unknown')}"]

        authors = paper.get("authors", [])
        if authors:
            if isinstance(authors, list):
                parts.append(f"Authors: {', '.join(authors[:5])}")
            else:
                parts.append(f"Authors: {authors}")

        sections = paper.get("sections", {})

        # Include abstract in full if present
        for name in ("abstract", "Abstract"):
            if name in sections:
                parts.append(f"\n## Abstract\n{sections[name]}")
                break

        # Section headers + truncated preview
        for name, text in sections.items():
            if name.lower() == "abstract":
                continue
            label = name.replace("_", " ").title()
            preview = text[:200].rsplit(" ", 1)[0] + "…" if len(text) > 200 else text
            parts.append(f"\n## {label}\n{preview}")

        result = "\n".join(parts)

        # If still too long, hard-truncate (shouldn't happen often)
        if len(result) > self._max_context_chars:
            result = result[: self._max_context_chars] + "\n[truncated]"

        return result

    # -- LLM interaction ------------------------------------------------------

    def _generate_header(self, chunk_text: str, doc_context: str) -> str:
        """Call Ollama to generate a contextual header for one chunk."""
        prompt = _PROMPT_TEMPLATE.format(
            document_context=doc_context,
            chunk_text=chunk_text[:2000],  # cap chunk in prompt to save tokens
        )

        try:
            resp = requests.post(
                f"{self.config.ollama_base_url}/api/generate",
                json={
                    "model": self.config.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # low temp for factual, consistent output
                        "num_predict": 150,  # 2-3 sentences ≈ 50-100 tokens
                    },
                },
                timeout=self.config.request_timeout,
            )
            resp.raise_for_status()
            header = resp.json().get("response", "").strip()

            # Sanity check: if LLM returned nothing useful, skip the header
            if len(header) < 20:
                logger.warning(f"LLM returned very short header ({len(header)} chars), skipping")
                return ""

            return header

        except requests.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            return ""

    # -- caching --------------------------------------------------------------

    @staticmethod
    def _hash_text(text: str) -> str:
        """SHA-256 hash of chunk text, used as cache key."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _cache_path(self, doc_id: str) -> Path:
        """Path to the cache file for a given document."""
        # Sanitize doc_id for use as filename
        safe_id = doc_id.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_id}.json"

    def _load_cache(self, doc_id: str) -> dict[str, str]:
        """Load cached headers for a document. Returns empty dict on miss."""
        path = self._cache_path(doc_id)
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load cache for {doc_id}: {e}")
        return {}

    def _save_cache(self, doc_id: str, cache: dict[str, str]) -> None:
        """Persist the header cache for a document."""
        path = self._cache_path(doc_id)
        try:
            path.write_text(json.dumps(cache, indent=2))
        except OSError as e:
            logger.warning(f"Failed to save cache for {doc_id}: {e}")
