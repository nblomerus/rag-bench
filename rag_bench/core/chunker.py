"""
chunker.py — Chunk parsed papers into retrieval-ready segments.

Handles AI/ML-specific challenges:
- Preserves mathematical equations as atomic units
- Keeps table rows with their headers
- Expands acronyms on first occurrence per chunk
- Attaches rich metadata for citation formatting
- Filters noisy sections (references, acknowledgments) that degrade retrieval
"""

import logging
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_bench.config import MIN_CHUNK_LENGTH, SECTION_BLOCKLIST
from rag_bench.core.configs import ChunkerConfig
from rag_bench.core.types import ChunkData
from rag_bench.utils.text import format_authors

logger = logging.getLogger(__name__)


# ── Equation detection patterns ──
EQUATION_PATTERNS = [
    re.compile(r"\$\$.*?\$\$", re.DOTALL),  # $$...$$
    re.compile(r"\\\[.*?\\\]", re.DOTALL),  # \[...\]
    re.compile(r"\\begin\{equation\}.*?\\end\{equation\}", re.DOTALL),
    re.compile(r"\\begin\{align\}.*?\\end\{align\}", re.DOTALL),
    re.compile(r"\\begin\{gather\}.*?\\end\{gather\}", re.DOTALL),
]

# ── Table detection ──
TABLE_ROW_PATTERN = re.compile(r"^\|.*\|$", re.MULTILINE)
TABLE_SEPARATOR_PATTERN = re.compile(r"^\|[\s\-:]+\|$", re.MULTILINE)


class PaperChunker:
    """
    Chunks AI/ML papers with domain-aware splitting.

    Key features:
    - Equation-aware: never splits inside math blocks
    - Table-aware: keeps table rows with column headers
    - Acronym expansion: expands first occurrence per chunk
    - Section filtering: skips references, acknowledgments, and other noise
    - Contextual prefix: prepends paper title + section for embedding quality
    - Rich metadata: every chunk carries citation-ready metadata
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        min_section_length: int = 50,
        *,
        config: ChunkerConfig | None = None,
    ):
        if config is not None:
            chunk_size = config.chunk_size
            chunk_overlap = config.chunk_overlap
            min_section_length = config.min_section_length

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_section_length = min_section_length

        # Separators prioritize keeping equations and paragraphs intact
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",  # paragraph break (highest priority)
                "\n",  # line break
                ". ",  # sentence break
                "; ",  # clause break
                ", ",  # phrase break
                " ",  # word break
                "",  # character break (last resort)
            ],
            length_function=len,
        )

    def chunk_paper(self, doc: dict) -> list[ChunkData]:
        """
        Chunk a parsed paper into retrieval-ready segments.

        Args:
            doc: Parsed paper dict with sections, acronyms, metadata.

        Returns:
            List of ChunkData, each with chunk_id, doc_id, text, section,
            and metadata (source_display, title, year, arxiv_id, topic, categories).
        """
        chunks: list[ChunkData] = []
        acronyms = doc.get("acronyms", {})

        # Build citation display string
        source_display = f'{format_authors(doc["authors"])} ({doc["year"]}) "{doc["title"]}"'

        sections = doc.get("sections", {})
        if not sections:
            # Fallback: chunk the full text as a single section
            sections = {"full_text": doc.get("full_text", "")}

        for section_name, section_text in sections.items():
            # Skip noisy sections that degrade retrieval at scale
            if section_name.lower() in SECTION_BLOCKLIST:
                continue

            if not section_text or len(section_text.strip()) < self.min_section_length:
                continue

            # Pre-process: protect equations from splitting
            protected_text = self._protect_equations(section_text)

            # Pre-process: handle tables
            protected_text = self._protect_tables(protected_text)

            # Split into chunks
            text_chunks = self.splitter.split_text(protected_text)

            for i, chunk_text in enumerate(text_chunks):
                # Restore any equation placeholders
                chunk_text = self._restore_equations(chunk_text)

                # Expand acronyms (first occurrence per chunk)
                chunk_text = self._expand_acronyms(chunk_text, acronyms)

                # Clean up whitespace
                chunk_text = self._clean_text(chunk_text)

                if len(chunk_text.strip()) < MIN_CHUNK_LENGTH:
                    continue  # skip trivially small chunks

                # Prepend contextual prefix for better embedding quality
                section_label = section_name.replace("_", " ").title()
                prefix = f"{doc['title']} — {section_label}\n\n"
                chunk_text = prefix + chunk_text

                # Flatten categories for ChromaDB (requires scalar values)
                categories = doc.get("categories", [])
                if isinstance(categories, list):
                    categories = ",".join(categories)

                chunk = ChunkData(
                    chunk_id=f"{doc['doc_id']}_{section_name}_{i:03d}",
                    doc_id=doc["doc_id"],
                    text=chunk_text,
                    section=section_name,
                    metadata={
                        "source_display": source_display,
                        "title": doc["title"],
                        "year": doc["year"],
                        "arxiv_id": doc.get("arxiv_id", ""),
                        "section": section_name,
                        "topic": doc.get("topic", ""),
                        "categories": categories,
                    },
                )
                chunks.append(chunk)

        return chunks

    # ── Equation handling ──

    def _protect_equations(self, text: str) -> str:
        """
        Replace equations with placeholders so the splitter doesn't break them.
        Stores equations in self._equation_store for later restoration.
        """
        if not hasattr(self, "_equation_store"):
            self._equation_store = {}

        for pattern in EQUATION_PATTERNS:
            for match in pattern.finditer(text):
                eq_id = f"__EQ_{len(self._equation_store):04d}__"
                self._equation_store[eq_id] = match.group(0)
                text = text.replace(match.group(0), eq_id, 1)

        return text

    def _restore_equations(self, text: str) -> str:
        """Restore equation placeholders back to original equations."""
        if not hasattr(self, "_equation_store"):
            return text

        for eq_id, equation in self._equation_store.items():
            text = text.replace(eq_id, equation)

        return text

    def _reset_equation_store(self):
        """Clear the equation store between papers."""
        self._equation_store = {}

    # ── Table handling ──

    def _protect_tables(self, text: str) -> str:
        """
        Ensure table rows stay with their column headers.

        Strategy: if a table is small enough, keep it as one block.
        If too large, at least keep the header row with each chunk of rows.
        """
        lines = text.split("\n")
        result_lines = []
        in_table = False

        for line in lines:
            is_table_row = bool(TABLE_ROW_PATTERN.match(line.strip()))
            is_separator = bool(TABLE_SEPARATOR_PATTERN.match(line.strip()))

            if is_table_row and not in_table:
                # Start of a new table
                in_table = True
                result_lines.append(line)
            elif (is_separator and in_table) or (is_table_row and in_table):
                result_lines.append(line)
            else:
                if in_table:
                    in_table = False
                result_lines.append(line)

        return "\n".join(result_lines)

    # ── Acronym expansion ──

    def _expand_acronyms(self, text: str, acronyms: dict[str, str]) -> str:
        """
        Expand the first occurrence of each acronym in the chunk.

        Example: "MIPS" -> "Maximum Inner Product Search (MIPS)"
        Only expands if the acronym appears as a standalone word.
        """
        if not acronyms:
            return text

        for acronym, full_form in acronyms.items():
            # Only expand if the acronym appears as a standalone word
            # and isn't already expanded (e.g., "Full Name (ACRONYM)")
            pattern = rf"\b{re.escape(acronym)}\b"
            already_expanded = f"{full_form} ({acronym})"

            if already_expanded not in text and re.search(pattern, text):
                text = re.sub(pattern, f"{full_form} ({acronym})", text, count=1)

        return text

    # ── Text cleanup ──

    def _clean_text(self, text: str) -> str:
        """Clean up whitespace and formatting artifacts."""
        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove leading/trailing whitespace per line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)
        # Remove leading/trailing whitespace overall
        text = text.strip()
        return text


def chunk_all_papers(
    docs: list[dict],
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    min_section_length: int = 50,
) -> list[dict]:
    """
    Chunk all parsed papers and return flat list of chunks.

    Args:
        docs: List of parsed paper dicts from ingest.py
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between adjacent chunks
        min_section_length: Minimum section length to chunk

    Returns:
        Flat list of chunk dicts ready for embedding
    """
    chunker = PaperChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_section_length=min_section_length,
    )

    all_chunks: list[dict] = []

    for doc in docs:
        chunker._reset_equation_store()
        paper_chunks = chunker.chunk_paper(doc)
        all_chunks.extend(c.to_dict() for c in paper_chunks)
        logger.debug(f"  {doc['title'][:60]}... -> {len(paper_chunks)} chunks")

    logger.info(
        f"Chunked {len(docs)} papers into {len(all_chunks)} total chunks "
        f"(avg {len(all_chunks) / max(len(docs), 1):.1f} chunks/paper)"
    )

    return all_chunks
