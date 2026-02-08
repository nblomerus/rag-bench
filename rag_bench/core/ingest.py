"""
ingest.py — Load AI/ML papers from HuggingFace and parse into document schema.

Handles:
- Loading jamescalam/ai-arxiv2 dataset
- Extracting structured sections from markdown-formatted paper text
- Building per-paper acronym dictionaries
- Producing clean document dicts ready for chunking
"""

import json
import logging
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from rag_bench.utils.text import (
    build_acronym_dict,
    extract_sections,
)

logger = logging.getLogger(__name__)


def load_arxiv_dataset(
    dataset_name: str = "jamescalam/ai-arxiv2",
    split: str = "train",
) -> list[dict]:
    """Load the HuggingFace dataset and return as list of dicts."""
    logger.info(f"Loading dataset: {dataset_name} (split={split})")
    ds = load_dataset(dataset_name, split=split)
    logger.info(f"Loaded {len(ds)} papers")
    return list(ds)


def extract_year(row: dict) -> int | None:
    """Extract publication year from dataset row."""
    # Try direct year field
    if "year" in row and row["year"]:
        try:
            return int(row["year"])
        except (ValueError, TypeError):
            pass

    # Try to extract from date fields
    for field in ["published", "date", "created"]:
        if field in row and row[field]:
            match = re.search(r"(20\d{2}|19\d{2})", str(row[field]))
            if match:
                return int(match.group(1))

    # Try to extract from arxiv_id (format: YYMM.NNNNN)
    arxiv_id = row.get("id", row.get("arxiv_id", ""))
    if arxiv_id:
        match = re.match(r"(\d{2})(\d{2})\.", str(arxiv_id))
        if match:
            year = int(match.group(1))
            return 2000 + year if year < 50 else 1900 + year

    return None


def parse_paper(row: dict) -> dict:
    """
    Convert a HuggingFace dataset row into our document schema.

    Returns a structured dict with:
    - doc_id, title, authors, year
    - sections: dict of section_name -> text
    - acronyms: dict of ACRONYM -> full form
    - full_text: the complete paper text
    """
    # Get the paper text — dataset may use different field names
    full_text = row.get("content", row.get("text", row.get("chunk", "")))
    if not full_text:
        full_text = ""

    # Get title
    title = row.get("title", "Unknown")
    if isinstance(title, list):
        title = title[0] if title else "Unknown"

    # Get authors
    authors = row.get("authors", [])
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(",")]

    # Get arxiv ID
    arxiv_id = str(row.get("id", row.get("arxiv_id", row.get("doi", "unknown"))))

    # Build document
    doc = {
        "doc_id": f"arxiv_{arxiv_id}".replace("/", "_"),
        "title": title.strip() if isinstance(title, str) else str(title),
        "authors": authors,
        "year": extract_year(row),
        "arxiv_id": arxiv_id,
        "full_text": full_text,
        "sections": extract_sections(full_text),
        "acronyms": build_acronym_dict(full_text),
    }

    return doc


def ingest_dataset(
    dataset_name: str = "jamescalam/ai-arxiv2",
    split: str = "train",
    save_path: Path | None = None,
) -> list[dict]:
    """
    Full ingestion pipeline: load dataset -> parse all papers -> return docs.

    Optionally saves parsed documents to JSON for caching.
    """
    raw_data = load_arxiv_dataset(dataset_name, split)

    docs = []
    skipped = 0

    for row in tqdm(raw_data, desc="Parsing papers"):
        doc = parse_paper(row)

        # Skip papers with no meaningful text
        if len(doc["full_text"]) < 100:
            skipped += 1
            continue

        docs.append(doc)

    logger.info(f"Parsed {len(docs)} papers ({skipped} skipped due to no text)")

    # Optionally cache to disk
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(docs, f, indent=2, default=str)
        logger.info(f"Saved parsed documents to {save_path}")

    return docs
